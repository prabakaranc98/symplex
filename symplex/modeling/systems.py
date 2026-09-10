"""Dimensional stock-and-flow integration with host-recomputed numerical invariants.

An agent declares stocks, flows, parameters, couplings, delays and interventions with
explicit units and dimensions. Host code parses each rate expression into a closed
arithmetic AST, performs dimensional analysis over that AST, integrates the resulting
ODE or SDE under a frozen integrator, and recomputes conservation, non-negativity and
step-refinement diagnostics itself.

Declared content has no execution authority: no attribute access, subscripting,
comprehension, lambda, import or non-allowlisted call survives compilation, and no
agent-authored Python is executed anywhere in this module. A satisfied numerical
invariant is a statement about the integrator and the declared algebra. It is not
evidence that the mechanism, the parameter values or the unit definitions correspond
to a real system.
"""

import ast
import bisect
import csv
import io
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated, Callable, Literal

import numpy as np
from pydantic import Field, model_validator

from symplex.core.contracts import Invalid, digest
from symplex.modeling.complex_system import ClosedContract, Identifier, Text

PROTOCOL = "dimensional-stock-flow-v1"
SCOPE = (
    "Conditional on declared mechanisms, parameters, units and rate expressions. "
    "Numerical invariants hold under the frozen integrator and the declared dimension "
    "algebra; they do not establish empirical validity, parameter identifiability, "
    "measurement correspondence or real-world impact."
)

MAX_STOCKS = 24
MAX_FLOWS = 48
MAX_AUXILIARIES = 24
MAX_PARAMETERS = 48
MAX_COUPLINGS = 64
MAX_DELAYS = 8
MAX_INTERVENTIONS = 12
MAX_GROUPS = 6
MAX_SCENARIOS = 6
MAX_STEPS = 20000
MAX_REPLICATES = 64
MAX_STATE_SAMPLES = 2_000_000
MAX_CSV_ROWS = 100_000
MAX_EXPRESSION_NODES = 200
MAX_EXPRESSION_CHARS = 400
MAX_DIMENSION_CHARS = 200
MAX_DIMENSION_EXPONENT = 12
MAX_POWER = 64

Expression = Annotated[
    str,
    Field(
        min_length=1,
        max_length=MAX_EXPRESSION_CHARS,
        pattern=r"^[a-zA-Z0-9_+\-*/^().,<>=! \t]+$",
        description="Declarative arithmetic over declared identifiers and t. Not Python source; the host compiles a closed AST allowlist.",
    ),
]
Dimension = Annotated[
    str,
    Field(
        min_length=1,
        max_length=MAX_DIMENSION_CHARS,
        description='Product of base dimensions with rational exponents, e.g. "mass*time^-1", "amount/length^3", or "1" for dimensionless.',
    ),
]
Unit = Annotated[str, Field(min_length=1, max_length=64)]


# --------------------------------------------------------------------------------------
# Dimension algebra
# --------------------------------------------------------------------------------------

_FACTOR = re.compile(r"^([a-z][a-z0-9_]*)(?:\^\(?(-?\d+(?:\.\d+)?(?:/\d+)?)\)?)?$")


def parse_dimension(text):
    """Parse a declared dimension string into {base name: rational exponent}."""
    if not isinstance(text, str) or not text.strip():
        raise Invalid("Expected a nonempty dimension string")
    body = text.strip()
    if len(body) > MAX_DIMENSION_CHARS:
        raise Invalid("Dimension string exceeds the permitted length")
    if body in ("1", "dimensionless"):
        return {}
    tokens, buffer, depth = [], "", 0
    for character in body:
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        if depth < 0:
            raise Invalid("Unbalanced parentheses in a dimension string")
        if character in "*/" and depth == 0:
            tokens.extend((buffer, character))
            buffer = ""
        else:
            buffer += character
    if depth != 0:
        raise Invalid("Unbalanced parentheses in a dimension string")
    tokens.append(buffer)
    exponents, sign = {}, 1
    for position, token in enumerate(tokens):
        if position % 2 == 1:
            sign = 1 if token == "*" else -1
            continue
        factor = token.strip()
        if factor == "1":
            continue
        matched = _FACTOR.match(factor)
        if not matched:
            raise Invalid("Unparseable dimension factor: " + repr(factor))
        try:
            power = Fraction(matched.group(2)) if matched.group(2) else Fraction(1)
        except (ValueError, ZeroDivisionError):
            raise Invalid("Unparseable dimension exponent: " + repr(factor)) from None
        if abs(power) > MAX_DIMENSION_EXPONENT:
            raise Invalid("Dimension exponent magnitude exceeds the permitted bound")
        exponents[matched.group(1)] = exponents.get(matched.group(1), Fraction(0)) + sign * power
    return {name: power for name, power in exponents.items() if power != 0}


def format_dimension(exponents):
    """Canonical text for a parsed dimension; the inverse of parse_dimension up to order."""
    if not exponents:
        return "1"
    return "*".join(
        name
        if power == 1
        else (f"{name}^{power}" if power.denominator == 1 else f"{name}^({power})")
        for name, power in sorted(exponents.items())
    )


def _dim_mul(left, right):
    out = dict(left)
    for name, power in right.items():
        out[name] = out.get(name, Fraction(0)) + power
    return {name: power for name, power in out.items() if power != 0}


def _dim_div(left, right):
    return _dim_mul(left, {name: -power for name, power in right.items()})


def _dim_pow(base, power):
    scaled = {name: exponent * power for name, exponent in base.items()}
    for exponent in scaled.values():
        if abs(exponent) > MAX_DIMENSION_EXPONENT:
            raise Invalid("Dimension exponent magnitude exceeds the permitted bound")
    return {name: exponent for name, exponent in scaled.items() if exponent != 0}


# --------------------------------------------------------------------------------------
# Closed arithmetic expressions
# --------------------------------------------------------------------------------------

_ARITY = {
    "min": (2, 8),
    "max": (2, 8),
    "exp": (1, 1),
    "log": (1, 2),
    "sqrt": (1, 1),
    "abs": (1, 1),
    "tanh": (1, 1),
    "clip": (3, 3),
    "where": (3, 3),
}
ALLOWED_FUNCTIONS = tuple(sorted(_ARITY))

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Name,
    ast.Call,
    ast.Compare,
    ast.BoolOp,
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Eq,
    ast.NotEq,
    ast.And,
    ast.Or,
)


def _reject(node, reason):
    raise Invalid(f"Rejected {type(node).__name__} in a declared expression: {reason}")


def _div(left, right):
    if right == 0:
        raise Invalid("Division by zero in a declared expression")
    return left / right


def _power(base, exponent):
    if not -MAX_POWER <= exponent <= MAX_POWER:
        raise Invalid("Exponent magnitude exceeds the permitted bound")
    if base < 0 and exponent != int(exponent):
        raise Invalid("Fractional power of a negative base")
    if base == 0 and exponent < 0:
        raise Invalid("Negative power of zero")
    try:
        return float(base) ** float(exponent)
    except OverflowError:
        raise Invalid("Overflow in a declared expression") from None


def _fn_exp(value):
    if value > 700:
        raise Invalid("Overflow in exp() of a declared expression")
    return math.exp(value)


def _fn_log(value, base=None):
    if value <= 0:
        raise Invalid("log() of a nonpositive value in a declared expression")
    if base is None:
        return math.log(value)
    if base <= 0 or base == 1:
        raise Invalid("log() base must be positive and not 1")
    return math.log(value, base)


def _fn_sqrt(value):
    if value < 0:
        raise Invalid("sqrt() of a negative value in a declared expression")
    return math.sqrt(value)


def _fn_clip(value, low, high):
    if low > high:
        raise Invalid("clip() lower bound exceeds its upper bound")
    return min(max(value, low), high)


def _fn_where(condition, when_true, when_false):
    return when_true if condition else when_false


_FUNCTIONS = {
    "min": min,
    "max": max,
    "exp": _fn_exp,
    "log": _fn_log,
    "sqrt": _fn_sqrt,
    "abs": abs,
    "tanh": math.tanh,
    "clip": _fn_clip,
    "where": _fn_where,
}

_BINARY = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: _div,
    ast.Pow: _power,
}
_COMPARE = {
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
}


@dataclass(frozen=True)
class CompiledExpression:
    """A closed arithmetic term. Holds no reference to any Python callable it did not build."""

    text: str
    tree: ast.Expression
    names: frozenset
    functions: frozenset
    evaluator: Callable

    def evaluate(self, bindings):
        value = self.evaluator(bindings)
        if isinstance(value, bool):
            value = 1.0 if value else 0.0
        value = float(value)
        if not math.isfinite(value):
            raise Invalid("Declared expression produced a nonfinite value: " + self.text)
        return value


def _compile_node(node, allowed, names, functions):
    if not isinstance(node, _ALLOWED_NODES):
        _reject(node, "node type is outside the arithmetic allowlist")
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            _reject(node, "only finite numeric literals are permitted")
        if not math.isfinite(node.value):
            _reject(node, "only finite numeric literals are permitted")
        literal = float(node.value)
        return lambda bindings: literal
    if isinstance(node, ast.Name):
        if not isinstance(node.ctx, ast.Load):
            _reject(node, "names are read-only")
        if "__" in node.id:
            _reject(node, "dunder identifiers are not addressable")
        if node.id not in allowed:
            _reject(node, "unknown identifier " + repr(node.id))
        names.add(node.id)
        return lambda bindings, key=node.id: bindings[key]
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.USub, ast.UAdd)):
            _reject(node, "only unary plus and minus are permitted")
        inner = _compile_node(node.operand, allowed, names, functions)
        if isinstance(node.op, ast.UAdd):
            return inner
        return lambda bindings: -inner(bindings)
    if isinstance(node, ast.BinOp):
        operation = _BINARY.get(type(node.op))
        if operation is None:
            _reject(node.op, "operator is outside the arithmetic allowlist")
        left = _compile_node(node.left, allowed, names, functions)
        right = _compile_node(node.right, allowed, names, functions)
        return lambda bindings: operation(left(bindings), right(bindings))
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            _reject(node, "chained comparisons are not permitted")
        operation = _COMPARE.get(type(node.ops[0]))
        if operation is None:
            _reject(node.ops[0], "comparison is outside the allowlist")
        left = _compile_node(node.left, allowed, names, functions)
        right = _compile_node(node.comparators[0], allowed, names, functions)
        return lambda bindings: operation(left(bindings), right(bindings))
    if isinstance(node, ast.BoolOp):
        parts = [_compile_node(v, allowed, names, functions) for v in node.values]
        if isinstance(node.op, ast.And):
            return lambda bindings: all(bool(p(bindings)) for p in parts)
        return lambda bindings: any(bool(p(bindings)) for p in parts)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            _reject(node, "only direct calls to allowlisted functions are permitted")
        name = node.func.id
        if name not in _ARITY:
            _reject(node, "call to non-allowlisted function " + repr(name))
        if node.keywords:
            _reject(node, "keyword arguments are not permitted")
        low, high = _ARITY[name]
        if not low <= len(node.args) <= high:
            _reject(node, f"{name}() takes {low}-{high} arguments")
        functions.add(name)
        arguments = [_compile_node(a, allowed, names, functions) for a in node.args]
        implementation = _FUNCTIONS[name]
        return lambda bindings: implementation(*[a(bindings) for a in arguments])
    _reject(node, "node type is outside the arithmetic allowlist")


def compile_expression(text, allowed_names):
    """Compile a declared arithmetic term; anything outside the allowlist raises Invalid."""
    if not isinstance(text, str) or not text.strip():
        raise Invalid("Expected a nonempty expression")
    if len(text) > MAX_EXPRESSION_CHARS:
        raise Invalid("Expression exceeds the permitted length")
    if "\n" in text or "\r" in text or "#" in text:
        raise Invalid("Expressions are single-line arithmetic without comments")
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        raise Invalid("Expression is not parseable arithmetic: " + text) from None
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_EXPRESSION_NODES:
        raise Invalid("Expression exceeds the permitted node count")
    for node in nodes:
        if not isinstance(node, _ALLOWED_NODES):
            _reject(node, "node type is outside the arithmetic allowlist")
    allowed = frozenset(allowed_names)
    names, functions = set(), set()
    evaluator = _compile_node(tree.body, allowed, names, functions)
    return CompiledExpression(
        text=text,
        tree=tree,
        names=frozenset(names),
        functions=frozenset(functions),
        evaluator=evaluator,
    )


def _infer_dimension(node, dimensions):
    if isinstance(node, ast.Constant):
        return {}
    if isinstance(node, ast.Name):
        return dict(dimensions[node.id])
    if isinstance(node, ast.UnaryOp):
        return _infer_dimension(node.operand, dimensions)
    if isinstance(node, ast.BinOp):
        left = _infer_dimension(node.left, dimensions)
        right = _infer_dimension(node.right, dimensions)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            if left != right:
                raise Invalid(
                    "Additive terms have different dimensions: "
                    f"{format_dimension(left)} and {format_dimension(right)}"
                )
            return left
        if isinstance(node.op, ast.Mult):
            return _dim_mul(left, right)
        if isinstance(node.op, ast.Div):
            return _dim_div(left, right)
        if right:
            raise Invalid("A dimensional exponent is not admissible")
        if not left:
            return {}
        exponent = _constant_exponent(node.right)
        if exponent is None:
            raise Invalid(
                "A dimensional base requires a constant rational exponent: " + ast.unparse(node)
            )
        return _dim_pow(left, exponent)
    if isinstance(node, ast.Compare):
        left = _infer_dimension(node.left, dimensions)
        right = _infer_dimension(node.comparators[0], dimensions)
        if left != right:
            raise Invalid(
                "Compared terms have different dimensions: "
                f"{format_dimension(left)} and {format_dimension(right)}"
            )
        return {}
    if isinstance(node, ast.BoolOp):
        for value in node.values:
            if _infer_dimension(value, dimensions):
                raise Invalid("Boolean operands must be dimensionless")
        return {}
    if isinstance(node, ast.Call):
        name = node.func.id
        parts = [_infer_dimension(a, dimensions) for a in node.args]
        if name in ("exp", "log", "tanh"):
            if any(parts):
                raise Invalid(f"{name}() requires dimensionless arguments")
            return {}
        if name == "sqrt":
            return _dim_pow(parts[0], Fraction(1, 2))
        if name == "abs":
            return parts[0]
        if name in ("min", "max", "clip"):
            if any(part != parts[0] for part in parts[1:]):
                raise Invalid(f"{name}() requires arguments of one dimension")
            return parts[0]
        if name == "where":
            if parts[0]:
                raise Invalid("where() requires a dimensionless condition")
            if parts[1] != parts[2]:
                raise Invalid("where() branches have different dimensions")
            return parts[1]
    raise Invalid("Dimensional analysis reached an unsupported term")


def _constant_exponent(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Fraction(node.value).limit_denominator(64)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _constant_exponent(node.operand)
        return None if inner is None else -inner
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _constant_exponent(node.operand)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left, right = _constant_exponent(node.left), _constant_exponent(node.right)
        if left is None or right is None or right == 0:
            return None
        return left / right
    return None


# --------------------------------------------------------------------------------------
# Typed contract
# --------------------------------------------------------------------------------------


class Stock(ClosedContract):
    """An accumulating quantity with a declared unit and dimension."""

    id: Identifier
    name: Text
    unit: Unit
    dimension: Dimension
    initial: float
    non_negative: bool = Field(
        description="Declares an expectation the host checks against the trajectory. The integrator does not clamp; a violation is reported, not hidden."
    )
    noise_sd: float = Field(
        ge=0,
        description="Additive per-stock diffusion in stock units per sqrt(time unit). Used only in stochastic mode.",
    )
    meaning: Text


class Flow(ClosedContract):
    """A declared transfer rate between stocks, or across the system boundary."""

    id: Identifier
    name: Text
    from_stock: Identifier | None = Field(
        description="Source stock, or null for an inflow from outside the declared boundary."
    )
    to_stock: Identifier | None = Field(
        description="Destination stock, or null for an outflow across the declared boundary."
    )
    rate: Expression
    unit: Unit
    dimension: Dimension = Field(
        description="Must equal the endpoint stock dimension divided by the time dimension."
    )
    delay_id: Identifier | None
    meaning: Text

    @model_validator(mode="after")
    def connected(self):
        if self.from_stock is None and self.to_stock is None:
            raise ValueError("A flow must touch at least one declared stock")
        if self.from_stock is not None and self.from_stock == self.to_stock:
            raise ValueError("A flow cannot have the same source and destination")
        return self


class Auxiliary(ClosedContract):
    """A derived quantity available to flow rates; it accumulates nothing."""

    id: Identifier
    name: Text
    expression: Expression
    unit: Unit
    dimension: Dimension
    meaning: Text


class Parameter(ClosedContract):
    id: Identifier
    name: Text
    value: float
    unit: Unit
    dimension: Dimension
    lower: float
    upper: float
    meaning: Text

    @model_validator(mode="after")
    def bounded(self):
        if not self.lower <= self.value <= self.upper:
            raise ValueError("Parameter value lies outside its declared bounds")
        return self


class CouplingEdge(ClosedContract):
    """A first-order network transfer between two stocks of the same dimension.

    The rate constant carries dimension time^-1 by construction; transfer moves
    weight*source, gradient moves weight*(source - target). Both conserve the pair.
    """

    id: Identifier
    source: Identifier
    target: Identifier
    kind: Literal["transfer", "gradient"]
    rate_constant: float = Field(ge=0)
    unit: Unit
    mechanism: Text

    @model_validator(mode="after")
    def distinct(self):
        if self.source == self.target:
            raise ValueError("A coupling edge must join two distinct stocks")
        return self


class Delay(ClosedContract):
    """A lag on a declared flow.

    fixed_lag lags the whole transfer, so the pair of endpoints stays conserved and no
    material is held in transit. first_order introduces an explicit pipeline state whose
    content the host counts inside a conservation group when both endpoints are members.
    """

    id: Identifier
    flow_id: Identifier
    kind: Literal["fixed_lag", "first_order"]
    duration: float = Field(gt=0)
    meaning: Text


class Intervention(ClosedContract):
    """A declared change applied to one parameter or one stock at a declared time."""

    id: Identifier
    name: Text
    target_kind: Literal["parameter", "stock"]
    target_id: Identifier
    mode: Literal["step", "pulse", "ramp"]
    time: float = Field(ge=0)
    duration: float | None = Field(
        description="Window length for pulse and ramp; null for step."
    )
    value: float
    absolute: bool = Field(
        description="True sets the target to value; False adds value to the declared baseline."
    )
    meaning: Text

    @model_validator(mode="after")
    def coherent(self):
        if self.mode == "step":
            if self.duration is not None:
                raise ValueError("A step intervention has no duration")
        elif self.duration is None or self.duration <= 0:
            raise ValueError("Pulse and ramp interventions require a positive duration")
        if self.target_kind == "stock":
            if self.mode != "step" and self.absolute:
                raise ValueError(
                    "Stock pulse and ramp interventions declare an added quantity, not an absolute level"
                )
        return self


class ConservationGroup(ClosedContract):
    """A set of stocks whose total the host tracks against declared external flux."""

    id: Identifier
    name: Text
    stock_ids: list[Identifier] = Field(min_length=2, max_length=MAX_STOCKS)
    tolerance: float = Field(ge=0)
    meaning: Text


class SystemScenario(ClosedContract):
    id: Identifier
    name: Text
    intervention_ids: list[Identifier] = Field(max_length=MAX_INTERVENTIONS)
    interpretation: Text


class SystemModel(ClosedContract):
    """A declared dimensional system. Structure is checked here; algebra in compile_model."""

    title: Text
    boundary: Text
    time_unit: Unit
    time_dimension: Dimension
    horizon: float = Field(gt=0)
    stocks: list[Stock] = Field(min_length=1, max_length=MAX_STOCKS)
    flows: list[Flow] = Field(max_length=MAX_FLOWS)
    auxiliaries: list[Auxiliary] = Field(max_length=MAX_AUXILIARIES)
    parameters: list[Parameter] = Field(max_length=MAX_PARAMETERS)
    couplings: list[CouplingEdge] = Field(max_length=MAX_COUPLINGS)
    delays: list[Delay] = Field(max_length=MAX_DELAYS)
    interventions: list[Intervention] = Field(max_length=MAX_INTERVENTIONS)
    conservation_groups: list[ConservationGroup] = Field(max_length=MAX_GROUPS)
    scenarios: list[SystemScenario] = Field(min_length=1, max_length=MAX_SCENARIOS)
    assumptions: list[Text] = Field(max_length=12)
    unmodeled: list[Text] = Field(max_length=12)

    @model_validator(mode="after")
    def structure(self):
        groups = {
            "stock": [s.id for s in self.stocks],
            "flow": [f.id for f in self.flows],
            "auxiliary": [a.id for a in self.auxiliaries],
            "parameter": [p.id for p in self.parameters],
            "coupling": [c.id for c in self.couplings],
            "delay": [d.id for d in self.delays],
            "intervention": [i.id for i in self.interventions],
            "conservation group": [g.id for g in self.conservation_groups],
            "scenario": [s.id for s in self.scenarios],
        }
        seen = {}
        for label, identifiers in groups.items():
            if len(set(identifiers)) != len(identifiers):
                raise ValueError(f"Duplicate {label} ID")
            for identifier in identifiers:
                if identifier in seen:
                    raise ValueError(
                        f"Identifier {identifier!r} is declared as both a {seen[identifier]} and a {label}"
                    )
                seen[identifier] = label
        if "t" in seen:
            raise ValueError("The identifier 't' is reserved for simulation time")
        stocks = set(groups["stock"])
        flows = set(groups["flow"])
        parameters = set(groups["parameter"])
        delays = set(groups["delay"])
        for flow in self.flows:
            for endpoint in (flow.from_stock, flow.to_stock):
                if endpoint is not None and endpoint not in stocks:
                    raise ValueError(f"Flow {flow.id} references unknown stock {endpoint!r}")
            if flow.delay_id is not None and flow.delay_id not in delays:
                raise ValueError(f"Flow {flow.id} references unknown delay {flow.delay_id!r}")
        for coupling in self.couplings:
            if not {coupling.source, coupling.target} <= stocks:
                raise ValueError(f"Coupling {coupling.id} references an unknown stock")
        owned = [d.flow_id for d in self.delays]
        if len(set(owned)) != len(owned):
            raise ValueError("A flow may carry at most one delay")
        for delay in self.delays:
            if delay.flow_id not in flows:
                raise ValueError(f"Delay {delay.id} references unknown flow {delay.flow_id!r}")
            back = [f for f in self.flows if f.id == delay.flow_id]
            if back[0].delay_id != delay.id:
                raise ValueError(f"Flow {delay.flow_id} does not declare delay {delay.id}")
        for intervention in self.interventions:
            pool = parameters if intervention.target_kind == "parameter" else stocks
            if intervention.target_id not in pool:
                raise ValueError(
                    f"Intervention {intervention.id} targets unknown {intervention.target_kind} {intervention.target_id!r}"
                )
            if intervention.time > self.horizon:
                raise ValueError(f"Intervention {intervention.id} starts after the horizon")
        for group in self.conservation_groups:
            if len(set(group.stock_ids)) != len(group.stock_ids):
                raise ValueError(f"Conservation group {group.id} repeats a stock")
            if not set(group.stock_ids) <= stocks:
                raise ValueError(f"Conservation group {group.id} references an unknown stock")
        declared = set(groups["intervention"])
        for scenario in self.scenarios:
            if len(set(scenario.intervention_ids)) != len(scenario.intervention_ids):
                raise ValueError(f"Scenario {scenario.id} repeats an intervention")
            if not set(scenario.intervention_ids) <= declared:
                raise ValueError(f"Scenario {scenario.id} references an unknown intervention")
            targets = [
                (i.target_kind, i.target_id)
                for i in self.interventions
                if i.id in scenario.intervention_ids
            ]
            if len(set(targets)) != len(targets):
                raise ValueError(
                    f"Scenario {scenario.id} applies two interventions to one target"
                )
        if self.scenarios[0].id != "baseline" or self.scenarios[0].intervention_ids:
            raise ValueError("The first scenario must be 'baseline' with no interventions")
        return self

    @classmethod
    def parse(cls, data):
        return cls.model_validate(data).model_dump()

    @classmethod
    def json_schema(cls):
        return cls.model_json_schema()


# --------------------------------------------------------------------------------------
# Compilation and dimensional analysis
# --------------------------------------------------------------------------------------


def _as_model(model):
    if isinstance(model, SystemModel):
        return model
    return SystemModel.model_validate(model)


def _value_names(spec):
    names = {"t"}
    names.update(s.id for s in spec.stocks)
    names.update(p.id for p in spec.parameters)
    names.update(a.id for a in spec.auxiliaries)
    return names


def _compile_expressions(spec):
    allowed = _value_names(spec)
    auxiliary = {a.id: compile_expression(a.expression, allowed) for a in spec.auxiliaries}
    flows = {f.id: compile_expression(f.rate, allowed) for f in spec.flows}
    identifiers = {a.id for a in spec.auxiliaries}
    pending = {a.id: auxiliary[a.id].names & identifiers for a in spec.auxiliaries}
    for identifier, dependencies in pending.items():
        if identifier in dependencies:
            raise Invalid(f"Auxiliary {identifier} references itself")
    order, resolved = [], set()
    while pending:
        ready = sorted(i for i, d in pending.items() if d <= resolved)
        if not ready:
            raise Invalid(
                "Auxiliary definitions form a cycle: " + ", ".join(sorted(pending))
            )
        for identifier in ready:
            order.append((identifier, auxiliary[identifier]))
            resolved.add(identifier)
            pending.pop(identifier)
    return tuple(order), flows


def analyze_dimensions(model):
    """Infer the dimension of every declared expression and compare it with the declaration.

    Structural or syntactic faults raise Invalid. A dimensional mismatch is reported as a
    finding with consistent=False so callers can inspect the whole set at once.
    """
    spec = _as_model(model)
    order, flow_expressions = _compile_expressions(spec)
    time = parse_dimension(spec.time_dimension)
    if not time:
        raise Invalid("The declared time dimension must not be dimensionless")
    bindings = {"t": time}
    for stock in spec.stocks:
        bindings[stock.id] = parse_dimension(stock.dimension)
    for parameter in spec.parameters:
        bindings[parameter.id] = parse_dimension(parameter.dimension)
    for auxiliary in spec.auxiliaries:
        bindings[auxiliary.id] = parse_dimension(auxiliary.dimension)

    findings, auxiliary_rows, flow_rows, coupling_rows = [], [], [], []
    for identifier, expression in order:
        declared = bindings[identifier]
        try:
            inferred = _infer_dimension(expression.tree.body, bindings)
            note = None
        except Invalid as error:
            inferred, note = None, str(error)
        agrees = inferred == declared
        if not agrees:
            findings.append(
                f"Auxiliary {identifier} declares {format_dimension(declared)} but its expression "
                + (f"yields {format_dimension(inferred)}" if note is None else f"is inadmissible: {note}")
            )
        auxiliary_rows.append(
            {
                "id": identifier,
                "declared": format_dimension(declared),
                "inferred": None if inferred is None else format_dimension(inferred),
                "consistent": agrees,
                "note": note,
            }
        )

    stocks = {s.id: s for s in spec.stocks}
    for flow in spec.flows:
        declared = parse_dimension(flow.dimension)
        endpoints = [e for e in (flow.from_stock, flow.to_stock) if e is not None]
        endpoint_dimensions = [parse_dimension(stocks[e].dimension) for e in endpoints]
        expected = _dim_div(endpoint_dimensions[0], time)
        consistent = True
        if len(endpoint_dimensions) == 2 and endpoint_dimensions[0] != endpoint_dimensions[1]:
            consistent = False
            findings.append(
                f"Flow {flow.id} joins stocks of different dimensions "
                f"({format_dimension(endpoint_dimensions[0])} and {format_dimension(endpoint_dimensions[1])})"
            )
        if declared != expected:
            consistent = False
            findings.append(
                f"Flow {flow.id} declares {format_dimension(declared)} but its endpoint stock "
                f"requires {format_dimension(expected)}"
            )
        try:
            inferred = _infer_dimension(flow_expressions[flow.id].tree.body, bindings)
            note = None
        except Invalid as error:
            inferred, note = None, str(error)
        if inferred != declared:
            consistent = False
            findings.append(
                f"Flow {flow.id} declares {format_dimension(declared)} but its rate expression "
                + (f"yields {format_dimension(inferred)}" if note is None else f"is inadmissible: {note}")
            )
        flow_rows.append(
            {
                "id": flow.id,
                "declared": format_dimension(declared),
                "inferred": None if inferred is None else format_dimension(inferred),
                "required_by_stocks": format_dimension(expected),
                "consistent": consistent,
                "note": note,
            }
        )

    for coupling in spec.couplings:
        source = parse_dimension(stocks[coupling.source].dimension)
        target = parse_dimension(stocks[coupling.target].dimension)
        agrees = source == target
        if not agrees:
            findings.append(
                f"Coupling {coupling.id} joins stocks of different dimensions "
                f"({format_dimension(source)} and {format_dimension(target)})"
            )
        coupling_rows.append(
            {
                "id": coupling.id,
                "stock_dimension": format_dimension(source),
                "rate_constant_dimension": format_dimension(_dim_div({}, time)),
                "consistent": agrees,
            }
        )

    return {
        "consistent": not findings,
        "time_dimension": format_dimension(time),
        "stocks": {s.id: format_dimension(parse_dimension(s.dimension)) for s in spec.stocks},
        "parameters": {
            p.id: format_dimension(parse_dimension(p.dimension)) for p in spec.parameters
        },
        "auxiliaries": auxiliary_rows,
        "flows": flow_rows,
        "couplings": coupling_rows,
        "findings": findings,
        "method": "declared dimension strings parsed as products of base dimensions with rational exponents and propagated through the expression AST",
        "scope": "Dimensional bookkeeping over declared strings. It does not establish that a declared base dimension or unit is scientifically meaningful.",
    }


@dataclass(frozen=True)
class _CompiledFlow:
    id: str
    position: int
    source: int | None
    target: int | None
    expression: CompiledExpression
    delay_kind: str | None
    duration: float
    pipeline: int | None
    ledger_terms: tuple


@dataclass(frozen=True)
class _CompiledCoupling:
    id: str
    source: int
    target: int
    kind: str
    weight: float
    ledger_terms: tuple


@dataclass(frozen=True)
class _CompiledGroup:
    id: str
    name: str
    members: tuple
    pipelines: tuple
    ledger: int
    tolerance: float
    structurally_closed: bool


@dataclass(frozen=True)
class CompiledModel:
    """Host-owned executable form of a declared system. Built only by compile_model."""

    spec: SystemModel
    stock_ids: tuple
    stock_index: dict
    state_size: int
    initial: np.ndarray
    noise_sd: np.ndarray
    auxiliaries: tuple
    flows: tuple
    couplings: tuple
    groups: tuple
    dimensions: dict


def compile_model(model):
    """Validate structure, compile every expression and require dimensional consistency."""
    spec = _as_model(model)
    report = analyze_dimensions(spec)
    if not report["consistent"]:
        raise Invalid("Dimensional inconsistency: " + "; ".join(report["findings"][:4]))
    order, flow_expressions = _compile_expressions(spec)
    stock_ids = tuple(s.id for s in spec.stocks)
    stock_index = {identifier: position for position, identifier in enumerate(stock_ids)}
    count = len(stock_ids)
    delays = {d.flow_id: d for d in spec.delays}

    pipelines, cursor = {}, count
    for flow in spec.flows:
        delay = delays.get(flow.id)
        if delay is not None and delay.kind == "first_order":
            pipelines[flow.id] = cursor
            cursor += 1
    ledger_start = cursor
    group_members = {
        group.id: tuple(sorted(stock_index[i] for i in group.stock_ids))
        for group in spec.conservation_groups
    }
    member_sets = {g: set(v) for g, v in group_members.items()}
    ledger_index = {
        group.id: ledger_start + position
        for position, group in enumerate(spec.conservation_groups)
    }
    state_size = ledger_start + len(spec.conservation_groups)

    closed = dict.fromkeys(member_sets, True)
    compiled_flows = []
    for position, flow in enumerate(spec.flows):
        delay = delays.get(flow.id)
        source = None if flow.from_stock is None else stock_index[flow.from_stock]
        target = None if flow.to_stock is None else stock_index[flow.to_stock]
        terms = []
        for group in spec.conservation_groups:
            members = member_sets[group.id]
            inside_source = source is not None and source in members
            inside_target = target is not None and target in members
            if inside_source and inside_target:
                continue
            if not inside_source and not inside_target:
                continue
            closed[group.id] = False
            terms.append(
                (
                    ledger_index[group.id],
                    -1.0 if inside_source else 0.0,
                    1.0 if inside_target else 0.0,
                )
            )
        compiled_flows.append(
            _CompiledFlow(
                id=flow.id,
                position=position,
                source=source,
                target=target,
                expression=flow_expressions[flow.id],
                delay_kind=None if delay is None else delay.kind,
                duration=0.0 if delay is None else float(delay.duration),
                pipeline=pipelines.get(flow.id),
                ledger_terms=tuple(terms),
            )
        )

    compiled_couplings = []
    for coupling in spec.couplings:
        source, target = stock_index[coupling.source], stock_index[coupling.target]
        terms = []
        for group in spec.conservation_groups:
            members = member_sets[group.id]
            inside_source, inside_target = source in members, target in members
            if inside_source == inside_target:
                continue
            closed[group.id] = False
            terms.append((ledger_index[group.id], -1.0 if inside_source else 1.0))
        compiled_couplings.append(
            _CompiledCoupling(
                id=coupling.id,
                source=source,
                target=target,
                kind=coupling.kind,
                weight=float(coupling.rate_constant),
                ledger_terms=tuple(terms),
            )
        )

    compiled_groups = tuple(
        _CompiledGroup(
            id=group.id,
            name=group.name,
            members=group_members[group.id],
            pipelines=tuple(
                sorted(
                    pipelines[flow.id]
                    for flow in spec.flows
                    if flow.id in pipelines
                    and flow.from_stock is not None
                    and flow.to_stock is not None
                    and stock_index[flow.from_stock] in member_sets[group.id]
                    and stock_index[flow.to_stock] in member_sets[group.id]
                )
            ),
            ledger=ledger_index[group.id],
            tolerance=float(group.tolerance),
            structurally_closed=closed[group.id],
        )
        for group in spec.conservation_groups
    )

    initial = np.zeros(state_size)
    initial[:count] = [float(s.initial) for s in spec.stocks]
    noise = np.array([float(s.noise_sd) for s in spec.stocks])
    return CompiledModel(
        spec=spec,
        stock_ids=stock_ids,
        stock_index=stock_index,
        state_size=state_size,
        initial=initial,
        noise_sd=noise,
        auxiliaries=order,
        flows=tuple(compiled_flows),
        couplings=tuple(compiled_couplings),
        groups=compiled_groups,
        dimensions=report,
    )


# --------------------------------------------------------------------------------------
# Deterministic and stochastic integration
# --------------------------------------------------------------------------------------


def _parameter_schedule(base, intervention):
    target = intervention.value if intervention.absolute else base + intervention.value
    start = float(intervention.time)
    if intervention.mode == "step":
        return lambda t: target if t >= start else base
    stop = start + float(intervention.duration)
    if intervention.mode == "pulse":
        return lambda t: target if start <= t < stop else base
    span = stop - start

    def ramp(t):
        if t <= start:
            return base
        if t >= stop:
            return target
        return base + (target - base) * (t - start) / span

    return ramp


def _stock_schedule(intervention):
    start = float(intervention.time)
    stop = start + float(intervention.duration)
    magnitude = float(intervention.value)
    if intervention.mode == "pulse":
        rate = magnitude / (stop - start)
        return lambda t: rate if start <= t < stop else 0.0
    span = stop - start

    def ramp(t):
        if t <= start:
            return 0.0
        if t >= stop:
            return magnitude
        return magnitude * (t - start) / span

    return ramp


class _Runtime:
    """Per-run evaluator. Holds compiled expressions and, for fixed lags, a rate history."""

    def __init__(self, compiled, interventions):
        self.compiled = compiled
        self.size = compiled.state_size
        self.stock_count = len(compiled.stock_ids)
        self.group_members = {g.ledger: g.members for g in compiled.groups}
        base = {p.id: float(p.value) for p in compiled.spec.parameters}
        self.parameter_functions = {}
        for identifier, value in base.items():
            applied = [
                i
                for i in interventions
                if i.target_kind == "parameter" and i.target_id == identifier
            ]
            self.parameter_functions[identifier] = (
                _parameter_schedule(value, applied[0])
                if applied
                else (lambda t, v=value: v)
            )
        self.stock_flux, self.jumps = [], []
        for intervention in interventions:
            if intervention.target_kind != "stock":
                continue
            index = compiled.stock_index[intervention.target_id]
            ledgers = tuple(
                ledger for ledger, members in self.group_members.items() if index in members
            )
            if intervention.mode == "step":
                self.jumps.append(
                    (float(intervention.time), index, float(intervention.value), intervention.absolute, ledgers)
                )
            else:
                self.stock_flux.append((index, _stock_schedule(intervention), ledgers))
        self.history_times, self.history_rates = [], []

    def environment(self, t, y):
        bindings = {"t": float(t)}
        for position, identifier in enumerate(self.compiled.stock_ids):
            bindings[identifier] = float(y[position])
        for identifier, schedule in self.parameter_functions.items():
            bindings[identifier] = schedule(t)
        for identifier, expression in self.compiled.auxiliaries:
            bindings[identifier] = expression.evaluate(bindings)
        return bindings

    def raw_rates(self, t, y):
        bindings = self.environment(t, y)
        return np.array(
            [flow.expression.evaluate(bindings) for flow in self.compiled.flows]
        )

    def record(self, t, y):
        self.history_times.append(float(t))
        self.history_rates.append(self.raw_rates(t, y))

    def lagged(self, position, duration, t):
        target = t - duration
        times = self.history_times
        if not times or target <= times[0]:
            return float(self.history_rates[0][position]) if self.history_rates else 0.0
        if target >= times[-1]:
            return float(self.history_rates[-1][position])
        upper = bisect.bisect_right(times, target)
        lower = upper - 1
        span = times[upper] - times[lower]
        weight = 0.0 if span <= 0 else (target - times[lower]) / span
        low = float(self.history_rates[lower][position])
        high = float(self.history_rates[upper][position])
        return low + weight * (high - low)

    def derivative(self, t, y):
        bindings = self.environment(t, y)
        derivative = np.zeros(self.size)
        for flow in self.compiled.flows:
            raw = flow.expression.evaluate(bindings)
            if flow.delay_kind == "first_order":
                effective = y[flow.pipeline] / flow.duration
                derivative[flow.pipeline] += raw - effective
                departing = raw
            elif flow.delay_kind == "fixed_lag":
                effective = self.lagged(flow.position, flow.duration, t)
                departing = effective
            else:
                effective = departing = raw
            if flow.source is not None:
                derivative[flow.source] -= departing
            if flow.target is not None:
                derivative[flow.target] += effective
            for ledger, out_coefficient, in_coefficient in flow.ledger_terms:
                derivative[ledger] += out_coefficient * departing + in_coefficient * effective
        for coupling in self.compiled.couplings:
            flux = coupling.weight * (
                y[coupling.source]
                if coupling.kind == "transfer"
                else y[coupling.source] - y[coupling.target]
            )
            derivative[coupling.source] -= flux
            derivative[coupling.target] += flux
            for ledger, coefficient in coupling.ledger_terms:
                derivative[ledger] += coefficient * flux
        for index, schedule, ledgers in self.stock_flux:
            flux = schedule(t)
            derivative[index] += flux
            for ledger in ledgers:
                derivative[ledger] += flux
        if not np.all(np.isfinite(derivative)):
            raise Invalid("Declared dynamics produced a nonfinite derivative")
        return derivative


def _apply_jumps(y, jumps):
    for _, index, value, absolute, ledgers in jumps:
        delta = (value - y[index]) if absolute else value
        y[index] += delta
        for ledger in ledgers:
            y[ledger] += delta


def _rk4_step(derivative, t, y, dt):
    k1 = derivative(t, y)
    k2 = derivative(t + dt / 2, y + (dt / 2) * k1)
    k3 = derivative(t + dt / 2, y + (dt / 2) * k2)
    k4 = derivative(t + dt, y + dt * k3)
    return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def _integrate_fixed(runtime, grid, method, stochastic, generator):
    compiled = runtime.compiled
    count = runtime.stock_count
    trajectory = np.zeros((len(grid), runtime.size))
    y = compiled.initial.copy()
    schedule = {}
    for jump in runtime.jumps:
        index = int(np.searchsorted(grid, jump[0] - 1e-12, side="left"))
        schedule.setdefault(min(index, len(grid) - 1), []).append(jump)
    _apply_jumps(y, schedule.get(0, []))
    runtime.record(grid[0], y)
    trajectory[0] = y
    noise = compiled.noise_sd
    for step in range(len(grid) - 1):
        dt = float(grid[step + 1] - grid[step])
        if method == "rk4":
            y = _rk4_step(runtime.derivative, float(grid[step]), y, dt)
        else:
            y = y + dt * runtime.derivative(float(grid[step]), y)
        if stochastic:
            # Draw the full innovation vector every step so scenarios share one stream.
            draw = generator.standard_normal(count)
            shock = noise * math.sqrt(dt) * draw
            y[:count] += shock
            for group in compiled.groups:
                y[group.ledger] += float(shock[list(group.members)].sum())
        _apply_jumps(y, schedule.get(step + 1, []))
        runtime.record(grid[step + 1], y)
        trajectory[step + 1] = y
    return trajectory


def _integrate_adaptive(runtime, grid, breakpoints, rtol, atol):
    from scipy.integrate import solve_ivp

    trajectory = np.zeros((len(grid), runtime.size))
    y = runtime.compiled.initial.copy()
    horizon = float(grid[-1])
    tolerance = 1e-12 * max(1.0, horizon)
    start = float(grid[0])
    _apply_jumps(y, [j for j in runtime.jumps if j[0] <= start + tolerance])
    trajectory[0] = y
    for stop in breakpoints[1:]:
        stop = float(stop)
        if stop <= start + tolerance:
            continue
        low = int(np.searchsorted(grid, start + tolerance, side="left"))
        high = int(np.searchsorted(grid, stop + tolerance, side="right"))
        indices = [k for k in range(low, high) if k < len(grid)]
        evaluations = [min(float(grid[k]), stop) for k in indices]
        if not evaluations or evaluations[-1] < stop - tolerance:
            evaluations.append(stop)
        solution = solve_ivp(
            runtime.derivative,
            (start, stop),
            y,
            method="RK45",
            t_eval=evaluations,
            rtol=rtol,
            atol=atol,
        )
        if not solution.success:
            raise Invalid("Adaptive integration failed: " + str(solution.message))
        for position, k in enumerate(indices):
            trajectory[k] = solution.y[:, position]
        y = solution.y[:, -1].copy()
        pending = [j for j in runtime.jumps if abs(j[0] - stop) <= tolerance]
        if pending:
            _apply_jumps(y, pending)
            if indices and abs(float(grid[indices[-1]]) - stop) <= tolerance:
                trajectory[indices[-1]] = y
        start = stop
    return trajectory


def _breakpoints(interventions, horizon):
    times = {0.0, float(horizon)}
    for intervention in interventions:
        for moment in (intervention.time, None if intervention.duration is None else intervention.time + intervention.duration):
            if moment is not None and 0.0 < float(moment) < float(horizon):
                times.add(float(moment))
    return sorted(times)


# --------------------------------------------------------------------------------------
# Host-recomputed invariants
# --------------------------------------------------------------------------------------


def _group_totals(compiled, group, trajectory):
    columns = list(group.members) + list(group.pipelines)
    return trajectory[:, columns].sum(axis=1)


def _conservation(compiled, group, trajectory):
    total = _group_totals(compiled, group, trajectory)
    ledger = trajectory[:, group.ledger]
    residual = (total - ledger) - (total[0] - ledger[0])
    return total, residual


def _network(compiled):
    parent = list(range(len(compiled.stock_ids)))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left, right):
        parent[find(left)] = find(right)

    degree = dict.fromkeys(compiled.stock_ids, 0)
    boundary = 0
    for flow in compiled.flows:
        if flow.source is not None:
            degree[compiled.stock_ids[flow.source]] += 1
        if flow.target is not None:
            degree[compiled.stock_ids[flow.target]] += 1
        if flow.source is None or flow.target is None:
            boundary += 1
        else:
            union(flow.source, flow.target)
    for coupling in compiled.couplings:
        degree[compiled.stock_ids[coupling.source]] += 1
        degree[compiled.stock_ids[coupling.target]] += 1
        union(coupling.source, coupling.target)
    return {
        "stock_count": len(compiled.stock_ids),
        "flow_count": len(compiled.flows),
        "coupling_count": len(compiled.couplings),
        "boundary_flow_count": boundary,
        "connected_components": len({find(n) for n in range(len(compiled.stock_ids))}),
        "degree": degree,
        "pipeline_state_count": sum(1 for f in compiled.flows if f.pipeline is not None),
        "ledger_state_count": len(compiled.groups),
    }


def _run_once(compiled, interventions, grid, integrator, stochastic, seed, replicate, rtol, atol, breakpoints):
    runtime = _Runtime(compiled, interventions)
    if integrator == "rk45":
        return _integrate_adaptive(runtime, grid, breakpoints, rtol, atol)
    generator = np.random.default_rng([int(seed), int(replicate)]) if stochastic else None
    method = "rk4" if integrator == "rk4" else "euler"
    return _integrate_fixed(runtime, grid, method, stochastic, generator)


def _refinement(compiled, scenario_interventions, grid, integrator, steps, horizon, rtol, atol, breakpoints, coarse):
    """Halve the step (or tighten the tolerance) and report the change in the trajectory."""
    if integrator == "rk45":
        tight_rtol, tight_atol = max(rtol * 1e-2, 1e-13), max(atol * 1e-2, 1e-16)
        fine = _run_once(
            compiled, scenario_interventions, grid, integrator, False, 0, 0, tight_rtol, tight_atol, breakpoints
        )
        method, control = "tolerance_tightening", {"rtol": tight_rtol, "atol": tight_atol}
        comparison = fine
    else:
        if 2 * steps > MAX_STEPS:
            return {"status": "skipped", "reason": "refined step count exceeds the permitted bound"}
        fine_grid = np.linspace(0.0, horizon, 2 * steps + 1)
        fine = _run_once(
            compiled, scenario_interventions, fine_grid, integrator, False, 0, 0, rtol, atol, breakpoints
        )
        method, control = "step_halving", {"steps": 2 * steps}
        comparison = fine[::2]
    count = len(compiled.stock_ids)
    difference = np.abs(comparison[:, :count] - coarse[:, :count])
    scale = np.maximum(np.abs(coarse[:, :count]).max(axis=0), 1.0)
    return {
        "status": "checked",
        "method": method,
        "control": control,
        "max_absolute_difference": float(difference.max()) if difference.size else 0.0,
        "max_relative_difference": float((difference.max(axis=0) / scale).max()) if difference.size else 0.0,
        "per_stock_max_absolute_difference": {
            identifier: float(difference[:, position].max())
            for position, identifier in enumerate(compiled.stock_ids)
        },
        "interpretation": "Difference between two integrations of the same declared model. It bounds discretisation error, not model error.",
    }


def simulate(
    model,
    steps=200,
    replicates=1,
    seed=2026,
    integrator="rk4",
    stochastic=False,
    horizon=None,
    rtol=1e-8,
    atol=1e-10,
    refine=True,
):
    """Integrate a declared system and recompute host-owned numerical invariants.

    Deterministic for a fixed model, integrator, grid and seed. Stochastic replicates
    reuse one innovation stream per replicate index across scenarios, so a scenario
    difference is paired rather than confounded with a different noise draw.
    """
    spec = _as_model(model)
    if integrator not in ("rk4", "rk45", "euler_maruyama"):
        raise Invalid("Unsupported integrator")
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= MAX_STEPS:
        raise Invalid(f"Simulation envelope exceeded: steps must be 1-{MAX_STEPS}")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or not 1 <= replicates <= MAX_REPLICATES:
        raise Invalid(f"Simulation envelope exceeded: replicates must be 1-{MAX_REPLICATES}")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise Invalid("Seed must be a 32-bit nonnegative integer")
    if stochastic and integrator != "euler_maruyama":
        raise Invalid("Stochastic forcing requires the euler_maruyama integrator")
    if not stochastic and replicates != 1:
        raise Invalid("Deterministic integration admits exactly one replicate")
    if not 1e-13 <= rtol <= 1e-3 or not 1e-16 <= atol <= 1e-3:
        raise Invalid("Adaptive tolerances lie outside the permitted range")
    horizon = float(spec.horizon if horizon is None else horizon)
    if not math.isfinite(horizon) or horizon <= 0:
        raise Invalid("Horizon must be a positive finite number")

    compiled = compile_model(spec)
    samples = len(spec.scenarios) * replicates * (steps + 1) * compiled.state_size
    if samples > MAX_STATE_SAMPLES:
        raise Invalid("Simulation envelope exceeded: total state samples above 2000000")
    dt = horizon / steps
    lagged = [f for f in compiled.flows if f.delay_kind == "fixed_lag"]
    if lagged and integrator == "rk45":
        raise Invalid("Fixed-lag delays require a fixed-step integrator")
    for flow in lagged:
        if flow.duration < dt - 1e-12:
            raise Invalid(
                f"Fixed-lag duration on flow {flow.id} is shorter than the integration step"
            )

    grid = np.linspace(0.0, horizon, steps + 1)
    by_id = {i.id: i for i in spec.interventions}
    count = len(compiled.stock_ids)
    runs, results = {}, {}
    for scenario in spec.scenarios:
        interventions = [by_id[i] for i in scenario.intervention_ids]
        breakpoints = _breakpoints(interventions, horizon)
        runs[scenario.id] = [
            _run_once(
                compiled, interventions, grid, integrator, stochastic, seed, replicate, rtol, atol, breakpoints
            )
            for replicate in range(replicates)
        ]

    baseline_id = spec.scenarios[0].id
    baseline_final = np.array([r[-1, :count] for r in runs[baseline_id]])
    trajectories = []
    for scenario in spec.scenarios:
        stack = np.stack(runs[scenario.id])
        mean = stack.mean(axis=0)
        finals = stack[:, -1, :count]
        difference = finals - baseline_final
        results[scenario.id] = {
            "name": scenario.name,
            "interpretation": scenario.interpretation,
            "intervention_ids": list(scenario.intervention_ids),
            "mean_trajectory": {
                identifier: [float(v) for v in mean[:, position]]
                for position, identifier in enumerate(compiled.stock_ids)
            },
            "final_mean": {
                identifier: float(finals[:, position].mean())
                for position, identifier in enumerate(compiled.stock_ids)
            },
            "final_by_replicate": {
                identifier: [float(v) for v in finals[:, position]]
                for position, identifier in enumerate(compiled.stock_ids)
            },
            "paired_difference": {
                identifier: float(difference[:, position].mean())
                for position, identifier in enumerate(compiled.stock_ids)
            },
            "fraction_above_baseline": {
                identifier: float((difference[:, position] > 0).mean())
                for position, identifier in enumerate(compiled.stock_ids)
            },
        }
        for replicate in range(replicates):
            trajectories.append(
                {
                    "scenario": scenario.id,
                    "replicate": replicate,
                    "stocks": {
                        identifier: [float(v) for v in runs[scenario.id][replicate][:, position]]
                        for position, identifier in enumerate(compiled.stock_ids)
                    },
                }
            )

    conservation = []
    for group in compiled.groups:
        members = set(group.members)
        noisy = stochastic and bool(np.any(compiled.noise_sd[list(group.members)] > 0))
        worst, worst_relative, per_scenario = 0.0, 0.0, {}
        for scenario in spec.scenarios:
            local = 0.0
            for trajectory in runs[scenario.id]:
                total, residual = _conservation(compiled, group, trajectory)
                local = max(local, float(np.abs(residual).max()))
                scale = max(abs(float(total[0])), 1e-30)
                worst_relative = max(worst_relative, float(np.abs(residual).max()) / scale)
            worst = max(worst, local)
            touched = any(
                by_id[i].target_kind == "stock" and compiled.stock_index[by_id[i].target_id] in members
                for i in scenario.intervention_ids
            )
            per_scenario[scenario.id] = {
                "closed": group.structurally_closed and not touched and not noisy,
                "initial_total": float(_group_totals(compiled, group, runs[scenario.id][0])[0]),
                "max_absolute_residual": local,
                "within_tolerance": local <= group.tolerance,
            }
        conservation.append(
            {
                "id": group.id,
                "name": group.name,
                "stock_ids": [compiled.stock_ids[m] for m in group.members],
                "internal_pipeline_states": len(group.pipelines),
                "initial_total": per_scenario[baseline_id]["initial_total"],
                "structurally_closed": group.structurally_closed,
                "closed": all(entry["closed"] for entry in per_scenario.values()),
                "per_scenario": per_scenario,
                "max_absolute_residual": worst,
                "max_relative_residual": worst_relative,
                "tolerance": group.tolerance,
                "within_tolerance": worst <= group.tolerance,
                "definition": "total of member stocks and internal delay pipelines, minus the integrated external flux ledger, minus the same quantity at t=0",
            }
        )

    negativity = []
    for position, stock in enumerate(spec.stocks):
        if not stock.non_negative:
            continue
        minimum = min(
            float(trajectory[:, position].min())
            for scenario in spec.scenarios
            for trajectory in runs[scenario.id]
        )
        negativity.append(
            {
                "id": stock.id,
                "declared_non_negative": True,
                "minimum": minimum,
                "violation": max(0.0, -minimum),
                "violated": minimum < 0.0,
            }
        )

    refinement = {"status": "skipped", "reason": "not requested"}
    if not refine:
        pass
    elif stochastic:
        refinement = {"status": "skipped", "reason": "stochastic forcing; refinement compares deterministic drift only"}
    else:
        refinement = {}
        for scenario in spec.scenarios:
            interventions = [by_id[i] for i in scenario.intervention_ids]
            refinement[scenario.id] = _refinement(
                compiled,
                interventions,
                grid,
                integrator,
                steps,
                horizon,
                rtol,
                atol,
                _breakpoints(interventions, horizon),
                runs[scenario.id][0],
            )

    return {
        "title": spec.title,
        "boundary": spec.boundary,
        "protocol": PROTOCOL,
        "integrator": integrator,
        "stochastic": bool(stochastic),
        "steps": steps,
        "step_size": dt,
        "horizon": horizon,
        "time_unit": spec.time_unit,
        "time_dimension": spec.time_dimension,
        "replicates": replicates,
        "seed": seed,
        "times": [float(t) for t in grid],
        "stocks": [
            {
                "id": s.id,
                "name": s.name,
                "unit": s.unit,
                "dimension": s.dimension,
                "initial": float(s.initial),
                "non_negative": bool(s.non_negative),
            }
            for s in spec.stocks
        ],
        "scenarios": results,
        "trajectories": trajectories,
        "diagnostics": {
            "conservation": conservation,
            "non_negativity": negativity,
            "step_refinement": refinement,
            "dimensions": compiled.dimensions,
            "network": _network(compiled),
            "expressions": {
                "allowed_functions": list(ALLOWED_FUNCTIONS),
                "compiled_flow_count": len(compiled.flows),
                "compiled_auxiliary_count": len(compiled.auxiliaries),
                "execution": "closed arithmetic AST; no agent-authored Python is executed",
            },
        },
        "model_digest": digest(spec.model_dump()),
        "units": "declared per-stock units; time in " + spec.time_unit,
        "equation": "dx/dt = sum(inflow rates) - sum(outflow rates) + coupling flux + declared intervention flux; euler_maruyama adds noise_sd*sqrt(dt)*N(0,1) per stock",
        "assumptions": list(spec.assumptions),
        "unmodeled": list(spec.unmodeled),
        "verdict": "inconclusive",
        "independently_validated": False,
        "empirically_validated": False,
        "synthetic": True,
        "scope": SCOPE,
    }


# --------------------------------------------------------------------------------------
# Tidy CSV emission for the host numerical checks
# --------------------------------------------------------------------------------------


def trajectory_csv(result):
    """Emit tidy CSV: time, scenario, replicate and one column per declared stock."""
    if not isinstance(result, dict) or not {"trajectories", "times", "stocks"} <= set(result):
        raise Invalid("Expected a simulate() result")
    identifiers = [s["id"] for s in result["stocks"]]
    times = result["times"]
    rows = len(result["trajectories"]) * len(times)
    if rows > MAX_CSV_ROWS:
        raise Invalid(f"Trajectory CSV exceeds the {MAX_CSV_ROWS}-row limit")
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["time", "scenario", "replicate", *identifiers])
    for entry in result["trajectories"]:
        columns = [entry["stocks"][i] for i in identifiers]
        for position, moment in enumerate(times):
            writer.writerow(
                [repr(float(moment)), entry["scenario"], str(entry["replicate"])]
                + [repr(float(column[position])) for column in columns]
            )
    return buffer.getvalue()


def numeric_check_specs(result, csv_filename="trajectory.csv"):
    """Build NumericCheck-shaped dicts matching trajectory_csv, for host recomputation.

    Returned as plain dicts so the evaluation layer stays the sole owner of the check
    contract. A check that passes says the emitted numbers behaved, nothing more.
    """
    identifiers = [s["id"] for s in result["stocks"]]
    grouping = ["scenario", "replicate"]
    specs = [
        {
            "id": "finite_states",
            "operation": "finite",
            "csv_filename": csv_filename,
            "columns": identifiers,
            "group_columns": grouping,
            "time_column": "time",
            "row_filters": [],
            "lower": None,
            "upper": None,
            "reference_value": None,
            "tolerance": 0.0,
            "direction": None,
            "units": "declared stock units",
            "rationale": "Every emitted state value must be a finite number under the frozen integrator.",
        }
    ]
    bounded = [s["id"] for s in result["stocks"] if s["non_negative"]]
    if bounded:
        specs.append(
            {
                "id": "non_negative_stocks",
                "operation": "bounds",
                "csv_filename": csv_filename,
                "columns": bounded,
                "group_columns": grouping,
                "time_column": "time",
                "row_filters": [],
                "lower": 0.0,
                "upper": None,
                "reference_value": None,
                "tolerance": 0.0,
                "direction": None,
                "units": "declared stock units",
                "rationale": "Stocks declared non-negative are checked against the emitted trajectory; the integrator does not clamp them.",
            }
        )
    for group in result["diagnostics"]["conservation"]:
        if group["internal_pipeline_states"]:
            # Material held in a delay pipeline is not a CSV column, so the row sum is not the group total.
            continue
        for scenario, entry in group["per_scenario"].items():
            if not entry["closed"]:
                continue
            specs.append(
                {
                    "id": f"conserved_{group['id']}_{scenario}"[:64],
                    "operation": "sum_conservation",
                    "csv_filename": csv_filename,
                    "columns": list(group["stock_ids"]),
                    "group_columns": grouping,
                    "time_column": "time",
                    "row_filters": [{"column": "scenario", "equals": scenario}],
                    "lower": None,
                    "upper": None,
                    "reference_value": entry["initial_total"],
                    "tolerance": max(group["tolerance"], 0.0),
                    "direction": None,
                    "units": "declared stock units",
                    "rationale": "A closed conservation group has no declared external flux, so its member total must hold at the declared tolerance.",
                }
            )
    return specs
