"""Dimensional algebra that makes a declared unit checkable instead of readable.

`UnitDefinition.dimension` in `symplex/modeling/complex_system.py` is documented as a
"canonical declared quantity dimension" written as a product of base dimensions. Nothing
parsed it: the host compared two dimension strings with `!=`, so `length/time` and
`length*time^-1` were treated as incompatible, and `mass` plus `time` was never noticed.
This module parses that text into an exponent vector, gives it algebra, and refuses the
arithmetic the declared dimension forbids.

Base dimensions are the seven SI bases (BIPM, SI Brochure 9th edition, table 1) plus two
pseudo-dimensions this product actually needs: `currency` (economics and market domains)
and `count` (agent and population counts). A currency exponent does not imply that two
currencies are interconvertible; it only keeps money from being added to mass. A count is
not a mole and is deliberately not reducible to `amount`.

Convention in this module: module-level public functions return a dict carrying a `scope`
string. The `Dimension` and `Unit` value types are values, not reports; their operators
return values and raise `Invalid` on a contract violation. `Dimension.parse` and `unit()`
are constructors/registry accessors, and are the two documented exceptions.

Nothing here establishes that a declared dimension is the scientifically correct one for a
quantity. It establishes only that the declarations are internally consistent.
"""

import ast
import math
import re
from dataclasses import dataclass, field
from fractions import Fraction

from symplex.core.contracts import Invalid

BASE_DIMENSIONS = (
    "mass",
    "length",
    "time",
    "current",
    "temperature",
    "amount",
    "luminous_intensity",
    "currency",
    "count",
)

MAX_DIMENSION_CHARS = 400
MAX_DIMENSION_TOKENS = 128
MAX_EXPONENT = 12
MAX_PARSE_DEPTH = 16
MAX_EXPRESSION_CHARS = 2000
MAX_EXPRESSION_NODES = 400

SCOPE_DIMENSION = (
    "Internal consistency of declared dimensions only. Parsing a dimension does not "
    "establish that it is the right dimension for the quantity, that the quantity was "
    "measured, or that the model using it is correct."
)
SCOPE_CONVERSION = (
    "Exact affine conversion between two declared units of one dimension. Not a check "
    "that either declared unit describes the observed quantity, and not a currency, "
    "calendar or coordinate-system conversion."
)
SCOPE_EXPRESSION = (
    "Dimensional consistency of the written expression under the supplied environment. "
    "A dimensionally consistent expression can still be the wrong model; a passing check "
    "is a necessary condition, never evidence of correctness."
)

_ALIASES = {
    "mass": "mass",
    "length": "length",
    "distance": "length",
    "time": "time",
    "duration": "time",
    "current": "current",
    "electric_current": "current",
    "temperature": "temperature",
    "thermodynamic_temperature": "temperature",
    "amount": "amount",
    "amount_of_substance": "amount",
    "substance": "amount",
    "luminous_intensity": "luminous_intensity",
    "luminosity": "luminous_intensity",
    "currency": "currency",
    "money": "currency",
    "monetary": "currency",
    "count": "count",
    "cardinality": "count",
    "headcount": "count",
}
_DIMENSIONLESS_WORDS = frozenset(
    {"dimensionless", "unitless", "none", "scalar", "pure_number", "one"}
)
_TOKEN = re.compile(r"\s*(\*\*|[*/^()]|[·⋅]|[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|[-+])")
_TIMES = frozenset({"*", "·", "⋅"})
_POWER = frozenset({"^", "**"})


def _exponent_text(exponent: Fraction) -> str:
    if exponent.denominator == 1:
        return str(exponent.numerator)
    text = repr(float(exponent))
    if Fraction(text) != exponent:
        raise Invalid("Exponent has no exact decimal form: " + str(exponent))
    return text


def _checked_exponent(exponent) -> Fraction:
    exponent = Fraction(exponent)
    if abs(exponent) > MAX_EXPONENT:
        raise Invalid(f"Dimension exponent exceeds the +/-{MAX_EXPONENT} envelope")
    _exponent_text(exponent)
    return exponent


@dataclass(frozen=True, order=False)
class Dimension:
    """An exponent vector over BASE_DIMENSIONS. Equality is exponent equality, not text."""

    exponents: tuple = field(default=())

    def __post_init__(self):
        cleaned = []
        seen = set()
        for name, exponent in self.exponents:
            if name not in BASE_DIMENSIONS:
                raise Invalid("Unknown base dimension: " + str(name))
            if name in seen:
                raise Invalid("Repeated base dimension: " + name)
            seen.add(name)
            exponent = _checked_exponent(exponent)
            if exponent:
                cleaned.append((name, exponent))
        cleaned.sort(key=lambda pair: BASE_DIMENSIONS.index(pair[0]))
        object.__setattr__(self, "exponents", tuple(cleaned))

    @classmethod
    def of(cls, mapping) -> "Dimension":
        return cls(tuple(mapping.items() if hasattr(mapping, "items") else mapping))

    @classmethod
    def base(cls, name: str) -> "Dimension":
        return cls(((_ALIASES.get(name, name), Fraction(1)),))

    @classmethod
    def dimensionless(cls) -> "Dimension":
        return cls(())

    @classmethod
    def parse(cls, text) -> "Dimension":
        """Parse a declared dimension product. Constructor, so it returns a Dimension."""
        return _parse_dimension_text(text)

    @property
    def is_dimensionless(self) -> bool:
        return not self.exponents

    @property
    def canonical(self) -> str:
        if not self.exponents:
            return "dimensionless"
        parts = []
        for name, exponent in self.exponents:
            parts.append(
                name if exponent == 1 else f"{name}^{_exponent_text(exponent)}"
            )
        return "*".join(parts)

    @property
    def vector(self) -> tuple:
        table = dict(self.exponents)
        return tuple(float(table.get(name, 0)) for name in BASE_DIMENSIONS)

    def as_dict(self) -> dict:
        return {name: _exponent_text(exponent) for name, exponent in self.exponents}

    def __mul__(self, other: "Dimension") -> "Dimension":
        if not isinstance(other, Dimension):
            return NotImplemented
        table = dict(self.exponents)
        for name, exponent in other.exponents:
            table[name] = table.get(name, Fraction(0)) + exponent
        return Dimension.of(table)

    def __truediv__(self, other: "Dimension") -> "Dimension":
        if not isinstance(other, Dimension):
            return NotImplemented
        return self * other**-1

    def __pow__(self, exponent) -> "Dimension":
        exponent = Fraction(exponent)
        return Dimension.of({name: value * exponent for name, value in self.exponents})

    def __str__(self) -> str:
        return self.canonical

    def __repr__(self) -> str:
        return f"Dimension({self.canonical!r})"


DIMENSIONLESS = Dimension.dimensionless()


class _Reader:
    def __init__(self, text):
        self.tokens = []
        position = 0
        while position < len(text):
            match = _TOKEN.match(text, position)
            if match is None:
                if text[position:].strip():
                    raise Invalid(
                        "Unreadable character in dimension: " + repr(text[position])
                    )
                break
            self.tokens.append(match.group(1))
            position = match.end()
            if len(self.tokens) > MAX_DIMENSION_TOKENS:
                raise Invalid("Dimension expression exceeds the token envelope")
        self.index = 0

    def peek(self):
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def take(self):
        token = self.peek()
        if token is None:
            raise Invalid("Dimension expression ended early")
        self.index += 1
        return token

    def expression(self, depth=0):
        if depth > MAX_PARSE_DEPTH:
            raise Invalid("Dimension expression nests too deeply")
        value = self.term(depth)
        while self.peek() in _TIMES or self.peek() == "/":
            operator = self.take()
            right = self.term(depth)
            value = value * right if operator in _TIMES else value / right
        return value

    def term(self, depth):
        value = self.factor(depth)
        if self.peek() in _POWER:
            self.take()
            value = value ** self.exponent()
        return value

    def factor(self, depth):
        token = self.take()
        if token == "(":
            value = self.expression(depth + 1)
            if self.take() != ")":
                raise Invalid("Unbalanced parenthesis in dimension")
            return value
        if token[0].isalpha() or token[0] == "_":
            lowered = token.lower()
            if lowered in _DIMENSIONLESS_WORDS:
                return DIMENSIONLESS
            if lowered in _ALIASES:
                return Dimension.base(_ALIASES[lowered])
            raise Invalid(
                "Unknown base dimension "
                + repr(token)
                + "; permitted: "
                + ", ".join(BASE_DIMENSIONS)
                + ", dimensionless"
            )
        if token[0].isdigit():
            if Fraction(token) != 1:
                raise Invalid(
                    "A dimension product carries no numeric factor other than 1: "
                    + token
                )
            return DIMENSIONLESS
        raise Invalid("Unexpected token in dimension: " + repr(token))

    def exponent(self):
        sign = 1
        while self.peek() in ("-", "+"):
            if self.take() == "-":
                sign = -sign
        token = self.take()
        if not token[0].isdigit():
            raise Invalid("Dimension exponent must be a number: " + repr(token))
        return _checked_exponent(sign * Fraction(token))


def _parse_dimension_text(text) -> Dimension:
    if isinstance(text, Dimension):
        return text
    if not isinstance(text, str) or not text.strip():
        raise Invalid("Expected a nonempty declared dimension string")
    if len(text) > MAX_DIMENSION_CHARS:
        raise Invalid(f"Dimension string exceeds {MAX_DIMENSION_CHARS} characters")
    reader = _Reader(text)
    if not reader.tokens:
        raise Invalid("Expected a nonempty declared dimension string")
    value = reader.expression()
    if reader.peek() is not None:
        raise Invalid("Trailing text in dimension: " + repr(reader.peek()))
    return value


def parse_dimension(text) -> dict:
    """Parse a declared dimension product into an exponent vector over BASE_DIMENSIONS."""
    dimension = _parse_dimension_text(text)
    return {
        "input": text if isinstance(text, str) else dimension.canonical,
        "canonical": dimension.canonical,
        "exponents": dimension.as_dict(),
        "vector": list(dimension.vector),
        "basis": list(BASE_DIMENSIONS),
        "dimensionless": dimension.is_dimensionless,
        "round_trips": Dimension.parse(dimension.canonical) == dimension,
        "scope": SCOPE_DIMENSION,
    }


@dataclass(frozen=True)
class Unit:
    """A declared unit: canonical = scale_to_canonical * value + offset_to_canonical.

    The field names mirror `UnitDefinition` in symplex/modeling/complex_system.py so a
    saved spec can be lifted into this algebra without translation.
    """

    id: str
    symbol: str
    dimension: Dimension
    scale_to_canonical: float = 1.0
    offset_to_canonical: float = 0.0
    note: str = ""

    def __post_init__(self):
        if not isinstance(self.dimension, Dimension):
            object.__setattr__(self, "dimension", _parse_dimension_text(self.dimension))
        scale = float(self.scale_to_canonical)
        offset = float(self.offset_to_canonical)
        if not math.isfinite(scale) or scale <= 0:
            raise Invalid("scale_to_canonical must be a positive finite number")
        if not math.isfinite(offset):
            raise Invalid("offset_to_canonical must be finite")
        object.__setattr__(self, "scale_to_canonical", scale)
        object.__setattr__(self, "offset_to_canonical", offset)

    @property
    def is_affine(self) -> bool:
        """True when the unit has a nonzero zero-point, such as degC or degF."""
        return self.offset_to_canonical != 0.0

    def to_canonical(self, value: float) -> float:
        return self.scale_to_canonical * float(value) + self.offset_to_canonical

    def from_canonical(self, value: float) -> float:
        return (float(value) - self.offset_to_canonical) / self.scale_to_canonical

    def _multiplicative(self, other, symbol):
        for operand in (self, other):
            if isinstance(operand, Unit) and operand.is_affine:
                raise Invalid(
                    "Refusing to compose the affine unit "
                    + operand.symbol
                    + " multiplicatively; its zero point is not at zero, so "
                    + f"'{self.symbol} {symbol} {getattr(other, 'symbol', other)}' has no "
                    "meaning. Convert to the canonical unit of its dimension first."
                )

    def __mul__(self, other: "Unit") -> "Unit":
        if not isinstance(other, Unit):
            return NotImplemented
        self._multiplicative(other, "*")
        return Unit(
            id=f"{self.id}_times_{other.id}",
            symbol=f"{self.symbol}*{other.symbol}",
            dimension=self.dimension * other.dimension,
            scale_to_canonical=self.scale_to_canonical * other.scale_to_canonical,
            offset_to_canonical=0.0,
            note="derived by composition; not a registered unit",
        )

    def __truediv__(self, other: "Unit") -> "Unit":
        if not isinstance(other, Unit):
            return NotImplemented
        self._multiplicative(other, "/")
        return Unit(
            id=f"{self.id}_per_{other.id}",
            symbol=f"{self.symbol}/{other.symbol}",
            dimension=self.dimension / other.dimension,
            scale_to_canonical=self.scale_to_canonical / other.scale_to_canonical,
            offset_to_canonical=0.0,
            note="derived by composition; not a registered unit",
        )

    def __pow__(self, exponent) -> "Unit":
        self._multiplicative(self, "^")
        power = Fraction(exponent)
        return Unit(
            id=f"{self.id}_pow_{str(power).replace('/', '_over_').replace('-', 'neg')}",
            symbol=f"{self.symbol}^{_exponent_text(power)}",
            dimension=self.dimension**power,
            scale_to_canonical=self.scale_to_canonical ** float(power),
            offset_to_canonical=0.0,
            note="derived by composition; not a registered unit",
        )


def _unit(ident, symbol, dimension, scale=1.0, offset=0.0, note=""):
    return Unit(ident, symbol, Dimension.parse(dimension), scale, offset, note)


_REGISTRY_ENTRIES = (
    _unit("one", "1", "dimensionless"),
    _unit("percent", "%", "dimensionless", 0.01),
    _unit("m", "m", "length"),
    _unit("km", "km", "length", 1000.0),
    _unit("cm", "cm", "length", 0.01),
    _unit("mm", "mm", "length", 0.001),
    _unit("s", "s", "time"),
    _unit("min", "min", "time", 60.0),
    _unit("h", "h", "time", 3600.0),
    _unit("day", "day", "time", 86400.0),
    _unit("year", "yr", "time", 31557600.0, 0.0, "Julian year of 365.25 days"),
    _unit("kg", "kg", "mass"),
    _unit("g", "g", "mass", 0.001),
    _unit("tonne", "t", "mass", 1000.0),
    _unit("k", "K", "temperature"),
    _unit("deg_c", "degC", "temperature", 1.0, 273.15),
    _unit("deg_f", "degF", "temperature", 5.0 / 9.0, 273.15 - 32.0 * 5.0 / 9.0),
    _unit("mol", "mol", "amount"),
    _unit("ampere", "A", "current"),
    _unit("candela", "cd", "luminous_intensity"),
    _unit(
        "usd",
        "USD",
        "currency",
        1.0,
        0.0,
        "Canonical currency unit. Cross-currency rates are time-varying data, not a unit conversion.",
    ),
    _unit("kusd", "kUSD", "currency", 1000.0),
    _unit("musd", "MUSD", "currency", 1000000.0),
    _unit("count", "count", "count"),
    _unit("person", "person", "count", 1.0, 0.0, "A count whose members are people."),
    _unit("thousand_persons", "kperson", "count", 1000.0),
    _unit("m2", "m^2", "length^2"),
    _unit("km2", "km^2", "length^2", 1000000.0),
    _unit("m3", "m^3", "length^3"),
    _unit("litre", "L", "length^3", 0.001),
    _unit("m_per_s", "m/s", "length/time"),
    _unit("km_per_h", "km/h", "length/time", 1000.0 / 3600.0),
    _unit("m_per_s2", "m/s^2", "length/time^2"),
    _unit("hz", "Hz", "time^-1"),
    _unit("newton", "N", "mass*length/time^2"),
    _unit("joule", "J", "mass*length^2/time^2"),
    _unit("watt", "W", "mass*length^2/time^3"),
    _unit("pascal", "Pa", "mass/(length*time^2)"),
    _unit("kg_per_m3", "kg/m^3", "mass/length^3"),
    _unit("usd_per_day", "USD/day", "currency/time", 1.0 / 86400.0),
    _unit("usd_per_person", "USD/person", "currency/count"),
    _unit("person_per_km2", "person/km^2", "count/length^2", 1e-6),
)

UNITS = {entry.id: entry for entry in _REGISTRY_ENTRIES}
_BY_SYMBOL = {}
for _entry in _REGISTRY_ENTRIES:
    _BY_SYMBOL.setdefault(_entry.symbol, _entry)
    _BY_SYMBOL.setdefault(_entry.symbol.lower(), _entry)


def unit(name) -> Unit:
    """Look a registered unit up by id or symbol. Registry accessor, so it returns a Unit."""
    if isinstance(name, Unit):
        return name
    if not isinstance(name, str) or not name.strip():
        raise Invalid("Expected a unit id or symbol")
    key = name.strip()
    for candidate in (UNITS.get(key), _BY_SYMBOL.get(key), _BY_SYMBOL.get(key.lower())):
        if candidate is not None:
            return candidate
    raise Invalid(
        "Unknown unit "
        + repr(name)
        + "; the registry is a convenience, not a complete unit system. "
        "Declare the unit explicitly with Unit(...) when it is absent."
    )


def registry() -> dict:
    """List the built-in units. Convenience only; a spec may declare units outside it."""
    return {
        "units": [
            {
                "id": entry.id,
                "symbol": entry.symbol,
                "dimension": entry.dimension.canonical,
                "scale_to_canonical": entry.scale_to_canonical,
                "offset_to_canonical": entry.offset_to_canonical,
                "affine": entry.is_affine,
                "note": entry.note,
            }
            for entry in _REGISTRY_ENTRIES
        ],
        "unit_count": len(_REGISTRY_ENTRIES),
        "basis": list(BASE_DIMENSIONS),
        "complete": False,
        "scope": (
            "A small convenience registry of common units. It is not a curated unit "
            "ontology; absence of a unit says nothing about its validity, and currency "
            "multiples are scale factors, not exchange rates."
        ),
    }


def convert(value, from_unit, to_unit) -> dict:
    """Convert between two declared units of the same dimension, refusing anything else."""
    source, target = unit(from_unit), unit(to_unit)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise Invalid("Expected a finite numeric value to convert")
    value = float(value)
    if not math.isfinite(value):
        raise Invalid("Expected a finite numeric value to convert")
    if source.dimension != target.dimension:
        raise Invalid(
            "Refusing to convert "
            + source.symbol
            + " ("
            + source.dimension.canonical
            + ") to "
            + target.symbol
            + " ("
            + target.dimension.canonical
            + "); different dimensions need an explicit modelled transformation, not a unit conversion"
        )
    canonical = source.to_canonical(value)
    result = target.from_canonical(canonical)
    if not (math.isfinite(canonical) and math.isfinite(result)):
        raise Invalid("Unit conversion exceeds finite numerical precision")
    return {
        "value": result,
        "input_value": value,
        "canonical_value": canonical,
        "from_unit": source.symbol,
        "to_unit": target.symbol,
        "dimension": source.dimension.canonical,
        "affine": source.is_affine or target.is_affine,
        "scale": source.scale_to_canonical / target.scale_to_canonical,
        "offset": (source.offset_to_canonical - target.offset_to_canonical)
        / target.scale_to_canonical,
        "scope": SCOPE_CONVERSION,
    }


def check_unit_definition(definition) -> dict:
    """Check one `UnitDefinition`-shaped mapping: parseable dimension, sane affine form."""
    if not isinstance(definition, dict):
        raise Invalid("Expected a UnitDefinition mapping")
    missing = {
        "id",
        "symbol",
        "dimension",
        "scale_to_canonical",
        "offset_to_canonical",
    } - set(definition)
    if missing:
        raise Invalid("UnitDefinition is missing: " + ", ".join(sorted(missing)))
    parsed = Unit(
        id=str(definition["id"]),
        symbol=str(definition["symbol"]),
        dimension=Dimension.parse(definition["dimension"]),
        scale_to_canonical=definition["scale_to_canonical"],
        offset_to_canonical=definition["offset_to_canonical"],
    )
    findings = []
    if parsed.is_affine and parsed.dimension != Dimension.base("temperature"):
        findings.append(
            {
                "code": "affine_non_temperature_unit",
                "severity": "warning",
                "detail": (
                    "A nonzero offset outside temperature is unusual; such a unit cannot "
                    "take part in any product or power without losing meaning."
                ),
            }
        )
    if parsed.dimension.is_dimensionless and parsed.is_affine:
        findings.append(
            {
                "code": "affine_dimensionless_unit",
                "severity": "warning",
                "detail": "A dimensionless unit with an offset is an index, not a ratio.",
            }
        )
    return {
        "id": parsed.id,
        "symbol": parsed.symbol,
        "declared_dimension": str(definition["dimension"]),
        "canonical_dimension": parsed.dimension.canonical,
        "normalized": parsed.dimension.canonical
        != str(definition["dimension"]).strip(),
        "exponents": parsed.dimension.as_dict(),
        "affine": parsed.is_affine,
        "composable": not parsed.is_affine,
        "findings": findings,
        "scope": SCOPE_DIMENSION,
    }


def check_unit_definitions(definitions) -> dict:
    """Check a spec's whole unit table and report string-distinct, dimensionally equal units.

    The host currently compares declared dimensions with string equality. Two units that
    are dimensionally identical but written differently are reported here as
    `equivalent_dimension_written_differently`; that is a false incompatibility waiting to
    happen, not a modelling error.
    """
    if not isinstance(definitions, list) or len(definitions) > 256:
        raise Invalid("Expected a list of at most 256 UnitDefinition mappings")
    checked, rejected = [], []
    by_canonical = {}
    for definition in definitions:
        try:
            report = check_unit_definition(definition)
        except Invalid as error:
            rejected.append(
                {
                    "id": (definition or {}).get("id")
                    if isinstance(definition, dict)
                    else None,
                    "declared_dimension": (definition or {}).get("dimension")
                    if isinstance(definition, dict)
                    else None,
                    "reason": str(error),
                }
            )
            continue
        checked.append(report)
        by_canonical.setdefault(report["canonical_dimension"], []).append(report)
    findings = []
    for canonical, group in sorted(by_canonical.items()):
        spellings = sorted({report["declared_dimension"] for report in group})
        if len(spellings) > 1:
            findings.append(
                {
                    "code": "equivalent_dimension_written_differently",
                    "severity": "warning",
                    "canonical_dimension": canonical,
                    "spellings": spellings,
                    "unit_ids": sorted(report["id"] for report in group),
                    "detail": (
                        "These declared dimensions are equal as exponent vectors but "
                        "differ as strings; a string comparison would reject a valid "
                        "coupling between them."
                    ),
                }
            )
    for report in checked:
        for finding in report["findings"]:
            findings.append({**finding, "unit_ids": [report["id"]]})
    return {
        "status": "checked" if not rejected else "failed",
        "units": checked,
        "rejected": rejected,
        "unit_count": len(checked),
        "distinct_dimensions": sorted(by_canonical),
        "findings": findings,
        "finding_count": len(findings),
        "scope": SCOPE_DIMENSION,
    }


_DIMENSIONLESS_FUNCTIONS = frozenset(
    {
        "exp",
        "expm1",
        "log",
        "log10",
        "log2",
        "log1p",
        "tanh",
        "sinh",
        "cosh",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "erf",
        "erfc",
        "sigmoid",
        "logit",
        "softplus",
    }
)
_SAME_DIMENSION_FUNCTIONS = frozenset({"abs", "fabs", "min", "max", "hypot"})
_UNARY_SQRT = frozenset({"sqrt"})
_RATIO_FUNCTIONS = frozenset({"atan2"})
_ARITHMETIC = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)


def _as_dimension(value) -> Dimension:
    if isinstance(value, Dimension):
        return value
    if isinstance(value, Unit):
        return value.dimension
    if isinstance(value, str):
        return Dimension.parse(value)
    raise Invalid(
        "Environment values must be a Dimension, a Unit or a declared dimension string"
    )


class _ExpressionChecker:
    def __init__(self, environment):
        self.environment = environment
        self.used = {}
        self.checks = []
        self.nodes = 0

    def note(self, code, detail):
        entry = {"code": code, "detail": detail}
        if entry not in self.checks:
            self.checks.append(entry)

    def visit(self, node) -> Dimension:
        self.nodes += 1
        if self.nodes > MAX_EXPRESSION_NODES:
            raise Invalid(
                f"Expression exceeds the {MAX_EXPRESSION_NODES}-node envelope"
            )
        method = getattr(self, "_" + type(node).__name__, None)
        if method is None:
            raise Invalid(
                "Unsupported syntax in dimensional check: " + type(node).__name__
            )
        return method(node)

    def _Expression(self, node):
        return self.visit(node.body)

    def _Constant(self, node):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise Invalid("Only finite numeric literals are permitted")
        if not math.isfinite(node.value):
            raise Invalid("Only finite numeric literals are permitted")
        return DIMENSIONLESS

    def _Name(self, node):
        if node.id not in self.environment:
            raise Invalid(
                "Symbol "
                + repr(node.id)
                + " has no declared dimension; every symbol must be declared before its "
                "expression can be checked"
            )
        dimension = _as_dimension(self.environment[node.id])
        self.used[node.id] = dimension.canonical
        return dimension

    def _UnaryOp(self, node):
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise Invalid("Only unary + and - are supported")
        return self.visit(node.operand)

    def _BinOp(self, node):
        if not isinstance(node.op, _ARITHMETIC):
            raise Invalid(
                "Only + - * / ** are supported in a dimensional check; "
                + type(node.op).__name__
                + " is not"
            )
        if isinstance(node.op, ast.Pow):
            return self._power(node)
        left, right = self.visit(node.left), self.visit(node.right)
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if left != right:
            raise Invalid(
                "Cannot add or subtract unlike dimensions: "
                + left.canonical
                + " and "
                + right.canonical
                + " in "
                + ast.unparse(node)
            )
        self.note(
            "additive_terms_share_a_dimension",
            "Every + and - joins terms of one dimension.",
        )
        return left

    def _power(self, node):
        base = self.visit(node.left)
        exponent_node = node.right
        exponent = None
        try:
            literal = ast.literal_eval(exponent_node)
            if isinstance(literal, (int, float)) and not isinstance(literal, bool):
                exponent = _checked_exponent(Fraction(str(literal)))
        except (ValueError, SyntaxError, TypeError, ZeroDivisionError):
            exponent = None
        exponent_dimension = self.visit(exponent_node)
        if not exponent_dimension.is_dimensionless:
            raise Invalid(
                "An exponent must be dimensionless, not "
                + exponent_dimension.canonical
                + " in "
                + ast.unparse(node)
            )
        if exponent is None:
            if not base.is_dimensionless:
                raise Invalid(
                    "A non-literal exponent requires a dimensionless base; "
                    + ast.unparse(node)
                    + " raises "
                    + base.canonical
                    + " to a symbolic power"
                )
            return DIMENSIONLESS
        self.note(
            "exponent_is_dimensionless",
            "Every exponent is dimensionless and, where the base is not, a literal.",
        )
        return base**exponent

    def _Compare(self, node):
        permitted = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)
        if not all(isinstance(op, permitted) for op in node.ops):
            raise Invalid("Only ordering and equality comparisons are supported")
        left = self.visit(node.left)
        for comparator in node.comparators:
            right = self.visit(comparator)
            if left != right:
                raise Invalid(
                    "Cannot compare unlike dimensions: "
                    + left.canonical
                    + " and "
                    + right.canonical
                    + " in "
                    + ast.unparse(node)
                )
            left = right
        self.note(
            "comparisons_share_a_dimension",
            "Every comparison holds between terms of one dimension.",
        )
        return DIMENSIONLESS

    def _Call(self, node):
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise Invalid("Only plain calls to whitelisted functions are supported")
        name = node.func.id
        arguments = [self.visit(argument) for argument in node.args]
        if not arguments:
            raise Invalid("Function " + name + " needs at least one argument")
        if name in _DIMENSIONLESS_FUNCTIONS:
            if len(arguments) != 1:
                raise Invalid(name + " takes exactly one argument")
            if not arguments[0].is_dimensionless:
                raise Invalid(
                    name
                    + " requires a dimensionless argument, but received "
                    + arguments[0].canonical
                    + " in "
                    + ast.unparse(node)
                    + "; divide by a reference quantity of the same dimension first"
                )
            self.note(
                "transcendental_arguments_dimensionless",
                "exp/log/tanh-family arguments are dimensionless.",
            )
            return DIMENSIONLESS
        if name in _UNARY_SQRT:
            if len(arguments) != 1:
                raise Invalid("sqrt takes exactly one argument")
            return arguments[0] ** Fraction(1, 2)
        if name in _SAME_DIMENSION_FUNCTIONS:
            for argument in arguments[1:]:
                if argument != arguments[0]:
                    raise Invalid(
                        name
                        + " requires arguments of one dimension, but received "
                        + ", ".join(sorted({a.canonical for a in arguments}))
                    )
            return arguments[0]
        if name in _RATIO_FUNCTIONS:
            if len(arguments) != 2 or arguments[0] != arguments[1]:
                raise Invalid("atan2 takes two arguments of one dimension")
            return DIMENSIONLESS
        raise Invalid(
            "Function "
            + name
            + " is not in the dimensional-check whitelist; add it deliberately or "
            "rewrite the expression"
        )


def check_expression_dimensions(expr, env) -> dict:
    """Infer the dimension of an arithmetic expression and refuse an inconsistent one.

    Rules enforced: `+`, `-` and comparisons require identical dimensions; `*` and `/`
    add and subtract exponents; an exponent is dimensionless and, over a dimensional
    base, a literal; `exp`, `log`, `tanh` and their family require a dimensionless
    argument. That last rule catches the most common modelling error in this repo's
    domains, where a raw quantity is fed to a transcendental function.

    `env` maps each symbol to a `Dimension`, a `Unit`, or a declared dimension string.
    Nothing is evaluated: the expression is parsed with `ast` and never executed.
    """
    if not isinstance(expr, str) or not expr.strip():
        raise Invalid("Expected a nonempty expression string")
    if len(expr) > MAX_EXPRESSION_CHARS:
        raise Invalid(f"Expression exceeds {MAX_EXPRESSION_CHARS} characters")
    if not isinstance(env, dict):
        raise Invalid("Expected an environment mapping symbols to dimensions")
    if len(env) > 512:
        raise Invalid("Environment exceeds the 512-symbol envelope")
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        raise Invalid("Expected a parseable arithmetic expression") from None
    checker = _ExpressionChecker(env)
    dimension = checker.visit(tree)
    return {
        "expression": expr,
        "dimension": dimension.canonical,
        "exponents": dimension.as_dict(),
        "vector": list(dimension.vector),
        "dimensionless": dimension.is_dimensionless,
        "symbols": sorted(checker.used),
        "symbol_dimensions": dict(sorted(checker.used.items())),
        "declared_but_unused": sorted(set(env) - set(checker.used)),
        "rules_enforced": checker.checks,
        "node_count": checker.nodes,
        "executed": False,
        "scope": SCOPE_EXPRESSION,
    }
