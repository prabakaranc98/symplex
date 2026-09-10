"""The permitted mutation grammar for scientific-program genomes, as typed operators.

Each operator is a host-owned, deterministic edit with declared preconditions, a
declared expected observable effect, and a declared result that would count
against it. An operator that cannot state what would disconfirm it is not a
scientific move, so this module refuses to apply one.

Refusal is the point. `recombine` merges components only when meaning, units,
timing and interfaces agree, and `validate_patch` enforces the specification's
denylist: an editor cannot touch final labels, the objective, the evaluator,
task allocation, execution permissions, budget enforcement or audit records.
Nothing here calls a model, the network, or a subprocess.
"""

import copy
import math
import random

from symplex.agents.qd_archive import ACCEPTING_OUTCOMES
from symplex.core.contracts import Invalid, number

VERSION = "scientific-program-mutation-grammar-v1"

GENOME_FIELDS = (
    "template",
    "template_version",
    "components",
    "couplings",
    "transforms",
    "observation_model",
    "parameter_domains",
    "inference",
    "supported_queries",
    "evidence_ids",
    "parent_ids",
    "constraints",
)
MUTABLE_FIELDS = tuple(f for f in GENOME_FIELDS if f != "template")

# The master specification: "The editor cannot change final labels, objective,
# evaluator, task allocation, execution permissions, budget enforcement or audit
# records." Aliases are listed because a denylist that only matches one spelling
# is decoration.
PROTECTED_FIELDS = (
    "audit",
    "audit_log",
    "audit_records",
    "budget",
    "budget_enforcement",
    "budgets",
    "evaluator",
    "evaluator_reference",
    "evaluator_version",
    "execution_permissions",
    "final_labels",
    "labels",
    "objective",
    "permissions",
    "promotion_rule",
    "split",
    "splits",
    "task_allocation",
)

COMPONENT_FIELDS = (
    "id",
    "kind",
    "meaning",
    "unit",
    "dimension",
    "time_scale",
    "step_seconds",
    "interface",
)
COUPLING_FIELDS = ("source", "target", "lag", "mechanism")

OBSERVATION_MODELS = {
    "gaussian_additive": "real",
    "student_t_additive": "real",
    "lognormal_multiplicative": "positive_real",
    "gamma_multiplicative": "positive_real",
    "poisson_count": "count",
    "negative_binomial_count": "count",
    "bernoulli_indicator": "binary",
}
INFERENCE_STRATEGIES = {
    "maximum_likelihood": ("deterministic_optimizer",),
    "map_penalized": ("deterministic_optimizer",),
    "variational": ("deterministic_optimizer", "stochastic_gradient"),
    "markov_chain_monte_carlo": ("sampler",),
    "sequential_monte_carlo": ("sampler",),
}
RESOLUTION_LADDER = ("coarse", "medium", "fine")

MAX_COMPONENTS = 32
MAX_COUPLINGS = 128
MAX_TRANSFORMS = 32
MAX_LAG = 8
MAX_QUERIES = 32
MAX_CONSTRAINTS = 64
MAX_OPERATOR_HISTORY = 10_000
MAX_TEXT = 1600

SCOPE = (
    "A closed edit grammar over a declared genome. Applying an operator produces "
    "a syntactically valid child and a falsifiable expectation; it establishes "
    "nothing about whether the child is a better model. Only the host evaluator "
    "and the archive decide what is retained, and neither this module nor the "
    "proposer may change the evaluator, the objective, the splits, permissions, "
    "budgets or the audit record."
)


def _text(value, field):
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_TEXT:
        raise Invalid(f"invalid_{field}: expected nonempty text under {MAX_TEXT} chars")
    return value


def _ident(value, field):
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise Invalid(f"invalid_{field}: expected a nonempty identifier")
    return value


def _interface(value, component_id):
    if not isinstance(value, dict) or set(value) != {"inputs", "outputs"}:
        raise Invalid(f"invalid_interface: {component_id} needs inputs and outputs")
    for key in ("inputs", "outputs"):
        if not isinstance(value[key], list) or any(
            not isinstance(v, str) or not v.strip() for v in value[key]
        ):
            raise Invalid(f"invalid_interface_{key}: {component_id}")
    return {"inputs": list(value["inputs"]), "outputs": list(value["outputs"])}


def _component(value):
    if not isinstance(value, dict):
        raise Invalid("invalid_component: expected a mapping")
    missing = [f for f in COMPONENT_FIELDS if f not in value]
    if missing:
        raise Invalid("component_missing_fields: " + ", ".join(missing))
    ident = _ident(value["id"], "component_id")
    for field in ("kind", "meaning", "unit", "dimension", "time_scale"):
        _text(value[field], "component_" + field)
    step = value["step_seconds"]
    if step is not None:
        number(step, 0)
        if step <= 0:
            raise Invalid(f"component_step_seconds_not_positive: {ident}")
    _interface(value["interface"], ident)
    return value


def _coupling(value, component_ids):
    if not isinstance(value, dict):
        raise Invalid("invalid_coupling: expected a mapping")
    missing = [f for f in COUPLING_FIELDS if f not in value]
    if missing:
        raise Invalid("coupling_missing_fields: " + ", ".join(missing))
    for end in ("source", "target"):
        if value[end] not in component_ids:
            raise Invalid(f"coupling_{end}_is_not_a_component: {value[end]}")
    lag = value["lag"]
    if not isinstance(lag, int) or isinstance(lag, bool) or not 0 <= lag <= MAX_LAG:
        raise Invalid(f"coupling_lag_out_of_bounds: expected 0..{MAX_LAG}")
    _text(value["mechanism"], "coupling_mechanism")
    return value


def _genome(value):
    if not isinstance(value, dict):
        raise Invalid("invalid_genome: expected a mapping")
    missing = [f for f in GENOME_FIELDS if f not in value]
    if missing:
        raise Invalid("genome_missing_fields: " + ", ".join(missing))
    present = sorted(set(value) & set(PROTECTED_FIELDS))
    if present:
        raise Invalid(
            "protected_field_in_genome: " + ", ".join(present) + " is host-owned and "
            "must never travel inside an editable genome"
        )
    components = value["components"]
    if not isinstance(components, list) or not 1 <= len(components) <= MAX_COMPONENTS:
        raise Invalid(f"component_count_out_of_bounds: expected 1..{MAX_COMPONENTS}")
    for component in components:
        _component(component)
    ids = [c["id"] for c in components]
    if len(set(ids)) != len(ids):
        raise Invalid("duplicate_component_ids")
    couplings = value["couplings"]
    if not isinstance(couplings, list) or len(couplings) > MAX_COUPLINGS:
        raise Invalid(f"coupling_count_out_of_bounds: at most {MAX_COUPLINGS}")
    for coupling in couplings:
        _coupling(coupling, set(ids))
    if len({(c["source"], c["target"]) for c in couplings}) != len(couplings):
        raise Invalid("duplicate_coupling_endpoints")
    observation = value["observation_model"]
    if not isinstance(observation, dict) or observation.get("kind") not in OBSERVATION_MODELS:
        raise Invalid(
            "unknown_observation_model: expected " + ", ".join(sorted(OBSERVATION_MODELS))
        )
    inference = value["inference"]
    if not isinstance(inference, dict):
        raise Invalid("invalid_inference_plan: expected a mapping")
    if inference.get("strategy") not in INFERENCE_STRATEGIES:
        raise Invalid(
            "unknown_inference_strategy: expected " + ", ".join(sorted(INFERENCE_STRATEGIES))
        )
    if inference.get("resolution") not in RESOLUTION_LADDER:
        raise Invalid("unknown_resolution: expected " + ", ".join(RESOLUTION_LADDER))
    _text(inference.get("backend"), "inference_backend")
    for field, limit in (
        ("transforms", MAX_TRANSFORMS),
        ("supported_queries", MAX_QUERIES),
        ("constraints", MAX_CONSTRAINTS),
        ("evidence_ids", MAX_CONSTRAINTS),
        ("parent_ids", 8),
    ):
        item = value[field]
        if not isinstance(item, list) or len(item) > limit:
            raise Invalid(f"{field}_out_of_bounds: at most {limit} entries")
    if not isinstance(value["parameter_domains"], dict):
        raise Invalid("invalid_parameter_domains: expected a mapping")
    return value


def seed_genome(**overrides):
    """A minimal valid genome, useful as a search seed and in tests."""
    genome = {
        "template": "coupled_component_graph",
        "template_version": "v1",
        "components": [
            {
                "id": "demand",
                "kind": "state",
                "meaning": "observed hourly demand",
                "unit": "unit_per_hour",
                "dimension": "rate",
                "time_scale": "hourly",
                "step_seconds": 3600.0,
                "interface": {"inputs": [], "outputs": ["level"]},
            },
            {
                "id": "capacity",
                "kind": "state",
                "meaning": "available service capacity",
                "unit": "unit_per_hour",
                "dimension": "rate",
                "time_scale": "hourly",
                "step_seconds": 3600.0,
                "interface": {"inputs": [], "outputs": ["level"]},
            },
        ],
        "couplings": [
            {
                "source": "capacity",
                "target": "demand",
                "lag": 0,
                "mechanism": "declared congestion feedback",
            }
        ],
        "transforms": [],
        "observation_model": {
            "kind": "gaussian_additive",
            "support": "real",
            "bias_model": None,
            "missingness_model": None,
        },
        "parameter_domains": {"coupling_weight": [-1.0, 1.0]},
        "inference": {
            "strategy": "maximum_likelihood",
            "backend": "deterministic_optimizer",
            "resolution": "medium",
        },
        "supported_queries": ["forecast"],
        "evidence_ids": ["evidence_1"],
        "parent_ids": [],
        "constraints": [],
    }
    genome.update(overrides)
    return _genome(genome)


seed_genome.scope = (
    "A syntactically valid placeholder genome with declared units, clocks and "
    "interfaces. Its components, couplings and evidence references are fictitious; "
    "it seeds a search and demonstrates the contract, and models nothing real."
)


# ------------------------------------------------------------------ operator cards

OPERATOR_CARDS = {
    "add_coupling": {
        "arity": 1,
        "preconditions": [
            "both endpoints are declared components of the parent",
            "the endpoint pair is not already coupled",
            "no constraint forbids this coupling",
            "the requested lag is within the declared lag bound",
        ],
        "expected_observable_effect": (
            "Development loss falls, or the target's residual autocorrelation at the "
            "declared lag falls, relative to the parent under the same protocol."
        ),
        "what_would_count_against_it": (
            "Development loss does not fall while the fitted coupling weight stays "
            "indistinguishable from zero, or the target's residuals are unchanged: the "
            "added edge then carries no information and should be removed."
        ),
    },
    "remove_coupling": {
        "arity": 1,
        "preconditions": [
            "the coupling exists in the parent",
            "no constraint declares this coupling required",
            "at least one coupling remains after removal",
        ],
        "expected_observable_effect": (
            "Development loss is unchanged within tolerance while parameter count and "
            "compute cost fall: the removed edge was not carrying information."
        ),
        "what_would_count_against_it": (
            "Development loss rises beyond tolerance, or the target's residuals gain "
            "structure at the removed lag: the edge was load-bearing."
        ),
    },
    "add_lag": {
        "arity": 1,
        "preconditions": [
            "the coupling exists in the parent",
            "the requested lag differs from the current lag and is within bounds",
            "the source component declares a sampling interval, so a lag is defined",
        ],
        "expected_observable_effect": (
            "Cross-correlation between source and target residuals peaks at the new lag "
            "rather than the old one, and development loss falls."
        ),
        "what_would_count_against_it": (
            "The residual cross-correlation peak does not move, or moves to a lag the "
            "declared sampling interval cannot represent: the timing claim is unsupported."
        ),
    },
    "introduce_shared_driver": {
        "arity": 1,
        "preconditions": [
            "at least two declared target components",
            "the driver identifier is new",
            "the driver declares a sampling interval no slower than every target",
            "the driver's dimension is declared",
        ],
        "expected_observable_effect": (
            "Residual correlation between the targets falls once the shared driver is "
            "included, without a rise in development loss."
        ),
        "what_would_count_against_it": (
            "Residual correlation between the targets is unchanged, or the driver's "
            "fitted effects on the targets have opposite signs with no declared "
            "mechanism: the common cause is not doing the work claimed for it."
        ),
    },
    "add_bias_or_missingness_model": {
        "arity": 1,
        "preconditions": [
            "kind is bias or missingness",
            "that slot is currently empty, so nothing is silently overwritten",
            "the model cites evidence already declared by the parent",
            "the model states a mechanism",
        ],
        "expected_observable_effect": (
            "Calibration error falls on the affected stratum, and the measured bias in "
            "that stratum's residuals shrinks toward zero."
        ),
        "what_would_count_against_it": (
            "Calibration error is unchanged or worsens elsewhere: the correction is "
            "absorbing variance rather than modelling a real selection process."
        ),
    },
    "change_observation_model": {
        "arity": 1,
        "preconditions": [
            "the requested observation model is in the declared catalogue",
            "it differs from the current one",
            "it shares the current model's support, so the measured quantity's "
            "admissible range is unchanged",
        ],
        "expected_observable_effect": (
            "Predictive log score improves and residual quantiles move closer to the "
            "assumed distribution, with the point predictions largely unchanged."
        ),
        "what_would_count_against_it": (
            "Log score does not improve, or residual quantiles move further from the "
            "assumed distribution: the noise assumption was not the binding problem."
        ),
    },
    "replace_component": {
        "arity": 1,
        "preconditions": [
            "the replaced component exists",
            "the replacement is a valid component",
            "meaning dimension, unit, time scale and interface all match, so every "
            "existing coupling remains meaningful",
        ],
        "expected_observable_effect": (
            "Development loss falls or compute cost falls while the couplings that "
            "reference this component keep their fitted signs."
        ),
        "what_would_count_against_it": (
            "Couplings that reference the component change sign, or downstream "
            "predictions shift while loss is flat: the replacement changed the meaning "
            "of the node rather than its implementation."
        ),
    },
    "adjust_resolution": {
        "arity": 1,
        "preconditions": [
            "the requested resolution is on the declared ladder",
            "it is exactly one rung from the current resolution",
        ],
        "expected_observable_effect": (
            "A finer rung lowers discretisation error at higher compute cost; a coarser "
            "rung lowers compute cost with development loss unchanged within tolerance."
        ),
        "what_would_count_against_it": (
            "Refining does not lower loss, meaning the error was not discretisation; or "
            "coarsening changes the conclusion, meaning the result was resolution "
            "dependent and the earlier finding was not converged."
        ),
    },
    "repair_alignment_error": {
        "arity": 1,
        "preconditions": [
            "a demonstrated alignment error is supplied, citing evidence declared by "
            "the parent and a nonzero count of affected records",
            "the named component exists",
            "the repair states the correction it applies",
        ],
        "expected_observable_effect": (
            "The demonstrated mismatch count falls to zero on re-inspection, and "
            "development loss falls or is unchanged."
        ),
        "what_would_count_against_it": (
            "The mismatch count is unchanged after the repair, or loss falls without the "
            "mismatch count changing: the improvement came from somewhere else."
        ),
    },
    "change_inference_strategy": {
        "arity": 1,
        "preconditions": [
            "the requested strategy is in the declared catalogue",
            "it differs from the current strategy",
            "the declared backend supports it",
        ],
        "expected_observable_effect": (
            "The same posterior or optimum is reached within tolerance at different "
            "compute cost, so cost moves and development loss does not."
        ),
        "what_would_count_against_it": (
            "Development loss moves materially: the two strategies are not solving the "
            "same problem, and one of them is misconfigured or not converged."
        ),
    },
    "recombine": {
        "arity": 2,
        "preconditions": [
            "every requested component exists in the donor parent",
            "each donor component matches its receiving counterpart, or is new",
            "matched components agree on dimension, unit, time scale, sampling "
            "interval and interface",
            "the two parents share an observation-model support",
        ],
        "expected_observable_effect": (
            "The child inherits both parents' behaviour on the strata where each parent "
            "was strongest, and its development loss is no worse than the better parent."
        ),
        "what_would_count_against_it": (
            "The child is worse than both parents on every stratum: the merged "
            "components were interacting in a way neither parent's evidence covers."
        ),
    },
}


def operator_card(name):
    """The declared contract for one operator: preconditions and falsifiability."""
    if name not in OPERATOR_CARDS:
        raise Invalid("unknown_operator: " + str(name))
    card = OPERATOR_CARDS[name]
    return {
        "operator": name,
        "arity": card["arity"],
        "preconditions": list(card["preconditions"]),
        "expected_observable_effect": card["expected_observable_effect"],
        "what_would_count_against_it": card["what_would_count_against_it"],
        "version": VERSION,
        "scope": SCOPE,
    }


def _result(name, child, parents, patch, detail):
    card = operator_card(name)
    child["parent_ids"] = [p for p in parents]
    _genome(child)
    return {
        "operator": name,
        "genome": child,
        "parent_ids": list(parents),
        "patch": patch,
        "changed_fields": sorted(patch),
        "detail": detail,
        "preconditions": card["preconditions"],
        "expected_observable_effect": card["expected_observable_effect"],
        "what_would_count_against_it": card["what_would_count_against_it"],
        "version": VERSION,
        "scope": SCOPE,
    }


def _constraints(genome, kind):
    return [
        c
        for c in genome["constraints"]
        if isinstance(c, dict) and c.get("kind") == kind
    ]


def _find(genome, source, target):
    for coupling in genome["couplings"]:
        if coupling["source"] == source and coupling["target"] == target:
            return coupling
    return None


# ---------------------------------------------------------------------- operators


def add_coupling(genome, *, source, target, lag, mechanism, parent_id=None):
    parent = _genome(genome)
    ids = {c["id"] for c in parent["components"]}
    source, target = _ident(source, "source"), _ident(target, "target")
    if source not in ids or target not in ids:
        raise Invalid("coupling_endpoint_is_not_a_component: " + source + " -> " + target)
    if source == target:
        raise Invalid("self_coupling_requires_an_explicit_declared_mechanism")
    if _find(parent, source, target) is not None:
        raise Invalid("coupling_already_present: " + source + " -> " + target)
    if any(
        c.get("source") == source and c.get("target") == target
        for c in _constraints(parent, "forbidden_coupling")
    ):
        raise Invalid("coupling_forbidden_by_constraint: " + source + " -> " + target)
    if len(parent["couplings"]) >= MAX_COUPLINGS:
        raise Invalid(f"coupling_count_out_of_bounds: at most {MAX_COUPLINGS}")
    if not isinstance(lag, int) or isinstance(lag, bool) or not 0 <= lag <= MAX_LAG:
        raise Invalid(f"coupling_lag_out_of_bounds: expected 0..{MAX_LAG}")
    child = copy.deepcopy(parent)
    child["couplings"].append(
        {
            "source": source,
            "target": target,
            "lag": lag,
            "mechanism": _text(mechanism, "mechanism"),
        }
    )
    return _result(
        "add_coupling",
        child,
        [parent_id] if parent_id else [],
        {"couplings": child["couplings"]},
        f"added {source} -> {target} at lag {lag}",
    )


def remove_coupling(genome, *, source, target, parent_id=None):
    parent = _genome(genome)
    coupling = _find(parent, _ident(source, "source"), _ident(target, "target"))
    if coupling is None:
        raise Invalid("coupling_not_present: " + str(source) + " -> " + str(target))
    if any(
        c.get("source") == source and c.get("target") == target
        for c in _constraints(parent, "required_coupling")
    ):
        raise Invalid("coupling_required_by_constraint: " + source + " -> " + target)
    if len(parent["couplings"]) <= 1:
        raise Invalid("removal_would_leave_an_uncoupled_graph")
    child = copy.deepcopy(parent)
    child["couplings"] = [
        c
        for c in child["couplings"]
        if not (c["source"] == source and c["target"] == target)
    ]
    return _result(
        "remove_coupling",
        child,
        [parent_id] if parent_id else [],
        {"couplings": child["couplings"]},
        f"removed {source} -> {target}",
    )


def add_lag(genome, *, source, target, lag, parent_id=None):
    parent = _genome(genome)
    coupling = _find(parent, _ident(source, "source"), _ident(target, "target"))
    if coupling is None:
        raise Invalid("coupling_not_present: " + str(source) + " -> " + str(target))
    if not isinstance(lag, int) or isinstance(lag, bool) or not 0 <= lag <= MAX_LAG:
        raise Invalid(f"coupling_lag_out_of_bounds: expected 0..{MAX_LAG}")
    if lag == coupling["lag"]:
        raise Invalid("lag_unchanged: a mutation must change something")
    origin = next(c for c in parent["components"] if c["id"] == source)
    if origin["step_seconds"] is None:
        raise Invalid(
            "lag_undefined_without_sampling_interval: " + source + " is event-indexed, "
            "so 'lag " + str(lag) + "' names no measurable delay"
        )
    child = copy.deepcopy(parent)
    for edge in child["couplings"]:
        if edge["source"] == source and edge["target"] == target:
            edge["lag"] = lag
    return _result(
        "add_lag",
        child,
        [parent_id] if parent_id else [],
        {"couplings": child["couplings"]},
        f"lag {coupling['lag']} -> {lag} on {source} -> {target} "
        f"({lag * origin['step_seconds']} seconds)",
    )


def introduce_shared_driver(genome, *, driver, target_ids, mechanism, parent_id=None):
    parent = _genome(genome)
    driver = _component(dict(driver))
    ids = {c["id"] for c in parent["components"]}
    if driver["id"] in ids:
        raise Invalid("driver_identifier_already_used: " + driver["id"])
    targets = [_ident(t, "target_id") for t in target_ids or []]
    if len(set(targets)) < 2:
        raise Invalid("shared_driver_requires_at_least_two_distinct_targets")
    missing = sorted(set(targets) - ids)
    if missing:
        raise Invalid("driver_target_is_not_a_component: " + ", ".join(missing))
    if driver["step_seconds"] is None:
        raise Invalid("driver_requires_a_declared_sampling_interval")
    for target in targets:
        component = next(c for c in parent["components"] if c["id"] == target)
        if component["step_seconds"] is not None and driver["step_seconds"] > component["step_seconds"]:
            raise Invalid(
                "driver_sampled_slower_than_target: " + driver["id"] + " cannot explain "
                "co-movement of " + target + " at the target's own resolution"
            )
    if len(parent["components"]) >= MAX_COMPONENTS:
        raise Invalid(f"component_count_out_of_bounds: expected 1..{MAX_COMPONENTS}")
    child = copy.deepcopy(parent)
    child["components"].append(driver)
    for target in targets:
        child["couplings"].append(
            {
                "source": driver["id"],
                "target": target,
                "lag": 0,
                "mechanism": _text(mechanism, "mechanism"),
            }
        )
    return _result(
        "introduce_shared_driver",
        child,
        [parent_id] if parent_id else [],
        {"components": child["components"], "couplings": child["couplings"]},
        f"{driver['id']} drives {', '.join(targets)}",
    )


def add_bias_or_missingness_model(genome, *, kind, model, parent_id=None):
    parent = _genome(genome)
    if kind not in ("bias", "missingness"):
        raise Invalid("unknown_correction_kind: expected bias or missingness")
    slot = "bias_model" if kind == "bias" else "missingness_model"
    if parent["observation_model"].get(slot) is not None:
        raise Invalid(
            "correction_slot_already_occupied: " + slot + " is set; replacing it "
            "silently would hide which correction produced a change"
        )
    if not isinstance(model, dict):
        raise Invalid("invalid_correction_model: expected a mapping")
    _text(model.get("mechanism"), "correction_mechanism")
    cited = model.get("evidence_ids") or []
    if not isinstance(cited, list) or not cited:
        raise Invalid("correction_model_cites_no_evidence")
    unknown = sorted(set(cited) - set(parent["evidence_ids"]))
    if unknown:
        raise Invalid("correction_cites_undeclared_evidence: " + ", ".join(unknown))
    child = copy.deepcopy(parent)
    child["observation_model"][slot] = copy.deepcopy(model)
    return _result(
        "add_bias_or_missingness_model",
        child,
        [parent_id] if parent_id else [],
        {"observation_model": child["observation_model"]},
        f"declared {slot} citing {', '.join(sorted(cited))}",
    )


def change_observation_model(genome, *, kind, parent_id=None):
    parent = _genome(genome)
    if kind not in OBSERVATION_MODELS:
        raise Invalid(
            "unknown_observation_model: expected " + ", ".join(sorted(OBSERVATION_MODELS))
        )
    current = parent["observation_model"]["kind"]
    if kind == current:
        raise Invalid("observation_model_unchanged: a mutation must change something")
    if OBSERVATION_MODELS[kind] != OBSERVATION_MODELS[current]:
        raise Invalid(
            "incompatible_observation_support: " + current + " observes "
            + OBSERVATION_MODELS[current] + " values and " + kind + " observes "
            + OBSERVATION_MODELS[kind] + "; swapping them changes what the "
            "measurement is, not how it is modelled"
        )
    child = copy.deepcopy(parent)
    child["observation_model"]["kind"] = kind
    child["observation_model"]["support"] = OBSERVATION_MODELS[kind]
    return _result(
        "change_observation_model",
        child,
        [parent_id] if parent_id else [],
        {"observation_model": child["observation_model"]},
        f"{current} -> {kind} on shared support {OBSERVATION_MODELS[kind]}",
    )


def component_compatibility(left, right):
    """Why two components may or may not stand in for one another."""
    left, right = _component(dict(left)), _component(dict(right))
    reasons = []
    if left["dimension"] != right["dimension"]:
        reasons.append(
            f"dimension {left['dimension']} != {right['dimension']}"
        )
    if left["unit"] != right["unit"]:
        reasons.append(f"unit {left['unit']} != {right['unit']}")
    if left["time_scale"] != right["time_scale"]:
        reasons.append(f"time_scale {left['time_scale']} != {right['time_scale']}")
    if left["step_seconds"] != right["step_seconds"]:
        reasons.append(
            f"step_seconds {left['step_seconds']} != {right['step_seconds']}"
        )
    for key in ("inputs", "outputs"):
        if sorted(left["interface"][key]) != sorted(right["interface"][key]):
            reasons.append(f"interface {key} differ")
    return {
        "compatible": not reasons,
        "incompatibilities": reasons,
        "compared": ["dimension", "unit", "time_scale", "step_seconds", "interface"],
        "rule": (
            "Two components may substitute for one another only when they measure the "
            "same declared quantity, in the same unit, on the same clock at the same "
            "sampling interval, behind the same interface. Declared text equality is "
            "the check; no unit conversion is attempted, and none is implied."
        ),
        "scope": (
            "Compatibility of declarations, not of underlying reality. Two components "
            "can pass this check and still mean different things if their declarations "
            "are wrong; the check only prevents merges that are wrong on their face."
        ),
    }


def replace_component(genome, *, component_id, replacement, parent_id=None):
    parent = _genome(genome)
    component_id = _ident(component_id, "component_id")
    existing = next((c for c in parent["components"] if c["id"] == component_id), None)
    if existing is None:
        raise Invalid("component_not_present: " + component_id)
    replacement = _component(copy.deepcopy(replacement))
    if replacement["id"] != component_id and replacement["id"] in {
        c["id"] for c in parent["components"]
    }:
        raise Invalid("replacement_identifier_already_used: " + replacement["id"])
    report = component_compatibility(existing, replacement)
    if not report["compatible"]:
        raise Invalid(
            "incompatible_replacement: " + "; ".join(report["incompatibilities"])
        )
    child = copy.deepcopy(parent)
    child["components"] = [
        replacement if c["id"] == component_id else c for c in child["components"]
    ]
    for edge in child["couplings"]:
        for end in ("source", "target"):
            if edge[end] == component_id:
                edge[end] = replacement["id"]
    return _result(
        "replace_component",
        child,
        [parent_id] if parent_id else [],
        {"components": child["components"], "couplings": child["couplings"]},
        f"{component_id} -> {replacement['id']} with matching unit, timing and interface",
    )


def adjust_resolution(genome, *, resolution, parent_id=None):
    parent = _genome(genome)
    if resolution not in RESOLUTION_LADDER:
        raise Invalid("unknown_resolution: expected " + ", ".join(RESOLUTION_LADDER))
    current = parent["inference"]["resolution"]
    if resolution == current:
        raise Invalid("resolution_unchanged: a mutation must change something")
    step = abs(RESOLUTION_LADDER.index(resolution) - RESOLUTION_LADDER.index(current))
    if step != 1:
        raise Invalid(
            "resolution_step_too_large: move one rung at a time so the effect of "
            "resolution can be attributed"
        )
    child = copy.deepcopy(parent)
    child["inference"]["resolution"] = resolution
    return _result(
        "adjust_resolution",
        child,
        [parent_id] if parent_id else [],
        {"inference": child["inference"]},
        f"resolution {current} -> {resolution}",
    )


def repair_alignment_error(genome, *, component_id, demonstration, correction, parent_id=None):
    parent = _genome(genome)
    component_id = _ident(component_id, "component_id")
    if component_id not in {c["id"] for c in parent["components"]}:
        raise Invalid("component_not_present: " + component_id)
    if not isinstance(demonstration, dict):
        raise Invalid("alignment_error_not_demonstrated: expected a demonstration mapping")
    evidence_id = demonstration.get("evidence_id")
    if evidence_id not in set(parent["evidence_ids"]):
        raise Invalid(
            "alignment_demonstration_cites_undeclared_evidence: " + str(evidence_id)
        )
    _text(demonstration.get("observed_discrepancy"), "observed_discrepancy")
    affected = demonstration.get("affected_records")
    if not isinstance(affected, int) or isinstance(affected, bool) or affected <= 0:
        raise Invalid(
            "alignment_error_not_demonstrated: affected_records must be a positive count, "
            "otherwise the repair is a guess with no observed mismatch behind it"
        )
    if len(parent["transforms"]) >= MAX_TRANSFORMS:
        raise Invalid(f"transforms_out_of_bounds: at most {MAX_TRANSFORMS} entries")
    child = copy.deepcopy(parent)
    child["transforms"].append(
        {
            "kind": "alignment_repair",
            "component_id": component_id,
            "correction": _text(correction, "correction"),
            "evidence_id": evidence_id,
            "observed_discrepancy": demonstration["observed_discrepancy"],
            "affected_records": affected,
        }
    )
    return _result(
        "repair_alignment_error",
        child,
        [parent_id] if parent_id else [],
        {"transforms": child["transforms"]},
        f"repaired {component_id} against {affected} demonstrated mismatches",
    )


def change_inference_strategy(genome, *, strategy, parent_id=None):
    parent = _genome(genome)
    if strategy not in INFERENCE_STRATEGIES:
        raise Invalid(
            "unknown_inference_strategy: expected " + ", ".join(sorted(INFERENCE_STRATEGIES))
        )
    current = parent["inference"]["strategy"]
    if strategy == current:
        raise Invalid("inference_strategy_unchanged: a mutation must change something")
    backend = parent["inference"]["backend"]
    if backend not in INFERENCE_STRATEGIES[strategy]:
        raise Invalid(
            "backend_does_not_support_strategy: " + backend + " cannot run " + strategy
            + "; supported backends are " + ", ".join(INFERENCE_STRATEGIES[strategy])
        )
    child = copy.deepcopy(parent)
    child["inference"]["strategy"] = strategy
    return _result(
        "change_inference_strategy",
        child,
        [parent_id] if parent_id else [],
        {"inference": child["inference"]},
        f"inference {current} -> {strategy} on backend {backend}",
    )


def recombine(parent_a, parent_b, *, component_ids, parent_a_id=None, parent_b_id=None):
    """Merge donor components into a receiver only when their declarations agree.

    The refusal is the scientific content: two components that disagree on
    meaning, unit, clock or interface cannot be merged into one program without
    someone deciding what the merged quantity means, and that decision is not a
    search operator's to make.
    """
    receiver = _genome(parent_a)
    donor = _genome(parent_b)
    requested = [_ident(c, "component_id") for c in component_ids or []]
    if not requested or len(set(requested)) != len(requested):
        raise Invalid("recombination_requires_distinct_named_components")
    donor_by_id = {c["id"]: c for c in donor["components"]}
    missing = sorted(set(requested) - set(donor_by_id))
    if missing:
        raise Invalid("donor_component_not_present: " + ", ".join(missing))
    receiver_support = OBSERVATION_MODELS[receiver["observation_model"]["kind"]]
    donor_support = OBSERVATION_MODELS[donor["observation_model"]["kind"]]
    if receiver_support != donor_support:
        raise Invalid(
            "incompatible_observation_support: parents observe " + receiver_support
            + " and " + donor_support + " values; their components are not measuring "
            "the same kind of quantity"
        )
    receiver_by_id = {c["id"]: c for c in receiver["components"]}
    reasons, merged, added = [], [], []
    for ident in requested:
        if ident in receiver_by_id:
            report = component_compatibility(receiver_by_id[ident], donor_by_id[ident])
            if not report["compatible"]:
                reasons.append(ident + ": " + "; ".join(report["incompatibilities"]))
            else:
                merged.append(ident)
        else:
            added.append(ident)
    if reasons:
        raise Invalid("incompatible_recombination: " + " | ".join(reasons))
    if len(receiver["components"]) + len(added) > MAX_COMPONENTS:
        raise Invalid(f"component_count_out_of_bounds: expected 1..{MAX_COMPONENTS}")
    child = copy.deepcopy(receiver)
    for ident in requested:
        component = copy.deepcopy(donor_by_id[ident])
        child["components"] = [c for c in child["components"] if c["id"] != ident]
        child["components"].append(component)
    present = {c["id"] for c in child["components"]}
    existing = {(c["source"], c["target"]) for c in child["couplings"]}
    for edge in donor["couplings"]:
        pair = (edge["source"], edge["target"])
        if (
            pair not in existing
            and edge["source"] in present
            and edge["target"] in present
            and (edge["source"] in requested or edge["target"] in requested)
        ):
            child["couplings"].append(copy.deepcopy(edge))
            existing.add(pair)
    child["evidence_ids"] = sorted(set(receiver["evidence_ids"]) | set(donor["evidence_ids"]))
    parents = [p for p in (parent_a_id, parent_b_id) if p]
    return _result(
        "recombine",
        child,
        parents,
        {"components": child["components"], "couplings": child["couplings"],
         "evidence_ids": child["evidence_ids"]},
        f"merged {', '.join(merged) or 'none'}, added {', '.join(added) or 'none'}",
    )


OPERATORS = {
    "add_coupling": add_coupling,
    "remove_coupling": remove_coupling,
    "add_lag": add_lag,
    "introduce_shared_driver": introduce_shared_driver,
    "add_bias_or_missingness_model": add_bias_or_missingness_model,
    "change_observation_model": change_observation_model,
    "replace_component": replace_component,
    "adjust_resolution": adjust_resolution,
    "repair_alignment_error": repair_alignment_error,
    "change_inference_strategy": change_inference_strategy,
    "recombine": recombine,
}


def apply_operator(name, *genomes, **arguments):
    """Dispatch one named operator; unknown names are refused, never improvised."""
    if name not in OPERATORS:
        raise Invalid("unknown_operator: " + str(name))
    arity = OPERATOR_CARDS[name]["arity"]
    if len(genomes) != arity:
        raise Invalid(f"operator_arity_mismatch: {name} takes {arity} parent genome(s)")
    return OPERATORS[name](*genomes, **arguments)


# ------------------------------------------------------------------ patch validation


def _protected_keys(value, depth=0):
    found = set()
    if depth > 8:
        raise Invalid("patch_nesting_too_deep")
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and key in PROTECTED_FIELDS:
                found.add(key)
            found |= _protected_keys(item, depth + 1)
    elif isinstance(value, list):
        for item in value[:512]:
            found |= _protected_keys(item, depth + 1)
    return found


def validate_patch(parent, patch):
    """Refuse any patch that reaches outside the permitted mutation grammar.

    The specification is emphatic: the editor cannot change final labels, the
    objective, the evaluator, task allocation, execution permissions, budget
    enforcement or audit records. Those are checked first, by name and at any
    nesting depth, so the error says which host-owned field was touched instead
    of a generic schema complaint.
    """
    if not isinstance(parent, dict):
        raise Invalid("invalid_parent_genome: expected a mapping")
    if not isinstance(patch, dict):
        raise Invalid("invalid_patch: expected a mapping of genome fields")
    if not patch:
        raise Invalid("empty_patch: a mutation must change something")
    protected = sorted(_protected_keys(patch))
    if protected:
        raise Invalid(
            "protected_field_not_editable: " + ", ".join(protected) + ". Final labels, "
            "the objective, the evaluator, task allocation, execution permissions, "
            "budget enforcement and audit records are host-owned."
        )
    outside = sorted(set(patch) - set(MUTABLE_FIELDS))
    if outside:
        raise Invalid("field_outside_mutation_grammar: " + ", ".join(outside))
    base = _genome(parent)
    child = copy.deepcopy(base)
    child.update(copy.deepcopy(patch))
    _genome(child)
    changed = sorted(k for k in patch if child[k] != base[k])
    if not changed:
        raise Invalid("patch_changes_nothing: a mutation must change something")
    return {
        "accepted": True,
        "changed_fields": changed,
        "child": child,
        "mutable_fields": list(MUTABLE_FIELDS),
        "protected_fields": list(PROTECTED_FIELDS),
        "checked": [
            "no host-owned field appears at any nesting depth of the patch",
            "every patched field is inside the declared mutation grammar",
            "the resulting genome still satisfies the genome contract",
            "the patch changes at least one field",
        ],
        "version": VERSION,
        "scope": (
            "A grammar and denylist check. Accepting a patch means it is inside the "
            "editable surface, not that the edit is scientifically sensible, supported "
            "by evidence, or an improvement. " + SCOPE
        ),
    }


# -------------------------------------------------------- adaptive operator selection


UNIFORM_PRIOR_NOTE = (
    "No operator outcome has been recorded yet, so there is nothing to learn from. "
    "The documented fallback is a uniform prior over the declared operators; the "
    "selection below is a seeded draw from that prior and carries no evidence that "
    "the chosen operator is any good."
)


def adaptive_operator_selection(
    history,
    seed,
    *,
    strategy="ucb1",
    exploration=math.sqrt(2.0),
    decay=0.9,
    operators=None,
):
    """A bandit over the mutation grammar, scored by accepted archive entries.

    `ucb1` is Auer et al.'s index: mean reward plus c * sqrt(ln N / n). `credit`
    is a recency-weighted credit-assignment variant: an outcome i steps back
    counts decay**i, so an operator that stopped working stops being credited for
    what it did fifty generations ago. All arithmetic is returned.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise Invalid("selection_seed_required: pass an explicit integer seed")
    if strategy not in ("ucb1", "credit"):
        raise Invalid("unknown_bandit_strategy: expected ucb1 or credit")
    exploration = number(exploration, 0)
    decay = number(decay, 0, 1)
    if not 0 < decay <= 1:
        raise Invalid("decay_out_of_bounds: expected a value in (0, 1]")
    arms = sorted(operators if operators is not None else OPERATORS)
    if not arms or any(a not in OPERATORS for a in arms):
        raise Invalid("unknown_operator_in_arm_set")
    if history is None:
        history = []
    if not isinstance(history, list):
        raise Invalid("invalid_history: expected a list of outcome records")
    if len(history) > MAX_OPERATOR_HISTORY:
        raise Invalid(f"history_exceeds_bound: at most {MAX_OPERATOR_HISTORY} records")
    relevant = []
    for record in history:
        if not isinstance(record, dict):
            raise Invalid("invalid_history_record: expected a mapping")
        name = record.get("operator")
        outcome = record.get("outcome")
        if name not in OPERATORS:
            raise Invalid("unknown_operator_in_history: " + str(name))
        if name in arms:
            relevant.append((name, outcome in ACCEPTING_OUTCOMES))
    rng = random.Random(seed)
    if not relevant:
        share = 1.0 / len(arms)
        return {
            "selected": arms[rng.randrange(len(arms))],
            "strategy": strategy,
            "basis": "uniform_prior_no_history",
            "prior": {name: share for name in arms},
            "arms": [
                {
                    "operator": name,
                    "plays": 0.0,
                    "accepted": 0.0,
                    "mean_reward": None,
                    "exploration_bonus": None,
                    "score": None,
                    "status": "untried",
                }
                for name in arms
            ],
            "history_records": 0,
            "seed": seed,
            "note": UNIFORM_PRIOR_NOTE,
            "arithmetic": "score(operator) = 1/" + str(len(arms)) + " for every operator.",
            "reward_definition": (
                "Reward 1 when the operator's child was accepted into the archive "
                "(" + ", ".join(ACCEPTING_OUTCOMES) + "), else 0."
            ),
            "version": VERSION,
            "scope": (
                "An exploration schedule over edit types. A high-scoring operator is "
                "one that has recently produced archive entries, which is a statement "
                "about this search, not about which mechanism is true of the system under study. "
                + SCOPE
            ),
        }
    plays = {name: 0.0 for name in arms}
    accepted = {name: 0.0 for name in arms}
    if strategy == "ucb1":
        for name, ok in relevant:
            plays[name] += 1.0
            accepted[name] += 1.0 if ok else 0.0
        arithmetic = (
            "score = accepted/plays + " + repr(exploration) + " * sqrt(ln(N)/plays); "
            "untried operators are played first."
        )
    else:
        for age, (name, ok) in enumerate(reversed(relevant)):
            weight = decay**age
            plays[name] += weight
            accepted[name] += weight if ok else 0.0
        arithmetic = (
            "credit weights an outcome i steps back by " + repr(decay) + "**i; "
            "score = credit/weighted_plays + " + repr(exploration)
            + " * sqrt(ln(weighted N)/weighted plays); untried operators are played first."
        )
    total = sum(plays.values())
    logarithm = math.log(total) if total > 1 else 0.0
    rows = []
    for name in arms:
        if plays[name] <= 0:
            rows.append(
                {
                    "operator": name,
                    "plays": 0.0,
                    "accepted": 0.0,
                    "mean_reward": None,
                    "exploration_bonus": None,
                    "score": math.inf,
                    "status": "untried",
                }
            )
            continue
        mean = accepted[name] / plays[name]
        bonus = exploration * math.sqrt(logarithm / plays[name]) if logarithm > 0 else 0.0
        rows.append(
            {
                "operator": name,
                "plays": plays[name],
                "accepted": accepted[name],
                "mean_reward": mean,
                "exploration_bonus": bonus,
                "score": mean + bonus,
                "status": "played",
            }
        )
    best = max(row["score"] for row in rows)
    tied = [row["operator"] for row in rows if row["score"] == best]
    selected = tied[rng.randrange(len(tied))]
    return {
        "selected": selected,
        "strategy": strategy,
        "basis": "untried_operators_first" if math.isinf(best) else strategy,
        "prior": None,
        "arms": [
            dict(row, score=None if math.isinf(row["score"]) else row["score"])
            for row in rows
        ],
        "untried": [row["operator"] for row in rows if row["status"] == "untried"],
        "tied_at_best": sorted(tied),
        "history_records": len(relevant),
        "total_plays": total,
        "seed": seed,
        "note": None,
        "arithmetic": arithmetic,
        "reward_definition": (
            "Reward 1 when the operator's child was accepted into the archive ("
            + ", ".join(ACCEPTING_OUTCOMES) + "), else 0."
        ),
        "version": VERSION,
        "scope": (
            "An exploration schedule over edit types. A high-scoring operator is one "
            "that has recently produced archive entries, which is a statement about "
            "this search under these descriptors and objectives, never about which "
            "mechanism is true of the system under study. " + SCOPE
        ),
    }
