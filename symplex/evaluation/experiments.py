"""Frozen comparative protocols and deterministic checks of executed maker outputs.

The host computes dominance; it does not treat model-written numbers or diagnostic
checks as independently verified science. This is a scoped exploration archive.
"""

import json
from typing import ClassVar, Literal

from pydantic import Field, model_validator

from symplex.agents.prompts import prompt
from symplex.agents.scientific_cycle import current_artifact
from symplex.core.contracts import Invalid, digest
from symplex.evaluation.numerics import NumericCheck, execute_numeric_checks
from symplex.modeling.complex_system import ClosedContract, Identifier, Text

EVALUATOR_VERSION = "frozen-csv-numerics-readiness-v2"


class Metric(ClosedContract):
    id: Identifier
    name: Text
    unit: Text
    direction: Literal["minimize", "maximize"]
    minimum_improvement: float = Field(ge=0)
    maximum_degradation: float = Field(ge=0)


class ExperimentProtocol(ClosedContract):
    output_token_budget: ClassVar[int] = 6500
    # Unknown is a loading state for immutable records saved before this field.
    # The proposal schema and parse entry point require an explicit new decision.
    execution_readiness: Literal["ready", "blocked", "unknown"] = Field(
        default="unknown",
        description="Declare ready only when this experiment can execute with the available inputs, specified controls and numerical checks. Declare blocked when unresolved requirements prevent execution. This is an agent declaration, not scientific validation.",
    )
    blocking_reasons: list[Text] = Field(
        default_factory=list, max_length=12,
        description="Nonempty reasons preventing execution when blocked; an empty list when ready. A blocked design is saved without freezing it for execution.",
    )
    experiment_id: Identifier
    baseline_id: Identifier
    candidate_ids: list[Identifier] = Field(min_length=1, max_length=5)
    metrics: list[Metric] = Field(min_length=1, max_length=6)
    required_check_ids: list[Identifier] = Field(min_length=1, max_length=8)
    numeric_checks: list[NumericCheck] = Field(default_factory=list, max_length=32)
    comparison_controls: Text
    uncertainty_plan: Text
    stopping_rule: Text
    evidence_gaps: list[Text] = Field(max_length=12)

    @model_validator(mode="after")
    def unique(self):
        if (self.execution_readiness == "blocked") != bool(self.blocking_reasons):
            raise ValueError("Blocked readiness requires reasons; ready or unknown cannot carry blocking reasons")
        for ids in (
            self.candidate_ids,
            self.required_check_ids,
            [m.id for m in self.metrics],
            [c.id for c in self.numeric_checks],
        ):
            if len(ids) != len(set(ids)):
                raise ValueError("Protocol identifiers must be unique")
        if self.baseline_id in self.candidate_ids:
            raise ValueError("Baseline cannot also be a candidate")
        return self

    @classmethod
    def parse(cls, value):
        parsed = cls.model_validate(value)
        if parsed.execution_readiness == "unknown" or "blocking_reasons" not in value:
            raise Invalid("New protocols require explicit ready or blocked execution_readiness and blocking_reasons")
        return parsed.model_dump()

    @classmethod
    def json_schema(cls):
        schema = cls.model_json_schema()
        # Defaults support stored legacy records only, never new proposals.
        schema["required"] = list(schema["properties"])
        for field in ("numeric_checks", "execution_readiness", "blocking_reasons"):
            schema["properties"][field].pop("default", None)
        schema["properties"]["execution_readiness"]["enum"] = ["ready", "blocked"]
        return schema


def require_execution_ready(record):
    """A host preflight for new computation, not a scientific-validity assertion."""
    try:
        protocol = ExperimentProtocol.model_validate({
            key: value for key, value in record["data"].items()
            if key in ExperimentProtocol.model_fields
        })
    except ValueError as exc:
        raise Invalid("Experiment protocol has an invalid readiness contract") from exc
    if protocol.execution_readiness != "ready":
        reason = "; ".join(protocol.blocking_reasons) or "Legacy protocol has no explicit execution-readiness declaration"
        raise Invalid("Experiment execution is " + protocol.execution_readiness + ": " + reason[:1500]
                      + ". Save a new ready protocol before starting new computation; existing outputs remain collectable.")
    if record["data"].get("status") != "frozen_for_comparison":
        raise Invalid("Ready experiment protocol must be frozen before starting new computation")
    _validate_numeric_freeze([check.model_dump() for check in protocol.numeric_checks])
    return protocol


def _validate_numeric_freeze(checks):
    checked_files = {check["csv_filename"] for check in checks}
    finite_files = {check["csv_filename"] for check in checks if check["operation"] == "finite"}
    if not checks or checked_files != finite_files or not any(check["operation"] != "finite" for check in checks):
        raise Invalid("New protocols require finite checks for every checked CSV and at least one substantive numerical check")


class MetricValue(ClosedContract):
    metric_id: Identifier
    value: float
    lower: float | None
    upper: float | None

    @model_validator(mode="after")
    def interval(self):
        if (self.lower is None) != (self.upper is None):
            raise ValueError("Provide both interval endpoints or neither")
        if self.lower is not None and not self.lower <= self.value <= self.upper:
            raise ValueError("Reported interval must include its estimate")
        return self


class Diagnostic(ClosedContract):
    check_id: Identifier
    passed: bool
    detail: Text


class Trial(ClosedContract):
    alternative_id: Identifier
    metrics: list[MetricValue] = Field(min_length=1, max_length=6)
    checks: list[Diagnostic] = Field(min_length=1, max_length=8)


class ExperimentOutput(ClosedContract):
    protocol_id: str
    basis: Literal["synthetic", "conditional_model", "observational_data"]
    trials: list[Trial] = Field(min_length=2, max_length=6)
    data_description: Text
    uncertainty_method: Text
    limitations: list[Text] = Field(min_length=1, max_length=12)


def plan_experiment(store, provider, problem_id, system_id=None, instruction=""):
    system = current_artifact(store, problem_id, "complex_system", system_id)
    from symplex.agents.context import working_context
    from symplex.agents.request_context import bounded_proposal_context

    supplied = bounded_proposal_context(store, provider, ExperimentProtocol, {
        "system": system["data"], "system_id": system["id"],
        "instruction": prompt("experiment_designer"),
        "requested_task": instruction,
    }, {"context": working_context(store, problem_id, excluded_kinds=("complex_system",))},
        problem_id, primary_ids=(system["id"],))

    proposal = provider.propose(
        ExperimentProtocol,
        supplied,
        problem_id,
    )
    proposal = ExperimentProtocol.parse(proposal)
    experiment = next(
        (
            e
            for e in system["data"]["experiments"]
            if e["id"] == proposal["experiment_id"]
        ),
        None,
    )
    if (
        not experiment
        or proposal["baseline_id"] != experiment["baseline_alternative_id"]
        or set(proposal["candidate_ids"])
        != set(experiment["candidate_alternative_ids"])
    ):
        raise Invalid("Protocol must preserve the selected experiment's alternatives")
    if not set(proposal["required_check_ids"]) <= {
        c["id"] for c in system["data"]["validation"]["checks"]
    }:
        raise Invalid("Protocol references unknown validation checks")
    if proposal["execution_readiness"] == "ready":
        _validate_numeric_freeze(proposal["numeric_checks"])
    return store.put(
        "experiment_protocol",
        {
            **proposal,
            "system_id": system["id"],
            "system_digest": system["digest"],
            "requested_task": instruction,
            "status": "frozen_for_comparison" if proposal["execution_readiness"] == "ready" else "blocked_design",
            "result_contract": ExperimentOutput.model_json_schema(),
            "scope": "Agent-declared execution readiness with host-checked structure. Ready protocols freeze numerical checks; blocked designs require a new ready protocol before execution. Neither readiness nor mathematical checks establish empirical validity.",
        },
        problem_id,
    )


def compare_computation(store, problem_id, package_id):
    from symplex.connectors.compute import read_blob

    package = current_artifact(store, problem_id, "compute_package", package_id)
    run = current_artifact(store, problem_id, "compute_run", package["data"]["run_id"])
    if not any(c.get("status") == "completed" for c in run["data"]["calls"]):
        raise Invalid("Comparison requires a completed sandbox execution")
    files = [
        current_artifact(store, problem_id, "file_blob", i)
        for i in package["data"]["file_ids"]
    ]
    outputs = [r for r in files if r["data"]["filename"] == "symplex_experiment.json"]
    if (
        len(outputs) != 1
        or outputs[0]["data"].get("run_id") != run["id"]
        or outputs[0]["data"].get("basis") != "generated"
    ):
        raise Invalid("One generated symplex_experiment.json from this run is required")
    raw = read_blob(store, outputs[0])
    if len(raw) > 200000:
        raise Invalid("Experiment result exceeds 200 KB")
    result = ExperimentOutput.model_validate(json.loads(raw))
    protocol_record = current_artifact(
        store, problem_id, "experiment_protocol", result.protocol_id
    )
    def supplied(record):
        return any(
            isinstance(item, dict)
            and item.get("id") == record["id"]
            and item.get("digest") == record["digest"]
            for item in run["data"].get("input_manifest", [])
        )

    if (
        protocol_record["data"].get("status") != "frozen_for_comparison"
        or protocol_record["created"] >= run["created"]
        or not supplied(protocol_record)
    ):
        raise Invalid("Protocol must be frozen and supplied before this execution")
    problem = store.get(problem_id)
    if not supplied(problem):
        raise Invalid("Comparison run must be bound to the current problem digest")
    system = current_artifact(
        store, problem_id, "complex_system", protocol_record["data"]["system_id"]
    )
    if (
        protocol_record["data"].get("system_digest") != system["digest"]
        or not supplied(system)
    ):
        raise Invalid("Frozen protocol system must match the supplied execution digest")
    protocol = ExperimentProtocol.model_validate(
        {
            k: protocol_record["data"][k]
            for k in ExperimentProtocol.model_fields
            if k in protocol_record["data"]
        }
    )
    if protocol.execution_readiness == "blocked":
        raise Invalid("Blocked experiment designs cannot qualify for comparison")
    expected = {protocol.baseline_id, *protocol.candidate_ids}
    if (
        len(result.trials) != len(expected)
        or {t.alternative_id for t in result.trials} != expected
    ):
        raise Invalid("Result must cover every frozen alternative exactly once")
    metrics = {m.id: m for m in protocol.metrics}
    trials = {}
    for trial in result.trials:
        if len(trial.metrics) != len(metrics) or {
            m.metric_id for m in trial.metrics
        } != set(metrics):
            raise Invalid("Every trial must report each frozen metric exactly once")
        if len(trial.checks) != len(protocol.required_check_ids) or {
            c.check_id for c in trial.checks
        } != set(protocol.required_check_ids):
            raise Invalid("Every trial must report each frozen diagnostic exactly once")
        trials[trial.alternative_id] = trial
    numerical = (
        execute_numeric_checks(
            store,
            problem_id,
            run["id"],
            package["data"]["file_ids"],
            protocol.numeric_checks,
        )
        if protocol.numeric_checks and protocol.execution_readiness == "ready"
        else {
            "status": "unavailable",
            "all_passed": False,
            "checks": [],
            "source_manifest": [],
            "independently_validated": False,
            "scope": "Legacy protocol has no explicit execution readiness or no frozen host numerical checks; no numerical eligibility is established",
        }
    )
    numerical_digest = digest(numerical)
    existing = [
        r
        for r in store.list("experiment_comparison")
        if r["parent"] == problem_id
        and not r["stale"]
        and r["data"].get("package_id") == package_id
        and r["data"].get("output_digest") == outputs[0]["digest"]
        and r["data"].get("protocol_digest") == protocol_record["digest"]
        and r["data"].get("evaluator_version") == EVALUATOR_VERSION
        and r["data"].get("numerical_result_digest") == numerical_digest
    ]
    if existing:
        verification_id = existing[-1]["data"].get("numerical_verification_id")
        if verification_id:
            current_artifact(
                store, problem_id, "numerical_verification", verification_id
            )
        return existing[-1]["id"]
    verification_id = store.put(
        "numerical_verification",
        {
            **numerical,
            "evaluator_version": EVALUATOR_VERSION,
            "protocol_id": protocol_record["id"],
            "protocol_digest": protocol_record["digest"],
            "package_id": package_id,
            "run_id": run["id"],
        },
        problem_id,
    )
    numerical_ok = numerical["all_passed"]
    baseline = {m.metric_id: m.value for m in trials[protocol.baseline_id].metrics}
    baseline_ok = all(c.passed for c in trials[protocol.baseline_id].checks)
    comparisons = []
    for ident in protocol.candidate_ids:
        trial = trials[ident]
        deltas = {
            m.metric_id: (m.value - baseline[m.metric_id])
            * (1 if metrics[m.metric_id].direction == "maximize" else -1)
            for m in trial.metrics
        }
        maker_ok = baseline_ok and all(c.passed for c in trial.checks)
        diagnostic_ok = maker_ok and numerical_ok
        no_regression = all(
            v >= -metrics[k].maximum_degradation for k, v in deltas.items()
        )
        improved = any(
            v > 0 and v >= metrics[k].minimum_improvement for k, v in deltas.items()
        )
        comparisons.append(
            {
                "candidate_id": ident,
                "improvement_deltas": deltas,
                "diagnostics_passed": diagnostic_ok,
                "maker_diagnostics_passed": maker_ok,
                "host_numerical_checks_passed": numerical_ok,
                "status": "meets_declared_comparison"
                if diagnostic_ok and no_regression and improved
                else "not_established",
                "intervals": [m.model_dump() for m in trial.metrics],
            }
        )
    # Nondominated candidates are retained even when they express different tradeoffs.
    feasible = [
        t for t in result.trials if numerical_ok and all(c.passed for c in t.checks)
    ]
    values = {
        t.alternative_id: {
            m.metric_id: m.value
            * (1 if metrics[m.metric_id].direction == "maximize" else -1)
            for m in t.metrics
        }
        for t in feasible
    }
    frontier = [
        i
        for i, v in values.items()
        if not any(
            j != i
            and all(w[k] >= v[k] for k in metrics)
            and any(w[k] > v[k] for k in metrics)
            for j, w in values.items()
        )
    ]
    return store.put(
        "experiment_comparison",
        {
            "protocol_id": result.protocol_id,
            "protocol_digest": protocol_record["digest"],
            "execution_readiness": protocol.execution_readiness,
            "system_id": protocol_record["data"]["system_id"],
            "package_id": package_id,
            "run_id": run["id"],
            "output_id": outputs[0]["id"],
            "output_digest": outputs[0]["digest"],
            "evaluator_version": EVALUATOR_VERSION,
            "numerical_verification_id": verification_id,
            "numerical_result_digest": numerical_digest,
            "numerical_verification": {
                "status": numerical["status"],
                "all_passed": numerical_ok,
                "check_count": len(numerical["checks"]),
                "failed_check_ids": [
                    c["id"] for c in numerical["checks"] if not c["passed"]
                ],
                "scope": numerical["scope"],
            },
            "baseline_id": protocol.baseline_id,
            "comparisons": comparisons,
            "pareto_alternative_ids": frontier,
            "basis": result.basis,
            "basis_authority": "maker_declared_unverified",
            "metrics": [m.model_dump() for m in protocol.metrics],
            "uncertainty_method": result.uncertainty_method,
            "limitations": result.limitations,
            "independently_validated": False,
            "promotion_allowed": False,
            "scope": "Host numerical checks verify selected generated CSV values under frozen limits. Summary metrics, fixture completeness, maker diagnostics and intervals remain unverified. This is not a statistical significance test, empirical validation or scientific promotion.",
        },
        problem_id,
    )
