"""Agent-authored, domain-general complex-system hypotheses.

This contract checks declared interfaces, references and comparison plans. It does
not certify causal claims, unit definitions, parameter estimates, or suitability
for a consequential decision. Hybrid models and multiple modalities are choices,
not prerequisites. Nothing in this module executes an agent-proposed mechanism.
"""

import math
from collections.abc import Iterable
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Text = Annotated[str, Field(min_length=1, max_length=1600)]
Identifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]


class ClosedContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, allow_inf_nan=False, str_strip_whitespace=True
    )


class UnitDefinition(ClosedContract):
    id: Identifier
    symbol: Text
    dimension: Text = Field(
        description="Canonical declared quantity dimension, identical for compatible units. Semantic counts and currencies must remain distinct."
    )
    scale_to_canonical: float = Field(gt=0)
    offset_to_canonical: float


class TimeScale(ClosedContract):
    id: Identifier
    name: Text
    step_seconds: float | None = Field(
        description="Positive sampling or update interval, or null for event-indexed/unknown timing."
    )
    clock: Text = Field(
        description="Shared reference clock, such as UTC or elapsed time after a defined event."
    )

    @model_validator(mode="after")
    def positive_step(self):
        if self.step_seconds is not None and self.step_seconds <= 0:
            raise ValueError("step_seconds must be positive or null")
        return self


class SystemBoundary(ClosedContract):
    included: list[Text] = Field(min_length=1, max_length=12)
    excluded: list[Text] = Field(max_length=12)
    scales: list[TimeScale] = Field(min_length=1, max_length=8)


class ComplexityCharacteristics(ClosedContract):
    nonlinearity: Text
    stochasticity: Text
    cross_scale_effects: Text
    emergence: Text


class Entity(ClosedContract):
    id: Identifier
    name: Text
    meaning: Text


class SystemState(ClosedContract):
    id: Identifier
    entity_id: Identifier
    name: Text
    kind: Literal["observed", "latent", "decision", "exogenous"]
    unit_id: Identifier
    scale_id: Identifier
    meaning: Text


class ObservationChannel(ClosedContract):
    id: Identifier
    name: Text
    modality: Literal[
        "text",
        "table",
        "image",
        "audio",
        "video",
        "spatial",
        "time_series",
        "graph",
        "molecular",
        "sensor",
        "simulation",
        "human_judgment",
        "other",
    ]
    source_artifact_ids: list[Text] = Field(max_length=12)
    state_ids: list[Identifier] = Field(min_length=1, max_length=12)
    measurement_process: Text
    missingness: Text
    uncertainty: Text
    status: Literal["available", "proposed", "missing"]

    @model_validator(mode="after")
    def available_has_source(self):
        if self.status == "available" and not self.source_artifact_ids:
            raise ValueError(
                "An available observation channel requires source artifacts"
            )
        return self


class ModelPort(ClosedContract):
    id: Identifier
    direction: Literal["input", "output"]
    state_id: Identifier
    unit_id: Identifier
    scale_id: Identifier


class ModelComponent(ClosedContract):
    id: Identifier
    name: Text
    kind: Literal[
        "mechanistic",
        "probabilistic",
        "learned",
        "symbolic",
        "agent_based",
        "optimization",
        "measurement",
        "adapter",
    ]
    mechanism: Text
    ports: list[ModelPort] = Field(min_length=1, max_length=12)
    assumptions: list[Text] = Field(max_length=12)
    evidence_ids: list[Text] = Field(max_length=12)


class AffineConversion(ClosedContract):
    scale: float = Field(gt=0)
    offset: float


class ModelCoupling(ClosedContract):
    id: Identifier
    source_port: Identifier
    target_port: Identifier
    conversion: AffineConversion = Field(
        description="target = scale * source + offset. Coefficients must agree with declared unit definitions; changing quantity needs a model component."
    )
    time_alignment: Literal[
        "synchronous", "sample_hold", "interpolate", "aggregate", "event_mapping"
    ]
    alignment_rationale: Text
    mechanism: Text


class FeedbackLoop(ClosedContract):
    id: Identifier
    component_ids: list[Identifier] = Field(
        min_length=1,
        max_length=12,
        description="Ordered directed cycle in the declared component coupling graph; the last component returns to the first.",
    )
    description: Text
    polarity: Literal["reinforcing", "balancing", "mixed", "unknown"]
    nonlinearity: Text
    stochasticity: Text


class RivalHypothesis(ClosedContract):
    id: Identifier
    claim: Text
    rivals: list[Identifier] = Field(min_length=1, max_length=5)
    component_ids: list[Identifier] = Field(min_length=1, max_length=12)
    evidence_ids: list[Text] = Field(max_length=12)
    discriminating_test: Text
    expected_observation: Text
    rejection_condition: Text


class Endpoint(ClosedContract):
    id: Identifier
    name: Text
    state_id: Identifier
    direction: Literal["minimize", "maximize", "target", "characterize"]
    criterion: Text


class DecisionIntervention(ClosedContract):
    state_id: Identifier
    change: Text


class DecisionAlternative(ClosedContract):
    id: Identifier
    name: Text
    interventions: list[DecisionIntervention] = Field(max_length=12)
    rationale: Text


class DecisionConstraint(ClosedContract):
    id: Identifier
    description: Text
    state_ids: list[Identifier] = Field(max_length=12)
    test: Text


class DecisionFrame(ClosedContract):
    question: Text
    beneficiary: Text
    endpoints: list[Endpoint] = Field(min_length=1, max_length=8)
    alternatives: list[DecisionAlternative] = Field(min_length=2, max_length=6)
    constraints: list[DecisionConstraint] = Field(max_length=12)


class ValidationCheck(ClosedContract):
    id: Identifier
    name: Text
    component_ids: list[Identifier] = Field(min_length=1, max_length=12)
    method: Text
    acceptance_condition: Text
    evidence_ids: list[Text] = Field(max_length=12)


class ValidationPlan(ClosedContract):
    identifiability_gaps: list[Text] = Field(max_length=12)
    checks: list[ValidationCheck] = Field(min_length=1, max_length=8)
    calibration_plan: Text
    extrapolation_limits: list[Text] = Field(min_length=1, max_length=12)


class ComparativeExperiment(ClosedContract):
    id: Identifier
    name: Text
    hypothesis_ids: list[Identifier] = Field(min_length=1, max_length=6)
    baseline_alternative_id: Identifier
    candidate_alternative_ids: list[Identifier] = Field(min_length=1, max_length=5)
    endpoint_ids: list[Identifier] = Field(min_length=1, max_length=8)
    method: Text
    comparison_controls: Text
    uncertainty_plan: Text
    stopping_rule: Text


class Disturbance(ClosedContract):
    id: Identifier
    name: Text
    affected_state_ids: list[Identifier] = Field(min_length=1, max_length=12)
    perturbation: Text


class RecoveryObservable(ClosedContract):
    state_id: Identifier
    criterion: Text
    observation_window: Text


class ResiliencePlan(ClosedContract):
    disturbances: list[Disturbance] = Field(min_length=1, max_length=6)
    recovery_observables: list[RecoveryObservable] = Field(min_length=1, max_length=6)


def _unique(items, name):
    identifiers = [item.id for item in items]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"Duplicate {name} ID")
    return {item.id: item for item in items}


def _references(values, known, label, *, nonself=None):
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {label} reference")
    if not set(values).issubset(known):
        raise ValueError(
            f"Unknown {label} reference: {sorted(set(values) - set(known))}"
        )
    if nonself is not None and nonself in values:
        raise ValueError(f"{label} must not reference itself")


class ComplexSystemSpec(ClosedContract):
    """A testable proposed representation, never a claim of validated system truth."""

    output_token_budget: ClassVar[int] = 11000
    title: Text
    representation_rationale: Text = Field(
        description="Justify modalities, model families and abstraction level. A single modality or model family is valid; do not add complexity without a decision-relevant reason."
    )
    decision: DecisionFrame
    boundary: SystemBoundary
    complexity: ComplexityCharacteristics = Field(
        description="State the proposed sources of complexity or explicitly say absent, unknown, or outside scope. Complexity is not presumed."
    )
    units: list[UnitDefinition] = Field(min_length=1, max_length=12)
    entities: list[Entity] = Field(min_length=1, max_length=12)
    states: list[SystemState] = Field(min_length=1, max_length=18)
    observations: list[ObservationChannel] = Field(max_length=10)
    components: list[ModelComponent] = Field(min_length=1, max_length=10)
    couplings: list[ModelCoupling] = Field(max_length=20)
    feedback: list[FeedbackLoop] = Field(max_length=6)
    hypotheses: list[RivalHypothesis] = Field(min_length=2, max_length=6)
    experiments: list[ComparativeExperiment] = Field(min_length=1, max_length=6)
    validation: ValidationPlan
    resilience: ResiliencePlan | None = Field(
        description="Use only when disturbance response and recovery are relevant to the decision; otherwise null."
    )
    assumptions: list[Text] = Field(max_length=12)
    unresolved_questions: list[Text] = Field(max_length=12)

    @model_validator(mode="after")
    def validate_graph(self):
        units = _unique(self.units, "unit")
        scales = _unique(self.boundary.scales, "time scale")
        entities = _unique(self.entities, "entity")
        states = _unique(self.states, "state")
        components = _unique(self.components, "component")
        ports = _unique([p for c in self.components for p in c.ports], "port")
        hypotheses = _unique(self.hypotheses, "hypothesis")
        alternatives = _unique(self.decision.alternatives, "alternative")
        endpoints = _unique(self.decision.endpoints, "endpoint")
        for items, name in (
            (self.observations, "observation"),
            (self.couplings, "coupling"),
            (self.feedback, "feedback"),
            (self.experiments, "experiment"),
            (self.validation.checks, "validation check"),
            (self.decision.constraints, "decision constraint"),
        ):
            _unique(items, name)

        for state in self.states:
            _references([state.entity_id], entities, "state entity")
            _references([state.unit_id], units, "state unit")
            _references([state.scale_id], scales, "state time scale")
        for port in ports.values():
            _references([port.state_id], states, "port state")
            _references([port.unit_id], units, "port unit")
            _references([port.scale_id], scales, "port time scale")
            state = states[port.state_id]
            if units[port.unit_id].dimension != units[state.unit_id].dimension:
                raise ValueError(
                    f"Port {port.id} has a different declared dimension from its state"
                )
            if port.scale_id != state.scale_id:
                raise ValueError(
                    f"Port {port.id} must use its state's declared time scale"
                )

        for observation in self.observations:
            _references(observation.state_ids, states, "observation state")
        owner = {p.id: c.id for c in self.components for p in c.ports}
        component_edges, coupled_inputs = set(), set()
        for coupling in self.couplings:
            _references([coupling.source_port], ports, "coupling source port")
            _references([coupling.target_port], ports, "coupling target port")
            source, target = ports[coupling.source_port], ports[coupling.target_port]
            if source.direction != "output" or target.direction != "input":
                raise ValueError(
                    "Couplings must connect an output port to an input port"
                )
            if target.id in coupled_inputs:
                raise ValueError(
                    "Multiple producers for an input require an explicit fusion component"
                )
            coupled_inputs.add(target.id)
            self._validate_interface(coupling, source, target, units, scales)
            component_edges.add((owner[source.id], owner[target.id]))

        for feedback in self.feedback:
            _references(feedback.component_ids, components, "feedback component")
            cycle = feedback.component_ids
            if any(
                (a, b) not in component_edges
                for a, b in zip(cycle, cycle[1:] + cycle[:1])
            ):
                raise ValueError(
                    "Feedback must form an explicit directed cycle through declared couplings"
                )
        for hypothesis in self.hypotheses:
            _references(
                hypothesis.rivals, hypotheses, "rival hypothesis", nonself=hypothesis.id
            )
            _references(hypothesis.component_ids, components, "hypothesis component")
        for endpoint in self.decision.endpoints:
            _references([endpoint.state_id], states, "decision endpoint state")
        for alternative in self.decision.alternatives:
            _references(
                [i.state_id for i in alternative.interventions],
                states,
                "intervention state",
            )
        for constraint in self.decision.constraints:
            _references(constraint.state_ids, states, "constraint state")
        for check in self.validation.checks:
            _references(check.component_ids, components, "validation component")
        for experiment in self.experiments:
            _references(experiment.hypothesis_ids, hypotheses, "experiment hypothesis")
            _references(
                [experiment.baseline_alternative_id],
                alternatives,
                "experiment baseline",
            )
            _references(
                experiment.candidate_alternative_ids,
                alternatives,
                "experiment candidate",
            )
            _references(experiment.endpoint_ids, endpoints, "experiment endpoint")
            if (
                experiment.baseline_alternative_id
                in experiment.candidate_alternative_ids
            ):
                raise ValueError("Experiment baseline cannot also be a candidate")
        if self.resilience is not None:
            _unique(self.resilience.disturbances, "disturbance")
            for disturbance in self.resilience.disturbances:
                _references(disturbance.affected_state_ids, states, "disturbance state")
            _references(
                [r.state_id for r in self.resilience.recovery_observables],
                states,
                "recovery observable state",
            )
        return self

    @staticmethod
    def _validate_interface(coupling, source, target, units, scales):
        source_unit, target_unit = units[source.unit_id], units[target.unit_id]
        if source_unit.dimension != target_unit.dimension:
            raise ValueError(
                "Incompatible declared dimensions; use an explicit transformation component"
            )
        expected_scale = source_unit.scale_to_canonical / target_unit.scale_to_canonical
        expected_offset = (
            source_unit.offset_to_canonical - target_unit.offset_to_canonical
        ) / target_unit.scale_to_canonical
        if not (
            math.isfinite(expected_scale)
            and expected_scale > 0
            and math.isfinite(expected_offset)
        ):
            raise ValueError(
                "Declared unit conversion exceeds finite numerical precision"
            )
        if not (
            math.isclose(
                coupling.conversion.scale, expected_scale, rel_tol=1e-8, abs_tol=0
            )
            and math.isclose(
                coupling.conversion.offset, expected_offset, rel_tol=1e-8, abs_tol=0
            )
        ):
            raise ValueError("Affine conversion does not agree with declared units")
        source_scale, target_scale = scales[source.scale_id], scales[target.scale_id]
        if source_scale.clock != target_scale.clock:
            raise ValueError(
                "Different reference clocks require an explicit clock-mapping component"
            )
        method = coupling.time_alignment
        source_step, target_step = source_scale.step_seconds, target_scale.step_seconds
        if method == "synchronous":
            if source_step is None or target_step is None:
                raise ValueError(
                    "Unknown/event timing requires an explicit event_mapping alignment"
                )
            if source_step != target_step:
                raise ValueError(
                    "Synchronous coupling requires equal declared sampling intervals"
                )
        elif method != "event_mapping":
            if source_step is None or target_step is None:
                raise ValueError(
                    "Unknown/event timing requires an explicit event_mapping alignment"
                )
            if method == "aggregate" and target_step < source_step:
                raise ValueError(
                    "Aggregation cannot map coarse samples to a finer interval"
                )
            if method == "interpolate" and target_step > source_step:
                raise ValueError(
                    "Interpolation to a coarser interval requires an explicit aggregation or sampling plan"
                )

    @classmethod
    def parse(cls, data):
        return cls.model_validate(data).model_dump()

    @classmethod
    def json_schema(cls):
        return cls.model_json_schema()


def validate_artifact_references(
    data: dict | ComplexSystemSpec, available_ids: Iterable[str]
) -> dict:
    """Admit only existing, caller-scoped evidence refs; this is not evidence verification.

    The caller supplies non-stale artifacts belonging to the current problem. IDs
    cannot establish source quality, applicability, entailment, or causal truth.
    """
    spec = ComplexSystemSpec.model_validate(data)
    known = set(available_ids)
    reference_lists = [o.source_artifact_ids for o in spec.observations]
    reference_lists.extend(c.evidence_ids for c in spec.components)
    reference_lists.extend(h.evidence_ids for h in spec.hypotheses)
    reference_lists.extend(c.evidence_ids for c in spec.validation.checks)
    for references in reference_lists:
        _references(references, known, "source artifact")
    return spec.model_dump()
