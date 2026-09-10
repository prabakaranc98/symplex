"""The agent's modeling plan, checked by the host before any compute is spent.

Cheap rejection before expensive execution is the highest-leverage check in the system.
Every violation `validate_plan` raises here costs nothing; the same defect discovered
after a hosted sandbox run costs a paid execution, and after delivery costs the user's
trust in the whole investigation.

The plan exists chiefly to stop one documented failure. `docs/release-audit.md` records
an airport siting problem that became a dimensionless influence graph: physical queues,
capacity, legal environmental limits, costs and land use were never modelled, because
the agent reached for the one representation it always reaches for. A plan that must
name the rejected alternatives and say why this family beats them cannot silently
default to a template. It can still be wrong -- but it is wrong *on the record*, and a
reviewer can see the choice that was made.

Nothing in this module executes a mechanism, verifies a unit against reality, or judges
whether the chosen family is scientifically appropriate. It checks internal coherence.
"""

from typing import ClassVar, Literal

from pydantic import Field, ValidationError, model_validator

from symplex.authoring.testspec import TestSpec
from symplex.core.contracts import Invalid, digest
from symplex.modeling.complex_system import ClosedContract, Identifier, Text

SCOPE = (
    "Host coherence check of an agent-authored plan before compute is spent. It does "
    "not establish that the chosen mechanism is appropriate, that declared units are "
    "physically meaningful, or that the model will be valid."
)

Provenance = Literal["measured", "fitted", "assumed", "synthetic"]

PROVENANCE_MEANING = {
    "measured": "Taken from a supplied observation. Cite the artifact it came from.",
    "fitted": "Estimated from data inside this work. Say from what, and against what held-out claim.",
    "assumed": "Chosen by the agent. Say why, and what would change if it were wrong.",
    "synthetic": "Invented for a fixture. Never present a synthetic value as measured.",
}


class StateVariable(ClosedContract):
    id: Identifier
    name: Text
    unit: Text | None = Field(
        description="Declared unit symbol. Null is permitted by the contract so that the host, not the parser, reports the omission -- but a state without a unit is not a modellable quantity and the plan will be rejected."
    )
    dimension: Text | None = Field(
        description="Canonical quantity dimension, e.g. mass, mass/time, dimensionless. Counts and currencies stay distinct from physical dimensions."
    )
    meaning: Text


class PlanParameter(ClosedContract):
    id: Identifier
    name: Text
    unit: Text | None
    provenance: Provenance
    source: Text = Field(
        description="Where the value came from: the artifact for measured, the fitting target for fitted, the reasoning for assumed, the fixture role for synthetic."
    )
    value: float | None


class RejectedAlternative(ClosedContract):
    """A model family that was considered and set aside, with the reason."""

    family: Text
    why_rejected: Text = Field(
        description="Why this family loses for THIS question. 'Too complex' is not a reason; 'cannot represent the queue that the decision turns on' is."
    )


class ModelVariant(ClosedContract):
    """One runnable structure. The baseline and the rival must differ structurally."""

    id: Identifier
    name: Text
    mechanism: Text
    structural_elements: list[Text] = Field(
        min_length=1,
        max_length=16,
        description="The structural commitments that make this variant what it is: terms, couplings, laws, agent rules. Two variants with the same set are the same model with different numbers.",
    )
    parameterization: list[Text] = Field(max_length=16)


class Invariant(ClosedContract):
    id: Identifier
    statement: Text
    quantity: Text
    tolerance: float | None = Field(
        description="Absolute tolerance for the invariant, or null when the invariant is structural rather than numerical."
    )


class ModelingPlan(ClosedContract):
    """A plan written before code, so the host can reject an incoherent one for free."""

    output_token_budget: ClassVar[int] = 7000

    title: Text
    question: Text = Field(
        description="The decision-relevant question this model is supposed to inform."
    )
    mechanism_family: Text = Field(
        description="The model family actually chosen: compartment ODE, agent-based population, spatial network, stochastic process, Bayesian hierarchical, symbolic constraint system, ensemble, or another named family."
    )
    mechanism_rationale: Text = Field(
        description="Why this family, for this question, given this evidence. Not why modelling is useful in general."
    )
    rejected_alternatives: list[RejectedAlternative] = Field(
        max_length=8,
        description="Families considered and set aside. An empty list means no choice was made; the plan is rejected.",
    )
    states: list[StateVariable] = Field(min_length=1, max_length=24)
    parameters: list[PlanParameter] = Field(max_length=32)
    baseline: ModelVariant
    rival: ModelVariant
    invariants: list[Invariant] = Field(max_length=12)
    tests: list[TestSpec] = Field(max_length=32)
    disconfirmation: Text = Field(
        description="The observation or result that would show this model is the wrong answer to this question. Not a caveat -- a condition."
    )
    declared_outputs: list[Text] = Field(
        max_length=8,
        description="The files this program will produce, matching the simulation-engineer output contract.",
    )
    known_gaps: list[Text] = Field(max_length=12)

    @model_validator(mode="after")
    def unique_identifiers(self):
        for items, label in (
            (self.states, "state"),
            (self.parameters, "parameter"),
            (self.invariants, "invariant"),
            (self.tests, "test"),
        ):
            ids = [i.id for i in items]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate {label} ID in the modeling plan")
        return self

    @classmethod
    def json_schema(cls):
        return cls.model_json_schema()


def _normalized_structure(variant):
    return tuple(sorted(e.strip().lower() for e in variant.structural_elements))


def parse_plan(plan):
    """Parse without judging coherence; raises Invalid on a malformed contract."""
    if isinstance(plan, ModelingPlan):
        return plan
    try:
        return ModelingPlan.model_validate(plan)
    except ValidationError as exc:
        raise Invalid("Modeling plan has an invalid contract: " + str(exc)[:1500]) from exc


def validate_plan(plan, *, require_protected_claim=True):
    """Reject an incoherent plan before any compute is spent. Raises `Invalid` on failure.

    The checks are structural and cheap on purpose:

    * every state carries a unit and a dimension -- an unnamed quantity cannot be
      conserved, converted or compared, and the host cannot check it later;
    * the rejected-alternatives list is non-empty -- a mechanism that was never chosen
      against anything is a template, and templates are how an airport became an
      influence graph;
    * the rival differs structurally from the baseline -- two parameterizations of one
      structure are a sensitivity sweep, not a discriminating comparison;
    * at least one invariant is declared -- otherwise nothing constrains the numerics;
    * at least one failure-case test exists -- a model that is never asserted to fail
      has no stated regime of validity;
    * at least one test is host-recomputable (`require_protected_claim`) -- without one,
      every claim the run produces is a maker-reported boolean, which is exactly the
      risk `docs/release-audit.md` finding 2 names.

    Passing establishes internal coherence only. It does not establish that the chosen
    mechanism suits the domain, that the units are physically right, or that the model
    will be valid. Those need evidence this function never sees.
    """
    parsed = parse_plan(plan)
    violations = []

    missing_units = [s.id for s in parsed.states if not (s.unit or "").strip()]
    if missing_units:
        violations.append(
            "States without a declared unit cannot be modelled or checked: "
            + ", ".join(sorted(missing_units)[:12])
        )
    missing_dimensions = [s.id for s in parsed.states if not (s.dimension or "").strip()]
    if missing_dimensions:
        violations.append(
            "States without a declared dimension cannot be dimension-checked: "
            + ", ".join(sorted(missing_dimensions)[:12])
        )

    if not parsed.rejected_alternatives:
        violations.append(
            "No rejected alternatives: name the model families considered and why this one "
            "beats them for this question, or the choice was a default rather than a decision"
        )

    if parsed.baseline.id == parsed.rival.id:
        violations.append("Baseline and rival must be distinct variants")
    if _normalized_structure(parsed.baseline) == _normalized_structure(parsed.rival):
        violations.append(
            "Rival is structurally identical to the baseline: the declared structural "
            "elements match, so this is a parameter sweep and not a discriminating comparison"
        )
    if parsed.baseline.mechanism.strip().lower() == parsed.rival.mechanism.strip().lower():
        violations.append(
            "Rival restates the baseline mechanism verbatim; a rival must propose a different mechanism"
        )

    if not parsed.invariants:
        violations.append(
            "No declared invariant: state at least one quantity the mechanism must conserve or bound"
        )

    kinds = {t.kind for t in parsed.tests}
    if "failure_case" not in kinds:
        violations.append(
            "No failure_case test: declare a regime in which this model is asserted to fail or refuse"
        )

    protected = [t.id for t in parsed.tests if t.numeric_check is not None]
    if require_protected_claim and not protected:
        violations.append(
            "No host-recomputable test: without at least one NumericCheck every result is a "
            "maker-reported boolean and nothing from this run can become evidence"
        )

    if violations:
        raise Invalid(
            "Modeling plan is not internally coherent: " + "; ".join(violations)[:4000]
        )

    provenance_counts = {}
    for parameter in parsed.parameters:
        provenance_counts[parameter.provenance] = (
            provenance_counts.get(parameter.provenance, 0) + 1
        )
    baseline_only = sorted(
        set(_normalized_structure(parsed.baseline)) - set(_normalized_structure(parsed.rival))
    )
    rival_only = sorted(
        set(_normalized_structure(parsed.rival)) - set(_normalized_structure(parsed.baseline))
    )
    data = parsed.model_dump()
    return {
        "status": "coherent",
        "plan": data,
        "plan_digest": digest(data),
        "mechanism_family": parsed.mechanism_family,
        "rejected_alternative_count": len(parsed.rejected_alternatives),
        "state_count": len(parsed.states),
        "parameter_count": len(parsed.parameters),
        "parameter_provenance_counts": provenance_counts,
        "unsupported_parameter_ids": [
            p.id for p in parsed.parameters if p.provenance in ("assumed", "synthetic")
        ],
        "invariant_ids": [i.id for i in parsed.invariants],
        "test_kinds": sorted(kinds),
        "protected_test_ids": protected,
        "structural_difference": {
            "baseline_only": baseline_only[:16],
            "rival_only": rival_only[:16],
        },
        "checks_performed": [
            "states_have_units",
            "states_have_dimensions",
            "rejected_alternatives_present",
            "rival_differs_structurally_from_baseline",
            "invariant_declared",
            "failure_case_test_declared",
            "host_recomputable_claim_declared"
            if require_protected_claim
            else "host_recomputable_claim_not_required",
        ],
        "independently_validated": False,
        "scope": SCOPE,
    }
