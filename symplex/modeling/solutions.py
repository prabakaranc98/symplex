"""Problem-to-solution design contracts, independent of any particular dataset."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from symplex.agents.prompts import prompt


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Hypothesis(Contract):
    id: str
    mechanism: str
    assumptions: list[str]
    observable_prediction: str
    disconfirmation: str
    test: str


class Component(Contract):
    id: str
    name: str
    kind: Literal["evidence", "rule", "model", "solver", "tool", "delivery"]
    responsibility: str
    inputs: list[str]
    outputs: list[str]
    implementation: str
    asset_id: str
    readiness: Literal[
        "existing_adapter",
        "requires_implementation",
        "requires_data",
        "external_validation",
    ]


class Coupling(Contract):
    source: str
    target: str
    meaning: str
    units: str
    timing: str


class ExperimentPlan(Contract):
    baseline: str
    candidate: str
    observable_target: str
    primary_metric: str
    split_unit: str
    protected_confirmation: str
    failure_conditions: list[str]


class Subproblem(Contract):
    id: str
    question: str
    depends_on: list[str]
    evidence_needed: list[str]
    acceptance_check: str


class Alternative(Contract):
    name: str
    intervention: str
    desired_outcome: str
    cost_or_constraint: str


class SolutionBlueprint(Contract):
    title: str
    beneficiary: str
    decision: str
    success_criterion: str
    boundary: str
    entities: list[str]
    subproblems: list[Subproblem]
    alternatives: list[Alternative]
    state_variables: list[str]
    feedback: list[str]
    hard_constraints: list[str]
    evidence_needed: list[str]
    hypotheses: list[Hypothesis]
    components: list[Component]
    couplings: list[Coupling]
    training_plan: str
    inference_plan: str
    experiment: ExperimentPlan
    next_action: str
    expected_information_gain: str
    stop_condition: str
    consumable: str
    reversal_conditions: list[str]
    dependency_gaps: list[str]

    @model_validator(mode="after")
    def references(self):
        ids = {c.id for c in self.components}
        if len(ids) != len(self.components) or not 2 <= len(self.components) <= 12:
            raise ValueError("Provide 2–12 uniquely identified components")
        if not 2 <= len(self.hypotheses) <= 4:
            raise ValueError("Provide 2–4 competing hypotheses")
        if any(c.source not in ids or c.target not in ids for c in self.couplings):
            raise ValueError("Coupling references an unknown component")
        sub_ids = {p.id for p in self.subproblems}
        if len(sub_ids) != len(self.subproblems) or any(
            d not in sub_ids or d == p.id
            for p in self.subproblems
            for d in p.depends_on
        ):
            raise ValueError(
                "Subproblem dependencies must reference other identified subproblems"
            )
        return self

    @classmethod
    def parse(cls, data):
        return cls.model_validate(data).model_dump()

    @classmethod
    def json_schema(cls):
        return cls.model_json_schema()


def design_solution(store, provider, problem_id, research=False):
    from symplex.connectors.registry import catalog
    from symplex.domains.catalog import DATASETS
    from symplex.modeling.assets import ASSETS

    problem = store.get(problem_id)
    from symplex.agents.context import working_context

    known = working_context(store, problem_id)
    # Keep proposal context bounded; exact capabilities come from the running registry.
    context = {
        "problem": problem["data"],
        "evidence": known,
        "source_catalog": DATASETS,
        "model_assets": ASSETS,
        "connectors": catalog(),
        "executable_adapters": [
            "P1 prepared threshold-market pack",
            "BamTwoogle research-policy pilot",
        ],
        "instruction": prompt("system_architect"),
    }
    result = provider.propose(SolutionBlueprint, context, problem_id)
    # Availability is an application decision, not an LLM assertion.
    available = {a["id"] for a in ASSETS if a["status"] in ("executable", "api")}
    for component in result["components"]:
        component["readiness"] = (
            "existing_adapter"
            if component["asset_id"] in available
            else "requires_implementation"
        )
    result.update(
        parent_solution_id=next(
            (r["id"] for r in known if r["kind"] == "solution"), None
        ),
        status="proposed",
        execution_status="not_executed",
        authority="Astra proposal; host validation required",
        immutable_evaluator="Selected only when an executable domain adapter and benchmark are admitted",
    )
    return store.put("solution", result, problem_id)
