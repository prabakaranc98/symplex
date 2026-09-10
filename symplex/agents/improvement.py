"""Separate intervention design from system representation and method proposals."""

from typing import ClassVar, Literal

from pydantic import Field

from symplex.agents.context import working_context
from symplex.agents.prompts import prompt, manifest
from symplex.agents.scientific_cycle import current_artifact
from symplex.core.contracts import Invalid
from symplex.modeling.complex_system import ClosedContract, Identifier, Text


class ProposalContract(ClosedContract):
    @classmethod
    def parse(cls, value):
        return cls.model_validate(value).model_dump()

    @classmethod
    def json_schema(cls):
        return cls.model_json_schema()


class DesignChange(ClosedContract):
    component_id: Identifier
    description: Text
    expected_observable_change: Text
    check_ids: list[Identifier] = Field(min_length=1, max_length=8)


class CandidateDesign(ProposalContract):
    output_token_budget: ClassVar[int] = 4000
    name: Text
    alternative_id: Identifier
    objective: Text
    design_specification: Text
    changes: list[DesignChange] = Field(min_length=1, max_length=10)
    evidence_ids: list[Text] = Field(max_length=12)
    assumptions: list[Text] = Field(max_length=12)
    feasibility_gaps: list[Text] = Field(max_length=12)
    expected_tradeoffs: list[Text] = Field(min_length=1, max_length=8)
    required_experiment_ids: list[Identifier] = Field(min_length=1, max_length=8)
    acceptance_criterion: Text
    reversal_conditions: list[Text] = Field(min_length=1, max_length=8)


class MethodChange(ClosedContract):
    role: Literal[
        "complexity_architect",
        "hypothesis_critic",
        "evidence_synthesist",
        "experiment_designer",
        "simulation_engineer",
        "metareasoner",
    ]
    current_failure: Text
    proposed_instruction: Text
    expected_effect: Text
    regression_risk: Text


class MethodCandidate(ProposalContract):
    output_token_budget: ClassVar[int] = 3500
    name: Text
    changes: list[MethodChange] = Field(min_length=1, max_length=4)
    feedback_ids: list[Text] = Field(min_length=1, max_length=12)
    applicable_problem_characteristics: list[Text] = Field(min_length=1, max_length=8)
    development_test: Text
    independent_validation_test: Text
    matched_resource_rule: Text
    acceptance_criterion: Text
    rollback_condition: Text
    transfer_limits: list[Text] = Field(min_length=1, max_length=8)


def develop_candidate(store, provider, problem_id, instruction="", parent_id=None):
    from symplex.agents.request_context import bounded_proposal_context
    system = current_artifact(store, problem_id, "complex_system")
    predecessor = (
        current_artifact(store, problem_id, "candidate_design", parent_id)
        if parent_id
        else None
    )
    supplied = bounded_proposal_context(store, provider, CandidateDesign, {
        "problem": store.get(problem_id)["data"], "system": system["data"],
        "task": instruction, "instruction": prompt("solution_designer"),
    }, {"context": working_context(store, problem_id, excluded_kinds=("complex_system",))},
        problem_id, primary_ids=(system["id"],), optional_records={"predecessor": predecessor})
    context = supplied["context"]
    result = provider.propose(
        CandidateDesign,
        supplied,
        problem_id,
    )
    result = CandidateDesign.parse(result)
    if result["alternative_id"] not in {
        a["id"] for a in system["data"]["decision"]["alternatives"]
    }:
        raise Invalid("Candidate must instantiate a declared decision alternative")
    if not set(result["required_experiment_ids"]) <= {
        e["id"] for e in system["data"]["experiments"]
    }:
        raise Invalid("Candidate references an unknown experiment")
    for experiment in system["data"]["experiments"]:
        if (experiment["id"] in result["required_experiment_ids"] and
            result["alternative_id"] not in {experiment["baseline_alternative_id"], *experiment["candidate_alternative_ids"]}):
            raise Invalid("Required experiment does not test this candidate alternative")
    for change in result["changes"]:
        if change["component_id"] not in {
            c["id"] for c in system["data"]["components"]
        } or not set(change["check_ids"]) <= {
            c["id"] for c in system["data"]["validation"]["checks"]
        }:
            raise Invalid("Candidate changes must target known components and checks")
    if not set(result["evidence_ids"]) <= {r["id"] for r in context}:
        raise Invalid("Candidate cited unavailable evidence")
    return store.put(
        "candidate_design",
        {
            **result,
            "system_id": system["id"],
            "system_digest": system["digest"],
            "parent_candidate_id": parent_id,
            "status": "proposed",
            "input_manifest": [
                {k: r[k] for k in ("id", "digest", "kind")} for r in context
            ],
            "scope": "Agent-authored intervention design; feasibility and benefit require linked execution/evidence",
        },
        problem_id,
    )


def propose_method(store, provider, problem_id, instruction=""):
    from symplex.agents.request_context import bounded_proposal_context
    allowed_feedback = {
        "model_critique",
        "hypothesis_review",
        "experiment_comparison",
        "review",
    }
    feedback = [
        r
        for r in store.list()
        if r["parent"] == problem_id
        and not r["stale"]
        and r["kind"] in allowed_feedback
    ][-8:]
    if not feedback:
        raise Invalid("Method improvement needs recorded evaluation feedback first")
    supplied = bounded_proposal_context(store, provider, MethodCandidate, {
        "role_versions": manifest(), "instruction": prompt("method_optimizer"),
        "requested_task": instruction,
    }, {"feedback": list(reversed(feedback))}, problem_id)
    feedback = supplied["feedback"]
    if not feedback:
        raise Invalid("No complete evaluation feedback fits the method proposal request")
    result = provider.propose(
        MethodCandidate,
        supplied,
        problem_id,
    )
    result = MethodCandidate.parse(result)
    if not set(result["feedback_ids"]) <= {r["id"] for r in feedback}:
        raise Invalid("Method proposal cited unavailable feedback")
    return store.put(
        "method_candidate",
        {
            **result,
            "parent_method": {p["name"]: p["digest"] for p in manifest()},
            "requested_task": instruction,
            "status": "awaiting_independent_evaluation",
            "installed": False,
            "promotion_allowed": False,
            "scope": "Frozen instruction proposal; no active change by itself. The supported architect role can enter staged contract replay and explicit operator review for a named-problem canary. Tool permissions, budgets and evaluation rules remain host-owned.",
        },
        problem_id,
    )
