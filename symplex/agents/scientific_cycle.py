"""Agent roles for representation and falsifiability review, with host-owned lineage."""

from typing import ClassVar, Literal

from pydantic import Field

from symplex.agents.context import working_context
from symplex.agents.prompts import prompt
from symplex.core.contracts import Invalid
from symplex.modeling.complex_system import (
    ClosedContract,
    ComplexSystemSpec,
    Identifier,
    Text,
    validate_artifact_references,
)


def current_artifact(store, problem_id, kind, ident=None):
    problem = store.get(problem_id)
    if problem["kind"] != "workspace_problem" or problem["stale"]:
        raise Invalid("Select a current problem")
    records = [
        r for r in store.list(kind) if r["parent"] == problem_id and not r["stale"]
    ]
    selected = (
        next((r for r in records if r["id"] == ident), None)
        if ident
        else next(iter(records[::-1]), None)
    )
    if selected is None:
        raise Invalid("A current " + kind + " artifact is required for this problem")
    return selected


def represent_system(store, provider, problem_id, instruction=""):
    from symplex.connectors.registry import catalog

    problem = store.get(problem_id)
    if problem["kind"] != "workspace_problem" or problem["stale"]:
        raise Invalid("Select a current problem")
    context = working_context(store, problem_id)
    result = provider.propose(
        ComplexSystemSpec,
        {
            "problem": problem["data"],
            "artifacts": context,
            "capabilities": catalog(),
            "user_task": instruction,
            "instruction": prompt("complexity_architect"),
        },
        problem_id,
    )
    # References must be in the actual context the agent received, not just guessed IDs.
    result = validate_artifact_references(result, [r["id"] for r in context])
    result.update(
        {
            "parent_system_id": next(
                (r["id"] for r in context if r["kind"] == "complex_system"), None
            ),
            "input_manifest": [
                {k: r[k] for k in ("id", "kind", "digest")} for r in context
            ],
            "status": "proposed",
            "execution_status": "not_executed",
            "validation_scope": "Host-checked references, declared units, ports and clocks; mechanisms remain hypotheses",
        }
    )
    return store.put("complex_system", result, problem_id)


class HypothesisAssessment(ClosedContract):
    hypothesis_id: Identifier
    status: Literal[
        "testable", "needs_reformulation", "not_identifiable_with_current_evidence"
    ]
    issue: Text
    discriminating_test: Text
    confounders: list[Text] = Field(max_length=8)
    evidence_ids: list[Text] = Field(max_length=12)


class HypothesisReview(ClosedContract):
    output_token_budget: ClassVar[int] = 4000
    assessments: list[HypothesisAssessment] = Field(min_length=1, max_length=6)
    coverage_gaps: list[Text] = Field(max_length=10)
    shared_failure_modes: list[Text] = Field(max_length=10)
    next_action: Text

    @classmethod
    def parse(cls, value):
        return cls.model_validate(value).model_dump()

    @classmethod
    def json_schema(cls):
        return cls.model_json_schema()


def review_hypotheses(store, provider, problem_id, target_id=None, instruction=""):
    from symplex.agents.request_context import bounded_proposal_context
    system = current_artifact(store, problem_id, "complex_system", target_id)
    supplied = bounded_proposal_context(store, provider, HypothesisReview, {
        "system_id": system["id"], "representation": system["data"],
        "instruction": prompt("hypothesis_critic"), "requested_task": instruction,
    }, {"evidence": working_context(store, problem_id, excluded_kinds=("complex_system",))},
        problem_id, role="validator", primary_ids=(system["id"],))
    context = supplied["evidence"]
    result = provider.propose(
        HypothesisReview,
        supplied,
        problem_id,
        role="validator",
    )
    result = HypothesisReview.parse(result)
    expected = {h["id"] for h in system["data"]["hypotheses"]}
    actual = [a["hypothesis_id"] for a in result["assessments"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise Invalid("Review must assess each current hypothesis exactly once")
    if any(
        i not in {r["id"] for r in context}
        for a in result["assessments"]
        for i in a["evidence_ids"]
    ):
        raise Invalid("Reviewer cited an unavailable artifact")
    return store.put(
        "hypothesis_review",
        {
            **result,
            "system_id": system["id"],
            "system_digest": system["digest"],
            "requested_task": instruction,
            "scope": "Separate-model testability critique; not proof of hypothesis truth",
        },
        problem_id,
    )
