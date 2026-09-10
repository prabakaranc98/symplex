"""Consumable decisions linked to source artifacts rather than unsupported summaries."""

from typing import ClassVar, Literal

from pydantic import Field

from symplex.agents.context import review_context
from symplex.agents.improvement import ProposalContract
from symplex.agents.prompts import prompt
from symplex.agents.scientific_cycle import current_artifact
from symplex.core.contracts import Invalid
from symplex.modeling.complex_system import ClosedContract, Identifier, Text


class OutcomeClaim(ClosedContract):
    statement: Text
    basis: Literal[
        "observed", "conditional_model", "synthetic_test", "assumption", "judgment", "artifact_inventory"
    ]
    source_ids: list[Text] = Field(min_length=1, max_length=8)
    limitation: Text


class AlternativeAssessment(ClosedContract):
    alternative_id: Identifier
    assessment: Text
    benefit_and_tradeoff: Text
    unresolved_test: Text


class DecisionBrief(ProposalContract):
    output_token_budget: ClassVar[int] = 4000
    title: Text
    decision_owner: Text
    recommendation: Text
    claims: list[OutcomeClaim] = Field(max_length=10)
    alternatives: list[AlternativeAssessment] = Field(min_length=1, max_length=6)
    next_actions: list[Text] = Field(min_length=1, max_length=8)
    reversal_conditions: list[Text] = Field(min_length=1, max_length=8)
    external_validation_needed: list[Text] = Field(min_length=1, max_length=8)
    consumable_artifact_ids: list[Text] = Field(max_length=12)


def build_outcome(store, provider, problem_id, instruction=""):
    from symplex.agents.request_context import bounded_proposal_context
    system = current_artifact(store, problem_id, "complex_system")
    context = review_context(store, problem_id)

    def citation_policy(supplied):
        evidence = supplied["evidence"]
        content_ids = {r["id"] for r in evidence.get("artifacts", []) + evidence.get("executed_files", [])} | {system["id"]}
        inventory_ids = {r["id"] for r in evidence.get("file_inventory", [])}
        supplied["citation_policy"] = {
            "claim_content_ids": sorted(content_ids),
            "inventory_metadata_ids": sorted(inventory_ids),
            "consumable_ids": sorted(content_ids | inventory_ids),
            "rule": "Substantive claims may cite only claim_content_ids. For claims solely about recorded file availability or inventory metadata, use basis artifact_inventory and cite consumable_ids. Inventory metadata does not establish file contents, numerical results, successful execution or scientific validity. A file omitted from this bounded context is not evidence that it is missing from storage.",
        }

    supplied = bounded_proposal_context(store, provider, DecisionBrief, {
        "problem": store.get(problem_id)["data"], "system": system,
        "evidence": {"scope": context["scope"]}, "instruction": prompt("decision_editor"),
        "requested_task": instruction,
    }, {("evidence", "executed_files"): context["executed_files"],
        ("evidence", "artifacts"): context["artifacts"],
        ("evidence", "file_inventory"): context.get("file_inventory", [])},
        problem_id, primary_ids=(system["id"],), derive_metadata=citation_policy)
    context = supplied["evidence"]
    available = set(supplied["citation_policy"]["claim_content_ids"])
    result = provider.propose(
        DecisionBrief,
        supplied,
        problem_id,
    )
    result = DecisionBrief.parse(result)
    consumable = set(supplied["citation_policy"]["consumable_ids"])
    if not set(result["consumable_artifact_ids"]) <= consumable or any(
        not set(c["source_ids"]) <= (consumable if c["basis"] == "artifact_inventory" else available)
        for c in result["claims"]
    ):
        raise Invalid("Outcome references an artifact not provided to its author")
    ids = [a["alternative_id"] for a in result["alternatives"]]
    if len(ids) != len(set(ids)) or not set(ids) <= {
        a["id"] for a in system["data"]["decision"]["alternatives"]
    }:
        raise Invalid("Outcome references duplicate or unknown alternatives")
    return store.put(
        "decision_brief",
        {
            **result,
            "system_id": system["id"],
            "system_digest": system["digest"],
            "requested_task": instruction,
            "citation_policy": supplied["citation_policy"],
            "context_manifest_id": supplied["context_selection"]["manifest_id"],
            "status": "requires_review",
            "independently_validated": False,
            "scope": "Agent-authored interpretation with admitted artifact references; citation presence does not establish claim truth",
        },
        problem_id,
    )
