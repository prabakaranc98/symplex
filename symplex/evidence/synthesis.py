"""Agent-authored evidence synthesis with host-checked scope and claim provenance.

Reference validation prevents nonexistent evidence from entering the record. It does
not establish that a source is correct, that an inference follows, or that a model
hypothesis is empirically validated.
"""

from typing import ClassVar, Literal

from pydantic import Field, model_validator

from symplex.agents.context import KINDS, working_context
from symplex.agents.prompts import prompt
from symplex.core.contracts import Invalid, digest
from symplex.modeling.complex_system import (
    ClosedContract,
    ComplexSystemSpec,
    Identifier,
    Text,
)


class HypothesisLink(ClosedContract):
    hypothesis_id: Identifier
    relation: Literal["supports", "contradicts", "unclear"]
    rationale: Text


class EvidenceClaim(ClosedContract):
    id: Identifier
    statement: Text
    kind: Literal["source_assertion", "inference", "assumption", "user_judgment"]
    source_ids: list[Text] = Field(max_length=8)
    premise_claim_ids: list[Identifier] = Field(max_length=8)
    hypothesis_links: list[HypothesisLink] = Field(max_length=6)
    measurement_or_proxy_limitations: list[Text] = Field(max_length=6)
    applicability_limits: list[Text] = Field(max_length=6)


class SourceDependence(ClosedContract):
    source_ids: list[Text] = Field(min_length=2, max_length=8)
    relation: Literal[
        "duplicate", "shared_origin", "shared_measurement", "dependence_suspected"
    ]
    explanation: Text
    consequence_for_synthesis: Text


class Disagreement(ClosedContract):
    claim_ids: list[Identifier] = Field(min_length=2, max_length=8)
    hypothesis_ids: list[Identifier] = Field(min_length=1, max_length=6)
    explanation: Text
    resolving_observation: Text


class DiscriminatingGap(ClosedContract):
    id: Identifier
    hypothesis_ids: list[Identifier] = Field(min_length=1, max_length=6)
    missing_information: Text
    discriminating_observation: Text
    next_action: Text


class HypothesisSynthesis(ClosedContract):
    hypothesis_id: Identifier
    claim_ids: list[Identifier] = Field(max_length=12)
    interpretation: Text
    unresolved_limit: Text
    next_discriminating_test: Text


class ExcludedSignal(ClosedContract):
    source_id: Text
    reason: Text
    reconsider_if: Text


class EvidenceSynthesis(ClosedContract):
    output_token_budget: ClassVar[int] = 4000
    summary: Text
    claims: list[EvidenceClaim] = Field(max_length=12)
    source_dependencies: list[SourceDependence] = Field(max_length=6)
    disagreements: list[Disagreement] = Field(max_length=6)
    gaps: list[DiscriminatingGap] = Field(max_length=8)
    hypothesis_assessments: list[HypothesisSynthesis] = Field(
        min_length=1, max_length=6
    )
    excluded_signals: list[ExcludedSignal] = Field(max_length=8)
    next_action: Text

    @model_validator(mode="after")
    def local_references(self):
        claims = {claim.id: claim for claim in self.claims}
        if len(claims) != len(self.claims):
            raise ValueError("Claim IDs must be unique")
        if len({gap.id for gap in self.gaps}) != len(self.gaps):
            raise ValueError("Gap IDs must be unique")
        for claim in self.claims:
            _references(claim.premise_claim_ids, claims, "premise claim")
            if claim.id in claim.premise_claim_ids:
                raise ValueError("A claim cannot justify itself")
            if (
                claim.kind in ("source_assertion", "user_judgment")
                and not claim.source_ids
            ):
                raise ValueError("Source assertions and user judgments require sources")
            if claim.kind == "source_assertion" and claim.premise_claim_ids:
                raise ValueError(
                    "Derived claims with premises must be labeled inference"
                )
            if claim.kind == "inference" and not (
                claim.source_ids or claim.premise_claim_ids
            ):
                raise ValueError("An inference requires sources or explicit premises")
            if claim.kind in ("assumption", "user_judgment") and any(
                link.relation != "unclear" for link in claim.hypothesis_links
            ):
                raise ValueError(
                    "Assumptions and user judgments cannot establish scientific support"
                )
            if len({link.hypothesis_id for link in claim.hypothesis_links}) != len(
                claim.hypothesis_links
            ):
                raise ValueError("A claim must link each hypothesis at most once")
        for assessment in self.hypothesis_assessments:
            _references(assessment.claim_ids, claims, "assessment claim")
        for disagreement in self.disagreements:
            _references(disagreement.claim_ids, claims, "disagreement claim")
        visiting, visited = set(), set()

        def visit(ident):
            if ident in visiting:
                raise ValueError("Circular claim justification")
            if ident in visited:
                return
            visiting.add(ident)
            for premise in claims[ident].premise_claim_ids:
                visit(premise)
            visiting.remove(ident)
            visited.add(ident)

        for ident in claims:
            visit(ident)
        return self

    @classmethod
    def parse(cls, value):
        return cls.model_validate(value).model_dump()

    @classmethod
    def json_schema(cls):
        return cls.model_json_schema()


def _references(references, known, label):
    if len(references) != len(set(references)):
        raise ValueError("Duplicate " + label + " reference")
    if not set(references).issubset(known):
        raise ValueError("Unknown " + label + " reference")


SOURCE_KINDS = {
    "context",
    "steering",
    "evidence_note",
    "binary_context",
    "compute_package",
    "scenario_report",
    "experiment_comparison",
}
UNOBSERVED_MATERIALS = {
    "literature_metadata",
    "binary_reference",
    "user_judgment",
    "declared_assumption",
}


def _material(record):
    kind, data = record["kind"], record["data"]
    if kind == "evidence_note":
        return (
            "literature_metadata"
            if data.get("mode") == "literature_metadata"
            else "research_summary"
        )
    if kind == "binary_context":
        return "binary_reference"
    if kind == "steering":
        return "user_judgment"
    if kind == "context":
        if data.get("basis") == "assumption":
            return "declared_assumption"
        return (
            "user_supplied_data"
            if data.get("format") in ("csv", "json", "geojson")
            else "user_supplied_text"
        )
    return "generated_result_summary"


def _validate_synthesis(value, sources, hypothesis_ids):
    parsed = EvidenceSynthesis.model_validate(value)
    actual = [a.hypothesis_id for a in parsed.hypothesis_assessments]
    if len(actual) != len(hypothesis_ids) or set(actual) != hypothesis_ids:
        raise Invalid(
            "Synthesis must assess each current system hypothesis exactly once"
        )
    claims = {c.id: c for c in parsed.claims}
    for claim in parsed.claims:
        _references(claim.source_ids, sources, "supplied source")
        _references(
            [link.hypothesis_id for link in claim.hypothesis_links],
            hypothesis_ids,
            "current hypothesis",
        )
        if claim.kind == "user_judgment" and any(
            sources[ident]["kind"] not in ("context", "steering")
            for ident in claim.source_ids
        ):
            raise Invalid("User judgments require user-supplied sources")
    for dependence in parsed.source_dependencies:
        _references(dependence.source_ids, sources, "dependent source")
    for excluded in parsed.excluded_signals:
        _references([excluded.source_id], sources, "excluded source")
    for item in [*parsed.gaps, *parsed.disagreements]:
        _references(item.hypothesis_ids, hypothesis_ids, "current hypothesis")

    def has_substantive_source(claim):
        if claim.kind in ("assumption", "user_judgment"):
            return False
        return any(
            sources[i]["material"] not in UNOBSERVED_MATERIALS for i in claim.source_ids
        ) or any(has_substantive_source(claims[i]) for i in claim.premise_claim_ids)

    for claim in parsed.claims:
        if any(
            link.relation != "unclear" for link in claim.hypothesis_links
        ) and not has_substantive_source(claim):
            raise Invalid(
                "Metadata, uninspected binary references and assumptions cannot alone support or contradict a scientific hypothesis"
            )
    return parsed.model_dump()


def synthesize_evidence(store, provider, problem_id, instruction=""):
    from symplex.agents.request_context import bounded_proposal_context
    problem = store.get(problem_id)
    if problem["kind"] != "workspace_problem" or problem["stale"]:
        raise Invalid("Select a current problem")
    systems = [
        r
        for r in store.list("complex_system")
        if r["parent"] == problem_id and not r["stale"]
    ]
    if not systems:
        raise Invalid(
            "A current complex-system representation is required for evidence synthesis"
        )
    system = systems[-1]
    spec = ComplexSystemSpec.model_validate(
        {k: v for k, v in system["data"].items() if k in ComplexSystemSpec.model_fields}
    )
    context = working_context(
        store,
        problem_id,
        max_chars=18000,
        # The system is supplied explicitly below. Plans and critiques should not
        # crowd actual evidence out of the synthesist's bounded working memory.
        excluded_kinds=set(KINDS) - SOURCE_KINDS,
    )
    sources = {}
    for entry in context:
        if entry["kind"] not in SOURCE_KINDS:
            continue
        current = store.get(entry["id"])
        if (
            current["parent"] != problem_id
            or current["stale"]
            or current["digest"] != entry["digest"]
        ):
            raise Invalid("Synthesis source is stale, changed or outside this problem")
        sources[entry["id"]] = {**entry, "material": _material(current)}

    def source_groups(value):
        groups = {}
        for source in value["sources"]:
            groups.setdefault(source["digest"], []).append(source["id"])
        value["exact_duplicate_source_groups"] = [ids for ids in groups.values() if len(ids) > 1]

    supplied = bounded_proposal_context(store, provider, EvidenceSynthesis, {
        "problem": problem["data"], "system_id": system["id"], "system_digest": system["digest"],
        "system": spec.model_dump(), "working_context": [],
        "instruction": prompt("evidence_synthesist"), "requested_task": instruction,
    }, {"sources": list(sources.values())}, problem_id, primary_ids=(system["id"],), derive_metadata=source_groups)
    sources = {entry["id"]: entry for entry in supplied["sources"]}
    context = list(sources.values())
    input_manifest = [
        {"id": problem["id"], "kind": problem["kind"], "digest": problem["digest"]},
        {"id": system["id"], "kind": system["kind"], "digest": system["digest"]},
        *[{k: r[k] for k in ("id", "kind", "digest")} for r in context],
    ]
    input_digest = digest(
        {
            "version": "evidence-synthesis-v1",
            "inputs": input_manifest,
            "prompt_digest": digest(prompt("evidence_synthesist")),
            "requested_task": instruction,
        }
    )
    for prior in store.list("evidence_synthesis"):
        if (
            prior["parent"] == problem_id
            and not prior["stale"]
            and prior["data"].get("input_digest") == input_digest
        ):
            return prior["id"]
    duplicate_groups = {}
    for source in sources.values():
        duplicate_groups.setdefault(source["digest"], []).append(source["id"])
    result = provider.propose(
        EvidenceSynthesis,
        supplied,
        problem_id,
        role="heavy",
    )
    try:
        result = _validate_synthesis(result, sources, {h.id for h in spec.hypotheses})
        for source in [system, *sources.values()]:
            current = store.get(source["id"])
            if current["stale"] or current["digest"] != source["digest"]:
                raise Invalid("Synthesis inputs changed while the model was reasoning")
    except (ValueError, Invalid) as exc:
        store.put(
            "synthesis_rejection",
            {"proposal": result, "error": str(exc), "input_digest": input_digest},
            problem_id,
        )
        raise
    return store.put(
        "evidence_synthesis",
        {
            **result,
            "system_id": system["id"],
            "system_digest": system["digest"],
            "input_manifest": input_manifest,
            "input_digest": input_digest,
            "requested_task": instruction,
            "source_manifest": [
                {k: s[k] for k in ("id", "kind", "digest", "material")}
                for s in sources.values()
            ],
            "exact_duplicate_source_groups": [
                ids for ids in duplicate_groups.values() if len(ids) > 1
            ],
            "status": "agent_synthesis",
            "empirically_validated": False,
            "scope": "Host-checked source and hypothesis references; source assertions, entailment, dependence and applicability still require verification. No numerical reliability score is inferred.",
        },
        problem_id,
    )
