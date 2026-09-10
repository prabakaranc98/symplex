"""Evidence synthesis admission tests; fixtures are synthetic and no model is called."""

import copy
import json
import tempfile
import unittest

from pydantic import ValidationError

from symplex.core.contracts import Invalid
from symplex.evidence.synthesis import EvidenceSynthesis, synthesize_evidence
from symplex.infrastructure.storage import Store
from tests.test_complex_system import system_spec


def synthesis_result(source_id=None):
    claims = (
        []
        if source_id is None
        else [
            {
                "id": "observation",
                "statement": "The supplied trace reports a delayed response.",
                "kind": "source_assertion",
                "source_ids": [source_id],
                "premise_claim_ids": [],
                "hypothesis_links": [
                    {
                        "hypothesis_id": "delayed",
                        "relation": "supports",
                        "rationale": "A reported lag is compatible with memory, conditional on instrument error.",
                    }
                ],
                "measurement_or_proxy_limitations": ["Instrument delay is unmeasured"],
                "applicability_limits": [
                    "A single user-supplied observation is not validated"
                ],
            }
        ]
    )
    return {
        "summary": "Evidence is insufficient to establish a mechanism.",
        "claims": claims,
        "source_dependencies": [],
        "disagreements": [],
        "gaps": [
            {
                "id": "instrument",
                "hypothesis_ids": ["memoryless", "delayed"],
                "missing_information": "Independent instrument response",
                "discriminating_observation": "Measure lag with a calibrated independent instrument",
                "next_action": "Acquire independent instrument observations",
            }
        ],
        "hypothesis_assessments": [
            {
                "hypothesis_id": ident,
                "claim_ids": ["observation"] if claims else [],
                "interpretation": "Not established",
                "unresolved_limit": "Instrument confounding",
                "next_discriminating_test": "Independent response measurement",
            }
            for ident in ("memoryless", "delayed")
        ],
        "excluded_signals": [],
        "next_action": "Measure a discriminating response",
    }


class FakeProvider:
    def __init__(self, value, callback=None):
        self.value = value
        self.calls = []
        self.callback = callback

    def propose(self, contract, context, parent=None, role="heavy"):
        self.calls.append((contract, copy.deepcopy(context), parent, role))
        if self.callback:
            self.callback()
        return copy.deepcopy(self.value)


class SynthesisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name)
        self.problem = self.store.put(
            "workspace_problem", {"question": "Distinguish lag mechanisms"}
        )
        self.system = self.store.put(
            "complex_system",
            {
                **system_spec(),
                "status": "proposed",
                "execution_status": "not_executed",
            },
            self.problem,
        )

    def source(self, content="A user reports response lag"):
        return self.store.put(
            "context",
            {
                "content": content,
                "title": "Observation",
                "format": "text",
                "basis": "user_context",
            },
            self.problem,
        )

    def run_synthesis(self, value, callback=None):
        provider = FakeProvider(value, callback)
        ident = synthesize_evidence(self.store, provider, self.problem)
        return self.store.get(ident), provider

    def test_synthesis_preserves_claim_types_current_hypotheses_and_host_lineage(self):
        source = self.source()
        result, provider = self.run_synthesis(synthesis_result(source))
        self.assertEqual(result["kind"], "evidence_synthesis")
        self.assertEqual(result["parent"], self.problem)
        self.assertEqual(result["data"]["claims"][0]["kind"], "source_assertion")
        self.assertEqual(result["data"]["system_id"], self.system)
        self.assertEqual(
            result["data"]["system_digest"], self.store.get(self.system)["digest"]
        )
        self.assertFalse(result["data"]["empirically_validated"])
        self.assertEqual(provider.calls[0][3], "heavy")
        self.assertEqual(provider.calls[0][0].output_token_budget, 4000)
        manifest = {r["id"]: r for r in result["data"]["input_manifest"]}
        self.assertEqual(manifest[source]["digest"], self.store.get(source)["digest"])
        self.assertEqual(
            set(h["hypothesis_id"] for h in result["data"]["hypothesis_assessments"]),
            {"memoryless", "delayed"},
        )

    def test_unchanged_synthesis_reuses_record_but_new_system_changes_identity(self):
        source = self.source()
        first, provider = self.run_synthesis(synthesis_result(source))
        self.assertEqual(
            synthesize_evidence(self.store, provider, self.problem), first["id"]
        )
        self.assertEqual(len(provider.calls), 1)
        self.store.put("complex_system", system_spec(), self.problem)
        second = synthesize_evidence(self.store, provider, self.problem)
        self.assertNotEqual(second, first["id"])
        self.assertEqual(len(provider.calls), 2)

    def test_missing_evidence_produces_gaps_without_invented_claims(self):
        result, provider = self.run_synthesis(synthesis_result())
        self.assertEqual(result["data"]["claims"], [])
        self.assertTrue(result["data"]["gaps"])
        self.assertEqual(provider.calls[0][1]["sources"], [])

    def test_unknown_stale_foreign_or_protected_source_references_rejected(self):
        stale = self.source("Obsolete observation")
        self.store.invalidate(stale)
        other = self.store.put("workspace_problem", {"question": "Other system"})
        foreign = self.store.put(
            "context",
            {"content": "Unrelated evidence", "title": "Other", "format": "text"},
            other,
        )
        protected = self.store.put(
            "dataset", {"secret": "held-out labels"}, self.problem
        )
        for source in ("missing_source", stale, foreign, protected):
            with self.subTest(source=source), self.assertRaises((Invalid, ValueError)):
                self.run_synthesis(synthesis_result(source))
        self.assertEqual(len(self.store.list("evidence_synthesis")), 0)
        self.assertEqual(len(self.store.list("synthesis_rejection")), 4)

    def test_metadata_and_uninspected_binary_cannot_be_scientific_support(self):
        metadata = self.store.put(
            "evidence_note",
            {"mode": "literature_metadata", "text": "Title and DOI only"},
            self.problem,
        )
        binary = self.store.put(
            "binary_context",
            {"title": "Image pointer", "blob_id": "uninspected"},
            self.problem,
        )
        for source in (metadata, binary):
            with self.subTest(source=source), self.assertRaises(Invalid):
                self.run_synthesis(synthesis_result(source))
            uncertain = synthesis_result(source)
            uncertain["claims"][0]["hypothesis_links"][0]["relation"] = "unclear"
            result, provider = self.run_synthesis(uncertain)
            self.assertFalse(result["data"]["empirically_validated"])
            source_entry = next(
                s for s in provider.calls[0][1]["sources"] if s["id"] == source
            )
            self.assertIn(
                source_entry["material"], ("literature_metadata", "binary_reference")
            )
            # Keep each proposal admission independent of the idempotent cache.
            self.store.invalidate(result["id"])

    def test_assumptions_and_user_judgments_are_not_empirical_support(self):
        source = self.source()
        for kind in ("assumption", "user_judgment"):
            value = synthesis_result(source)
            value["claims"][0]["kind"] = kind
            with self.subTest(kind=kind), self.assertRaises(ValidationError):
                self.run_synthesis(value)
        value = synthesis_result(source)
        value["claims"][0]["kind"] = "user_judgment"
        value["claims"][0]["hypothesis_links"][0]["relation"] = "unclear"
        result, _ = self.run_synthesis(value)
        self.assertEqual(result["data"]["claims"][0]["kind"], "user_judgment")

    def test_current_hypotheses_and_claim_references_are_exact(self):
        source = self.source()
        for mutation in (
            lambda value: value["hypothesis_assessments"].pop(),
            lambda value: value["hypothesis_assessments"][0].update(
                hypothesis_id="invented"
            ),
            lambda value: value["claims"][0]["hypothesis_links"][0].update(
                hypothesis_id="candidate_solution"
            ),
            lambda value: value["hypothesis_assessments"][0].update(
                claim_ids=["unseen_claim"]
            ),
        ):
            value = synthesis_result(source)
            mutation(value)
            with self.assertRaises((Invalid, ValueError)):
                self.run_synthesis(value)

    def test_circular_inferences_and_reliability_scores_rejected(self):
        value = synthesis_result()
        value["reliability_score"] = 0.99
        with self.assertRaises(ValidationError):
            EvidenceSynthesis.parse(value)
        source = self.source()
        value = synthesis_result(source)
        first = value["claims"][0]
        first.update(kind="inference", premise_claim_ids=["other"])
        other = copy.deepcopy(first)
        other.update(id="other", premise_claim_ids=["observation"])
        value["claims"].append(other)
        with self.assertRaises(ValidationError):
            EvidenceSynthesis.parse(value)

    def test_exact_duplicates_and_disagreement_are_preserved(self):
        a, b = self.source(), self.source()
        value = synthesis_result(a)
        other = copy.deepcopy(value["claims"][0])
        other.update(
            id="contrary",
            source_ids=[b],
            statement="A conflicting interpretation is reported",
        )
        other["hypothesis_links"][0]["relation"] = "contradicts"
        value["claims"].append(other)
        value["source_dependencies"] = [
            {
                "source_ids": [a, b],
                "relation": "duplicate",
                "explanation": "Identical stored content",
                "consequence_for_synthesis": "Do not count as independent confirmation",
            }
        ]
        value["disagreements"] = [
            {
                "claim_ids": ["observation", "contrary"],
                "hypothesis_ids": ["delayed"],
                "explanation": "Conflicting interpretations of the same account",
                "resolving_observation": "Inspect independently measured response",
            }
        ]
        result, provider = self.run_synthesis(value)
        self.assertEqual(result["data"]["exact_duplicate_source_groups"], [[a, b]])
        self.assertEqual(
            provider.calls[0][1]["exact_duplicate_source_groups"], [[a, b]]
        )
        self.assertEqual(
            result["data"]["disagreements"][0]["claim_ids"], ["observation", "contrary"]
        )

    def test_source_invalidated_during_reasoning_is_rejected(self):
        source = self.source()
        with self.assertRaises(Invalid):
            self.run_synthesis(
                synthesis_result(source), lambda: self.store.invalidate(source)
            )
        self.assertFalse(self.store.list("evidence_synthesis"))
        self.assertEqual(len(self.store.list("synthesis_rejection")), 1)

    def test_schema_is_closed_with_no_confidence_or_reliability_fields(self):
        schema = EvidenceSynthesis.json_schema()
        self.assertNotIn("confidence", json.dumps(schema))
        for definition in [schema, *schema["$defs"].values()]:
            if definition.get("type") == "object":
                self.assertFalse(definition["additionalProperties"])
                self.assertEqual(
                    set(definition["required"]), set(definition["properties"])
                )


if __name__ == "__main__":
    unittest.main()
