"""Full primary models fit actual serialized provider requests without duplicate memory."""

import copy
import json
import tempfile
import unittest
from types import SimpleNamespace

from symplex.agents.improvement import develop_candidate
from symplex.agents.outcomes import build_outcome
from symplex.agents.request_context import bounded_proposal_context, proposal_request_bytes
from symplex.agents.scientific_cycle import HypothesisReview, review_hypotheses
from symplex.core.contracts import Invalid, canonical
from symplex.evaluation.experiments import plan_experiment
from symplex.evidence.synthesis import synthesize_evidence
from symplex.infrastructure.providers import OpenAIProvider
from symplex.infrastructure.storage import Budget, Store
from symplex.modeling.complex_system import ComplexSystemSpec
from tests.test_complex_system import system_spec


class CapturedRequest(BaseException):
    pass


def large_valid_system():
    """Match the roughly 39 KB accepted live model using valid synthetic narrative fields."""
    value = system_spec()
    slots = []

    def visit(item):
        if isinstance(item, dict):
            for key, child in item.items():
                if isinstance(child, str) and len(child) > 30 and not key.endswith("_id"):
                    slots.append((item, key))
                else:
                    visit(child)
        elif isinstance(item, list):
            for index, child in enumerate(item):
                if isinstance(child, str) and len(child) > 30:
                    slots.append((item, index))
                else:
                    visit(child)

    visit(value)
    for parent, key in slots:
        extra = min(39000 - len(canonical(value)), 1600 - len(parent[key]))
        if extra <= 0:
            break
        parent[key] += " " + ("Synthetic explanatory context. " * 60)[:extra - 1]
    return ComplexSystemSpec.parse(value)


class RequestContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name)
        self.problem = self.store.put("workspace_problem", {"question": "Investigate a large valid scientific representation"})
        self.system_data = large_valid_system()
        self.system = self.store.put("complex_system", self.system_data, self.problem)
        for index in range(3):
            self.store.put("context", {"title": f"Evidence {index}", "content": "Fresh observations. " * 200,
                                      "format": "text", "basis": "observed_data"}, self.problem)
        self.store.put("problem_dna", {"decision": "Explicit decision details. " * 130}, self.problem)
        self.store.put("model_critique", {"issue": "Unresolved assumptions. " * 200}, self.problem)
        self.store.put("hypothesis_review", {"assessment": "Earlier critique findings. " * 180}, self.problem)
        self.requests = []

        def capture(**kwargs):
            self.requests.append(kwargs)
            raise CapturedRequest()

        client = SimpleNamespace(responses=SimpleNamespace(create=capture))
        self.provider = OpenAIProvider(self.store, Budget(self.store, {
            "usd": 50, "input_tokens": 1000000, "output_tokens": 200000}), client=client)

    def test_large_primary_models_fit_every_affected_actual_provider_request(self):
        self.assertGreaterEqual(len(canonical(self.system_data)), 38500)
        original = self.store.get(self.system)
        calls = ((review_hypotheses, "representation"), (plan_experiment, "system"),
                 (synthesize_evidence, "system"), (develop_candidate, "system"),
                 (build_outcome, "system"))
        for function, primary_key in calls:
            with self.subTest(role=function.__name__), self.assertRaises(CapturedRequest):
                function(self.store, self.provider, self.problem)
            request = self.requests[-1]
            # These kwargs have passed through the real provider's _call preparation.
            size = len(canonical(request).encode("utf-8")) + 2048
            self.assertLessEqual(size, 58000)
            wire_input = request["input"]
            context = json.loads(wire_input if isinstance(wire_input, str) else wire_input[0]["content"])
            primary = context[primary_key]
            if function is build_outcome:
                primary = primary["data"]
            self.assertEqual(primary, self.system_data)
            manifest = self.store.get(context["context_selection"]["manifest_id"])["data"]
            self.assertEqual(manifest["request_bytes"], size)
            self.assertIn(self.system, manifest["primary_artifact_ids"])
            self.assertNotIn(self.system, {r.get("id") for r in manifest["included"]})
        self.assertEqual(self.store.get(self.system), original)

    def test_omitted_whole_records_have_manifest_reasons_and_no_fake_truncated_contracts(self):
        primary = {"representation": {"complete": "primary"}}
        entries = [{"id": "primary", "kind": "complex_system", "digest": "p", "data": {"body": "duplicate"}},
                   {"id": "large", "kind": "context", "digest": "l", "data": {"body": "z" * 50000}},
                   {"id": "small", "kind": "context", "digest": "s", "data": {"body": "Short evidence"}}]
        before = copy.deepcopy(entries)
        result = bounded_proposal_context(self.store, self.provider, HypothesisReview, primary,
                                          {"evidence": entries}, self.problem,
                                          primary_ids=("primary",), max_request_bytes=10000)
        self.assertEqual(result["representation"], primary["representation"])
        self.assertEqual(result["evidence"], [entries[-1]])
        self.assertEqual(entries, before)
        manifest = self.store.get(result["context_selection"]["manifest_id"])["data"]
        omitted = {r["id"]: r["reason"] for r in manifest["omitted"]}
        self.assertIn("already supplied", omitted["primary"])
        self.assertEqual(omitted["large"], "request byte budget")
        self.assertNotIn("excerpt", json.dumps(result))
        self.assertLessEqual(proposal_request_bytes(HypothesisReview, result, self.provider), 10000)

    def test_irreducible_primary_and_schema_overflow_fail_without_truncation_or_model_call(self):
        before = len(self.store.list("role_context_manifest"))
        with self.assertRaisesRegex(Invalid, "Complete primary contract"):
            bounded_proposal_context(self.store, self.provider, HypothesisReview,
                                      {"representation": {"body": "x" * 96000}},
                                      {"evidence": []}, self.problem)
        self.assertEqual(len(self.store.list("role_context_manifest")), before)
        self.assertFalse(self.requests)

    def test_complete_primary_can_expand_without_filling_context_with_history(self):
        primary = {"representation": {"body": "x" * 65000}}
        result = bounded_proposal_context(self.store, self.provider, HypothesisReview,
            primary, {"evidence": [{"id": "history", "data": "old" * 1000}]}, self.problem)
        self.assertEqual(result["representation"], primary["representation"])
        self.assertEqual(result["evidence"], [])
        manifest = self.store.get(result["context_selection"]["manifest_id"])["data"]
        self.assertTrue(manifest["expanded_for_primary"])
        self.assertLessEqual(manifest["request_bytes"], 96000)
        self.assertLessEqual(manifest["max_request_bytes"] - manifest["primary_request_bytes"], 512)


if __name__ == "__main__":
    unittest.main()
