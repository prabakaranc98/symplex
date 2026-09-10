"""Reviewer context and consumable links; fixtures never execute stored source."""

import copy
import tempfile
import unittest

from symplex.agents.context import review_context
from symplex.agents.outcomes import build_outcome
from symplex.connectors.compute import save_blob
from symplex.core.contracts import Invalid
from symplex.infrastructure.storage import Store
from tests.test_complex_system import system_spec
from tests.test_improvement import brief_spec


class FakeProvider:
    def __init__(self, result):
        self.result = result
        self.context = None

    def propose(self, contract, context, parent=None, role="heavy"):
        self.context = copy.deepcopy(context)
        return self.result


class ReviewInventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name)
        self.problem = self.store.put(
            "workspace_problem", {"question": "Compare synthetic mechanisms"}
        )
        self.system = self.store.put("complex_system", system_spec(), self.problem)
        self.run = self.store.put(
            "compute_run", {"summary": "Synthetic fixture", "calls": []}, self.problem
        )

    def blob(self, name, content, *, problem=None, run=None, basis="generated"):
        return save_blob(
            self.store,
            content,
            name,
            problem or self.problem,
            basis,
            run or self.run,
        )

    def package(self, ids):
        return self.store.put(
            "compute_package",
            {"run_id": self.run, "file_ids": ids, "summary": "Synthetic fixture"},
            self.problem,
        )

    def test_source_and_diagnostics_precede_large_csv_with_complete_inventory(self):
        csv = self.blob("raw.csv", b"time,value\n" + b"1,1\n" * 10000)
        result = self.blob("results.json", b'{"padding":"' + b"x" * 12000 + b'"}')
        source = self.blob("symplex_model.py", b"# source fixture\n" * 1000)
        diagnostics = self.blob("symplex_experiment.json", b'{"basis":"synthetic"}')
        image = self.blob("plot.png", b"image fixture; not a scientific visualization")
        ids = [csv, result, image, diagnostics, source]
        self.package(ids)

        context = review_context(self.store, self.problem, max_chars=16000)
        excerpts = context["executed_files"]
        self.assertEqual([r["id"] for r in excerpts[:2]], [source, diagnostics])
        self.assertEqual(excerpts[0]["content"], ("# source fixture\n" * 1000)[:7000])
        self.assertTrue(excerpts[0]["truncated"])
        self.assertFalse(excerpts[1]["truncated"])
        self.assertNotIn(csv, {r["id"] for r in excerpts})
        inventory = {r["id"]: r for r in context["file_inventory"]}
        self.assertEqual(set(inventory), set(ids))
        self.assertEqual(inventory[image]["run_id"], self.run)
        self.assertEqual(inventory[source]["digest"], self.store.get(source)["digest"])
        self.assertIn("Available", inventory[image]["content_status"])

    def test_protocol_projection_keeps_comparison_without_mutating_full_contract(self):
        protocol_data = {
            "title": "Frozen synthetic comparison",
            "result_contract": {"padding": "x" * 24000},
            "acceptance": "Independent validation is unavailable",
        }
        protocol = self.store.put("experiment_protocol", protocol_data, self.problem)
        comparison = self.store.put(
            "experiment_comparison",
            {"status": "conditional", "independently_validated": False},
            self.problem,
        )
        context = review_context(self.store, self.problem)
        artifacts = {r["id"]: r for r in context["artifacts"]}
        self.assertIn(comparison, artifacts)
        self.assertNotIn("result_contract", artifacts[protocol]["data"])
        self.assertEqual(
            artifacts[protocol]["data"]["full_output_contract_artifact_id"], protocol
        )
        self.assertEqual(self.store.get(protocol)["data"], protocol_data)

    def test_uninspected_image_can_be_consumable_without_becoming_claim_evidence(self):
        image = self.blob("plot.png", b"uninspected plot fixture")
        self.package([image])
        brief = brief_spec(self.system, self.system)
        brief["consumable_artifact_ids"] = [image]
        provider = FakeProvider(brief)
        ident = build_outcome(self.store, provider, self.problem)
        self.assertEqual(
            self.store.get(ident)["data"]["consumable_artifact_ids"], [image]
        )
        evidence = provider.context["evidence"]
        self.assertIn(image, {r["id"] for r in evidence["file_inventory"]})
        self.assertNotIn(image, {r["id"] for r in evidence["executed_files"]})

        brief["claims"][0]["source_ids"] = [image]
        with self.assertRaisesRegex(Invalid, "not provided"):
            build_outcome(self.store, FakeProvider(brief), self.problem)

    def test_inventory_excludes_foreign_stale_and_unbound_files(self):
        other = self.store.put("workspace_problem", {"question": "Other system"})
        foreign = self.blob("foreign.png", b"foreign", problem=other)
        wrong_run = self.blob("wrong.png", b"wrong run", run="compute_run_unrelated")
        user_file = self.blob("user.png", b"user input", basis="user_context")
        stale = self.blob("stale.png", b"stale")
        self.store.invalidate(stale)
        accepted = self.blob("accepted.png", b"valid generated fixture")
        self.package([foreign, wrong_run, user_file, stale, accepted])
        context = review_context(self.store, self.problem)
        self.assertEqual([r["id"] for r in context["file_inventory"]], [accepted])
        for ident in (foreign, wrong_run, user_file, stale):
            brief = brief_spec(self.system, self.system)
            brief["consumable_artifact_ids"] = [ident]
            with self.subTest(ident=ident), self.assertRaises(Invalid):
                build_outcome(self.store, FakeProvider(brief), self.problem)

    def test_inventory_basis_allows_availability_claim_without_granting_scientific_content(self):
        image = self.blob("plot.png", b"uninspected plot fixture")
        self.package([image])
        brief = brief_spec(self.system, self.system)
        brief["claims"] = [{"statement": "A plot.png output is available in the recorded package.",
            "basis": "artifact_inventory", "source_ids": [image],
            "limitation": "Inventory metadata only; image contents and scientific validity were not inspected."}]
        brief["consumable_artifact_ids"] = [image]
        provider = FakeProvider(brief)
        ident = build_outcome(self.store, provider, self.problem)
        policy = provider.context["citation_policy"]
        self.assertIn(image, policy["inventory_metadata_ids"])
        self.assertIn(image, policy["consumable_ids"])
        self.assertNotIn(image, policy["claim_content_ids"])
        self.assertEqual(self.store.get(ident)["data"]["claims"], brief["claims"])
        self.assertEqual(self.store.get(ident)["data"]["citation_policy"], policy)
        for basis in ("observed", "conditional_model", "synthetic_test", "assumption", "judgment"):
            with self.subTest(basis=basis):
                changed = copy.deepcopy(brief)
                changed["claims"][0]["basis"] = basis
                with self.assertRaisesRegex(Invalid, "not provided"):
                    build_outcome(self.store, FakeProvider(changed), self.problem)
        changed = copy.deepcopy(brief)
        changed["claims"][0]["source_ids"] = ["foreign_or_missing_file"]
        with self.assertRaisesRegex(Invalid, "not provided"):
            build_outcome(self.store, FakeProvider(changed), self.problem)

    def test_citation_policy_is_derived_only_after_actual_context_admission(self):
        from unittest.mock import patch
        from symplex.agents.request_context import bounded_proposal_context
        image = self.blob("plot.png", b"uninspected plot fixture")
        self.package([image])
        brief = brief_spec(self.system, self.system)
        provider = FakeProvider(brief)
        def omit_inventory(store, provider, contract, primary, collections, problem_id, **kwargs):
            collections = {key: value for key, value in collections.items()}
            collections[("evidence", "file_inventory")] = []
            return bounded_proposal_context(store, provider, contract, primary, collections, problem_id, **kwargs)
        with patch("symplex.agents.request_context.bounded_proposal_context", side_effect=omit_inventory):
            build_outcome(self.store, provider, self.problem)
        self.assertNotIn(image, provider.context["citation_policy"]["consumable_ids"])
        self.assertEqual(provider.context["citation_policy"]["inventory_metadata_ids"], [])


if __name__ == "__main__":
    unittest.main()
