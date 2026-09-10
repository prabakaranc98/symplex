"""Mandatory assessment and cache validity, using local synthetic artifacts only."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from symplex.connectors.compute import collect_outputs
from symplex.core.contracts import Invalid
from symplex.evaluation.execution import VERSION, assess_package
from tests import test_experiments as fixtures


class ExecutionAssessmentTests(unittest.TestCase):
    def setUp(self):
        # Reuse contract-fixture setup; none of its test methods or remote code runs.
        self.fixture = fixtures.ExperimentTests(
            "test_protocol_is_frozen_with_system_digest_and_output_contract"
        )
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.store = self.fixture.store
        self.problem = self.fixture.problem_id

    def assess(self, package):
        return self.store.get(assess_package(self.store, self.problem, package))

    def test_valid_assessment_links_host_checks_and_remains_idempotent(self):
        package, run, _ = self.fixture.package()
        record = self.assess(package)
        data = record["data"]
        self.assertEqual(data["status"], "checked")
        self.assertTrue(data["numerical_verification"]["all_passed"])
        self.assertIsNotNone(data["comparison_id"])
        self.assertIsNotNone(data["numerical_verification_id"])
        self.assertEqual(data["run_id"], run)
        self.assertEqual(data["evaluator_version"], VERSION)
        self.assertFalse(data["independently_validated"])
        self.assertEqual(self.assess(package), record)
        self.assertEqual(len(self.store.list("execution_assessment")), 1)

    def test_numerical_failure_never_becomes_checked_merely_because_link_exists(self):
        package, _, _ = self.fixture.package(
            csv_text="alternative,quality,cost\nbaseline,5,10\nperturbed,-7,9\n"
        )
        data = self.assess(package)["data"]
        self.assertEqual(data["status"], "failed")
        self.assertIsNotNone(data["comparison_id"])
        self.assertIsNotNone(data["numerical_verification_id"])
        self.assertFalse(data["numerical_verification"]["all_passed"])
        self.assertIn("unchanged protocol", data["next_requirement"])

    def test_missing_or_legacy_protocol_is_unavailable_not_success(self):
        package, _, _ = self.fixture.package(manifest=[])
        with patch("symplex.evaluation.experiments.compare_computation") as compare:
            data = self.assess(package)["data"]
        compare.assert_not_called()
        self.assertEqual(data["status"], "unavailable")
        self.assertIsNone(data["comparison_id"])

        legacy = self.store.get(self.fixture.protocol_id)["data"]
        legacy.pop("numeric_checks")
        self.fixture.protocol_id = self.store.put(
            "experiment_protocol", legacy, self.problem
        )
        package, _, _ = self.fixture.package()
        data = self.assess(package)["data"]
        self.assertIsNotNone(data["numerical_verification_id"])
        self.assertEqual(data["status"], "unavailable")
        self.assertFalse(data["numerical_verification"]["all_passed"])

    def test_malformed_output_and_incomplete_run_are_recorded_failures(self):
        for options in (
            {"filename": "incorrect.json"},
            {"csv_text": "alternative,quality,cost\nbaseline,NaN,1\n"},
            {"calls": []},
        ):
            with self.subTest(options=options):
                package, _, _ = self.fixture.package(**options)
                data = self.assess(package)["data"]
                self.assertEqual(data["status"], "failed")
                self.assertTrue(data["error"])
                self.assertFalse(data["independently_validated"])

    def test_protocol_without_bound_problem_or_system_cannot_be_checked(self):
        protocol = self.store.get(self.fixture.protocol_id)
        manifest = [{k: protocol[k] for k in ("id", "kind", "digest")}]
        package, _, _ = self.fixture.package(manifest=manifest)
        data = self.assess(package)["data"]
        self.assertEqual(data["status"], "failed")
        self.assertIn("current problem digest", data["error"])
        self.assertIsNone(data["comparison_id"])
        self.assertFalse(data["numerical_verification"]["all_passed"])

    def test_cached_check_does_not_authorize_changed_bytes_or_stale_run(self):
        package, _, _ = self.fixture.package()
        checked = self.assess(package)
        csv_id = self.store.get(package)["data"]["file_ids"][1]
        blob = self.store.get(csv_id)
        (self.store.root / "blobs" / blob["data"]["sha256"]).write_bytes(b"corrupt")
        failed = self.assess(package)
        self.assertNotEqual(failed["id"], checked["id"])
        self.assertEqual(failed["data"]["status"], "failed")
        self.assertEqual(self.store.get(checked["id"]), checked)

        package, run, _ = self.fixture.package(
            csv_text="alternative,quality,cost\nbaseline,5,8\nperturbed,7,7\n"
        )
        self.assess(package)
        self.store.invalidate(run)
        with self.assertRaises(Invalid):
            self.assess(package)

    def test_output_collection_assesses_fresh_and_existing_packages_by_default(self):
        run = self.store.put(
            "compute_run",
            {
                "calls": [{"status": "completed"}],
                "summary": "No frozen protocol or outputs",
            },
            self.problem,
        )
        provider = SimpleNamespace()  # No client/network surface is available.
        package = collect_outputs(self.store, provider, self.problem, run)
        assessments = self.store.list("execution_assessment")
        self.assertEqual(len(assessments), 1)
        self.assertEqual(assessments[0]["data"]["package_id"], package)
        self.assertEqual(assessments[0]["data"]["status"], "unavailable")
        self.assertEqual(
            collect_outputs(self.store, provider, self.problem, run), package
        )
        self.assertEqual(len(self.store.list("execution_assessment")), 1)

        valid, valid_run, _ = self.fixture.package()
        self.assertEqual(
            collect_outputs(self.store, provider, self.problem, valid_run), valid
        )
        latest = self.store.list("execution_assessment")[-1]["data"]
        self.assertEqual(latest["package_id"], valid)
        self.assertEqual(latest["status"], "checked")


if __name__ == "__main__":
    unittest.main()
