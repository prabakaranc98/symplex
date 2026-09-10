"""Archive selection uses frozen host numerical diagnostics, never maker scores."""

import copy
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from symplex.agents.search_policy import build_search_archive
from symplex.agents.solver import NextStep
from symplex.agents.tools import evolve_model
from symplex.core.contracts import Invalid
from symplex.connectors.compute import save_blob
from symplex.evaluation.experiments import (
    EVALUATOR_VERSION,
    ExperimentProtocol,
    compare_computation,
)
from symplex.infrastructure.storage import Store

try:
    from tests.test_complex_system import system_spec
except ModuleNotFoundError:
    from test_complex_system import system_spec


class SearchPolicyTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.store = Store(folder.name)
        self.problem_id = self.store.put(
            "workspace_problem", {"question": "Inspect hypothetical model candidates."}
        )
        self.system_id = self.store.put(
            "complex_system", system_spec(), self.problem_id
        )
        self.protocol_id = self.freeze("First frozen comparison")

    def freeze(self, controls):
        checks = []
        for ident, operation in (
            ("finite_values", "finite"),
            ("bounded_values", "bounds"),
        ):
            checks.append(
                {
                    "id": ident,
                    "operation": operation,
                    "csv_filename": "values.csv",
                    "columns": ["value"],
                    "group_columns": [],
                    "time_column": None,
                    "row_filters": [],
                    "lower": 0.0 if operation == "bounds" else None,
                    "upper": 10.0 if operation == "bounds" else None,
                    "reference_value": None,
                    "tolerance": 0.0,
                    "direction": None,
                    "units": "declared units",
                    "rationale": "Synthetic software diagnostic.",
                }
            )
        protocol = ExperimentProtocol.parse(
            {
                "execution_readiness": "ready",
                "blocking_reasons": [],
                "experiment_id": "lag_test",
                "baseline_id": "baseline",
                "candidate_ids": ["perturbed"],
                "metrics": [
                    {
                        "id": "quality",
                        "name": "Maker score",
                        "unit": "declared score",
                        "direction": "maximize",
                        "minimum_improvement": 1.0,
                        "maximum_degradation": 0.0,
                    }
                ],
                "required_check_ids": ["instrument_check"],
                "numeric_checks": checks,
                "comparison_controls": controls,
                "uncertainty_plan": "Unknown scientific uncertainty.",
                "stopping_rule": "One frozen software test.",
                "evidence_gaps": ["Synthetic fixtures only."],
            }
        )
        return self.store.put(
            "experiment_protocol",
            {
                **protocol,
                "system_id": self.system_id,
                "system_digest": self.store.get(self.system_id)["digest"],
                "status": "frozen_for_comparison",
            },
            self.problem_id,
        )

    def execute(
        self,
        value=5.0,
        *,
        basis="synthetic",
        protocol_id=None,
        predecessor=None,
        evaluate=True,
        maker_score=100.0,
        predecessor_in_manifest=True,
        source_code=False,
    ):
        protocol_id = protocol_id or self.protocol_id
        manifest = [
            {"id": ident, "digest": self.store.get(ident)["digest"]}
            for ident in (self.problem_id, self.system_id, protocol_id)
        ]
        if predecessor and predecessor_in_manifest:
            manifest.append(
                {"id": predecessor, "digest": self.store.get(predecessor)["digest"]}
            )
        run_id = self.store.put(
            "compute_run",
            {
                "calls": [{"status": "completed"}],
                "input_manifest": manifest,
                "predecessor_package_id": predecessor,
            },
            self.problem_id,
        )
        csv_id = save_blob(
            self.store,
            ("value\n" + str(value) + "\n").encode(),
            "values.csv",
            self.problem_id,
            "generated",
            run_id,
        )
        trials = [
            {
                "alternative_id": alternative,
                "metrics": [
                    {
                        "metric_id": "quality",
                        "value": 0.0 if alternative == "baseline" else maker_score,
                        "lower": None,
                        "upper": None,
                    }
                ],
                "checks": [
                    {
                        "check_id": "instrument_check",
                        "passed": True,
                        "detail": "Unverified maker diagnostic.",
                    }
                ],
            }
            for alternative in ("baseline", "perturbed")
        ]
        output_id = save_blob(
            self.store,
            json.dumps(
                {
                    "protocol_id": protocol_id,
                    "basis": basis,
                    "trials": trials,
                    "data_description": "Synthetic fixtures for software tests.",
                    "uncertainty_method": "None established.",
                    "limitations": ["Not scientific evidence."],
                }
            ).encode(),
            "symplex_experiment.json",
            self.problem_id,
            "generated",
            run_id,
        )
        files = [csv_id, output_id]
        if source_code:
            files.append(save_blob(self.store, b'print("synthetic fixture")\n', "model.py", self.problem_id, "generated", run_id))
        package_id = self.store.put(
            "compute_package",
            {"run_id": run_id, "file_ids": files},
            self.problem_id,
        )
        comparison_id = (
            compare_computation(self.store, self.problem_id, package_id)
            if evaluate
            else None
        )
        return package_id, run_id, comparison_id, csv_id

    def test_host_pass_fraction_is_the_only_quality_and_all_alternatives_remain(self):
        poor, _, _, _ = self.execute(value=-1.0, maker_score=1e10)
        good, _, _, _ = self.execute(value=5.0, maker_score=-1e10)
        archive = build_search_archive(self.store, self.problem_id)
        self.assertEqual(archive["evaluator_version"], EVALUATOR_VERSION)
        self.assertEqual(len(archive["cells"]), 1)
        cell = archive["cells"][0]
        self.assertEqual(cell["elite_package_id"], good)
        self.assertEqual(archive["suggested_parent_id"], good)
        entries = {e["package_id"]: e for e in cell["entries"]}
        self.assertEqual(entries[poor]["diagnostic_pass_fraction"], 0.5)
        self.assertEqual(entries[good]["diagnostic_pass_fraction"], 1.0)
        self.assertEqual(entries[poor]["alternative_ids"], ["baseline", "perturbed"])
        self.assertEqual(
            cell["descriptor"]["declared_component_kinds"], ["mechanistic"]
        )
        self.assertEqual(
            cell["descriptor"]["protocol_digest"],
            self.store.get(self.protocol_id)["digest"],
        )
        self.assertIs(archive["promotion_allowed"], False)

    def test_missing_or_legacy_evaluation_has_no_fitness(self):
        missing, _, _, _ = self.execute(evaluate=False)
        old, _, comparison_id, _ = self.execute()
        original = self.store.get(comparison_id)
        self.store.invalidate(comparison_id)
        self.store.put(
            "experiment_comparison",
            dict(original["data"], evaluator_version="legacy-maker-scores"),
            self.problem_id,
        )
        archive = build_search_archive(self.store, self.problem_id)
        self.assertFalse(archive["cells"])
        self.assertIsNone(archive["suggested_parent_id"])
        self.assertEqual(
            {x["package_id"] for x in archive["excluded_candidates"]}, {missing, old}
        )
        self.assertTrue(
            all(x["fitness"] is None for x in archive["excluded_candidates"])
        )

    def test_ties_are_retained_and_reproducible_without_model_calls(self):
        first, _, _, _ = self.execute()
        second, _, _, _ = self.execute()
        before = len(self.store.list())
        a = build_search_archive(self.store, self.problem_id)
        b = build_search_archive(self.store, self.problem_id)
        self.assertEqual(a, b)
        self.assertEqual(len(self.store.list()), before)
        self.assertEqual(set(a["cells"][0]["tie_package_ids"]), {first, second})
        self.assertEqual(a["suggested_parent_id"], first)

    def test_actual_bound_revision_runs_drive_underexplored_niche_selection(self):
        visited, _, _, _ = self.execute(basis="synthetic")
        fresh, _, _, _ = self.execute(basis="conditional_model")
        self.execute(basis="synthetic", predecessor=visited, evaluate=False)
        archive = build_search_archive(self.store, self.problem_id)
        self.assertEqual(archive["suggested_parent_id"], fresh)
        cells = {c["descriptor"]["evidence_basis"]: c for c in archive["cells"]}
        self.assertEqual(cells["synthetic"]["visit_count"], 1)
        self.assertEqual(cells["conditional_model"]["visit_count"], 0)
        entry = cells["synthetic"]["entries"][0]
        self.assertEqual(entry["parent_visit_count"], 1)

    def test_unbound_predecessor_claim_does_not_count_as_a_visit(self):
        package, _, _, _ = self.execute()
        _, run_id, _, _ = self.execute(
            predecessor=package, predecessor_in_manifest=False, evaluate=False
        )
        archive = build_search_archive(self.store, self.problem_id)
        self.assertEqual(archive["cells"][0]["visit_count"], 0)
        self.assertEqual(archive["ignored_visit_records"][0]["run_id"], run_id)

    def test_distinct_frozen_protocols_are_never_ranked_by_incomparable_scores(self):
        earlier, _, _, _ = self.execute(value=-1.0)
        second_protocol = self.freeze(
            "A different frozen comparison; incomparable suite identity"
        )
        later, _, _, _ = self.execute(value=5.0, protocol_id=second_protocol)
        archive = build_search_archive(self.store, self.problem_id)
        self.assertEqual(len(archive["cells"]), 2)
        self.assertEqual(archive["suggested_parent_id"], earlier)
        self.assertNotEqual(archive["suggested_parent_id"], later)
        self.assertEqual(
            len({c["descriptor"]["protocol_digest"] for c in archive["cells"]}), 2
        )

    def test_evolution_selects_current_protocol_even_when_older_group_is_less_visited(self):
        earlier, _, _, _ = self.execute()
        current_protocol = self.freeze("A second frozen comparison")
        current, _, _, _ = self.execute(protocol_id=current_protocol, source_code=True)
        self.execute(protocol_id=current_protocol, predecessor=current, evaluate=False)
        global_archive = build_search_archive(self.store, self.problem_id)
        self.assertEqual(global_archive["suggested_parent_id"], earlier)
        ctx = SimpleNamespace(store=self.store, problem_id=self.problem_id)
        action = NextStep("evolve_model", "", "Investigate convergence", "Discretization", "Lower error")
        with patch("symplex.agents.tools.run_model_code", return_value={}) as run:
            result = evolve_model(ctx, action)
        self.assertEqual(run.call_args.args[1].target_id, current)
        self.assertEqual(result["selected_parent_id"], current)
        archived = self.store.get(result["search_archive_id"])["data"]
        self.assertEqual(len(archived["cells"]), 2)
        self.assertEqual(archived["selection_scope"], "specified_frozen_protocol")
        self.assertEqual(archived["selected_protocol_digest"], self.store.get(current_protocol)["digest"])

    def test_scoped_selection_never_falls_back_to_a_different_protocol(self):
        self.execute()
        current_protocol = self.freeze("New suite not yet executed")
        archive = build_search_archive(
            self.store, self.problem_id,
            protocol_digest=self.store.get(current_protocol)["digest"],
        )
        self.assertEqual(len(archive["cells"]), 1)
        self.assertIsNone(archive["suggested_parent_id"])
        ctx = SimpleNamespace(store=self.store, problem_id=self.problem_id)
        action = NextStep("evolve_model", "", "Investigate convergence", "Discretization", "Lower error")
        with patch("symplex.agents.tools.run_model_code") as run:
            with self.assertRaisesRegex(Invalid, "current frozen protocol"):
                evolve_model(ctx, action)
        run.assert_not_called()

    def test_evolution_requires_source_in_actual_revision_input_envelope(self):
        self.execute()
        ctx = SimpleNamespace(store=self.store, problem_id=self.problem_id)
        action = NextStep("evolve_model", "", "Investigate convergence", "Discretization", "Lower error")
        with patch("symplex.agents.tools.run_model_code") as run:
            with self.assertRaisesRegex(Invalid, "included predecessor Python"):
                evolve_model(ctx, action)
        run.assert_not_called()

    def test_stale_verification_or_mismatched_result_digest_excludes_candidate(self):
        first, _, comparison_id, _ = self.execute()
        original = self.store.get(comparison_id)
        self.store.invalidate(original["data"]["numerical_verification_id"])
        second, _, second_comparison_id, _ = self.execute()
        second_data = copy.deepcopy(self.store.get(second_comparison_id)["data"])
        self.store.invalidate(second_comparison_id)
        second_data["numerical_result_digest"] = "wrong"
        self.store.put("experiment_comparison", second_data, self.problem_id)
        archive = build_search_archive(self.store, self.problem_id)
        self.assertFalse(archive["cells"])
        excluded = {e["package_id"]: e for e in archive["excluded_candidates"]}
        self.assertIn(
            "not_current_scoped_numerical_verification", excluded[first]["reasons"]
        )
        self.assertIn("numerical_result_digest_mismatch", excluded[second]["reasons"])

    def test_tampered_checked_csv_does_not_retain_fitness(self):
        package, _, _, csv_id = self.execute()
        record = self.store.get(csv_id)
        (self.store.root / "blobs" / record["data"]["sha256"]).write_bytes(
            b"value\n999\n"
        )
        archive = build_search_archive(self.store, self.problem_id)
        self.assertFalse(archive["cells"])
        self.assertEqual(archive["excluded_candidates"][0]["package_id"], package)
        self.assertIsNone(archive["excluded_candidates"][0]["fitness"])

    def test_other_problem_and_stale_packages_are_not_candidates(self):
        package, _, _, _ = self.execute()
        self.store.invalidate(package)
        other = self.store.put("workspace_problem", {"question": "Other problem"})
        self.store.put("compute_package", {"run_id": "foreign"}, other)
        archive = build_search_archive(self.store, self.problem_id)
        self.assertFalse(archive["cells"])
        self.assertEqual(len(archive["excluded_candidates"]), 1)
        self.assertEqual(archive["excluded_candidates"][0]["package_id"], package)


if __name__ == "__main__":
    unittest.main()
