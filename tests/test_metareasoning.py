"""Host measurement of the metareasoner: uncertainty ledger, allocation prior, calibration.

Synthetic fixtures only. Nothing here executes model code, reaches a network, or
establishes anything scientific; the assertions check that the host arithmetic is
deterministic, provenance-bound and honest about what it could not measure.
"""

import copy
import tempfile
import unittest

from symplex.agents.calibration_ledger import (
    METHOD_FEEDBACK_KINDS,
    action_history,
    calibration_report,
    improvement_signal,
    record_prediction,
    score_outcome,
)
from symplex.agents.policies import DEPTHS
from symplex.agents.uncertainty import (
    RESOLVING_TOOLS,
    TYPES,
    ledger_digest,
    uncertainty_ledger,
)
from symplex.agents.voi import (
    PRIOR_RESOLUTION_RATE,
    check_tool_tables,
    ledger_delta,
    resolution_rate,
    score_actions,
)
from symplex.core.contracts import Invalid
from symplex.infrastructure.storage import Store

try:
    from tests.test_complex_system import system_spec
except ModuleNotFoundError:
    from test_complex_system import system_spec


def comparison_record(protocol_id, failed=("bounded_values",), passed=("finite_values",)):
    """A minimal recorded comparison; only the host-check summary matters here."""
    return {
        "protocol_id": protocol_id,
        "package_id": "compute_package_fixture",
        "run_id": "compute_run_fixture",
        "numerical_verification": {
            "status": "failed" if failed else "checked",
            "all_passed": not failed,
            "check_count": len(failed) + len(passed),
            "failed_check_ids": list(failed),
            "scope": "synthetic fixture",
        },
        "comparisons": [{"candidate_id": "perturbed", "status": "not_established"}],
        "scope": "Synthetic fixture; no experiment was executed.",
    }


class LedgerTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.store = Store(folder.name)
        self.problem_id = self.store.put(
            "workspace_problem", {"question": "Which mechanism explains the response?"}
        )
        self.system_id = self.store.put("complex_system", system_spec(), self.problem_id)

    def ledger(self):
        return uncertainty_ledger(self.store, self.problem_id)

    def items_of(self, ledger, kind):
        return [i for i in ledger["items"] if i["type"] == kind]

    def test_unmeasured_state_produces_exactly_one_observational_item_citing_its_artifact(self):
        ledger = self.ledger()
        observational = self.items_of(ledger, "observational")
        self.assertEqual(len(observational), 1)
        item = observational[0]
        self.assertEqual(item["reference"], "states:response")
        self.assertEqual(item["derived_from"], [self.system_id])
        self.assertEqual(item["detail"]["state_id"], "response")
        self.assertEqual(item["blocking_for"], "endpoint:endpoint")
        self.assertEqual(item["resolvable_by"], list(RESOLVING_TOOLS["observational"]))
        self.assertIn("no", ledger["scope"].lower())

    def test_unresolved_rivals_are_structural_and_a_failed_host_check_is_numerical(self):
        structural = self.items_of(self.ledger(), "structural")
        self.assertEqual(len(structural), 1)
        self.assertEqual(structural[0]["detail"]["hypothesis_ids"], ["delayed", "memoryless"])
        self.assertIn("no completed frozen comparison", structural[0]["statement"])
        self.assertEqual(structural[0]["blocking_for"], "experiment:lag_test")

        protocol_id = self.store.put(
            "experiment_protocol",
            {"experiment_id": "lag_test", "execution_readiness": "ready",
             "blocking_reasons": [], "status": "frozen_for_comparison"},
            self.problem_id,
        )
        comparison_id = self.store.put(
            "experiment_comparison", comparison_record(protocol_id), self.problem_id
        )
        numerical = self.items_of(self.ledger(), "numerical")
        self.assertEqual([i["reference"] for i in numerical], ["host_check:bounded_values"])
        self.assertEqual(numerical[0]["derived_from"], [comparison_id])
        self.assertEqual(numerical[0]["severity"], 0.9)
        # A failed comparison does not discriminate, so the rivals stay open.
        self.assertEqual(len(self.items_of(self.ledger(), "structural")), 1)

    def test_a_passing_comparison_closes_the_rival_pair_it_covers(self):
        protocol_id = self.store.put(
            "experiment_protocol",
            {"experiment_id": "lag_test", "execution_readiness": "ready",
             "blocking_reasons": [], "status": "frozen_for_comparison"},
            self.problem_id,
        )
        record = comparison_record(protocol_id, failed=())
        record["comparisons"] = [{"candidate_id": "perturbed", "status": "meets_declared_comparison"}]
        self.store.put("experiment_comparison", record, self.problem_id)
        self.assertEqual(self.items_of(self.ledger(), "structural"), [])

    def test_every_item_cites_a_real_stored_artifact_of_this_problem(self):
        self.store.put(
            "evidence_synthesis",
            {"disagreements": [{"claim_ids": ["a", "b"], "hypothesis_ids": ["delayed"],
                                "explanation": "Two recorded sources report opposite lag.",
                                "resolving_observation": "A repeated held perturbation."}],
             "gaps": [{"id": "lag_gap", "hypothesis_ids": ["delayed"],
                       "missing_information": "No independent lag measurement exists.",
                       "discriminating_observation": "Held perturbation.", "next_action": "Measure."}]},
            self.problem_id,
        )
        self.store.put(
            "model_critique",
            {"assessment": "Fixture", "unsupported_claims": ["The lag is physically established."],
             "next_validation": "Measure the lag."},
            self.problem_id,
        )
        self.store.put(
            "experiment_protocol",
            {"experiment_id": "lag_test", "execution_readiness": "blocked",
             "blocking_reasons": ["No calibrated instrument is available."],
             "status": "blocked_design"},
            self.problem_id,
        )
        ledger = self.ledger()
        self.assertGreaterEqual(len(ledger["items"]), 7)
        for item in ledger["items"]:
            self.assertTrue(item["derived_from"], item["id"])
            self.assertIn(item["type"], TYPES)
            for artifact_id in item["derived_from"]:
                record = self.store.get(artifact_id)
                self.assertEqual(record["parent"], self.problem_id)
                self.assertFalse(record["stale"])
        self.assertEqual(
            sorted(ledger["derived_from_artifact_ids"]),
            sorted({a for i in ledger["items"] for a in i["derived_from"]}),
        )
        self.assertEqual(
            {i["type"] for i in ledger["items"]},
            {"parametric", "structural", "observational", "decision", "evidential"},
        )

    def test_empty_problem_returns_an_empty_ledger_without_raising(self):
        empty_id = self.store.put("workspace_problem", {"question": "Nothing recorded yet."})
        ledger = uncertainty_ledger(self.store, empty_id)
        self.assertEqual(ledger["items"], [])
        self.assertEqual(ledger["total_severity_mass"], 0.0)
        self.assertIsNone(ledger["system_id"])
        self.assertFalse(ledger["truncated"])
        self.assertEqual(ledger["ledger_digest"], ledger_digest(ledger))
        self.assertIn("scope", ledger)

    def test_ledger_is_deterministic_and_bounded_and_rejects_bad_input(self):
        first, second = self.ledger(), self.ledger()
        self.assertEqual(first, second)
        self.assertEqual(first["ledger_digest"], ledger_digest(second))
        bounded = uncertainty_ledger(self.store, self.problem_id, max_items=1)
        self.assertEqual(len(bounded["items"]), 1)
        self.assertTrue(bounded["truncated"])
        self.assertEqual(bounded["omitted_item_count"], len(first["items"]) - 1)
        self.assertNotEqual(bounded["ledger_digest"], first["ledger_digest"])
        with self.assertRaises(Invalid):
            uncertainty_ledger(self.store, self.system_id)
        with self.assertRaises(Invalid):
            uncertainty_ledger(self.store, "no_such_record")
        with self.assertRaises(Invalid):
            uncertainty_ledger(self.store, self.problem_id, max_items=0)

    def test_ledger_delta_reports_resolved_new_and_unchanged(self):
        before = self.ledger()
        measured = copy.deepcopy(system_spec())
        measured["observations"][0]["state_ids"] = ["temperature", "response"]
        self.store.put("complex_system", measured, self.problem_id)
        critique_id = self.store.put(
            "model_critique",
            {"assessment": "Fixture", "unsupported_claims": ["Lag is established."],
             "next_validation": "Measure."},
            self.problem_id,
        )
        after = self.ledger()
        delta = ledger_delta(before, after)

        resolved = {i["id"]: i for i in before["items"] if i["id"] in delta["resolved_ids"]}
        self.assertEqual(
            sorted(i["reference"] for i in resolved.values()),
            ["decision.endpoints:endpoint", "states:response"],
        )
        created = {i["id"]: i for i in after["items"] if i["id"] in delta["new_ids"]}
        self.assertEqual(len(created), 1)
        self.assertEqual(next(iter(created.values()))["derived_from"], [critique_id])
        unchanged = {i["id"]: i for i in after["items"] if i["id"] in delta["unchanged_ids"]}
        self.assertEqual(
            sorted(i["rule"] for i in unchanged.values()),
            ["identifiability_gap", "undiscriminated_rivals"],
        )
        self.assertEqual(delta["resolved_count"] + delta["unchanged_count"], len(before["items"]))
        self.assertEqual(delta["direction"], "reduced")
        self.assertAlmostEqual(
            delta["net_severity_change"],
            after["total_severity_mass"] - before["total_severity_mass"], places=6)
        self.assertEqual(delta["by_type"]["observational"]["resolved"], 1)
        self.assertEqual(delta["by_type"]["evidential"]["new"], 1)
        self.assertEqual(delta["by_type"]["structural"]["unchanged"], 1)
        self.assertEqual(delta["after_digest"], after["ledger_digest"])
        with self.assertRaises(Invalid):
            ledger_delta(before, {"items": "not a list"})


class ValueOfInformationTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.store = Store(folder.name)
        self.problem_id = self.store.put("workspace_problem", {"question": "Fixture."})
        self.system_id = self.store.put("complex_system", system_spec(), self.problem_id)
        self.ledger = uncertainty_ledger(self.store, self.problem_id)
        self.envelope = DEPTHS["balanced"]
        self.tools = ["plan_experiment", "run_model_code", "discover_datasets",
                      "research_evidence", "propose_method", "deliver"]

    def test_ranking_is_deterministic_and_repeats_the_same_order(self):
        first = score_actions(self.ledger, self.tools, self.envelope, [])
        second = score_actions(self.ledger, list(reversed(self.tools)), self.envelope, [])
        self.assertEqual([r["tool"] for r in first["ranked"]], [r["tool"] for r in second["ranked"]])
        self.assertEqual([r["score"] for r in first["ranked"]], [r["score"] for r in second["ranked"]])
        self.assertEqual([r["rank"] for r in first["ranked"]], list(range(1, len(first["ranked"]) + 1)))
        self.assertEqual(first["unscored_tools"], [
            {"tool": "deliver",
             "reason": "delivery is a host action, not an information-gathering action"}])
        self.assertEqual(first["ledger_digest"], self.ledger["ledger_digest"])

    def test_each_score_multiplies_and_sums_exactly_as_documented(self):
        ranking = score_actions(self.ledger, self.tools, self.envelope, [])
        scored_any = False
        for entry in ranking["ranked"]:
            terms = entry["terms"]
            for kind, mass in terms["severity_mass_by_type"].items():
                self.assertAlmostEqual(
                    terms["weighted_gain_by_type"][kind],
                    mass * terms["resolution_rate_by_type"][kind]["rate"], places=12)
            self.assertAlmostEqual(
                terms["expected_information_gain"],
                sum(terms["weighted_gain_by_type"].values()), places=12)
            self.assertAlmostEqual(
                entry["score"],
                terms["expected_information_gain"]
                / terms["expected_cost"]["expected_cost"]
                * terms["exploration"]["exploration_bonus"], places=12)
            scored_any = scored_any or entry["score"] > 0
        self.assertTrue(scored_any)
        method = next(r for r in ranking["ranked"] if r["tool"] == "propose_method")
        self.assertEqual(method["score"], 0.0)
        self.assertEqual(method["addressed_item_ids"], [])
        self.assertIn("resolvable_by", method["note"])
        self.assertIn("not", ranking["scope"].lower())
        self.assertIn("limit_reason", ranking["authority"])

    def test_severity_mass_matches_the_ledger_items_the_tool_could_address(self):
        ranking = score_actions(self.ledger, self.tools, self.envelope, [])
        entry = next(r for r in ranking["ranked"] if r["tool"] == "plan_experiment")
        expected = [i for i in self.ledger["items"] if "plan_experiment" in i["resolvable_by"]]
        self.assertEqual(entry["addressed_item_ids"], sorted(i["id"] for i in expected))
        for kind, mass in entry["terms"]["severity_mass_by_type"].items():
            self.assertAlmostEqual(
                mass, sum(i["severity"] for i in expected if i["type"] == kind), places=6)

    def test_exploration_bonus_favours_the_under_visited_dimension(self):
        history = [{"tool": "run_model_code", "search_dimension": "computational",
                    "uncertainty_type": "structural", "status": "pending", "hit": None}] * 4
        ranking = score_actions(self.ledger, self.tools, self.envelope, history)
        crowded = next(r for r in ranking["ranked"] if r["tool"] == "run_model_code")
        fresh = next(r for r in ranking["ranked"] if r["tool"] == "plan_experiment")
        self.assertEqual(crowded["terms"]["exploration"]["exploration_bonus"], 1.0)
        self.assertEqual(fresh["terms"]["exploration"]["exploration_bonus"], 1.5)
        self.assertEqual(ranking["dimension_visits"], {"computational": 4})

    def test_measured_history_replaces_the_prior_and_a_zero_rate_zeroes_the_gain(self):
        history = [
            {"tool": "run_model_code", "search_dimension": "computational",
             "uncertainty_type": "structural", "status": "scored", "hit": False},
            {"tool": "run_model_code", "search_dimension": "computational",
             "uncertainty_type": "structural", "status": "scored", "hit": False},
            {"tool": "run_model_code", "search_dimension": "computational",
             "uncertainty_type": "parametric", "status": "unscorable", "hit": None},
        ]
        ranking = score_actions(self.ledger, self.tools, self.envelope, history)
        entry = next(r for r in ranking["ranked"] if r["tool"] == "run_model_code")
        structural = entry["terms"]["resolution_rate_by_type"]["structural"]
        self.assertEqual(structural["basis"], "measured")
        self.assertEqual((structural["attempts"], structural["resolutions"]), (2, 0))
        self.assertEqual(structural["rate"], 0.0)
        self.assertTrue(structural["low_sample"])
        self.assertEqual(entry["terms"]["weighted_gain_by_type"]["structural"], 0.0)
        # An unscorable attempt never becomes a measured failure.
        self.assertEqual(entry["terms"]["resolution_rate_by_type"]["parametric"]["basis"], "prior")
        self.assertIn("run_model_code|parametric", ranking["prior_used_for"])
        self.assertNotIn("run_model_code|structural", ranking["prior_used_for"])

    def test_scarce_allowance_raises_expected_cost(self):
        used = [{"tool": "research_evidence", "search_dimension": "evidence",
                 "uncertainty_type": "evidential", "status": "pending", "hit": None}] * 2
        cheap = score_actions(self.ledger, self.tools, self.envelope, [])
        dear = score_actions(self.ledger, self.tools, self.envelope, used)
        cheap_cost = next(r for r in cheap["ranked"] if r["tool"] == "research_evidence")["terms"]["expected_cost"]
        dear_cost = next(r for r in dear["ranked"] if r["tool"] == "research_evidence")["terms"]["expected_cost"]
        self.assertEqual(cheap_cost["capped_allowance"], "research")
        self.assertEqual(cheap_cost["capped_allowance_used_fraction"], 0.0)
        self.assertEqual(dear_cost["allowance_used"], 2)
        self.assertGreater(dear_cost["expected_cost"], cheap_cost["expected_cost"])

    def test_no_history_returns_the_documented_prior_and_says_so(self):
        rate = resolution_rate(self.store, self.problem_id, "run_model_code", "structural")
        self.assertEqual(rate["basis"], "prior")
        self.assertEqual(rate["rate"], PRIOR_RESOLUTION_RATE)
        self.assertEqual(rate["attempts"], 0)
        self.assertEqual(rate["resolutions"], 0)
        self.assertEqual(rate["history_basis"], "no history")
        self.assertEqual(rate["history_entries_scanned"], 0)
        self.assertIn("documented prior", rate["note"])
        self.assertIn("not fitted", rate["prior_basis"])
        self.assertIn("scope", rate)

    def test_bad_input_is_refused(self):
        with self.assertRaises(Invalid):
            score_actions({"items": {}}, self.tools, self.envelope, [])
        with self.assertRaises(Invalid):
            score_actions(self.ledger, "run_model_code", self.envelope, [])
        with self.assertRaises(Invalid):
            score_actions(self.ledger, self.tools, self.envelope, ["not a dict"])
        with self.assertRaises(Invalid):
            resolution_rate(self.store, self.problem_id, "run_model_code", "imaginary")
        with self.assertRaises(Invalid):
            resolution_rate(self.store, self.problem_id, "", "structural")

    def test_declared_planning_tables_cover_every_registered_tool(self):
        coverage = check_tool_tables()
        if not coverage["available"]:
            self.skipTest("tool registry unavailable: " + coverage["reason"])
        self.assertEqual(coverage["missing_dimension"], [])
        self.assertEqual(coverage["missing_cost"], [])
        self.assertEqual(coverage["unknown_declared_tools"], [])
        self.assertEqual(coverage["revision_tool_drift"], [])


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.store = Store(folder.name)
        self.problem_id = self.store.put("workspace_problem", {"question": "Fixture."})
        self.system_id = self.store.put("complex_system", system_spec(), self.problem_id)
        self.critique_id = self.store.put(
            "model_critique",
            {"assessment": "Fixture", "unsupported_claims": ["Lag is established."],
             "next_validation": "Measure the lag."},
            self.problem_id,
        )

    def ledger(self):
        return uncertainty_ledger(self.store, self.problem_id)

    def item_id(self, ledger, rule):
        return next(i["id"] for i in ledger["items"] if i["rule"] == rule)

    def measure_the_response_state(self):
        measured = copy.deepcopy(system_spec())
        measured["observations"][0]["state_ids"] = ["temperature", "response"]
        return self.store.put("complex_system", measured, self.problem_id)

    def test_a_correct_and_a_wrong_prediction_give_the_expected_hit_rate(self):
        before = self.ledger()
        right = record_prediction(
            self.store, self.problem_id, tool="discover_datasets", search_dimension="evidence",
            uncertainty="The response state has no measurement channel.",
            expected_change="A measurement channel for the response state appears.",
            predicted_cost=1.0, ledger=before,
            uncertainty_item_ids=[self.item_id(before, "state_without_observation_channel")],
        )
        self.assertTrue(right["scorable"])
        self.assertEqual(right["uncertainty_type"], "observational")

        self.measure_the_response_state()
        hit = score_outcome(self.store, self.problem_id, right["prediction_id"], actual_cost=2.0)
        self.assertEqual(hit["status"], "scored")
        self.assertIs(hit["hit"], True)
        self.assertEqual(len(hit["resolved_cited_item_ids"]), 1)
        self.assertEqual(hit["cost"]["direction"], "underestimated")
        self.assertEqual(hit["cost"]["ratio"], 2.0)

        current = self.ledger()
        wrong = record_prediction(
            self.store, self.problem_id, tool="run_model_code", search_dimension="computational",
            uncertainty="The rival mechanisms are not discriminated.",
            expected_change="The rival hypotheses are discriminated.",
            predicted_cost=4.0, ledger=current,
            uncertainty_item_ids=[self.item_id(current, "undiscriminated_rivals")],
        )
        miss = score_outcome(self.store, self.problem_id, wrong["prediction_id"], actual_cost=2.0)
        self.assertEqual(miss["status"], "scored")
        self.assertIs(miss["hit"], False)
        self.assertEqual(miss["still_open_cited_item_ids"], miss["cited_item_ids"])
        self.assertEqual(miss["cost"]["direction"], "overestimated")

        report = calibration_report(self.store, self.problem_id)
        self.assertEqual((report["scored"], report["unscorable"], report["pending"]), (2, 0, 0))
        self.assertEqual(report["hit_rate"], 0.5)
        self.assertEqual(report["by_search_dimension"]["evidence"]["hit_rate"], 1.0)
        self.assertEqual(report["by_search_dimension"]["computational"]["hit_rate"], 0.0)
        self.assertEqual(report["by_tool"]["discover_datasets"]["hits"], 1)
        self.assertEqual(report["by_tool"]["run_model_code"]["misses"], 1)
        self.assertEqual(report["cost_error"]["underestimates"], 1)
        self.assertEqual(report["cost_error"]["overestimates"], 1)
        self.assertEqual(report["cost_error"]["count"], 2)
        self.assertEqual(report["cost_error"]["mean_ratio"], 1.25)
        self.assertEqual(report["cost_error"]["median_ratio"], 1.25)
        self.assertEqual(report["brier"]["score"], 0.5)
        self.assertTrue(report["brier"]["degenerate"])

    def test_an_unscorable_prediction_is_counted_as_unscorable_not_as_a_miss(self):
        before = self.ledger()
        wrong = record_prediction(
            self.store, self.problem_id, tool="run_model_code", search_dimension="computational",
            uncertainty="The rival mechanisms are not discriminated.",
            expected_change="The rival hypotheses are discriminated.",
            predicted_cost=1.0, ledger=before,
            uncertainty_item_ids=[self.item_id(before, "undiscriminated_rivals")],
        )
        score_outcome(self.store, self.problem_id, wrong["prediction_id"])
        vague = record_prediction(
            self.store, self.problem_id, tool="review_model", search_dimension="structural",
            uncertainty="The model may be wrong somewhere.",
            expected_change="The review improves the model.", predicted_cost=1.0, ledger=before,
        )
        self.assertFalse(vague["scorable"])
        outcome = score_outcome(self.store, self.problem_id, vague["prediction_id"])
        self.assertEqual(outcome["status"], "unscorable")
        self.assertIsNone(outcome["hit"])
        self.assertIn("cited no ledger item", outcome["unscorable_reason"])

        pendant = record_prediction(
            self.store, self.problem_id, tool="plan_experiment", search_dimension="experimental",
            uncertainty="The rival mechanisms are not discriminated.",
            expected_change="A frozen protocol exists.", predicted_cost=1.0, ledger=before,
            uncertainty_item_ids=[self.item_id(before, "undiscriminated_rivals")],
        )
        self.assertTrue(pendant["prediction_id"])

        report = calibration_report(self.store, self.problem_id)
        self.assertEqual(report["predictions_total"], 3)
        self.assertEqual((report["scored"], report["unscorable"], report["pending"]), (1, 1, 1))
        self.assertEqual((report["hits"], report["misses"]), (0, 1))
        self.assertEqual(report["hit_rate"], 0.0)
        self.assertEqual(sum(report["unscorable_reasons"].values()), 1)
        self.assertIsNone(report["by_search_dimension"]["structural"]["hit_rate"])
        self.assertEqual(report["by_search_dimension"]["structural"]["unscorable"], 1)
        self.assertEqual(report["by_search_dimension"]["structural"]["misses"], 0)
        self.assertEqual(report["by_search_dimension"]["experimental"]["pending"], 1)
        self.assertEqual(report["cost_error"]["count"], 0)
        self.assertIsNone(report["cost_error"]["mean_ratio"])

    def test_empty_history_reports_no_history_rather_than_a_confident_number(self):
        report = calibration_report(self.store, self.problem_id)
        self.assertEqual(report["predictions_total"], 0)
        self.assertIsNone(report["hit_rate"])
        self.assertEqual(report["hit_rate_basis"], "no scored predictions")
        self.assertIsNone(report["brier"]["score"])
        self.assertEqual(action_history(self.store, self.problem_id)["history_basis"], "no history")
        signal = improvement_signal(self.store, self.problem_id)
        self.assertEqual(signal["signals"], [])
        self.assertEqual(signal["supporting_artifact_ids"], [])
        self.assertEqual(signal["method_candidate_inputs"]["observed_failure"], "")

    def test_improvement_signal_names_a_recurring_failure_mode_with_supporting_ids(self):
        expected_ids = set()
        for _ in range(2):
            ledger = self.ledger()
            prediction = record_prediction(
                self.store, self.problem_id, tool="run_model_code", search_dimension="computational",
                uncertainty="The rival mechanisms are not discriminated.",
                expected_change="The rival hypotheses are discriminated.",
                predicted_cost=1.0, ledger=ledger,
                uncertainty_item_ids=[self.item_id(ledger, "undiscriminated_rivals")],
            )
            outcome = score_outcome(
                self.store, self.problem_id, prediction["prediction_id"], actual_cost=3.0)
            expected_ids.update({prediction["prediction_id"], outcome["outcome_id"]})

        signal = improvement_signal(self.store, self.problem_id)
        unresolved = next(s for s in signal["signals"] if s["kind"] == "unresolved_pairing")
        self.assertIn("structural", unresolved["failure_mode"])
        self.assertIn("run_model_code", unresolved["failure_mode"])
        self.assertEqual((unresolved["occurrences"], unresolved["hits"]), (2, 0))
        self.assertEqual(unresolved["uncertainty_type"], "structural")
        self.assertEqual(set(unresolved["supporting_artifact_ids"]), expected_ids)
        self.assertEqual(set(signal["supporting_artifact_ids"]), expected_ids)
        for artifact_id in signal["supporting_artifact_ids"]:
            self.assertEqual(self.store.get(artifact_id)["parent"], self.problem_id)

        cost = next(s for s in signal["signals"] if s["kind"] == "cost_underestimation")
        self.assertIn("underestimated", cost["failure_mode"])

        inputs = signal["method_candidate_inputs"]
        self.assertEqual(inputs["role"], "metareasoner")
        self.assertEqual(inputs["observed_failure"], unresolved["failure_mode"])
        self.assertEqual(inputs["feedback_ids"], [self.critique_id])
        for artifact_id in inputs["feedback_ids"]:
            self.assertIn(self.store.get(artifact_id)["kind"], METHOD_FEEDBACK_KINDS)
        self.assertIn("propose_method admits", inputs["note"])
        self.assertEqual(signal["min_occurrences"], 2)

    def test_prediction_and_outcome_reject_bad_input(self):
        ledger = self.ledger()
        with self.assertRaises(Invalid):
            record_prediction(
                self.store, self.problem_id, tool="run_model_code", search_dimension="imaginary",
                uncertainty="x", expected_change="y", predicted_cost=1.0, ledger=ledger)
        with self.assertRaises(Invalid):
            record_prediction(
                self.store, self.problem_id, tool="run_model_code", search_dimension="computational",
                uncertainty="x", expected_change="y", predicted_cost=-1.0, ledger=ledger)
        with self.assertRaises(Invalid):
            record_prediction(
                self.store, self.problem_id, tool="run_model_code", search_dimension="computational",
                uncertainty="x", expected_change="y", predicted_cost=1.0,
                ledger=ledger, uncertainty_type="imaginary")
        with self.assertRaises(Invalid):
            score_outcome(self.store, self.problem_id, self.system_id)
        with self.assertRaises(Invalid):
            score_outcome(self.store, self.problem_id, "no_such_prediction")
        with self.assertRaises(Invalid):
            improvement_signal(self.store, self.problem_id, min_occurrences=0)
        with self.assertRaises(Invalid):
            calibration_report(self.store, self.problem_id, max_history=0)

    def test_a_prediction_citing_an_item_that_was_not_open_is_unscorable(self):
        ledger = self.ledger()
        prediction = record_prediction(
            self.store, self.problem_id, tool="run_model_code", search_dimension="computational",
            uncertainty="An uncertainty that is not in this ledger.",
            expected_change="Something changes.", predicted_cost=1.0, ledger=ledger,
            uncertainty_item_ids=["unc_not_in_the_ledger"],
        )
        self.assertFalse(prediction["scorable"])
        outcome = score_outcome(self.store, self.problem_id, prediction["prediction_id"])
        self.assertEqual(outcome["status"], "unscorable")
        self.assertIn("not open in the ledger", outcome["unscorable_reason"])

    def test_secondary_observations_record_repaired_host_checks(self):
        protocol_id = self.store.put(
            "experiment_protocol",
            {"experiment_id": "lag_test", "execution_readiness": "ready",
             "blocking_reasons": [], "status": "frozen_for_comparison"},
            self.problem_id,
        )
        self.store.put(
            "numerical_verification",
            {"status": "failed", "all_passed": False,
             "checks": [{"id": "bounded_values", "passed": False}]},
            self.problem_id,
        )
        ledger = self.ledger()
        prediction = record_prediction(
            self.store, self.problem_id, tool="run_model_code", search_dimension="computational",
            uncertainty="The rival mechanisms are not discriminated.",
            expected_change="The failing bounds check passes.", predicted_cost=1.0, ledger=ledger,
            uncertainty_item_ids=[self.item_id(ledger, "undiscriminated_rivals")],
        )
        self.store.put(
            "numerical_verification",
            {"status": "checked", "all_passed": True,
             "checks": [{"id": "bounded_values", "passed": True}]},
            self.problem_id,
        )
        outcome = score_outcome(self.store, self.problem_id, prediction["prediction_id"])
        self.assertEqual(outcome["secondary_observations"]["host_checks_repaired"], ["bounded_values"])
        self.assertEqual(outcome["secondary_observations"]["host_checks_broken"], [])
        self.assertEqual(outcome["secondary_observations"]["new_artifact_kinds"], [])
        self.assertIs(outcome["hit"], False)
        self.assertEqual(protocol_id, self.store.get(protocol_id)["id"])


class ScopeDisciplineTests(unittest.TestCase):
    def test_every_public_return_states_what_is_not_established(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        store = Store(folder.name)
        problem_id = store.put("workspace_problem", {"question": "Fixture."})
        store.put("complex_system", system_spec(), problem_id)
        ledger = uncertainty_ledger(store, problem_id)
        prediction = record_prediction(
            store, problem_id, tool="plan_experiment", search_dimension="experimental",
            uncertainty="Rivals undiscriminated.", expected_change="A frozen protocol exists.",
            predicted_cost=1.0, ledger=ledger,
            uncertainty_item_ids=[ledger["items"][0]["id"]],
        )
        returns = [
            ledger,
            score_actions(ledger, ["plan_experiment"], DEPTHS["focused"], []),
            resolution_rate(store, problem_id, "plan_experiment", "structural"),
            ledger_delta(ledger, ledger),
            prediction,
            score_outcome(store, problem_id, prediction["prediction_id"]),
            action_history(store, problem_id),
            calibration_report(store, problem_id),
            improvement_signal(store, problem_id),
            check_tool_tables(),
        ]
        for value in returns:
            self.assertIsInstance(value, dict)
            self.assertIsInstance(value.get("scope"), str)
            self.assertTrue(value["scope"].strip())


if __name__ == "__main__":
    unittest.main()
