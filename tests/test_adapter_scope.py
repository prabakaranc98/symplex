"""Dedicated adapter measurements cannot validate an unrelated workspace problem."""

import unittest
from dataclasses import replace
from unittest.mock import patch

from symplex.agents.solver import solve
from symplex.agents.tools import REGISTRY
from symplex.core.proposals import DeliveryReview
from tests import test_investigation as investigation_fixtures
from tests.test_investigation import SequenceProvider, action


class AdapterScopeTests(unittest.TestCase):
    setUp = investigation_fixtures.InvestigationTests.setUp
    def test_p1_results_are_bound_as_auxiliary_evidence_with_their_actual_scope(self):
        dataset = self.store.put("imported_dataset", {"summary": {}, "pack": {}})
        adapter_problem = self.store.put("problem", {"question": "Forecast reliability"})
        for synthetic in (False, True):
            with self.subTest(synthetic=synthetic):
                problem = self.store.put("workspace_problem", {"question": "Protein recovery"})
                decision = self.store.put("decision", {
                    "problem_id": adapter_problem, "execution_completed": True,
                    "synthetic": synthetic, "scope": "Synthetic plumbing only" if synthetic else "Retrospective supplied-slice comparison",
                    "status": "inconclusive", "comparison": {"paired_brier_delta": 0.01},
                }, adapter_problem)
                provider = SequenceProvider([replace(action("run_p1"), target_id=dataset), action("deliver")])
                with patch("symplex.agents.tools.Engine.investigate", return_value={
                    "id": "adapter-job", "status": "succeeded", "result_id": decision,
                }):
                    result = solve(self.store, self.budget, provider, problem)
                self.assertEqual(result["status"], "succeeded")
                delivery = self.store.get(result["result_id"])["data"]
                self.assertEqual(delivery["status"], "dependency_gap")
                self.assertEqual(delivery["evaluated_decision_ids"], [])
                wrapper = self.store.get(delivery["auxiliary_adapter_run_ids"][0])
                self.assertEqual(wrapper["parent"], problem)
                self.assertEqual(wrapper["data"]["source_result_id"], decision)
                self.assertEqual(wrapper["data"]["source_result_digest"], self.store.get(decision)["digest"])
                self.assertFalse(wrapper["data"]["problem_specific_validation"])
                self.assertEqual(wrapper["data"]["synthetic"], synthetic)
                review = provider.contexts(DeliveryReview)[0]
                self.assertEqual(review["auxiliary_adapter_results"][0]["data"], wrapper["data"])
                self.assertNotIn(decision, delivery["artifact_ids"])

    def test_delivery_does_not_promote_legacy_foreign_synthetic_or_stale_decisions(self):
        other = self.store.put("workspace_problem", {"question": "Different question"})
        for parent, synthetic, stale, claimed_problem in (
            (other, False, False, other),
            (self.problem, True, False, self.problem),
            (self.problem, False, True, self.problem),
            (self.problem, False, False, other),
        ):
            with self.subTest(parent=parent, synthetic=synthetic, stale=stale, claimed_problem=claimed_problem):
                decision = self.store.put("decision", {
                    "execution_completed": True, "synthetic": synthetic, "problem_id": claimed_problem,
                }, parent)
                if stale:
                    self.store.invalidate(decision)
                self.store.put("steering", {"instruction": "Reassess this result"}, self.problem)
                provider = SequenceProvider([action("inspect_context"), action("deliver")])
                def supply_legacy_result(name, context, chosen, permissions):
                    context.result_ids.append(decision)
                    return {"result_id": decision}
                with patch.object(REGISTRY, "execute", side_effect=supply_legacy_result):
                    result = solve(self.store, self.budget, provider, self.problem)
                delivery = self.store.get(result["result_id"])["data"]
                self.assertEqual(delivery["status"], "dependency_gap")
                self.assertEqual(delivery["evaluated_decision_ids"], [])
