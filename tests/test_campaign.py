"""Campaign retries preserve paid usage, including interrupted reservations."""

import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

from scripts.run_case_campaign import (
    CaseBudget, campaign_lock, case_scope_id, case_usage, recover_attempts,
    recover_scoped_attempts, run_campaign, should_run,
)
from symplex.core.contracts import digest
from symplex.infrastructure.scoped_budget import ScopedBudget
from symplex.infrastructure.storage import Budget, BudgetExhausted, Store


class CampaignTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = Store(directory.name)
        self.budget = Budget(self.store, {"usd": 20})
        self.campaign = self.store.put("campaign_plan", {})
        self.plan = {
            "proposed_additional_budget_usd": 20, "case_budget_usd": 5,
            "cases": [{"id": name, "problem": "Investigate " + name} for name in ("first", "second")],
            "shared_context": "Use synthetic data honestly", "depth": "balanced",
            "milestones": ["compute_package", "experiment_comparison", "model_critique"],
        }
        self.args = SimpleNamespace(case_id=None, resume=False, retry_incomplete=False,
                                    campaign_budget_usd=None, case_budget_usd=None)

    def attempt(self, case="first"):
        return self.store.put("campaign_attempt", {
            "case_id": case, "budget_start": self.budget.snapshot(),
        }, self.campaign)

    def test_hard_interruption_charges_reserved_delta_only_to_interrupted_case(self):
        self.budget.reserve(usd=2, model_requests=1)
        self.store.put("campaign_result", {
            "case_id": "first", "budget": self.budget.snapshot(),
        }, self.campaign)
        self.attempt("second")
        self.budget.reserve(usd=3, model_requests=2)
        recover_attempts(self.store, self.budget, self.campaign)
        recover_attempts(self.store, self.budget, self.campaign)
        self.assertEqual(len(self.store.list("campaign_attempt_end")), 1)
        self.assertEqual(case_usage(self.store, self.campaign, "first")["usd"], 2)
        consumed = case_usage(self.store, self.campaign, "second")
        self.assertEqual(consumed["usd"], 3)
        resumed = CaseBudget(self.budget, 5, consumed)
        resumed.reserve(usd=2)
        with self.assertRaises(BudgetExhausted):
            resumed.reserve(usd=0.01)
        self.assertEqual(self.budget.snapshot()["usd"]["cap"], 20)

    def test_attempt_and_report_do_not_double_count_or_require_report_to_survive(self):
        first = self.attempt()
        self.budget.reserve(usd=1)
        recover_attempts(self.store, self.budget, self.campaign)
        self.store.put("campaign_result", {
            "case_id": "first", "budget": self.budget.snapshot(),
            "attempt_id": first, "usage_delta": {"usd": 1},
        }, self.campaign)
        self.attempt()
        self.budget.reserve(usd=2)
        recover_attempts(self.store, self.budget, self.campaign)
        self.assertEqual(case_usage(self.store, self.campaign, "first")["usd"], 3)

    def test_legacy_explicit_zero_usage_is_not_replaced_by_campaign_delta(self):
        self.budget.reserve(usd=3)
        self.store.put("campaign_result", {
            "case_id": "first", "budget": self.budget.snapshot(), "usage_delta": {},
        }, self.campaign)
        self.assertEqual(case_usage(self.store, self.campaign, "first"), {})

    def test_unrecovered_or_overlapping_attempts_fail_closed(self):
        self.attempt()
        with self.assertRaises(RuntimeError):
            case_usage(self.store, self.campaign, "first")
        self.attempt("second")
        with self.assertRaises(RuntimeError):
            recover_attempts(self.store, self.budget, self.campaign)

    def test_case_overrun_blocks_every_resource_and_does_not_raise_global_cap(self):
        allocation = CaseBudget(self.budget, 5, {"usd": 4})
        reservation = allocation.reserve(usd=1)
        allocation.settle(reservation, usd=2)
        with self.assertRaises(BudgetExhausted):
            allocation.reserve(model_requests=1)
        self.assertEqual(self.budget.snapshot()["usd"]["cap"], 20)

    def test_retry_filter_only_reopens_success_missing_execution_or_evaluation(self):
        report = {"data": {"job": {"status": "succeeded"}, "milestones": {}}}
        self.assertFalse(should_run([report], self.plan, resume=True))
        self.assertTrue(should_run([report], self.plan, retry_incomplete=True))
        report["data"]["milestones"] = {name: [name + "_id"] for name in self.plan["milestones"]}
        self.assertFalse(should_run([report], self.plan, retry_incomplete=True))
        report["data"]["job"]["status"] = "failed"
        self.assertTrue(should_run([report], self.plan, resume=True))
        self.assertFalse(should_run([report], self.plan))

    def test_soft_interruption_is_persisted_before_exception_and_retry_keeps_usage(self):
        def interrupted(store, budget, provider, problem, depth):
            budget.reserve(usd=3)
            raise KeyboardInterrupt()

        with patch("scripts.run_case_campaign.OpenAIProvider"), patch("scripts.run_case_campaign.solve", interrupted):
            with self.assertRaises(KeyboardInterrupt):
                run_campaign(self.store, self.plan, self.args)
        attempt = self.store.list("campaign_attempt")[-1]
        ending = self.store.list("campaign_attempt_end")[-1]
        self.assertEqual(ending["parent"], attempt["id"])
        self.assertEqual(ending["data"]["status"], "interrupted")
        self.assertEqual(case_usage(self.store, attempt["parent"], "first")["usd"], 3)

        def resumed(store, budget, provider, problem, depth):
            with self.assertRaises(BudgetExhausted):
                budget.reserve(usd=2.01)
            budget.reserve(usd=2)
            return {"status": "succeeded", "id": "job_test"}

        self.args.case_id = "first"
        with patch("scripts.run_case_campaign.OpenAIProvider"), patch("scripts.run_case_campaign.solve", resumed):
            run_campaign(self.store, self.plan, self.args)
        self.assertEqual(case_usage(self.store, attempt["parent"], "first")["usd"], 5)

    def test_case_selection_and_incomplete_retry_add_fresh_context_without_reset(self):
        self.args.case_id = "second"
        contexts = []

        def completed(store, budget, provider, problem, depth):
            contexts.append([r["digest"] for r in store.list("context") if r["parent"] == problem])
            budget.reserve(usd=1)
            return {"status": "succeeded", "id": "job_test"}

        with patch("scripts.run_case_campaign.OpenAIProvider"), patch("scripts.run_case_campaign.solve", completed):
            run_campaign(self.store, self.plan, self.args)
            self.args.retry_incomplete = True
            run_campaign(self.store, self.plan, self.args)
        self.assertEqual(len(self.store.list("workspace_problem")), 1)
        self.assertEqual(len(contexts), 2)
        self.assertNotEqual(contexts[0], contexts[1])
        reports = self.store.list("campaign_result")
        self.assertEqual(reports[-1]["data"]["prior_case_usage"]["usd"], 1)
        self.assertEqual(case_usage(self.store, reports[-1]["parent"], "second")["usd"], 2)

    def test_budget_override_requires_existing_grant_and_does_not_modify_caps(self):
        self.args.campaign_budget_usd = 40
        with self.assertRaises(ValueError):
            run_campaign(self.store, self.plan, self.args)
        self.assertEqual(self.budget.snapshot()["usd"]["cap"], 20)
        self.assertEqual(self.budget.snapshot()["usd"]["used_or_reserved"], 0)

    def test_expanded_case_preserves_spend_and_narrower_campaign_ceiling(self):
        self.budget.reserve(usd=4)
        expanded = CaseBudget(self.budget, 10, {"usd": 4}, campaign_usd=8)
        self.assertEqual(expanded.limits["output_tokens"], 90000)
        self.assertEqual(expanded.limits["model_requests"], 80)
        expanded.reserve(usd=4)
        with self.assertRaises(BudgetExhausted):
            expanded.reserve(usd=0.01)
        self.assertEqual(self.budget.snapshot()["usd"]["cap"], 20)

    def test_parallel_cases_keep_foreign_attempt_open_and_report_only_own_delta(self):
        active = threading.Barrier(2)
        spent = threading.Barrier(2)

        def concurrent_solve(store, budget, provider, problem, depth):
            case_id = store.get(problem)["data"]["case_id"]
            active.wait(timeout=10)
            attempts = [r for r in store.list("campaign_attempt") if r["data"].get("scope_id")]
            self.assertEqual(len(attempts), 2)
            self.assertFalse(store.list("campaign_attempt_end"))
            budget.reserve(usd=1 if case_id == "first" else 2)
            spent.wait(timeout=10)
            return {"status": "succeeded", "id": "job_" + case_id}

        args = [SimpleNamespace(**dict(vars(self.args), case_id=case)) for case in ("first", "second")]
        with patch("scripts.run_case_campaign.OpenAIProvider"), patch("scripts.run_case_campaign.solve", concurrent_solve), patch("builtins.print"):
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(lambda options: run_campaign(self.store, self.plan, options), args))
        reports = self.store.list("campaign_result")
        self.assertEqual(len(reports), 2)
        self.assertEqual(len({r["parent"] for r in reports}), 1)
        for report in reports:
            data = report["data"]
            expected = 1 if data["case_id"] == "first" else 2
            self.assertEqual(data["usage_delta"]["usd"], expected)
            self.assertEqual(data["scope_budget"]["usd"]["used_or_reserved"], expected)
            self.assertEqual(case_usage(self.store, report["parent"], data["case_id"])["usd"], expected)
        self.assertEqual(self.budget.snapshot()["usd"]["used_or_reserved"], 3)

    def test_scoped_recovery_never_closes_foreign_case_or_uses_global_span(self):
        scopes = {case: ScopedBudget(self.budget, case_scope_id(self.campaign, case), {"usd": 5}) for case in ("first", "second")}
        attempts = {}
        for case, scope in scopes.items():
            attempts[case] = self.store.put("campaign_attempt", {
                "case_id": case, "scope_id": scope.scope_id,
                "budget_start": self.budget.snapshot(), "scope_start": scope.scope_snapshot(),
            }, self.campaign)
        scopes["first"].reserve(usd=1)
        scopes["second"].reserve(usd=2)
        recover_attempts(self.store, self.budget, self.campaign)
        self.assertFalse(self.store.list("campaign_attempt_end"))
        recover_scoped_attempts(self.store, scopes["first"], self.campaign, "first")
        endings = self.store.list("campaign_attempt_end")
        self.assertEqual(len(endings), 1)
        self.assertEqual(endings[0]["parent"], attempts["first"])
        self.assertEqual(endings[0]["data"]["usage_delta"]["usd"], 1)
        self.assertEqual(case_usage(self.store, self.campaign, "second")["usd"], 2)

    def test_legacy_migration_seeds_once_before_any_scoped_paid_work(self):
        campaign = self.store.put("campaign_plan", {"plan": self.plan, "plan_digest": digest(self.plan)})
        self.budget.reserve(usd=3)
        self.store.put("campaign_result", {
            "case_id": "first", "budget": self.budget.snapshot(),
            "job": {"status": "failed"}, "milestones": {},
        }, campaign)
        self.store.put("campaign_attempt", {"case_id": "second", "budget_start": self.budget.snapshot()}, campaign)
        self.budget.reserve(usd=2)

        def continuation(store, budget, provider, problem, depth):
            self.assertEqual(budget.scope_snapshot()["usd"]["used_or_reserved"], 3)
            self.assertEqual(budget.parent.snapshot()["usd"]["used_or_reserved"], 5)
            with self.assertRaises(BudgetExhausted):
                budget.reserve(usd=2.01)
            budget.reserve(usd=1)
            return {"status": "succeeded", "id": "job_test"}

        self.args.case_id, self.args.resume = "first", True
        with patch("scripts.run_case_campaign.OpenAIProvider"), patch("scripts.run_case_campaign.solve", continuation), patch("builtins.print"):
            run_campaign(self.store, self.plan, self.args)
            run_campaign(self.store, self.plan, self.args)
        self.assertEqual(self.budget.snapshot()["usd"]["used_or_reserved"], 6)
        self.assertEqual(case_usage(self.store, campaign, "first")["usd"], 4)
        self.assertEqual(case_usage(self.store, campaign, "second")["usd"], 2)

    def test_legacy_recovery_refuses_global_attribution_after_scoped_work(self):
        self.attempt()
        scope = ScopedBudget(self.budget, case_scope_id(self.campaign, "second"), {"usd": 5})
        self.store.put("campaign_attempt", {"case_id": "second", "scope_id": scope.scope_id,
                       "scope_start": scope.scope_snapshot()}, self.campaign)
        with self.assertRaises(RuntimeError):
            recover_attempts(self.store, self.budget, self.campaign)

    def test_legacy_process_exclusive_lock_prevents_scoped_runner_start(self):
        with campaign_lock(self.store, "campaign.lock"):
            with self.assertRaises(RuntimeError):
                run_campaign(self.store, self.plan, self.args)
        self.assertFalse(self.store.list("campaign_attempt"))

    def test_same_case_lock_prevents_second_solver_but_different_case_locks_coexist(self):
        first = "campaign-case-" + digest(case_scope_id(self.campaign, "first")) + ".lock"
        second = "campaign-case-" + digest(case_scope_id(self.campaign, "second")) + ".lock"
        with campaign_lock(self.store, first):
            with self.assertRaises(RuntimeError):
                with campaign_lock(self.store, first):
                    self.fail("Same case must not execute twice concurrently")
            with campaign_lock(self.store, second):
                pass

    def test_reopening_with_different_scope_cap_cannot_silently_change_allocation(self):
        def completed(store, budget, provider, problem, depth):
            return {"status": "succeeded", "id": "job_test"}

        self.args.case_id = "first"
        with patch("scripts.run_case_campaign.OpenAIProvider"), patch("scripts.run_case_campaign.solve", completed), patch("builtins.print"):
            run_campaign(self.store, self.plan, self.args)
            self.args.case_budget_usd = 10
            with self.assertRaises(ValueError):
                run_campaign(self.store, self.plan, self.args)
        self.assertEqual(self.budget.snapshot()["usd"]["used_or_reserved"], 0)


if __name__ == "__main__":
    unittest.main()
