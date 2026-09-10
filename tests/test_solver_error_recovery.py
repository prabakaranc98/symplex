"""A rejected tool action is bounded feedback; host stop conditions remain final."""

import tempfile
import unittest
from unittest.mock import patch

from symplex.agents.solver import solve
from symplex.agents.tools import REGISTRY
from symplex.core.contracts import Invalid
from symplex.infrastructure.runner import Cancelled
from symplex.infrastructure.storage import Budget, BudgetExhausted, Store
from tests.test_investigation import SequenceProvider, action


class SolverErrorRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name)
        self.budget = Budget(self.store)
        self.problem = self.store.put(
            "workspace_problem",
            {"question": "Investigate a rejected numerical comparison"},
        )

    def test_invalid_tool_result_reaches_next_step_and_allows_different_action(self):
        provider = SequenceProvider(
            [action("inspect_context"), action("discover_datasets"), action("deliver")]
        )
        original = REGISTRY.execute

        def execute(name, context, proposal, permissions):
            if name == "inspect_context":
                reservation = context.budget.reserve(worker_seconds=1)
                context.budget.settle(reservation, worker_seconds=1)
                raise Invalid("Frozen CSV check rejected a missing column")
            return original(name, context, proposal, permissions)

        with patch.object(REGISTRY, "execute", side_effect=execute) as call:
            result = solve(
                self.store, self.budget, provider, self.problem, depth="focused"
            )
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(
            [c.args[0] for c in call.call_args_list],
            ["inspect_context", "discover_datasets"],
        )
        following = provider.next_contexts[1]
        rejected = following["completed_actions"][0]["result"]
        self.assertEqual(rejected["status"], "rejected")
        self.assertIn("missing column", rejected["error"])
        self.assertIn("unchanged contracts", rejected["required_response"])
        self.assertEqual(following["tool_counts"]["inspect_context"], 1)
        self.assertNotIn("inspect_context", following["available_tools"])
        recorded = next(
            r["data"]
            for r in self.store.list("agent_result")
            if r["data"]["status"] == "rejected"
        )
        self.assertEqual(
            recorded["resource_usage_or_reservations"]["worker_seconds"], 1
        )
        self.assertFalse(self.store.list("failure"))

    def test_failed_tool_still_consumes_limit_and_replans_once_within_permissions(self):
        provider = SequenceProvider(
            [
                action("inspect_context"),
                action("inspect_context"),
                action("discover_datasets"),
                action("deliver"),
            ]
        )
        original = REGISTRY.execute

        def execute(name, context, proposal, permissions):
            if name == "inspect_context":
                raise Invalid("Invalid scoped input")
            return original(name, context, proposal, permissions)

        with patch.object(REGISTRY, "execute", side_effect=execute) as call:
            result = solve(
                self.store, self.budget, provider, self.problem, depth="focused"
            )
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(
            [c.args[0] for c in call.call_args_list],
            ["inspect_context", "discover_datasets"],
        )
        replans = [c for c in provider.next_contexts if "host_feedback" in c]
        self.assertEqual(len(replans), 1)
        self.assertNotIn("inspect_context", replans[0]["available_tools"])
        self.assertEqual(len(self.store.list("action_rejection")), 1)

    def test_budget_and_cancellation_are_not_converted_to_recoverable_rejections(self):
        for exception, expected in (
            (BudgetExhausted("Ceiling reached"), "budget-exhausted"),
            (Cancelled("Stopped"), "cancelled"),
        ):
            with self.subTest(expected=expected):
                problem = self.store.put("workspace_problem", {"question": expected})
                provider = SequenceProvider([action("inspect_context")])
                with patch.object(REGISTRY, "execute", side_effect=exception):
                    result = solve(self.store, self.budget, provider, problem)
                self.assertEqual(result["status"], expected)
                self.assertEqual(len(provider.next_contexts), 1)
                self.assertFalse(
                    [
                        r
                        for r in self.store.list("deliverable")
                        if r["parent"] == problem
                    ]
                )

    def test_rejected_protocol_attempt_can_recover_without_spending_revision_allowance(self):
        provider = SequenceProvider([action("plan_experiment"), action("plan_experiment"), action("plan_experiment"), action("deliver")])
        attempts = []
        def execute(name, context, proposal, permissions):
            attempts.append(name)
            reservation = context.budget.reserve(worker_seconds=1)
            context.budget.settle(reservation, worker_seconds=1)
            if len(attempts) == 1:
                raise Invalid("Technical proposal-size rejection before protocol creation")
            return {"protocol_id": "accepted-offline-protocol"}
        with patch.object(REGISTRY, "execute", side_effect=execute):
            result = solve(self.store, self.budget, provider, self.problem, depth="focused")
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(attempts, ["plan_experiment", "plan_experiment"])
        after_failure = provider.next_contexts[1]
        self.assertEqual(after_failure["tool_counts"]["plan_experiment"], 1)
        self.assertEqual(after_failure["accepted_revision_counts"]["plan_experiment"], 0)
        self.assertIn("plan_experiment", after_failure["available_tools"])
        after_success = provider.next_contexts[2]
        self.assertEqual(after_success["tool_counts"]["plan_experiment"], 2)
        self.assertEqual(after_success["accepted_revision_counts"]["plan_experiment"], 1)
        self.assertNotIn("plan_experiment", after_success["available_tools"])
        self.assertEqual(self.budget.snapshot()["worker_seconds"]["used_or_reserved"], 2)

    def test_repeated_rejections_remain_bounded_by_unchanged_total_steps_and_charges(self):
        from symplex.agents.policies import DEPTHS
        steps = DEPTHS["focused"]["steps"]
        provider = SequenceProvider([action("plan_experiment") for _ in range(steps)])
        def reject(name, context, proposal, permissions):
            reservation = context.budget.reserve(worker_seconds=1)
            context.budget.settle(reservation, worker_seconds=1)
            raise Invalid("No valid frozen contract produced")
        with patch.object(REGISTRY, "execute", side_effect=reject) as execute:
            result = solve(self.store, self.budget, provider, self.problem, depth="focused")
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(execute.call_count, steps - 1)
        self.assertEqual(len(provider.next_contexts), steps)
        self.assertEqual(self.budget.snapshot()["worker_seconds"]["used_or_reserved"], steps - 1)
        self.assertEqual(self.store.list("agent_action")[-1]["data"]["tool"], "deliver")

    def test_executed_candidate_with_failed_numerics_still_consumes_revision_cap(self):
        from symplex.agents.policies import DEPTHS
        from symplex.agents.solver import limit_reason
        events = [{"tool": "run_model_code", "result": {"status": "failed", "package_id": "recorded-candidate"}}]
        self.assertIsNotNone(limit_reason("run_model_code", events, DEPTHS["focused"]))


if __name__ == "__main__":
    unittest.main()
