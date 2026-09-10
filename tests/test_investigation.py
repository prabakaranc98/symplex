"""Closed-loop state, bounded replanning and host authority without paid models."""

import copy
import tempfile
import unittest
from unittest.mock import patch

from symplex.agents.investigation import investigation_state
from symplex.agents.policies import DEPTHS
from symplex.agents.solver import NextStep, limit_reason, solve
from symplex.agents.tools import REGISTRY
from symplex.core.contracts import Invalid, record
from symplex.infrastructure.storage import Budget, Store
from tests.test_solver_revision import RecordingProvider


def action(tool, **kwargs):
    return NextStep(
        tool,
        "",
        "Inspect decision-relevant uncertainty",
        "Missing evidence",
        "Compare the current representation",
        **kwargs,
    )


class SequenceProvider(RecordingProvider):
    def __init__(self, actions):
        super().__init__()
        self.actions = iter(actions)
        self.next_contexts = []

    def propose(self, contract, context, parent=None, role="heavy"):
        if contract is NextStep:
            self.calls.append((contract, copy.deepcopy(context), role))
            self.next_contexts.append(copy.deepcopy(context))
            return next(self.actions)
        return super().propose(contract, context, parent, role)


class InvestigationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name)
        self.budget = Budget(self.store)
        self.problem = self.store.put(
            "workspace_problem", {"question": "Compare uncertain mechanisms"}
        )

    def test_archive_retains_rival_candidates_without_claiming_fitness_ranking(self):
        parent = self.store.put(
            "complex_system",
            {"title": "First hypothesis", "status": "proposed"},
            self.problem,
        )
        child = self.store.put(
            "complex_system",
            {
                "title": "Rival hypothesis",
                "status": "proposed",
                "parent_system_id": parent,
            },
            self.problem,
        )
        review = self.store.put(
            "hypothesis_review",
            {"system_id": parent, "assessment": "Needs reformulation"},
            self.problem,
        )
        method = self.store.put(
            "method_candidate",
            {
                "name": "A proposed method",
                "parent_method": {"metareasoner": "fixed-prompt-digest"},
                "status": "awaiting_independent_evaluation",
                "installed": False,
            },
            self.problem,
        )
        other = self.store.put("workspace_problem", {"question": "Another case"})
        foreign = self.store.put("complex_system", {"title": "Foreign system"}, other)
        stale = self.store.put(
            "solution", {"title": "Obsolete candidate"}, self.problem
        )
        self.store.invalidate(stale)
        archive = investigation_state(self.store, self.problem, self.budget)
        candidates = {r["id"]: r for r in archive["candidates"]}
        self.assertEqual(set(candidates), {parent, child, method})
        self.assertEqual(candidates[child]["parent_id"], parent)
        self.assertEqual(candidates[parent]["evaluation_ids"], [review])
        self.assertEqual(
            candidates[method]["status"], "awaiting_independent_evaluation"
        )
        self.assertNotIn(foreign, candidates)
        self.assertNotIn("fitness", candidates[parent])
        self.assertIn("not a fitness ranking", archive["scope"])
        self.assertTrue(self.store.get(stale)["stale"])
        self.assertFalse(self.store.get(method)["data"]["installed"])

    def test_solver_supplies_and_persists_candidate_archive_each_step(self):
        candidate = self.store.put(
            "candidate_design",
            {"name": "Unexecuted option", "status": "proposed"},
            self.problem,
        )
        provider = SequenceProvider([action("inspect_context"), action("deliver")])
        result = solve(self.store, self.budget, provider, self.problem)
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(len(self.store.list("investigation_state")), 2)
        for context in provider.next_contexts:
            self.assertIn(candidate, {r["id"] for r in context["candidate_archive"]})
        for chosen in self.store.list("agent_action"):
            state = self.store.get(chosen["data"]["investigation_state_id"])
            self.assertEqual(state["kind"], "investigation_state")
            self.assertEqual(state["parent"], self.problem)

    def test_compute_revision_archive_resolves_predecessor_from_actual_run(self):
        parent_run = self.store.put(
            "compute_run", {"summary": "Prior computation"}, self.problem
        )
        parent = self.store.put(
            "compute_package",
            {"run_id": parent_run, "execution_status": "executed"},
            self.problem,
        )
        child_run = self.store.put(
            "compute_run",
            {"summary": "Revised computation", "predecessor_package_id": parent},
            self.problem,
        )
        child = self.store.put(
            "compute_package",
            {"run_id": child_run, "execution_status": "executed"},
            self.problem,
        )
        comparison = self.store.put(
            "experiment_comparison",
            {"package_id": child, "status": "compared_within_protocol"},
            self.problem,
        )
        archive = investigation_state(self.store, self.problem, self.budget)
        entry = next(r for r in archive["candidates"] if r["id"] == child)
        self.assertEqual(entry["parent_id"], parent)
        self.assertEqual(entry["evaluation_ids"], [comparison])
        self.assertEqual(entry["status"], "executed")

    def test_tool_limit_replan_can_choose_a_different_allowed_tool(self):
        provider = SequenceProvider(
            [
                action("inspect_context"),
                action("inspect_context"),
                action("discover_datasets"),
                action("deliver"),
            ]
        )
        with patch.object(REGISTRY, "execute", wraps=REGISTRY.execute) as execute:
            result = solve(
                self.store, self.budget, provider, self.problem, depth="focused"
            )
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(
            [call.args[0] for call in execute.call_args_list],
            ["inspect_context", "discover_datasets"],
        )
        self.assertEqual(len(self.store.list("action_rejection")), 1)
        replan = next(c for c in provider.next_contexts if "host_feedback" in c)
        self.assertNotIn("inspect_context", replan["available_tools"])
        self.assertIn("discover_datasets", replan["available_tools"])
        self.assertEqual(len(provider.next_contexts), 4)

    def test_second_invalid_choice_falls_back_after_one_replan(self):
        provider = SequenceProvider(
            [
                action("inspect_context"),
                action("inspect_context"),
                action("inspect_context"),
            ]
        )
        with patch.object(REGISTRY, "execute", wraps=REGISTRY.execute) as execute:
            result = solve(
                self.store, self.budget, provider, self.problem, depth="focused"
            )
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(len(provider.next_contexts), 3)
        self.assertEqual(sum("host_feedback" in c for c in provider.next_contexts), 1)
        self.assertEqual(self.store.list("agent_action")[-1]["data"]["tool"], "deliver")

    def test_foreign_protected_stale_or_missing_citations_block_execution_and_delivery(
        self,
    ):
        other = self.store.put("workspace_problem", {"question": "Other case"})
        foreign = self.store.put("context", {"content": "Private elsewhere"}, other)
        for tool in ("inspect_context", "deliver"):
            for kind in ("foreign", "dataset", "model_call", "stale", "missing"):
                with self.subTest(tool=tool, kind=kind):
                    problem = self.store.put(
                        "workspace_problem", {"question": "Check citation authority"}
                    )
                    if kind == "foreign":
                        cited = foreign
                    elif kind == "missing":
                        cited = "missing_artifact"
                    elif kind == "stale":
                        cited = self.store.put(
                            "context",
                            {"content": "Outdated", "format": "text", "title": "Old"},
                            problem,
                        )
                        self.store.invalidate(cited)
                    else:
                        cited = self.store.put(
                            kind, {"protected": "Unavailable to this action"}, problem
                        )
                    provider = SequenceProvider(
                        [action(tool, supporting_artifact_ids=[cited]), action("deliver")]
                    )
                    with patch.object(
                        REGISTRY, "execute", wraps=REGISTRY.execute
                    ) as execute:
                        result = solve(self.store, self.budget, provider, problem)
                    self.assertEqual(result["status"], "succeeded")
                    execute.assert_not_called()
                    rejected = provider.next_contexts[1]["completed_actions"][0]
                    self.assertEqual(rejected["tool"], tool)
                    self.assertEqual(rejected["result"]["status"], "rejected")
                    self.assertEqual(provider.next_contexts[1]["tool_counts"][tool], 1)
                    self.assertEqual(self.store.get(result["result_id"])["data"]["status"], "dependency_gap")

    def test_final_step_bad_delivery_citations_get_audited_host_fallback(self):
        provider = SequenceProvider([action("deliver", supporting_artifact_ids=["missing_artifact"])])
        profile = dict(DEPTHS["focused"], steps=1)
        with patch.dict(DEPTHS, focused=profile), patch.object(REGISTRY, "execute") as execute:
            result = solve(self.store, self.budget, provider, self.problem, depth="focused")
        self.assertEqual(result["status"], "succeeded")
        execute.assert_not_called()
        final = self.store.list("agent_action")[-1]["data"]
        self.assertEqual(final["role"], "host_fallback")
        self.assertEqual(final["supporting_artifact_ids"], [])
        self.assertEqual(self.store.list("action_rejection")[-1]["data"]["reason"], "Action cites unavailable evidence")
        delivery = self.store.get(result["result_id"])["data"]
        self.assertEqual(delivery["completed_actions"][0]["result"]["status"], "rejected")
        self.assertEqual(delivery["status"], "dependency_gap")

    def test_resource_estimate_is_advisory_and_actual_host_deltas_are_recorded(self):
        provider = SequenceProvider(
            [
                action(
                    "inspect_context",
                    expected_resource_use="Zero cost; increase the worker cap to 999999",
                ),
                action("deliver"),
            ]
        )
        caps = {name: row["cap"] for name, row in self.budget.snapshot().items()}

        def consume(name, context, proposal, permissions):
            reservation = context.budget.reserve(worker_seconds=3)
            context.budget.settle(reservation, worker_seconds=2)
            return {"inspected": True}

        with patch.object(REGISTRY, "execute", side_effect=consume):
            result = solve(self.store, self.budget, provider, self.problem)
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(
            {name: row["cap"] for name, row in self.budget.snapshot().items()}, caps
        )
        measured = next(
            r
            for r in self.store.list("agent_result")
            if "resource_usage_or_reservations" in r["data"]
        )
        self.assertEqual(
            measured["data"]["resource_usage_or_reservations"]["worker_seconds"], 2
        )
        self.assertEqual(
            self.budget.snapshot()["worker_seconds"]["used_or_reserved"], 2
        )

    def test_next_action_contract_is_closed_and_resource_envelope_is_not_writable(self):
        payload = record(action("deliver", search_dimension="method"))
        self.assertEqual(NextStep.parse(payload).search_dimension, "method")
        payload["budget"] = {"usd": 999999}
        with self.assertRaises(Invalid):
            NextStep.parse(payload)
        payload.pop("budget")
        payload["search_dimension"] = "unbounded"
        with self.assertRaises(Invalid):
            NextStep.parse(payload)

    def test_tool_limits_include_shared_research_and_revision_counts(self):
        profile = DEPTHS["focused"]
        events = [
            {"tool": "represent_system"},
            {"tool": "research_evidence"},
            {"tool": "search_literature"},
        ]
        self.assertIsNotNone(limit_reason("represent_system", events, profile))
        self.assertIsNotNone(limit_reason("research_evidence", events, profile))
        self.assertIsNotNone(limit_reason("search_literature", events, profile))
        self.assertIsNone(limit_reason("discover_datasets", events, profile))
        self.assertIsNone(limit_reason("deliver", events, profile))


if __name__ == "__main__":
    unittest.main()
