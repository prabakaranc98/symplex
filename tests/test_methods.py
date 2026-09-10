"""Offline paired method replay proves boundaries, not model or scientific improvement."""

import copy
import json
import tempfile
import unittest

from symplex.agents.context import working_context
from symplex.agents.improvement import propose_method
from symplex.agents.prompts import manifest, prompt, prompt_overlay, snapshot
from symplex.connectors.retrieval import search
from symplex.core.contracts import Invalid
from symplex.evaluation.methods import evaluate_method, freeze_method_evaluation
from symplex.infrastructure.storage import Budget, Store
from tests.test_complex_system import system_spec
from tests.test_improvement import FakeProposer, method_spec


class ReplayProvider:
    def __init__(self, store, budget, calls, mode="improve", account=True, extra_request=False, settle=True):
        self.store, self.budget, self.calls = store, budget, calls
        self.models = {"heavy": "offline-fake"}
        self.mode, self.account, self.extra_request = mode, account, extra_request
        self.settle = settle

    def propose(self, contract, context, parent=None, role="heavy"):
        assert self.store.list("method_task_exposure"), "Exposure must precede the first call"
        self.calls.append({"context": copy.deepcopy(context), "parent": parent,
                           "instruction": prompt("complexity_architect"), "budget": self.budget})
        child = "Evaluated method addition:" in context["instruction"]
        if self.account:
            ident = self.budget.reserve(model_requests=1, input_tokens=100, output_tokens=100, usd=0.02)
            if self.settle:
                self.budget.settle(ident, model_requests=1, input_tokens=80, output_tokens=50, usd=0.01)
        if self.extra_request:
            self.budget.reserve(model_requests=1)
        if self.mode == "exception" and not child:
            raise RuntimeError("Recorded provider failure")
        result = system_spec()
        ids = [r["id"] for r in context["artifacts"]]
        for collection, field in ((result["observations"], "source_artifact_ids"),
                                  (result["components"], "evidence_ids"),
                                  (result["hypotheses"], "evidence_ids"),
                                  (result["validation"]["checks"], "evidence_ids")):
            for item in collection:
                item[field] = ids if item[field] else []
        if (self.mode == "improve" and not child) or (self.mode == "regress" and child):
            result["components"][0]["ports"][0]["scale_id"] = "undeclared_clock"
        if self.mode == "bad_reference":
            result["observations"][0]["source_artifact_ids"] = ["invented_source"]
        return result


class MethodEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name)
        self.budget = Budget(self.store, {"usd": 10})
        self.origin = self.store.put("workspace_problem", {"question": "Original development question"})
        self.feedback = self.store.put("model_critique", {"issue": "ORIGIN_FEEDBACK_SECRET clock mismatch"}, self.origin)
        self.candidate = self.create_candidate()
        self.calls = []

    def create_candidate(self, role="complexity_architect"):
        value = method_spec(self.feedback)
        value["changes"][0].update(role=role, proposed_instruction="Check each port clock against its declared state before returning.")
        return propose_method(self.store, FakeProposer(value), self.origin)

    def freeze(self, **kwargs):
        return freeze_method_evaluation(self.store, self.candidate, **kwargs)

    def task(self, protocol_id, question="FRESH_HOLDOUT_TASK compare a new system", source_text="Fresh task evidence"):
        problem = self.store.put("workspace_problem", {"question": question})
        source = self.store.put("context", {"title": "Held-out input", "format": "text", "content": source_text}, problem)
        task = self.store.put("method_task", {"problem_id": problem, "context_artifact_ids": [source]}, protocol_id)
        return task

    def factory(self, **kwargs):
        return lambda budget: ReplayProvider(self.store, budget, self.calls, **kwargs)

    def test_paired_replay_freezes_inputs_scores_contracts_records_cost_and_preserves_prompts(self):
        before, candidate = snapshot(), self.store.get(self.candidate)
        protocol = self.freeze(usd_per_arm=0.1, requests_per_arm=1)
        tasks = [self.task(protocol, question="Fresh system one"), self.task(protocol, question="Fresh system two")]
        report_id = evaluate_method(self.store, self.budget, self.factory(), protocol, tasks)
        report = self.store.get(report_id)["data"]
        self.assertEqual(report["status"], "contract_reliability_improved_on_replay")
        self.assertEqual((report["parent_passes"], report["child_passes"]), (0, 2))
        self.assertTrue(report["matched_resources_verified"])
        self.assertFalse(report["installed"])
        self.assertFalse(report["promotion_allowed"])
        self.assertIn("no scientific truth", report["scope"])
        self.assertEqual(snapshot(), before)
        self.assertEqual(self.store.get(self.candidate), candidate)
        self.assertAlmostEqual(self.budget.snapshot()["usd"]["used_or_reserved"], 0.04)
        self.assertEqual(len(self.store.list("method_evaluation_run")), 4)
        instructions = [call["instruction"] for call in self.calls]
        self.assertEqual(["Evaluated method addition:" in text for text in instructions], [False, True, True, False])
        for call in self.calls:
            self.assertEqual(call["context"]["instruction"], call["instruction"])
            self.assertNotIn("ORIGIN_FEEDBACK_SECRET", json.dumps(call["context"]))
            self.assertNotIn("prior_problem_dna", call["context"])
            self.assertEqual(call["budget"].scope_snapshot()["model_requests"]["cap"], 1)
            self.assertEqual(call["budget"].scope_snapshot()["usd"]["cap"], 0.1)
        self.assertEqual(len(self.store.list("method_task_exposure")), 2)

    def test_unsupported_roles_and_parent_prompt_drift_reject_before_replay(self):
        other = self.create_candidate("metareasoner")
        with self.assertRaisesRegex(Invalid, "complexity_architect"):
            freeze_method_evaluation(self.store, other)
        with prompt_overlay({"complexity_architect": "A different parent method"}):
            with self.assertRaisesRegex(Invalid, "parent prompt digests"):
                self.freeze()
        self.assertFalse(self.calls)

    def test_origin_old_tasks_foreign_feedback_and_previously_investigated_problems_are_rejected(self):
        old_problem = self.store.put("workspace_problem", {"question": "Old independent problem"})
        protocol = self.freeze()
        invalid = [self.store.put("method_task", {"problem_id": self.origin, "context_artifact_ids": []}, protocol),
                   self.store.put("method_task", {"problem_id": old_problem, "context_artifact_ids": []}, protocol),
                   self.task(protocol, question=" ORIGINAL   development QUESTION ")]
        fresh = self.store.put("workspace_problem", {"question": "A distinct fresh problem"})
        invalid.append(self.store.put("method_task", {"problem_id": fresh, "context_artifact_ids": [self.feedback]}, protocol))
        investigated = self.task(protocol, question="Already investigated fresh task")
        self.store.put("agent_action", {"tool": "inspect_context"}, self.store.get(investigated)["data"]["problem_id"])
        invalid.append(investigated)
        for task_id in invalid:
            with self.subTest(task_id=task_id), self.assertRaises(Invalid):
                evaluate_method(self.store, self.budget, self.factory(), protocol, [task_id])
        self.assertFalse(self.store.list("method_task_exposure"))
        self.assertFalse(self.calls)

    def test_consumed_tasks_and_cloned_task_contents_cannot_be_reused(self):
        protocol = self.freeze()
        task = self.task(protocol)
        evaluate_method(self.store, self.budget, self.factory(), protocol, [task])
        count = len(self.calls)
        with self.assertRaises(Invalid):
            evaluate_method(self.store, self.budget, self.factory(), protocol, [task])
        another = self.freeze()
        clone = self.task(another)
        with self.assertRaises(Invalid):
            evaluate_method(self.store, self.budget, self.factory(), another, [clone])
        self.assertEqual(len(self.calls), count)

    def test_missing_usage_and_budget_failures_are_inconclusive(self):
        for index, options in enumerate(({"account": False}, {"extra_request": True}, {"settle": False})):
            with self.subTest(options=options):
                protocol = self.freeze(requests_per_arm=1)
                task = self.task(protocol, question=f"Resource case {index}")
                report = self.store.get(evaluate_method(self.store, self.budget, self.factory(**options), protocol, [task]))["data"]
                self.assertEqual(report["status"], "inconclusive_resource_accounting")
                self.assertFalse(report["matched_resources_verified"])
        statuses = [r["data"]["usage_status"] for r in self.store.list("method_evaluation_run")]
        self.assertIn("missing_usage_accounting", statuses)
        self.assertIn("unsettled_reservations", statuses)
        failures = [r["data"]["failure"] for r in self.store.list("method_evaluation_run") if r["data"]["failure"]]
        self.assertTrue(any(f["type"] == "BudgetExhausted" for f in failures))

    def test_deterministic_reference_and_regression_checks_preserve_failures(self):
        for index, mode in enumerate(("bad_reference", "regress", "exception")):
            protocol = self.freeze()
            task = self.task(protocol, question=f"Validation case {index}")
            report = self.store.get(evaluate_method(self.store, self.budget, self.factory(mode=mode), protocol, [task]))["data"]
            if mode == "bad_reference":
                self.assertEqual((report["parent_passes"], report["child_passes"]), (0, 0))
            elif mode == "regress":
                self.assertEqual(report["paired_regressions"], 1)
                self.assertEqual(report["status"], "no_contract_reliability_improvement")
        runs = self.store.list("method_evaluation_run")
        self.assertTrue(any(r["data"]["failure"] and r["data"]["failure"]["type"] == "RuntimeError" for r in runs))

    def test_protected_replay_details_never_enter_origin_working_context_or_retrieval(self):
        protocol = self.freeze()
        task = self.task(protocol)
        report = evaluate_method(self.store, self.budget, self.factory(), protocol, [task])
        visible = working_context(self.store, self.origin)
        retrieved = search(self.store, self.origin, "FRESH_HOLDOUT_TASK")
        self.assertNotIn("FRESH_HOLDOUT_TASK", json.dumps(visible))
        self.assertNotIn("FRESH_HOLDOUT_TASK", json.dumps(retrieved))
        protected = {r["id"] for kind in ("method_task", "method_evaluation_protocol", "method_task_exposure", "method_evaluation_run", "method_evaluation_report")
                     for r in self.store.list(kind)}
        self.assertTrue(all(self.store.get(ident)["parent"] != self.origin for ident in protected))
        aggregate = self.store.list("method_evaluation")[-1]
        self.assertEqual(aggregate["parent"], self.origin)
        self.assertEqual(aggregate["data"]["report_id"], report)
        self.assertNotIn("FRESH_HOLDOUT_TASK", json.dumps(aggregate["data"]))
        self.assertNotIn("output", aggregate["data"])

    def test_nested_overlay_restores_active_prompts_even_when_role_calls_fail(self):
        protocol = self.freeze()
        task = self.task(protocol)
        before = manifest()
        with prompt_overlay({"metareasoner": "Outer invocation instructions"}):
            outer = manifest()
            evaluate_method(self.store, self.budget, self.factory(mode="exception"), protocol, [task])
            self.assertEqual(manifest(), outer)
            self.assertEqual(prompt("metareasoner"), "Outer invocation instructions")
        self.assertEqual(manifest(), before)

    def test_provider_factory_cannot_drop_scope_or_change_models_between_arms(self):
        protocol = self.freeze()
        task = self.task(protocol)
        count = 0

        def changing_factory(budget):
            nonlocal count
            count += 1
            provider = ReplayProvider(self.store, budget, self.calls)
            provider.models = {"heavy": "model-" + str(count)}
            return provider

        with self.assertRaisesRegex(Invalid, "identical fixed model"):
            evaluate_method(self.store, self.budget, changing_factory, protocol, [task])
        self.assertFalse(self.calls)
        other = self.freeze()
        task = self.task(other, question="Bad scope task")
        with self.assertRaisesRegex(Invalid, "scoped budget"):
            evaluate_method(self.store, self.budget, lambda _: ReplayProvider(self.store, self.budget, self.calls), other, [task])
        self.assertFalse(self.calls)


if __name__ == "__main__":
    unittest.main()
