"""Governance uses actual offline replay artifacts; no fabricated evaluation reports."""

import unittest
from unittest.mock import patch

from symplex.agents.prompts import prompt, snapshot
from symplex.agents.solver import solve
from symplex.agents.tools import REGISTRY
from symplex.core.contracts import Invalid
from symplex.evaluation.methods import evaluate_method, freeze_method_evaluation
from symplex.infrastructure.method_registry import activate_canary, gates, lab_status, operator_review, rollback, shadow
from tests.test_investigation import SequenceProvider, action
from tests import test_methods as method_fixtures
from tests.test_solver_revision import RecordingProvider


class MethodRegistryTests(unittest.TestCase):
    setUp = method_fixtures.MethodEvaluationTests.setUp
    create_candidate = method_fixtures.MethodEvaluationTests.create_candidate
    factory = method_fixtures.MethodEvaluationTests.factory
    task = method_fixtures.MethodEvaluationTests.task

    def stage(self, name, *, mode="improve", usd=0.1, account=True):
        protocol = freeze_method_evaluation(self.store, self.candidate, stage=name, usd_per_arm=usd, requests_per_arm=1)
        task = self.task(protocol, question="Fresh " + name + " problem " + protocol)
        return evaluate_method(self.store, self.budget, self.factory(mode=mode, account=account), protocol, [task])

    def staged(self):
        return {name: self.stage(name) for name in ("development", "regression", "holdout")}

    def review(self, reports=None, **changes):
        body = {"candidate_id": self.candidate, "decision": "approve", "actor": "Offline operator",
                "reason": "Review this contract-only canary", "report_ids": reports or {}, "problem_ids": [self.origin],
                "execution_failure_limit": 1, "numerical_failure_limit": 1}
        body.update(changes)
        return operator_review(self.store, body)

    def activate(self, review):
        return activate_canary(self.store, {"review_id": review["id"], "actor": "Offline operator", "reason": "Begin named-problem canary"})

    def test_missing_stages_block_approval_and_agent_registry_has_no_approval_tool(self):
        report = self.stage("development")
        evaluation = gates(self.store, self.candidate)
        self.assertFalse(evaluation["eligible"])
        self.assertEqual(evaluation["report_ids"], {"development": report})
        review = self.review()
        self.assertEqual(review["status"], "blocked")
        with self.assertRaises(Invalid):
            self.activate(review)
        self.assertFalse(set(REGISTRY.descriptions()) & {"approve_method", "activate_canary", "operator_review", "method_canary"})
        self.assertFalse(self.store.list("method_deployment"))

    def test_staged_replay_requires_gain_no_regressions_and_matching_caps(self):
        self.stage("development")
        self.stage("regression", mode="regress")
        self.stage("holdout", usd=0.2)
        evaluation = gates(self.store, self.candidate)
        self.assertFalse(evaluation["eligible"])
        self.assertTrue(any("regression" in b for b in evaluation["blockers"]))
        self.assertTrue(any("settings differ" in b for b in evaluation["blockers"]))

    def test_later_favorable_stage_cannot_replace_first_consumed_failure(self):
        self.stage("development")
        self.stage("regression")
        failed = self.stage("holdout", mode="regress")
        later = self.stage("holdout")
        self.assertEqual(gates(self.store, self.candidate)["report_ids"]["holdout"], failed)
        evaluation = gates(self.store, self.candidate, {"holdout": later})
        self.assertFalse(evaluation["eligible"])
        self.assertIn("first frozen", evaluation["stages"]["holdout"]["error"])

    def test_stages_frozen_before_prior_results_are_blocked(self):
        self.stage("development")
        regression_protocol = freeze_method_evaluation(self.store, self.candidate, stage="regression", usd_per_arm=0.1, requests_per_arm=1)
        holdout_protocol = freeze_method_evaluation(self.store, self.candidate, stage="holdout", usd_per_arm=0.1, requests_per_arm=1)
        regression_task = self.task(regression_protocol, question="Fresh regression chronology task")
        evaluate_method(self.store, self.budget, self.factory(), regression_protocol, [regression_task])
        holdout_task = self.task(holdout_protocol, question="Fresh held-out chronology task")
        evaluate_method(self.store, self.budget, self.factory(), holdout_protocol, [holdout_task])
        result = gates(self.store, self.candidate)
        self.assertFalse(result["eligible"])
        self.assertIn("preceding stage", result["stages"]["holdout"]["error"])

    def test_holdout_without_gain_and_exact_candidate_clone_cannot_pass(self):
        self.stage("development")
        self.stage("regression")
        self.stage("holdout", mode="equal")
        result = gates(self.store, self.candidate)
        self.assertFalse(result["eligible"])
        self.assertIn("gain", result["stages"]["holdout"]["error"])
        original = self.candidate
        self.candidate = self.create_candidate()
        self.assertEqual(self.store.get(original)["digest"], self.store.get(self.candidate)["digest"])
        later = self.staged()
        self.assertFalse(gates(self.store, self.candidate, later)["eligible"])

    def test_missing_accounting_and_unavailable_selected_report_produce_audited_block(self):
        self.stage("development", account=False)
        result = self.review({"holdout": "missing_report"})
        self.assertEqual(result["status"], "blocked")
        data = self.store.get(result["id"])["data"]
        self.assertIsNone(data["report_digests"]["holdout"])
        self.assertTrue(any("accounting" in b for b in result["gates"]["blockers"]))

    def test_shadow_leaves_baseline_prompts_and_job_identity_unchanged(self):
        provider = RecordingProvider()
        baseline = solve(self.store, self.budget, provider, self.origin)
        before = snapshot()
        record = shadow(self.store, {"candidate_id": self.candidate, "actor": "Reviewer", "reason": "Observe candidate only", "problem_ids": [self.origin]})
        again = solve(self.store, self.budget, provider, self.origin)
        self.assertEqual(baseline["id"], again["id"])
        self.assertEqual(snapshot(), before)
        self.assertEqual(lab_status(self.store, self.candidate)["state"], "shadow")
        self.assertFalse(record["affects_solver"])
        self.assertIsNone(self.store.list("run_manifest")[-1]["data"]["method_version"])

    def test_approved_canary_applies_only_to_named_new_invocations_and_rolls_back(self):
        reports = self.staged()
        self.assertTrue(gates(self.store, self.candidate)["eligible"])
        baseline = solve(self.store, self.budget, RecordingProvider(), self.origin)
        review = self.review(reports)
        self.assertEqual(review["status"], "approved")
        before = snapshot()
        deployment = self.activate(review)
        observed = []
        class InspectingProvider(RecordingProvider):
            def propose(inner_self, *args, **kwargs):
                observed.append(prompt("complexity_architect"))
                return super().propose(*args, **kwargs)
        result = solve(self.store, self.budget, InspectingProvider(), self.origin)
        self.assertNotEqual(result["id"], baseline["id"])
        self.assertTrue(all("Evaluated method addition:" in p for p in observed))
        self.assertEqual(snapshot(), before)
        manifest = self.store.list("run_manifest")[-1]["data"]
        self.assertEqual(manifest["method_version"]["deployment_id"], deployment["id"])
        other = self.store.put("workspace_problem", {"question": "Outside approved scope"})
        observed.clear()
        solve(self.store, self.budget, InspectingProvider(), other)
        self.assertTrue(all("Evaluated method addition:" not in p for p in observed))
        rollback(self.store, {"deployment_id": deployment["id"], "actor": "Reviewer", "reason": "End the trial"})
        restored = solve(self.store, self.budget, RecordingProvider(), self.origin)
        self.assertEqual(restored["id"], baseline["id"])
        self.assertEqual(len(self.store.list("method_rollback")), 1)
        self.assertFalse(self.store.get(self.candidate)["data"]["installed"])

    def test_host_numerical_failure_triggers_rollback_without_changing_live_snapshot(self):
        deployment = self.activate(self.review(self.staged()))
        provider = SequenceProvider([action("inspect_context"), action("deliver")])
        observed = []
        def fail_check(name, ctx, chosen, permissions):
            self.store.put("execution_assessment", {"status": "failed"}, self.origin)
            observed.append(prompt("complexity_architect"))
            return {"status": "succeeded"}
        with patch.object(REGISTRY, "execute", side_effect=fail_check):
            result = solve(self.store, self.budget, provider, self.origin)
        self.assertEqual(result["status"], "succeeded")
        self.assertIn("Evaluated method addition:", observed[0])
        rollback_record = self.store.list("method_rollback")[-1]
        self.assertEqual(rollback_record["data"]["deployment_id"], deployment["id"])
        self.assertEqual(rollback_record["data"]["trigger"], "failure_threshold")
        self.assertEqual(rollback_record["data"]["actor"], "host")
        self.assertFalse(self.store.list("method_canary_observation")[-1]["data"]["execution_failed"])

    def test_failed_resume_of_observed_waiting_canary_is_counted_and_rolled_back(self):
        self.activate(self.review(self.staged()))
        first = solve(self.store, self.budget, SequenceProvider([action("ask_user")]), self.origin)
        self.assertEqual(first["status"], "waiting_user")
        self.store.put("steering", {"instruction": "Resume this computation"}, self.origin)
        with patch.object(REGISTRY, "execute", side_effect=RuntimeError("Offline execution failure")):
            resumed = solve(self.store, self.budget, SequenceProvider([action("inspect_context")]), self.origin)
        self.assertEqual(resumed["id"], first["id"])
        self.assertEqual(resumed["status"], "failed")
        observed = self.store.list("method_canary_observation")
        self.assertEqual(len(observed), 2)
        self.assertTrue(observed[-1]["data"]["execution_failed"])
        self.assertEqual(self.store.list("method_rollback")[-1]["data"]["trigger"], "failure_threshold")

    def test_scope_expansion_superseded_review_and_stale_evidence_cannot_activate(self):
        review = self.review(self.staged())
        rejected = self.review(decision="reject", problem_ids=[])
        self.assertEqual(rejected["status"], "rejected")
        with self.assertRaises(Invalid):
            self.activate(review)
        newer = self.review()
        holdout = self.store.get(newer["id"])["data"]["report_ids"]["holdout"]
        self.store.invalidate(holdout)
        with self.assertRaises(Invalid):
            self.activate(newer)
        self.assertFalse(self.store.list("method_deployment"))

    def test_canary_api_is_explicit_token_protected_and_missing_evidence_stays_blocked(self):
        from fastapi.testclient import TestClient
        from symplex.interfaces.api import create_app
        client = TestClient(create_app(self.tmp.name))
        headers = {"X-Symplex-Token": client.get("/api/state").json()["token"]}
        status = client.get("/api/method-lab/" + self.candidate)
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json()["state"], "blocked")
        body = {"candidate_id": self.candidate, "decision": "approve", "actor": "Reviewer", "reason": "Inspect missing gates", "problem_ids": [self.origin]}
        self.assertEqual(client.post("/api/method-lab/review", json=body).status_code, 403)
        recorded = client.post("/api/method-lab/review", json=body, headers=headers)
        self.assertEqual(recorded.status_code, 200)
        self.assertEqual(recorded.json()["status"], "blocked")
        self.assertEqual(client.post("/api/method-lab/canary", json={"review_id": recorded.json()["id"], "actor": "Reviewer", "reason": "Try blocked review"}, headers=headers).status_code, 400)
        self.assertFalse(self.calls)
