"""Workbench and CLI entrypoints retain allocations and exclusive run ownership."""

import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from scripts.run_case_campaign import campaign_lock
from symplex.agents.solver import solve
from symplex.core.contracts import Invalid, digest
from symplex.infrastructure.execution_scope import ExecutionBusy
from symplex.infrastructure.scoped_budget import ScopedBudget
from symplex.infrastructure.storage import Budget, Store
from symplex.interfaces.api import create_app
from tests.test_solver_revision import RecordingProvider


class ChargingProvider(RecordingProvider):
    def __init__(self, store, budget, pause=False):
        super().__init__(pause_for_steering=pause)
        self.store, self.budget = store, budget

    def propose(self, *args, **kwargs):
        reservation = self.budget.reserve(usd=0.1, model_requests=1)
        self.budget.settle(reservation, usd=0.1, model_requests=1)
        return super().propose(*args, **kwargs)


class ExecutionScopeTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = Store(directory.name)
        self.budget = Budget(self.store, {"usd": 20})
        self.problem = self.store.put("workspace_problem", {"question": "Preserve the case allocation"})
        self.scope = ScopedBudget(self.budget, "arbitrary-campaign:arbitrary-case", {"usd": 2})
        self.store.put("campaign_attempt", {"problem_id": self.problem, "scope_id": self.scope.scope_id})
        self.key = patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test-key"})
        self.key.start()
        self.addCleanup(self.key.stop)
        self.client = TestClient(create_app(directory.name))
        self.headers = {"X-Symplex-Token": self.client.get("/api/state").json()["token"]}

    def post(self, route, body):
        return self.client.post(route, json=body, headers=self.headers)

    def wait_task(self, task_id):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            task = next(t for t in self.client.get("/api/state").json()["active"] if t["id"] == task_id)
            if task["status"] != "running":
                return task
            threading.Event().wait(0.01)
        self.fail("Offline background task did not finish")

    def test_direct_global_solver_reopens_scope_and_preserves_provider_binding(self):
        provider = ChargingProvider(self.store, self.budget)
        result = solve(self.store, self.budget, provider, self.problem)
        self.assertEqual(result["status"], "succeeded")
        self.assertAlmostEqual(self.scope.scope_snapshot()["usd"]["used_or_reserved"], 0.3)
        self.assertIs(provider.budget, self.budget)
        self.assertEqual(self.store.list("run_manifest")[-1]["data"]["budget_scope_id"], self.scope.scope_id)
        self.assertEqual(self.store.list("problem_budget_scope")[-1]["parent"], self.problem)

    def test_steering_queues_under_runner_lock_then_resumes_same_job_with_scope(self):
        provider = ChargingProvider(self.store, self.scope, pause=True)
        initial = solve(self.store, self.scope, provider, self.problem)
        self.assertEqual(initial["status"], "waiting_user")
        seen = []
        def factory(store, budget):
            seen.append(budget)
            return ChargingProvider(store, budget)
        case_lock = "campaign-case-" + digest(self.scope.scope_id) + ".lock"
        with patch("symplex.interfaces.api.OpenAIProvider", side_effect=factory):
            with campaign_lock(self.store, case_lock):
                queued = self.post("/api/steer", {"problem_id": self.problem, "instruction": "Use the measured recovery time"})
                self.assertEqual(queued.status_code, 200)
                self.assertEqual(queued.json()["resume"], "queued_for_scope_owner")
                self.assertFalse(seen)
                self.assertEqual(len(self.store.list("steering")), 1)
            resumed = self.post("/api/steer", {"problem_id": self.problem, "instruction": "Continue within the original allocation"})
            self.assertEqual(resumed.status_code, 200)
            task = self.wait_task(resumed.json()["task_id"])
        self.assertEqual(task["status"], "succeeded")
        self.assertEqual(task["result"]["id"], initial["id"])
        self.assertEqual([s.scope_id for s in seen], [self.scope.scope_id])
        self.assertAlmostEqual(self.scope.scope_snapshot()["usd"]["used_or_reserved"], 0.4)
        self.assertEqual(len(self.store.list("problem_dna")), 1)

    def test_scoped_plan_and_compute_use_the_same_provider_and_resource_allocation(self):
        def factory(store, budget):
            return ChargingProvider(store, budget)
        def compute(store, budget, provider, problem_id, instruction, **kwargs):
            self.assertEqual(budget.scope_id, self.scope.scope_id)
            self.assertIs(provider.budget, budget)
            reservation = budget.reserve(worker_seconds=1)
            budget.settle(reservation, worker_seconds=1)
            return "offline-compute-result"
        with patch("symplex.interfaces.api.OpenAIProvider", side_effect=factory), patch("symplex.connectors.compute.run", side_effect=compute):
            planned = self.post("/api/plan", {"id": self.problem})
            self.assertEqual(planned.status_code, 200, planned.text)
            started = self.post("/api/compute", {"id": self.problem})
            self.assertEqual(started.status_code, 200, started.text)
            self.assertEqual(self.wait_task(started.json()["task_id"])["status"], "succeeded")
        usage = self.scope.scope_snapshot()
        self.assertAlmostEqual(usage["usd"]["used_or_reserved"], 0.1)
        self.assertEqual(usage["worker_seconds"]["used_or_reserved"], 1)

    def test_exhausted_scope_cannot_borrow_global_capacity_through_api(self):
        self.scope.reserve(usd=2)
        with patch("symplex.interfaces.api.OpenAIProvider", side_effect=lambda store, budget: ChargingProvider(store, budget)):
            response = self.post("/api/plan", {"id": self.problem})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.budget.snapshot()["usd"]["used_or_reserved"], 2)
        self.assertFalse(self.store.list("problem_dna"))

    def test_external_case_owner_blocks_direct_solve_and_synchronous_api_work(self):
        with campaign_lock(self.store, "campaign-case-" + digest(self.scope.scope_id) + ".lock"):
            # Different thread has no in-process ownership token, like the API worker.
            errors = []
            def attempt():
                try:
                    solve(self.store, self.budget, ChargingProvider(self.store, self.budget), self.problem)
                except ExecutionBusy as error:
                    errors.append(error)
            worker = threading.Thread(target=attempt)
            worker.start()
            worker.join(2)
            with patch("symplex.interfaces.api.OpenAIProvider") as factory:
                response = self.post("/api/plan", {"id": self.problem})
                factory.assert_not_called()
        self.assertEqual(len(errors), 1)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.budget.snapshot()["usd"]["used_or_reserved"], 0)

    def test_missing_and_conflicting_scopes_fail_closed(self):
        orphan = self.store.put("workspace_problem", {"question": "Missing allocation", "campaign_id": "campaign"})
        with self.assertRaises(Invalid):
            solve(self.store, self.budget, RecordingProvider(), orphan)
        other = ScopedBudget(self.budget, "another-case")
        with self.assertRaises(Invalid):
            solve(self.store, other, RecordingProvider(), self.problem)
        self.assertFalse(self.store.jobs())
