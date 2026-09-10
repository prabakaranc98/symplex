"""User steering, multimodal inventory and reviewer evidence without paid calls."""

import copy
import json
import tempfile
import unittest

from symplex.agents.context import review_context
from symplex.agents.solver import NextStep, solve
from symplex.connectors.compute import save_blob
from symplex.core.contracts import ProblemDNA
from symplex.core.proposals import DeliveryReview
from symplex.infrastructure.storage import Budget, Store


class RecordingProvider:
    def __init__(self, pause_for_steering=False):
        self.calls = []
        self.pause_for_steering = pause_for_steering

    def propose(self, contract, context, parent=None, role="heavy"):
        self.calls.append((contract, copy.deepcopy(context), role))
        if contract is ProblemDNA:
            return ProblemDNA(
                beneficiary="Decision owner",
                decision="Compare interventions",
                objective="Reduce uncertainty",
                actors=["System users"],
                constraints=["No measured outcomes available"],
                assumptions=["Scenario parameters are provisional"],
                missing_evidence=["Independent validation data"],
            )
        if contract is NextStep:
            tool = (
                "ask_user"
                if self.pause_for_steering and not context["user_steering"]
                else "deliver"
            )
            return NextStep(
                tool,
                "",
                "Identify the relevant outcome",
                "Objective is unclear",
                "Record the decision boundary",
            )
        if contract is DeliveryReview:
            return {
                "assessment": "Exploratory evidence only",
                "unsupported_claims": [],
                "next_validation": "Independent measurement",
            }
        raise AssertionError("Unexpected contract: " + contract.__name__)

    def contexts(self, contract):
        return [context for seen, context, role in self.calls if seen is contract]


class SolverRevisionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name)
        self.budget = Budget(self.store)
        self.problem = self.store.put(
            "workspace_problem",
            {"question": "Understand recovery after a perturbation"},
        )
        self.provider = RecordingProvider()

    def attach_context(self, text):
        return self.store.put(
            "context",
            {
                "title": "User observation",
                "format": "text",
                "content": text,
                "basis": "user_context",
            },
            self.problem,
        )

    def steer(self, instruction):
        return self.store.put(
            "steering", {"instruction": instruction, "kind": "judgment"}, self.problem
        )

    def execute(self, provider=None):
        return solve(self.store, self.budget, provider or self.provider, self.problem)

    def test_new_steering_after_success_creates_job_and_reframes_from_context(self):
        context_id = self.attach_context(
            "Recovery time matters more than maximum output"
        )
        first = self.execute()
        self.assertEqual(first["status"], "succeeded")
        original = self.store.get(first["result_id"])
        direction = self.steer("Prefer robust recovery under uncertain disturbances")
        second = self.execute()
        self.assertEqual(second["status"], "succeeded")
        self.assertNotEqual(first["id"], second["id"])
        frames = self.provider.contexts(ProblemDNA)
        self.assertEqual(len(frames), 2)
        supplied = {r["id"]: r for r in frames[-1]["supplied_context"]}
        self.assertIn(context_id, supplied)
        self.assertIn(direction, supplied)
        self.assertIn("robust recovery", supplied[direction]["data"]["instruction"])
        self.assertEqual(self.store.get(first["result_id"]), original)
        result = self.store.get(second["result_id"])["data"]
        self.assertEqual(result["steering_ids"], [direction])

    def test_unchanged_success_returns_same_job_without_model_calls(self):
        self.attach_context("A defined intervention boundary")
        self.steer("Measure robustness")
        first = self.execute()
        count = len(self.provider.calls)
        second = self.execute()
        self.assertEqual(second["id"], first["id"])
        self.assertEqual(second["result_id"], first["result_id"])
        self.assertEqual(len(self.provider.calls), count)
        self.assertEqual(len(self.store.jobs()), 1)

    def test_new_attached_context_changes_the_investigation_identity(self):
        first = self.execute()
        new_context = self.attach_context("Additional measurement: delayed recovery")
        second = self.execute()
        self.assertNotEqual(first["id"], second["id"])
        latest = self.provider.contexts(ProblemDNA)[-1]["supplied_context"]
        self.assertIn(new_context, {r["id"] for r in latest})

    def test_waiting_user_does_not_repeat_calls_and_steering_resumes_same_job(self):
        provider = RecordingProvider(pause_for_steering=True)
        first = self.execute(provider)
        self.assertEqual(first["status"], "waiting_user")
        count = len(provider.calls)
        waiting = self.execute(provider)
        self.assertEqual(waiting["id"], first["id"])
        self.assertEqual(waiting["status"], "waiting_user")
        self.assertEqual(len(provider.calls), count)
        direction = self.steer("Use time to recovery as the outcome")
        resumed = self.execute(provider)
        self.assertEqual(resumed["id"], first["id"])
        self.assertEqual(resumed["status"], "succeeded")
        self.assertEqual(len(self.store.jobs()), 1)
        self.assertEqual(len(self.store.list("resume")), 1)
        self.assertEqual(len(provider.contexts(ProblemDNA)), 1)
        self.assertIn(direction, {r["id"] for r in provider.contexts(NextStep)[-1]["user_steering"]})
        self.assertTrue(provider.contexts(NextStep)[-1]["input_delta"]["changed"])
        self.assertTrue(provider.contexts(NextStep)[-1]["problem_dna_basis"]["predates_current_inputs"])

    def test_binary_context_inventory_reaches_framer_and_planner(self):
        blob = save_blob(
            self.store,
            b"image fixture",
            "observed-system.png",
            self.problem,
            "user_context",
        )
        binary = self.store.put(
            "binary_context",
            {
                "title": "Observed diagram",
                "blob_id": blob,
                "filename": "observed-system.png",
                "format": "image/png",
            },
            self.problem,
        )
        self.execute()
        frame = self.provider.contexts(ProblemDNA)[0]["supplied_context"]
        next_step = self.provider.contexts(NextStep)[0]["working_context"]
        for context in (frame, next_step):
            entry = next(r for r in context if r["id"] == binary)
            self.assertEqual(entry["kind"], "binary_context")
            self.assertEqual(entry["data"]["blob_id"], blob)
            self.assertEqual(entry["digest"], self.store.get(binary)["digest"])

    def executed_package(self):
        run_id = self.store.put(
            "compute_run",
            {"summary": "Recorded synthetic execution", "calls": []},
            self.problem,
        )
        code = save_blob(
            self.store,
            b"# actual generated code\nprint(2 + 2)\n",
            "model.py",
            self.problem,
            "generated",
            run_id,
        )
        results = save_blob(
            self.store,
            b'{"basis":"synthetic","residual":0.02}',
            "results.json",
            self.problem,
            "generated",
            run_id,
        )
        package = self.store.put(
            "compute_package",
            {
                "run_id": run_id,
                "file_ids": [code, results],
                "summary": "Exploratory result",
                "file_failures": [],
            },
            self.problem,
        )
        return package, code, results

    def test_separate_delivery_reviewer_receives_actual_code_and_result_excerpts(self):
        package, code, results = self.executed_package()
        self.store.put(
            "dataset", {"secret": "protected confirmation labels"}, self.problem
        )
        self.store.put(
            "model_call", {"output": "private maker reasoning"}, self.problem
        )
        response = self.execute()
        self.assertEqual(response["status"], "succeeded")
        reviews = [
            (context, role)
            for contract, context, role in self.provider.calls
            if contract is DeliveryReview
        ]
        self.assertEqual(len(reviews), 1)
        evidence, role = reviews[0][0]["executed_evidence"], reviews[0][1]
        self.assertEqual(role, "validator")
        files = {r["id"]: r for r in evidence["executed_files"]}
        self.assertEqual(set(files), {code, results})
        self.assertIn("print(2 + 2)", files[code]["content"])
        self.assertEqual(json.loads(files[results]["content"])["residual"], 0.02)
        self.assertFalse(files[code]["truncated"])
        self.assertNotIn("protected confirmation labels", json.dumps(evidence))
        self.assertNotIn("private maker reasoning", json.dumps(evidence))
        self.assertIn(package, {r["id"] for r in evidence["artifacts"]})

    def test_review_context_marks_long_file_excerpts_and_skips_foreign_or_stale(self):
        run_id = self.store.put(
            "compute_run",
            {"summary": "Recorded synthetic execution", "calls": []},
            self.problem,
        )
        long_code = save_blob(
            self.store,
            b"# diagnostic\n" * 1000,
            "model.py",
            self.problem,
            "generated",
            run_id,
        )
        stale = save_blob(
            self.store,
            b'{"obsolete":true}',
            "stale.json",
            self.problem,
            "generated",
            run_id,
        )
        self.store.invalidate(stale)
        other = self.store.put("workspace_problem", {"question": "Different system"})
        foreign = save_blob(
            self.store, b'{"foreign":true}', "foreign.json", other, "generated", run_id
        )
        self.store.put(
            "compute_package",
            {
                "run_id": run_id,
                "file_ids": [long_code, stale, foreign],
                "summary": "Inspect bounded evidence",
                "file_failures": [],
            },
            self.problem,
        )
        evidence = review_context(self.store, self.problem, max_chars=8000)
        self.assertEqual([r["id"] for r in evidence["executed_files"]], [long_code])
        self.assertTrue(evidence["executed_files"][0]["truncated"])
        self.assertEqual(len(evidence["executed_files"][0]["content"]), 4000)


if __name__ == "__main__":
    unittest.main()
