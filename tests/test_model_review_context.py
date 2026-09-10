"""The reviewer sees complete primary state and the planner's requested critique."""

import json
import unittest
from types import SimpleNamespace

from symplex.agents.tools import ToolContext, review_model
from symplex.connectors.compute import save_blob
from symplex.core.contracts import canonical
from tests import test_request_context as request_fixtures
from tests.test_solver_revision import RecordingProvider


class ModelReviewContextTests(unittest.TestCase):
    setUp = request_fixtures.RequestContextTests.setUp

    def add_outputs(self):
        run = self.store.put("compute_run", {"calls": [{"status": "completed"}]}, self.problem)
        ids = [save_blob(self.store, b"# model source\n" * 1000, "model.py", self.problem, "generated", run),
               save_blob(self.store, b"time,value\n" + b"1,2\n" * 5000, "data.csv", self.problem, "generated", run)]
        self.store.put("compute_package", {"run_id": run, "file_ids": ids}, self.problem)
        return ids

    def test_real_wire_request_is_bounded_without_losing_system_or_requested_review(self):
        self.add_outputs()
        instruction = "Review whether delayed recovery changes the chosen intervention; do not widen the experiment."
        context = ToolContext(self.store, None, self.provider, self.problem, None, [], [])
        with self.assertRaises(request_fixtures.CapturedRequest):
            review_model(context, SimpleNamespace(instruction=instruction))
        request = self.requests[-1]
        payload = request["input"]
        text = payload if isinstance(payload, str) else payload[0]["content"]
        if isinstance(text, list):
            text = "".join(part["text"] for part in text if part.get("type") == "input_text")
        supplied = json.loads(text)
        self.assertEqual(supplied["system"], self.system_data)
        self.assertEqual(supplied["requested_review"], instruction)
        manifest = self.store.get(supplied["context_selection"]["manifest_id"])["data"]
        self.assertEqual(manifest["role"], "validator")
        self.assertIn(self.system, manifest["primary_artifact_ids"])
        self.assertLessEqual(len(canonical(request).encode("utf-8")) + 2048, manifest["max_request_bytes"])
        self.assertNotIn(self.system, {r["id"] for r in supplied["executed_evidence"]["artifacts"]})
        self.assertTrue(manifest["omitted"])

    def test_saved_inspection_ids_match_only_the_evidence_actually_supplied(self):
        self.add_outputs()
        provider = RecordingProvider()
        context = ToolContext(self.store, None, provider, self.problem, None, [], [])
        result = review_model(context, SimpleNamespace(instruction="Inspect the saved source assumptions"))
        evidence = provider.calls[-1][1]["executed_evidence"]
        self.assertEqual(result["reviewed_file_ids"], [r["id"] for r in evidence["executed_files"]])
        self.assertEqual(set(result["available_file_ids"]), {r["id"] for r in evidence["executed_files"] + evidence["file_inventory"]})
        self.assertEqual(self.store.get(result["context_manifest_id"])["kind"], "role_context_manifest")
        self.assertEqual(len(context.result_ids), 1)
