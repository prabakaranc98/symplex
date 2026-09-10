import tempfile
import unittest
from types import SimpleNamespace

from symplex.agents.context import working_context
from symplex.agents.planner_context import bounded_planner_context
from symplex.connectors.compute import save_blob
from symplex.connectors.inspection import inspect
from symplex.core.contracts import Invalid
from symplex.infrastructure.storage import Store


class InspectionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = Store(directory.name)
        self.problem = self.store.put(
            "workspace_problem", {"question": "Inspect current mechanisms"}
        )

    def test_exact_record_selection_and_pagination_reaches_current_working_memory(self):
        ident = self.store.put(
            "complex_system", {"hypotheses": [{"claim": "x" * 10000}]}, self.problem
        )
        first = inspect(
            self.store,
            self.problem,
            ident,
            {"pointer": "/hypotheses/0/claim", "limit": 8000},
        )
        second = inspect(
            self.store,
            self.problem,
            ident,
            {
                "pointer": "/hypotheses/0/claim",
                "offset": first["next_offset"],
                "limit": 8000,
            },
        )
        self.assertEqual(len(first["excerpt"]), 8000)
        self.assertEqual(first["excerpt"] + second["excerpt"], '"' + "x" * 10000 + '"')
        self.assertIsNone(second["next_offset"])
        memory = working_context(self.store, self.problem)
        observed = next(r for r in memory if r["kind"] == "artifact_inspection")
        self.assertEqual(observed["data"]["excerpt"], second["excerpt"])
        self.assertEqual(
            observed["data"]["artifact_digest"], self.store.get(ident)["digest"]
        )

    def test_protected_foreign_stale_and_unknown_artifacts_are_denied(self):
        for kind in (
            "method_task",
            "method_evaluation_run",
            "dataset",
            "model_response",
            "model_call",
        ):
            ident = self.store.put(kind, {"secret": "protected"}, self.problem)
            with self.assertRaises(Invalid):
                inspect(self.store, self.problem, ident)
        foreign = self.store.put("complex_system", {}, "another_problem")
        current = self.store.put("complex_system", {}, self.problem)
        self.store.invalidate(current)
        for ident in (foreign, current, "unknown"):
            with self.assertRaises(Invalid):
                inspect(self.store, self.problem, ident)

    def test_source_file_hash_and_binary_boundary_are_enforced(self):
        ident = save_blob(
            self.store,
            b"def mechanism(): return 1",
            "source.py",
            self.problem,
            "generated",
        )
        self.assertIn("mechanism", inspect(self.store, self.problem, ident)["excerpt"])
        blob = self.store.get(ident)
        (self.store.root / "blobs" / blob["data"]["sha256"]).write_bytes(b"changed")
        with self.assertRaises(Invalid):
            inspect(self.store, self.problem, ident)
        binary = save_blob(
            self.store, b"binary", "image.png", self.problem, "generated"
        )
        with self.assertRaises(Invalid):
            inspect(self.store, self.problem, binary)

    def test_retrieval_result_is_actually_presented_to_next_planner(self):
        result = {
            "hits": [{"artifact_id": "source", "text": "A discriminating observation"}]
        }
        context = bounded_planner_context(
            {
                "completed_actions": [
                    {
                        "tool": "search_artifacts",
                        "artifact_id": "action",
                        "result": result,
                    }
                ]
            }
        )
        self.assertEqual(
            context["completed_actions"][-1]["result"]["observed_result"], result
        )


    def test_typed_tool_argument_error_is_feedback_instead_of_workflow_crash(self):
        from symplex.agents.tools import REGISTRY
        ident = self.store.put("complex_system", {"hypotheses": []}, self.problem)
        context = SimpleNamespace(store=self.store, problem_id=self.problem)
        with self.assertRaisesRegex(Invalid, "less_than_equal"):
            REGISTRY.execute("inspect_artifact", context,
                SimpleNamespace(target_id=ident, instruction='{"limit":24000}'), {"artifacts.read"})
        self.assertFalse(self.store.list("artifact_inspection"))
        result = REGISTRY.execute("inspect_artifact", context,
            SimpleNamespace(target_id=ident, instruction='{"limit":5000}'), {"artifacts.read"})
        self.assertIn('hypotheses', result['excerpt'])


if __name__ == "__main__":
    unittest.main()
