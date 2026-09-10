"""Revision semantics and provenance, using a fake hosted service and no model calls."""

import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from symplex.connectors.compute import run, save_blob
from symplex.core.contracts import Invalid
from symplex.infrastructure.storage import Store


class FakeProvider:
    def __init__(self):
        self.uploads = []
        self.deleted = []
        self.calls = []
        self.client = SimpleNamespace(
            files=SimpleNamespace(create=self.upload, delete=self.deleted.append)
        )

    def upload(self, file, purpose):
        self.uploads.append(file)
        return SimpleNamespace(id="remote-" + str(len(self.uploads)))

    def _call(self, role, request, parent, compute):
        self.calls.append(request)
        call = SimpleNamespace(type="code_interpreter_call")
        call.model_dump = lambda: {"status": "completed", "container_id": "fake"}
        return SimpleNamespace(
            id="response-" + str(len(self.calls)),
            model="fake",
            output=[call],
            output_text="Computed a synthetic revision",
        )


class ComputeRevisionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name)
        self.problem = self.store.put(
            "workspace_problem", {"question": "Compare system hypotheses"}
        )
        self.provider = FakeProvider()
        collector = patch(
            "symplex.connectors.compute.collect_outputs", side_effect=self.collect
        )
        collector.start()
        self.addCleanup(collector.stop)

    def collect(self, store, provider, problem_id, run_id):
        for package in store.list("compute_package"):
            if package["data"]["run_id"] == run_id:
                return package["id"]
        return store.put(
            "compute_package",
            {
                "run_id": run_id,
                "file_ids": [],
                "summary": "Synthetic execution",
                "file_failures": [],
            },
            problem_id,
        )

    def predecessor(self, files=None, problem=None):
        problem = problem or self.problem
        run_id = self.store.put(
            "compute_run",
            {"summary": "Prior synthetic calculation", "calls": []},
            problem,
        )
        ids = [
            save_blob(self.store, raw, name, problem, "generated", run_id)
            for name, raw in (
                files
                or [
                    (
                        "model.py",
                        b"raise RuntimeError('never run on application host')",
                    ),
                    ("results.json", b'{"basis":"synthetic","metric":0.4}'),
                    ("observations.csv", b"time,value\n0,0.4\n"),
                    ("plot.png", b"plot fixture"),
                ]
            )
        ]
        return self.store.put(
            "compute_package",
            {
                "run_id": run_id,
                "file_ids": ids,
                "summary": "Prior candidate, synthetic",
                "file_failures": [],
            },
            problem,
        )

    def context_sent(self):
        request = self.provider.calls[-1]["input"]
        if isinstance(request, list):
            request = request[0]["content"][0]["text"]
        return json.loads(request)

    def test_explicit_revision_receives_code_results_and_exact_predecessor(self):
        predecessor = self.predecessor()
        first = run(
            self.store,
            None,
            self.provider,
            self.problem,
            "Test a rival",
            predecessor_package_id=predecessor,
        )
        self.assertEqual(len(self.provider.uploads), 3)
        uploaded = [raw for name, raw in self.provider.uploads]
        self.assertIn(b"raise RuntimeError('never run on application host')", uploaded)
        sent = self.context_sent()
        self.assertEqual(sent["predecessor"]["package_id"], predecessor)
        self.assertEqual(
            sent["predecessor"]["package_digest"], self.store.get(predecessor)["digest"]
        )
        record = self.store.list("compute_run")[-1]["data"]
        actual_files = {
            r["id"] for r in record["input_manifest"] if r["channel"] == "file"
        }
        self.assertEqual(actual_files, set(sent["predecessor"]["file_ids"]))
        self.assertEqual(record["predecessor_package_id"], predecessor)
        self.assertEqual(len(self.provider.deleted), 3)

        # The package produced by this execution must not become its own new input.
        second = run(
            self.store,
            None,
            self.provider,
            self.problem,
            "Test a rival",
            predecessor_package_id=predecessor,
        )
        self.assertEqual(second, first)
        self.assertEqual(len(self.provider.calls), 1)
        self.assertEqual(len(self.provider.uploads), 3)

    def test_new_predecessor_creates_new_revision_but_default_does_not_select_one(self):
        first = run(self.store, None, self.provider, self.problem, "Inspect")
        predecessor = self.predecessor()
        again = run(self.store, None, self.provider, self.problem, "Inspect")
        self.assertEqual(again, first)
        self.assertFalse(self.provider.uploads)
        revision = run(
            self.store,
            None,
            self.provider,
            self.problem,
            "Inspect",
            predecessor_package_id=predecessor,
        )
        self.assertNotEqual(revision, first)
        self.assertEqual(len(self.provider.calls), 2)

    def test_revision_gets_own_parent_diagnostics_and_ignores_child_feedback_on_retry(self):
        predecessor = self.predecessor()
        diagnostic = self.store.put("execution_assessment", {"package_id": predecessor,
                                     "status": "failed", "error": "Parent failed a frozen check"}, self.problem)
        first = run(self.store, None, self.provider, self.problem, "Repair", predecessor_package_id=predecessor)
        supplied = self.context_sent()["predecessor"]["diagnostics"]
        self.assertEqual([r["id"] for r in supplied], [diagnostic])
        self.assertIn("Parent failed", supplied[0]["data_excerpt"])
        self.store.put("execution_assessment", {"package_id": first, "status": "checked"}, self.problem)
        again = run(self.store, None, self.provider, self.problem, "Repair", predecessor_package_id=predecessor)
        self.assertEqual(first, again)
        self.assertEqual(len(self.provider.calls), 1)
        manifest = self.store.get(self.store.get(first)["data"]["run_id"])["data"]["input_manifest"]
        self.assertIn(diagnostic, [r["id"] for r in manifest if r["channel"] == "predecessor_diagnostic"])

    def test_output_does_not_displace_source_context_and_change_retry_identity(self):
        for kind in ("solution", "model_critique", "evidence_note"):
            self.store.put(kind, {"text": "evidence " * 650}, self.problem)
        self.store.put(
            "context",
            {
                "title": "Important late input",
                "content": "x" * 4000,
                "format": "text",
                "basis": "user_context",
            },
            self.problem,
        )
        first = run(self.store, None, self.provider, self.problem, "Inspect")
        self.store.put(
            "compute_package",
            {
                "summary": "Long generated output " * 400,
                "run_id": self.store.get(first)["data"]["run_id"],
                "file_ids": [],
            },
            self.problem,
        )
        again = run(self.store, None, self.provider, self.problem, "Inspect")
        self.assertEqual(first, again)
        self.assertEqual(len(self.provider.calls), 1)

    def test_changed_system_hypothesis_or_protocol_invalidates_cache(self):
        previous = run(self.store, None, self.provider, self.problem, "Inspect")
        for kind in ("complex_system", "hypothesis_review", "experiment_protocol"):
            from tests.test_experiments import protocol_spec
            data = dict(protocol_spec(), status="frozen_for_comparison") if kind == "experiment_protocol" else {"instruction": "New " + kind}
            ident = self.store.put(kind, data, self.problem)
            current = run(self.store, None, self.provider, self.problem, "Inspect")
            self.assertNotEqual(current, previous)
            source_ids = self.store.list("compute_run")[-1]["data"]["source_ids"]
            self.assertIn(ident, source_ids)
            previous = current
        self.assertEqual(len(self.provider.calls), 4)

    def test_actual_sources_include_binary_blob_and_omit_unsupplied_context(self):
        contexts = [
            self.store.put(
                "context",
                {
                    "title": str(i),
                    "format": "text",
                    "basis": "user_context",
                    "content": "input " + str(i),
                },
                self.problem,
            )
            for i in range(8)
        ]
        blob = save_blob(
            self.store,
            b"\x89PNG\r\n\x1a\nfixture",
            "diagram.png",
            self.problem,
            "user_context",
        )
        binary = self.store.put(
            "binary_context", {"title": "Diagram", "blob_id": blob}, self.problem
        )
        run(self.store, None, self.provider, self.problem)
        record = self.store.list("compute_run")[-1]["data"]
        self.assertNotIn(contexts[0], record["source_ids"])
        self.assertIn(binary, record["source_ids"])
        self.assertIn(blob, record["source_ids"])
        reference = next(r for r in record["input_manifest"] if r["id"] == blob)
        self.assertEqual(reference["digest"], self.store.get(blob)["digest"])
        self.assertEqual(
            reference["content_sha256"], self.store.get(blob)["data"]["sha256"]
        )
        image = self.provider.calls[-1]["input"][0]["content"][-1]
        self.assertEqual(image["type"], "input_image")

    def test_missing_stale_or_foreign_predecessor_rejected_before_upload(self):
        foreign_problem = self.store.put(
            "workspace_problem", {"question": "Other case"}
        )
        foreign = self.predecessor(problem=foreign_problem)
        stale = self.predecessor()
        self.store.invalidate(stale)
        for ident in ("compute_package_missing", foreign, stale):
            with self.subTest(ident=ident), self.assertRaises(Invalid):
                run(
                    self.store,
                    None,
                    self.provider,
                    self.problem,
                    predecessor_package_id=ident,
                )
        self.assertFalse(self.provider.calls)
        self.assertFalse(self.provider.uploads)

    def test_missing_or_foreign_binary_blob_rejected_before_upload(self):
        other = self.store.put("workspace_problem", {"question": "Other case"})
        blob = save_blob(self.store, b"{}", "data.json", other, "user_context")
        ident = self.store.put(
            "binary_context", {"title": "Wrong scope", "blob_id": blob}, self.problem
        )
        with self.assertRaises(Invalid):
            run(self.store, None, self.provider, self.problem)
        self.store.invalidate(ident)
        self.store.put(
            "binary_context", {"title": "Missing", "blob_id": "missing"}, self.problem
        )
        with self.assertRaises(Invalid):
            run(self.store, None, self.provider, self.problem)
        self.assertFalse(self.provider.uploads)

    def test_predecessor_integrity_and_run_binding_checked(self):
        predecessor = self.predecessor()
        blob = self.store.get(self.store.get(predecessor)["data"]["file_ids"][0])
        (self.store.root / "blobs" / blob["data"]["sha256"]).write_bytes(b"tampered")
        with self.assertRaises(Invalid):
            run(
                self.store,
                None,
                self.provider,
                self.problem,
                predecessor_package_id=predecessor,
            )
        self.assertFalse(self.provider.uploads)

        legitimate = self.predecessor()
        foreign_run = self.store.put(
            "compute_run", {"summary": "Other model"}, self.problem
        )
        wrong = self.store.put(
            "compute_package",
            {
                "run_id": foreign_run,
                "file_ids": self.store.get(legitimate)["data"]["file_ids"],
                "summary": "Misbound outputs",
            },
            self.problem,
        )
        with self.assertRaises(Invalid):
            run(
                self.store,
                None,
                self.provider,
                self.problem,
                predecessor_package_id=wrong,
            )
        self.assertFalse(self.provider.uploads)

    def test_revision_file_caps_and_omissions_are_explicit(self):
        predecessor = self.predecessor(
            files=[("model.py", b"# previous program")]
            + [(f"part-{i}.csv", b"x\n" + b"1\n" * 400000) for i in range(6)]
        )
        run(
            self.store,
            None,
            self.provider,
            self.problem,
            predecessor_package_id=predecessor,
        )
        self.assertLessEqual(
            sum(len(raw) for name, raw in self.provider.uploads), 4000000
        )
        self.assertTrue(self.context_sent()["predecessor"]["omitted_files"])
        self.assertLessEqual(len(self.provider.uploads), 6)
        self.assertTrue(self.provider.uploads[0][0].endswith("model.py"))


if __name__ == "__main__":
    unittest.main()
