import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from symplex.connectors.compute import collect_outputs, read_blob, run, save_blob
from symplex.core.contracts import Invalid
from symplex.infrastructure.storage import Store


class ComputeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name)
        self.problem = self.store.put(
            "workspace_problem", {"question": "Model a system"}
        )

    def readiness_provider(self):
        uploads, calls = [], []

        def upload(**kwargs):
            uploads.append(kwargs)
            return SimpleNamespace(id="offline-file-" + str(len(uploads)))

        def invoke(*args, **kwargs):
            calls.append((args, kwargs))
            execution = SimpleNamespace(type="code_interpreter_call", model_dump=lambda: {"status": "completed"})
            return SimpleNamespace(id="offline-response", model="offline", output=[execution], output_text="No output files")

        return SimpleNamespace(client=SimpleNamespace(files=SimpleNamespace(create=upload, delete=lambda *args: None)),
                               _call=invoke, uploads=uploads, calls=calls)

    def test_blocked_and_unknown_protocols_stop_before_upload_or_paid_execution(self):
        from tests.test_experiments import protocol_spec
        for readiness in ("blocked", "unknown"):
            with self.subTest(readiness=readiness):
                data = protocol_spec()
                if readiness == "blocked":
                    data.update(execution_readiness="blocked", blocking_reasons=["Missing numerical specification"], status="blocked_design")
                else:
                    data.pop("execution_readiness")
                    data.pop("blocking_reasons")
                    data["status"] = "frozen_for_comparison"
                self.store.put("experiment_protocol", data, self.problem)
                provider = self.readiness_provider()
                with self.assertRaisesRegex(Invalid, "execution is " + readiness):
                    run(self.store, None, provider, self.problem, "Run the planned experiment")
                self.assertEqual(provider.uploads, [])
                self.assertEqual(provider.calls, [])
                self.assertEqual(self.store.list("compute_run"), [])

    def test_explicit_ready_protocol_can_execute_and_identical_retry_collects(self):
        from tests.test_experiments import protocol_spec
        self.store.put("experiment_protocol", dict(protocol_spec(), status="frozen_for_comparison"), self.problem)
        provider = self.readiness_provider()
        first = run(self.store, None, provider, self.problem, "Execute the frozen experiment")
        self.assertEqual(len(provider.calls), 1)
        self.assertGreater(len(provider.uploads), 0)
        second = run(self.store, None, provider, self.problem, "Execute the frozen experiment")
        self.assertEqual(first, second)
        self.assertEqual(len(provider.calls), 1)

    def test_legacy_run_remains_collectable_and_identical_retry_does_not_reexecute(self):
        from tests.test_experiments import protocol_spec
        legacy = protocol_spec()
        legacy.pop("execution_readiness")
        legacy.pop("blocking_reasons")
        legacy["status"] = "frozen_for_comparison"
        self.store.put("experiment_protocol", legacy, self.problem)
        provider = self.readiness_provider()
        # Simulate a run recorded before the new preflight existed. All execution
        # and upload methods are local fakes; the second call uses the real gate.
        with patch("symplex.evaluation.experiments.require_execution_ready"):
            first = run(self.store, None, provider, self.problem, "Historical experiment")
        record = self.store.get(first)
        uploads = len(provider.uploads)
        self.assertEqual(collect_outputs(self.store, provider, self.problem, record["data"]["run_id"]), first)
        second = run(self.store, None, provider, self.problem, "Historical experiment")
        self.assertEqual(first, second)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(len(provider.uploads), uploads)
        self.assertEqual(self.store.get(first), record)

    def test_actual_collection_assessment_cannot_trigger_an_identical_paid_retry(self):
        class Provider:
            client = SimpleNamespace()

            def __init__(self):
                self.calls = []

            def _call(self, role, request, parent, compute):
                self.calls.append(request)
                execution = SimpleNamespace(type="code_interpreter_call", model_dump=lambda: {"status": "completed"})
                return SimpleNamespace(id="offline-response", model="offline", output=[execution], output_text="No output files")

        provider = Provider()
        first = run(self.store, None, provider, self.problem, "Inspect the model")
        self.assertEqual(len(self.store.list("execution_assessment")), 1)
        self.store.put("artifact_inspection", {"source_id": first, "excerpt": "Previously saved output"}, self.problem)
        second = run(self.store, None, provider, self.problem, "Inspect the model")
        self.assertEqual(first, second)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(len(self.store.list("compute_run")), 1)
        self.assertEqual(len(self.store.list("execution_assessment")), 1)

    def test_file_collection_handles_unknown_size_and_does_not_rerun(self):
        run = self.store.put(
            "compute_run",
            {
                "calls": [{"container_id": "container", "status": "completed"}],
                "summary": "Executed code",
            },
            self.problem,
        )
        count = []

        class Stream:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def iter_bytes(self):
                yield b'{"basis":"synthetic"}'

        class File:
            id = "file"

            def model_dump(self):
                return {
                    "id": "file",
                    "path": "/mnt/data/results.json",
                    "bytes": None,
                    "source": "assistant",
                }

        def list_files(*args, **kwargs):
            count.append(1)
            return [File()]

        provider = SimpleNamespace(
            client=SimpleNamespace(
                containers=SimpleNamespace(
                    files=SimpleNamespace(
                        list=list_files,
                        content=SimpleNamespace(
                            with_streaming_response=SimpleNamespace(
                                retrieve=lambda *a, **k: Stream()
                            )
                        ),
                    )
                )
            )
        )
        package = collect_outputs(self.store, provider, self.problem, run)
        data = self.store.get(package)["data"]
        self.assertEqual(len(data["file_ids"]), 1)
        self.assertFalse(data["file_failures"])
        blob = self.store.get(data["file_ids"][0])
        self.assertEqual(json.loads(read_blob(self.store, blob))["basis"], "synthetic")
        self.assertEqual(
            collect_outputs(self.store, provider, self.problem, run), package
        )
        self.assertEqual(len(count), 1)

    def test_blob_integrity_and_unsupported_active_content(self):
        with self.assertRaises(Invalid):
            save_blob(
                self.store,
                b"<script>bad</script>",
                "view.html",
                self.problem,
                "generated",
            )
        ident = save_blob(
            self.store, b"print(1)", "model.py", self.problem, "generated"
        )
        r = self.store.get(ident)
        (self.store.root / "blobs" / r["data"]["sha256"]).write_bytes(b"changed")
        with self.assertRaises(Invalid):
            read_blob(self.store, r)

    def test_multimodal_upload_and_download_are_scoped_artifacts(self):
        import base64

        from fastapi.testclient import TestClient

        from symplex.interfaces.api import create_app

        client = TestClient(create_app(self.tmp.name))
        state = client.get("/api/state").json()
        headers = {"X-Symplex-Token": state["token"], "Origin": "http://testserver"}
        payload = {
            "problem_id": self.problem,
            "filename": "diagram.png",
            "content_base64": base64.b64encode(b"\x89PNG\r\n\x1a\nfixture").decode(),
        }
        response = client.post("/api/uploads", json=payload, headers=headers)
        self.assertEqual(response.status_code, 200)
        result = client.get("/api/files/" + response.json()["blob_id"])
        self.assertEqual(result.headers["content-type"], "image/png")
        self.assertEqual(result.headers["x-content-type-options"], "nosniff")
        payload["filename"] = "evil.html"
        response = client.post("/api/uploads", json=payload, headers=headers)
        self.assertEqual(response.status_code, 400)

    def test_working_context_preserves_latest_state_excludes_holdout(self):
        from symplex.agents.context import working_context

        self.store.put("problem_dna", {"decision": "inspect"}, self.problem)
        self.store.put("dataset", {"secret": "confirmation labels"}, self.problem)
        for i in range(6):
            self.store.put("model_critique", {"assessment": str(i)}, self.problem)
        context = working_context(self.store, self.problem)
        self.assertIn("problem_dna", {r["kind"] for r in context})
        self.assertEqual(
            [r["data"]["assessment"] for r in context if r["kind"] == "model_critique"],
            ["5"],
        )
        self.assertNotIn("confirmation labels", json.dumps(context))
