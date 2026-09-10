"""Audio rendering preserves the original ledger and uses official SDK transports."""

import io
import json
import sqlite3
import tempfile
import unittest
import wave
from pathlib import Path

import httpx
from openai import OpenAI

from scripts.render_demo_audio import render, narration_text
from symplex.core.contracts import Invalid
from symplex.infrastructure.storage import Budget, BudgetExhausted, Store


class DemoAudioTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.store = Store(self.root / "ledger")
        self.budget = Budget(self.store, {"usd": 0.5})
        self.narration = self.root / "narration.md"
        self.narration.write_text(
            "<!-- narration:start --> This is a synthetic demonstration with an AI-generated voice. <!-- narration:end -->"
        )
        self.output = self.root / "audio"
        self.calls = []
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as audio:
            audio.setparams((1, 2, 24000, 24000, "NONE", "not compressed"))
            audio.writeframes(b"\0\0" * 24000)
        self.audio = buffer.getvalue()

    def client(self, fail_transcription=False):
        def respond(request):
            self.calls.append(request.url.path)
            if request.url.path == "/v1/audio/speech":
                body = json.loads(request.content)
                self.assertEqual(body["response_format"], "wav")
                self.assertEqual(body["model"], "gpt-4o-mini-tts")
                return httpx.Response(
                    200, content=self.audio, headers={"content-type": "audio/wav"}
                )
            if fail_transcription:
                return httpx.Response(
                    503,
                    json={
                        "error": {
                            "message": "sk-never-log-this",
                            "type": "server_error",
                        }
                    },
                )
            self.assertIn(b"gpt-transcribe", request.content)
            self.assertIn(self.audio, request.content)
            return httpx.Response(200, json={"text": narration_text(self.narration)})

        client = OpenAI(
            api_key="sk-test",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        )
        self.addCleanup(client.close)
        return client

    def run_render(self, **kwargs):
        return render(self.narration, self.output, self.store.root, **kwargs)

    def test_dry_run_reads_only_and_never_creates_scope_or_output(self):
        before = self.budget.snapshot()
        result = self.run_render(client=self.client())
        self.assertEqual(result["mode"], "dry_run")
        self.assertFalse(self.calls)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.budget.snapshot(), before)
        with sqlite3.connect(self.store.root / "metadata.sqlite3") as db:
            self.assertEqual(
                db.execute("select count(*) from budget_scopes").fetchone()[0], 0
            )

    def test_native_sdk_speech_and_transcription_produce_audited_assets(self):
        result = self.run_render(execute=True, client=self.client())
        self.assertEqual(self.calls, ["/v1/audio/speech", "/v1/audio/transcriptions"])
        self.assertEqual(result["audio_seconds"], 1)
        self.assertEqual(result["transcript_similarity"], 1)
        self.assertAlmostEqual(result["accounted_usd"], 0.250075)
        self.assertAlmostEqual(
            self.budget.snapshot()["usd"]["used_or_reserved"], 0.250075
        )
        self.assertEqual(self.budget.snapshot()["usd"]["cap"], 0.5)
        self.assertEqual((self.output / "narration.wav").read_bytes(), self.audio)
        self.assertTrue((self.output / "audio-manifest.json").is_file())
        self.assertEqual(len(self.store.list("demo_audio_transcript")), 1)
        with self.assertRaises(Invalid):
            self.run_render(execute=True, client=self.client())
        self.assertEqual(len(self.calls), 2)

    def test_scope_usage_survives_output_directory_change(self):
        self.run_render(execute=True, client=self.client())
        self.output = self.root / "different-output"
        with self.assertRaises(BudgetExhausted):
            self.run_render(execute=True, client=self.client())
        self.assertEqual(len(self.calls), 2)

    def test_global_budget_blocks_whole_bundle_before_first_request(self):
        reserved = self.budget.reserve(usd=0.3)
        self.budget.settle(reserved, usd=0.3)
        with self.assertRaises(BudgetExhausted):
            self.run_render(execute=True, client=self.client())
        self.assertFalse(self.calls)
        self.assertEqual(self.budget.snapshot()["usd"]["cap"], 0.5)

    def test_failure_keeps_reservation_audio_and_safe_metadata_without_retry(self):
        with self.assertRaises(Exception):
            self.run_render(execute=True, client=self.client(fail_transcription=True))
        self.assertEqual(len(self.calls), 2)
        self.assertAlmostEqual(self.budget.snapshot()["usd"]["used_or_reserved"], 0.3)
        self.assertTrue((self.output / "narration.wav").is_file())
        failure = self.store.list("demo_audio_failure")[0]["data"]
        self.assertNotIn("sk-never-log-this", json.dumps(failure))
        self.assertIsNotNone(failure["speech_artifact_id"])

    def test_unknown_models_and_unbounded_narration_fail_without_calls(self):
        with self.assertRaises(Invalid):
            self.run_render(
                execute=True, tts_model="unpriced-model", client=self.client()
            )
        self.narration.write_text(
            "<!-- narration:start -->" + "x" * 2000 + "<!-- narration:end -->"
        )
        with self.assertRaises(Invalid):
            self.run_render(execute=True, client=self.client())
        self.assertFalse(self.calls)
        self.assertFalse(self.store.list("demo_audio_started"))


if __name__ == "__main__":
    unittest.main()
