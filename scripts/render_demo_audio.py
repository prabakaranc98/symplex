"""Budgeted official OpenAI speech + transcription; dry run unless --execute."""

import argparse
import difflib
import hashlib
import json
import math
import re
import sqlite3
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from symplex.core.contracts import Invalid
from symplex.infrastructure.scoped_budget import ScopedBudget
from symplex.infrastructure.storage import Budget, Store


SCOPE_ID = "demo_audio_v1"
SCOPE_CAP_USD = 0.40
SPEECH_ALLOCATION_USD = 0.25
TOTAL_RESERVATION_USD = 0.30
TTS_MODELS = ("gpt-4o-mini-tts", "tts-1", "tts-1-hd")
TRANSCRIPTION_RATES = {"gpt-transcribe": 0.0045, "whisper-1": 0.006}
VOICES = (
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "onyx",
    "nova",
    "sage",
    "shimmer",
    "verse",
)
INSTRUCTIONS = "Speak clearly and calmly at a steady documentary pace. Preserve every word. Do not add commentary."


def narration_text(path):
    content = Path(path).read_text()
    match = re.search(
        r"<!-- narration:start -->(.*?)<!-- narration:end -->", content, re.S
    )
    if not match:
        raise Invalid("Narration requires explicit start and end markers")
    text = " ".join(match.group(1).split())
    if not text or len((text + INSTRUCTIONS).encode()) > 1950:
        raise Invalid("Narration and instructions must fit the 1950-byte input bound")
    return text


def ledger_snapshot(path):
    database = Path(path).resolve() / "metadata.sqlite3"
    if not database.is_file():
        raise Invalid("An existing workspace budget ledger is required")
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
        rows = db.execute("SELECT name,cap,used FROM budget").fetchall()
    values = {name: {"cap": cap, "used_or_reserved": used} for name, cap, used in rows}
    if "usd" not in values:
        raise Invalid("The existing ledger has no monetary cap")
    return values


def wav_duration(path):
    with wave.open(str(path), "rb") as audio:
        # Streamed WAV headers can have a placeholder frame count. Measure the
        # frames actually present, not the placeholder duration in the header.
        frames = audio.readframes(audio.getnframes())
        return len(frames) / (
            audio.getframerate() * audio.getnchannels() * audio.getsampwidth()
        )


def similarity(expected, actual):
    def tokenize(value):
        return re.findall(r"[a-z0-9]+", value.lower())

    return difflib.SequenceMatcher(None, tokenize(expected), tokenize(actual)).ratio()


def render(
    narration,
    output,
    ledger,
    *,
    execute=False,
    tts_model="gpt-4o-mini-tts",
    voice="coral",
    transcription_model="gpt-transcribe",
    client=None,
):
    if (
        tts_model not in TTS_MODELS
        or transcription_model not in TRANSCRIPTION_RATES
        or voice not in VOICES
    ):
        raise Invalid("Choose an explicitly supported model and standard voice")
    if tts_model != "gpt-4o-mini-tts" and voice not in (
        "alloy",
        "echo",
        "fable",
        "onyx",
        "nova",
        "shimmer",
    ):
        raise Invalid("This legacy TTS model requires a supported legacy voice")
    text = narration_text(narration)
    before = ledger_snapshot(ledger)
    plan = {
        "mode": "execute" if execute else "dry_run",
        "scope_id": SCOPE_ID,
        "scope_cap_usd": SCOPE_CAP_USD,
        "reservation_usd": TOTAL_RESERVATION_USD,
        "ledger_usd": before["usd"],
        "tts_model": tts_model,
        "transcription_model": transcription_model,
        "voice": voice,
        "narration_words": len(text.split()),
        "narration_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "ai_generated_voice": True,
    }
    if not execute:
        return plan
    output = Path(output)
    paths = {
        "audio": output / "narration.wav",
        "transcript": output / "transcript.txt",
        "manifest": output / "audio-manifest.json",
    }
    if any(path.exists() for path in paths.values()):
        raise Invalid(
            "Demo audio assets already exist; refusing an accidental paid rerun"
        )
    # No constructor grant: existing caps remain unchanged. Both API requests
    # must fit atomically before the first is sent.
    store = Store(ledger)
    budget = Budget(store)
    scope = ScopedBudget(budget, SCOPE_ID, {"usd": SCOPE_CAP_USD, "model_requests": 2})
    if client is None:
        from openai import OpenAI

        client = OpenAI(
            base_url="https://api.openai.com/v1", max_retries=0, timeout=180
        )
    reservation = scope.reserve(usd=TOTAL_RESERVATION_USD, model_requests=2)
    parent = store.put("demo_audio_started", {**plan, "reservation": reservation})
    output.mkdir(parents=True, exist_ok=True)
    speech_id = None
    try:
        kwargs = {
            "model": tts_model,
            "voice": voice,
            "input": text,
            "response_format": "wav",
            "speed": 1.0,
        }
        if tts_model == "gpt-4o-mini-tts":
            kwargs["instructions"] = INSTRUCTIONS
        with client.audio.speech.with_streaming_response.create(**kwargs) as response:
            with paths["audio"].open("xb") as target:
                size = 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > 20_000_000:
                        raise Invalid(
                            "Speech output exceeded the bounded demo file size"
                        )
                    target.write(chunk)
        audio = paths["audio"].read_bytes()
        duration = wav_duration(paths["audio"])
        speech_id = store.put(
            "demo_audio_asset",
            {
                "path": str(paths["audio"]),
                "sha256": hashlib.sha256(audio).hexdigest(),
                "bytes": len(audio),
                "seconds": duration,
                "model": tts_model,
                "voice": voice,
                "ai_generated": True,
                "reservation": reservation,
                "cost_basis": "Speech binary response has no reported token usage; retain conservative speech allocation",
            },
            parent,
        )
        if not math.isfinite(duration) or not 0 < duration <= 180:
            raise Invalid(
                "Generated narration duration exceeds the three-minute demo bound"
            )
        with paths["audio"].open("rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model=transcription_model,
                file=audio_file,
                response_format="json",
            )
        transcript_text = transcript.text
        paths["transcript"].write_text(transcript_text + "\n")
        transcript_id = store.put(
            "demo_audio_transcript",
            {
                "text": transcript_text,
                "model": transcription_model,
                "speech_id": speech_id,
                "usage": getattr(transcript, "usage", None).model_dump()
                if getattr(transcript, "usage", None) is not None
                else None,
            },
            parent,
        )
        # Per-minute transcription pricing, rounded up one second conservatively.
        transcript_usd = (
            math.ceil(duration) / 60 * TRANSCRIPTION_RATES[transcription_model]
        )
        speech_usd = (
            SPEECH_ALLOCATION_USD
            if tts_model == "gpt-4o-mini-tts"
            else len(text) * (30 if tts_model == "tts-1-hd" else 15) / 1e6
        )
        accounting = scope.settle(
            reservation, usd=speech_usd + transcript_usd, model_requests=2
        )
        report = {
            **plan,
            "status": "completed",
            "audio_seconds": duration,
            "audio_sha256": hashlib.sha256(audio).hexdigest(),
            "audio_bytes": len(audio),
            "speech_artifact_id": speech_id,
            "transcript_artifact_id": transcript_id,
            "transcript_similarity": similarity(text, transcript_text),
            "similarity_basis": "Normalized word-sequence similarity; not an independent factual evaluation",
            "accounted_usd": speech_usd + transcript_usd,
            "accounting": accounting,
            "cost_basis": "Conservative speech allocation plus duration-priced transcription; not an invoice",
            "scope_budget": scope.scope_snapshot(),
            "ledger_usd_after": budget.snapshot()["usd"],
        }
        store.put("demo_audio_result", report, parent)
        paths["manifest"].write_text(json.dumps(report, indent=2) + "\n")
        return report
    except BaseException as exc:
        store.put(
            "demo_audio_failure",
            {
                "error_type": type(exc).__name__,
                "reservation": reservation,
                "speech_artifact_id": speech_id,
                "accounting_status": "Reservation retained; no automatic retry or cap increase",
            },
            parent,
        )
        raise


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--narration", type=Path, default=repo / "docs/demo-narration.md"
    )
    parser.add_argument("--output", type=Path, default=repo / "docs/demo")
    parser.add_argument("--ledger", type=Path, default=repo / ".symplex")
    parser.add_argument("--tts-model", choices=TTS_MODELS, default=TTS_MODELS[0])
    parser.add_argument("--voice", choices=VOICES, default="coral")
    parser.add_argument(
        "--transcription-model", choices=TRANSCRIPTION_RATES, default="gpt-transcribe"
    )
    args = parser.parse_args()
    if args.execute:
        from dotenv import load_dotenv

        load_dotenv(repo / ".env")
    try:
        if args.execute:
            import httpx

            # Unauthenticated, unbilled reachability check before reserving.
            reachable = httpx.get(
                "https://api.openai.com/v1/models", timeout=15, follow_redirects=False
            )
            if reachable.status_code != 401:
                raise Invalid(
                    "Official API reachability preflight did not return the expected authentication response"
                )
        result = render(
            args.narration,
            args.output,
            args.ledger,
            execute=args.execute,
            tts_model=args.tts_model,
            voice=args.voice,
            transcription_model=args.transcription_model,
        )
        print(json.dumps(result), flush=True)
    except Exception as exc:
        # Error bodies can contain request/credential details. Local failure
        # records retain the exception class and reservation instead.
        print(
            json.dumps({"status": "failed", "error_type": type(exc).__name__}),
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
