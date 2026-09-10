# Narration check

The [AI-generated narration](narration.wav) was produced by OpenAI `gpt-4o-mini-tts`, using the standard `coral` voice. It is 82.15 seconds long. OpenAI `gpt-transcribe` then transcribed that exact audio file.

The [raw transcript](transcript.txt) matches the [source script](../demo-narration.md) except that it spells the product name “Simplex” rather than “Symplex.” Normalized word-sequence similarity is 0.99405. This verifies wording only; it does not validate scientific claims. The source script is the appropriate spelling reference for captions.

The [manifest](audio-manifest.json) records the original audio SHA-256, duration, local artifact IDs and accounting. Two API requests were made without automatic retries. The original workspace ledger accounts for $0.256225: a conservative $0.25 speech allocation because the binary response supplies no token usage, plus $0.006225 for transcription at the duration-based rate. This is an accounting estimate, not an invoice. During audio generation, the $5 ledger cap and $0.40 audio scope cap were unchanged; later operator grants are recorded separately.

To fit a video under 80 seconds without another API call, the editor can apply a 1.05× audio tempo during video muxing, giving approximately 78.24 seconds. Keep the original WAV and its hash intact as the source artifact.
