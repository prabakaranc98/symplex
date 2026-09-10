# Symplex demo narration

This narration uses an AI-generated voice. The recording demonstrates the interface and a labeled synthetic biology example; it does not establish a biological discovery, clinical result, completed research campaign, or advantage over other tools.

<!-- narration:start -->
This is Symplex, a workspace for investigating complex systems. This narration uses an AI-generated voice.

Start with a question, define the scope, and add the evidence you actually have. Here, a synthetic biology example illustrates the workflow. Its values are demonstration fixtures, not measurements from a real organism.

The workspace separates the problem definition, system representation, proposed solutions, and supporting artifacts. You can inspect assumptions and dependencies instead of relying on a final paragraph alone.

Model specialists now run through the OpenAI Agents SDK. The host controls permissions, validates proposals, and records each model request against persistent budgets.

As an investigation develops, computed outputs, comparisons, and critiques belong alongside their source inputs. Missing evidence remains visible. A successful execution does not by itself validate a scientific claim.

Human steering can change what the next step should address. Budget and activity views help track what has happened and what remains unresolved.

The aim is an inspectable investigation: a clear question, traceable work, and conclusions whose limits stay attached.
<!-- narration:end -->

Recording cues: show the selected synthetic example, its input context, system and solution views, artifacts or results, then activity and budget. Show only records that exist. If an output is unavailable, keep its unavailable state visible.

Generate with `python scripts/render_demo_audio.py --execute`; without `--execute` the script only shows its plan and reads the existing ledger. Defaults use the official [speech API](https://developers.openai.com/api/docs/guides/text-to-speech) with `gpt-4o-mini-tts` and a standard `coral` voice, then the [file transcription API](https://developers.openai.com/api/docs/guides/speech-to-text) with `gpt-transcribe`. Prices were checked against the [model page](https://developers.openai.com/api/docs/models/gpt-4o-mini-tts) and [pricing guide](https://developers.openai.com/api/docs/pricing) on 2026-09-10. Transcript similarity is a rough narration check, not a scientific evaluation.
