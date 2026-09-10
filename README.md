# Symplex

Symplex is a local scientific investigation workbench for complex systems. You start from a decision problem, attach evidence, and run an agent-guided investigation that produces inspectable model artifacts, experiments, and decisions.

## Quick start

Requirements: Python 3.11+, macOS or Linux.

```sh
uv sync
cp .env.example .env  # Only for first-time setup
# Set OPENAI_API_KEY in .env
uv run python -m symplex serve --port 8765
```

Then open: http://127.0.0.1:8765

> This checkout includes an installed `.venv`, so `.venv/bin/python -m symplex serve` also works.

## How it works

1. Create a problem (not just a dataset).
2. Add context in **Evidence**.
3. Choose investigation depth.
4. Click **Launch solver**.

The agent selects actions, while the host validates and executes them. Chat can clarify intent and perform web research, but experiments are never run silently.

## Key capabilities

- FastAPI backend and responsive local workbench with persistent chats and problem versions.
- Adaptive, bounded solver with focused/balanced/thorough profiles (8/16/24 action envelopes).
- Typed solution graphs for subproblems, hypotheses, components, couplings, and decision contracts.
- Versioned inputs and explicit problem reframing with inspectable historical context.
- Hosted OpenAI Python execution that persists generated code and artifacts (CSV/JSON/PNG).
- Frozen experiment protocols with host-side numerical check recomputation.
- Deterministic search archive for protocol-scoped package comparison and evolution.
- Method Lab workflow for candidate review, stage gates, scoped canaries, and rollback controls.
- Optional acquisition and validation flows (including Croissant metadata validation).
- Immutable artifact storage, explicit job states, lineage exports, and reservation-based accounting.

## Scientific and execution boundaries

- Symplex provides exploratory modeling and comparison; it does not prove scientific truth.
- Host checks validate specified numerical properties under frozen limits.
- Model-reported metrics and evidence basis are not independently scientifically validated by the platform.
- The trusted local runner executes bundled templates only; generated arbitrary code does not run on the local host.
- Method Lab reliability metrics measure software-contract behavior, not scientific correctness.

## Configuration and budget

Environment configuration is loaded from `.env` (excluded from Git):

```dotenv
OPENAI_API_KEY=...
OPENAI_CHAT_MODEL=gpt-5.6-luna
OPENAI_HEAVY_MODEL=gpt-6-astra
OPENAI_VALIDATOR_MODEL=gpt-5.6-sol
SYMPLEX_MAX_USD=5
```

Notes:

- Workspace budget ceilings are frozen at first creation.
- Campaign problems retain allocations across solver calls.
- Default global caps include request, token, worker-time, evidence, and USD limits.
- Cost tracking is reservation-based and local; it is not a provider invoice.

## Common commands

```sh
uv run python -m unittest discover -s tests -v
node --check symplex/interfaces/web/common.js
uv run python -m symplex doctor
uv run python -m symplex run --synthetic
uv run python -m symplex run --data /path/to/prepared-p1.json --live
uv run python -m symplex acquire-bamtwoogle
uv run python -m symplex research-pilot
```

`doctor` checks model access without generating text.

## Repository layout

```text
symplex/
  core/                  Problem inputs, DNA, and proposal contracts
  agents/                Solver orchestration, tools, evolution, profiles, prompts
  modeling/              Solution graphs, numerical models, extension contracts
  evidence/              Data admission, metadata validation, spatial inputs
  connectors/            Compute, literature, retrieval, RDF, and visualization connectors
  evaluation/            Metrics, fixtures, and method-policy comparisons
  infrastructure/        OpenAI gateway, storage, budgets, workers, method registry
  observability/         Trace spans and metadata export
  domains/manifests/     Domain packs with full case specifications
  interfaces/            API, CLI, and web views
```

## Security model

The server binds to loopback and enforces origin/session-token checks for writes. This is single-user software and does not provide production authentication, tenant isolation, or public deployment hardening.

## Documentation

- [Architecture](docs/architecture.md)
- [Data contract](docs/data-contract.md)
- [Connector and simulator design](docs/connectors-and-simulation.md)
- [Verification record](docs/verification.md)
- [Master specification](docs/master-specification.md)
- In-app design brief: `symplex/interfaces/web/research.html`
