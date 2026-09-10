# Symplex

A local scientific investigation workbench for complex systems, with bounded method evaluation and operator-reviewed canaries. A user brings a decision problem; Symplex's agents frame it, break it into connected subproblems, gather evidence, propose rival mechanisms, design model/tool/rule assemblies, execute supported experiments, and deliver inspectable results.

## Run

Python 3.11+ on macOS or Linux:

```sh
uv sync
cp .env.example .env  # Only on a new installation; preserve your existing .env.
# Set OPENAI_API_KEY in .env.
uv run python -m symplex serve --port 8765
```

Open http://127.0.0.1:8765. The current checkout also has an installed `.venv`, so `.venv/bin/python -m symplex serve` works directly.

The interface starts with a problem, not a dataset. Create a problem, attach context under **Evidence**, choose investigation depth, and **Launch solver**. The agent chooses actions; the host validates and executes them. Chat clarifies intent and can research the web, but does not silently run experiments. System, component, scenario, evaluation and delivery views expose the same persisted artifacts.

## Implemented

- FastAPI backend, responsive local workbench, persistent chat and problem versions.
- OpenAI SDK role routing: `gpt-5.6-luna` for chat, `gpt-6-astra` for synthesis/modeling/metareasoning/code proposals, `gpt-5.6-sol` for separate model review. No silent model fallback.
- An adaptive solver with bounded tool selection, model critique and revisits; focused/balanced/thorough profiles allow 8/16/24 actions under one global ledger.
- Decision-boundary input snapshots record new, changed and removed context. The agent can explicitly reframe Problem DNA; earlier frames and the inputs behind them remain inspectable. Prompt files are frozen for each solver invocation.
- Typed solution graphs: subproblem dependencies, rival hypotheses, model/rule/tool components, couplings, training/inference plans, experiment and decision contracts.
- User context and CSV/JSON/GeoJSON artifacts; point/polygon coordinate visualization. Geography is optional, not the product's central abstraction.
- A trusted stochastic-system template: normalized states, delayed influences, inertia, noise, interventions, common-random-number scenarios and coupling ablation. What-if edits create and execute a new model version.
- Prepared P1 empirical adapter: temporal/semantic admission, raw and constrained baselines, training-only shrinkage, typed parent–child mutation, sparse archive, agent-selected action, protected confirmation and result-linked decision export.
- Optional Google BamTwoogle acquisition, actual `mlcroissant` metadata validation and a small separate research-policy pilot. Pilot labels never enter the answering model context.
- SQLite metadata, immutable content-addressed artifact files, idempotency keys, explicit job states, global reservations, stale-descendant tracking and JSON lineage exports.
- Hosted OpenAI Python execution: Astra writes models, executes comparisons and diagnostics, and returns persisted code, CSV/JSON outputs and PNG plots. No generated code runs on the local application host.
- Frozen experiment protocols bind a system version, alternatives and numerical checks to execution. Every collected package receives an assessment; the host recomputes the specified checks from generated CSV bytes and records passing, failed or unavailable checks separately from model critique.
- Optional component maps link each declared mechanism to saved Python symbols or record its omission, and link state outputs to CSV columns and declared units. The host checks source/run provenance, symbol existence and column references; it does not prove equation fidelity or that a saved symbol was executed.
- A deterministic search archive groups evaluated packages by declared model kinds, evidence basis and protocol. Parent selection combines prior visits with the fraction of frozen checks passed, comparing scores only within the same protocol. `evolve_model` selects from the current protocol and supplies the selected predecessor's Python source and results to a new execution.
- A paired method-evaluation backend supports one `complexity_architect` instruction change through development, regression and protected fresh-task stages. It freezes parent/candidate prompts and the contract, consumes operator-supplied tasks once, runs both arms under matching settings and equal persistent caps, and reports contract/reference pass counts with resource accounting.
- Method Lab displays actual stage gates, operator reviews, shadow designations, scoped canaries and rollback observations. Review and activation are separate explicit actions. The registry revalidates the first frozen report for each stage, rejects paired regressions, and requires a held-out contract gain before approval. A shadow designation leaves baseline execution unchanged and runs no extra evaluation.
- Approved canaries affect only named problems and future solver invocations. Method versions participate in job identity; recorded execution failures and failed host assessments trigger rollback at the reviewed limits. Manual rollback and operator rejection also restore the baseline for future invocations. Existing results and prompt files remain preserved.
- Interactive 3D replay of recorded trajectories, with exact CSV-column projections and optional row filters. Views retain source/run digests and expose the original coordinates; missing samples remain absent.
- Multimodal PNG/JPEG/PDF inputs; images can enter Astra’s visual context and supplied files enter the hosted compute container.
- Registered DuckDB, RDFLib, scikit-learn retrieval, Crossref, Croissant, chart export and OpenTelemetry connectors.
- User steering at decision boundaries; agent-requested clarification pauses and resumes the same checkpoint after a reply.
- Five domain manifests containing the ten complete problem specifications from the supplied build contract.

## Execution and scientific scope

The general engine produces exploratory models and comparisons. Its host numerical checks establish specified properties of generated values under frozen limits. The comparison's summary metrics, evidence basis and intervals are maker-reported; separate model critique does not independently validate them. Scientific use requires external evidence and domain validation appropriate to the decision.

The bundled stochastic runner uses dimensionless, agent-proposed hypotheses and preference weights. Its outputs are conditional simulations. Specialist physical, biological and social models can be authored in hosted Python when their dependencies are available, but their scientific accuracy is not established by successful execution.

P1 requires a prepared real archive. No Kalshi or METR-LA archive was supplied. The built-in P1 fixture is explicitly synthetic. The other original domain cases remain specifications. BamTwoogle is a research-method benchmark, not a replacement for a domain's empirical validation. The host does not verify the truth of user-declared raw source hashes, availability semantics or settlement metadata against an external archive.

The runner executes bundled trusted templates only. The subprocess strips inherited credentials, bounds CPU and wall time, caps file size, and on Linux caps address space. It is **not** a sandbox for untrusted generated code; macOS does not enforce its memory cap. Draft code proposals remain unexecuted until the agent runs code through the separate hosted Python backend. Specialist native simulators, GPU workloads, large neural training, weight merging, general model-asset loading and deployment still require suitable backends. The hosted compute backend checks its available packages instead of claiming every scientific library exists.

Method candidates remain immutable proposals; an explicit operator review and a separately requested scoped canary govern their use. The paired replay backend measures software-contract reliability for the supported architect role; its counts do not measure scientific correctness, decision utility or generalization. Other role evaluators and automatic promotion are not implemented. Shadow designation is a registry record, not an executed comparison. Monitoring covers job execution and host-assessment failures, not scientific correctness or full operational drift. The replay, optional one-question-per-split policy pilot and diagnostic search archive do not establish recursive self-improvement or cross-domain scientific superiority.

In **Method Lab**, refresh a candidate's gate status, inspect its development/regression/holdout reports, and record a review with an operator name, reason, problem allowlist and failure limits. Activation requires the latest approved review and rechecks its frozen evidence. Fresh stage tasks and paid replays are prepared through the evaluation backend; the Method Lab review controls do not generate model calls. No missing gate is displayed as passed.

## Configuration and cost

`.env` is loaded on the server and excluded from Git. Only whether a key is configured is exposed to the frontend.

```dotenv
OPENAI_API_KEY=...
OPENAI_CHAT_MODEL=gpt-5.6-luna
OPENAI_HEAVY_MODEL=gpt-6-astra
OPENAI_VALIDATOR_MODEL=gpt-5.6-sol
SYMPLEX_MAX_USD=5
```

The first workspace creation freezes the ceiling. Reopening does not reset or increase it. Use a separate explicit workspace for a separately budgeted experiment (`--workspace PATH`); do not use new workspaces to evade an agreed total spend.

Campaign problems retain their persisted allocation across solver calls and workbench actions. Steering is saved while another runner owns the problem; a waiting checkpoint resumes under the same allocation when the execution lock is available. Creating a new job or reopening the server does not reset case usage.

Default global caps: 64 model requests, 250k input tokens, 50k output tokens, 900 numerical-worker seconds, 20 evidence requests and $5. Model pricing is a conservative Standard short-context estimate including the upper cache-write input rate, verified September 10, 2026. All requests use the Standard service tier. Hosted compute reserves four 1 GB container charges conservatively; API billing may reuse a container. Tool-generated context means these are local reservations and estimates, not a provider-enforced spending limit. Failed requests retain reservations when usage is unknown. The local dollar estimate is not an account invoice. Changing model IDs requires a verified rate in `infrastructure/providers.py`. Automatic SDK retries are disabled so failed requests cannot bypass accounting.

Cancellation stops the numerical worker promptly and prevents subsequent agent steps; an in-flight SDK request may finish. Completed jobs are idempotently returned. Failed solver checkpoints resume explicitly without repeating completed actions; a paused clarification resumes when the user supplies steering. New assumptions or context versions create a new investigation identity. Interrupted/failed jobs retain checkpoints but are not automatically restarted through a consumed confirmation seal. This is not a durable distributed job queue.

## Validation and commands

```sh
uv run python -m unittest discover -s tests -v
node --check symplex/interfaces/web/common.js
uv run python -m symplex doctor
uv run python -m symplex run --synthetic
uv run python -m symplex run --data /path/to/prepared-p1.json --live
uv run python -m symplex acquire-bamtwoogle
uv run python -m symplex research-pilot
```

`doctor` checks account model access without generating text. `scripts/live_smoke.py` is an explicit paid integration test: Symplex's own agent designs and simulates a hypothetical planning system. It does not generate an empirical site recommendation.

See [architecture](docs/architecture.md), [data contract](docs/data-contract.md), and the in-app [design brief](symplex/interfaces/web/research.html). The original user specification is preserved in `docs/master-specification.md`; current implementation status is described here rather than implied by that specification.

## Repository layout

```text
symplex/
  core/                  User inputs, problem DNA, proposal contracts
  agents/
    solver.py            Adaptive controller, checkpoints, user steering
    tools.py             Capability registry and trusted dispatch
    evolution.py         Prepared P1 program mutation and archive
    search_policy.py     Protocol-scoped numerical diagnostic archive
    input_state.py       Input revisions and explicit problem reframing
    context.py           Scoped working memory
    profiles.json        Depth, reasoning effort and permission envelopes
    prompts/             Proposer, collaborator, architect, metareasoner,
                         system modeler and simulation engineer instructions
  modeling/              Solution graphs, numerical models, asset contracts
    extensions.json      Scientific SDK families and validation requirements
  evidence/              Data admission, Croissant acquisition, spatial inputs
  connectors/            Compute, literature, tables, retrieval, RDF,
                         visualization and connector contracts
  evaluation/            Protected metrics, fixtures, method-policy comparison
  infrastructure/        OpenAI gateway, storage, budgets, numerical worker,
                         reviewed method registry and scoped canary rollback
  observability/         Nested spans and metadata-only trace export
  domains/manifests/     Five domain packs with ten full case specifications
  interfaces/
    api.py               Local HTTP boundary and background execution
    cli.py               Server and explicit experiment commands
    web/views/           Separate UI modules for problems, evidence, models,
                         scenarios, modeling studio, workflow and connectors
```

The server binds to loopback and checks origin/session tokens for writes. It is single-user software, without production authentication, tenant isolation or a public deployment configuration.

See [connector and simulator design](docs/connectors-and-simulation.md) and [verification record](docs/verification.md). The registries define supported tools and SDKs; problem-specific equations and mechanisms are proposed by the agent. Fixed numerical templates are optional, explicitly scoped capabilities rather than a universal modeling method.
