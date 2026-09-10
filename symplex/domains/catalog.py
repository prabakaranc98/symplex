import json
from pathlib import Path


def domains():
    return [
        json.loads(p.read_text())
        for p in sorted(Path(__file__).with_name("manifests").glob("*.json"))
    ]


def use_cases():
    return json.loads(Path(__file__).with_name("use_cases.json").read_text())


DATASETS = [
    {
        "id": "bamtwoogle",
        "name": "BamTwoogle",
        "publisher": "Google Research",
        "domain": "Research methods",
        "use_case": "When does checking intermediate evidence improve answers to multi-step research questions?",
        "description": "100 human-authored information-seeking questions. Compare frozen research procedures on reserved questions.",
        "url": "https://github.com/google-research-datasets/BamTwoogle",
        "license": "CC-BY-4.0",
        "size": "About 10 KB",
        "status": "download_available",
        "limitations": "Public benchmark contamination; exact-match scoring can miss equivalent answers; not a scientific forecasting benchmark.",
    },
    {
        "id": "caravan",
        "name": "Caravan",
        "publisher": "Hydrology community / Google Research collaborators",
        "domain": "Climate & agriculture",
        "use_case": "Which catchments need extra observation after a rainfall deficit?",
        "description": "Compare lagged-flow and nonlinear storage models for low-flow recovery; add evidence about basin attributes.",
        "url": "https://github.com/kratzert/Caravan",
        "license": "Check each contributing dataset",
        "size": "Prepare a basin subset",
        "status": "specification_only",
        "limitations": "Reanalysis is not a point-in-time weather forecast. Original and MultiMet daily timezone conventions differ.",
    },
    {
        "id": "power2019",
        "name": "Data Center Power 2019",
        "publisher": "Google",
        "domain": "Infrastructure",
        "use_case": "Which power domains are approaching a peak that merits an operator review?",
        "description": "57 power domains. Compare persistence with shared-demand residual models on contiguous held-out blocks.",
        "url": "https://github.com/google/cluster-data/blob/master/PowerData2019.md",
        "license": "CC-BY",
        "size": "Prepare a time slice",
        "status": "specification_only",
        "limitations": "Historical utilization does not establish the effect of a new capping policy.",
    },
    {
        "id": "p1",
        "name": "Kalshi resolved-event archive",
        "publisher": "Kalshi",
        "domain": "Prediction markets",
        "use_case": "Do exact logical event relations improve forecast reliability?",
        "description": "Prepared contract definitions, timestamped price observations and binary resolution labels.",
        "url": "https://docs.kalshi.com/",
        "license": "Verify selected archive terms",
        "size": "Prepared JSON slice",
        "status": "awaiting_prepared_data",
        "limitations": "Quote prices are proxies; logical coherence is not accuracy or executable arbitrage.",
    },
]

MODULES = [
    ("Evidence synthesis", "evidence/synthesis.py", "Hypothesis-linked claims, source dependence, disagreements and evidence gaps", "implemented"),
    ("Design and method evolution", "agents/improvement.py", "Separate intervention designs and uninstalled method candidates with explicit evaluation requirements", "implemented"),
    ("Decision synthesis", "agents/outcomes.py", "Artifact-linked claims, alternatives, next actions and reversal conditions", "implemented"),
    (
        "Complex-system contracts",
        "modeling/complex_system.py",
        "Typed modalities, units, clocks, hybrid couplings, hypotheses and decision endpoints",
        "implemented",
    ),
    (
        "Scientific reasoning roles",
        "agents/scientific_cycle.py",
        "Agent-authored system representation and separate hypothesis testability review",
        "implemented",
    ),
    (
        "Comparative experiment contracts",
        "evaluation/experiments.py",
        "Frozen protocols, executed-result admission, deterministic metric comparison and Pareto tradeoffs",
        "exploratory_comparison",
    ),
    (
        "Hosted modeling studio",
        "connectors/compute.py",
        "Astra-authored Python, multimodal context, actual plots and reusable model outputs",
        "implemented",
    ),
    (
        "Connector registry",
        "connectors/registry.py",
        "Discoverable data, semantics, research, compute, visualization and trace contracts",
        "implemented",
    ),
    (
        "Agent tools",
        "agents/tools.py",
        "Registered handlers with explicit capability permissions",
        "implemented",
    ),
    (
        "Role prompts",
        "agents/prompts/",
        "Versioned role instructions and prompt digests in run manifests",
        "implemented",
    ),
    (
        "Working memory",
        "agents/context.py",
        "Problem-scoped context with current critiques and model results; protected labels excluded",
        "implemented",
    ),
    (
        "Observability",
        "observability/tracing.py",
        "Nested OpenTelemetry spans, persisted metadata and JSONL exports",
        "implemented",
    ),
    (
        "Problem-solving agent",
        "agents/solver.py",
        "Adaptive tool selection, bounded revisits, persistent actions and deliverables",
        "implemented",
    ),
    (
        "System simulation",
        "modeling/dynamics.py",
        "Stochastic feedback graphs, lag, interventions, paired rollouts and structural ablation",
        "conditional_simulation",
    ),
    (
        "Solution specification",
        "modeling/solutions.py",
        "Subproblems, rival mechanisms, component graph and decision contract",
        "implemented",
    ),
    (
        "Problem contracts",
        "core/contracts.py",
        "Typed objectives, assumptions, patches and action proposals",
        "implemented",
    ),
    (
        "Evidence & provenance",
        "evidence/admission.py",
        "Cutoff checks, exact semantics, versioned source hashes",
        "implemented",
    ),
    (
        "Dataset discovery",
        "evaluation/research.py",
        "Source catalog, bounded downloads, Croissant metadata validation",
        "implemented",
    ),
    (
        "Model gateway",
        "infrastructure/providers.py",
        "OpenAI SDK; chat, heavy reasoning and validation roles",
        "implemented",
    ),
    (
        "Training & composition",
        "modeling/templates.py",
        "Scalar fitting, probability projection, age-aware shrinkage",
        "implemented",
    ),
    (
        "Evolution & metareasoning",
        "agents/evolution.py",
        "Parent–child grammar, archive cells, guarded research actions",
        "implemented",
    ),
    (
        "Numerical runner",
        "infrastructure/runner.py",
        "Credential-free trusted subprocess with CPU and wall limits",
        "implemented",
    ),
    (
        "Protected evaluation",
        "evaluation/metrics.py",
        "Host-owned scoring, whole-event splits, single-use confirmation",
        "implemented",
    ),
    (
        "Persistent memory",
        "infrastructure/storage.py",
        "SQLite, content hashes, jobs, budget reservations, stale descendants",
        "implemented",
    ),
    (
        "Method history",
        "evaluation/research.py",
        "Frozen research policy comparison on disjoint benchmark questions",
        "implemented",
    ),
    (
        "Model assets",
        "modeling/assets.py",
        "Scientific, API and Hugging Face descriptors with compatibility checks",
        "implemented",
    ),
    (
        "Coding workspace",
        "modeling/assets.py",
        "Draft components and hosted agent-authored code; no local arbitrary-code execution",
        "implemented",
    ),
    (
        "Workbench & chat",
        "interfaces/api.py",
        "FastAPI, persistent conversations, evidence and decision exports",
        "implemented",
    ),
]
