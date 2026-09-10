"""Discoverable connector contracts with a trusted, explicit dispatch map."""

import os
from dataclasses import asdict, dataclass

from pydantic import BaseModel, ConfigDict, Field

from symplex.core.contracts import Invalid


class ConnectorInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    connector: str
    problem_id: str
    query: str = Field(default="", max_length=2000)
    artifact_id: str = ""
    url: str = Field(default="", max_length=2048)
    filename: str = Field(default="", max_length=120)
    expected_sha256: str = Field(default="", max_length=64)
    expected_csv_columns: list[str] = Field(default_factory=list, max_length=100)


@dataclass(frozen=True)
class Connector:
    id: str
    category: str
    name: str
    status: str
    operations: tuple
    package: str
    documentation: str
    limits: str
    permission: str


CATALOG = [
    Connector(
        "public_artifact",
        "data",
        "Public source files",
        "ready",
        ("fetch_artifact",),
        "httpx",
        "https://www.python-httpx.org/advanced/transports/",
        "2 MB; operator-owned exact-host allowlist; pinned public-IP HTTPS; source bytes and structural profiles are not measurement validation",
        "network.dataset",
    ),
    Connector(
        "hosted_compute",
        "compute",
        "Astra simulation engineer",
        "requires_key",
        ("run",),
        "openai",
        "https://developers.openai.com/api/docs/guides/tools-code-interpreter",
        "Agent-authored Python in hosted sandbox; 1 GB; 4 tool calls; code, plots and artifacts persisted; specialist imports are probed",
        "compute.hosted",
    ),
    Connector(
        "local_retrieval",
        "semantics",
        "Scoped artifact retrieval",
        "ready",
        ("search",),
        "scikit-learn",
        "https://scikit-learn.org/stable/modules/feature_extraction.html",
        "TF-IDF lexical relevance; no embedding or entailment claim",
        "artifacts.read",
    ),
    Connector(
        "system_rdf",
        "semantics",
        "System ontology projection",
        "ready",
        ("project",),
        "rdflib",
        "https://rdflib.readthedocs.io/en/stable/",
        "RDF triples preserve proposed mechanisms; no causal verification",
        "artifacts.read",
    ),
    Connector(
        "csv_profile",
        "data",
        "Uploaded tables",
        "ready",
        ("profile",),
        "duckdb",
        "https://duckdb.org/docs/stable/clients/python/overview",
        "Fixed read-only queries; CSV up to 1 MB; no arbitrary SQL",
        "artifacts.read",
    ),
    Connector(
        "crossref",
        "research",
        "Crossref literature discovery",
        "ready",
        ("search",),
        "httpx",
        "https://www.crossref.org/documentation/retrieve-metadata/rest-api/",
        "5 metadata records; 500 KB; does not retrieve full papers",
        "network.research",
    ),
    Connector(
        "web_research",
        "research",
        "Astra evidence search",
        "requires_key",
        ("research",),
        "openai",
        "https://developers.openai.com/api/docs/guides/tools-web-search",
        "Two search calls per action; citations retained; synthesis is not validation",
        "network.research",
    ),
    Connector(
        "croissant",
        "data",
        "Google datasets / Croissant",
        "ready",
        ("acquire_bamtwoogle",),
        "mlcroissant",
        "https://docs.mlcommons.org/croissant/",
        "One verified source mapping; arbitrary datasets require a reviewed mapping",
        "network.dataset",
    ),
    Connector(
        "scenario_chart",
        "visualization",
        "Scenario chart export",
        "ready",
        ("vega_lite",),
        "Vega-Lite specification",
        "https://vega.github.io/vega-lite/docs/spec.html",
        "Inline data from executed reports only; UI has native trajectory charts",
        "artifacts.read",
    ),
    Connector(
        "local_trace",
        "observability",
        "Agent and model tracing",
        "ready",
        ("export",),
        "opentelemetry-sdk",
        "https://opentelemetry.io/docs/languages/python/instrumentation/",
        "Metadata-only trace export; prompts and source documents omitted; external exporters disabled",
        "artifacts.read",
    ),
    Connector(
        "openai_deep_research",
        "research",
        "Dedicated deep-research service",
        "planned",
        ("start", "poll", "cancel"),
        "openai",
        "https://developers.openai.com/api/docs/guides/deep-research",
        "Requires durable background response polling, dedicated pricing envelope and model-access test",
        "network.research",
    ),
    Connector(
        "remote_mcp",
        "enterprise",
        "Remote MCP data sources",
        "planned",
        ("list", "search", "fetch"),
        "mcp / openai",
        "https://developers.openai.com/api/docs/guides/tools-connectors-mcp",
        "Per-server allowlist, scoped OAuth and tool schemas required; no inherited plugin credentials",
        "network.enterprise",
    ),
    Connector(
        "object_store",
        "data",
        "S3 / Parquet / Arrow",
        "planned",
        ("list", "read_subset"),
        "boto3 / pyarrow",
        "https://arrow.apache.org/docs/python/parquet.html",
        "Read-only prefixes, schema and partition selection, license and point-in-time provenance",
        "network.dataset",
    ),
    Connector(
        "earth_observation",
        "data",
        "STAC / climate arrays",
        "planned",
        ("search", "subset"),
        "pystac-client / xarray",
        "https://pystac-client.readthedocs.io/en/stable/",
        "Optional spatial adapter; CRS, units, grid/time alignment and chunk budgets required",
        "network.dataset",
    ),
    Connector(
        "scientific_compute",
        "compute",
        "Domain simulator backends",
        "planned",
        ("validate", "run", "collect"),
        "PyMC / SciPy / BioSimulators adapters",
        "https://docs.biosimulators.org/",
        "Select by governing mechanisms; immutable images, units, solver tolerances and independent validation required",
        "compute.isolated",
    ),
    Connector(
        "interactive_graphics",
        "visualization",
        "Graph, map and multidimensional scenes",
        "planned",
        ("render",),
        "Cytoscape.js / deck.gl",
        "https://js.cytoscape.org/",
        "Current UI renders component graphs and trajectories; advanced graph/map engines not installed",
        "artifacts.read",
    ),
]


def catalog():
    return [
        dict(
            asdict(c),
            status=("ready" if os.getenv("OPENAI_API_KEY") else "requires_key")
            if c.id in ("web_research", "hosted_compute")
            else c.status,
        )
        for c in CATALOG
    ]


def execute(store, budget, provider, request, permissions):
    entry = next((c for c in CATALOG if c.id == request.connector), None)
    if (
        entry is None
        or entry.status == "planned"
        or entry.permission not in permissions
    ):
        raise Invalid("Connector unavailable or outside the permission envelope")
    problem = store.get(request.problem_id)
    if problem["kind"] != "workspace_problem" or problem["stale"]:
        raise Invalid("Select a current problem")
    from symplex.observability.tracing import span

    with span(store, request.problem_id, "connector." + entry.id):
        result = HANDLERS[entry.id](store, budget, provider, request)
        ident = store.put(
            "connector_result",
            {"connector": entry.id, "request": request.model_dump(), "result": result},
            request.problem_id,
        )
    return {"id": ident, "result": result}


def _retrieval(s, b, p, r):
    from .retrieval import search

    return search(s, r.problem_id, r.query)


def _rdf(s, b, p, r):
    from .semantics import project

    return project(s, r.problem_id)


def _table(s, b, p, r):
    from .tables import profile

    return profile(s, r.problem_id, r.artifact_id)


def _literature(s, b, p, r):
    from .literature import search

    return search(s, b, r.problem_id, r.query)


def _web(s, b, p, r):
    if p is None:
        raise Invalid("OpenAI API key required")
    result = p.chat(
        [{"role": "user", "content": r.query}],
        {"problem": s.get(r.problem_id)["data"]},
        r.problem_id,
        research=True,
    )
    ident = s.put("evidence_note", result, r.problem_id)
    return {"evidence_id": ident, **result}


def _croissant(s, b, p, r):
    from symplex.evidence.acquisition import download_bam

    return download_bam(s, b)


def _public_artifact(s, b, p, r):
    from .public_artifacts import PublicArtifactRequest, fetch

    return fetch(s, b, r.problem_id, PublicArtifactRequest(
        url=r.url, filename=r.filename, expected_sha256=r.expected_sha256,
        expected_csv_columns=r.expected_csv_columns,
    ))


def _chart(s, b, p, r):
    from .visualization import scenario_chart

    return scenario_chart(s, r.problem_id, r.artifact_id)


def _trace(s, b, p, r):
    from symplex.observability.tracing import export_trace

    return {"events": export_trace(s, r.problem_id), "payload_policy": "metadata_only"}


def _compute(s, b, p, r):
    from .compute import run

    if p is None:
        raise Invalid("OpenAI API key required")
    ident = run(s, b, p, r.problem_id, r.query)
    return {"package_id": ident, **s.get(ident)["data"]}


HANDLERS = {
    "public_artifact": _public_artifact,
    "hosted_compute": _compute,
    "local_retrieval": _retrieval,
    "system_rdf": _rdf,
    "csv_profile": _table,
    "crossref": _literature,
    "web_research": _web,
    "croissant": _croissant,
    "scenario_chart": _chart,
    "local_trace": _trace,
}
