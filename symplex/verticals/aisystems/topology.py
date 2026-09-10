"""The AI system as a typed graph: routers, models, tools, retrievers, verifiers, humans.

Structure only. This module declares what a deployment could look like and checks that the
declaration is executable at all: every node reachable, every retry cycle bounded, every
fallback chain terminating at something that can answer. It measures no deployed system and
calls no model. Node parameters (service time, cost, capability) are stated assumptions that
the rest of the vertical consumes; they are not fitted to any observed service.
"""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from symplex.core.contracts import Invalid, digest
from symplex.modeling.complex_system import ClosedContract, Identifier, Text

MAX_NODES = 64
MAX_EDGES = 256
MAX_FALLBACK_DEPTH = 8
MAX_CYCLE_BUDGET = 64
PROBABILITY_TOLERANCE = 1e-9

NODE_KINDS = (
    "router",
    "model",
    "tool",
    "retriever",
    "verifier",
    "human_reviewer",
    "cache",
    "queue",
    "memory",
)
MODEL_TIERS = ("cheap", "mid", "frontier")
ARCHITECTURES = (
    "centralized_planner",
    "specialist_hierarchy",
    "debate",
    "verifier_loop",
    "swarm",
    "workflow_graph",
)

SCOPE = (
    "Declared architecture with host-checked structure. Reachability, retry bounds and "
    "fallback termination are properties of the declaration, not evidence that any deployed "
    "system has this shape or these service parameters."
)

NodeKind = Literal[
    "router",
    "model",
    "tool",
    "retriever",
    "verifier",
    "human_reviewer",
    "cache",
    "queue",
    "memory",
]
Tier = Literal["cheap", "mid", "frontier"]
ServiceLaw = Literal["exponential", "lognormal", "pareto", "deterministic"]
Architecture = Literal[
    "centralized_planner",
    "specialist_hierarchy",
    "debate",
    "verifier_loop",
    "swarm",
    "workflow_graph",
]


class RetryPolicy(ClosedContract):
    """Per-edge transient-failure retry. Bounded by contract; there is no infinite retry."""

    max_attempts: Annotated[int, Field(ge=1, le=5)]
    backoff_seconds: Annotated[float, Field(ge=0.0, le=60.0)]
    jitter: Annotated[float, Field(ge=0.0, le=1.0)]


class Node(ClosedContract):
    """A typed station. Service, cost, energy and capability are declared assumptions."""

    id: Identifier
    kind: NodeKind
    name: Text
    tier: Tier | None
    servers: Annotated[int, Field(ge=1, le=512)]
    queue_capacity: Annotated[int, Field(ge=0, le=4096)]
    service_seconds: Annotated[float, Field(gt=0.0, le=600.0)]
    service_law: ServiceLaw
    service_dispersion: Annotated[float, Field(ge=0.0, le=4.0)]
    rate_limit_per_second: Annotated[float, Field(gt=0.0, le=10000.0)] | None
    rate_burst: Annotated[int, Field(ge=1, le=1024)]
    batch_size: Annotated[int, Field(ge=1, le=64)]
    batch_wait_seconds: Annotated[float, Field(ge=0.0, le=10.0)]
    capability: Annotated[float, Field(ge=0.0, le=1.0)]
    cost_per_call_usd: Annotated[float, Field(ge=0.0, le=100.0)]
    energy_per_call_joules: Annotated[float, Field(ge=0.0, le=1_000_000.0)]
    compute_units: Annotated[float, Field(ge=0.0, le=1000.0)]
    timeout_seconds: Annotated[float, Field(gt=0.0, le=600.0)]

    @model_validator(mode="after")
    def kind_coherent(self):
        if (self.tier is not None) != (self.kind == "model"):
            raise ValueError("Only model nodes carry a tier, and every model node needs one")
        if self.kind == "human_reviewer" and self.servers > 64:
            raise ValueError("Human reviewer concurrency must stay in a staffable range")
        if self.service_law in ("lognormal", "pareto") and self.service_dispersion <= 0:
            raise ValueError("Heavy-tailed service laws require a positive dispersion")
        if self.service_law == "pareto" and self.service_dispersion <= 1.0:
            raise ValueError("Pareto service requires tail index above 1 for a finite mean")
        if self.batch_size > 1 and self.kind not in ("model", "verifier", "retriever"):
            raise ValueError("Batching is only declared for model, verifier or retriever nodes")
        return self


class Edge(ClosedContract):
    """A routing option. `max_traversals` is the per-request loop bound; None means unbounded."""

    source: Identifier
    target: Identifier
    probability: Annotated[float, Field(ge=0.0, le=1.0)]
    timeout_seconds: Annotated[float, Field(gt=0.0, le=600.0)]
    retry: RetryPolicy
    fallback: Identifier | None
    max_traversals: Annotated[int, Field(ge=1, le=16)] | None
    role: Text

    @model_validator(mode="after")
    def distinct_fallback(self):
        if self.fallback == self.source:
            raise ValueError("A fallback cannot be the node that already failed")
        if self.source == self.target and self.max_traversals is None:
            raise ValueError("A self-loop must declare a finite max_traversals")
        return self


class Topology(ClosedContract):
    id: Identifier
    name: Text
    architecture: Architecture
    entry: Identifier
    terminals: list[Identifier] = Field(min_length=1, max_length=8)
    nodes: list[Node] = Field(min_length=2, max_length=MAX_NODES)
    edges: list[Edge] = Field(min_length=1, max_length=MAX_EDGES)
    assumptions: list[Text] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def referential(self):
        ids = [n.id for n in self.nodes]
        if len(set(ids)) != len(ids):
            raise ValueError("Node identifiers must be unique")
        known = set(ids)
        if self.entry not in known:
            raise ValueError("Entry node is not declared")
        if not set(self.terminals) <= known or len(set(self.terminals)) != len(self.terminals):
            raise ValueError("Terminals must be unique declared nodes")
        for edge in self.edges:
            if edge.source not in known or edge.target not in known:
                raise ValueError("Edge references an undeclared node: " + edge.source)
            if edge.fallback is not None and edge.fallback not in known:
                raise ValueError("Fallback references an undeclared node: " + str(edge.fallback))
        if len({(e.source, e.target) for e in self.edges}) != len(self.edges):
            raise ValueError("Duplicate edge between the same pair of nodes")
        return self

    def node(self, ident):
        for candidate in self.nodes:
            if candidate.id == ident:
                return candidate
        raise Invalid("Unknown node: " + str(ident))

    def by_kind(self, kind):
        return [n for n in self.nodes if n.kind == kind]

    def model_tier(self, tier):
        for candidate in self.nodes:
            if candidate.kind == "model" and candidate.tier == tier:
                return candidate
        raise Invalid("No model node at tier: " + str(tier))

    def out_edges(self, ident):
        return [e for e in self.edges if e.source == ident]


def _strongly_connected(adjacency, nodes):
    """Tarjan's SCC, iterative, so a deep graph cannot exhaust the interpreter stack."""
    index = {}
    low = {}
    on_stack = {}
    stack = []
    order = [0]
    components = []
    for root in nodes:
        if root in index:
            continue
        work = [(root, iter(adjacency.get(root, ())))]
        index[root] = low[root] = order[0]
        order[0] += 1
        stack.append(root)
        on_stack[root] = True
        while work:
            node, children = work[-1]
            advanced = False
            for child in children:
                if child not in index:
                    index[child] = low[child] = order[0]
                    order[0] += 1
                    stack.append(child)
                    on_stack[child] = True
                    work.append((child, iter(adjacency.get(child, ()))))
                    advanced = True
                    break
                if on_stack.get(child):
                    low[node] = min(low[node], index[child])
            if advanced:
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[node])
            if low[node] == index[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack[member] = False
                    component.append(member)
                    if member == node:
                        break
                components.append(sorted(component))
    return components


def _cyclic_components(edges, nodes):
    adjacency = {}
    for edge in edges:
        adjacency.setdefault(edge.source, []).append(edge.target)
    selfish = {e.source for e in edges if e.source == e.target}
    return [
        component
        for component in _strongly_connected(adjacency, nodes)
        if len(component) > 1 or component[0] in selfish
    ]


def _reachable(edges, start):
    adjacency = {}
    for edge in edges:
        adjacency.setdefault(edge.source, []).append(edge.target)
    seen = {start}
    frontier = [start]
    while frontier:
        node = frontier.pop()
        for target in adjacency.get(node, ()):
            if target not in seen:
                seen.add(target)
                frontier.append(target)
    return seen


def _reaches_terminal(edges, terminals):
    reverse = {}
    for edge in edges:
        reverse.setdefault(edge.target, []).append(edge.source)
    seen = set(terminals)
    frontier = list(terminals)
    while frontier:
        node = frontier.pop()
        for source in reverse.get(node, ()):
            if source not in seen:
                seen.add(source)
                frontier.append(source)
    return seen


def validate(topology):
    """Structural admission. Raises Invalid on a topology that cannot be executed honestly.

    Rejects: unreachable nodes, dead ends that can never deliver a response, retry cycles with
    no finite traversal bound, loop budgets beyond the declared envelope, fallback chains that
    revisit a node or never reach a node able to answer, and degenerate routing mass.
    """
    if not isinstance(topology, Topology):
        topology = Topology.model_validate(topology)
    ids = [n.id for n in topology.nodes]
    reachable = _reachable(topology.edges, topology.entry)
    unreachable = sorted(set(ids) - reachable)
    if unreachable:
        raise Invalid("Unreachable nodes: " + ", ".join(unreachable))
    productive = _reaches_terminal(topology.edges, topology.terminals)
    dead_ends = sorted(set(ids) - productive)
    if dead_ends:
        raise Invalid("Nodes that can never deliver a response: " + ", ".join(dead_ends))

    unbounded_edges = [e for e in topology.edges if e.max_traversals is None]
    unbounded_cycles = _cyclic_components(unbounded_edges, ids)
    if unbounded_cycles:
        raise Invalid(
            "Unbounded retry cycle: "
            + "; ".join(" -> ".join(c) for c in unbounded_cycles)
        )
    cycles = _cyclic_components(topology.edges, ids)
    budgets = {}
    for component in cycles:
        member = set(component)
        internal = [
            e
            for e in topology.edges
            if e.source in member and e.target in member and e.max_traversals is not None
        ]
        budget = sum(e.max_traversals * e.retry.max_attempts for e in internal)
        budgets["|".join(component)] = budget
        if budget > MAX_CYCLE_BUDGET:
            raise Invalid(
                "Loop budget exceeds the declared envelope in cycle: " + "|".join(component)
            )

    fallback_targets = {}
    for edge in topology.edges:
        if edge.fallback is None:
            continue
        chain = [edge.fallback]
        seen = {edge.source, edge.fallback}
        while True:
            successors = sorted(
                {e.fallback for e in topology.out_edges(chain[-1]) if e.fallback is not None}
            )
            if not successors:
                break
            if len(chain) >= MAX_FALLBACK_DEPTH:
                raise Invalid("Fallback chain exceeds depth bound from edge: " + edge.source)
            following = successors[0]
            if following in seen:
                raise Invalid(
                    "Non-terminating fallback cycle: " + " -> ".join(chain + [following])
                )
            seen.add(following)
            chain.append(following)
        if chain[-1] not in productive:
            raise Invalid("Fallback chain never reaches a node that can answer: " + chain[-1])
        fallback_targets[f"{edge.source}->{edge.target}"] = chain

    routing_mass = {}
    for ident in ids:
        outgoing = topology.out_edges(ident)
        if not outgoing:
            if ident not in topology.terminals:
                raise Invalid("Non-terminal node has no outgoing edge: " + ident)
            continue
        mass = sum(e.probability for e in outgoing)
        routing_mass[ident] = mass
        if mass <= 0 or mass > 1 + PROBABILITY_TOLERANCE:
            raise Invalid("Routing probability mass out of range at node: " + ident)
    if abs(routing_mass.get(topology.entry, 0.0) - 1.0) > 1e-6:
        raise Invalid("Entry node must dispatch a full unit of routing probability")

    for tier in MODEL_TIERS:
        topology.model_tier(tier)
    capabilities = [topology.model_tier(t).capability for t in MODEL_TIERS]
    if not capabilities[0] < capabilities[1] < capabilities[2]:
        raise Invalid("Model tiers must be strictly ordered in capability: cheap < mid < frontier")

    return {
        "valid": True,
        "topology_id": topology.id,
        "architecture": topology.architecture,
        "node_count": len(topology.nodes),
        "edge_count": len(topology.edges),
        "unreachable_nodes": [],
        "dead_end_nodes": [],
        "cyclic_components": cycles,
        "cycle_loop_budgets": budgets,
        "unbounded_retry_cycles": [],
        "fallback_chains": fallback_targets,
        "routing_mass": routing_mass,
        "kinds_present": sorted({n.kind for n in topology.nodes}),
        "topology_digest": digest(topology.model_dump()),
        "checks": [
            "reachability_from_entry",
            "every_node_reaches_a_terminal",
            "no_unbounded_retry_cycle",
            "loop_budget_within_envelope",
            "fallback_chains_terminate",
            "routing_probability_mass",
            "model_tier_capability_order",
        ],
        "scope": SCOPE,
    }


_KIND_GROUP = {
    "router": "control",
    "queue": "control",
    "model": "inference",
    "verifier": "assurance",
    "human_reviewer": "assurance",
    "tool": "external",
    "retriever": "external",
    "cache": "state",
    "memory": "state",
}


def to_graph_json(topology):
    """A frontend-ready graph. Layers are a drawing hint, not a claim about execution order."""
    if not isinstance(topology, Topology):
        topology = Topology.model_validate(topology)
    report = validate(topology)
    layer = {topology.entry: 0}
    frontier = [topology.entry]
    while frontier:
        node = frontier.pop(0)
        for edge in topology.out_edges(node):
            if edge.target not in layer:
                layer[edge.target] = layer[node] + 1
                frontier.append(edge.target)
    return {
        "id": topology.id,
        "name": topology.name,
        "architecture": topology.architecture,
        "entry": topology.entry,
        "terminals": list(topology.terminals),
        "nodes": [
            {
                "id": n.id,
                "kind": n.kind,
                "group": _KIND_GROUP[n.kind],
                "label": n.name,
                "tier": n.tier,
                "layer": layer.get(n.id, 0),
                "servers": n.servers,
                "queue_capacity": n.queue_capacity,
                "service_seconds": n.service_seconds,
                "service_law": n.service_law,
                "rate_limit_per_second": n.rate_limit_per_second,
                "cost_per_call_usd": n.cost_per_call_usd,
                "energy_per_call_joules": n.energy_per_call_joules,
                "capability": n.capability,
            }
            for n in topology.nodes
        ],
        "edges": [
            {
                "source": e.source,
                "target": e.target,
                "probability": e.probability,
                "timeout_seconds": e.timeout_seconds,
                "max_attempts": e.retry.max_attempts,
                "backoff_seconds": e.retry.backoff_seconds,
                "fallback": e.fallback,
                "max_traversals": e.max_traversals,
                "role": e.role,
                "cyclic": any(
                    e.source in c and e.target in c for c in report["cyclic_components"]
                ),
            }
            for e in topology.edges
        ],
        "legend": {
            "control": "routing and admission",
            "inference": "model calls by tier",
            "assurance": "verification and human review",
            "external": "tools and retrieval, the parts that fail independently",
            "state": "cache and shared memory, the parts that carry errors forward",
        },
        "topology_digest": report["topology_digest"],
        "validation": {k: report[k] for k in ("valid", "cyclic_components", "cycle_loop_budgets")},
        "scope": SCOPE,
    }


def architecture_profile(topology):
    """Reduce the graph to the few execution facts the simulator honours.

    The simulator does not walk arbitrary graphs; it walks a policy-chosen station sequence.
    These derived counts are how the declared architecture actually changes that sequence.
    """
    if not isinstance(topology, Topology):
        topology = Topology.model_validate(topology)
    verify_rounds = max(
        [
            e.max_traversals or 1
            for e in topology.edges
            if topology.node(e.source).kind == "verifier"
            and topology.node(e.target).kind == "model"
        ]
        or [0]
    )
    peer_edges = [
        e
        for e in topology.edges
        if topology.node(e.source).kind == "model" and topology.node(e.target).kind == "model"
    ]
    parallel = {
        "centralized_planner": 2,
        "specialist_hierarchy": 1,
        "debate": 3,
        "verifier_loop": 1,
        "swarm": 2,
        "workflow_graph": 1,
    }[topology.architecture]
    return {
        "architecture": topology.architecture,
        "model_calls_per_request": parallel,
        "max_verify_rounds": verify_rounds,
        "peer_delegation_edges": len(peer_edges),
        "has_cache": bool(topology.by_kind("cache")),
        "has_memory": bool(topology.by_kind("memory")),
        "has_human_review": bool(topology.by_kind("human_reviewer")),
        "aggregation": "majority_of_parallel_calls" if parallel > 1 else "single_call",
        "scope": SCOPE,
    }


_DEFAULT_RETRY = {"max_attempts": 2, "backoff_seconds": 0.25, "jitter": 0.5}

_FALLBACK_BY_SOURCE = {
    "router": "model_mid",
    "cache": "model_cheap",
    "retriever": "model_mid",
    "tool_search": "model_frontier",
    "model_cheap": "model_mid",
    "model_mid": "model_frontier",
    "model_frontier": "human_review",
    "verifier": "human_review",
}


def _station(ident, kind, name, **overrides):
    spec = {
        "id": ident,
        "kind": kind,
        "name": name,
        "tier": None,
        "servers": 16,
        "queue_capacity": 128,
        "service_seconds": 0.05,
        "service_law": "exponential",
        "service_dispersion": 0.0,
        "rate_limit_per_second": None,
        "rate_burst": 16,
        "batch_size": 1,
        "batch_wait_seconds": 0.0,
        "capability": 0.0,
        "cost_per_call_usd": 0.0,
        "energy_per_call_joules": 0.0,
        "compute_units": 0.0,
        "timeout_seconds": 30.0,
    }
    spec.update(overrides)
    return Node.model_validate(spec)


def _standard_nodes(queue_scale, tool_rate_limit):
    def capacity(base):
        return max(0, int(round(base * queue_scale)))

    return [
        _station(
            "intake",
            "queue",
            "Ingress admission buffer",
            servers=64,
            queue_capacity=capacity(512),
            service_seconds=0.002,
            service_law="deterministic",
            timeout_seconds=60.0,
        ),
        _station(
            "router",
            "router",
            "Policy router",
            servers=32,
            queue_capacity=capacity(256),
            service_seconds=0.004,
            timeout_seconds=10.0,
        ),
        _station(
            "cache",
            "cache",
            "Semantic response cache",
            servers=64,
            queue_capacity=capacity(128),
            service_seconds=0.003,
            service_law="deterministic",
            capability=0.30,
            timeout_seconds=5.0,
        ),
        _station(
            "retriever",
            "retriever",
            "Document retriever",
            servers=8,
            queue_capacity=capacity(64),
            service_seconds=0.09,
            service_law="lognormal",
            service_dispersion=0.65,
            capability=0.70,
            cost_per_call_usd=0.00008,
            energy_per_call_joules=1.5,
            compute_units=0.2,
            timeout_seconds=8.0,
        ),
        _station(
            "tool_search",
            "tool",
            "External search tool",
            servers=6,
            queue_capacity=capacity(32),
            service_seconds=0.40,
            service_law="lognormal",
            service_dispersion=0.95,
            rate_limit_per_second=tool_rate_limit,
            rate_burst=12,
            capability=0.75,
            cost_per_call_usd=0.0006,
            energy_per_call_joules=4.0,
            compute_units=0.3,
            timeout_seconds=12.0,
        ),
        _station(
            "model_cheap",
            "model",
            "Small model",
            tier="cheap",
            servers=4,
            queue_capacity=capacity(128),
            service_seconds=0.45,
            service_law="lognormal",
            service_dispersion=0.70,
            batch_size=4,
            batch_wait_seconds=0.04,
            capability=0.45,
            cost_per_call_usd=0.00040,
            energy_per_call_joules=12.0,
            compute_units=1.0,
            timeout_seconds=20.0,
        ),
        _station(
            "model_mid",
            "model",
            "Mid model",
            tier="mid",
            servers=5,
            queue_capacity=capacity(64),
            service_seconds=1.10,
            service_law="lognormal",
            service_dispersion=0.80,
            capability=0.68,
            cost_per_call_usd=0.0030,
            energy_per_call_joules=60.0,
            compute_units=3.0,
            timeout_seconds=30.0,
        ),
        _station(
            "model_frontier",
            "model",
            "Frontier model",
            tier="frontier",
            servers=3,
            queue_capacity=capacity(24),
            service_seconds=2.60,
            service_law="lognormal",
            service_dispersion=0.90,
            rate_limit_per_second=4.0,
            rate_burst=8,
            capability=0.88,
            cost_per_call_usd=0.0200,
            energy_per_call_joules=320.0,
            compute_units=8.0,
            timeout_seconds=45.0,
        ),
        _station(
            "verifier",
            "verifier",
            "Answer verifier",
            servers=8,
            queue_capacity=capacity(64),
            service_seconds=0.35,
            service_law="lognormal",
            service_dispersion=0.60,
            capability=0.72,
            cost_per_call_usd=0.00080,
            energy_per_call_joules=18.0,
            compute_units=1.0,
            timeout_seconds=15.0,
        ),
        _station(
            "human_review",
            "human_reviewer",
            "Human reviewer pool",
            servers=3,
            queue_capacity=capacity(12),
            service_seconds=45.0,
            service_law="lognormal",
            service_dispersion=0.50,
            capability=0.95,
            cost_per_call_usd=1.20,
            energy_per_call_joules=0.0,
            compute_units=0.0,
            timeout_seconds=600.0,
        ),
        _station(
            "memory",
            "memory",
            "Shared episodic memory",
            servers=64,
            queue_capacity=capacity(256),
            service_seconds=0.005,
            service_law="deterministic",
            timeout_seconds=5.0,
        ),
        _station(
            "delivery",
            "queue",
            "Response egress",
            servers=128,
            queue_capacity=capacity(1024),
            service_seconds=0.001,
            service_law="deterministic",
            timeout_seconds=60.0,
        ),
    ]


_CORE_EDGES = (
    ("intake", "router", 1.0, 1, "admission buffer to router"),
    ("human_review", "delivery", 1.0, 1, "reviewed answer released"),
    ("memory", "delivery", 1.0, 1, "write-back then release"),
)

_ARCHITECTURE_EDGES = {
    "workflow_graph": (
        ("router", "cache", 0.20, 1, "cache lookup"),
        ("router", "retriever", 0.80, 1, "grounded path"),
        ("cache", "delivery", 0.60, 1, "cache hit answers directly"),
        ("cache", "model_cheap", 0.40, 1, "cache miss"),
        ("retriever", "model_mid", 0.70, 1, "grounded generation"),
        ("retriever", "model_cheap", 0.30, 1, "cheap grounded generation"),
        ("model_cheap", "verifier", 0.50, 1, "check the small model"),
        ("model_cheap", "memory", 0.20, 1, "persist result"),
        ("model_cheap", "delivery", 0.30, 1, "unchecked release"),
        ("model_mid", "tool_search", 0.30, 1, "tool call"),
        ("model_mid", "verifier", 0.50, 1, "check the mid model"),
        ("model_mid", "memory", 0.20, 1, "persist result"),
        ("tool_search", "verifier", 0.70, 1, "check the tool-grounded answer"),
        ("tool_search", "model_frontier", 0.30, 1, "hard synthesis after tool use"),
        ("model_frontier", "verifier", 0.60, 1, "check the frontier answer"),
        ("model_frontier", "memory", 0.20, 1, "persist result"),
        ("model_frontier", "delivery", 0.20, 1, "unchecked release"),
        ("verifier", "delivery", 0.80, 1, "verified release"),
        ("verifier", "human_review", 0.20, 1, "escalate to a person"),
    ),
    "verifier_loop": (
        ("router", "cache", 0.20, 1, "cache lookup"),
        ("router", "retriever", 0.80, 1, "grounded path"),
        ("cache", "delivery", 0.60, 1, "cache hit answers directly"),
        ("cache", "model_cheap", 0.40, 1, "cache miss"),
        ("retriever", "model_mid", 0.70, 1, "grounded generation"),
        ("retriever", "model_cheap", 0.30, 1, "cheap grounded generation"),
        ("model_cheap", "verifier", 0.60, 2, "check the small model"),
        ("model_cheap", "memory", 0.20, 1, "persist result"),
        ("model_cheap", "delivery", 0.20, 1, "unchecked release"),
        ("model_mid", "tool_search", 0.25, 2, "tool call"),
        ("model_mid", "verifier", 0.55, 2, "check the mid model"),
        ("model_mid", "memory", 0.20, 1, "persist result"),
        ("tool_search", "verifier", 1.00, 2, "check the tool-grounded answer"),
        ("model_frontier", "verifier", 0.80, 4, "check the frontier answer"),
        ("model_frontier", "memory", 0.20, 1, "persist result"),
        ("verifier", "delivery", 0.60, 1, "verified release"),
        ("verifier", "human_review", 0.15, 1, "escalate to a person"),
        ("verifier", "model_frontier", 0.25, 3, "rejected: regenerate at a stronger tier"),
    ),
    "centralized_planner": (
        ("router", "cache", 0.15, 1, "cache lookup"),
        ("router", "model_mid", 0.85, 1, "hand the request to the planner"),
        ("cache", "delivery", 0.60, 1, "cache hit answers directly"),
        ("cache", "model_mid", 0.40, 1, "cache miss to planner"),
        ("model_mid", "retriever", 0.25, 2, "planner requests evidence"),
        ("model_mid", "tool_search", 0.20, 2, "planner requests a tool"),
        ("model_mid", "model_cheap", 0.25, 2, "planner delegates a subtask"),
        ("model_mid", "verifier", 0.20, 2, "planner submits for checking"),
        ("model_mid", "memory", 0.10, 1, "planner writes shared state"),
        ("retriever", "model_mid", 1.00, 2, "evidence returned to planner"),
        ("tool_search", "model_mid", 1.00, 2, "tool result returned to planner"),
        ("model_cheap", "model_mid", 0.60, 2, "subtask result returned to planner"),
        ("model_cheap", "delivery", 0.40, 1, "trivial subtask answers directly"),
        ("verifier", "delivery", 0.70, 1, "verified release"),
        ("verifier", "human_review", 0.10, 1, "escalate to a person"),
        ("verifier", "model_frontier", 0.20, 2, "rejected: replan at a stronger tier"),
        ("model_frontier", "verifier", 0.50, 2, "check the replan"),
        ("model_frontier", "delivery", 0.30, 1, "unchecked release"),
        ("model_frontier", "memory", 0.20, 1, "persist result"),
    ),
    "specialist_hierarchy": (
        ("router", "cache", 0.10, 1, "cache lookup"),
        ("router", "retriever", 0.30, 1, "grounded path"),
        ("router", "model_cheap", 0.60, 1, "first-line specialist"),
        ("cache", "delivery", 0.60, 1, "cache hit answers directly"),
        ("cache", "model_cheap", 0.40, 1, "cache miss"),
        ("retriever", "model_cheap", 0.40, 1, "cheap grounded generation"),
        ("retriever", "model_mid", 0.60, 1, "grounded generation"),
        ("model_cheap", "model_mid", 0.25, 1, "escalate one level"),
        ("model_cheap", "verifier", 0.35, 1, "check the small model"),
        ("model_cheap", "memory", 0.10, 1, "persist result"),
        ("model_cheap", "delivery", 0.30, 1, "unchecked release"),
        ("model_mid", "model_frontier", 0.20, 1, "escalate one level"),
        ("model_mid", "tool_search", 0.20, 1, "tool call"),
        ("model_mid", "verifier", 0.40, 1, "check the mid model"),
        ("model_mid", "memory", 0.20, 1, "persist result"),
        ("tool_search", "model_frontier", 0.40, 1, "hard synthesis after tool use"),
        ("tool_search", "verifier", 0.60, 1, "check the tool-grounded answer"),
        ("model_frontier", "verifier", 0.60, 1, "check the frontier answer"),
        ("model_frontier", "memory", 0.20, 1, "persist result"),
        ("model_frontier", "delivery", 0.20, 1, "unchecked release"),
        ("verifier", "delivery", 0.75, 1, "verified release"),
        ("verifier", "human_review", 0.25, 1, "escalate to a person"),
    ),
    "debate": (
        ("router", "cache", 0.10, 1, "cache lookup"),
        ("router", "retriever", 0.30, 1, "grounded path"),
        ("router", "model_mid", 0.60, 1, "open the debate"),
        ("cache", "delivery", 0.60, 1, "cache hit answers directly"),
        ("cache", "model_mid", 0.40, 1, "cache miss"),
        ("retriever", "model_mid", 0.60, 1, "grounded proponent"),
        ("retriever", "model_frontier", 0.40, 1, "grounded opponent"),
        ("model_mid", "model_frontier", 0.35, 2, "argue against the stronger peer"),
        ("model_mid", "model_cheap", 0.20, 2, "delegate a sub-argument"),
        ("model_mid", "verifier", 0.35, 2, "submit to the judge"),
        ("model_mid", "memory", 0.10, 1, "write the transcript"),
        ("model_frontier", "model_mid", 0.30, 2, "rebut the peer"),
        ("model_frontier", "verifier", 0.50, 2, "submit to the judge"),
        ("model_frontier", "memory", 0.20, 1, "write the transcript"),
        ("model_cheap", "model_mid", 0.40, 2, "return a sub-argument"),
        ("model_cheap", "tool_search", 0.20, 1, "look up a claim"),
        ("model_cheap", "delivery", 0.40, 1, "uncontested release"),
        ("tool_search", "verifier", 1.00, 1, "evidence goes to the judge"),
        ("verifier", "delivery", 0.80, 1, "judged release"),
        ("verifier", "human_review", 0.20, 1, "escalate to a person"),
    ),
    "swarm": (
        ("router", "cache", 0.10, 1, "cache lookup"),
        ("router", "retriever", 0.30, 1, "grounded path"),
        ("router", "model_cheap", 0.60, 1, "drop into the swarm"),
        ("cache", "delivery", 0.60, 1, "cache hit answers directly"),
        ("cache", "model_cheap", 0.40, 1, "cache miss"),
        ("retriever", "model_cheap", 0.50, 1, "cheap grounded generation"),
        ("retriever", "model_mid", 0.50, 1, "grounded generation"),
        ("model_cheap", "model_mid", 0.25, 2, "pass to a peer"),
        ("model_cheap", "tool_search", 0.20, 2, "tool call"),
        ("model_cheap", "verifier", 0.25, 1, "check the small model"),
        ("model_cheap", "memory", 0.10, 1, "write shared state"),
        ("model_cheap", "delivery", 0.20, 1, "unchecked release"),
        ("model_mid", "model_cheap", 0.25, 2, "hand back to a peer"),
        ("model_mid", "model_frontier", 0.20, 2, "pull in the strongest peer"),
        ("model_mid", "verifier", 0.35, 1, "check the mid model"),
        ("model_mid", "memory", 0.20, 1, "write shared state"),
        ("model_frontier", "model_mid", 0.25, 2, "hand back to a peer"),
        ("model_frontier", "verifier", 0.50, 1, "check the frontier answer"),
        ("model_frontier", "memory", 0.25, 1, "write shared state"),
        ("tool_search", "model_cheap", 0.50, 2, "tool result to a peer"),
        ("tool_search", "verifier", 0.50, 1, "tool result to the checker"),
        ("verifier", "delivery", 0.80, 1, "verified release"),
        ("verifier", "human_review", 0.20, 1, "escalate to a person"),
    ),
}

_ASSUMPTIONS = {
    "workflow_graph": "A fixed pipeline: retrieval, generation, optional tool call, verification.",
    "verifier_loop": "Rejected answers are regenerated at a stronger tier, at most three rounds.",
    "centralized_planner": "One planner holds the task and delegates; every result returns to it.",
    "specialist_hierarchy": "Work starts cheap and escalates one level at a time; no back edges.",
    "debate": "Two model tiers argue, a verifier judges, and a cheap peer supplies sub-arguments.",
    "swarm": "Peers hand work sideways with bounded traversal; shared memory is the coupling.",
}


def build_topology(
    name,
    queue_scale=1.0,
    tool_retry_attempts=3,
    tool_backoff_seconds=0.5,
    tool_jitter=0.6,
    tool_rate_limit_per_second=12.0,
):
    """Assemble one of the six named architectures over a shared station set.

    Only the wiring changes between architectures; stations are held fixed so that a policy
    comparison across architectures is not confounded by different service parameters.
    """
    if name not in _ARCHITECTURE_EDGES:
        raise Invalid("Unknown architecture: " + str(name))
    if not 0.0 <= queue_scale <= 8.0:
        raise Invalid("queue_scale must lie in [0, 8]")
    if not 1 <= int(tool_retry_attempts) <= 5:
        raise Invalid("tool_retry_attempts must lie in [1, 5]")
    tool_retry = {
        "max_attempts": int(tool_retry_attempts),
        "backoff_seconds": float(tool_backoff_seconds),
        "jitter": float(tool_jitter),
    }
    nodes = _standard_nodes(queue_scale, float(tool_rate_limit_per_second))
    kinds = {n.id: n.kind for n in nodes}
    timeouts = {n.id: n.timeout_seconds for n in nodes}
    edges = []
    for source, target, probability, traversals, role in (
        _CORE_EDGES + _ARCHITECTURE_EDGES[name]
    ):
        retry = dict(tool_retry if kinds[source] == "tool" else _DEFAULT_RETRY)
        if kinds[source] in ("queue", "memory", "cache"):
            retry = {"max_attempts": 1, "backoff_seconds": 0.0, "jitter": 0.0}
        fallback = _FALLBACK_BY_SOURCE.get(source)
        if fallback == source or fallback not in kinds:
            fallback = None
        edges.append(
            Edge.model_validate(
                {
                    "source": source,
                    "target": target,
                    "probability": probability,
                    "timeout_seconds": min(600.0, timeouts[target]),
                    "retry": retry,
                    "fallback": fallback,
                    "max_traversals": traversals,
                    "role": role,
                }
            )
        )
    topology = Topology.model_validate(
        {
            "id": "aisystem_" + name,
            "name": "AI system: " + name.replace("_", " "),
            "architecture": name,
            "entry": "intake",
            "terminals": ["delivery"],
            "nodes": [n.model_dump() for n in nodes],
            "edges": [e.model_dump() for e in edges],
            "assumptions": [
                _ASSUMPTIONS[name],
                "Service, cost, energy and capability parameters are declared, not measured.",
                "Model tiers are ordered cheap < mid < frontier in capability by construction.",
            ],
        }
    )
    validate(topology)
    return topology


def reference_topology(seed=None):
    """The vertical's default: a verifier loop, the architecture the owner's thesis centres on."""
    return build_topology("verifier_loop")
