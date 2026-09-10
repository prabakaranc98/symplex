"""Causal structure over an assumed DAG, with the boundary of what that buys stated plainly.

Every verdict in this module is a property of the graph it was given. The graph is an
assumption, usually an agent-authored one, and this module does not test it against data.
What it does provide is the set of conditional independencies the assumed graph implies,
which is the part of a causal claim that data can actually refute.

Definitions follow Pearl, *Causality: Models, Reasoning, and Inference*, 2nd edition,
2009: d-separation (definition 1.2.3), the back-door criterion (definition 3.3.1), the
front-door criterion (definition 3.3.4) and the structural definition of an instrument
(section 8.4). The d-separation routine is the reachability algorithm of Koller and
Friedman, *Probabilistic Graphical Models*, 2009, algorithm 3.1, which is equivalent to
Shachter's Bayes-Ball and handles the collider-descendant rule directly.

This is NOT the complete identification (ID) algorithm of Tian (2002) and Shpitser and
Pearl (2006). A verdict of `not_identifiable_by_backdoor_or_frontdoor` means those two
criteria failed on this graph, not that no identification strategy exists.
"""

from itertools import combinations
from typing import Literal

from pydantic import Field

from symplex.core.contracts import Invalid, digest
from symplex.modeling.complex_system import ClosedContract, Identifier, Text

MAX_NODES = 64
MAX_EDGES = 256
MAX_CONDITIONING_CANDIDATES = 14
MAX_ADJUSTMENT_SETS = 64
MAX_IMPLICATIONS = 512
MAX_FRONTDOOR_MEDIATOR_SETS = 128

SCOPE_GRAPH = (
    "Structure of an assumed DAG. The DAG itself is an unvalidated assumption; nothing "
    "in this module tests it against data or establishes that an arrow exists."
)
SCOPE_IDENTIFICATION = (
    "Identifiability is a property of the ASSUMED graph, not of the world and not of any "
    "dataset. The back-door and front-door criteria are sufficient, not necessary: this "
    "is not the complete ID algorithm, so a negative verdict rules out these two "
    "criteria only. A positive verdict still requires positivity, consistency, no "
    "measurement error in the adjustment set, and correct model specification, none of "
    "which are checked here."
)
SCOPE_IMPLICATIONS = (
    "Conditional independencies implied by the assumed DAG over its observed nodes. "
    "These are what data can refute; they are not evidence for the graph, since many "
    "graphs share an independence model. Statements touching latent nodes are dropped, "
    "and equality (Verma-type) constraints induced by latent structure are not "
    "enumerated. No statistical test is performed here."
)


class CausalNode(ClosedContract):
    id: Identifier
    latent: bool = Field(
        description="True when the variable is not measured; a latent node may appear in structure but never in an adjustment set or a testable implication."
    )
    meaning: Text


class CausalEdge(ClosedContract):
    source: Identifier
    target: Identifier
    sign: Literal["positive", "negative", "ambiguous", "unspecified"]
    mechanism: Text


def _cycle(nodes, parents):
    """Return one directed cycle as a node list, or None. Kahn's algorithm remainder."""
    indegree = {node: len(parents[node]) for node in nodes}
    children = {node: [] for node in nodes}
    for node in nodes:
        for parent in parents[node]:
            children[parent].append(node)
    ready = sorted(node for node in nodes if not indegree[node])
    order = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for child in children[node]:
            indegree[child] -= 1
            if not indegree[child]:
                ready.append(child)
                ready.sort()
    if len(order) == len(nodes):
        return None, order
    # Every remaining node keeps a parent among the remaining nodes, so walking parents
    # is guaranteed to close a cycle; walking children is not.
    remaining = {node for node in nodes if node not in set(order)}
    start = sorted(remaining)[0]
    path, seen = [start], {start}
    node = start
    for _ in range(len(nodes) + 1):
        node = sorted(p for p in parents[node] if p in remaining)[0]
        if node in seen:
            index = path.index(node)
            return [node] + list(reversed(path[index:])), None
        seen.add(node)
        path.append(node)
    raise Invalid("Cycle detection exceeded its node envelope")


class CausalGraph:
    """A signed, labelled DAG with explicit latent nodes.

    Latency is a modelling declaration: a latent node participates in structure and in
    d-separation, but is refused as a member of any adjustment set or instrument, and is
    dropped from the testable implications, because you cannot condition on what you did
    not measure.
    """

    def __init__(self, nodes, edges):
        parsed_nodes = [
            CausalNode.model_validate(
                n.model_dump() if isinstance(n, CausalNode) else n
            )
            for n in (nodes or [])
        ]
        if not 1 <= len(parsed_nodes) <= MAX_NODES:
            raise Invalid(f"A causal graph carries 1 to {MAX_NODES} nodes")
        if len({n.id for n in parsed_nodes}) != len(parsed_nodes):
            raise Invalid("Duplicate causal node ID")
        parsed_edges = [
            CausalEdge.model_validate(
                e.model_dump() if isinstance(e, CausalEdge) else e
            )
            for e in (edges or [])
        ]
        if len(parsed_edges) > MAX_EDGES:
            raise Invalid(f"A causal graph carries at most {MAX_EDGES} edges")
        self.nodes = {n.id: n for n in parsed_nodes}
        self.order = [n.id for n in parsed_nodes]
        self.parents = {n: set() for n in self.order}
        self.children = {n: set() for n in self.order}
        self.edges = []
        seen = set()
        for edge in parsed_edges:
            if edge.source not in self.nodes or edge.target not in self.nodes:
                raise Invalid("Causal edge references an undeclared node")
            if edge.source == edge.target:
                raise Invalid("A causal edge cannot be a self-loop")
            if (edge.source, edge.target) in seen:
                raise Invalid("Duplicate causal edge")
            if (edge.target, edge.source) in seen:
                raise Invalid("A pair of nodes carries at most one directed edge")
            seen.add((edge.source, edge.target))
            self.parents[edge.target].add(edge.source)
            self.children[edge.source].add(edge.target)
            self.edges.append(edge)
        cycle, topological = _cycle(self.order, self.parents)
        if cycle is not None:
            raise Invalid(
                "A causal graph must be acyclic; found "
                + " -> ".join(cycle)
                + ". A feedback loop needs an explicit time index before d-separation applies."
            )
        self.topological = topological

    @classmethod
    def from_pairs(cls, pairs, latent=(), meanings=None) -> "CausalGraph":
        """Terse constructor for a graph whose signs and mechanisms are not yet declared."""
        latent = set(latent)
        meanings = meanings or {}
        names = []
        for source, target in pairs:
            for name in (source, target):
                if name not in names:
                    names.append(name)
        for name in sorted(latent):
            if name not in names:
                names.append(name)
        nodes = [
            {
                "id": name,
                "latent": name in latent,
                "meaning": meanings.get(name, "Declared without a stated meaning."),
            }
            for name in names
        ]
        edges = [
            {
                "source": source,
                "target": target,
                "sign": "unspecified",
                "mechanism": "Declared without a stated mechanism.",
            }
            for source, target in pairs
        ]
        return cls(nodes, edges)

    @property
    def latent(self) -> set:
        return {n for n, node in self.nodes.items() if node.latent}

    @property
    def observed(self) -> set:
        return {n for n, node in self.nodes.items() if not node.latent}

    def require(self, *names):
        for name in names:
            if name not in self.nodes:
                raise Invalid("Unknown causal node: " + str(name))
        return names

    def digest(self) -> str:
        return digest(
            {
                "nodes": [n.model_dump() for n in self.nodes.values()],
                "edges": [e.model_dump() for e in self.edges],
            }
        )

    def without_outgoing(self, node) -> "CausalGraph":
        """Pearl's G-underline-X: the graph with every edge emanating from X deleted."""
        self.require(node)
        return CausalGraph(
            [n.model_dump() for n in self.nodes.values()],
            [e.model_dump() for e in self.edges if e.source != node],
        )

    def without_incoming(self, node) -> "CausalGraph":
        """Pearl's G-overline-X: the graph with every edge entering X deleted."""
        self.require(node)
        return CausalGraph(
            [n.model_dump() for n in self.nodes.values()],
            [e.model_dump() for e in self.edges if e.target != node],
        )

    def without_node(self, node) -> "CausalGraph":
        self.require(node)
        return CausalGraph(
            [n.model_dump() for n in self.nodes.values() if n.id != node],
            [
                e.model_dump()
                for e in self.edges
                if e.source != node and e.target != node
            ],
        )

    def _closure(self, start, step):
        frontier, seen = list(start), set(start)
        while frontier:
            node = frontier.pop()
            for neighbour in step[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    frontier.append(neighbour)
        return seen - set(start)

    def ancestor_set(self, nodes) -> set:
        return self._closure(list(nodes), self.parents)

    def descendant_set(self, nodes) -> set:
        return self._closure(list(nodes), self.children)


def _graph(value) -> CausalGraph:
    if isinstance(value, CausalGraph):
        return value
    if isinstance(value, dict) and "nodes" in value and "edges" in value:
        return CausalGraph(value["nodes"], value["edges"])
    raise Invalid(
        "Expected a CausalGraph or a {'nodes': [...], 'edges': [...]} mapping"
    )


def acyclic_check(graph) -> dict:
    """Report whether a declared node/edge set is a DAG, naming a cycle when it is not.

    Accepts a `CausalGraph` (always acyclic, since the constructor refuses cycles) or a
    raw `{"nodes": [...], "edges": [...]}` mapping, so a declaration can be screened
    before a graph is built from it.
    """
    if isinstance(graph, CausalGraph):
        return {
            "acyclic": True,
            "cycle": None,
            "topological_order": list(graph.topological),
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "scope": SCOPE_GRAPH,
        }
    if not isinstance(graph, dict) or "nodes" not in graph or "edges" not in graph:
        raise Invalid(
            "Expected a CausalGraph or a {'nodes': [...], 'edges': [...]} mapping"
        )
    names = []
    for node in graph["nodes"]:
        name = node["id"] if isinstance(node, dict) else getattr(node, "id", node)
        names.append(name)
    if len(set(names)) != len(names):
        raise Invalid("Duplicate causal node ID")
    if not 1 <= len(names) <= MAX_NODES:
        raise Invalid(f"A causal graph carries 1 to {MAX_NODES} nodes")
    parents = {name: set() for name in names}
    for edge in graph["edges"]:
        pair = (
            (edge["source"], edge["target"])
            if isinstance(edge, dict)
            else (edge.source, edge.target)
        )
        if pair[0] not in parents or pair[1] not in parents:
            raise Invalid("Causal edge references an undeclared node")
        parents[pair[1]].add(pair[0])
    cycle, topological = _cycle(names, parents)
    return {
        "acyclic": cycle is None,
        "cycle": cycle,
        "topological_order": topological,
        "node_count": len(names),
        "edge_count": len(graph["edges"]),
        "scope": SCOPE_GRAPH,
    }


def ancestors(graph, node) -> dict:
    """Proper ancestors of a node in the assumed graph."""
    graph = _graph(graph)
    graph.require(node)
    found = graph.ancestor_set([node])
    return {
        "node": node,
        "ancestors": sorted(found),
        "count": len(found),
        "latent_ancestors": sorted(found & graph.latent),
        "scope": SCOPE_GRAPH,
    }


def descendants(graph, node) -> dict:
    """Proper descendants of a node in the assumed graph."""
    graph = _graph(graph)
    graph.require(node)
    found = graph.descendant_set([node])
    return {
        "node": node,
        "descendants": sorted(found),
        "count": len(found),
        "latent_descendants": sorted(found & graph.latent),
        "scope": SCOPE_GRAPH,
    }


def _reachable(graph: CausalGraph, sources, given) -> set:
    """Koller and Friedman algorithm 3.1: nodes reachable from `sources` by an active trail."""
    given = set(given)
    ancestors_of_given = given | graph.ancestor_set(given)
    frontier = [(node, "up") for node in sources]
    visited, reached = set(), set()
    while frontier:
        node, direction = frontier.pop()
        if (node, direction) in visited:
            continue
        visited.add((node, direction))
        if node not in given:
            reached.add(node)
        if direction == "up" and node not in given:
            for parent in graph.parents[node]:
                frontier.append((parent, "up"))
            for child in graph.children[node]:
                frontier.append((child, "down"))
        elif direction == "down":
            if node not in given:
                for child in graph.children[node]:
                    frontier.append((child, "down"))
            if node in ancestors_of_given:
                for parent in graph.parents[node]:
                    frontier.append((parent, "up"))
    return reached


def _separated(graph: CausalGraph, x, y, given) -> bool:
    xs = {x} if isinstance(x, str) else set(x)
    ys = {y} if isinstance(y, str) else set(y)
    return not (_reachable(graph, xs, given) & ys)


def d_separation(graph, x, y, given=()) -> dict:
    """Decide whether x and y are d-separated by `given` in the assumed DAG.

    Pearl 2009, definition 1.2.3. A path is blocked by Z when it contains a chain or a
    fork whose middle node is in Z, or a collider whose middle node and all of whose
    descendants are outside Z. Conditioning on a collider or on any descendant of a
    collider therefore OPENS a path that was closed; this implementation gets that from
    the `ancestors_of_given` test in the reachability pass rather than from a special case.
    """
    graph = _graph(graph)
    if isinstance(given, str):
        raise Invalid("Pass the conditioning set as a collection, not a bare string")
    given = list(given)
    if len(given) > MAX_NODES:
        raise Invalid("Conditioning set exceeds the node envelope")
    graph.require(x, y, *given)
    if x == y:
        raise Invalid("d-separation is asked of two distinct nodes")
    if x in given or y in given:
        raise Invalid(
            "Refusing a d-separation query whose conditioning set contains the queried "
            "node; the statement is not well posed"
        )
    reached = _reachable(graph, {x}, given)
    separated = y not in reached
    conditioned_latent = sorted(set(given) & graph.latent)
    return {
        "x": x,
        "y": y,
        "given": sorted(given),
        "d_separated": separated,
        "verdict": "d_separated" if separated else "d_connected",
        "implies_independence": separated,
        "reachable_from_x": sorted(reached),
        "colliders_opened": sorted(
            node
            for node in graph.nodes
            if len(graph.parents[node]) >= 2
            and ({node} | graph.descendant_set([node])) & set(given)
        ),
        "conditioned_on_latent": conditioned_latent,
        "conditioning_warning": (
            "The conditioning set contains latent nodes ("
            + ", ".join(conditioned_latent)
            + "); no dataset can realise this conditioning."
        )
        if conditioned_latent
        else None,
        "graph_digest": graph.digest(),
        "scope": SCOPE_GRAPH,
    }


def _backdoor_valid(graph: CausalGraph, treatment, outcome, adjustment) -> bool:
    """Pearl 2009 definition 3.3.1, applied via the G-underline-X separation test."""
    forbidden = graph.descendant_set([treatment]) | {treatment, outcome}
    if set(adjustment) & forbidden:
        return False
    return _separated(graph.without_outgoing(treatment), treatment, outcome, adjustment)


def backdoor_sets(graph, treatment, outcome) -> dict:
    """Enumerate minimal observed adjustment sets satisfying the back-door criterion.

    A set Z is admissible when no member is a descendant of the treatment and Z blocks
    every path from treatment to outcome that starts with an arrow into the treatment.
    Only observed nodes are candidates: an adjustment set you cannot measure is not an
    adjustment set. Minimal means no proper subset is itself admissible.
    """
    graph = _graph(graph)
    graph.require(treatment, outcome)
    if treatment == outcome:
        raise Invalid("Treatment and outcome must be distinct")
    forbidden = graph.descendant_set([treatment]) | {treatment, outcome}
    candidates = sorted(graph.observed - forbidden)
    truncated = False
    if len(candidates) > MAX_CONDITIONING_CANDIDATES:
        candidates = candidates[:MAX_CONDITIONING_CANDIDATES]
        truncated = True
    minimal, considered = [], 0
    for size in range(len(candidates) + 1):
        for subset in combinations(candidates, size):
            considered += 1
            if any(set(known) <= set(subset) for known in minimal):
                continue
            if _backdoor_valid(graph, treatment, outcome, subset):
                minimal.append(list(subset))
                if len(minimal) >= MAX_ADJUSTMENT_SETS:
                    truncated = True
                    break
        if len(minimal) >= MAX_ADJUSTMENT_SETS:
            break
    latent_backdoor = sorted(
        node
        for node in graph.latent
        if node in graph.ancestor_set([treatment]) | {treatment}
        and node in graph.ancestor_set([outcome])
    )
    return {
        "treatment": treatment,
        "outcome": outcome,
        "adjustment_sets": minimal,
        "set_count": len(minimal),
        "empty_set_sufficient": [] in minimal,
        "candidate_nodes": candidates,
        "excluded_as_descendants": sorted(graph.descendant_set([treatment])),
        "excluded_as_latent": sorted(graph.latent - {treatment, outcome}),
        "latent_confounders_of_treatment_and_outcome": latent_backdoor,
        "subsets_considered": considered,
        "truncated": truncated,
        "satisfied": bool(minimal),
        "graph_digest": graph.digest(),
        "scope": SCOPE_IDENTIFICATION,
    }


def frontdoor_check(graph, treatment, outcome, mediators) -> dict:
    """Test the three front-door conditions for a proposed mediator set (Pearl 2009, 3.3.4).

    (1) the mediators intercept every directed path from treatment to outcome;
    (2) no unblocked back-door path runs from treatment to the mediators;
    (3) every back-door path from mediators to outcome is blocked by the treatment.
    All mediators must be observed, or the estimator cannot be computed.
    """
    graph = _graph(graph)
    mediators = list(mediators)
    if not mediators:
        raise Invalid("The front-door criterion needs at least one mediator")
    graph.require(treatment, outcome, *mediators)
    if treatment in mediators or outcome in mediators:
        raise Invalid("Mediators must differ from the treatment and the outcome")
    unmeasured = sorted(set(mediators) & graph.latent)
    reduced = graph
    for mediator in mediators:
        reduced = reduced.without_node(mediator)
    intercepts = (
        treatment not in reduced.nodes
        or outcome not in reduced.nodes
        or (outcome not in reduced.descendant_set([treatment]))
    )
    no_backdoor_to_mediators = all(
        _separated(graph.without_outgoing(treatment), treatment, mediator, ())
        for mediator in mediators
    )
    mediators_blocked_by_treatment = all(
        _separated(graph.without_outgoing(mediator), mediator, outcome, [treatment])
        for mediator in mediators
    )
    conditions = [
        {
            "code": "mediators_intercept_all_directed_paths",
            "holds": intercepts,
            "detail": "Removing the mediators leaves no directed path from treatment to outcome.",
        },
        {
            "code": "no_unblocked_backdoor_treatment_to_mediators",
            "holds": no_backdoor_to_mediators,
            "detail": "No confounding path runs from treatment into a mediator.",
        },
        {
            "code": "treatment_blocks_backdoor_mediators_to_outcome",
            "holds": mediators_blocked_by_treatment,
            "detail": "Conditioning on the treatment blocks every back-door path from a mediator to the outcome.",
        },
        {
            "code": "mediators_observed",
            "holds": not unmeasured,
            "detail": "Every mediator is measured, so the front-door estimator is computable.",
        },
    ]
    satisfied = all(condition["holds"] for condition in conditions)
    return {
        "treatment": treatment,
        "outcome": outcome,
        "mediators": sorted(mediators),
        "satisfied": satisfied,
        "conditions": conditions,
        "failing_conditions": [c["code"] for c in conditions if not c["holds"]],
        "unmeasured_mediators": unmeasured,
        "graph_digest": graph.digest(),
        "scope": SCOPE_IDENTIFICATION,
    }


def instrument_check(graph, z, x, y, given=()) -> dict:
    """Test the three instrumental-variable conditions structurally (Pearl 2009, 8.4).

    Relevance: Z is d-connected to X, so it moves the treatment. Exclusion: no directed
    path from Z to Y avoids X, so Z affects the outcome only through the treatment.
    Independence: Z is d-separated from Y in the graph with X's outgoing edges deleted,
    so Z shares no confounder with the outcome. A structurally valid instrument still
    yields only a local or parametric effect, never a nonparametric point identification.
    """
    graph = _graph(graph)
    given = list(given)
    graph.require(z, x, y, *given)
    if len({z, x, y}) != 3:
        raise Invalid("An instrument, a treatment and an outcome must be distinct")
    if set(given) & {z, x, y}:
        raise Invalid(
            "The conditioning set must exclude the instrument, treatment and outcome"
        )
    relevance = not _separated(graph, z, x, given)
    without_x = graph.without_node(x)
    exclusion = (
        z not in without_x.nodes
        or y not in without_x.nodes
        or y not in without_x.descendant_set([z])
    )
    independence = _separated(graph.without_outgoing(x), z, y, given)
    observed = z not in graph.latent and not (set(given) & graph.latent)
    conditions = [
        {
            "code": "relevance",
            "holds": relevance,
            "detail": "Z is d-connected to X given the conditioning set, so it has structural relevance.",
        },
        {
            "code": "exclusion_restriction",
            "holds": exclusion,
            "detail": "Deleting X leaves no directed path from Z to Y.",
        },
        {
            "code": "independence_no_shared_confounder",
            "holds": independence,
            "detail": "Z is d-separated from Y once X's outgoing edges are removed.",
        },
        {
            "code": "instrument_observed",
            "holds": observed,
            "detail": "The instrument and any conditioning variables are measured.",
        },
    ]
    valid = all(condition["holds"] for condition in conditions)
    return {
        "instrument": z,
        "treatment": x,
        "outcome": y,
        "given": sorted(given),
        "valid_instrument": valid,
        "conditions": conditions,
        "failing_conditions": [c["code"] for c in conditions if not c["holds"]],
        "estimand_if_valid": "local or parametric average effect under an additional homogeneity or monotonicity assumption",
        "point_identified": False,
        "graph_digest": graph.digest(),
        "scope": SCOPE_IDENTIFICATION,
    }


def identifiable(graph, treatment, outcome) -> dict:
    """Verdict on the total effect of treatment on outcome under the ASSUMED graph.

    Returns one of `identifiable`, `requires_stronger_assumptions` or
    `not_identifiable_by_backdoor_or_frontdoor`. The middle verdict is used when neither
    criterion applies but a structurally valid instrument exists, since an instrument
    buys identification only under an added homogeneity, monotonicity or parametric
    assumption. This is deliberately conservative and is NOT the complete ID algorithm.
    """
    graph = _graph(graph)
    graph.require(treatment, outcome)
    if treatment == outcome:
        raise Invalid("Treatment and outcome must be distinct")
    backdoor = backdoor_sets(graph, treatment, outcome)
    has_effect_path = outcome in graph.descendant_set([treatment])
    frontdoor_results, frontdoor = [], None
    if not backdoor["satisfied"]:
        candidates = sorted(
            (graph.observed & graph.descendant_set([treatment]))
            & (graph.ancestor_set([outcome]) | set())
        )
        candidates = [c for c in candidates if c != outcome][
            :MAX_CONDITIONING_CANDIDATES
        ]
        attempted = 0
        for size in (1, 2):
            for subset in combinations(candidates, size):
                attempted += 1
                if attempted > MAX_FRONTDOOR_MEDIATOR_SETS:
                    break
                result = frontdoor_check(graph, treatment, outcome, list(subset))
                frontdoor_results.append(
                    {"mediators": result["mediators"], "satisfied": result["satisfied"]}
                )
                if result["satisfied"]:
                    frontdoor = result
                    break
            if frontdoor is not None:
                break
    instruments = []
    if not backdoor["satisfied"] and frontdoor is None:
        for candidate in sorted(graph.observed - {treatment, outcome}):
            check = instrument_check(graph, candidate, treatment, outcome)
            if check["valid_instrument"]:
                instruments.append(candidate)
    if backdoor["satisfied"]:
        verdict, method = "identifiable", "backdoor_adjustment"
    elif frontdoor is not None:
        verdict, method = "identifiable", "frontdoor_adjustment"
    elif instruments:
        verdict, method = "requires_stronger_assumptions", "instrumental_variable"
    else:
        verdict, method = "not_identifiable_by_backdoor_or_frontdoor", None
    return {
        "treatment": treatment,
        "outcome": outcome,
        "verdict": verdict,
        "identifiable": verdict == "identifiable",
        "method": method,
        "adjustment_sets": backdoor["adjustment_sets"],
        "frontdoor": {
            "mediators": frontdoor["mediators"] if frontdoor else None,
            "attempted": frontdoor_results,
        },
        "candidate_instruments": instruments,
        "latent_confounders_of_treatment_and_outcome": backdoor[
            "latent_confounders_of_treatment_and_outcome"
        ],
        "directed_path_exists": has_effect_path,
        "complete_id_algorithm_applied": False,
        "graph_digest": graph.digest(),
        "scope": SCOPE_IDENTIFICATION,
    }


def testable_implications(graph) -> dict:
    """Enumerate the conditional independencies the assumed DAG implies over observed nodes.

    Uses the ordered local Markov basis: every node is independent of its non-descendants
    given its parents (Pearl 2009, theorem 1.2.7). Each statement is emitted pairwise and
    re-verified with `d_separation`, so the returned list is a basis from which the
    remaining implications follow by the semi-graphoid axioms. This is the falsifiable
    content of the graph: a statement here that data rejects rejects the graph.
    """
    graph = _graph(graph)
    seen, statements, suppressed = set(), [], 0
    for node in graph.topological:
        parents = graph.parents[node]
        nondescendants = (
            set(graph.nodes) - graph.descendant_set([node]) - {node} - parents
        )
        for other in sorted(nondescendants):
            key = (frozenset({node, other}), frozenset(parents))
            if key in seen:
                continue
            seen.add(key)
            if ({node, other} | parents) & graph.latent:
                suppressed += 1
                continue
            if len(statements) >= MAX_IMPLICATIONS:
                break
            pair = sorted((node, other))
            verified = d_separation(graph, pair[0], pair[1], sorted(parents))
            statements.append(
                {
                    "x": pair[0],
                    "y": pair[1],
                    "given": sorted(parents),
                    "statement": (
                        f"{pair[0]} _||_ {pair[1]}"
                        + (" | " + ", ".join(sorted(parents)) if parents else "")
                    ),
                    "verified_by_d_separation": verified["d_separated"],
                }
            )
    statements.sort(key=lambda s: (s["x"], s["y"], tuple(s["given"])))
    unverified = [s for s in statements if not s["verified_by_d_separation"]]
    if unverified:
        raise Invalid(
            "Internal inconsistency: a local Markov statement failed its own d-separation check"
        )
    return {
        "implications": statements,
        "implication_count": len(statements),
        "suppressed_by_latent_nodes": suppressed,
        "latent_nodes": sorted(graph.latent),
        "basis": "ordered local Markov (each node against its non-descendants given its parents)",
        "truncated": len(statements) >= MAX_IMPLICATIONS,
        "tested_against_data": False,
        "graph_digest": graph.digest(),
        "scope": SCOPE_IMPLICATIONS,
    }
