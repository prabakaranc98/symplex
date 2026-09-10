"""Deterministic network generators and numpy-only structural measures.

Generators return a `Network` that carries its own `scope` field; every other
public function returns a dict containing a `scope` string. A generated graph is
a stated construction, not a measured social, biological or engineered network.
Reproducing one summary statistic of a real system does not establish that the
real system was produced by the matching generator, and a structural measure
computed here describes the supplied adjacency matrix only.

No network access, no model calls, no subprocess. numpy and scipy.sparse.csgraph
(connected components only) are the sole numerical dependencies.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from symplex.core.contracts import Invalid

MAX_NODES = 2000
MAX_EDGES = 60000
MAX_COMMUNITIES = 24
MAX_PATH_NODES = 800
MAX_BETWEENNESS_NODES = 800

GENERATOR_SCOPE = (
    "synthetic construction under a stated generator and seed; not a measured "
    "network and not evidence that any real system shares this topology"
)
MEASURE_SCOPE = (
    "descriptive structure of the supplied adjacency matrix; no causal, "
    "predictive or empirical claim about a real system is established"
)


def _rng(seed):
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise Invalid("Seed must be an integer")
    if not 0 <= int(seed) < 2**32:
        raise Invalid("Seed must lie in [0, 2**32)")
    return np.random.default_rng(int(seed))


def _count(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise Invalid(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise Invalid(f"{name} must lie in [{low}, {high}]")
    return value


def _fraction(value, name, low=0.0, high=1.0):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise Invalid(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise Invalid(f"{name} must be a finite number in [{low}, {high}]")
    return value


@dataclasses.dataclass(eq=False)
class Network:
    """An undirected simple graph plus the optional labels a diffusion study needs.

    `communities` and `attributes` are declared node properties. When a generator
    supplies them they are the construction's own ground truth, which exists only
    inside the simulation and is never available for a real observed cascade.
    """

    adjacency: np.ndarray
    name: str = "network"
    generator: str = "supplied"
    parameters: dict = dataclasses.field(default_factory=dict)
    communities: np.ndarray | None = None
    attributes: np.ndarray | None = None
    seed: int | None = None
    scope: str = GENERATOR_SCOPE

    def __post_init__(self):
        matrix = np.asarray(self.adjacency)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise Invalid("Adjacency must be a square matrix")
        if not 2 <= matrix.shape[0] <= MAX_NODES:
            raise Invalid(f"Node count must lie in [2, {MAX_NODES}]")
        matrix = matrix.astype(np.float64, copy=True)
        if not np.all(np.isfinite(matrix)):
            raise Invalid("Adjacency contains nonfinite entries")
        if not np.array_equal(matrix, matrix.T):
            raise Invalid("Adjacency must be symmetric; directed graphs are unsupported")
        if np.any(np.diag(matrix) != 0):
            raise Invalid("Adjacency must have a zero diagonal; self-loops are unsupported")
        if not np.all((matrix == 0) | (matrix == 1)):
            raise Invalid("Adjacency must be binary; edge weights belong to a mechanism")
        self.adjacency = matrix
        self.n = int(matrix.shape[0])
        upper = np.triu(matrix, 1)
        rows, cols = np.nonzero(upper)
        self.edges = np.column_stack([rows, cols]).astype(np.int64)
        self.edge_count = int(self.edges.shape[0])
        if self.edge_count > MAX_EDGES:
            raise Invalid(f"Edge count must not exceed {MAX_EDGES}")
        self.degree = matrix.sum(axis=1).astype(np.int64)
        if self.communities is not None:
            labels = np.asarray(self.communities)
            if labels.shape != (self.n,) or not np.issubdtype(labels.dtype, np.integer):
                raise Invalid("Community labels must be one integer per node")
            if labels.min() < 0 or len(np.unique(labels)) > MAX_COMMUNITIES:
                raise Invalid("Community labels must be small nonnegative integers")
            self.communities = labels.astype(np.int64)
        if self.attributes is not None:
            values = np.asarray(self.attributes, dtype=np.float64)
            if values.shape != (self.n,) or not np.all(np.isfinite(values)):
                raise Invalid("Node attributes must be one finite number per node")
            self.attributes = values

    @property
    def mean_degree(self):
        return float(self.degree.mean())

    def neighbours(self, node):
        return np.nonzero(self.adjacency[node])[0]

    def to_dict(self):
        return {
            "name": self.name,
            "generator": self.generator,
            "parameters": dict(self.parameters),
            "seed": self.seed,
            "nodes": self.n,
            "edges": self.edge_count,
            "mean_degree": self.mean_degree,
            "max_degree": int(self.degree.max()),
            "has_communities": self.communities is not None,
            "has_attributes": self.attributes is not None,
            "scope": self.scope,
        }


def from_adjacency(matrix, name="supplied", communities=None, attributes=None, parameters=None):
    """Wrap a caller-supplied binary symmetric adjacency matrix."""
    return Network(
        adjacency=matrix,
        name=name,
        generator="supplied",
        parameters=dict(parameters or {}),
        communities=communities,
        attributes=attributes,
        seed=None,
        scope="caller-supplied adjacency; provenance, sampling frame and edge "
        "semantics of this graph are not established here",
    )


def _from_edges(n, pairs, **kwargs):
    matrix = np.zeros((n, n), dtype=np.float64)
    if len(pairs):
        pairs = np.asarray(pairs, dtype=np.int64)
        keep = pairs[:, 0] != pairs[:, 1]
        pairs = pairs[keep]
        matrix[pairs[:, 0], pairs[:, 1]] = 1.0
        matrix[pairs[:, 1], pairs[:, 0]] = 1.0
    return Network(adjacency=matrix, **kwargs)


# --------------------------------------------------------------------------- #
# Generators
# --------------------------------------------------------------------------- #


def erdos_renyi(n, p, seed, name="erdos_renyi"):
    """Independent edges at probability p. No clustering, no hubs, no communities."""
    n = _count(n, "n", 2, MAX_NODES)
    p = _fraction(p, "p")
    rng = _rng(seed)
    draws = rng.random((n, n))
    upper = np.triu(draws < p, 1)
    matrix = (upper | upper.T).astype(np.float64)
    return Network(
        adjacency=matrix,
        name=name,
        generator="erdos_renyi",
        parameters={"n": n, "p": p},
        seed=int(seed),
    )


def watts_strogatz(n, k, beta, seed, name="watts_strogatz"):
    """Ring lattice with k neighbours per node, each edge rewired with probability beta."""
    n = _count(n, "n", 4, MAX_NODES)
    k = _count(k, "k", 2, min(n - 1, 64))
    if k % 2:
        raise Invalid("k must be even so the ring lattice is symmetric")
    if k >= n:
        raise Invalid("k must be smaller than n")
    beta = _fraction(beta, "beta")
    rng = _rng(seed)
    matrix = np.zeros((n, n), dtype=bool)
    for offset in range(1, k // 2 + 1):
        targets = (np.arange(n) + offset) % n
        matrix[np.arange(n), targets] = True
        matrix[targets, np.arange(n)] = True
    for offset in range(1, k // 2 + 1):
        for node in range(n):
            partner = (node + offset) % n
            if rng.random() >= beta:
                continue
            candidates = np.nonzero(~matrix[node])[0]
            candidates = candidates[candidates != node]
            if not candidates.size:
                continue
            new = int(candidates[rng.integers(candidates.size)])
            matrix[node, partner] = matrix[partner, node] = False
            matrix[node, new] = matrix[new, node] = True
    return Network(
        adjacency=matrix.astype(np.float64),
        name=name,
        generator="watts_strogatz",
        parameters={"n": n, "k": k, "beta": beta},
        seed=int(seed),
    )


def barabasi_albert(n, m, seed, name="barabasi_albert"):
    """Preferential attachment: each new node attaches m edges proportional to degree."""
    n = _count(n, "n", 4, MAX_NODES)
    m = _count(m, "m", 1, min(n - 1, 32))
    rng = _rng(seed)
    matrix = np.zeros((n, n), dtype=bool)
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            matrix[i, j] = matrix[j, i] = True
    repeated = [node for node in range(m + 1) for _ in range(m)]
    for node in range(m + 1, n):
        chosen = set()
        guard = 0
        while len(chosen) < m and guard < 200 * m:
            guard += 1
            pick = int(repeated[rng.integers(len(repeated))])
            if pick != node:
                chosen.add(pick)
        if len(chosen) < m:  # Degenerate draw: fall back to the lowest free indices.
            for candidate in range(node):
                if len(chosen) >= m:
                    break
                chosen.add(candidate)
        for pick in sorted(chosen):
            matrix[node, pick] = matrix[pick, node] = True
            repeated.append(pick)
            repeated.append(node)
    return Network(
        adjacency=matrix.astype(np.float64),
        name=name,
        generator="barabasi_albert",
        parameters={"n": n, "m": m},
        seed=int(seed),
    )


def stochastic_block_model(
    sizes, p_in, p_out, seed, name="stochastic_block_model", attribute_assortativity=1.0
):
    """Planted communities: within-block probability p_in, between-block p_out.

    `p_in` may be a scalar or one probability per block. Node attributes are drawn
    around each block's centre so that homophily and contagion are genuinely
    confounded, which is the point of this generator in this vertical.
    """
    if not isinstance(sizes, (list, tuple)) or not 2 <= len(sizes) <= MAX_COMMUNITIES:
        raise Invalid(f"sizes must list between 2 and {MAX_COMMUNITIES} block sizes")
    sizes = [_count(s, "block size", 2, MAX_NODES) for s in sizes]
    n = sum(sizes)
    if n > MAX_NODES:
        raise Invalid(f"Total node count must not exceed {MAX_NODES}")
    if isinstance(p_in, (list, tuple)):
        if len(p_in) != len(sizes):
            raise Invalid("p_in must be a scalar or one probability per block")
        within = [_fraction(p, "p_in") for p in p_in]
    else:
        within = [_fraction(p_in, "p_in")] * len(sizes)
    p_out = _fraction(p_out, "p_out")
    assortativity = _fraction(attribute_assortativity, "attribute_assortativity")
    rng = _rng(seed)
    labels = np.concatenate([np.full(size, index) for index, size in enumerate(sizes)])
    probability = np.full((n, n), p_out)
    start = 0
    for index, size in enumerate(sizes):
        probability[start : start + size, start : start + size] = within[index]
        start += size
    draws = rng.random((n, n))
    upper = np.triu(draws < probability, 1)
    matrix = (upper | upper.T).astype(np.float64)
    centres = np.linspace(0.15, 0.85, len(sizes))
    noise = rng.random(n)
    attributes = assortativity * centres[labels] + (1 - assortativity) * noise
    attributes = np.clip(attributes + 0.08 * (noise - 0.5), 0.0, 1.0)
    return Network(
        adjacency=matrix,
        name=name,
        generator="stochastic_block_model",
        parameters={
            "sizes": list(sizes),
            "p_in": within,
            "p_out": p_out,
            "attribute_assortativity": assortativity,
        },
        communities=labels.astype(np.int64),
        attributes=attributes,
        seed=int(seed),
    )


def configuration_model(degree_sequence, seed, name="configuration_model"):
    """Stub matching against a prescribed degree sequence.

    Self-loops and repeated pairs are discarded rather than retried, so the
    realized degree sequence is reported and is usually slightly below target.
    """
    sequence = np.asarray(degree_sequence)
    if sequence.ndim != 1 or not 2 <= sequence.size <= MAX_NODES:
        raise Invalid(f"degree_sequence must hold between 2 and {MAX_NODES} degrees")
    if not np.issubdtype(sequence.dtype, np.integer):
        raise Invalid("degree_sequence must contain integers")
    if sequence.min() < 0 or sequence.max() >= sequence.size:
        raise Invalid("Degrees must be nonnegative and smaller than the node count")
    if int(sequence.sum()) % 2:
        raise Invalid("The degree sum must be even for stub matching")
    if int(sequence.sum()) // 2 > MAX_EDGES:
        raise Invalid(f"Requested edge count exceeds {MAX_EDGES}")
    rng = _rng(seed)
    stubs = np.repeat(np.arange(sequence.size), sequence)
    stubs = rng.permutation(stubs)
    pairs = stubs.reshape(-1, 2)
    network = _from_edges(
        int(sequence.size),
        pairs,
        name=name,
        generator="configuration_model",
        parameters={"target_degree_sum": int(sequence.sum())},
        seed=int(seed),
    )
    network.parameters["realized_degree_sum"] = int(network.degree.sum())
    network.parameters["discarded_stub_pairs"] = int(
        (sequence.sum() - network.degree.sum()) // 2
    )
    return network


def geometric_graph(n, radius, seed, name="geometric_graph"):
    """Random points in the unit square joined below a distance threshold.

    Node attributes are the x coordinate, so spatial proximity, attribute
    similarity and adjacency are correlated by construction.
    """
    n = _count(n, "n", 2, MAX_NODES)
    radius = _fraction(radius, "radius", 0.0, 1.5)
    rng = _rng(seed)
    points = rng.random((n, 2))
    delta = points[:, None, :] - points[None, :, :]
    distance = np.sqrt((delta**2).sum(-1))
    matrix = (distance < radius).astype(np.float64)
    np.fill_diagonal(matrix, 0.0)
    return Network(
        adjacency=matrix,
        name=name,
        generator="geometric_graph",
        parameters={"n": n, "radius": radius},
        attributes=points[:, 0],
        seed=int(seed),
    )


GENERATORS = {
    "erdos_renyi": erdos_renyi,
    "watts_strogatz": watts_strogatz,
    "barabasi_albert": barabasi_albert,
    "stochastic_block_model": stochastic_block_model,
    "configuration_model": configuration_model,
    "geometric_graph": geometric_graph,
}


def generate(kind, seed, **parameters):
    """Dispatch to a named generator. Unknown names fail closed."""
    if kind not in GENERATORS:
        raise Invalid(
            "unsupported_operation: unknown generator " + str(kind) + "; available: "
            + ", ".join(sorted(GENERATORS))
        )
    return GENERATORS[kind](seed=seed, **parameters)


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #


def _network(value):
    if not isinstance(value, Network):
        raise Invalid("Expected a Network built by this module")
    return value


def degree_distribution(network, tail_fraction=0.2):
    """Degree summary plus a Hill tail-exponent estimate.

    The Hill estimator assumes a Pareto upper tail. A finite estimate is not
    evidence that the degree distribution is a power law; it is one number
    computed from the largest observed degrees.
    """
    network = _network(network)
    tail_fraction = _fraction(tail_fraction, "tail_fraction", 0.02, 0.9)
    degree = network.degree
    counts = np.bincount(degree, minlength=int(degree.max()) + 1)
    ordered = np.sort(degree)[::-1]
    take = max(3, int(round(tail_fraction * degree.size)))
    take = min(take, int((degree > 0).sum()))
    exponent = None
    if take >= 3:
        tail = ordered[:take].astype(np.float64)
        cutoff = float(ordered[take - 1])
        if cutoff > 0 and tail[0] > cutoff:
            logs = np.log(tail[:-1] / cutoff)
            total = float(logs.sum())
            if total > 0:
                exponent = 1.0 + (take - 1) / total
    mean = float(degree.mean())
    variance = float(degree.var())
    return {
        "counts": counts.tolist(),
        "mean": mean,
        "variance": variance,
        "dispersion": variance / mean if mean > 0 else float("nan"),
        "max": int(degree.max()),
        "min": int(degree.min()),
        "gini": _gini(degree.astype(np.float64)),
        "tail_exponent": exponent,
        "tail_sample": take,
        "isolated_nodes": int((degree == 0).sum()),
        "scope": MEASURE_SCOPE
        + "; a Hill tail exponent assumes a Pareto tail and does not test that assumption",
    }


def _gini(values):
    if values.sum() <= 0:
        return 0.0
    ordered = np.sort(values)
    index = np.arange(1, ordered.size + 1)
    return float(
        (2 * (index * ordered).sum()) / (ordered.size * ordered.sum())
        - (ordered.size + 1) / ordered.size
    )


def clustering(network):
    """Global transitivity and mean local clustering coefficient."""
    network = _network(network)
    matrix = network.adjacency
    triangles = np.diag(matrix @ matrix @ matrix)
    degree = network.degree.astype(np.float64)
    pairs = degree * (degree - 1)
    local = np.zeros(network.n)
    usable = pairs > 0
    local[usable] = triangles[usable] / pairs[usable]
    total_pairs = float(pairs.sum())
    return {
        "global_transitivity": float(triangles.sum() / total_pairs) if total_pairs else 0.0,
        "average_local": float(local.mean()),
        "local": local,
        "triangle_count": float(triangles.sum() / 6.0),
        "scope": MEASURE_SCOPE,
    }


def _bfs_distances(matrix, sources):
    n = matrix.shape[0]
    neighbours = matrix > 0
    distances = np.full((len(sources), n), np.inf)
    for row, source in enumerate(sources):
        seen = np.zeros(n, dtype=bool)
        seen[source] = True
        frontier = np.zeros(n, dtype=bool)
        frontier[source] = True
        distances[row, source] = 0.0
        step = 0
        while frontier.any():
            step += 1
            reached = neighbours[frontier].any(axis=0) & ~seen
            if not reached.any():
                break
            distances[row, reached] = step
            seen |= reached
            frontier = reached
    return distances


def path_lengths(network, sources=None):
    """Unweighted shortest-path summary over connected pairs only."""
    network = _network(network)
    if network.n > MAX_PATH_NODES:
        raise Invalid(f"Path lengths are bounded to {MAX_PATH_NODES} nodes")
    if sources is None:
        sources = np.arange(network.n)
    sources = np.asarray(sources, dtype=np.int64)
    if sources.ndim != 1 or sources.size == 0 or sources.max() >= network.n or sources.min() < 0:
        raise Invalid("sources must be valid node indices")
    distances = _bfs_distances(network.adjacency, sources)
    offdiag = distances.copy()
    offdiag[np.arange(sources.size), sources] = np.inf
    finite = np.isfinite(offdiag)
    reachable = float(finite.sum())
    total = float(offdiag.size - sources.size)
    return {
        "mean_path_length": float(offdiag[finite].mean()) if reachable else float("inf"),
        "diameter_over_connected_pairs": float(offdiag[finite].max()) if reachable else float("inf"),
        "connected_pair_fraction": reachable / total if total else 0.0,
        "distances": distances,
        "sources": sources,
        "scope": MEASURE_SCOPE
        + "; means are taken over connected pairs, so a disconnected graph reports "
        "a shorter path length than its true accessibility",
    }


def components(network):
    """Connected components via scipy.sparse.csgraph."""
    network = _network(network)
    count, labels = connected_components(
        csr_matrix(network.adjacency), directed=False, return_labels=True
    )
    sizes = np.bincount(labels)
    return {
        "count": int(count),
        "labels": labels.astype(np.int64),
        "sizes": sizes.tolist(),
        "largest_fraction": float(sizes.max() / network.n),
        "scope": MEASURE_SCOPE,
    }


def _kmeans(points, k, iterations=60):
    """Deterministic k-means: farthest-point initialization, then Lloyd updates."""
    n = points.shape[0]
    centres = [int(np.argmax(((points - points.mean(0)) ** 2).sum(1)))]
    while len(centres) < k:
        distance = np.min(
            ((points[:, None, :] - points[centres][None, :, :]) ** 2).sum(-1), axis=1
        )
        distance[centres] = -1.0
        centres.append(int(np.argmax(distance)))
    centroids = points[centres].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iterations):
        distance = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
        new = np.argmin(distance, axis=1).astype(np.int64)
        if np.array_equal(new, labels):
            break
        labels = new
        for index in range(k):
            member = labels == index
            if member.any():
                centroids[index] = points[member].mean(0)
    return labels


def _canonical_labels(labels):
    order = {}
    output = np.zeros(labels.size, dtype=np.int64)
    for position, value in enumerate(labels):
        if value not in order:
            order[value] = len(order)
        output[position] = order[value]
    return output


def detect_communities(network, k=None, method="spectral"):
    """Deterministic community detection: normalized spectral clustering or label propagation.

    Recovering a planted partition in a generated graph shows that the estimator
    works on that construction. It does not establish that any detected group in
    an observed network corresponds to a real social community.
    """
    network = _network(network)
    if method not in ("spectral", "label_propagation"):
        raise Invalid("unsupported_operation: community method " + str(method))
    if method == "spectral":
        if k is None:
            raise Invalid("Spectral clustering requires an explicit community count k")
        k = _count(k, "k", 2, min(MAX_COMMUNITIES, network.n))
        degree = network.degree.astype(np.float64)
        inverse = 1.0 / np.sqrt(np.maximum(degree, 1e-9))
        normalized = network.adjacency * inverse[:, None] * inverse[None, :]
        values, vectors = np.linalg.eigh(normalized)
        embedding = vectors[:, -k:]
        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        embedding = embedding / np.maximum(norm, 1e-12)
        labels = _canonical_labels(_kmeans(embedding, k))
        spectrum = values[-min(k + 1, values.size) :][::-1]
    else:
        labels = _label_propagation(network)
        spectrum = np.array([])
        k = int(labels.max()) + 1
    sizes = np.bincount(labels, minlength=k)
    return {
        "labels": labels,
        "community_count": int(k),
        "sizes": sizes.tolist(),
        "method": method,
        "modularity": modularity(network, labels)["modularity"],
        "normalized_spectrum": spectrum.tolist(),
        "scope": MEASURE_SCOPE
        + "; a detected partition is an estimate from adjacency alone and is not a "
        "validated social, organizational or functional grouping",
    }


def _label_propagation(network, iterations=40):
    order = np.lexsort((np.arange(network.n), -network.degree))
    labels = np.arange(network.n, dtype=np.int64)
    adjacency = network.adjacency > 0
    for _ in range(iterations):
        changed = False
        for node in order:
            neighbours = np.nonzero(adjacency[node])[0]
            if not neighbours.size:
                continue
            counts = np.bincount(labels[neighbours], minlength=network.n)
            best = int(np.argmax(counts))  # np.argmax breaks ties on the smallest label.
            if best != labels[node]:
                labels[node] = best
                changed = True
        if not changed:
            break
    return _canonical_labels(labels)


def modularity(network, labels):
    """Newman modularity of a supplied partition."""
    network = _network(network)
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape != (network.n,):
        raise Invalid("Partition labels must be one integer per node")
    two_m = float(network.adjacency.sum())
    if two_m == 0:
        return {"modularity": 0.0, "scope": MEASURE_SCOPE}
    same = labels[:, None] == labels[None, :]
    degree = network.degree.astype(np.float64)
    expected = np.outer(degree, degree) / two_m
    value = float(((network.adjacency - expected) * same).sum() / two_m)
    return {"modularity": value, "scope": MEASURE_SCOPE}


def partition_agreement(first, second):
    """Adjusted Rand index between two partitions of the same node set."""
    first = _canonical_labels(np.asarray(first, dtype=np.int64))
    second = _canonical_labels(np.asarray(second, dtype=np.int64))
    if first.shape != second.shape or first.size < 2:
        raise Invalid("Partitions must cover the same node set")
    table = np.zeros((first.max() + 1, second.max() + 1), dtype=np.float64)
    np.add.at(table, (first, second), 1.0)

    def choose2(values):
        return float((values * (values - 1) / 2).sum())

    total = choose2(np.array([float(first.size)]))
    cells = choose2(table)
    rows = choose2(table.sum(1))
    cols = choose2(table.sum(0))
    expected = rows * cols / total if total else 0.0
    maximum = 0.5 * (rows + cols)
    index = (cells - expected) / (maximum - expected) if maximum != expected else 1.0
    return {
        "adjusted_rand_index": float(index),
        "contingency": table,
        "scope": "label-permutation-invariant agreement between two partitions; "
        "agreement with a planted partition is recovery on a construction, not "
        "validation against a real grouping",
    }


def k_core_decomposition(network):
    """Core number per node by iterative degree peeling."""
    network = _network(network)
    adjacency = network.adjacency > 0
    degree = network.degree.copy().astype(np.int64)
    removed = np.zeros(network.n, dtype=bool)
    core = np.zeros(network.n, dtype=np.int64)
    level = 0
    while not removed.all():
        level = max(level, int(degree[~removed].min()))
        peel = (~removed) & (degree <= level)
        while peel.any():
            core[peel] = level
            removed |= peel
            degree -= adjacency[peel].sum(axis=0).astype(np.int64)
            peel = (~removed) & (degree <= level)
    return {
        "core_number": core,
        "max_core": int(core.max()),
        "core_sizes": np.bincount(core).tolist(),
        "scope": MEASURE_SCOPE
        + "; core membership is a structural position, not an established measure "
        "of influence or of operational importance",
    }


def betweenness_centrality(network, normalized=True):
    """Brandes' unweighted betweenness, implemented directly on the adjacency matrix."""
    network = _network(network)
    if network.n > MAX_BETWEENNESS_NODES:
        raise Invalid(f"Betweenness is bounded to {MAX_BETWEENNESS_NODES} nodes")
    adjacency = [np.nonzero(row)[0].tolist() for row in (network.adjacency > 0)]
    n = network.n
    score = np.zeros(n)
    for source in range(n):
        stack = []
        predecessors = [[] for _ in range(n)]
        sigma = np.zeros(n)
        sigma[source] = 1.0
        distance = np.full(n, -1, dtype=np.int64)
        distance[source] = 0
        queue = [source]
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            stack.append(node)
            for neighbour in adjacency[node]:
                if distance[neighbour] < 0:
                    distance[neighbour] = distance[node] + 1
                    queue.append(neighbour)
                if distance[neighbour] == distance[node] + 1:
                    sigma[neighbour] += sigma[node]
                    predecessors[neighbour].append(node)
        delta = np.zeros(n)
        for node in reversed(stack):
            for predecessor in predecessors[node]:
                delta[predecessor] += (sigma[predecessor] / sigma[node]) * (1 + delta[node])
            if node != source:
                score[node] += delta[node]
    score /= 2.0
    if normalized and n > 2:
        score = score * (2.0 / ((n - 1) * (n - 2)))
    return {
        "betweenness": score,
        "normalized": bool(normalized and n > 2),
        "top_nodes": np.argsort(-score)[: min(10, n)].tolist(),
        "scope": MEASURE_SCOPE
        + "; shortest-path betweenness assumes traffic follows shortest paths, "
        "which no diffusion mechanism in this vertical actually obeys",
    }


def _leading_eigenpair(matrix, iterations=2000, tolerance=1e-12):
    n = matrix.shape[0]
    shift = float(np.abs(matrix).sum(axis=1).max()) + 1.0
    shifted = matrix + shift * np.eye(n)
    vector = np.ones(n) + matrix.sum(axis=1)
    vector /= np.linalg.norm(vector)
    value = 0.0
    for _ in range(iterations):
        product = shifted @ vector
        norm = float(np.linalg.norm(product))
        if norm == 0:
            return 0.0, np.zeros(n)
        product /= norm
        if float(np.linalg.norm(product - vector)) < tolerance:
            vector = product
            break
        vector = product
        value = norm
    value = float(vector @ (shifted @ vector))
    vector = np.abs(vector)
    total = float(np.linalg.norm(vector))
    if total > 0:
        vector /= total
    return value - shift, vector


def eigenvector_centrality(network):
    """Leading adjacency eigenvector by shifted power iteration."""
    network = _network(network)
    value, vector = _leading_eigenpair(network.adjacency)
    return {
        "eigenvector": vector,
        "leading_eigenvalue": float(value),
        "top_nodes": np.argsort(-vector)[: min(10, network.n)].tolist(),
        "scope": MEASURE_SCOPE
        + "; on a disconnected graph the leading eigenvector concentrates on one "
        "component and understates every other component's nodes",
    }


def epidemic_threshold(network):
    """Transmissibility threshold 1 / lambda_max from the leading adjacency eigenvalue.

    This is the mean-field / branching-process threshold. It ignores dynamical
    correlations, finite size and clustering, so cascades near the threshold are
    not predicted by it; it separates the clearly subcritical from the clearly
    supercritical regime and nothing finer.
    """
    network = _network(network)
    value, _ = _leading_eigenpair(network.adjacency)
    value = float(value)
    return {
        "leading_eigenvalue": value,
        "threshold": 1.0 / value if value > 0 else float("inf"),
        "mean_degree": network.mean_degree,
        "degree_ratio_threshold": (
            float(network.degree.mean() / (network.degree**2).mean())
            if (network.degree**2).mean() > 0
            else float("inf")
        ),
        "interpretation": "per-contact transmission probabilities well below the "
        "threshold produce cascades that die out; probabilities well above it "
        "produce cascades that reach a finite fraction of the graph",
        "scope": MEASURE_SCOPE
        + "; a mean-field threshold ignores clustering, dynamical correlation and "
        "finite-size effects and is not a validated tipping point",
    }


def cross_community_connectivity(network, labels):
    """Fraction of cross-community node pairs that remain connected in the graph."""
    network = _network(network)
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape != (network.n,):
        raise Invalid("Community labels must be one integer per node")
    _, component_labels = connected_components(
        csr_matrix(network.adjacency), directed=False, return_labels=True
    )
    connected_cross = 0.0
    for component in np.unique(component_labels):
        member = component_labels == component
        size = float(member.sum())
        counts = np.bincount(labels[member])
        same = float((counts * (counts - 1) / 2).sum())
        connected_cross += size * (size - 1) / 2 - same
    counts = np.bincount(labels, minlength=int(labels.max()) + 1).astype(np.float64)
    total_cross = float(network.n * (network.n - 1) / 2 - (counts * (counts - 1) / 2).sum())
    cross_edges = float(
        (network.adjacency * (labels[:, None] != labels[None, :])).sum() / 2
    )
    return {
        "connected_cross_pairs": connected_cross,
        "total_cross_pairs": total_cross,
        "fraction": connected_cross / total_cross if total_cross else 0.0,
        "cross_community_edges": cross_edges,
        "scope": MEASURE_SCOPE,
    }


def remove_nodes(network, nodes):
    """Return a copy with the given nodes isolated (indices are preserved)."""
    network = _network(network)
    nodes = np.asarray(list(nodes), dtype=np.int64)
    matrix = network.adjacency.copy()
    if nodes.size:
        if nodes.min() < 0 or nodes.max() >= network.n:
            raise Invalid("Node indices out of range")
        matrix[nodes, :] = 0.0
        matrix[:, nodes] = 0.0
    return Network(
        adjacency=matrix,
        name=network.name + "_minus_nodes",
        generator=network.generator,
        parameters=dict(network.parameters, removed_nodes=nodes.tolist()),
        communities=network.communities,
        attributes=network.attributes,
        seed=network.seed,
        scope=network.scope + "; nodes isolated by an intervention, not by observation",
    )


def remove_edges(network, edges):
    """Return a copy with the given undirected edges deleted."""
    network = _network(network)
    matrix = network.adjacency.copy()
    for a, b in edges:
        a, b = int(a), int(b)
        if not (0 <= a < network.n and 0 <= b < network.n):
            raise Invalid("Edge endpoints out of range")
        matrix[a, b] = matrix[b, a] = 0.0
    return Network(
        adjacency=matrix,
        name=network.name + "_minus_edges",
        generator=network.generator,
        parameters=dict(network.parameters, removed_edges=[list(map(int, e)) for e in edges]),
        communities=network.communities,
        attributes=network.attributes,
        seed=network.seed,
        scope=network.scope + "; edges deleted by an intervention, not by observation",
    )


def bridge_nodes(network, labels=None, budget=5, candidate_pool=48):
    """Greedy sequential search for nodes whose removal most reduces cross-community reach.

    The objective is the connected cross-community pair fraction, so a node scores
    only when its removal actually disconnects communities. The search is greedy
    over a bounded candidate pool and is therefore not the optimal removal set.
    """
    network = _network(network)
    if labels is None:
        labels = network.communities
    if labels is None:
        raise Invalid("Bridge identification requires community labels")
    labels = np.asarray(labels, dtype=np.int64)
    budget = _count(budget, "budget", 1, min(40, network.n - 1))
    candidate_pool = _count(candidate_pool, "candidate_pool", budget, min(200, network.n))
    cross_incident = (network.adjacency * (labels[:, None] != labels[None, :])).sum(1)
    priority = cross_incident + 1e-6 * network.degree
    pool = np.argsort(-priority)[:candidate_pool]
    pool = [int(node) for node in pool if priority[node] > 0]
    if not pool:
        return {
            "nodes": [],
            "gains": [],
            "baseline_cross_fraction": cross_community_connectivity(network, labels)["fraction"],
            "final_cross_fraction": cross_community_connectivity(network, labels)["fraction"],
            "candidate_pool": 0,
            "scope": MEASURE_SCOPE + "; no node in this graph joins distinct communities",
        }
    baseline = cross_community_connectivity(network, labels)["fraction"]
    current = network
    chosen, gains = [], []
    score = baseline
    for _ in range(budget):
        best_node, best_score = None, score
        for node in pool:
            if node in chosen:
                continue
            trial = remove_nodes(current, [node])
            value = cross_community_connectivity(trial, labels)["fraction"]
            if value < best_score - 1e-12:
                best_node, best_score = node, value
        if best_node is None:
            break
        gains.append(score - best_score)
        score = best_score
        chosen.append(best_node)
        current = remove_nodes(current, [best_node])
    if not chosen:  # No removal disconnects anything: fall back to cross-edge load.
        chosen = [int(node) for node in pool[:budget]]
        gains = [0.0] * len(chosen)
    return {
        "nodes": chosen,
        "gains": gains,
        "baseline_cross_fraction": baseline,
        "final_cross_fraction": score,
        "candidate_pool": len(pool),
        "objective": "connected cross-community pair fraction",
        "scope": MEASURE_SCOPE
        + "; greedy over a bounded candidate pool, so this is a good removal set "
        "and not a proven optimal cut, and structural bridging is not evidence "
        "that these nodes carry real influence",
    }


def bridge_edges(network, labels=None, budget=5, candidate_pool=64):
    """Cross-community edges ranked by the drop in connected cross-community pairs."""
    network = _network(network)
    if labels is None:
        labels = network.communities
    if labels is None:
        raise Invalid("Bridge identification requires community labels")
    labels = np.asarray(labels, dtype=np.int64)
    budget = _count(budget, "budget", 1, 200)
    candidate_pool = _count(candidate_pool, "candidate_pool", budget, 400)
    cross = [
        (int(a), int(b))
        for a, b in network.edges
        if labels[a] != labels[b]
    ]
    if not cross:
        return {
            "edges": [],
            "gains": [],
            "baseline_cross_fraction": cross_community_connectivity(network, labels)["fraction"],
            "scope": MEASURE_SCOPE + "; this graph has no cross-community edge",
        }
    cross = cross[:candidate_pool]
    baseline = cross_community_connectivity(network, labels)["fraction"]
    current = network
    chosen, gains, score = [], [], baseline
    for _ in range(min(budget, len(cross))):
        best_edge, best_score = None, score
        for edge in cross:
            if edge in chosen:
                continue
            value = cross_community_connectivity(remove_edges(current, [edge]), labels)["fraction"]
            if value < best_score - 1e-12:
                best_edge, best_score = edge, value
        if best_edge is None:
            break
        gains.append(score - best_score)
        score = best_score
        chosen.append(best_edge)
        current = remove_edges(current, [best_edge])
    if not chosen:
        chosen = cross[:budget]
        gains = [0.0] * len(chosen)
    return {
        "edges": chosen,
        "gains": gains,
        "baseline_cross_fraction": baseline,
        "final_cross_fraction": score,
        "candidate_count": len(cross),
        "scope": MEASURE_SCOPE
        + "; greedy over cross-community edges only, so a within-community edge "
        "that matters dynamically will not appear here",
    }


def structure_report(network, community_count=None):
    """One bounded structural summary, for the investigation record."""
    network = _network(network)
    report = {
        "network": network.to_dict(),
        "degree": degree_distribution(network),
        "clustering": clustering(network),
        "components": components(network),
        "k_core": k_core_decomposition(network),
        "eigenvector": eigenvector_centrality(network),
        "epidemic_threshold": epidemic_threshold(network),
    }
    if network.n <= MAX_PATH_NODES:
        report["paths"] = path_lengths(network)
    if network.n <= MAX_BETWEENNESS_NODES:
        report["betweenness"] = betweenness_centrality(network)
    if community_count is not None:
        detected = detect_communities(network, k=community_count)
        report["communities"] = detected
        if network.communities is not None:
            report["community_recovery"] = partition_agreement(
                network.communities, detected["labels"]
            )
        report["bridges"] = bridge_nodes(network, detected["labels"], budget=min(5, network.n - 1))
    report["scope"] = (
        "structural description of one supplied graph; nothing here establishes a "
        "diffusion mechanism, an intervention effect, or correspondence with a "
        "real observed network"
    )
    return report
