"""Deterministic diagnostics that characterize a supplied dynamical system.

This module is a pure numeric layer. It never imports a modeling runtime and holds no
state. Every entry point accepts one of three plain inputs:

* a vector field ``f(t, x) -> ndarray`` (a callable, plus starting points that fix the
  state dimension),
* a recorded trajectory as ``times (T,)`` and ``states (T, N)``,
* a signed influence matrix ``adjacency (N, N)`` with optional node labels.

Convention for every matrix argument named ``adjacency``: ``adjacency[i, j]`` is the
signed influence of node ``i`` on node ``j``, so row ``i`` holds the out-edges of ``i``.
A Jacobian uses the opposite convention (``J[i, j] = d f_i / d x_j``); functions that
accept a Jacobian transpose it internally and say so in their result.

Every public function returns a dict carrying a ``scope`` string that states what the
number does not establish. Numerical failure is reported as failure: a solver that does
not converge is recorded as non-convergence, never as the absence of an equilibrium, and
never as a plausible-looking substitute value.

``behavior_descriptor`` and ``descriptor_key`` exist so that an archive can niche on
measured dynamics instead of declared metadata; see their docstrings for the contract a
caller must keep for the key to stay stable.
"""

import math
from collections import deque

import numpy as np
from scipy import optimize, stats

from symplex.core.contracts import Invalid

VERSION = "analysis-v1"

MAX_STATE_DIM = 32
MAX_NODES = 128
MAX_CYCLE_LENGTH = 10
MAX_CYCLES = 5000
MAX_LOOPS_RANKED = 500
MAX_GUESSES = 200
MAX_EQUILIBRIA = 64
MAX_SWEEP_POINTS = 400
MAX_SERIES_POINTS = 200_000
MAX_TRAJECTORY_POINTS = 200_000
MAX_INTEGRATION_STEPS = 500_000
MAX_JACOBIAN_SEQUENCE = 20_000
MAX_EMBEDDED_POINTS = 20_000
_CHUNK_ELEMENTS = 2_000_000

STABILITY_CLASSES = (
    "stable_node",
    "stable_focus",
    "saddle",
    "unstable_node",
    "unstable_focus",
    "center_marginal",
)
BEHAVIOR_CLASSES = (
    "converged",
    "damped_oscillation",
    "sustained_oscillation",
    "drifting",
    "divergent",
)


# --------------------------------------------------------------------------------------
# input validation and small conversions
# --------------------------------------------------------------------------------------


def _array(value, name, ndim=None, finite=True):
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise Invalid(f"{name} must be a real numeric array") from None
    if ndim is not None and array.ndim != ndim:
        raise Invalid(f"{name} must be a {ndim}-dimensional array")
    if array.size == 0:
        raise Invalid(f"{name} must be nonempty")
    if finite and not bool(np.all(np.isfinite(array))):
        raise Invalid(f"{name} must contain only finite values")
    return array


def _positive(value, name, low=0.0, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Invalid(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value) or value <= low:
        raise Invalid(f"{name} must be a finite number greater than {low}")
    if high is not None and value > high:
        raise Invalid(f"{name} must not exceed {high}")
    return value


def _count(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, int):
        raise Invalid(f"{name} must be an integer")
    if not low <= value <= high:
        raise Invalid(f"{name} must be between {low} and {high}")
    return value


def _callable(f, name):
    if not callable(f):
        raise Invalid(f"{name} must be callable")
    return f


def _f(value):
    """Plain float, or None when the value is not finite."""
    value = float(value)
    return value if math.isfinite(value) else None


def _floats(values):
    return [_f(v) for v in np.asarray(values, dtype=float).ravel()]


def _matrix(values):
    return [[_f(v) for v in row] for row in np.asarray(values, dtype=float)]


def _labels_for(labels, count, name="labels"):
    if labels is None:
        return [f"node_{i}" for i in range(count)]
    if not isinstance(labels, (list, tuple)) or len(labels) != count:
        raise Invalid(f"{name} must be a sequence of {count} names")
    names = []
    for label in labels:
        if not isinstance(label, str) or not label.strip() or len(label) > 200:
            raise Invalid(f"{name} entries must be nonempty strings under 200 characters")
        names.append(label)
    if len(set(names)) != len(names):
        raise Invalid(f"{name} entries must be unique")
    return names


def _square(matrix, name, limit):
    array = _array(matrix, name, ndim=2)
    if array.shape[0] != array.shape[1]:
        raise Invalid(f"{name} must be square")
    if not 1 <= array.shape[0] <= limit:
        raise Invalid(f"{name} must have between 1 and {limit} rows")
    return array


def _call_field(f, t, x, dimension):
    """Evaluate the supplied field, converting any failure into Invalid."""
    try:
        value = f(float(t), np.asarray(x, dtype=float))
    except Invalid:
        raise
    except Exception as exc:  # a caller-supplied callable may raise anything
        raise Invalid("Vector field raised during evaluation: " + str(exc)[:200]) from None
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        raise Invalid("Vector field must return a real numeric vector") from None
    if array.shape[0] != dimension:
        raise Invalid(
            f"Vector field returned {array.shape[0]} components for a {dimension}-state system"
        )
    if not bool(np.all(np.isfinite(array))):
        raise Invalid("Vector field returned a nonfinite value")
    return array


# --------------------------------------------------------------------------------------
# 1. equilibria and linear stability
# --------------------------------------------------------------------------------------


def _jacobian_matrix(f, x, eps, t=0.0):
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.shape[0]
    columns = []
    for i in range(n):
        step = eps * max(1.0, abs(float(x[i])))
        forward = x.copy()
        backward = x.copy()
        forward[i] += step
        backward[i] -= step
        # Use the realized step so representation error in x_i +/- h does not bias the slope.
        realized = float(forward[i] - backward[i])
        if realized == 0.0:
            raise Invalid("Finite-difference step underflowed; increase eps")
        columns.append(
            (_call_field(f, t, forward, n) - _call_field(f, t, backward, n)) / realized
        )
    matrix = np.stack(columns, axis=1)
    if not bool(np.all(np.isfinite(matrix))):
        raise Invalid("Finite-difference Jacobian contains nonfinite entries")
    return matrix


def jacobian(f, x, eps=1e-6, t=0.0):
    """Central-difference Jacobian ``J[i, j] = d f_i / d x_j`` at ``x``.

    The per-coordinate step is ``eps * max(1, |x_j|)``. Nothing here checks that the
    field is smooth at ``x``; a discontinuous or stiff field returns a finite matrix that
    is not a derivative.
    """
    _callable(f, "f")
    x = _array(x, "x", ndim=1)
    if x.shape[0] > MAX_STATE_DIM:
        raise Invalid(f"State dimension must not exceed {MAX_STATE_DIM}")
    eps = _positive(eps, "eps", 0.0, 0.1)
    matrix = _jacobian_matrix(f, x, eps, t)
    return {
        "matrix": _matrix(matrix),
        "state": _floats(x),
        "time": _f(t),
        "dimension": int(x.shape[0]),
        "eps": eps,
        "frobenius_norm": _f(np.linalg.norm(matrix)),
        "scheme": "central difference, step eps*max(1,|x_j|), truncation error O(step^2)",
        "scope": (
            "Finite-difference derivative of the supplied field at one point. It is not "
            "an exact derivative, it does not establish that the field is differentiable "
            "or correctly specified there, and the step size trades truncation against "
            "roundoff without an error estimate."
        ),
    }


def stability(J, tol=None):
    """Classify the linearization ``J`` by its spectrum.

    Accepts a matrix or the dict returned by :func:`jacobian`. ``tol`` is the absolute
    distance from the imaginary axis inside which an eigenvalue is treated as marginal;
    it defaults to ``1e-8 * max(1, spectral radius)``. Classification of a marginal
    spectrum is reported as ``center_marginal`` because a linearization decides nothing
    there.
    """
    if isinstance(J, dict):
        if "matrix" not in J:
            raise Invalid("Jacobian dict must carry a 'matrix' entry")
        J = J["matrix"]
    matrix = _square(J, "J", MAX_STATE_DIM)
    n = matrix.shape[0]
    values, vectors = np.linalg.eig(matrix)
    radius = float(np.max(np.abs(values))) if n else 0.0
    if tol is None:
        tol = 1e-8 * max(1.0, radius)
    else:
        tol = _positive(tol, "tol")
    real = values.real
    imag = values.imag
    oscillatory = bool(np.any(np.abs(imag) > tol))
    if bool(np.any(np.abs(real) <= tol)):
        classification = "center_marginal"
    elif bool(np.all(real < 0.0)):
        classification = "stable_focus" if oscillatory else "stable_node"
    elif bool(np.all(real > 0.0)):
        classification = "unstable_focus" if oscillatory else "unstable_node"
    else:
        classification = "saddle"

    order = sorted(range(n), key=lambda i: (-real[i], -abs(imag[i]), i))
    dominant = order[0]
    spectrum = []
    for index in order:
        magnitude = float(abs(values[index]))
        spectrum.append(
            {
                "real": _f(real[index]),
                "imaginary": _f(imag[index]),
                "magnitude": magnitude,
                "damping_ratio": _f(-real[index] / magnitude) if magnitude > 0 else None,
                "period": _f(2.0 * math.pi / abs(imag[index]))
                if abs(imag[index]) > tol
                else None,
                "time_constant": _f(1.0 / abs(real[index]))
                if abs(real[index]) > tol
                else None,
                "growth_rate": _f(real[index]),
            }
        )

    vector = vectors[:, dominant]
    magnitudes = np.abs(vector)
    total = float(magnitudes.sum())
    participation = (
        [_f(v) for v in (magnitudes / total)] if total > 0 else [None] * n
    )
    anchor = int(np.argmax(magnitudes))
    aligned = vector * np.exp(-1j * np.angle(vector[anchor])) if magnitudes[anchor] > 0 else vector
    phases = np.angle(aligned)
    co_movement = []
    for index in range(n):
        if magnitudes[index] <= 1e-12 * max(1.0, float(magnitudes[anchor])):
            co_movement.append(0)
        elif abs(phases[index]) < 1e-6:
            co_movement.append(1)
        elif abs(abs(phases[index]) - math.pi) < 1e-6:
            co_movement.append(-1)
        else:
            co_movement.append(0)

    return {
        "dimension": n,
        "classification": classification,
        "is_stable": classification in ("stable_node", "stable_focus"),
        "is_hyperbolic": classification != "center_marginal",
        "oscillatory": oscillatory,
        "spectral_abscissa": _f(np.max(real)),
        "spectral_radius": _f(radius),
        "trace": _f(np.trace(matrix)),
        "determinant": _f(np.linalg.det(matrix)),
        "eigenvalues": spectrum,
        "damping_ratios": [e["damping_ratio"] for e in spectrum],
        "oscillation_periods": [e["period"] for e in spectrum],
        "dominant_eigenvalue": spectrum[0],
        "dominant_participation": participation,
        "dominant_co_movement": co_movement,
        "marginal_tolerance": tol,
        "participation_meaning": (
            "Normalized magnitudes of the dominant eigenvector: the share of that mode "
            "carried by each state. co_movement is +1/-1 when a component is in phase or "
            "antiphase with the largest component and 0 otherwise."
        ),
        "scope": (
            "Linearized stability of the supplied matrix. It does not establish the real "
            "system's behavior, that the matrix is the correct linearization, or anything "
            "outside a neighbourhood of the point it was taken at. A center_marginal "
            "spectrum means the linearization decides nothing; nonlinear terms do."
        ),
    }


def find_equilibria(
    f,
    x0_guesses,
    t=0.0,
    tol=1e-9,
    dedup_tol=1e-6,
    eps=1e-6,
    method="hybr",
    dimension=None,
):
    """Locate zeros of ``f(t, .)`` with ``scipy.optimize.root`` from several starts.

    Solutions within ``dedup_tol`` (relative to their magnitude) are merged. Starts that
    do not converge are listed under ``failures``; they never become an equilibrium and
    their absence never becomes evidence that no equilibrium exists.
    """
    _callable(f, "f")
    guesses = _array(x0_guesses, "x0_guesses")
    if guesses.ndim == 1:
        guesses = guesses.reshape(1, -1)
    if guesses.ndim != 2:
        raise Invalid("x0_guesses must be a (G, N) array or a single (N,) vector")
    if not 1 <= guesses.shape[0] <= MAX_GUESSES:
        raise Invalid(f"Supply between 1 and {MAX_GUESSES} starting points")
    n = int(guesses.shape[1])
    if not 1 <= n <= MAX_STATE_DIM:
        raise Invalid(f"State dimension must be between 1 and {MAX_STATE_DIM}")
    if dimension is not None and int(dimension) != n:
        raise Invalid("dimension does not match the width of x0_guesses")
    if method not in ("hybr", "lm", "broyden1", "krylov", "df-sane"):
        raise Invalid("Unsupported root-finding method")
    tol = _positive(tol, "tol")
    dedup_tol = _positive(dedup_tol, "dedup_tol")
    eps = _positive(eps, "eps", 0.0, 0.1)

    def residual(x):
        return _call_field(f, t, x, n)

    found, failures = [], []
    for index, guess in enumerate(guesses):
        try:
            with np.errstate(all="ignore"):
                result = optimize.root(residual, guess, method=method)
            candidate = np.asarray(result.x, dtype=float).reshape(-1)
            norm = float(np.linalg.norm(residual(candidate)))
            message = str(getattr(result, "message", ""))[:200]
            success = bool(result.success)
        except Invalid as exc:
            failures.append(
                {
                    "guess_index": index,
                    "guess": _floats(guess),
                    "status": "field_evaluation_failed",
                    "message": str(exc)[:200],
                    "residual_norm": None,
                }
            )
            continue
        if not (success and math.isfinite(norm) and norm <= tol):
            failures.append(
                {
                    "guess_index": index,
                    "guess": _floats(guess),
                    "status": "not_converged",
                    "message": message,
                    "residual_norm": _f(norm),
                    "final_state": _floats(candidate),
                }
            )
            continue
        for entry in found:
            reference = entry["state_array"]
            scale = max(1.0, float(np.linalg.norm(reference)))
            if float(np.linalg.norm(candidate - reference)) <= dedup_tol * scale:
                entry["guess_indices"].append(index)
                if norm < entry["residual_norm"]:
                    entry["state_array"] = candidate
                    entry["residual_norm"] = norm
                break
        else:
            if len(found) >= MAX_EQUILIBRIA:
                raise Invalid(f"More than {MAX_EQUILIBRIA} distinct equilibria located")
            found.append(
                {
                    "state_array": candidate,
                    "residual_norm": norm,
                    "guess_indices": [index],
                }
            )

    found.sort(key=lambda e: tuple(np.round(e["state_array"], 9)))
    equilibria = []
    for entry in found:
        state = entry["state_array"]
        matrix = _jacobian_matrix(f, state, eps, t)
        equilibria.append(
            {
                "state": _floats(state),
                "residual_norm": _f(entry["residual_norm"]),
                "guess_indices": sorted(entry["guess_indices"]),
                "jacobian": _matrix(matrix),
                "stability": stability(matrix),
            }
        )

    if equilibria:
        status = "located"
    elif failures:
        status = "not_converged"
    else:
        status = "no_starts_supplied"
    return {
        "status": status,
        "equilibria": equilibria,
        "equilibrium_count": len(equilibria),
        "failures": failures,
        "failure_count": len(failures),
        "converged_start_count": sum(len(e["guess_indices"]) for e in equilibria),
        "start_count": int(guesses.shape[0]),
        "dimension": n,
        "time": _f(t),
        "method": method,
        "residual_tolerance": tol,
        "deduplication_tolerance": dedup_tol,
        "absence_established": False,
        "scope": (
            "Zeros of the supplied field located numerically from the supplied starting "
            "points at a fixed time. Equilibria no start reaches are not reported, and a "
            "solver failure is recorded as non-convergence, not as the absence of an "
            "equilibrium. Nothing here establishes that the field is correctly specified."
        ),
    }


# --------------------------------------------------------------------------------------
# 2. feedback loop structure
# --------------------------------------------------------------------------------------


def feedback_loops(signed_adjacency, labels=None, max_len=6, weight_tol=0.0, max_cycles=MAX_CYCLES):
    """Enumerate simple directed cycles of a signed influence matrix and classify them.

    ``signed_adjacency[i, j]`` is the signed influence of node ``i`` on node ``j``. A
    cycle is ``reinforcing`` when it contains an even number of negative edges and
    ``balancing`` when odd; its gain is the product of the edge weights along it. Cycles
    are enumerated by bounded depth-first search rooted at each cycle's lowest node
    index, so every simple cycle up to ``max_len`` appears exactly once.
    """
    matrix = _square(signed_adjacency, "signed_adjacency", MAX_NODES)
    n = int(matrix.shape[0])
    names = _labels_for(labels, n)
    max_len = _count(max_len, "max_len", 1, MAX_CYCLE_LENGTH)
    max_cycles = _count(max_cycles, "max_cycles", 1, MAX_CYCLES)
    if weight_tol < 0:
        raise Invalid("weight_tol must not be negative")
    weight_tol = float(weight_tol)

    present = np.abs(matrix) > weight_tol
    successors = [[j for j in range(n) if present[i, j]] for i in range(n)]

    cycles = []

    def extend(start, path, visited):
        node = path[-1]
        for nxt in successors[node]:
            if nxt == start:
                cycles.append(tuple(path))
                if len(cycles) > max_cycles:
                    raise Invalid(
                        f"Cycle enumeration exceeded {max_cycles} simple cycles; "
                        "lower max_len or raise weight_tol"
                    )
            elif nxt > start and nxt not in visited and len(path) < max_len:
                visited.add(nxt)
                path.append(nxt)
                extend(start, path, visited)
                path.pop()
                visited.discard(nxt)

    for start in range(n):
        extend(start, [start], {start})

    cycles.sort(key=lambda c: (len(c), c))
    loops = []
    for rank, cycle in enumerate(cycles):
        edges, gain, negatives = [], 1.0, 0
        for position, source in enumerate(cycle):
            target = cycle[(position + 1) % len(cycle)]
            weight = float(matrix[source, target])
            gain *= weight
            negatives += int(weight < 0)
            edges.append(
                {
                    "source": names[source],
                    "target": names[target],
                    "source_index": int(source),
                    "target_index": int(target),
                    "weight": _f(weight),
                    "sign": int(np.sign(weight)),
                }
            )
        loops.append(
            {
                "id": f"loop_{rank + 1:03d}",
                "indices": [int(i) for i in cycle],
                "nodes": [names[i] for i in cycle],
                "signature": "->".join([names[i] for i in cycle] + [names[cycle[0]]]),
                "length": len(cycle),
                "polarity": "reinforcing" if negatives % 2 == 0 else "balancing",
                "negative_edge_count": negatives,
                "gain": _f(gain),
                "abs_gain": _f(abs(gain)),
                "edges": edges,
            }
        )

    return {
        "loops": loops,
        "loop_count": len(loops),
        "reinforcing_count": sum(1 for entry in loops if entry["polarity"] == "reinforcing"),
        "balancing_count": sum(1 for entry in loops if entry["polarity"] == "balancing"),
        "self_loop_count": sum(1 for entry in loops if entry["length"] == 1),
        "longest_loop": max((entry["length"] for entry in loops), default=0),
        "node_count": n,
        "labels": names,
        "edge_count": int(present.sum()),
        "max_len": max_len,
        "weight_tol": weight_tol,
        "convention": "adjacency[i, j] is the signed influence of node i on node j",
        "polarity_rule": "even count of negative edges is reinforcing; odd is balancing",
        "scope": (
            "Structural enumeration of simple cycles in the supplied matrix. Polarity and "
            "gain are properties of that matrix, not of an observed system: they assume "
            "the influence structure and signs are correct, treat the weights as constant, "
            "and ignore cycles longer than max_len. No loop here is shown to drive any "
            "observed behavior."
        ),
    }


def loop_dominance(loops, jacobian_sequence=None, times=None, top=10):
    """Rank feedback loops by absolute gain, optionally per time step.

    ``loops`` is the dict returned by :func:`feedback_loops` or its ``loops`` list.
    ``jacobian_sequence`` is a ``(T, N, N)`` stack of Jacobians (``J[i, j] = d f_i / d
    x_j``); it is transposed internally to the influence convention before the loop gains
    are recomputed at each step. Contiguous runs with the same top-ranked loop are
    reported as phases.
    """
    if isinstance(loops, dict):
        entries = loops.get("loops")
    else:
        entries = loops
    if not isinstance(entries, list) or not entries:
        raise Invalid("Supply the loops from feedback_loops")
    if len(entries) > MAX_LOOPS_RANKED:
        raise Invalid(f"Rank at most {MAX_LOOPS_RANKED} loops")
    top = _count(top, "top", 1, MAX_LOOPS_RANKED)

    parsed = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict) or "indices" not in entry or "gain" not in entry:
            raise Invalid("Each loop needs 'indices' and 'gain'")
        indices = [int(i) for i in entry["indices"]]
        if not indices or len(set(indices)) != len(indices) or any(i < 0 for i in indices):
            raise Invalid("Loop indices must be unique nonnegative integers")
        gain = entry["gain"]
        if gain is None or not math.isfinite(float(gain)):
            raise Invalid("Loop gain must be a finite number")
        parsed.append(
            {
                "id": entry.get("id", f"loop_{position + 1:03d}"),
                "indices": indices,
                "nodes": entry.get("nodes"),
                "polarity": entry.get("polarity"),
                "length": len(indices),
                "gain": float(gain),
            }
        )

    total = sum(abs(entry["gain"]) for entry in parsed)
    ranked = sorted(parsed, key=lambda e: (-abs(e["gain"]), e["length"], e["id"]))
    ranking = []
    for rank, entry in enumerate(ranked[:top]):
        ranking.append(
            {
                "rank": rank + 1,
                "id": entry["id"],
                "nodes": entry["nodes"],
                "polarity": entry["polarity"],
                "gain": _f(entry["gain"]),
                "abs_gain": _f(abs(entry["gain"])),
                "gain_share": _f(abs(entry["gain"]) / total) if total > 0 else None,
            }
        )

    phases, gain_series, time_points = None, None, None
    if jacobian_sequence is not None:
        stack = _array(jacobian_sequence, "jacobian_sequence", ndim=3)
        steps, rows, columns = stack.shape
        if rows != columns:
            raise Invalid("jacobian_sequence entries must be square")
        if rows > MAX_NODES:
            raise Invalid(f"jacobian_sequence width must not exceed {MAX_NODES}")
        if not 1 <= steps <= MAX_JACOBIAN_SEQUENCE:
            raise Invalid(f"Supply between 1 and {MAX_JACOBIAN_SEQUENCE} Jacobians")
        if any(max(entry["indices"]) >= rows for entry in parsed):
            raise Invalid("A loop references a node outside the Jacobian")
        if times is not None:
            time_array = _array(times, "times", ndim=1)
            if time_array.shape[0] != steps:
                raise Invalid("times must have one entry per Jacobian")
            if not bool(np.all(np.diff(time_array) > 0)):
                raise Invalid("times must be strictly increasing")
            time_points = _floats(time_array)
        influence = np.transpose(stack, (0, 2, 1))
        gains = np.ones((len(parsed), steps), dtype=float)
        for position, entry in enumerate(parsed):
            cycle = entry["indices"]
            for offset, source in enumerate(cycle):
                target = cycle[(offset + 1) % len(cycle)]
                gains[position] *= influence[:, source, target]
        gain_series = {
            entry["id"]: _floats(gains[position]) for position, entry in enumerate(parsed)
        }
        dominant = np.argmax(np.abs(gains), axis=0)
        phases = []
        start = 0
        for step in range(1, steps + 1):
            if step == steps or dominant[step] != dominant[start]:
                index = int(dominant[start])
                entry = parsed[index]
                signed_window = gains[index, start:step]
                window = np.abs(signed_window)
                mean_gain = float(signed_window.mean())
                phases.append(
                    {
                        "start_index": start,
                        "end_index": step - 1,
                        "start_time": time_points[start] if time_points else None,
                        "end_time": time_points[step - 1] if time_points else None,
                        "dominant_loop_id": entry["id"],
                        "polarity": entry["polarity"],
                        "nodes": entry["nodes"],
                        "mean_gain": _f(mean_gain),
                        "mean_abs_gain": _f(window.mean()),
                        "max_abs_gain": _f(window.max()),
                        "polarity_from_jacobian": "reinforcing" if mean_gain > 0 else "balancing",
                    }
                )
                start = step

    return {
        "ranking": ranking,
        "loop_count": len(parsed),
        "ranked_count": len(ranking),
        "dominant_loop_id": ranking[0]["id"] if ranking else None,
        "dominant_polarity": ranking[0]["polarity"] if ranking else None,
        "total_abs_gain": _f(total),
        "time_varying": jacobian_sequence is not None,
        "phases": phases,
        "phase_count": len(phases) if phases is not None else None,
        "gain_series": gain_series,
        "times": time_points,
        "jacobian_convention": "J[i, j] = d f_i / d x_j; transposed internally to influence i -> j",
        "scope": (
            "Structural, linearized dominance: loops are ranked by the absolute product of "
            "edge weights. That is not a causal attribution of observed behavior. Loops "
            "share edges so gains are not additive contributions, a large-gain loop can be "
            "inactive under the current state, and with a Jacobian sequence dominance is "
            "reported only at the supplied time indices. Eigenvalue elasticity analysis, "
            "which would relate loops to observed modes, is not performed."
        ),
    }


# --------------------------------------------------------------------------------------
# 3. tipping points and early-warning signals
# --------------------------------------------------------------------------------------


def _series(values, name="series", limit=MAX_SERIES_POINTS, minimum=4):
    array = _array(values, name, ndim=1)
    if not minimum <= array.shape[0] <= limit:
        raise Invalid(f"{name} must hold between {minimum} and {limit} points")
    return array


def _detrend(values, mode):
    if mode == "none":
        return values
    if mode != "linear":
        raise Invalid("detrend must be 'linear' or 'none'")
    index = np.arange(values.shape[0], dtype=float)
    slope, intercept = np.polyfit(index, values, 1)
    return values - (slope * index + intercept)


def _rolling_indicators(residual, window, lag):
    total = residual.shape[0] - window + 1
    view = np.lib.stride_tricks.sliding_window_view(residual, window)
    variance = np.empty(total)
    skewness = np.empty(total)
    autocorrelation = np.empty(total)
    block = max(1, _CHUNK_ELEMENTS // window)
    for start in range(0, total, block):
        chunk = np.array(view[start : start + block], dtype=float)
        centered = chunk - chunk.mean(axis=1, keepdims=True)
        squares = centered * centered
        sum_squares = squares.sum(axis=1)
        variance[start : start + chunk.shape[0]] = sum_squares / (window - 1)
        second = squares.mean(axis=1)
        third = (centered**3).mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            skewness[start : start + chunk.shape[0]] = np.where(
                second > 0, third / np.power(second, 1.5), np.nan
            )
            autocorrelation[start : start + chunk.shape[0]] = np.where(
                sum_squares > 0,
                (centered[:, :-lag] * centered[:, lag:]).sum(axis=1) / sum_squares,
                np.nan,
            )
    return autocorrelation, variance, skewness


def _kendall(values):
    finite = np.isfinite(values)
    if int(finite.sum()) < 3 or float(np.nanstd(values[finite])) == 0.0:
        return {"kendall_tau": None, "p_value": None, "point_count": int(finite.sum())}
    index = np.arange(values.shape[0], dtype=float)
    result = stats.kendalltau(index[finite], values[finite])
    return {
        "kendall_tau": _f(result.statistic),
        "p_value": _f(result.pvalue),
        "point_count": int(finite.sum()),
    }


def early_warning(series, window=None, detrend="linear", lag=1, min_window=8):
    """Rolling critical-slowing-down indicators with a Kendall tau trend on each.

    Computes rolling lag-1 autocorrelation, variance and skewness over trailing windows,
    then the Kendall rank correlation of each indicator against time. Rising
    autocorrelation and variance are the classical signature of critical slowing down as
    a system approaches a bifurcation; see Scheffer et al., "Early-warning signals for
    critical transitions", Nature 461:53-59 (2009), and Dakos et al., PLoS ONE 7:e41010
    (2012) for the estimation caveats this function inherits.
    """
    values = _series(series)
    total = values.shape[0]
    min_window = _count(min_window, "min_window", 4, total)
    lag = _count(lag, "lag", 1, max(1, min_window - 1))
    if window is None:
        window = max(min_window, total // 5)
    window = _count(window, "window", min_window, total)
    if window <= lag:
        raise Invalid("window must exceed lag")
    if total - window + 1 < 3:
        raise Invalid("Need at least three rolling windows to report a trend")
    if detrend not in ("linear", "none"):
        raise Invalid("detrend must be 'linear' or 'none'")

    residual = _detrend(values, detrend)
    autocorrelation, variance, skewness = _rolling_indicators(residual, window, lag)
    trend = {
        "lag1_autocorrelation": _kendall(autocorrelation),
        "variance": _kendall(variance),
        "skewness": _kendall(skewness),
    }
    rising = sorted(
        name
        for name, result in trend.items()
        if result["kendall_tau"] is not None and result["kendall_tau"] > 0
    )
    undefined = int(
        np.count_nonzero(~np.isfinite(autocorrelation))
        + np.count_nonzero(~np.isfinite(variance))
        + np.count_nonzero(~np.isfinite(skewness))
    )
    autocorrelation_tau = trend["lag1_autocorrelation"]["kendall_tau"]
    variance_tau = trend["variance"]["kendall_tau"]
    return {
        "window": window,
        "lag": lag,
        "detrend": detrend,
        "window_count": int(autocorrelation.shape[0]),
        "window_end_index": list(range(window - 1, total)),
        "lag1_autocorrelation": _floats(autocorrelation),
        "variance": _floats(variance),
        "skewness": _floats(skewness),
        "trend": trend,
        "rising_indicators": rising,
        "slowing_down_consistent": bool(
            autocorrelation_tau is not None
            and variance_tau is not None
            and autocorrelation_tau > 0
            and variance_tau > 0
        ),
        "undefined_indicator_count": undefined,
        "series_length": total,
        "scope": (
            "Rolling indicators of critical slowing down on the supplied series. Rising "
            "autocorrelation and variance are consistent with an approaching bifurcation "
            "but do not establish one: trends in drift, in forcing, in noise level or in "
            "sampling produce the same signature, indicators computed on overlapping "
            "windows are strongly serially dependent so the reported p-values are "
            "anticonservative, and no multiple-comparison correction is applied across "
            "the three indicators. The result depends on the window and detrending choice."
        ),
    }


def bifurcation_scan(
    f_factory,
    param_values,
    x0_guesses,
    parameter_name="parameter",
    t=0.0,
    tol=1e-9,
    dedup_tol=1e-6,
    eps=1e-6,
    continuation=True,
    match_tol=None,
    zero_tol=1e-8,
):
    """Sweep one parameter, track equilibria and their spectra, and flag crossings.

    ``f_factory(p)`` must return a field ``f(t, x)`` for parameter value ``p``. Parameter
    values must be strictly increasing. With ``continuation`` the equilibria located at
    one parameter seed the search at the next. Events are labelled from the spectrum
    alone: a complex pair crossing the imaginary axis is called ``hopf``, a real
    eigenvalue crossing is called ``transcritical_or_pitchfork``, and a change in the
    number of located equilibria is called ``saddle_node``. These are heuristics, not
    normal-form proofs.
    """
    _callable(f_factory, "f_factory")
    parameters = _array(param_values, "param_values", ndim=1)
    if not 2 <= parameters.shape[0] <= MAX_SWEEP_POINTS:
        raise Invalid(f"Supply between 2 and {MAX_SWEEP_POINTS} parameter values")
    if not bool(np.all(np.diff(parameters) > 0)):
        raise Invalid("param_values must be strictly increasing")
    guesses = _array(x0_guesses, "x0_guesses")
    if guesses.ndim == 1:
        guesses = guesses.reshape(1, -1)
    if guesses.ndim != 2:
        raise Invalid("x0_guesses must be a (G, N) array or a single (N,) vector")
    zero_tol = _positive(zero_tol, "zero_tol")
    if match_tol is not None:
        match_tol = _positive(match_tol, "match_tol")

    points, previous = [], []
    for value in parameters:
        field = f_factory(float(value))
        if not callable(field):
            raise Invalid("f_factory must return a callable field f(t, x)")
        starts = list(guesses)
        if continuation:
            starts.extend(previous)
        located = find_equilibria(
            field,
            np.asarray(starts[:MAX_GUESSES], dtype=float),
            t=t,
            tol=tol,
            dedup_tol=dedup_tol,
            eps=eps,
        )
        previous = [np.asarray(e["state"], dtype=float) for e in located["equilibria"]]
        points.append(
            {
                "parameter": _f(value),
                "equilibrium_count": located["equilibrium_count"],
                "unconverged_start_count": located["failure_count"],
                "equilibria": [
                    {
                        "state": entry["state"],
                        "spectral_abscissa": entry["stability"]["spectral_abscissa"],
                        "classification": entry["stability"]["classification"],
                        "oscillatory": entry["stability"]["oscillatory"],
                        "eigenvalues": entry["stability"]["eigenvalues"],
                    }
                    for entry in located["equilibria"]
                ],
            }
        )

    extent = 0.0
    for point in points:
        for entry in point["equilibria"]:
            extent = max(extent, float(np.max(np.abs(np.asarray(entry["state"], dtype=float)))))
    if match_tol is None:
        match_tol = 0.2 * (extent + 1.0)

    branches = []
    active = {}
    for index, point in enumerate(points):
        pairs = []
        for branch_id, tip in active.items():
            for position, entry in enumerate(point["equilibria"]):
                distance = float(
                    np.linalg.norm(np.asarray(entry["state"], dtype=float) - tip)
                )
                if distance <= match_tol:
                    pairs.append((distance, branch_id, position))
        pairs.sort()
        taken_branches, taken_points, assignment = set(), set(), {}
        for distance, branch_id, position in pairs:
            if branch_id in taken_branches or position in taken_points:
                continue
            taken_branches.add(branch_id)
            taken_points.add(position)
            assignment[position] = branch_id
        new_active = {}
        for position, entry in enumerate(point["equilibria"]):
            branch_id = assignment.get(position)
            if branch_id is None:
                branch_id = f"branch_{len(branches) + 1:03d}"
                branches.append(
                    {
                        "id": branch_id,
                        "parameter_indices": [],
                        "parameters": [],
                        "states": [],
                        "spectral_abscissa": [],
                        "classifications": [],
                        "eigenvalues": [],
                    }
                )
            branch = next(b for b in branches if b["id"] == branch_id)
            branch["parameter_indices"].append(index)
            branch["parameters"].append(point["parameter"])
            branch["states"].append(entry["state"])
            branch["spectral_abscissa"].append(entry["spectral_abscissa"])
            branch["classifications"].append(entry["classification"])
            branch["eigenvalues"].append(entry["eigenvalues"])
            new_active[branch_id] = np.asarray(entry["state"], dtype=float)
        active = new_active

    events = []
    for branch in branches:
        abscissa = branch["spectral_abscissa"]
        signed = None
        for position, value in enumerate(abscissa):
            if value is None or abs(value) <= zero_tol:
                continue  # a marginal sample sits on the axis and decides no sign
            if signed is not None and (value > 0) != (abscissa[signed] > 0):
                contiguous = branch["parameter_indices"][position] - branch["parameter_indices"][
                    signed
                ] == position - signed
                if contiguous:
                    before, after = abscissa[signed], value
                    crossing = min(
                        branch["eigenvalues"][position], key=lambda e: abs(e["real"] or 0.0)
                    )
                    complex_pair = (
                        crossing["imaginary"] is not None
                        and abs(crossing["imaginary"]) > zero_tol
                    )
                    span = after - before
                    estimate = branch["parameters"][signed]
                    if span != 0:
                        estimate = branch["parameters"][signed] + (
                            branch["parameters"][position] - branch["parameters"][signed]
                        ) * (-before / span)
                    events.append(
                        {
                            "type": "spectral_abscissa_crossing",
                            "label": "hopf" if complex_pair else "transcritical_or_pitchfork",
                            "branch_id": branch["id"],
                            "parameter_interval": [
                                branch["parameters"][signed],
                                branch["parameters"][position],
                            ],
                            "parameter_estimate": _f(estimate),
                            "abscissa_before": before,
                            "abscissa_after": after,
                            "crossing_eigenvalue": crossing,
                            "direction": "destabilizing" if after > before else "stabilizing",
                            "label_basis": "sign change of the spectral abscissa along one tracked "
                            "branch; the crossing eigenvalue's imaginary part selects the label",
                        }
                    )
            signed = position
    for index in range(len(points) - 1):
        before = points[index]["equilibrium_count"]
        after = points[index + 1]["equilibrium_count"]
        if before == after:
            continue
        events.append(
            {
                "type": "equilibrium_count_change",
                "label": "saddle_node",
                "branch_id": None,
                "parameter_interval": [
                    points[index]["parameter"],
                    points[index + 1]["parameter"],
                ],
                "parameter_estimate": _f(
                    0.5 * (points[index]["parameter"] + points[index + 1]["parameter"])
                ),
                "from_count": before,
                "to_count": after,
                "unconverged_start_count": [
                    points[index]["unconverged_start_count"],
                    points[index + 1]["unconverged_start_count"],
                ],
                "label_basis": "the number of located equilibria changed between adjacent "
                "parameter values; a fold is the generic mechanism, but a transcritical "
                "exchange, a branch leaving the region the starts cover, or solver "
                "non-convergence produce the same signature",
            }
        )
    events.sort(key=lambda e: (e["parameter_interval"][0], e["type"], e["label"]))

    return {
        "parameter_name": str(parameter_name)[:200],
        "parameter_values": _floats(parameters),
        "point_count": len(points),
        "points": points,
        "branches": branches,
        "branch_count": len(branches),
        "events": events,
        "event_count": len(events),
        "bifurcation_detected": bool(events),
        "labels_detected": sorted({event["label"] for event in events}),
        "branch_match_tolerance": _f(match_tol),
        "zero_tolerance": zero_tol,
        "continuation": bool(continuation),
        "parameter_values_without_located_equilibria": [
            point["parameter"] for point in points if point["equilibrium_count"] == 0
        ],
        "label_confidence": "heuristic",
        "scope": (
            "A parameter sweep of numerically located equilibria and their spectra. Event "
            "labels are heuristics read off the spectrum and the equilibrium count; they "
            "are not a normal-form analysis and do not establish that a bifurcation of the "
            "named type occurs. Resolution is limited to the supplied parameter spacing, "
            "branches are matched by distance so they can be swapped near collisions, "
            "equilibria outside the reach of the starts are invisible, and no equilibrium "
            "is shown to be reachable from any real initial condition."
        ),
    }


def _segment_cost(prefix, prefix_squares, start, end, statistic, floor):
    length = end - start
    total = prefix[end] - prefix[start]
    squares = prefix_squares[end] - prefix_squares[start]
    deviation = squares - (total * total) / length
    deviation = np.maximum(deviation, 0.0)
    if statistic == "mean":
        return deviation
    return length * np.log(np.maximum(deviation / length, floor))


def regime_segments(series, max_segments=6, min_size=5, penalty=None, statistic="mean_variance"):
    """Deterministic binary segmentation of a series under a penalized Gaussian cost.

    ``mean_variance`` uses the Gaussian log-likelihood cost ``n * log(sigma^2)``, which
    responds to shifts in level and in spread; ``mean`` uses the residual sum of squares,
    which responds to level only. Splits are accepted while the cost reduction exceeds
    ``penalty``. There is no randomness and no restart: the same series always yields the
    same segmentation.
    """
    values = _series(series, minimum=4)
    total = int(values.shape[0])
    max_segments = _count(max_segments, "max_segments", 1, 32)
    min_size = _count(min_size, "min_size", 2, max(2, total // 2))
    if statistic not in ("mean_variance", "mean"):
        raise Invalid("statistic must be 'mean_variance' or 'mean'")
    parameters = 2 if statistic == "mean_variance" else 1
    if penalty is None:
        # BIC for one change point: the segment parameters plus the location itself.
        penalty = (parameters + 1) * math.log(total)
        if statistic == "mean":
            penalty *= float(np.var(values))  # the residual-sum cost carries the data scale
    penalty = float(penalty)
    if not math.isfinite(penalty) or penalty < 0:
        raise Invalid("penalty must be a finite nonnegative number")

    prefix = np.concatenate([[0.0], np.cumsum(values)])
    prefix_squares = np.concatenate([[0.0], np.cumsum(values * values)])
    floor = 1e-12 * max(1.0, float(np.var(values)))

    def cost(start, end):
        return float(
            _segment_cost(prefix, prefix_squares, start, end, statistic, floor)
        )

    boundaries = [0, total]
    accepted = []
    for _ in range(max_segments - 1):
        best = None
        for position in range(len(boundaries) - 1):
            start, end = boundaries[position], boundaries[position + 1]
            if end - start < 2 * min_size:
                continue
            splits = np.arange(start + min_size, end - min_size + 1)
            if splits.size == 0:
                continue
            left = _segment_cost(prefix, prefix_squares, start, splits, statistic, floor)
            right = _segment_cost(prefix, prefix_squares, splits, end, statistic, floor)
            gains = cost(start, end) - (left + right)
            index = int(np.argmax(gains))
            gain = float(gains[index])
            if best is None or gain > best[0]:
                best = (gain, int(splits[index]))
        if best is None or best[0] <= penalty:
            break
        accepted.append({"index": best[1], "cost_reduction": _f(best[0])})
        boundaries = sorted(boundaries + [best[1]])

    segments = []
    for position in range(len(boundaries) - 1):
        start, end = boundaries[position], boundaries[position + 1]
        window = values[start:end]
        segments.append(
            {
                "index_start": start,
                "index_end": end - 1,
                "length": end - start,
                "mean": _f(window.mean()),
                "std": _f(window.std(ddof=1)) if end - start > 1 else 0.0,
                "min": _f(window.min()),
                "max": _f(window.max()),
            }
        )
    accepted.sort(key=lambda entry: entry["index"])
    return {
        "segments": segments,
        "segment_count": len(segments),
        "change_points": [entry["index"] for entry in accepted],
        "change_point_details": accepted,
        "penalty": _f(penalty),
        "statistic": statistic,
        "min_size": min_size,
        "max_segments": max_segments,
        "total_cost": _f(sum(cost(b, e) for b, e in zip(boundaries[:-1], boundaries[1:]))),
        "series_length": total,
        "algorithm": "binary segmentation, exhaustive split scan, deterministic tie-break to the earliest index",
        "scope": (
            "Best split points of the supplied series under the stated cost and penalty. "
            "They are not detected regime changes in any system: the number of segments "
            "follows directly from the penalty, no significance test is performed, serial "
            "correlation and slow drift both manufacture apparent change points, and "
            "binary segmentation is greedy so the segmentation need not be globally "
            "optimal."
        ),
    }


# --------------------------------------------------------------------------------------
# 4. chaos and predictability
# --------------------------------------------------------------------------------------


def _rk4(f, t, x, dt, dimension):
    k1 = _call_field(f, t, x, dimension)
    k2 = _call_field(f, t + 0.5 * dt, x + 0.5 * dt * k1, dimension)
    k3 = _call_field(f, t + 0.5 * dt, x + 0.5 * dt * k2, dimension)
    k4 = _call_field(f, t + dt, x + dt * k3, dimension)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def lyapunov_max(
    f,
    x0,
    dt=0.01,
    steps=20000,
    transient=0,
    discrete=False,
    delta=1e-8,
    t0=0.0,
    divergence_bound=1e12,
):
    """Largest Lyapunov exponent of a known field by Benettin renormalization.

    Two trajectories separated by ``delta`` are advanced together and the separation is
    renormalized to ``delta`` after every step, accumulating the log expansion factor.
    With ``discrete=True`` the callable is treated as a map ``x_{n+1} = f(n, x_n)`` and
    ``dt`` is forced to 1; otherwise it is integrated with fixed-step RK4. The estimate
    over the second half is reported alongside the full-run estimate so a caller can see
    whether it has settled. For a bare time series use :func:`lyapunov_rosenstein`.
    """
    _callable(f, "f")
    state = _array(x0, "x0", ndim=1)
    dimension = int(state.shape[0])
    if dimension > MAX_STATE_DIM:
        raise Invalid(f"State dimension must not exceed {MAX_STATE_DIM}")
    steps = _count(steps, "steps", 10, MAX_INTEGRATION_STEPS)
    transient = _count(transient, "transient", 0, MAX_INTEGRATION_STEPS)
    if transient + steps > MAX_INTEGRATION_STEPS:
        raise Invalid(f"transient + steps must not exceed {MAX_INTEGRATION_STEPS}")
    delta = _positive(delta, "delta", 0.0, 1e-2)
    divergence_bound = _positive(divergence_bound, "divergence_bound")
    if discrete:
        dt = 1.0
    else:
        dt = _positive(dt, "dt", 0.0, 1e3)

    def advance(time, point):
        if discrete:
            return _call_field(f, time, point, dimension)
        return _rk4(f, time, point, dt, dimension)

    time = float(t0)
    for _ in range(transient):
        state = advance(time, state)
        time += dt
        if not bool(np.all(np.isfinite(state))) or float(np.max(np.abs(state))) > divergence_bound:
            return _lyapunov_failure("diverged_during_transient", dt, steps, transient, discrete)

    direction = np.array([1.0 / (i + 1) for i in range(dimension)])
    direction /= np.linalg.norm(direction)
    partner = state + delta * direction

    accumulated = 0.0
    half = steps // 2
    accumulated_second_half = 0.0
    completed = 0
    for step in range(steps):
        state = advance(time, state)
        partner = advance(time, partner)
        time += dt
        separation = partner - state
        distance = float(np.linalg.norm(separation))
        if (
            not math.isfinite(distance)
            or distance == 0.0
            or not bool(np.all(np.isfinite(state)))
            or float(np.max(np.abs(state))) > divergence_bound
        ):
            return _lyapunov_failure(
                "collapsed_separation" if distance == 0.0 else "diverged",
                dt,
                steps,
                transient,
                discrete,
                completed=completed,
            )
        growth = math.log(distance / delta)
        accumulated += growth
        if step >= half:
            accumulated_second_half += growth
        partner = state + separation * (delta / distance)
        completed += 1

    exponent = accumulated / (steps * dt)
    second_half_steps = steps - half
    second_half = accumulated_second_half / (second_half_steps * dt)
    scale = max(abs(exponent), abs(second_half), 1e-12)
    return {
        "status": "estimated",
        "lambda_max": _f(exponent),
        "lambda_max_second_half": _f(second_half),
        "relative_drift": _f(abs(second_half - exponent) / scale),
        "settled": bool(abs(second_half - exponent) <= 0.1 * scale),
        "positive": bool(exponent > 0.0),
        "predictability_horizon": _f(1.0 / exponent) if exponent > 0 else None,
        "horizon_definition": "1/lambda_max in the same time units as dt; reported only for a positive exponent",
        "steps": steps,
        "transient": transient,
        "dt": dt,
        "discrete": bool(discrete),
        "delta": delta,
        "dimension": dimension,
        "method": "Benettin two-trajectory renormalization",
        "scope": (
            "A finite-time estimate for the supplied field from one initial condition. It "
            "does not establish chaos: the value depends on the initial condition, the "
            "transient discarded, the step size and the renormalization distance, no "
            "convergence proof is attempted, and a positive number can come from a "
            "transient, from integration error, or from an unbounded orbit. The "
            "predictability horizon is 1/lambda, an order-of-magnitude statement about "
            "this model, not about a real system."
        ),
    }


def _lyapunov_failure(status, dt, steps, transient, discrete, completed=0):
    return {
        "status": status,
        "lambda_max": None,
        "lambda_max_second_half": None,
        "relative_drift": None,
        "settled": False,
        "positive": None,
        "predictability_horizon": None,
        "horizon_definition": "not reported; the run did not complete",
        "steps": steps,
        "completed_steps": completed,
        "transient": transient,
        "dt": dt,
        "discrete": bool(discrete),
        "method": "Benettin two-trajectory renormalization",
        "scope": (
            "The run terminated before an exponent could be formed, so no exponent is "
            "reported. An unbounded or collapsing orbit is a fact about this integration, "
            "not evidence about the system's exponents."
        ),
    }


def _mean_period(values, dt=1.0):
    centered = values - values.mean()
    if float(np.dot(centered, centered)) == 0.0:
        return None
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    spectrum[0] = 0.0
    if float(spectrum.sum()) <= 0:
        return None
    peak = int(np.argmax(spectrum))
    if peak == 0:
        return None
    return float(values.shape[0]) * dt / peak


def lyapunov_rosenstein(
    series,
    dim=None,
    lag=None,
    dt=1.0,
    theiler=None,
    horizon=None,
    fit_range=None,
    max_references=1500,
):
    """Rosenstein-style largest Lyapunov exponent for a bare scalar time series.

    Embeds the series, tracks the mean logarithmic separation of nearest-neighbour pairs
    over ``horizon`` steps, and fits a slope over ``fit_range``. Follows Rosenstein,
    Collins and De Luca, Physica D 65:117-134 (1993). Embedding parameters default to the
    suggestions from :func:`attractor_embedding`.
    """
    values = _series(series, minimum=32)
    dt = _positive(dt, "dt")
    embedding = attractor_embedding(values, dim=dim, lag=lag)
    dim = embedding["dim"]
    lag = embedding["lag"]
    points = np.asarray(embedding["embedding"], dtype=float)
    count = int(points.shape[0])
    if count < 32:
        raise Invalid("Embedded trajectory is too short for a divergence estimate")
    if horizon is None:
        horizon = max(4, min(count // 10, 64))
    horizon = _count(horizon, "horizon", 2, max(2, count // 2))
    if theiler is None:
        period = _mean_period(values)
        theiler = int(max(lag, min(count // 10, int(period) if period else lag)))
    theiler = _count(theiler, "theiler", 1, max(1, count // 4))
    max_references = _count(max_references, "max_references", 10, 5000)

    usable = count - horizon
    if usable <= theiler + 2:
        raise Invalid("Series is too short for the requested horizon and Theiler window")
    stride = max(1, math.ceil(usable / max_references))
    references = np.arange(0, usable, stride)

    index = np.arange(count)
    divergence = np.zeros(horizon + 1)
    counts = np.zeros(horizon + 1)
    block = max(1, _CHUNK_ELEMENTS // max(1, count))
    for start in range(0, references.shape[0], block):
        chunk = references[start : start + block]
        distances = np.linalg.norm(points[chunk, None, :] - points[None, :, :], axis=2)
        mask = np.abs(index[None, :] - chunk[:, None]) <= theiler
        mask |= index[None, :] > usable - 1
        distances[mask] = np.inf
        neighbours = np.argmin(distances, axis=1)
        valid = np.isfinite(distances[np.arange(chunk.shape[0]), neighbours])
        chunk = chunk[valid]
        neighbours = neighbours[valid]
        if chunk.size == 0:
            continue
        for step in range(horizon + 1):
            separation = np.linalg.norm(
                points[chunk + step] - points[neighbours + step], axis=1
            )
            positive = separation > 0
            divergence[step] += float(np.log(separation[positive]).sum())
            counts[step] += int(positive.sum())

    if not bool(np.all(counts > 0)):
        raise Invalid("Neighbour separations collapsed to zero; the series may be constant")
    curve = divergence / counts
    if fit_range is None:
        fit_range = (1, max(2, horizon // 2))
    if (
        not isinstance(fit_range, (tuple, list))
        or len(fit_range) != 2
        or not 0 <= int(fit_range[0]) < int(fit_range[1]) <= horizon
    ):
        raise Invalid("fit_range must be an increasing (start, end) pair inside the horizon")
    first, last = int(fit_range[0]), int(fit_range[1])
    steps = np.arange(first, last + 1, dtype=float)
    window = curve[first : last + 1]
    slope, intercept = np.polyfit(steps, window, 1)
    predicted = slope * steps + intercept
    variance = float(np.sum((window - window.mean()) ** 2))
    r_squared = 1.0 - float(np.sum((window - predicted) ** 2)) / variance if variance > 0 else None
    exponent = float(slope) / dt
    return {
        "status": "estimated",
        "lambda_max": _f(exponent),
        "positive": bool(exponent > 0),
        "predictability_horizon": _f(1.0 / exponent) if exponent > 0 else None,
        "divergence_curve": _floats(curve),
        "fit_range": [first, last],
        "fit_slope_per_step": _f(slope),
        "fit_intercept": _f(intercept),
        "fit_r_squared": _f(r_squared) if r_squared is not None else None,
        "dim": dim,
        "lag": lag,
        "dt": dt,
        "theiler_window": theiler,
        "horizon": horizon,
        "reference_count": int(references.shape[0]),
        "embedded_point_count": count,
        "method": "Rosenstein mean log divergence of nearest-neighbour pairs",
        "scope": (
            "A slope fitted to the mean logarithmic divergence of the supplied series "
            "under one embedding. It is not a proof of chaos and not a property of any "
            "system: the value depends on the embedding dimension, the lag, the Theiler "
            "window and above all on the fit window, measurement noise inflates the early "
            "slope, and a stochastic series with short correlation time yields a positive "
            "slope too. No surrogate-data test is performed."
        ),
    }


def _delay_embed(values, dim, lag):
    count = values.shape[0] - (dim - 1) * lag
    if count <= 0:
        raise Invalid("Series is too short for the requested embedding dimension and lag")
    return np.stack([values[i * lag : i * lag + count] for i in range(dim)], axis=1)


def _mutual_information(values, lag, bins):
    first, second = values[:-lag], values[lag:]
    counts, _, _ = np.histogram2d(first, second, bins=bins)
    total = counts.sum()
    if total <= 0:
        return 0.0
    joint = counts / total
    row = joint.sum(axis=1, keepdims=True)
    column = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        contribution = np.where(joint > 0, joint * np.log(joint / (row * column)), 0.0)
    return float(np.sum(contribution))


def attractor_embedding(
    series,
    dim=None,
    lag=None,
    max_lag=64,
    max_dim=10,
    bins=None,
    fnn_threshold=0.01,
    r_tol=15.0,
    a_tol=2.0,
    max_references=1000,
):
    """Time-delay embedding with a mutual-information lag and a false-nearest-neighbour dimension.

    The suggested lag is the first local minimum of the average mutual information
    between ``x(t)`` and ``x(t + lag)`` (Fraser and Swinney, Phys. Rev. A 33:1134, 1986);
    if there is none, the first lag where the information has fallen by a factor of e.
    The suggested dimension is the smallest ``m`` whose false-nearest-neighbour fraction
    falls to ``fnn_threshold`` (Kennel, Brown and Abarbanel, Phys. Rev. A 45:3403, 1992);
    ``r_tol`` and ``a_tol`` are that paper's two false-neighbour criteria.
    ``embedding`` is returned as an ndarray; every other value is JSON-native.
    """
    values = _series(series, minimum=32)
    total = int(values.shape[0])
    max_lag = _count(max_lag, "max_lag", 1, max(1, total // 5))
    max_dim = _count(max_dim, "max_dim", 1, 16)
    max_references = _count(max_references, "max_references", 10, 5000)
    r_tol = _positive(r_tol, "r_tol")
    a_tol = _positive(a_tol, "a_tol")
    if not 0.0 <= float(fnn_threshold) <= 1.0:
        raise Invalid("fnn_threshold must lie in [0, 1]")
    fnn_threshold = float(fnn_threshold)
    if bins is None:
        bins = int(min(32, max(4, round(math.sqrt(total / 5.0)))))
    bins = _count(bins, "bins", 2, 128)

    information = [_mutual_information(values, offset, bins) for offset in range(1, max_lag + 1)]
    minimum_lag = None
    for position in range(1, len(information) - 1):
        if (
            information[position] < information[position - 1]
            and information[position] <= information[position + 1]
        ):
            minimum_lag = position + 1
            break
    decline_lag = None
    threshold = information[0] / math.e if information else 0.0
    for position, value in enumerate(information):
        if value <= threshold:
            decline_lag = position + 1
            break
    # Take the earlier of the two standard criteria: an over-long lag folds the
    # reconstruction, while an over-short one only leaves it correlated.
    candidates = [lag for lag in (minimum_lag, decline_lag) if lag is not None]
    if not candidates:
        suggested_lag = max_lag
        lag_basis = "neither a local minimum nor an e-fold decline of the mutual information; fell back to max_lag"
    else:
        suggested_lag = min(candidates)
        if suggested_lag == minimum_lag and suggested_lag == decline_lag:
            lag_basis = "first local minimum of the mutual information, which is also its first e-fold decline"
        elif suggested_lag == minimum_lag:
            lag_basis = "first local minimum of the average mutual information"
        else:
            lag_basis = "first lag at which the mutual information falls below I(1)/e"

    chosen_lag = suggested_lag if lag is None else _count(int(lag), "lag", 1, max(1, total // 4))
    spread = float(np.std(values))
    fractions = {}
    suggested_dim, dim_basis = max_dim, f"no dimension reached the {fnn_threshold} threshold; fell back to max_dim"
    for candidate in range(1, max_dim + 1):
        try:
            low = _delay_embed(values, candidate, chosen_lag)
            high = _delay_embed(values, candidate + 1, chosen_lag)
        except Invalid:
            break
        count = int(high.shape[0])
        if count < 10:
            break
        low = low[:count]
        stride = max(1, math.ceil(count / max_references))
        references = np.arange(0, count, stride)
        false_count, checked = 0, 0
        block = max(1, _CHUNK_ELEMENTS // max(1, count))
        for start in range(0, references.shape[0], block):
            chunk = references[start : start + block]
            distances = np.linalg.norm(low[chunk, None, :] - low[None, :, :], axis=2)
            distances[np.arange(chunk.shape[0]), chunk] = np.inf
            distances[distances == 0.0] = np.inf
            neighbours = np.argmin(distances, axis=1)
            base = distances[np.arange(chunk.shape[0]), neighbours]
            usable = np.isfinite(base)
            if not bool(np.any(usable)):
                continue
            extra = np.abs(
                high[chunk[usable], candidate] - high[neighbours[usable], candidate]
            )
            ratio = extra / base[usable]
            expanded = np.linalg.norm(high[chunk[usable]] - high[neighbours[usable]], axis=1)
            false_count += int(
                np.count_nonzero(
                    (ratio > r_tol) | (expanded > a_tol * spread if spread > 0 else False)
                )
            )
            checked += int(np.count_nonzero(usable))
        if checked == 0:
            break
        fraction = false_count / checked
        fractions[candidate] = _f(fraction)
        if fraction <= fnn_threshold and suggested_dim == max_dim and dim_basis.startswith("no dimension"):
            suggested_dim = candidate
            dim_basis = "smallest dimension whose false-nearest-neighbour fraction reached the threshold"

    chosen_dim = suggested_dim if dim is None else _count(int(dim), "dim", 1, max_dim + 4)
    embedding = _delay_embed(values, chosen_dim, chosen_lag)
    if embedding.shape[0] > MAX_EMBEDDED_POINTS:
        raise Invalid(f"Embedding would exceed {MAX_EMBEDDED_POINTS} points")
    return {
        "embedding": embedding,
        "point_count": int(embedding.shape[0]),
        "dim": chosen_dim,
        "lag": chosen_lag,
        "suggested_dim": suggested_dim,
        "suggested_lag": suggested_lag,
        "lag_basis": lag_basis,
        "dim_basis": dim_basis,
        "mutual_information": _floats(information),
        "mutual_information_lags": list(range(1, max_lag + 1)),
        "mutual_information_bins": bins,
        "false_nearest_neighbour_fraction": fractions,
        "fnn_threshold": fnn_threshold,
        "series_length": total,
        "scope": (
            "A delay reconstruction of the supplied series with parameters chosen by two "
            "standard heuristics. Takens' theorem guarantees an embedding only for a "
            "noise-free deterministic system observed exactly; these estimates are "
            "sensitive to noise, to the histogram binning, to the record length and to "
            "non-stationarity, and nothing here establishes that the series came from a "
            "low-dimensional deterministic system."
        ),
    }


# --------------------------------------------------------------------------------------
# 5. network structure
# --------------------------------------------------------------------------------------


def _undirected(matrix, tol=0.0):
    present = np.abs(matrix) > tol
    np.fill_diagonal(present, False)
    return (present | present.T).astype(float)


def _betweenness(successors, n):
    """Brandes (2001) accumulation on the unweighted directed graph."""
    scores = np.zeros(n)
    for source in range(n):
        order, predecessors = [], [[] for _ in range(n)]
        paths = np.zeros(n)
        distance = -np.ones(n)
        paths[source] = 1.0
        distance[source] = 0.0
        queue = deque([source])
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbour in successors[node]:
                if distance[neighbour] < 0:
                    distance[neighbour] = distance[node] + 1
                    queue.append(neighbour)
                if distance[neighbour] == distance[node] + 1:
                    paths[neighbour] += paths[node]
                    predecessors[neighbour].append(node)
        dependency = np.zeros(n)
        while order:
            node = order.pop()
            for predecessor in predecessors[node]:
                dependency[predecessor] += (paths[predecessor] / paths[node]) * (
                    1.0 + dependency[node]
                )
            if node != source:
                scores[node] += dependency[node]
    return scores


def _components(neighbours, n):
    seen, groups = set(), []
    for start in range(n):
        if start in seen:
            continue
        group, queue = [], deque([start])
        seen.add(start)
        while queue:
            node = queue.popleft()
            group.append(node)
            for neighbour in neighbours[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
        groups.append(sorted(group))
    return sorted(groups, key=lambda g: (-len(g), g))


def _strong_components(successors, predecessors, n):
    """Iterative Kosaraju: order by finish time, then peel on the reversed graph."""
    order, seen = [], set()
    for start in range(n):
        if start in seen:
            continue
        stack = [(start, iter(successors[start]))]
        seen.add(start)
        while stack:
            node, iterator = stack[-1]
            advanced = False
            for neighbour in iterator:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append((neighbour, iter(successors[neighbour])))
                    advanced = True
                    break
            if not advanced:
                order.append(stack.pop()[0])
    assigned, groups = set(), []
    for node in reversed(order):
        if node in assigned:
            continue
        group, queue = [], deque([node])
        assigned.add(node)
        while queue:
            current = queue.popleft()
            group.append(current)
            for neighbour in predecessors[current]:
                if neighbour not in assigned:
                    assigned.add(neighbour)
                    queue.append(neighbour)
        groups.append(sorted(group))
    return sorted(groups, key=lambda g: (-len(g), g))


def _eigenvector_centrality(magnitudes, iterations=1000, tol=1e-12):
    """Power iteration on |A|^T + sI. The shift keeps a bipartite graph from oscillating."""
    n = magnitudes.shape[0]
    bound = float(max(magnitudes.sum(axis=0).max(), magnitudes.sum(axis=1).max()))
    if bound <= 0:
        return np.zeros(n), 0.0, False, 0, "no_edges"
    operator = magnitudes.T + bound * np.eye(n)
    vector = np.ones(n) / math.sqrt(n)
    value, converged, used = 0.0, False, 0
    for step in range(iterations):
        updated = operator @ vector
        norm = float(np.linalg.norm(updated))
        used = step + 1
        if norm <= 1e-300:
            return np.zeros(n), 0.0, False, used, "no_positive_eigenvector"
        updated /= norm
        movement = float(np.linalg.norm(updated - vector))
        vector, value = updated, norm
        if movement < tol:
            converged = True
            break
    total = float(np.abs(vector).sum())
    scores = np.abs(vector) / total if total > 0 else np.zeros(n)
    return scores, value - bound, converged, used, "converged" if converged else "iteration_limit"


def network_metrics(adjacency, labels=None, tol=0.0):
    """Degree, strength, betweenness, eigenvector centrality, clustering and components.

    ``adjacency[i, j]`` is the influence of ``i`` on ``j``. Degrees exclude self-loops,
    which are counted separately. Betweenness uses Brandes' unweighted shortest-path
    accumulation on the directed graph and is normalized by the number of ordered pairs
    (halved when the matrix is symmetric). Eigenvector centrality is a power iteration on
    the absolute-value matrix and its convergence status is reported; clustering is
    computed on the binarized undirected projection. Implemented with numpy only.
    """
    matrix = _square(adjacency, "adjacency", MAX_NODES)
    n = int(matrix.shape[0])
    names = _labels_for(labels, n)
    if tol < 0:
        raise Invalid("tol must not be negative")
    tol = float(tol)

    present = np.abs(matrix) > tol
    self_loops = [names[i] for i in range(n) if present[i, i]]
    off_diagonal = present.copy()
    np.fill_diagonal(off_diagonal, False)
    magnitudes = np.abs(matrix) * off_diagonal

    out_degree = off_diagonal.sum(axis=1)
    in_degree = off_diagonal.sum(axis=0)
    out_strength = magnitudes.sum(axis=1)
    in_strength = magnitudes.sum(axis=0)

    successors = [[j for j in range(n) if off_diagonal[i, j]] for i in range(n)]
    predecessors = [[i for i in range(n) if off_diagonal[i, j]] for j in range(n)]
    symmetric = bool(np.array_equal(off_diagonal, off_diagonal.T))
    raw_betweenness = _betweenness(successors, n)
    if n > 2:
        pairs = (n - 1) * (n - 2)
        betweenness = (raw_betweenness / 2.0) / (pairs / 2.0) if symmetric else raw_betweenness / pairs
    else:
        betweenness = np.zeros(n)

    centrality, spectral, converged, iterations, reason = _eigenvector_centrality(magnitudes)

    undirected = _undirected(matrix, tol)
    degree = undirected.sum(axis=1)
    triangles = np.einsum("ij,jk,ki->i", undirected, undirected, undirected) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        clustering = np.where(degree > 1, 2.0 * triangles / (degree * (degree - 1)), 0.0)
    triples = float(np.sum(degree * (degree - 1)))
    transitivity = float(np.trace(undirected @ undirected @ undirected) / triples) if triples > 0 else None

    neighbours = [[j for j in range(n) if undirected[i, j] > 0] for i in range(n)]
    weak = _components(neighbours, n)
    strong = _strong_components(successors, predecessors, n)

    nodes = [
        {
            "label": names[index],
            "index": index,
            "in_degree": int(in_degree[index]),
            "out_degree": int(out_degree[index]),
            "degree": int(degree[index]),
            "in_strength": _f(in_strength[index]),
            "out_strength": _f(out_strength[index]),
            "betweenness": _f(betweenness[index]),
            "betweenness_raw": _f(raw_betweenness[index]),
            "eigenvector_centrality": _f(centrality[index]),
            "clustering": _f(clustering[index]),
            "self_loop": bool(present[index, index]),
        }
        for index in range(n)
    ]
    return {
        "nodes": nodes,
        "labels": names,
        "node_count": n,
        "edge_count": int(off_diagonal.sum()),
        "self_loop_labels": self_loops,
        "density": _f(off_diagonal.sum() / (n * (n - 1))) if n > 1 else None,
        "symmetric": symmetric,
        "mean_degree": _f(degree.mean()),
        "average_clustering": _f(clustering.mean()),
        "transitivity": _f(transitivity) if transitivity is not None else None,
        "weak_components": [[names[i] for i in group] for group in weak],
        "weak_component_count": len(weak),
        "strong_components": [[names[i] for i in group] for group in strong],
        "strong_component_count": len(strong),
        "largest_weak_component_size": len(weak[0]) if weak else 0,
        "largest_strong_component_size": len(strong[0]) if strong else 0,
        "eigenvector_centrality_status": reason,
        "eigenvector_centrality_converged": bool(converged),
        "eigenvector_centrality_iterations": iterations,
        "spectral_radius_estimate": _f(spectral),
        "betweenness_normalization": "ordered pairs excluding the node; halved for a symmetric matrix",
        "clustering_basis": "binarized undirected projection of the influence matrix",
        "eigenvector_basis": "shifted power iteration on |adjacency| transposed, so importance flows from in-neighbours; signs are discarded",
        "scope": (
            "Graph statistics of the supplied matrix. They describe that matrix, not an "
            "observed system: they assume the edge set is complete and correct, discard "
            "edge signs for centrality and clustering, use unweighted shortest paths for "
            "betweenness, and carry no sampling uncertainty. A central node here is not "
            "shown to be an effective intervention point."
        ),
    }


def percolation_threshold(adjacency, tol=0.0):
    """Mean-field cascade threshold ``<k>/(<k^2> - <k>)`` and the leading-eigenvalue criterion.

    Both are annealed-network estimates of the transmissibility at which a giant
    connected/infected component appears (Molloy and Reed, Random Struct. Algorithms
    6:161, 1995; Newman, Phys. Rev. E 66:016128, 2002). When the denominator is
    nonpositive or the spectral radius is zero, the corresponding threshold is reported
    as ``None`` with a reason rather than as a number.
    """
    matrix = _square(adjacency, "adjacency", MAX_NODES)
    if tol < 0:
        raise Invalid("tol must not be negative")
    undirected = _undirected(matrix, float(tol))
    degree = undirected.sum(axis=1)
    mean_degree = float(degree.mean())
    mean_square = float((degree * degree).mean())
    denominator = mean_square - mean_degree

    if mean_degree <= 0:
        mean_field, mean_field_reason = None, "graph has no edges"
    elif denominator <= 0:
        mean_field, mean_field_reason = None, "second degree moment does not exceed the first; no giant component under this criterion"
    else:
        mean_field, mean_field_reason = _f(mean_degree / denominator), "defined"

    magnitudes = np.abs(matrix).copy()
    np.fill_diagonal(magnitudes, 0.0)
    spectral = float(np.max(np.abs(np.linalg.eigvals(magnitudes)))) if magnitudes.size else 0.0
    if spectral <= 1e-12:
        spectral_threshold, spectral_reason = None, "spectral radius is zero; no spreading pathway"
    else:
        spectral_threshold, spectral_reason = _f(1.0 / spectral), "defined"

    return {
        "mean_field_threshold": mean_field,
        "mean_field_status": mean_field_reason,
        "spectral_threshold": spectral_threshold,
        "spectral_status": spectral_reason,
        "spectral_radius": _f(spectral),
        "mean_degree": _f(mean_degree),
        "mean_square_degree": _f(mean_square),
        "heterogeneity_ratio": _f(mean_square / mean_degree) if mean_degree > 0 else None,
        "molloy_reed_ratio": _f(mean_square / mean_degree) if mean_degree > 0 else None,
        "supercritical_by_molloy_reed": bool(mean_degree > 0 and mean_square / mean_degree > 2.0),
        "node_count": int(matrix.shape[0]),
        "definitions": {
            "mean_field_threshold": "<k>/(<k^2>-<k>) on the undirected projection",
            "spectral_threshold": "1/lambda_max of |adjacency| with the diagonal removed",
        },
        "scope": (
            "Annealed mean-field estimates. They assume a locally tree-like graph with no "
            "degree correlations, an independent per-edge transmission probability, and a "
            "large network; clustering, community structure, edge signs, edge weights, "
            "finite size and any actual contagion dynamics are ignored. On a small or "
            "clustered graph the numbers can be wrong by a wide margin, and neither "
            "establishes that a cascade would occur in a real system."
        ),
    }


# --------------------------------------------------------------------------------------
# 6. behavioral descriptors for quality-diversity search
# --------------------------------------------------------------------------------------


def _uniform_grid(times, states):
    steps = np.diff(times)
    span = float(steps.max() - steps.min())
    if span <= 1e-9 * float(steps.mean()):
        return times, states, False
    grid = np.linspace(float(times[0]), float(times[-1]), times.shape[0])
    resampled = np.stack(
        [np.interp(grid, times, states[:, column]) for column in range(states.shape[1])],
        axis=1,
    )
    return grid, resampled, True


def _fft_period(signal, dt):
    """Dominant period from a Hann-windowed periodogram, with parabolic peak refinement."""
    count = signal.shape[0]
    index = np.arange(count, dtype=float)
    slope, intercept = np.polyfit(index, signal, 1)
    detrended = (signal - (slope * index + intercept)) * np.hanning(count)
    spectrum = np.abs(np.fft.rfft(detrended)) ** 2
    spectrum[0] = 0.0
    total = float(spectrum.sum())
    if total <= 0:
        return None, 0.0
    peak = int(np.argmax(spectrum))
    if peak == 0:
        return None, 0.0
    offset = 0.0
    if 0 < peak < spectrum.shape[0] - 1:
        left, middle, right = (
            math.log(max(spectrum[peak - 1], 1e-300)),
            math.log(max(spectrum[peak], 1e-300)),
            math.log(max(spectrum[peak + 1], 1e-300)),
        )
        curvature = left - 2.0 * middle + right
        if curvature < 0:
            offset = 0.5 * (left - right) / curvature
            offset = float(min(max(offset, -0.5), 0.5))
    frequency = (peak + offset) / (count * dt)
    return (1.0 / frequency if frequency > 0 else None), float(spectrum[peak] / total)


def _autocorrelation_period(signal, dt):
    centered = signal - signal.mean()
    energy = float(np.dot(centered, centered))
    if energy <= 0:
        return None
    correlation = np.correlate(centered, centered, mode="full")[centered.shape[0] - 1 :]
    correlation = correlation / energy
    negative = np.nonzero(correlation < 0)[0]
    if negative.size == 0:
        return None
    start = int(negative[0])
    if start >= correlation.shape[0] - 1:
        return None
    peak = int(np.argmax(correlation[start:])) + start
    return float(peak * dt) if peak > 0 else None


def behavior_descriptor(times, states, settle_tolerance=0.02, tail_fraction=0.1):
    """Measure a bounded, deterministic behavioral signature of one trajectory.

    Returns a small feature set describing how the trajectory behaves rather than how the
    model was declared: a stability class, whether it oscillates and at what period,
    monotone versus overshooting approach, settling time, the share of deviation energy
    in the transient, the participation-ratio effective dimensionality of the state
    covariance, and a coefficient of variation. ``features`` holds the continuous
    components, each already scaled into [0, 1], in a fixed order; :func:`descriptor_key`
    bins them into a discrete niche key. The same input always produces the same output.

    A nonfinite state is treated as divergence: the finite prefix is measured and
    ``divergent`` is reported.
    """
    times = _array(times, "times", ndim=1)
    raw = _array(states, "states", finite=False)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    if raw.ndim != 2:
        raise Invalid("states must be a (T, N) array")
    if raw.shape[0] != times.shape[0]:
        raise Invalid("times and states must have the same number of rows")
    if not 8 <= times.shape[0] <= MAX_TRAJECTORY_POINTS:
        raise Invalid(f"Supply between 8 and {MAX_TRAJECTORY_POINTS} trajectory samples")
    if raw.shape[1] > MAX_STATE_DIM:
        raise Invalid(f"State dimension must not exceed {MAX_STATE_DIM}")
    if not bool(np.all(np.diff(times) > 0)):
        raise Invalid("times must be strictly increasing")
    settle_tolerance = _positive(settle_tolerance, "settle_tolerance", 0.0, 1.0)
    tail_fraction = _positive(tail_fraction, "tail_fraction", 0.0, 0.5)

    finite_rows = np.all(np.isfinite(raw), axis=1)
    divergent = not bool(np.all(finite_rows))
    if divergent:
        cut = int(np.argmin(finite_rows))
        if cut < 8:
            raise Invalid("Fewer than eight finite samples precede the first nonfinite state")
        times, raw = times[:cut], raw[:cut]

    times, values, resampled = _uniform_grid(times, raw)
    count, width = values.shape
    duration = float(times[-1] - times[0])
    dt = duration / (count - 1)

    ranges = values.max(axis=0) - values.min(axis=0)
    scales = np.maximum(ranges, 1e-12 * np.maximum(1.0, np.abs(values).max(axis=0)))
    overall_scale = float(max(scales.max(), 1e-12))
    tail = max(1, int(round(tail_fraction * count)))
    final = values[-tail:].mean(axis=0)

    magnitude = np.linalg.norm(values, axis=1)
    third = max(2, count // 3)
    early_magnitude = float(magnitude[:third].max())
    late_magnitude = float(magnitude[-third:].max())
    early_rate = float(np.abs(np.diff(magnitude[:third])).mean())
    late_rate = float(np.abs(np.diff(magnitude[-third:])).mean())
    accelerating = bool(
        late_magnitude > early_magnitude and late_rate > 2.0 * max(early_rate, 1e-9 * overall_scale)
    )

    tolerance = settle_tolerance * scales
    within = np.abs(values - final) <= tolerance
    all_within = np.all(within, axis=1)
    outside = np.nonzero(~all_within)[0]
    settle_index = int(outside[-1]) + 1 if outside.size else 0
    settled = settle_index < count
    settling_time = float(times[min(settle_index, count - 1)] - times[0]) if settled else duration
    settling_fraction = settling_time / duration if duration > 0 else 0.0

    overshoot = 0.0
    for column in range(width):
        travel = float(final[column] - values[0, column])
        if abs(travel) > tolerance[column]:
            excess = float(np.max((values[:, column] - final[column]) * np.sign(travel)))
            overshoot = max(overshoot, max(0.0, excess) / abs(travel))
        else:
            excursion = float(np.max(np.abs(values[:, column] - final[column])))
            overshoot = max(overshoot, excursion / float(scales[column]))

    differences = np.diff(values, axis=0)
    significant = np.abs(differences) > 1e-9 * scales
    monotone = True
    for column in range(width):
        signs = np.sign(differences[significant[:, column], column])
        if signs.size and not bool(np.all(signs == signs[0])):
            monotone = False
            break

    dominant = int(np.argmax(values.var(axis=0))) if width > 1 else 0
    signal = values[:, dominant]
    centered = signal - final[dominant]
    threshold = 1e-6 * float(scales[dominant])
    signs = np.sign(np.where(np.abs(centered) > threshold, centered, 0.0))
    nonzero = signs[signs != 0]
    crossings = int(np.count_nonzero(np.diff(nonzero) != 0)) if nonzero.size > 1 else 0
    oscillatory = crossings >= 2
    period, peak_share = _fft_period(signal, dt)
    autocorrelation_period = _autocorrelation_period(signal, dt)
    cycles = duration / period if period and period > 0 else 0.0
    period_fraction = min(1.0, period / duration) if (oscillatory and period and duration > 0) else 0.0

    early_amplitude = float(signal[:third].max() - signal[:third].min())
    late_amplitude = float(signal[-third:].max() - signal[-third:].min())
    amplitude_ratio = late_amplitude / early_amplitude if early_amplitude > 0 else 0.0

    normalized = (values - final) / scales
    energy = np.sum(normalized * normalized, axis=1)
    midpoint = count // 2
    first_energy = float(np.trapezoid(energy[: midpoint + 1], times[: midpoint + 1]))
    second_energy = float(np.trapezoid(energy[midpoint:], times[midpoint:]))
    energy_total = first_energy + second_energy
    transient_ratio = first_energy / energy_total if energy_total > 0 else 0.5

    centered_states = values - values.mean(axis=0)
    covariance = (centered_states.T @ centered_states) / max(1, count - 1)
    spectrum = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    squares = float(np.sum(spectrum * spectrum))
    participation = float(np.sum(spectrum) ** 2 / squares) if squares > 0 else 1.0
    participation = float(min(max(participation, 1.0), float(width)))
    effective_dimension = (participation - 1.0) / (width - 1.0) if width > 1 else 0.0

    means = values.mean(axis=0)
    deviations = values.std(axis=0, ddof=1)
    defined = np.abs(means) > 1e-9 * scales
    if bool(np.any(defined)):
        coefficient = float(np.max(deviations[defined] / np.abs(means[defined])))
    else:
        coefficient = float(np.max(deviations / scales))

    if divergent:
        stability_class = "divergent"
    elif settled:
        stability_class = "converged"
    elif oscillatory:
        if amplitude_ratio > 2.0:
            stability_class = "divergent"
        elif amplitude_ratio >= 0.5:
            stability_class = "sustained_oscillation"
        else:
            stability_class = "damped_oscillation"
    elif accelerating:
        stability_class = "divergent"
    else:
        stability_class = "drifting"

    features = {
        "period_fraction": float(period_fraction),
        "overshoot": float(min(overshoot, 2.0) / 2.0),
        "settling_fraction": float(min(max(settling_fraction, 0.0), 1.0)),
        "transient_energy_ratio": float(min(max(transient_ratio, 0.0), 1.0)),
        "effective_dimension": float(min(max(effective_dimension, 0.0), 1.0)),
        "variability": float(min(coefficient, 4.0) / 4.0),
        "monotonicity": 1.0 if monotone else 0.0,
    }
    return {
        "version": VERSION,
        "stability_class": stability_class,
        "oscillatory": bool(oscillatory),
        "dominant_period": _f(period) if (oscillatory and period) else None,
        "autocorrelation_period": _f(autocorrelation_period)
        if (oscillatory and autocorrelation_period)
        else None,
        "spectral_peak_share": _f(peak_share),
        "zero_crossings": crossings,
        "cycles_observed": _f(cycles),
        "amplitude_ratio": _f(amplitude_ratio),
        "monotone": bool(monotone),
        "settled": bool(settled),
        "settling_time": _f(settling_time),
        "settling_fraction": _f(settling_fraction),
        "overshoot_fraction": _f(overshoot),
        "transient_energy_ratio": _f(transient_ratio),
        "effective_dimension": _f(participation),
        "effective_dimension_normalized": _f(features["effective_dimension"]),
        "coefficient_of_variation": _f(coefficient),
        "dominant_component_index": dominant,
        "sample_count": int(count),
        "state_dimension": int(width),
        "duration": _f(duration),
        "resampled_to_uniform_grid": bool(resampled),
        "truncated_at_nonfinite_state": not bool(np.all(finite_rows)),
        "growth_accelerating": bool(accelerating),
        "feature_names": list(features),
        "features": features,
        "feature_vector": [features[name] for name in features],
        "deterministic": True,
        "scope": (
            "Measured descriptors of one recorded trajectory over its recorded window. "
            "They characterize that run, not the system: a different initial condition, "
            "forcing, horizon or sampling rate changes them, 'converged' means only that "
            "the record stopped moving within the stated tolerance, an oscillation period "
            "shorter than a few samples or longer than the record is not resolved, and "
            "nothing here establishes stability, correctness or empirical validity."
        ),
    }


_AXES = {
    "coarse": (
        ("stability_class", "categorical", BEHAVIOR_CLASSES),
        ("oscillatory", "boolean", None),
        ("overshoot", "numeric", (0.02, 0.2)),
        ("settling_fraction", "numeric", (0.25, 0.75)),
        ("effective_dimension", "numeric", (0.2, 0.5)),
        ("variability", "numeric", (0.1, 0.4)),
    ),
    "fine": (
        ("stability_class", "categorical", BEHAVIOR_CLASSES),
        ("oscillatory", "boolean", None),
        ("period_fraction", "numeric", (0.02, 0.05, 0.15, 0.4)),
        ("overshoot", "numeric", (0.02, 0.1, 0.3, 0.75)),
        ("settling_fraction", "numeric", (0.1, 0.3, 0.6, 0.95)),
        ("transient_energy_ratio", "numeric", (0.3, 0.5, 0.7, 0.9)),
        ("effective_dimension", "numeric", (0.15, 0.35, 0.6)),
        ("variability", "numeric", (0.05, 0.15, 0.4)),
        ("monotonicity", "boolean", None),
    ),
}


def descriptor_key(descriptor, resolution="coarse"):
    """Bin a behavior descriptor into a stable discrete niche key.

    The bin edges are fixed constants, never derived from the data, so the same
    descriptor always maps to the same cell and cells stay comparable across runs and
    archives. The key carries the descriptor version and the resolution; changing either
    changes the key, which is the intended signal that an archive built under the old
    binning is no longer comparable.
    """
    if not isinstance(descriptor, dict):
        raise Invalid("descriptor must be the dict returned by behavior_descriptor")
    if resolution not in _AXES:
        raise Invalid("resolution must be 'coarse' or 'fine'")
    features = descriptor.get("features")
    if not isinstance(features, dict):
        raise Invalid("descriptor must carry a 'features' dict")

    axes, coordinates, cells = [], [], 1
    for name, kind, edges in _AXES[resolution]:
        if kind == "categorical":
            value = descriptor.get(name)
            if value not in edges:
                raise Invalid(f"descriptor field {name} is not a known category")
            index = edges.index(value)
            size = len(edges)
            axes.append({"name": name, "kind": kind, "categories": list(edges), "value": value, "bin": index})
        else:
            if kind == "boolean":
                raw = descriptor.get(name, features.get(name))
                if isinstance(raw, bool):
                    index = int(raw)
                elif isinstance(raw, (int, float)):
                    index = int(float(raw) >= 0.5)
                else:
                    raise Invalid(f"descriptor field {name} must be boolean or numeric")
                size = 2
                axes.append({"name": name, "kind": kind, "value": bool(index), "bin": index})
            else:
                if name not in features:
                    raise Invalid(f"descriptor features are missing {name}")
                value = features[name]
                if value is None or not math.isfinite(float(value)):
                    raise Invalid(f"descriptor feature {name} must be a finite number")
                index = int(np.searchsorted(np.asarray(edges, dtype=float), float(value), side="right"))
                size = len(edges) + 1
                axes.append(
                    {
                        "name": name,
                        "kind": kind,
                        "edges": list(edges),
                        "value": float(value),
                        "bin": index,
                    }
                )
        coordinates.append(index)
        cells *= size

    key = f"{VERSION}/{resolution}/" + "-".join(str(index) for index in coordinates)
    return {
        "key": key,
        "version": VERSION,
        "resolution": resolution,
        "coordinates": coordinates,
        "axes": axes,
        "axis_names": [axis["name"] for axis in axes],
        "cell_count": cells,
        "stability_class": descriptor.get("stability_class"),
        "binning": "fixed constant edges; never derived from the data",
        "scope": (
            "A discrete niche label for the supplied behavior descriptor. It records "
            "measured dynamics of one trajectory, so two runs sharing a key are similar "
            "only in these coarse features under this binning, and a key says nothing "
            "about model quality, correctness or empirical validity. Keys from different "
            "versions or resolutions are not comparable."
        ),
    }
