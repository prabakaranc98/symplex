"""Sparse symbolic recovery of governing equations from trajectory data.

Implements SINDy: Brunton, Proctor and Kutz, "Discovering governing equations from data
by sparse identification of nonlinear dynamical systems", PNAS 113(15):3932-3937, 2016
(doi:10.1073/pnas.1517384113). The integral/test-function variant follows Reinbold,
Gurevich and Grigoriev, "Using noisy or incomplete data to discover models of spatio-
temporal dynamics", Phys. Rev. E 101:010203, 2020, and Messenger and Bortz, "Weak SINDy",
J. Comput. Phys. 443:110525, 2021.

A recovered equation is a sparse least-squares fit to the supplied samples inside the
chosen candidate library. It is not a derivation and it does not establish the real
system's mechanism. Which terms survive depends on the library, the sparsity threshold,
the derivative estimator and the noise level; `stability_selection` and `pareto_sweep`
exist to make that dependence visible, not to remove it. Naive differencing of noisy
samples is the standard way this method fails, so the derivative estimator is an explicit,
reported choice rather than a hidden default.

Pure numeric surface: arrays in, dicts out. Nothing here imports the modeling runtime.
"""

import itertools
import math

import numpy as np
from scipy.signal import savgol_filter

from symplex.core.contracts import Invalid

MAX_SAMPLES = 200_000
MAX_STATES = 12
MAX_DEGREE = 5
MAX_LIBRARY_TERMS = 200
MAX_ITERATIONS = 100
MAX_BOOTSTRAP = 200
MAX_THRESHOLDS = 32
MAX_TV_SAMPLES = 1_000  # Total-variation differentiation solves a dense n-by-n system.
MAX_TEST_FUNCTIONS = 500

DERIVATIVE_METHODS = ("finite_difference", "savitzky_golay", "total_variation")

SCOPE = (
    "Sparse regression fit to the supplied samples within the declared candidate library. "
    "Not a derivation, not a validated mechanism, and not evidence about states, regimes or "
    "noise levels outside the data used."
)


def _matrix(values, label, max_rows=MAX_SAMPLES, max_cols=MAX_STATES):
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.size == 0:
        raise Invalid(f"{label} must be a nonempty 1-D or 2-D numeric array")
    if not np.all(np.isfinite(array)):
        raise Invalid(f"{label} contains nonfinite values")
    if array.shape[0] > max_rows:
        raise Invalid(f"{label} exceeds the {max_rows}-sample limit")
    if array.shape[1] > max_cols:
        raise Invalid(f"{label} exceeds the {max_cols}-column limit")
    return array


def _positive(value, label, low=0.0, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Invalid(f"{label} must be a number")
    value = float(value)
    if not math.isfinite(value) or value <= low or (high is not None and value > high):
        raise Invalid(f"{label} is outside its permitted range")
    return value


def _integer(value, label, low, high):
    if isinstance(value, bool) or not isinstance(value, int):
        raise Invalid(f"{label} must be an integer")
    if not low <= value <= high:
        raise Invalid(f"{label} must lie in [{low}, {high}]")
    return value


def default_state_names(n_states):
    """Readable default symbols; explicit names are always preferable."""
    if n_states <= 3:
        return ["x", "y", "z"][:n_states]
    return [f"x{i}" for i in range(n_states)]


def _check_names(state_names, n_states):
    if state_names is None:
        return default_state_names(n_states)
    names = list(state_names)
    if len(names) != n_states or any(
        not isinstance(n, str) or not n.strip() or len(n) > 32 for n in names
    ):
        raise Invalid("state_names must be one short nonempty string per state")
    if len(set(names)) != len(names):
        raise Invalid("state_names must be unique")
    return [n.strip() for n in names]


def _term_name(combo, names):
    counts = {}
    for index in combo:
        counts[index] = counts.get(index, 0) + 1
    parts = []
    for index in sorted(counts):
        power = counts[index]
        parts.append(names[index] if power == 1 else f"{names[index]}^{power}")
    return " ".join(parts)


def polynomial_library(
    X,
    degree=2,
    include_trig=False,
    include_interactions=True,
    state_names=None,
    custom_terms=None,
    include_constant=True,
):
    """Build a candidate-term matrix with human-readable names.

    `custom_terms` is the extension hook: a sequence of ``(name, function)`` pairs or
    ``{"name": ..., "function": ...}`` mappings, where ``function(X) -> (n,)`` is a
    caller-supplied column. The library is the hypothesis space; a mechanism absent from
    it cannot be recovered, and that absence is not evidence against the mechanism.
    """
    X = _matrix(X, "X")
    degree = _integer(degree, "degree", 0, MAX_DEGREE)
    n_samples, n_states = X.shape
    names = _check_names(state_names, n_states)
    if not isinstance(include_trig, bool) or not isinstance(include_interactions, bool):
        raise Invalid("include_trig and include_interactions must be booleans")

    columns, term_names = [], []
    if include_constant:
        columns.append(np.ones(n_samples))
        term_names.append("1")
    for order in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(n_states), order):
            if not include_interactions and len(set(combo)) > 1:
                continue
            column = np.ones(n_samples)
            for index in combo:
                column = column * X[:, index]
            columns.append(column)
            term_names.append(_term_name(combo, names))
    if include_trig:
        for index in range(n_states):
            columns.append(np.sin(X[:, index]))
            term_names.append(f"sin({names[index]})")
            columns.append(np.cos(X[:, index]))
            term_names.append(f"cos({names[index]})")
    for entry in custom_terms or []:
        if isinstance(entry, dict):
            name, function = entry.get("name"), entry.get("function")
        else:
            try:
                name, function = entry
            except (TypeError, ValueError):
                raise Invalid("custom_terms entries must be (name, function)") from None
        if not isinstance(name, str) or not name.strip() or len(name) > 64:
            raise Invalid("Each custom term needs a short nonempty name")
        if not callable(function):
            raise Invalid("Each custom term needs a callable function(X) -> column")
        column = np.asarray(function(X), dtype=float).reshape(-1)
        if column.shape != (n_samples,) or not np.all(np.isfinite(column)):
            raise Invalid(f"Custom term {name!r} must return {n_samples} finite values")
        columns.append(column)
        term_names.append(name.strip())

    if len(term_names) != len(set(term_names)):
        raise Invalid("Candidate term names must be unique")
    if not columns:
        raise Invalid("The candidate library is empty")
    if len(columns) > MAX_LIBRARY_TERMS:
        raise Invalid(f"Candidate library exceeds {MAX_LIBRARY_TERMS} terms")
    features = np.column_stack(columns)
    if not np.all(np.isfinite(features)):
        raise Invalid("Candidate library contains nonfinite entries")
    norms = np.linalg.norm(features, axis=0)
    return {
        "features": features,
        "names": term_names,
        "state_names": names,
        "n_terms": len(term_names),
        "n_samples": int(n_samples),
        "degree": degree,
        "include_trig": bool(include_trig),
        "include_interactions": bool(include_interactions),
        "include_constant": bool(include_constant),
        "column_norms": [float(v) for v in norms],
        "degenerate_columns": [
            term_names[i] for i in range(len(term_names)) if norms[i] <= 1e-12
        ],
        "scope": (
            "A candidate basis chosen by the caller. Terms outside it cannot be recovered; "
            "their absence from the result is not evidence about the system."
        ),
    }


def _ridge_solve(A, y, alpha):
    """Least squares with Tikhonov regularization, solved as an augmented system."""
    if A.shape[1] == 0:
        return np.zeros(0)
    if alpha > 0:
        A = np.vstack([A, math.sqrt(alpha) * np.eye(A.shape[1])])
        y = np.concatenate([y, np.zeros(A.shape[1])])
    solution, *_ = np.linalg.lstsq(A, y, rcond=None)
    return solution


def _stlsq_column(Theta, y, threshold, alpha, max_iter):
    n_terms = Theta.shape[1]
    coefficients = _ridge_solve(Theta, y, alpha)
    active = np.abs(coefficients) >= threshold
    converged, iterations = False, 0
    for iterations in range(1, max_iter + 1):
        updated = np.zeros(n_terms)
        if active.any():
            updated[active] = _ridge_solve(Theta[:, active], y, alpha)
        next_active = np.abs(updated) >= threshold
        coefficients = updated
        if np.array_equal(next_active, active):
            converged = True
            break
        active = next_active
    if not converged:
        coefficients = np.where(np.abs(coefficients) >= threshold, coefficients, 0.0)
        active = coefficients != 0.0
    return coefficients, active, iterations, converged


def stlsq(Theta, dXdt, threshold=0.05, alpha=0.0, max_iter=20):
    """Sequentially thresholded least squares (the SINDy sparsity algorithm).

    Ridge-regularized least squares, then repeated elimination of coefficients below
    `threshold` and refit on the surviving support until the support stops changing.
    The threshold is compared against raw coefficients, so it carries the units of each
    library column: rescaling a term rescales its inclusion decision. Non-convergence
    within `max_iter` is reported, not smoothed over.
    """
    Theta = _matrix(Theta, "Theta", max_cols=MAX_LIBRARY_TERMS)
    targets = _matrix(dXdt, "dXdt")
    if targets.shape[0] != Theta.shape[0]:
        raise Invalid("Theta and dXdt must have the same number of rows")
    threshold = _positive(threshold, "threshold", low=-1e-12)
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise Invalid("alpha must be a number")
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha < 0:
        raise Invalid("alpha must be a finite nonnegative number")
    max_iter = _integer(max_iter, "max_iter", 1, MAX_ITERATIONS)

    coefficients, actives, iterations, flags = [], [], [], []
    for column in range(targets.shape[1]):
        w, active, iters, ok = _stlsq_column(
            Theta, targets[:, column], threshold, alpha, max_iter
        )
        coefficients.append(w)
        actives.append(active)
        iterations.append(int(iters))
        flags.append(bool(ok))
    matrix = np.column_stack(coefficients)
    return {
        "coefficients": [[float(v) for v in w] for w in coefficients],
        "coefficient_matrix": matrix,
        "active": [[bool(v) for v in a] for a in actives],
        "n_active": [int(a.sum()) for a in actives],
        "iterations": iterations,
        "converged": bool(all(flags)),
        "converged_per_target": flags,
        "threshold": threshold,
        "alpha": alpha,
        "max_iter": max_iter,
        "scope": (
            "A sparsity decision, not a significance test. The surviving support depends on "
            "the threshold, the ridge weight and the scaling of each library column."
        ),
    }


def render_equation(coefficients, term_names, lhs="dx/dt", precision=3):
    """Render one sparse coefficient row as a readable equation string."""
    coefficients = np.asarray(coefficients, dtype=float).reshape(-1)
    if len(term_names) != coefficients.size:
        raise Invalid("render_equation needs one name per coefficient")
    precision = _integer(precision, "precision", 1, 12)
    if not isinstance(lhs, str) or not lhs.strip():
        raise Invalid("lhs must be a nonempty string")
    pieces = []
    for value, name in zip(coefficients, term_names):
        if value == 0.0:
            continue
        magnitude = f"{abs(value):.{precision}f}"
        body = magnitude if name == "1" else f"{magnitude} {name}"
        if not pieces:
            pieces.append(("-" + body) if value < 0 else body)
        else:
            pieces.append((" - " if value < 0 else " + ") + body)
    equation = f"{lhs} = " + ("".join(pieces) if pieces else "0")
    return {
        "equation": equation,
        "n_active_terms": int(np.count_nonzero(coefficients)),
        "scope": (
            "A rendering of fitted numbers. Printed precision is a display choice and says "
            "nothing about the uncertainty of any coefficient."
        ),
    }


def _uniform_step(t, dt, n_samples, require_uniform):
    if t is not None and dt is not None:
        raise Invalid("Supply t or dt, not both")
    if t is None:
        if dt is None:
            raise Invalid("Supply sample times t or a time step dt")
        return _positive(dt, "dt"), None
    times = np.asarray(t, dtype=float).reshape(-1)
    if times.shape[0] != n_samples or not np.all(np.isfinite(times)):
        raise Invalid("t must hold one finite time per sample")
    steps = np.diff(times)
    if steps.size == 0 or np.any(steps <= 0):
        raise Invalid("t must be strictly increasing")
    step = float(steps.mean())
    if require_uniform and not np.allclose(steps, steps[0], rtol=1e-6, atol=1e-12):
        raise Invalid("This derivative estimator requires a uniform time grid")
    return step, times


def _tv_derivative(signal, dt, alpha, iterations, epsilon=1e-8):
    """Chartrand-style total-variation regularized differentiation (lagged diffusivity).

    Chartrand, "Numerical differentiation of noisy, nonsmooth data", ISRN Applied
    Mathematics, 2011. Dense linear algebra, hence the sample cap.
    """
    n = signal.size
    integrate = dt * (np.tril(np.ones((n, n))) - 0.5 * np.eye(n))
    difference = (np.eye(n, k=1) - np.eye(n))[:-1] / dt
    target = signal - signal[0]
    u = np.gradient(signal, dt)
    gram = integrate.T @ integrate
    rhs = integrate.T @ target
    converged = False
    for _ in range(iterations):
        weights = 1.0 / np.sqrt((difference @ u) ** 2 + epsilon)
        laplacian = dt * (difference.T * weights) @ difference
        gradient = gram @ u - rhs + alpha * (laplacian @ u)
        hessian = gram + alpha * laplacian + 1e-10 * np.eye(n)
        try:
            step = np.linalg.solve(hessian, -gradient)
        except np.linalg.LinAlgError:
            break
        u = u + step
        if np.linalg.norm(step) <= 1e-8 * (1.0 + np.linalg.norm(u)):
            converged = True
            break
    return u, converged


def estimate_derivatives(
    X,
    t=None,
    dt=None,
    method="finite_difference",
    window=None,
    polyorder=3,
    tv_alpha=0.02,
    tv_iterations=30,
):
    """Estimate dX/dt from sampled states, and report which estimator was used.

    `finite_difference` is second-order central differencing and amplifies noise by
    roughly 1/dt; it is only appropriate on clean or heavily oversampled data.
    `savitzky_golay` fits a local polynomial and returns both the smoothed states and
    their analytic derivative, which is what should feed the library on noisy data.
    `total_variation` is for piecewise-smooth or sharply varying signals and is capped
    because it solves a dense n-by-n system per state.
    """
    X = _matrix(X, "X")
    if method not in DERIVATIVE_METHODS:
        raise Invalid("method must be one of: " + ", ".join(DERIVATIVE_METHODS))
    n_samples, n_states = X.shape
    if n_samples < 5:
        raise Invalid("Derivative estimation needs at least 5 samples")
    step, times = _uniform_step(t, dt, n_samples, method != "finite_difference")
    notes = []

    if method == "finite_difference":
        axis_coordinate = times if times is not None else step
        derivatives = np.gradient(X, axis_coordinate, axis=0)
        smoothed = X.copy()
        edge_trim, converged = 1, True
        notes.append(
            "Endpoint derivatives are one-sided and lower order; edge samples are trimmed by default."
        )
    elif method == "savitzky_golay":
        polyorder = _integer(polyorder, "polyorder", 1, 8)
        if window is None:
            window = max(polyorder + 2, min(n_samples // 10 * 2 + 1, 51))
        window = _integer(window, "window", polyorder + 2, n_samples)
        if window % 2 == 0:
            window += 1
        if window > n_samples:
            raise Invalid("Savitzky-Golay window exceeds the number of samples")
        smoothed = savgol_filter(X, window, polyorder, axis=0)
        derivatives = savgol_filter(X, window, polyorder, deriv=1, delta=step, axis=0)
        edge_trim, converged = window // 2, True
        notes.append(
            "Smoothing biases sharp features; the window and polynomial order are a bias/variance choice."
        )
    else:
        if n_samples > MAX_TV_SAMPLES:
            raise Invalid(
                f"total_variation differentiation is capped at {MAX_TV_SAMPLES} samples "
                "because it solves a dense n-by-n system per state"
            )
        tv_alpha = _positive(tv_alpha, "tv_alpha")
        tv_iterations = _integer(tv_iterations, "tv_iterations", 1, MAX_ITERATIONS)
        columns, flags = [], []
        for index in range(n_states):
            derivative, ok = _tv_derivative(X[:, index], step, tv_alpha, tv_iterations)
            columns.append(derivative)
            flags.append(ok)
        derivatives = np.column_stack(columns)
        smoothed = X.copy()  # TV differentiation returns u; states are not re-integrated.
        edge_trim, converged = 1, bool(all(flags))
        notes.append(
            "The regularization weight tv_alpha trades noise suppression against flattening of true variation."
        )
        if not converged:
            notes.append("Lagged-diffusivity iteration did not meet its step tolerance.")

    return {
        "derivatives": derivatives,
        "smoothed_states": smoothed,
        "method": method,
        "dt": float(step),
        "window": int(window) if method == "savitzky_golay" else None,
        "polyorder": int(polyorder) if method == "savitzky_golay" else None,
        "suggested_edge_trim": int(edge_trim),
        "converged": bool(converged),
        "notes": notes,
        "scope": (
            "An estimate of a derivative, not a measurement of one. Every downstream "
            "coefficient inherits this estimator's bias and noise amplification."
        ),
    }


def _prepare(
    X,
    t=None,
    dt=None,
    dXdt=None,
    degree=2,
    include_trig=False,
    include_interactions=True,
    state_names=None,
    custom_terms=None,
    derivative_method="finite_difference",
    window=None,
    polyorder=3,
    edge_trim=None,
):
    X = _matrix(X, "X")
    names = _check_names(state_names, X.shape[1])
    if dXdt is not None:
        derivatives = _matrix(dXdt, "dXdt")
        if derivatives.shape[0] != X.shape[0]:
            raise Invalid("Supplied dXdt must have one row per sample of X")
        states, derivative_info = X, {
            "method": "supplied",
            "dt": None,
            "converged": True,
            "notes": ["Derivatives were supplied by the caller and are not checked here."],
            "suggested_edge_trim": 0,
            "window": None,
            "polyorder": None,
        }
    else:
        estimate = estimate_derivatives(
            X, t=t, dt=dt, method=derivative_method, window=window, polyorder=polyorder
        )
        states = estimate["smoothed_states"]
        derivatives = estimate["derivatives"]
        derivative_info = {k: v for k, v in estimate.items() if k not in ("derivatives", "smoothed_states", "scope")}
    trim = derivative_info["suggested_edge_trim"] if edge_trim is None else edge_trim
    trim = _integer(int(trim), "edge_trim", 0, max(0, X.shape[0] // 3))
    if trim:
        states = states[trim:-trim]
        derivatives = derivatives[trim:-trim]
    if states.shape[0] < 10:
        raise Invalid("Fewer than 10 usable samples remain after edge trimming")
    library = polynomial_library(
        states,
        degree=degree,
        include_trig=include_trig,
        include_interactions=include_interactions,
        state_names=names,
        custom_terms=custom_terms,
    )
    return states, derivatives, library, derivative_info, trim


def _target_labels(target_names, n_targets, state_names):
    """Left-hand-side labels: one per fitted equation, not necessarily one per state."""
    if target_names is None:
        if n_targets != len(state_names):
            raise Invalid(
                "target_names is required when dXdt has a different number of columns than X"
            )
        return list(state_names)
    labels = list(target_names)
    if len(labels) != n_targets or any(
        not isinstance(n, str) or not n.strip() or len(n) > 32 for n in labels
    ):
        raise Invalid("target_names must be one short nonempty string per fitted equation")
    if len(set(labels)) != len(labels):
        raise Invalid("target_names must be unique")
    return [n.strip() for n in labels]


def _scores(Theta, y, coefficients):
    prediction = Theta @ coefficients
    residual = y - prediction
    ss_res = float(residual @ residual)
    centred = y - y.mean()
    ss_tot = float(centred @ centred)
    r_squared = None if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return {
        "rmse": float(math.sqrt(ss_res / y.size)),
        "r_squared": r_squared,
        "max_absolute_residual": float(np.max(np.abs(residual))) if residual.size else 0.0,
    }


def _condition_number(Theta):
    gram = Theta.T @ Theta
    try:
        eigenvalues = np.linalg.eigvalsh(gram)
    except np.linalg.LinAlgError:
        return None
    smallest = float(eigenvalues[0])
    largest = float(eigenvalues[-1])
    if smallest <= 0 or largest <= 0:
        return None
    return float(math.sqrt(largest / smallest))


def fit_sindy(
    X,
    t=None,
    dt=None,
    dXdt=None,
    degree=2,
    include_trig=False,
    include_interactions=True,
    threshold=0.05,
    alpha=1e-8,
    max_iter=20,
    state_names=None,
    custom_terms=None,
    derivative_method="finite_difference",
    window=None,
    polyorder=3,
    edge_trim=None,
    precision=3,
    target_names=None,
):
    """Fit one sparse equation per target column and render it as readable text.

    Returns per-state coefficients over the named candidate library, the rendered
    equation strings, R-squared against the estimated derivative, and the active-term
    count. R-squared here measures agreement with an *estimated* derivative, not with
    the true dynamics, and a high value with a badly chosen library is a warning rather
    than a result. `condition_number` reports the collinearity of the library: a large
    value means several terms explain the same variation and the reported support is
    not identifiable from these data alone.
    """
    states, derivatives, library, derivative_info, trim = _prepare(
        X, t, dt, dXdt, degree, include_trig, include_interactions,
        state_names, custom_terms, derivative_method, window, polyorder, edge_trim,
    )
    Theta = library["features"]
    names = library["names"]
    state_names = library["state_names"]
    labels = _target_labels(target_names, derivatives.shape[1], state_names)
    fit = stlsq(Theta, derivatives, threshold=threshold, alpha=alpha, max_iter=max_iter)
    coefficients = fit["coefficient_matrix"]

    equations, scores = [], []
    for index, name in enumerate(labels):
        equations.append(
            render_equation(
                coefficients[:, index], names, lhs=f"d{name}/dt", precision=precision
            )["equation"]
        )
        scores.append(_scores(Theta, derivatives[:, index], coefficients[:, index]))

    active_terms = [
        [names[j] for j in range(len(names)) if coefficients[j, index] != 0.0]
        for index in range(len(labels))
    ]
    condition = _condition_number(Theta)
    warnings = list(derivative_info.get("notes", []))
    if not fit["converged"]:
        warnings.append(
            "Sequential thresholding did not reach a fixed support within max_iter; "
            "the reported support is the last iterate, not a converged selection."
        )
    if condition is not None and condition > 1e6:
        warnings.append(
            f"Library condition number {condition:.3g} indicates near-collinear candidate "
            "terms; term attribution is not identifiable from these samples."
        )
    if library["degenerate_columns"]:
        warnings.append(
            "Zero-variation candidate columns present: "
            + ", ".join(library["degenerate_columns"])
        )
    return {
        "equations": equations,
        "coefficients": [[float(v) for v in coefficients[:, i]] for i in range(coefficients.shape[1])],
        "coefficient_matrix": coefficients,
        "term_names": names,
        "state_names": state_names,
        "target_names": labels,
        "active_terms": active_terms,
        "active_term_count": [len(a) for a in active_terms],
        "total_active_terms": int(sum(len(a) for a in active_terms)),
        "r_squared": [s["r_squared"] for s in scores],
        "rmse": [s["rmse"] for s in scores],
        "max_absolute_residual": [s["max_absolute_residual"] for s in scores],
        "converged": fit["converged"],
        "iterations": fit["iterations"],
        "threshold": fit["threshold"],
        "alpha": fit["alpha"],
        "degree": library["degree"],
        "n_samples": int(Theta.shape[0]),
        "n_library_terms": library["n_terms"],
        "condition_number": condition,
        "derivative": derivative_info,
        "edge_trim": int(trim),
        "warnings": warnings,
        "scope": SCOPE
        + " R-squared is measured against an estimated derivative, and term inclusion is "
        "contingent on the threshold and the noise in these samples.",
    }


def stability_selection(
    X,
    t=None,
    dt=None,
    dXdt=None,
    n_bootstrap=40,
    sample_fraction=0.7,
    seed=2026,
    inclusion_cut=0.8,
    threshold_jitter=0.0,
    **fit_options,
):
    """Resample the fit and report how often each candidate term survives.

    Meinshausen and Buhlmann, "Stability selection", JRSS-B 72(4):417-473, 2010, applied
    to the SINDy support. A term that appears in almost every subsample is robust to
    which samples were drawn; a term that appears occasionally is fitted to noise. This
    is the difference between a recovered equation that is trustworthy and one that is
    merely plausible.

    The inclusion probability is a frequency over resamples of *these* data under a fixed
    library and threshold. It is not a posterior probability that the term belongs to the
    real mechanism, and a term absent from the library has probability zero by construction.
    """
    n_bootstrap = _integer(n_bootstrap, "n_bootstrap", 2, MAX_BOOTSTRAP)
    sample_fraction = _positive(sample_fraction, "sample_fraction", low=0.05, high=1.0)
    inclusion_cut = _positive(inclusion_cut, "inclusion_cut", low=0.0, high=1.0)
    if isinstance(threshold_jitter, bool) or not isinstance(threshold_jitter, (int, float)):
        raise Invalid("threshold_jitter must be a number")
    threshold_jitter = float(threshold_jitter)
    if not 0.0 <= threshold_jitter < 1.0:
        raise Invalid("threshold_jitter must lie in [0, 1)")
    seed = _integer(int(seed), "seed", 0, 2**31 - 1)

    threshold = fit_options.get("threshold", 0.05)
    alpha = fit_options.get("alpha", 1e-8)
    max_iter = fit_options.get("max_iter", 20)
    target_names = fit_options.get("target_names")
    prep_options = {
        k: v
        for k, v in fit_options.items()
        if k not in ("threshold", "alpha", "max_iter", "precision", "target_names")
    }
    states, derivatives, library, derivative_info, trim = _prepare(
        X, t, dt, dXdt, **{
            "degree": prep_options.get("degree", 2),
            "include_trig": prep_options.get("include_trig", False),
            "include_interactions": prep_options.get("include_interactions", True),
            "state_names": prep_options.get("state_names"),
            "custom_terms": prep_options.get("custom_terms"),
            "derivative_method": prep_options.get("derivative_method", "finite_difference"),
            "window": prep_options.get("window"),
            "polyorder": prep_options.get("polyorder", 3),
            "edge_trim": prep_options.get("edge_trim"),
        }
    )
    Theta = library["features"]
    names = library["names"]
    state_names = library["state_names"]
    labels = _target_labels(target_names, derivatives.shape[1], state_names)
    n_samples, n_terms = Theta.shape
    draw = max(10, int(round(sample_fraction * n_samples)))
    if draw > n_samples:
        draw = n_samples

    counts = np.zeros((len(labels), n_terms))
    sums = np.zeros((len(labels), n_terms))
    squares = np.zeros((len(labels), n_terms))
    generator = np.random.default_rng(seed)
    non_convergences = 0
    for _ in range(n_bootstrap):
        rows = generator.choice(n_samples, size=draw, replace=False)
        local_threshold = threshold * (
            1.0 + threshold_jitter * (2.0 * generator.random() - 1.0)
        )
        for index in range(len(labels)):
            w, active, _iters, ok = _stlsq_column(
                Theta[rows], derivatives[rows, index], local_threshold, alpha, max_iter
            )
            non_convergences += 0 if ok else 1
            counts[index] += active
            sums[index] += np.where(active, w, 0.0)
            squares[index] += np.where(active, w * w, 0.0)

    probability = counts / n_bootstrap
    entries = []
    for index, state in enumerate(labels):
        for term in range(n_terms):
            observed = counts[index, term]
            mean = float(sums[index, term] / observed) if observed else 0.0
            variance = (
                float(squares[index, term] / observed - mean * mean) if observed else 0.0
            )
            entries.append(
                {
                    "state": state,
                    "equation": f"d{state}/dt",
                    "term": names[term],
                    "inclusion_probability": float(probability[index, term]),
                    "mean_coefficient_when_included": mean,
                    "sd_coefficient_when_included": float(math.sqrt(max(variance, 0.0))),
                }
            )
    entries.sort(key=lambda e: (-e["inclusion_probability"], e["state"], e["term"]))
    robust = [e for e in entries if e["inclusion_probability"] >= inclusion_cut]
    fragile = [
        e
        for e in entries
        if 0.0 < e["inclusion_probability"] < inclusion_cut
    ]
    return {
        "terms": entries,
        "inclusion_probability": [[float(v) for v in row] for row in probability],
        "robust_terms": robust,
        "fragile_terms": fragile,
        "never_selected": [e["term"] for e in entries if e["inclusion_probability"] == 0.0],
        "term_names": names,
        "state_names": state_names,
        "target_names": labels,
        "n_bootstrap": n_bootstrap,
        "sample_fraction": sample_fraction,
        "subsample_size": int(draw),
        "inclusion_cut": inclusion_cut,
        "threshold": float(threshold),
        "threshold_jitter": threshold_jitter,
        "seed": seed,
        "nonconvergent_fits": int(non_convergences),
        "derivative": derivative_info,
        "scope": (
            "Frequency of term survival under row resampling of the supplied data with a fixed "
            "library and threshold. Not a posterior probability, not a p-value, and not evidence "
            "about terms the library never contained."
        ),
    }


def pareto_sweep(X, thresholds, t=None, dt=None, dXdt=None, **fit_options):
    """Accuracy against sparsity across an explicit threshold grid.

    Model selection in SINDy is a trade-off, and leaving the threshold as a single hidden
    hyperparameter hides it. The returned front lets a reader see what accuracy each level
    of parsimony buys. `suggested_threshold` is the sparsest fit whose R-squared stays
    within `tolerance` of the best on the grid: a stated convention, not an optimum.

    Sequential thresholding is not guaranteed to be monotone in active-term count, so the
    observed monotonicity is reported rather than assumed.
    """
    if not isinstance(thresholds, (list, tuple)) or not thresholds:
        raise Invalid("thresholds must be a nonempty list")
    if len(thresholds) > MAX_THRESHOLDS:
        raise Invalid(f"At most {MAX_THRESHOLDS} thresholds may be swept")
    values = sorted(_positive(v, "threshold", low=-1e-12) for v in thresholds)
    tolerance = fit_options.pop("tolerance", 0.01)
    tolerance = _positive(tolerance, "tolerance", low=-1e-12, high=1.0)

    front = []
    for value in values:
        fit = fit_sindy(X, t=t, dt=dt, dXdt=dXdt, threshold=value, **fit_options)
        scores = [r for r in fit["r_squared"] if r is not None]
        front.append(
            {
                "threshold": value,
                "total_active_terms": fit["total_active_terms"],
                "active_term_count": fit["active_term_count"],
                "mean_rmse": float(np.mean(fit["rmse"])),
                "mean_r_squared": float(np.mean(scores)) if scores else None,
                "equations": fit["equations"],
                "converged": fit["converged"],
            }
        )
    counts = [entry["total_active_terms"] for entry in front]
    monotone = all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1))
    scored = [e for e in front if e["mean_r_squared"] is not None]
    suggested, reason = None, "No R-squared was defined on this grid."
    if scored:
        best = max(e["mean_r_squared"] for e in scored)
        eligible = [e for e in scored if e["mean_r_squared"] >= best - tolerance]
        chosen = min(eligible, key=lambda e: (e["total_active_terms"], -e["threshold"]))
        suggested = chosen["threshold"]
        reason = (
            f"Sparsest fit on the grid within {tolerance:g} mean R-squared of the best "
            f"({best:.4f}); a stated convention, not a validated selection rule."
        )
    return {
        "front": front,
        "thresholds": values,
        "active_term_counts": counts,
        "monotone_in_active_terms": bool(monotone),
        "suggested_threshold": suggested,
        "suggested_reason": reason,
        "tolerance": tolerance,
        "scope": (
            "A trade-off curve over one hyperparameter on one dataset. It does not select a "
            "true model, and every point on it inherits the library and derivative choices."
        ),
    }


def weak_form_sindy(
    X,
    t=None,
    dt=None,
    degree=2,
    include_trig=False,
    include_interactions=True,
    state_names=None,
    custom_terms=None,
    n_test_functions=100,
    support_points=25,
    test_polynomial_order=4,
    threshold=0.05,
    alpha=1e-8,
    max_iter=20,
    precision=3,
):
    """Integral (weak) formulation: never differentiate the noisy signal at all.

    Each row multiplies the trajectory by a compactly supported test function phi that
    vanishes at the ends of its window, integrates by parts so that the unknown derivative
    is moved onto phi, and integrates the candidate library against phi. Integration
    averages noise instead of amplifying it, which is why this variant survives noise
    levels that break the differencing form.

    Windows overlap, so the rows are correlated and the residual statistics are not those
    of independent observations.
    """
    X = _matrix(X, "X")
    n_samples, n_states = X.shape
    names = _check_names(state_names, n_states)
    step, times = _uniform_step(t, dt, n_samples, True)
    if times is None:
        times = np.arange(n_samples) * step
    support_points = _integer(support_points, "support_points", 2, max(2, n_samples // 3))
    n_test_functions = _integer(n_test_functions, "n_test_functions", 1, MAX_TEST_FUNCTIONS)
    order = _integer(test_polynomial_order, "test_polynomial_order", 2, 12)

    library = polynomial_library(
        X,
        degree=degree,
        include_trig=include_trig,
        include_interactions=include_interactions,
        state_names=names,
        custom_terms=custom_terms,
    )
    Theta = library["features"]
    half = support_points
    centres = np.unique(
        np.linspace(half, n_samples - 1 - half, n_test_functions).round().astype(int)
    )
    if centres.size < library["n_terms"]:
        raise Invalid(
            "Fewer usable test-function windows than candidate terms; widen the data or "
            "narrow the library"
        )
    rows_G, rows_b = [], []
    for centre in centres:
        window = slice(centre - half, centre + half + 1)
        local_t = times[window]
        scaled = (local_t - times[centre]) / (half * step)
        phi = (1.0 - scaled**2) ** order
        dphi = order * (1.0 - scaled**2) ** (order - 1) * (-2.0 * scaled) / (half * step)
        weight = np.trapezoid(phi, local_t)
        if weight <= 0:
            continue
        rows_G.append(np.trapezoid(phi[:, None] * Theta[window], local_t, axis=0) / weight)
        rows_b.append(-np.trapezoid(dphi[:, None] * X[window], local_t, axis=0) / weight)
    if len(rows_G) < library["n_terms"]:
        raise Invalid("Too few valid test-function windows for the candidate library")
    G = np.vstack(rows_G)
    b = np.vstack(rows_b)

    fit = stlsq(G, b, threshold=threshold, alpha=alpha, max_iter=max_iter)
    coefficients = fit["coefficient_matrix"]
    equations, scores = [], []
    for index, name in enumerate(names):
        equations.append(
            render_equation(
                coefficients[:, index], library["names"], lhs=f"d{name}/dt", precision=precision
            )["equation"]
        )
        scores.append(_scores(G, b[:, index], coefficients[:, index]))
    active_terms = [
        [library["names"][j] for j in range(library["n_terms"]) if coefficients[j, i] != 0.0]
        for i in range(len(names))
    ]
    return {
        "formulation": "weak",
        "equations": equations,
        "coefficients": [[float(v) for v in coefficients[:, i]] for i in range(coefficients.shape[1])],
        "coefficient_matrix": coefficients,
        "term_names": library["names"],
        "state_names": names,
        "active_terms": active_terms,
        "active_term_count": [len(a) for a in active_terms],
        "total_active_terms": int(sum(len(a) for a in active_terms)),
        "r_squared": [s["r_squared"] for s in scores],
        "rmse": [s["rmse"] for s in scores],
        "n_test_functions": int(len(rows_G)),
        "support_points": support_points,
        "test_polynomial_order": order,
        "condition_number": _condition_number(G),
        "converged": fit["converged"],
        "threshold": fit["threshold"],
        "alpha": fit["alpha"],
        "warnings": [
            "Test-function windows overlap, so weak-form rows are correlated; residual "
            "statistics are not those of independent observations."
        ],
        "scope": SCOPE
        + " Residuals are measured on integrated quantities, so a small weak-form RMSE does "
        "not imply a small pointwise derivative error.",
    }
