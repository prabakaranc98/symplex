"""Host-owned parameter estimation, asymptotic uncertainty and identifiability checks.

Every entry point takes a caller-supplied numeric simulator ``simulate(theta) -> ndarray``
whose output is aligned element-by-element with ``observations``. Nothing here imports a
model class, reads storage or executes generated code: a fit is a numerical operation on a
callable the caller already owns.

A fitted parameter carrying a standard error is the easiest number in this repository to
over-trust. Everything reported here is conditional on the supplied simulator being the
correct mechanism and on independent, correctly scaled Gaussian observation error. Under
those assumptions the intervals are asymptotic linearizations around the reported optimum.
They are not empirical validation, they do not cover model discrepancy or systematic
observation error, and a narrow interval on a non-identifiable parameter is an artifact of
the regularization used to invert a singular information matrix, not evidence.
"""

import numpy as np
from scipy import optimize, stats

from symplex.core.contracts import Invalid

MAX_PARAMETERS = 32
MAX_OBSERVATIONS = 200_000
MAX_EVALUATIONS = 200_000
MAX_GRID_POINTS = 201
MAX_DRAWS = 2_000
MAX_MATRIX_ENTRIES = 20_000
DEFAULT_EVALUATIONS = 20_000
DEFAULT_RCOND = 1e-10
# Standard central-difference step exponent; the truncation and roundoff terms balance here.
CENTRAL_STEP = float(np.cbrt(np.finfo(float).eps))

ASYMPTOTIC_SCOPE = (
    "Asymptotic linearization around the reported optimum, conditional on the supplied "
    "simulator being the correct mechanism and on independent Gaussian observation error "
    "of the supplied scale. Not empirical validation and not a model-discrepancy bound."
)


def _bounded_int(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise Invalid(name + " must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise Invalid(f"{name} must lie in [{low}, {high}]")
    return value


def _unit_interval(value, name, low=0.0, high=1.0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise Invalid(name + " must be a number") from None
    if not np.isfinite(number) or not low < number < high:
        raise Invalid(f"{name} must lie strictly inside ({low}, {high})")
    return number


def _vector(value, name, limit):
    try:
        array = np.atleast_1d(np.asarray(value, dtype=float))
    except (TypeError, ValueError):
        raise Invalid(name + " must be a numeric array") from None
    if array.ndim != 1:
        raise Invalid(name + " must be one-dimensional")
    if not 1 <= array.size <= limit:
        raise Invalid(f"{name} must hold between 1 and {limit} values")
    if not np.all(np.isfinite(array)):
        raise Invalid(name + " must be finite")
    return array


def _matrix(value, name, max_rows, max_columns):
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise Invalid(name + " must be a numeric matrix") from None
    if array.ndim != 2:
        raise Invalid(name + " must be two-dimensional")
    if not 1 <= array.shape[0] <= max_rows or not 1 <= array.shape[1] <= max_columns:
        raise Invalid(f"{name} exceeds the permitted {max_rows} x {max_columns} envelope")
    if not np.all(np.isfinite(array)):
        raise Invalid(name + " must be finite")
    return array


def _square(value, name):
    array = _matrix(value, name, MAX_PARAMETERS, MAX_PARAMETERS)
    if array.shape[0] != array.shape[1]:
        raise Invalid(name + " must be square")
    return 0.5 * (array + array.T)


def _theta(value, name="theta"):
    return _vector(value, name, MAX_PARAMETERS)


def _observations(value):
    return _vector(value, "observations", MAX_OBSERVATIONS)


def _bounds(value, size, finite=False):
    """Bounds are a sequence of one (lower, upper) pair per parameter, never two arrays."""
    if value is None:
        raise Invalid("bounds are required: supply one (lower, upper) pair per parameter")
    try:
        pairs = list(value)
    except TypeError:
        raise Invalid("bounds must be a sequence of (lower, upper) pairs") from None
    if len(pairs) != size:
        raise Invalid(f"bounds must supply exactly {size} (lower, upper) pairs")
    lower = np.empty(size)
    upper = np.empty(size)
    for index, pair in enumerate(pairs):
        try:
            low, high = (float(v) for v in pair)
        except (TypeError, ValueError):
            raise Invalid("Each bound must be a (lower, upper) pair of numbers") from None
        if np.isnan(low) or np.isnan(high) or not low < high:
            raise Invalid(f"bounds[{index}] must satisfy lower < upper")
        if finite and not (np.isfinite(low) and np.isfinite(high)):
            raise Invalid(f"bounds[{index}] must be finite for this routine")
        lower[index], upper[index] = low, high
    return lower, upper


def _scale(sigma, size):
    """Return the observation-error scale and whether the caller actually supplied one."""
    if sigma is None:
        return np.ones(size), False
    array = np.asarray(sigma, dtype=float)
    if array.ndim == 0:
        array = np.full(size, float(array))
    array = _vector(array, "sigma", MAX_OBSERVATIONS)
    if array.size != size:
        raise Invalid("sigma must be scalar or match the observation count")
    if np.any(array <= 0):
        raise Invalid("sigma must be strictly positive")
    return array, True


def _names(parameter_names, size):
    if parameter_names is None:
        return [f"theta_{index}" for index in range(size)]
    names = list(parameter_names)
    if len(names) != size or any(not isinstance(n, str) or not n.strip() for n in names):
        raise Invalid("parameter_names must supply one nonempty name per parameter")
    if len(set(names)) != len(names):
        raise Invalid("parameter_names must be unique")
    return names


def _listed(array, entries=MAX_MATRIX_ENTRIES):
    """Serialize a matrix only when it fits the reporting envelope."""
    array = np.asarray(array)
    return array.tolist() if array.size <= entries else None


class _Simulator:
    """Counts every simulator call against a hard budget and validates each output.

    The budget is a resource envelope, not a convergence criterion: exhausting it raises
    rather than returning a partially explored result that would read like a finished one.
    """

    def __init__(self, simulate, max_evaluations, size=None):
        if not callable(simulate):
            raise Invalid("simulate must be callable as simulate(theta) -> ndarray")
        self._simulate = simulate
        self.budget = _bounded_int(max_evaluations, "max_evaluations", 1, MAX_EVALUATIONS)
        self.size = size
        self.calls = 0

    def __call__(self, theta):
        if self.calls >= self.budget:
            raise Invalid(
                f"Simulator evaluation budget of {self.budget} calls is exhausted; "
                "raise max_evaluations deliberately or reduce the requested design"
            )
        self.calls += 1
        try:
            raw = self._simulate(np.array(theta, dtype=float))
        except Invalid:
            raise
        except Exception as exc:  # A caller simulator failure is an input failure here.
            raise Invalid(f"Simulator call {self.calls} failed: {exc!r}") from exc
        try:
            out = np.atleast_1d(np.asarray(raw, dtype=float))
        except (TypeError, ValueError):
            raise Invalid("Simulator must return a numeric array") from None
        if out.ndim != 1:
            raise Invalid("Simulator must return a one-dimensional prediction vector")
        if self.size is None:
            if not 1 <= out.size <= MAX_OBSERVATIONS:
                raise Invalid("Simulator output exceeds the permitted observation envelope")
            self.size = int(out.size)
        elif out.size != self.size:
            raise Invalid("Simulator returned a prediction vector of changing length")
        if not np.all(np.isfinite(out)):
            raise Invalid("Simulator returned a nonfinite prediction")
        return out

    def require(self, count, label):
        if count > self.budget:
            raise Invalid(
                f"{label} needs {count} simulator calls but max_evaluations is {self.budget}"
            )


def _at_bound(theta, lower, upper):
    tolerance = 1e-8 * np.maximum(1.0, np.abs(theta))
    low = np.isfinite(lower) & (theta - lower <= tolerance)
    high = np.isfinite(upper) & (upper - theta <= tolerance)
    return [bool(v) for v in (low | high)]


def least_squares_fit(
    simulate,
    observations,
    theta0,
    bounds,
    sigma=None,
    max_evaluations=DEFAULT_EVALUATIONS,
    max_nfev=None,
):
    """Weighted least squares against a caller-supplied simulator.

    Residuals are weighted by ``1/sigma``; with ``sigma=None`` the weights are unity and the
    reported reduced chi-square is an unscaled residual variance rather than a chi-square.
    Non-convergence is reported as such: ``theta_hat`` is ``None`` when the optimizer did not
    terminate on a convergence criterion, and only ``terminal_theta`` carries the last point.
    """
    obs = _observations(observations)
    start = _theta(theta0, "theta0")
    lower, upper = _bounds(bounds, start.size)
    if np.any(start < lower) or np.any(start > upper):
        raise Invalid("theta0 must lie inside bounds")
    scale, supplied = _scale(sigma, obs.size)
    if max_nfev is not None:
        max_nfev = _bounded_int(max_nfev, "max_nfev", 1, MAX_EVALUATIONS)
    simulator = _Simulator(simulate, max_evaluations, obs.size)

    def residual(theta):
        return (simulator(theta) - obs) / scale

    result = optimize.least_squares(
        residual, start, bounds=(lower, upper), max_nfev=max_nfev
    )
    theta = np.asarray(result.x, dtype=float)
    weighted = np.asarray(result.fun, dtype=float)
    rss = float(np.dot(weighted, weighted))
    dof = int(obs.size - theta.size)
    reduced = float(rss / dof) if dof > 0 else None
    converged = bool(result.status > 0)
    return {
        "converged": converged,
        "status": int(result.status),
        "termination": str(result.message),
        "theta_hat": theta.tolist() if converged else None,
        "terminal_theta": theta.tolist(),
        "residuals": _listed(weighted),
        "residuals_omitted": weighted.size > MAX_MATRIX_ENTRIES,
        "rss": rss,
        "degrees_of_freedom": dof,
        "reduced_chi_square": reduced,
        "residual_scale": float(np.sqrt(reduced)) if reduced is not None else None,
        "at_bound": _at_bound(theta, lower, upper),
        "optimality": float(result.optimality),
        "observation_count": int(obs.size),
        "parameter_count": int(theta.size),
        "sigma_supplied": supplied,
        "evaluations": simulator.calls,
        "max_evaluations": simulator.budget,
        "scope": (
            (
                "Converged weighted least-squares optimum. "
                if converged
                else "Optimizer did NOT reach a convergence criterion; theta_hat is withheld and "
                "terminal_theta is the last visited point, not an estimate. "
            )
            + "A local optimum only; no global-optimum, identifiability or goodness-of-fit claim "
            "is made here. "
            + (
                ""
                if supplied
                else "No sigma was supplied, so reduced_chi_square is a residual variance and "
                "cannot be read as a chi-square goodness-of-fit statistic. "
            )
            + "Parameters resting on a bound invalidate the asymptotic theory downstream."
        ),
    }


def fisher_information(
    simulate,
    theta_hat,
    sigma=None,
    bounds=None,
    relative_step=CENTRAL_STEP,
    max_evaluations=DEFAULT_EVALUATIONS,
    rcond=DEFAULT_RCOND,
):
    """Finite-difference sensitivity matrix S and the weighted information S^T W S.

    ``S[i, j] = d prediction_i / d theta_j`` by central differences, ``W = diag(1/sigma^2)``.
    Steps are ``relative_step * max(|theta_j|, 1)``; where a bound blocks one side the
    difference falls back to the feasible side and is flagged in ``one_sided``.
    """
    theta = _theta(theta_hat, "theta_hat")
    size = theta.size
    step = float(relative_step)
    if not np.isfinite(step) or not 0 < step < 1:
        raise Invalid("relative_step must lie strictly inside (0, 1)")
    lower, upper = _bounds(bounds, size) if bounds is not None else (
        np.full(size, -np.inf),
        np.full(size, np.inf),
    )
    if np.any(theta < lower) or np.any(theta > upper):
        raise Invalid("theta_hat must lie inside bounds")
    simulator = _Simulator(simulate, max_evaluations)
    simulator.require(1 + 2 * size, "fisher_information")
    base = simulator(theta)
    scale, supplied = _scale(sigma, base.size)

    steps = step * np.maximum(np.abs(theta), 1.0)
    columns = []
    one_sided = []
    for index in range(size):
        h = steps[index]
        forward = theta[index] + h <= upper[index]
        backward = theta[index] - h >= lower[index]
        if not forward and not backward:
            raise Invalid(
                f"bounds[{index}] are narrower than the finite-difference step; widen the "
                "bounds or reduce relative_step"
            )
        if forward and backward:
            plus = np.array(theta)
            plus[index] += h
            minus = np.array(theta)
            minus[index] -= h
            columns.append((simulator(plus) - simulator(minus)) / (2 * h))
            one_sided.append(False)
        else:
            offset = h if forward else -h
            shifted = np.array(theta)
            shifted[index] += offset
            columns.append((simulator(shifted) - base) / offset)
            one_sided.append(True)
    sensitivity = np.column_stack(columns)
    weights = 1.0 / (scale**2)
    fim = sensitivity.T @ (sensitivity * weights[:, None])
    fim = 0.5 * (fim + fim.T)
    eigenvalues = np.linalg.eigvalsh(fim)[::-1]
    largest = float(eigenvalues[0])
    smallest = float(eigenvalues[-1])
    condition = None
    if largest > 0 and smallest > 0 and largest / smallest < 1e300:
        condition = float(largest / smallest)
    floor = max(largest, 0.0) * float(rcond)
    return {
        "fim": fim.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "eigenvalue_ratio_min": float(smallest / largest) if largest > 0 else None,
        "condition_number": condition,
        "numerically_singular": condition is None,
        "numerical_rank": int(np.sum(eigenvalues > floor)),
        "sensitivity_matrix": _listed(sensitivity),
        "sensitivity_matrix_omitted": sensitivity.size > MAX_MATRIX_ENTRIES,
        "steps": steps.tolist(),
        "one_sided": one_sided,
        "observation_count": int(base.size),
        "parameter_count": int(size),
        "sigma_supplied": supplied,
        "evaluations": simulator.calls,
        "max_evaluations": simulator.budget,
        "scope": (
            "Local finite-difference curvature of the supplied simulator at one point. The "
            "information matrix is only as meaningful as the model that produced it: a large "
            "condition number or a rank below the parameter count means the data cannot "
            "separate some parameter combination, not that the fit is wrong. "
            "One-sided columns carry first-order truncation error. " + ASYMPTOTIC_SCOPE
        ),
    }


def parameter_uncertainty(
    fim,
    scale=1.0,
    theta=None,
    parameter_names=None,
    rcond=DEFAULT_RCOND,
    confidence=0.95,
):
    """Asymptotic covariance, standard errors and correlations from a spectral pseudo-inverse.

    The inverse is taken on the eigenvalue spectrum with a floor at ``rcond * lambda_max``.
    Flooring rather than truncating keeps the near-null directions visible: a strict
    Moore-Penrose inverse zeroes them and would report a *small* standard error for a
    parameter the data cannot determine at all. In a floored direction the reported standard
    error is a lower bound set by the floor, and the true asymptotic uncertainty is unbounded.

    With ``sigma`` supplied to ``fisher_information`` use ``scale=1``. With unweighted
    residuals pass ``scale=reduced_chi_square`` from the fit to recover the usual
    ``s^2 (X^T X)^-1`` covariance.
    """
    information = _square(fim, "fim")
    size = information.shape[0]
    names = _names(parameter_names, size)
    variance_scale = float(scale)
    if not np.isfinite(variance_scale) or variance_scale <= 0:
        raise Invalid("scale must be a positive finite number")
    floor_ratio = float(rcond)
    if not np.isfinite(floor_ratio) or not 0 < floor_ratio < 1:
        raise Invalid("rcond must lie strictly inside (0, 1)")
    level = _unit_interval(confidence, "confidence")
    values, vectors = np.linalg.eigh(information)
    largest = float(values[-1])
    if largest <= 0:
        raise Invalid("fim must have at least one positive eigenvalue")
    floor = largest * floor_ratio
    rank = int(np.sum(values > floor))
    covariance = (vectors * (1.0 / np.maximum(values, floor))) @ vectors.T
    covariance = 0.5 * (covariance + covariance.T) * variance_scale
    variances = np.clip(np.diag(covariance), 0.0, None)
    errors = np.sqrt(variances)
    outer = np.outer(errors, errors)
    correlation = np.zeros_like(covariance)
    usable = outer > 0
    correlation[usable] = np.clip(covariance[usable] / outer[usable], -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    interval = None
    if theta is not None:
        estimate = _theta(theta)
        if estimate.size != size:
            raise Invalid("theta must match the information matrix dimension")
        half = float(stats.norm.ppf(0.5 + level / 2)) * errors
        interval = [
            {"name": names[j], "estimate": float(estimate[j]),
             "lower": float(estimate[j] - half[j]), "upper": float(estimate[j] + half[j])}
            for j in range(size)
        ]
    return {
        "parameter_names": names,
        "covariance": covariance.tolist(),
        "standard_errors": errors.tolist(),
        "correlation": correlation.tolist(),
        "numerical_rank": rank,
        "floored_directions": int(size - rank),
        "eigenvalues": values[::-1].tolist(),
        "eigenvalue_floor": float(floor),
        "variance_scale": variance_scale,
        "confidence": level,
        "intervals": interval,
        "scope": (
            "Asymptotic covariance from a floored spectral pseudo-inverse of the information "
            "matrix. " + ASYMPTOTIC_SCOPE + " "
            + (
                f"{size - rank} of {size} directions were floored at rcond*lambda_max: their "
                "standard errors are a floor artifact and understate an unbounded uncertainty."
                if rank < size
                else "No direction required flooring at this rcond."
            )
            + " Intervals are Wald intervals and are not valid for a parameter resting on a "
            "bound or entering the model nonlinearly over the interval width; check "
            "profile_likelihood before quoting them."
        ),
    }


def practical_identifiability(
    fim,
    theta=None,
    parameter_names=None,
    scale=1.0,
    rcond=DEFAULT_RCOND,
    collinearity_threshold=0.95,
    weak_correlation=0.9,
    null_tolerance=1e-8,
    weak_tolerance=1e-4,
):
    """Classify each parameter from information-eigenvalue decay and the correlation matrix.

    Two signals are combined. The first is the loading of each parameter axis in the
    numerically null eigenspace of the information matrix (eigenvalue ratio below
    ``null_tolerance``); a parameter lying mostly in a null direction cannot be determined
    from these data at all. The second is near-collinearity: any pair with
    ``|corr| > collinearity_threshold`` is jointly unresolvable, which is the failure mode
    that silently invalidates a fit while every individual standard error still looks finite.
    Collinear pairs are therefore reported explicitly rather than summarized away.

    Thresholds are reporting conventions, not tests, and are echoed back in ``thresholds``.
    """
    information = _square(fim, "fim")
    size = information.shape[0]
    names = _names(parameter_names, size)
    for value, label in ((collinearity_threshold, "collinearity_threshold"),
                         (weak_correlation, "weak_correlation")):
        _unit_interval(value, label)
    for value, label in ((null_tolerance, "null_tolerance"), (weak_tolerance, "weak_tolerance")):
        _unit_interval(value, label)
    if not weak_tolerance > null_tolerance:
        raise Invalid("weak_tolerance must exceed null_tolerance")
    if not collinearity_threshold > weak_correlation:
        raise Invalid("collinearity_threshold must exceed weak_correlation")
    uncertainty = parameter_uncertainty(
        information, scale=scale, theta=theta, parameter_names=names, rcond=rcond
    )
    correlation = np.asarray(uncertainty["correlation"], dtype=float)
    errors = np.asarray(uncertainty["standard_errors"], dtype=float)
    values, vectors = np.linalg.eigh(information)
    largest = float(values[-1])
    ratios = values / largest
    null_loading = np.sqrt(np.sum(vectors[:, ratios < null_tolerance] ** 2, axis=1))
    weak_loading = np.sqrt(np.sum(vectors[:, ratios < weak_tolerance] ** 2, axis=1))

    offdiagonal = np.abs(correlation) - np.eye(size)
    pairs = []
    for i in range(size):
        for j in range(i + 1, size):
            if abs(correlation[i, j]) > collinearity_threshold:
                pairs.append(
                    {
                        "indices": [i, j],
                        "names": [names[i], names[j]],
                        "correlation": float(correlation[i, j]),
                    }
                )
    estimate = _theta(theta) if theta is not None else None
    if estimate is not None and estimate.size != size:
        raise Invalid("theta must match the information matrix dimension")

    entries = []
    for index in range(size):
        row = offdiagonal[index]
        partner = int(np.argmax(row)) if size > 1 else None
        peak = float(row[partner]) if size > 1 else 0.0
        if null_loading[index] > 0.5 or peak > collinearity_threshold or information[index, index] <= 0:
            verdict = "non_identifiable"
        elif weak_loading[index] > 0.5 or peak > weak_correlation:
            verdict = "weakly_identifiable"
        else:
            verdict = "identifiable"
        relative = None
        if estimate is not None and estimate[index] != 0:
            relative = float(errors[index] / abs(estimate[index]))
        entries.append(
            {
                "index": index,
                "name": names[index],
                "verdict": verdict,
                "max_abs_correlation": peak,
                "most_correlated_with": names[partner] if partner is not None else None,
                "null_space_loading": float(null_loading[index]),
                "weak_space_loading": float(weak_loading[index]),
                "standard_error": float(errors[index]),
                "relative_standard_error": relative,
            }
        )
    counts = {
        key: sum(1 for e in entries if e["verdict"] == key)
        for key in ("identifiable", "weakly_identifiable", "non_identifiable")
    }
    overall = (
        "non_identifiable"
        if counts["non_identifiable"]
        else "weakly_identifiable"
        if counts["weakly_identifiable"]
        else "identifiable"
    )
    return {
        "overall": overall,
        "parameters": entries,
        "verdict_counts": counts,
        "collinear_pairs": pairs,
        "eigenvalues": values[::-1].tolist(),
        "eigenvalue_ratios": ratios[::-1].tolist(),
        "condition_number": (
            float(largest / values[0]) if values[0] > 0 and largest / values[0] < 1e300 else None
        ),
        "numerical_rank": uncertainty["numerical_rank"],
        "correlation": uncertainty["correlation"],
        "standard_errors": uncertainty["standard_errors"],
        "parameter_names": names,
        "thresholds": {
            "collinearity_threshold": float(collinearity_threshold),
            "weak_correlation": float(weak_correlation),
            "null_tolerance": float(null_tolerance),
            "weak_tolerance": float(weak_tolerance),
            "rcond": float(rcond),
        },
        "scope": (
            "A local, linearized identifiability verdict at one point, computed from the "
            "information matrix alone. It detects flat and confounded directions in the "
            "supplied design; it does not prove that an 'identifiable' parameter is estimable "
            "over a wide range, and a nonlinear model can be locally identifiable and globally "
            "not. Confirm any consequential verdict with profile_likelihood. The thresholds "
            "are conventions, not hypothesis tests, and no significance claim is made."
        ),
    }


def profile_likelihood(
    simulate,
    observations,
    theta_hat,
    index,
    grid,
    bounds,
    sigma=None,
    confidence=0.95,
    flat_tolerance=1e-3,
    max_evaluations=DEFAULT_EVALUATIONS,
    max_nfev=None,
):
    """Fix one parameter across a grid, re-optimize the rest, and report the profile.

    The profile likelihood distinguishes the two failure modes that a covariance matrix
    cannot: a completely flat profile is structural non-identifiability (the parameter has no
    effect the remaining parameters cannot absorb), while a profile that rises but never
    crosses the threshold on one side is practical non-identifiability, leaving an open
    confidence interval. Raue et al., "Structural and practical identifiability analysis of
    partially observed dynamical models by exploiting the profile likelihood", Bioinformatics
    25(15):1923-1929, 2009.

    Points are solved outward from the grid node nearest ``theta_hat[index]`` and warm-started
    from the neighbouring solution. The threshold is the chi-square quantile on one degree of
    freedom, which is itself an asymptotic approximation.
    """
    obs = _observations(observations)
    theta = _theta(theta_hat, "theta_hat")
    size = theta.size
    lower, upper = _bounds(bounds, size)
    if np.any(theta < lower) or np.any(theta > upper):
        raise Invalid("theta_hat must lie inside bounds")
    position = _bounded_int(index, "index", 0, size - 1)
    nodes = _vector(grid, "grid", MAX_GRID_POINTS)
    if np.any(np.diff(nodes) <= 0):
        raise Invalid("grid must be strictly increasing")
    if np.any(nodes < lower[position]) or np.any(nodes > upper[position]):
        raise Invalid("grid must lie inside the bounds of the profiled parameter")
    if not nodes[0] <= theta[position] <= nodes[-1]:
        raise Invalid("grid must bracket theta_hat[index]")
    level = _unit_interval(confidence, "confidence")
    tolerance = float(flat_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise Invalid("flat_tolerance must be positive")
    if max_nfev is not None:
        max_nfev = _bounded_int(max_nfev, "max_nfev", 1, MAX_EVALUATIONS)
    scale, supplied = _scale(sigma, obs.size)
    simulator = _Simulator(simulate, max_evaluations, obs.size)
    free = [j for j in range(size) if j != position]

    def residual_at(value, free_theta):
        full = np.empty(size)
        full[position] = value
        if free:
            full[free] = free_theta
        return (simulator(full) - obs) / scale

    reference = residual_at(theta[position], theta[free])
    reference_rss = float(np.dot(reference, reference))

    solutions = [None] * nodes.size
    anchor = int(np.argmin(np.abs(nodes - theta[position])))

    def solve(node_index, start):
        value = float(nodes[node_index])
        if not free:
            residuals = residual_at(value, np.zeros(0))
            rss = float(np.dot(residuals, residuals))
            solutions[node_index] = (rss, True, start)
            return start
        result = optimize.least_squares(
            lambda f: residual_at(value, f),
            start,
            bounds=(lower[free], upper[free]),
            max_nfev=max_nfev,
        )
        rss = float(2.0 * result.cost)
        converged = bool(result.status > 0)
        solutions[node_index] = (rss, converged, np.asarray(result.x, dtype=float))
        return np.asarray(result.x, dtype=float) if converged else start

    warm = solve(anchor, theta[free])
    forward = warm
    for node_index in range(anchor + 1, nodes.size):
        forward = solve(node_index, forward)
    backward = warm
    for node_index in range(anchor - 1, -1, -1):
        backward = solve(node_index, backward)

    rss_values = np.array([s[0] for s in solutions])
    converged = [bool(s[1]) for s in solutions]
    minimum = float(min(reference_rss, float(rss_values.min())))
    if supplied or minimum <= 0:
        # A zero-residual minimum has no variance to profile out, so use the plain difference.
        deltas = rss_values - minimum
    else:
        # Without a supplied sigma the noise level is profiled out too, which is the
        # n*log(RSS/RSS_min) likelihood ratio rather than a plain residual-sum difference.
        deltas = obs.size * np.log(np.maximum(rss_values, 1e-300) / max(minimum, 1e-300))
        deltas = np.where(np.isfinite(deltas), deltas, 0.0)
    threshold = float(stats.chi2.ppf(level, 1))

    def crossing(side):
        order = range(anchor, -1, -1) if side == "left" else range(anchor, nodes.size)
        previous = None
        for node_index in order:
            if deltas[node_index] >= threshold:
                if previous is None:
                    return float(nodes[node_index])
                span = deltas[node_index] - deltas[previous]
                if span <= 0:
                    return float(nodes[node_index])
                fraction = (threshold - deltas[previous]) / span
                return float(nodes[previous] + fraction * (nodes[node_index] - nodes[previous]))
            previous = node_index
        return None

    left = crossing("left")
    right = crossing("right")
    peak = float(deltas.max())
    if peak < tolerance:
        verdict = "structurally_non_identifiable"
    elif left is None or right is None:
        verdict = "practically_non_identifiable"
    else:
        verdict = "identifiable"
    return {
        "index": position,
        "grid": nodes.tolist(),
        "rss": rss_values.tolist(),
        "delta_chi_square": deltas.tolist(),
        "threshold": threshold,
        "confidence": level,
        "minimum_rss": minimum,
        "reference_rss": reference_rss,
        "max_delta": peak,
        "flat": bool(peak < tolerance),
        "flat_tolerance": tolerance,
        "identifiability": verdict,
        "confidence_interval": {"lower": left, "upper": right},
        "open_left": left is None,
        "open_right": right is None,
        "converged": converged,
        "non_converged_points": int(sum(1 for c in converged if not c)),
        "sigma_supplied": supplied,
        "evaluations": simulator.calls,
        "max_evaluations": simulator.budget,
        "scope": (
            "A profile over the supplied grid only. An interval endpoint reported here is a "
            "threshold crossing inside that grid; 'open' means the profile did not cross "
            "within the explored range, never that the parameter is bounded there. Flatness "
            "is asserted relative to flat_tolerance and to the inner optimizer's own "
            "convergence, so a poorly converged inner fit can imitate either verdict; "
            "non_converged_points must be read before the verdict. The chi-square threshold "
            "is asymptotic and conditional on the supplied model being correct. "
            + ("" if supplied else "Without sigma the profile uses the log-scaled residual "
               "ratio, which is an approximation with an unknown noise level.")
        ),
    }


def bootstrap_intervals(
    simulate,
    observations,
    theta_hat,
    bounds,
    sigma=None,
    draws=200,
    seed=0,
    confidence=0.95,
    max_evaluations=DEFAULT_EVALUATIONS,
    max_nfev=None,
):
    """Deterministic seeded residual bootstrap around a fitted parameter vector.

    Residuals are standardized by ``sigma``, resampled with replacement under an explicit
    seed, added back to the fitted prediction, and the model is refitted. Draws whose refit
    does not converge are discarded and counted rather than silently included.
    """
    obs = _observations(observations)
    theta = _theta(theta_hat, "theta_hat")
    lower, upper = _bounds(bounds, theta.size)
    if np.any(theta < lower) or np.any(theta > upper):
        raise Invalid("theta_hat must lie inside bounds")
    count = _bounded_int(draws, "draws", 2, MAX_DRAWS)
    seed_value = _bounded_int(seed, "seed", 0, 2**32 - 1)
    level = _unit_interval(confidence, "confidence")
    if max_nfev is not None:
        max_nfev = _bounded_int(max_nfev, "max_nfev", 1, MAX_EVALUATIONS)
    scale, supplied = _scale(sigma, obs.size)
    simulator = _Simulator(simulate, max_evaluations, obs.size)
    fitted = simulator(theta)
    standardized = (obs - fitted) / scale
    generator = np.random.default_rng(seed_value)
    indices = generator.integers(0, obs.size, size=(count, obs.size))

    samples = []
    failures = 0
    for row in indices:
        replicate = fitted + scale * standardized[row]

        def residual(candidate, target=replicate):
            return (simulator(candidate) - target) / scale

        result = optimize.least_squares(
            residual, theta, bounds=(lower, upper), max_nfev=max_nfev
        )
        if result.status > 0:
            samples.append(np.asarray(result.x, dtype=float))
        else:
            failures += 1
    if not samples:
        raise Invalid("No bootstrap refit converged; the interval is not defined")
    matrix = np.vstack(samples)
    tail = (1.0 - level) / 2
    lower_quantile = np.quantile(matrix, tail, axis=0)
    upper_quantile = np.quantile(matrix, 1.0 - tail, axis=0)
    return {
        "theta_hat": theta.tolist(),
        "lower": lower_quantile.tolist(),
        "upper": upper_quantile.tolist(),
        "bootstrap_standard_errors": matrix.std(axis=0, ddof=1).tolist()
        if matrix.shape[0] > 1
        else [0.0] * theta.size,
        "bootstrap_mean": matrix.mean(axis=0).tolist(),
        "bias": (matrix.mean(axis=0) - theta).tolist(),
        "draws_requested": count,
        "draws_used": int(matrix.shape[0]),
        "failed_draws": failures,
        "seed": seed_value,
        "confidence": level,
        "samples": _listed(matrix),
        "samples_omitted": matrix.size > MAX_MATRIX_ENTRIES,
        "sigma_supplied": supplied,
        "evaluations": simulator.calls,
        "max_evaluations": simulator.budget,
        "scope": (
            "Residual bootstrap conditional on the supplied model being correct and on "
            "residuals that are exchangeable and identically distributed after scaling by "
            "sigma. It does not cover model discrepancy, serially correlated or "
            "heteroscedastic error beyond sigma, or optimizer multimodality. Percentile "
            "intervals are uncorrected for bias or acceleration, and discarded draws bias the "
            "interval toward regions where the optimizer converges."
        ),
    }
