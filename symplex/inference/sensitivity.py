"""Local and global sensitivity analysis over a caller-supplied numeric simulator.

Local elasticities describe one point. Morris screening ranks parameters by detected effect
across a sampled box without decomposing variance. Sobol indices decompose the variance of a
scalar output and require the inputs to be independent. Forward propagation turns a supplied
parameter ensemble into a quantile fan.

None of these establishes that the simulator is right. A sensitivity index measures the model,
not the system: a parameter with a large index is one the *model* leans on, which is a
statement about where evidence would pay off, not about a real-world mechanism.
"""

import numpy as np
from scipy import stats

from symplex.core.contracts import Invalid
from symplex.inference.estimation import (
    CENTRAL_STEP,
    DEFAULT_EVALUATIONS,
    MAX_MATRIX_ENTRIES,
    MAX_OBSERVATIONS,
    MAX_PARAMETERS,
    _bounded_int,
    _bounds,
    _listed,
    _matrix,
    _Simulator,
    _theta,
    _unit_interval,
)

MAX_TRAJECTORIES = 200
MAX_LEVELS = 32
MAX_SOBOL_SAMPLES = 65_536
MAX_ENSEMBLE = 20_000
MAX_BOOTSTRAP = 1_000
DEFAULT_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def _boxed_bounds(bounds):
    """A finite sampling box; an unbounded design cannot be sampled uniformly."""
    try:
        pairs = list(bounds)
    except TypeError:
        raise Invalid("bounds must be a sequence of (lower, upper) pairs") from None
    if not 1 <= len(pairs) <= MAX_PARAMETERS:
        raise Invalid(f"bounds must describe 1 to {MAX_PARAMETERS} parameters")
    return _bounds(pairs, len(pairs), finite=True)


def _scalarizer(aggregate):
    """Morris and Sobol need one scalar per run; the caller declares which one."""
    if aggregate is None:
        def reduce_output(values):
            if values.size != 1:
                raise Invalid(
                    "Variance-based and screening designs need a scalar output: return a "
                    "single value from simulate or supply aggregate=callable(ndarray)->float"
                )
            return float(values[0])

        return reduce_output, "single scalar simulator output"
    if not callable(aggregate):
        raise Invalid("aggregate must be callable as aggregate(ndarray) -> float")

    def reduce_supplied(values):
        try:
            scalar = float(aggregate(values))
        except Invalid:
            raise
        except Exception as exc:
            raise Invalid(f"aggregate failed on a simulator output: {exc!r}") from exc
        if not np.isfinite(scalar):
            raise Invalid("aggregate returned a nonfinite value")
        return scalar

    return reduce_supplied, "caller-supplied aggregate of the simulator output vector"


def _quantile_key(fraction):
    return "p" + format(round(float(fraction) * 100, 6), "g")


def local_sensitivity(
    simulate,
    theta,
    bounds=None,
    relative_step=CENTRAL_STEP,
    max_evaluations=DEFAULT_EVALUATIONS,
):
    """Normalized elasticities ``(dY/Y) / (dTheta/Theta)`` at one parameter vector.

    An elasticity of 1 means a one-percent change in the parameter moves that prediction by
    one percent. Entries where the base prediction or the parameter is zero have no
    elasticity and are reported as ``None`` rather than as a large finite number.
    """
    point = _theta(theta)
    size = point.size
    step = float(relative_step)
    if not np.isfinite(step) or not 0 < step < 1:
        raise Invalid("relative_step must lie strictly inside (0, 1)")
    lower, upper = _bounds(bounds, size) if bounds is not None else (
        np.full(size, -np.inf),
        np.full(size, np.inf),
    )
    if np.any(point < lower) or np.any(point > upper):
        raise Invalid("theta must lie inside bounds")
    simulator = _Simulator(simulate, max_evaluations)
    simulator.require(1 + 2 * size, "local_sensitivity")
    base = simulator(point)
    steps = step * np.maximum(np.abs(point), 1.0)
    columns = []
    one_sided = []
    for index in range(size):
        h = steps[index]
        forward = point[index] + h <= upper[index]
        backward = point[index] - h >= lower[index]
        if not forward and not backward:
            raise Invalid(f"bounds[{index}] are narrower than the finite-difference step")
        if forward and backward:
            plus = np.array(point)
            plus[index] += h
            minus = np.array(point)
            minus[index] -= h
            columns.append((simulator(plus) - simulator(minus)) / (2 * h))
            one_sided.append(False)
        else:
            offset = h if forward else -h
            shifted = np.array(point)
            shifted[index] += offset
            columns.append((simulator(shifted) - base) / offset)
            one_sided.append(True)
    jacobian = np.column_stack(columns)
    with np.errstate(divide="ignore", invalid="ignore"):
        elasticity = jacobian * (point[None, :] / base[:, None])
    defined = np.isfinite(elasticity)
    undefined = int(elasticity.size - int(defined.sum()))
    safe = np.where(defined, elasticity, 0.0)
    counts = defined.sum(axis=0)
    summary = []
    for index in range(size):
        column = np.abs(safe[:, index])
        entries = int(counts[index])
        summary.append(
            {
                "index": index,
                "mean_abs_elasticity": float(column.sum() / entries) if entries else None,
                "max_abs_elasticity": float(column.max()) if entries else None,
                "mean_abs_derivative": float(np.abs(jacobian[:, index]).mean()),
                "defined_entries": entries,
                "one_sided": one_sided[index],
            }
        )
    return {
        "theta": point.tolist(),
        "base_prediction": _listed(base),
        "jacobian": _listed(jacobian),
        "elasticity": [
            [float(elasticity[i, j]) if defined[i, j] else None for j in range(size)]
            for i in range(jacobian.shape[0])
        ]
        if jacobian.size <= MAX_MATRIX_ENTRIES
        else None,
        "matrices_omitted": jacobian.size > MAX_MATRIX_ENTRIES,
        "parameter_summary": summary,
        "undefined_entries": undefined,
        "steps": steps.tolist(),
        "observation_count": int(base.size),
        "parameter_count": int(size),
        "evaluations": simulator.calls,
        "max_evaluations": simulator.budget,
        "scope": (
            "First-order local derivatives at a single point, by central differences. They do "
            "not describe behaviour away from that point, interactions between parameters, or "
            "any regime change the model may cross. Undefined entries are reported, not "
            "imputed. This is a property of the supplied simulator, not of the system it "
            "represents."
        ),
    }


def morris_screening(
    simulate,
    bounds,
    trajectories=10,
    levels=4,
    seed=0,
    aggregate=None,
    max_evaluations=DEFAULT_EVALUATIONS,
):
    """Morris elementary effects with mu, mu* and sigma, deterministic under ``seed``.

    Uses the standard one-at-a-time trajectory construction on a ``levels``-point grid over the
    unit hypercube (Morris 1991; Campolongo et al. 2007 for mu*). Each trajectory costs
    ``parameters + 1`` simulator calls. Effects are computed in unit-scaled parameter space, so
    mu* is comparable across parameters with different units but only within the supplied box.
    """
    lower, upper = _boxed_bounds(bounds)
    size = lower.size
    count = _bounded_int(trajectories, "trajectories", 2, MAX_TRAJECTORIES)
    grid_levels = _bounded_int(levels, "levels", 2, MAX_LEVELS)
    if grid_levels % 2:
        raise Invalid("levels must be even so that the Morris step keeps the design balanced")
    seed_value = _bounded_int(seed, "seed", 0, 2**32 - 1)
    reduce_output, reduction = _scalarizer(aggregate)
    simulator = _Simulator(simulate, max_evaluations)
    simulator.require(count * (size + 1), "morris_screening")

    delta = grid_levels / (2.0 * (grid_levels - 1))
    span = upper - lower
    generator = np.random.default_rng(seed_value)
    lower_triangle = np.tril(np.ones((size + 1, size)), -1)
    effects = np.zeros((count, size))
    for trajectory in range(count):
        base = generator.integers(0, grid_levels // 2, size=size) / (grid_levels - 1)
        signs = generator.choice([-1.0, 1.0], size=size)
        permutation = generator.permutation(size)
        design = base[None, :] + (delta / 2.0) * (
            (2.0 * lower_triangle - 1.0) * signs[None, :] + 1.0
        )
        design = np.clip(design[:, permutation], 0.0, 1.0)
        values = np.array(
            [reduce_output(simulator(lower + span * row)) for row in design]
        )
        for step in range(size):
            change = design[step + 1] - design[step]
            index = int(np.argmax(np.abs(change)))
            movement = change[index]
            effects[trajectory, index] = (
                (values[step + 1] - values[step]) / movement if movement != 0 else 0.0
            )
    mu = effects.mean(axis=0)
    mu_star = np.abs(effects).mean(axis=0)
    sigma = effects.std(axis=0, ddof=1) if count > 1 else np.zeros(size)
    order = np.argsort(-mu_star, kind="stable")
    ranks = np.empty(size, dtype=int)
    ranks[order] = np.arange(1, size + 1)
    return {
        "parameters": [
            {
                "index": index,
                "mu": float(mu[index]),
                "mu_star": float(mu_star[index]),
                "sigma": float(sigma[index]),
                "rank": int(ranks[index]),
            }
            for index in range(size)
        ],
        "ranking": [int(i) for i in order],
        "elementary_effects": _listed(effects),
        "trajectories": count,
        "levels": grid_levels,
        "delta": float(delta),
        "seed": seed_value,
        "output_reduction": reduction,
        "evaluations": simulator.calls,
        "max_evaluations": simulator.budget,
        "scope": (
            "A screening ranking over the supplied box, not a variance decomposition and not a "
            "significance test. A small mu* means no effect was detected under this design, "
            "not that the parameter is inert; a large sigma indicates interaction or "
            "nonlinearity without saying which. Effects are scaled to the box, so widening a "
            "bound changes the ranking. Ranks are relative and carry no interval."
        ),
    }


def sobol_indices(
    simulate,
    bounds,
    n,
    seed=0,
    aggregate=None,
    bootstrap=200,
    confidence=0.95,
    max_evaluations=DEFAULT_EVALUATIONS,
):
    """First-order and total-order Sobol indices via the Saltelli A/B/AB cross-sampling scheme.

    Two independent scrambled Sobol' sample matrices A and B of ``n`` rows are drawn from a
    single 2p-dimensional low-discrepancy sequence; ``AB_i`` is A with column i replaced from
    B. First-order indices use the Saltelli (2010) estimator
    ``S_i = mean(f(B) * (f(AB_i) - f(A))) / Var`` and total-order indices the Jansen (1999)
    estimator ``ST_i = mean((f(A) - f(AB_i))^2) / (2 Var)``. Cost is ``n * (p + 2)`` calls.

    Confidence intervals come from a seeded percentile bootstrap over sample rows and describe
    estimator noise at this ``n`` only.
    """
    lower, upper = _boxed_bounds(bounds)
    size = lower.size
    samples = _bounded_int(n, "n", 8, MAX_SOBOL_SAMPLES)
    if samples & (samples - 1):
        raise Invalid("n must be a power of two so the Sobol' sequence stays balanced")
    seed_value = _bounded_int(seed, "seed", 0, 2**32 - 1)
    draws = _bounded_int(bootstrap, "bootstrap", 0, MAX_BOOTSTRAP)
    level = _unit_interval(confidence, "confidence")
    reduce_output, reduction = _scalarizer(aggregate)
    simulator = _Simulator(simulate, max_evaluations)
    simulator.require(samples * (size + 2), "sobol_indices")

    engine = stats.qmc.Sobol(d=2 * size, scramble=True, seed=seed_value)
    unit = engine.random(samples)
    span = upper - lower
    matrix_a = lower + span * unit[:, :size]
    matrix_b = lower + span * unit[:, size:]

    def evaluate(rows):
        return np.array([reduce_output(simulator(row)) for row in rows])

    values_a = evaluate(matrix_a)
    values_b = evaluate(matrix_b)
    values_ab = np.empty((size, samples))
    for index in range(size):
        crossed = np.array(matrix_a)
        crossed[:, index] = matrix_b[:, index]
        values_ab[index] = evaluate(crossed)

    def indices_for(rows):
        a = values_a[rows]
        b = values_b[rows]
        variance = float(np.var(np.concatenate([a, b]), ddof=1))
        if variance <= 0:
            raise Invalid("Simulator output has zero variance over the sampled box")
        first = np.array(
            [float(np.mean(b * (values_ab[i][rows] - a)) / variance) for i in range(size)]
        )
        total = np.array(
            [
                float(np.mean((a - values_ab[i][rows]) ** 2) / (2.0 * variance))
                for i in range(size)
            ]
        )
        return first, total, variance

    all_rows = np.arange(samples)
    first_order, total_order, variance = indices_for(all_rows)
    first_ci = [None] * size
    total_ci = [None] * size
    if draws:
        generator = np.random.default_rng(seed_value + 1)
        resampled_first = np.empty((draws, size))
        resampled_total = np.empty((draws, size))
        for draw in range(draws):
            rows = generator.integers(0, samples, size=samples)
            resampled_first[draw], resampled_total[draw], _ = indices_for(rows)
        tail = (1.0 - level) / 2
        low_first = np.quantile(resampled_first, tail, axis=0)
        high_first = np.quantile(resampled_first, 1.0 - tail, axis=0)
        low_total = np.quantile(resampled_total, tail, axis=0)
        high_total = np.quantile(resampled_total, 1.0 - tail, axis=0)
        first_ci = [[float(low_first[i]), float(high_first[i])] for i in range(size)]
        total_ci = [[float(low_total[i]), float(high_total[i])] for i in range(size)]
    return {
        "parameters": [
            {
                "index": index,
                "first_order": float(first_order[index]),
                "first_order_interval": first_ci[index],
                "total_order": float(total_order[index]),
                "total_order_interval": total_ci[index],
            }
            for index in range(size)
        ],
        "first_order": first_order.tolist(),
        "total_order": total_order.tolist(),
        "sum_first_order": float(first_order.sum()),
        "interaction_share": float(1.0 - first_order.sum()),
        "output_variance": variance,
        "n": samples,
        "seed": seed_value,
        "bootstrap": draws,
        "confidence": level,
        "output_reduction": reduction,
        "evaluations": simulator.calls,
        "max_evaluations": simulator.budget,
        "scope": (
            "Sobol indices assume the inputs are INDEPENDENT and uniform over the supplied "
            "box; with correlated or physically coupled parameters the decomposition does not "
            "hold and these numbers should not be reported. Estimates are Monte Carlo "
            "quantities and can fall outside [0, 1] at finite n; the intervals describe "
            "estimator noise at this n, not model or structural uncertainty. Indices apportion "
            "the variance of the declared scalar output under the model, which is not evidence "
            "about the real system."
        ),
    }


def uncertainty_propagation(
    simulate,
    theta_samples,
    quantiles=DEFAULT_QUANTILES,
    max_evaluations=DEFAULT_EVALUATIONS,
):
    """Forward ensemble through the simulator: mean, spread and a quantile fan.

    ``theta_samples`` is a ``(draws, parameters)`` matrix the caller already produced, for
    example from ``bootstrap_intervals`` or a posterior sample. The returned fan is the object
    a decision-support chart is drawn from, so its scope matters: it carries only the
    variation present in the supplied samples.
    """
    matrix = _matrix(theta_samples, "theta_samples", MAX_ENSEMBLE, MAX_PARAMETERS)
    fractions = [_unit_interval(q, "quantiles") for q in list(quantiles)]
    if not 1 <= len(fractions) <= 11:
        raise Invalid("Supply between 1 and 11 quantile fractions")
    if len(set(fractions)) != len(fractions):
        raise Invalid("quantile fractions must be unique")
    if sorted(fractions) != fractions:
        raise Invalid("quantile fractions must be increasing")
    simulator = _Simulator(simulate, max_evaluations)
    simulator.require(matrix.shape[0], "uncertainty_propagation")
    ensemble = np.vstack([simulator(row) for row in matrix])
    if ensemble.shape[1] > MAX_OBSERVATIONS:
        raise Invalid("Simulator output exceeds the permitted observation envelope")
    fan = {
        _quantile_key(f): np.quantile(ensemble, f, axis=0).tolist() for f in fractions
    }
    return {
        "mean": ensemble.mean(axis=0).tolist(),
        "standard_deviation": (
            ensemble.std(axis=0, ddof=1) if ensemble.shape[0] > 1 else np.zeros(ensemble.shape[1])
        ).tolist(),
        "quantiles": fan,
        "quantile_fractions": fractions,
        "minimum": ensemble.min(axis=0).tolist(),
        "maximum": ensemble.max(axis=0).tolist(),
        "ensemble": _listed(ensemble),
        "ensemble_omitted": ensemble.size > MAX_MATRIX_ENTRIES,
        "draws": int(ensemble.shape[0]),
        "output_length": int(ensemble.shape[1]),
        "evaluations": simulator.calls,
        "max_evaluations": simulator.budget,
        "scope": (
            "Propagates only the variation present in the supplied parameter samples through "
            "the supplied simulator. It excludes observation error, model-structure error, "
            "any parameter the sample holds fixed, and any uncertainty the sample itself "
            "understates. The fan is therefore a conditional spread of model outputs, not a "
            "calibrated predictive interval; check it against held-out outcomes with "
            "pit_coverage or interval_score before treating it as one."
        ),
    }
