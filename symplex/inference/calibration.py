"""Generic forecast calibration and predictive-check scoring.

These routines take arrays, not prepared cases: probabilities and binary outcomes, forecast
ensembles and realized values, interval endpoints, simulated replicates. They are the
domain-independent counterpart to ``symplex.evaluation.metrics``, which scores one prepared
prediction-market protocol.

Calibration is a necessary property of a usable forecast, not a sufficient one. A perfectly
calibrated forecast can be useless (the climatological base rate is perfectly calibrated and
has no resolution), and a well-fitted model can be badly calibrated out of sample. Everything
here describes the supplied sample under the supplied binning; nothing here is a hypothesis
test of model adequacy, and no result establishes that the forecasts will hold on new data.
"""

import numpy as np
from scipy import stats

from symplex.core.contracts import Invalid
from symplex.inference.estimation import (
    MAX_MATRIX_ENTRIES,
    MAX_OBSERVATIONS,
    _bounded_int,
    _listed,
    _matrix,
    _unit_interval,
    _vector,
)

MAX_BINS = 100
MAX_ENSEMBLE_MEMBERS = 20_000
MAX_REPLICATES = 20_000
MAX_STATISTICS = 16
DEFAULT_LEVELS = (0.5, 0.8, 0.9, 0.95)


def _probabilities(values):
    array = _vector(values, "probabilities", MAX_OBSERVATIONS)
    if np.any(array < 0) or np.any(array > 1):
        raise Invalid("probabilities must lie in [0, 1]")
    return array


def _outcomes(values, size):
    array = _vector(values, "outcomes", MAX_OBSERVATIONS)
    if array.size != size:
        raise Invalid("outcomes must align one-to-one with the forecasts")
    if not np.all((array == 0) | (array == 1)):
        raise Invalid("outcomes must be binary 0 or 1")
    return array


def _ensemble(values, name=" ensemble"):
    """A ``(members,)`` or ``(members, observations)`` forecast ensemble."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raise Invalid(name.strip() + " must be a numeric array") from None
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise Invalid(name.strip() + " must be one- or two-dimensional")
    if not 2 <= array.shape[0] <= MAX_ENSEMBLE_MEMBERS:
        raise Invalid(f"{name.strip()} needs 2 to {MAX_ENSEMBLE_MEMBERS} members")
    if not 1 <= array.shape[1] <= MAX_OBSERVATIONS:
        raise Invalid(name.strip() + " exceeds the permitted observation envelope")
    if not np.all(np.isfinite(array)):
        raise Invalid(name.strip() + " must be finite")
    return array


def reliability(probabilities, outcomes, bins=10):
    """ECE, MCE, Brier score with its Murphy decomposition, and the reliability table.

    The decomposition is ``BS = reliability - resolution + uncertainty`` evaluated on the
    supplied equal-width binning (Murphy 1973). It is exact only when the forecast is constant
    within each bin, so ``decomposition_residual`` reports the discrepancy rather than hiding
    it. Bins are ``[k/B, (k+1)/B)`` with the top bin closed at 1.
    """
    predicted = _probabilities(probabilities)
    observed = _outcomes(outcomes, predicted.size)
    count = _bounded_int(bins, "bins", 1, MAX_BINS)
    total = predicted.size
    edges = np.linspace(0.0, 1.0, count + 1)
    assignment = np.clip(np.floor(predicted * count).astype(int), 0, count - 1)
    base_rate = float(observed.mean())
    brier = float(np.mean((predicted - observed) ** 2))

    table = []
    ece = 0.0
    mce = 0.0
    reliability_term = 0.0
    resolution_term = 0.0
    for index in range(count):
        mask = assignment == index
        size = int(mask.sum())
        entry = {
            "bin": index,
            "lower_edge": float(edges[index]),
            "upper_edge": float(edges[index + 1]),
            "count": size,
            "mean_prediction": None,
            "observed_rate": None,
            "gap": None,
        }
        if size:
            mean_prediction = float(predicted[mask].mean())
            observed_rate = float(observed[mask].mean())
            gap = mean_prediction - observed_rate
            weight = size / total
            ece += weight * abs(gap)
            mce = max(mce, abs(gap))
            reliability_term += weight * gap**2
            resolution_term += weight * (observed_rate - base_rate) ** 2
            entry.update(
                {
                    "mean_prediction": mean_prediction,
                    "observed_rate": observed_rate,
                    "gap": float(gap),
                }
            )
        table.append(entry)
    uncertainty_term = base_rate * (1.0 - base_rate)
    decomposed = reliability_term - resolution_term + uncertainty_term
    return {
        "brier": brier,
        "ece": float(ece),
        "mce": float(mce),
        "reliability_component": float(reliability_term),
        "resolution_component": float(resolution_term),
        "uncertainty_component": float(uncertainty_term),
        "decomposed_brier": float(decomposed),
        "decomposition_residual": float(brier - decomposed),
        "base_rate": base_rate,
        "mean_prediction": float(predicted.mean()),
        "reliability_table": table,
        "occupied_bins": int(sum(1 for entry in table if entry["count"])),
        "bins": count,
        "observations": total,
        "scope": (
            "Descriptive calibration of the supplied sample under the supplied equal-width "
            "binning. ECE and MCE are binning-dependent and biased downward when bins are "
            "sparsely populated; MCE in particular can be driven by a single bin, so read "
            "reliability_table counts before quoting either. The Murphy decomposition is exact "
            "only for constant forecasts within a bin and decomposition_residual reports the "
            "gap. Observations are treated as independent units: correlated or repeated events "
            "make every quantity here optimistic. No significance or out-of-sample claim."
        ),
    }


def crps_ensemble(forecast_ensemble, observation):
    """Continuous ranked probability score from the exact empirical formula.

    ``CRPS = mean|x_i - y| - 0.5 * mean_{i,j}|x_i - x_j|`` over ensemble members ``x``. For a
    point-mass ensemble the second term vanishes and the score reduces to the absolute error,
    which is the property that makes CRPS a proper generalization of it.
    """
    ensemble = _ensemble(forecast_ensemble, "forecast_ensemble")
    members, columns = ensemble.shape
    truth = _vector(observation, "observation", MAX_OBSERVATIONS)
    if truth.size != columns:
        raise Invalid("observation must align with the ensemble's observation axis")
    ordered = np.sort(ensemble, axis=0)
    weights = (2 * np.arange(members) - members + 1).reshape(-1, 1)
    spread = np.sum(weights * ordered, axis=0) / (members**2)
    accuracy = np.mean(np.abs(ensemble - truth[None, :]), axis=0)
    scores = accuracy - spread
    return {
        "crps": float(scores.mean()),
        "crps_per_observation": scores.tolist() if scores.size <= MAX_MATRIX_ENTRIES else None,
        "per_observation_omitted": scores.size > MAX_MATRIX_ENTRIES,
        "mean_absolute_error_term": float(accuracy.mean()),
        "ensemble_spread_term": float(spread.mean()),
        "members": int(members),
        "observations": int(columns),
        "scope": (
            "The empirical CRPS of the supplied finite ensemble. A finite ensemble is a biased "
            "estimator of the CRPS of the underlying predictive distribution and the bias "
            "shrinks with member count, so scores are comparable only between ensembles of the "
            "same size. Lower is better within one comparison; the absolute value carries the "
            "units of the observation and is not interpretable on its own."
        ),
    }


def pit_coverage(ensemble, observations, levels=DEFAULT_LEVELS, bins=10):
    """PIT histogram, Kolmogorov-Smirnov uniformity statistic and nominal-vs-empirical coverage.

    The PIT value of an observation is the ensemble fraction below it, with ties split evenly.
    Under a correctly specified forecast these values are uniform, so systematic departure
    diagnoses the failure: a U shape means the ensemble is too narrow, a dome means too wide, a
    slope means bias. Coverage compares each declared central interval's nominal level with the
    fraction of observations that actually fall inside it.
    """
    members_matrix = _ensemble(ensemble, "ensemble")
    members, columns = members_matrix.shape
    truth = _vector(observations, "observations", MAX_OBSERVATIONS)
    if truth.size != columns:
        raise Invalid("observations must align with the ensemble's observation axis")
    count = _bounded_int(bins, "bins", 2, MAX_BINS)
    declared = [_unit_interval(level, "levels") for level in list(levels)]
    if not 1 <= len(declared) <= 10 or len(set(declared)) != len(declared):
        raise Invalid("Supply 1 to 10 unique coverage levels strictly inside (0, 1)")

    below = np.sum(members_matrix < truth[None, :], axis=0)
    equal = np.sum(members_matrix == truth[None, :], axis=0)
    pit = (below + 0.5 * equal) / members
    test = stats.kstest(pit, "uniform")
    histogram, edges = np.histogram(pit, bins=count, range=(0.0, 1.0))

    coverage = []
    for level in declared:
        tail = (1.0 - level) / 2
        low = np.quantile(members_matrix, tail, axis=0)
        high = np.quantile(members_matrix, 1.0 - tail, axis=0)
        inside = float(np.mean((truth >= low) & (truth <= high)))
        coverage.append(
            {
                "nominal": level,
                "empirical": inside,
                "difference": float(inside - level),
                "mean_width": float(np.mean(high - low)),
            }
        )
    return {
        "pit": pit.tolist() if pit.size <= MAX_MATRIX_ENTRIES else None,
        "pit_omitted": pit.size > MAX_MATRIX_ENTRIES,
        "pit_histogram": [
            {
                "bin": index,
                "lower_edge": float(edges[index]),
                "upper_edge": float(edges[index + 1]),
                "count": int(histogram[index]),
                "frequency": float(histogram[index] / pit.size),
            }
            for index in range(count)
        ],
        "pit_mean": float(pit.mean()),
        "ks_statistic": float(test.statistic),
        "ks_pvalue": float(test.pvalue),
        "uniform_rejected_at_5_percent": bool(test.pvalue < 0.05),
        "coverage": coverage,
        "members": int(members),
        "observations": int(columns),
        "bins": count,
        "scope": (
            "PIT uniformity is necessary for calibration, not sufficient: a forecast can be "
            "uniform in PIT and still have no resolution. The KS test assumes independent "
            "observations and a continuous predictive distribution; a finite ensemble makes PIT "
            "discrete, which biases the statistic upward for small member counts, and serially "
            "correlated observations invalidate the p-value entirely. A non-rejection is not "
            "evidence of calibration, only an absence of detected departure at this sample size."
        ),
    }


def interval_score(lower, upper, observation, alpha):
    """Winkler interval score for a central ``1 - alpha`` prediction interval.

    ``IS = (u - l) + (2/alpha)(l - y) 1{y < l} + (2/alpha)(y - u) 1{y > u}``. The score is
    proper: it rewards sharpness through the width term while the penalty terms punish
    intervals that miss, so it cannot be improved by reporting an interval one does not
    believe. Lower is better and the units are those of the observation.
    """
    low = _vector(lower, "lower", MAX_OBSERVATIONS)
    high = _vector(upper, "upper", MAX_OBSERVATIONS)
    truth = _vector(observation, "observation", MAX_OBSERVATIONS)
    if not low.size == high.size == truth.size:
        raise Invalid("lower, upper and observation must have the same length")
    if np.any(high < low):
        raise Invalid("upper must be greater than or equal to lower")
    level = _unit_interval(alpha, "alpha")
    width = high - low
    under = (2.0 / level) * np.clip(low - truth, 0.0, None)
    over = (2.0 / level) * np.clip(truth - high, 0.0, None)
    scores = width + under + over
    inside = (truth >= low) & (truth <= high)
    return {
        "interval_score": float(scores.mean()),
        "scores": scores.tolist() if scores.size <= MAX_MATRIX_ENTRIES else None,
        "scores_omitted": scores.size > MAX_MATRIX_ENTRIES,
        "sharpness": float(width.mean()),
        "underprediction_penalty": float(under.mean()),
        "overprediction_penalty": float(over.mean()),
        "empirical_coverage": float(inside.mean()),
        "nominal_coverage": float(1.0 - level),
        "coverage_difference": float(inside.mean() - (1.0 - level)),
        "misses_below": int(np.sum(truth < low)),
        "misses_above": int(np.sum(truth > high)),
        "alpha": level,
        "observations": int(truth.size),
        "scope": (
            "A proper score on the supplied sample, in the units of the observation. It is a "
            "relative measure: comparable between forecasters on the same observations, not "
            "interpretable on its own. Empirical coverage on a finite sample carries binomial "
            "noise and correlated observations make it optimistic; matching nominal coverage "
            "does not by itself establish that the intervals are well calibrated elsewhere."
        ),
    }


def _lag1_autocorrelation(values):
    if values.size < 3:
        raise Invalid("lag-1 autocorrelation needs at least three ordered values")
    centered = values - values.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 0:
        return 0.0
    return float(np.dot(centered[:-1], centered[1:]) / denominator)


STATISTICS = {
    "mean": lambda v: float(np.mean(v)),
    "variance": lambda v: float(np.var(v, ddof=1)) if v.size > 1 else 0.0,
    "standard_deviation": lambda v: float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
    "max": lambda v: float(np.max(v)),
    "min": lambda v: float(np.min(v)),
    "range": lambda v: float(np.max(v) - np.min(v)),
    "lag1_autocorrelation": _lag1_autocorrelation,
}


def _statistics(requested):
    if isinstance(requested, dict):
        items = list(requested.items())
        for name, function in items:
            if not isinstance(name, str) or not name.strip() or not callable(function):
                raise Invalid("statistics mapping must be name -> callable(ndarray) -> float")
    else:
        items = []
        for name in list(requested):
            if not isinstance(name, str) or name not in STATISTICS:
                raise Invalid(
                    "Unknown summary statistic; supply a name from "
                    + ", ".join(sorted(STATISTICS))
                    + " or a mapping of name -> callable"
                )
            items.append((name, STATISTICS[name]))
    if not 1 <= len(items) <= MAX_STATISTICS:
        raise Invalid(f"Supply between 1 and {MAX_STATISTICS} summary statistics")
    if len({name for name, _ in items}) != len(items):
        raise Invalid("Summary statistic names must be unique")
    return items


def posterior_predictive_check(
    simulated_replicates,
    observations,
    statistics=("mean", "variance", "max", "lag1_autocorrelation"),
):
    """Bayesian p-values for caller-supplied summary statistics of the replicated data.

    Each statistic is evaluated on the observations and on every replicate; the p-value is
    ``P(T(replicate) >= T(observed))`` estimated by the replicate fraction. A value near 0 or 1
    means the model reproduces that feature of the data poorly. Values below 0.05 or above 0.95
    are flagged as model-discrepancy signals, which locates a mismatch and does not measure it.
    """
    replicates = _matrix(simulated_replicates, "simulated_replicates", MAX_REPLICATES, MAX_OBSERVATIONS)
    truth = _vector(observations, "observations", MAX_OBSERVATIONS)
    if replicates.shape[1] != truth.size:
        raise Invalid("Each replicate must align one-to-one with the observations")
    chosen = _statistics(statistics)

    def evaluate(function, name, values):
        try:
            scalar = float(function(values))
        except Invalid:
            raise
        except Exception as exc:
            raise Invalid(f"Summary statistic {name!r} failed: {exc!r}") from exc
        if not np.isfinite(scalar):
            raise Invalid(f"Summary statistic {name!r} returned a nonfinite value")
        return scalar

    results = []
    for name, function in chosen:
        observed = evaluate(function, name, truth)
        replicated = np.array(
            [evaluate(function, name, replicates[index]) for index in range(replicates.shape[0])]
        )
        pvalue = float(np.mean(replicated >= observed))
        tail = float(min(pvalue, 1.0 - pvalue))
        results.append(
            {
                "statistic": name,
                "observed": observed,
                "replicate_mean": float(replicated.mean()),
                "replicate_standard_deviation": float(replicated.std(ddof=1))
                if replicated.size > 1
                else 0.0,
                "bayesian_pvalue": pvalue,
                "tail_probability": tail,
                "discrepancy": bool(pvalue < 0.05 or pvalue > 0.95),
                "direction": "observed_above_replicates"
                if pvalue < 0.05
                else "observed_below_replicates"
                if pvalue > 0.95
                else "not_flagged",
                "replicate_values": _listed(replicated),
            }
        )
    flagged = [entry["statistic"] for entry in results if entry["discrepancy"]]
    return {
        "checks": results,
        "flagged_statistics": flagged,
        "discrepancy_detected": bool(flagged),
        "replicates": int(replicates.shape[0]),
        "observations": int(truth.size),
        "statistic_names": [name for name, _ in chosen],
        "scope": (
            "Posterior predictive p-values are conservative: the same data enter the fit and "
            "the check, so they are stochastically non-uniform under a correct model and must "
            "not be read as frequentist tail probabilities. A flagged statistic locates a "
            "feature the model fails to reproduce; an unflagged one is not evidence of "
            "adequacy, only that this statistic did not detect a mismatch at this replicate "
            "count. The p-value resolution is limited to 1/replicates."
        ),
    }
