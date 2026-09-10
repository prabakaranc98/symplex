"""Model discrepancy: what the model gets systematically wrong, and where.

Kennedy and O'Hagan, "Bayesian calibration of computer models", J. R. Statist. Soc. B
63(3):425-464, 2001. The observation is modelled as

    y(x) = f(x, theta) + delta(x) + epsilon

where `delta` is a smooth systematic discrepancy between the model and reality and
`epsilon` is independent observation error. `delta` is fitted here as a Gaussian process
over the residual, so that a residual with structure is separated from a residual that is
only noise.

The central caveat is Kennedy and O'Hagan's own: `delta` and `theta` are not jointly
identifiable from a single dataset. A discrepancy term can absorb a parameter error, and a
recalibrated parameter can absorb a missing mechanism. The split reported here is the split
implied by the assumed covariance and the supplied noise information, not a measurement of
which is which. `decompose_error` labels its residual bucket rather than distributing it.

The reason to run `discrepancy_diagnosis` at all is that a *structured* residual is a
pointer to a missing mechanism: it says the model is wrong in a particular region and in a
particular direction, which is a lead to model next. An unstructured residual is consistent
with noise and is not evidence that the model is right.
"""

import math

import numpy as np
from scipy.stats import chi2, norm

from symplex.core.contracts import Invalid
from symplex.hybrid.surrogate import MAX_GP_POINTS, GPEmulator

MAX_SAMPLES = 200_000
MAX_INPUT_DIMENSION = 32
MAX_ENSEMBLE = 2_000
MAX_BINS = 32
MAX_LAG = 50

SCOPE = (
    "A decomposition implied by the assumed discrepancy covariance and the supplied noise "
    "information. Discrepancy, parameter error and observation noise are not separately "
    "identifiable from one dataset; the split reported is a modelling consequence, not a "
    "measurement."
)


def _vector(values, label, length=None):
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise Invalid(f"{label} must be nonempty and finite")
    if array.size > MAX_SAMPLES:
        raise Invalid(f"{label} exceeds the {MAX_SAMPLES}-sample limit")
    if length is not None and array.size != length:
        raise Invalid(f"{label} must hold {length} values")
    return array


def _inputs(values, length):
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.shape[0] != length or not np.all(np.isfinite(array)):
        raise Invalid("inputs must be a finite 2-D array with one row per observation")
    if array.shape[1] > MAX_INPUT_DIMENSION:
        raise Invalid(f"inputs exceed the {MAX_INPUT_DIMENSION}-dimension limit")
    return array


def _integer(value, label, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise Invalid(f"{label} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise Invalid(f"{label} must lie in [{low}, {high}]")
    return value


def _number(value, label, low=None):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise Invalid(f"{label} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise Invalid(f"{label} must be finite")
    if low is not None and value < low:
        raise Invalid(f"{label} must be at least {low}")
    return value


def _runs_test(residuals):
    """Wald-Wolfowitz runs test on residual signs, in the supplied order."""
    signs = np.sign(residuals)
    signs = signs[signs != 0]
    n = signs.size
    positives = int(np.sum(signs > 0))
    negatives = n - positives
    if positives == 0 or negatives == 0 or n < 8:
        return {
            "runs": None,
            "expected_runs": None,
            "z": None,
            "p_value": None,
            "applicable": False,
            "reason": "Too few samples, or all residuals share one sign.",
        }
    runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
    expected = 2.0 * positives * negatives / n + 1.0
    variance = (
        2.0 * positives * negatives * (2.0 * positives * negatives - n)
        / (n * n * (n - 1.0))
    )
    if variance <= 0:
        return {
            "runs": runs,
            "expected_runs": float(expected),
            "z": None,
            "p_value": None,
            "applicable": False,
            "reason": "Degenerate variance for the runs statistic.",
        }
    z = (runs - expected) / math.sqrt(variance)
    return {
        "runs": runs,
        "expected_runs": float(expected),
        "z": float(z),
        "p_value": float(2.0 * norm.sf(abs(z))),
        "applicable": True,
        "reason": "",
    }


def _ljung_box(residuals, max_lag):
    n = residuals.size
    centred = residuals - residuals.mean()
    denominator = float(centred @ centred)
    lags = min(max_lag, max(1, n // 5))
    if n < 12 or denominator <= 0:
        return {
            "statistic": None,
            "degrees_of_freedom": None,
            "p_value": None,
            "lag_1_autocorrelation": None,
            "applicable": False,
            "reason": "Too few samples, or a residual series with no variation.",
        }
    correlations = []
    for lag in range(1, lags + 1):
        correlations.append(float(centred[lag:] @ centred[:-lag] / denominator))
    statistic = n * (n + 2.0) * sum(
        (rho * rho) / (n - lag) for lag, rho in enumerate(correlations, start=1)
    )
    return {
        "statistic": float(statistic),
        "degrees_of_freedom": int(lags),
        "p_value": float(chi2.sf(statistic, lags)),
        "lag_1_autocorrelation": correlations[0],
        "autocorrelations": correlations[:10],
        "applicable": True,
        "reason": "",
    }


def _binned_means(residuals, ordering_values, n_bins):
    n = residuals.size
    bins = max(2, min(n_bins, n // 5))
    edges = np.quantile(ordering_values, np.linspace(0.0, 1.0, bins + 1))
    edges[0] -= 1e-12
    edges[-1] += 1e-12
    assignment = np.clip(np.searchsorted(edges, ordering_values, side="left") - 1, 0, bins - 1)
    entries = []
    for index in range(bins):
        mask = assignment == index
        count = int(mask.sum())
        if count < 3:
            continue
        block = residuals[mask]
        mean = float(block.mean())
        spread = float(block.std(ddof=1))
        statistic = mean / (spread / math.sqrt(count)) if spread > 0 else None
        entries.append(
            {
                "bin": index,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "n": count,
                "mean_residual": mean,
                "sd_residual": spread,
                "t_statistic": None if statistic is None else float(statistic),
                "abs_mean_over_global_rms": float(
                    abs(mean) / math.sqrt(float(residuals @ residuals) / n)
                )
                if float(residuals @ residuals) > 0
                else 0.0,
            }
        )
    return entries


def discrepancy_diagnosis(residuals, inputs=None, n_bins=8, alpha=0.01, max_lag=10, order_by=0):
    """Is the residual structured (a missing mechanism) or unstructured (noise)?

    Two deterministic tests on the residual sequence, ordered by an input coordinate when
    one is supplied and otherwise in the order given:

    * a Wald-Wolfowitz runs test on residual signs - long stretches of one sign mean the
      model is biased over a region of input space;
    * a Ljung-Box portmanteau test on the residual autocorrelations - dependence between
      neighbouring residuals means something varying with the input is unmodelled.

    A `structured` verdict is the useful one: the binned means localize where the model is
    wrong and in which direction, and that is the lead to model next. An `unstructured`
    verdict means these tests found no pattern at this sample size; it is not evidence that
    the model is correct, and structure orthogonal to the chosen ordering will be missed.
    """
    residuals = _vector(residuals, "residuals")
    alpha = _number(alpha, "alpha", low=0.0)
    if not 0.0 < alpha < 1.0:
        raise Invalid("alpha must lie strictly between 0 and 1")
    n_bins = _integer(n_bins, "n_bins", 2, MAX_BINS)
    max_lag = _integer(max_lag, "max_lag", 1, MAX_LAG)

    if inputs is None:
        ordering_values = np.arange(residuals.size, dtype=float)
        ordering_label = "supplied order"
    else:
        points = _inputs(inputs, residuals.size)
        order_by = _integer(order_by, "order_by", 0, points.shape[1] - 1)
        ordering_values = points[:, order_by]
        ordering_label = f"input dimension {order_by}"
    order = np.argsort(ordering_values, kind="stable")
    ordered = residuals[order]
    ordered_values = ordering_values[order]

    runs = _runs_test(ordered)
    ljung = _ljung_box(ordered, max_lag)
    bins = _binned_means(ordered, ordered_values, n_bins)
    significant = [
        test["p_value"]
        for test in (runs, ljung)
        if test["applicable"] and test["p_value"] is not None
    ]
    structured = any(p < alpha for p in significant)
    worst = max(bins, key=lambda b: abs(b["mean_residual"]), default=None)
    pointer = None
    if structured and worst is not None:
        direction = "above" if worst["mean_residual"] > 0 else "below"
        pointer = {
            "ordering": ordering_label,
            "region_lower": worst["lower"],
            "region_upper": worst["upper"],
            "mean_residual": worst["mean_residual"],
            "direction": f"observations lie {direction} the model in this region",
            "relative_size": worst["abs_mean_over_global_rms"],
            "next_step": (
                "Look for a mechanism that acts in this region and in this direction, add it to "
                "the model, and check whether the structure disappears. Refitting existing "
                "parameters to remove the pattern would move the misfit rather than explain it."
            ),
        }
    return {
        "verdict": "structured" if structured else "unstructured",
        "structured": bool(structured),
        "alpha": alpha,
        "runs_test": runs,
        "ljung_box_test": ljung,
        "binned_means": bins,
        "worst_region": worst,
        "missing_mechanism_pointer": pointer,
        "ordering": ordering_label,
        "n": int(residuals.size),
        "residual_rms": float(np.sqrt(float(residuals @ residuals) / residuals.size)),
        "residual_mean": float(residuals.mean()),
        "interpretation": (
            "Structured residual: the model is systematically wrong over part of the input "
            "range, which points at a missing mechanism there rather than at noise."
            if structured
            else "Unstructured residual: no pattern detected by these tests at this sample "
            "size. Consistent with observation noise; not evidence that the model is correct."
        ),
        "scope": (
            "Two order-dependent tests on one residual sequence. They detect structure along the "
            "chosen ordering only, they assume the residuals are comparable in scale across that "
            "ordering, and failing to reject is not acceptance."
        ),
    }


def fit_discrepancy(
    model_predictions,
    observations,
    inputs,
    kernel="matern52",
    noise_variance=None,
    max_points=MAX_GP_POINTS,
    seed=2026,
    n_restarts=2,
    max_iterations=200,
):
    """Fit a Gaussian process over the systematic residual and separate it from noise.

    The GP signal variance carries the smooth, input-dependent part of the residual - the
    discrepancy - while the fitted nugget carries the part that varies independently between
    neighbouring inputs. That separation is made by the covariance assumption: with a short
    enough length scale a discrepancy is indistinguishable from noise, and `noise_variance`
    should be supplied whenever the observation error is actually known, because doing so is
    the only thing that makes the split anything more than an assumption.
    """
    predictions = _vector(model_predictions, "model_predictions")
    observed = _vector(observations, "observations", length=predictions.size)
    points = _inputs(inputs, predictions.size)
    max_points = _integer(max_points, "max_points", 4, MAX_GP_POINTS)
    residual = observed - predictions

    stride, note = 1, ""
    if residual.size > max_points:
        stride = int(np.ceil(residual.size / max_points))
        note = (
            f"Fitted on every {stride}th sample: a Gaussian process factorizes an n-by-n "
            f"covariance matrix, so training is capped at {max_points} points."
        )
    training_inputs = points[::stride]
    training_residual = residual[::stride]

    emulator = GPEmulator(
        kernel=kernel,
        nugget=1e-4,
        n_restarts=n_restarts,
        max_iterations=max_iterations,
        seed=seed,
    )
    fit = emulator.fit(
        training_inputs,
        training_residual,
        fixed_noise_variance=None if noise_variance is None else _number(noise_variance, "noise_variance", low=0.0),
    )
    posterior = emulator.predict(points)
    delta = np.asarray(posterior["mean"], dtype=float)
    corrected = predictions + delta
    signal = float(fit["signal_variance"])
    estimated_noise = float(fit["nugget"])
    total = signal + estimated_noise
    diagnosis = discrepancy_diagnosis(residual, inputs=points)
    warnings = list(fit["warnings"])
    if note:
        warnings.append(note)
    if noise_variance is None:
        warnings.append(
            "Observation noise was estimated from the same residuals as the discrepancy. "
            "Without independent noise information or replicates the split between them is "
            "set by the kernel, not by the data."
        )
    return {
        "discrepancy_mean": delta,
        "discrepancy_variance": np.asarray(posterior["variance"], dtype=float),
        "corrected_predictions": corrected,
        "residual": residual,
        "signal_variance": signal,
        "noise_variance": estimated_noise,
        "noise_variance_supplied": bool(fit["nugget_fixed"]),
        "signal_to_noise": float(signal / estimated_noise) if estimated_noise > 0 else None,
        "structured_fraction": float(signal / total) if total > 0 else None,
        "rmse_before": float(np.sqrt(np.mean(residual**2))),
        "rmse_after": float(np.sqrt(np.mean((observed - corrected) ** 2))),
        "discrepancy_rms": float(np.sqrt(np.mean(delta**2))),
        "length_scale_input_units": fit["length_scale_input_units"],
        "kernel": kernel,
        "n_observations": int(residual.size),
        "n_training_points": int(training_residual.size),
        "training_stride": stride,
        "converged": fit["converged"],
        "log_marginal_likelihood": fit["log_marginal_likelihood"],
        "support": fit["support"],
        "diagnosis": diagnosis,
        "warnings": warnings,
        "scope": SCOPE
        + " rmse_after is an in-sample figure: the discrepancy was fitted on these residuals, "
        "so it will always look better here than on held-out data.",
    }


def decompose_error(
    model_predictions,
    observations,
    inputs,
    parameter_ensemble=None,
    noise_variance=None,
    replicate_groups=None,
    kernel="matern52",
    seed=2026,
):
    """Split total squared error into observation noise, parameter uncertainty and discrepancy.

    The audit asks for exactly this separation, and the honest version of it has four
    buckets, not three:

    * **observation noise** - supplied, or estimated from replicate groups when the same
      input was measured more than once, or otherwise taken from the discrepancy fit's
      nugget, which is the weakest of the three and is labelled as such;
    * **parameter uncertainty** - the mean predictive variance across a supplied ensemble of
      predictions under sampled parameters. Without an ensemble this is reported as absent,
      not as zero;
    * **structural discrepancy** - the mean square of the fitted systematic residual;
    * **unattributed** - total minus the other three. It can come out negative, because the
      three sources are not orthogonal and the buckets overlap. That is reported as it comes
      rather than being clipped, since a large negative value means the decomposition's
      assumptions do not hold for these data.

    Ignorance is not a bucket here. It is the reason the other four cannot be trusted to sum.
    """
    predictions = _vector(model_predictions, "model_predictions")
    observed = _vector(observations, "observations", length=predictions.size)
    points = _inputs(inputs, predictions.size)
    residual = observed - predictions
    total = float(np.mean(residual**2))

    noise_source, noise = "unavailable", None
    if noise_variance is not None:
        noise = _number(noise_variance, "noise_variance", low=0.0)
        noise_source = "supplied by the caller"
    elif replicate_groups is not None:
        groups = np.asarray(replicate_groups).reshape(-1)
        if groups.size != residual.size:
            raise Invalid("replicate_groups must hold one label per observation")
        within, degrees = 0.0, 0
        for label in np.unique(groups):
            block = observed[groups == label]
            if block.size > 1:
                within += float(((block - block.mean()) ** 2).sum())
                degrees += block.size - 1
        if degrees > 0:
            noise = within / degrees
            noise_source = f"pooled within-replicate variance over {degrees} degrees of freedom"

    parameter_variance, ensemble_size = None, 0
    if parameter_ensemble is not None:
        ensemble = np.asarray(parameter_ensemble, dtype=float)
        if ensemble.ndim != 2 or ensemble.shape[1] != residual.size:
            raise Invalid("parameter_ensemble must be shaped (n_parameter_draws, n_observations)")
        if ensemble.shape[0] < 2 or ensemble.shape[0] > MAX_ENSEMBLE:
            raise Invalid(f"parameter_ensemble must hold 2 to {MAX_ENSEMBLE} draws")
        if not np.all(np.isfinite(ensemble)):
            raise Invalid("parameter_ensemble must be finite")
        ensemble_size = int(ensemble.shape[0])
        parameter_variance = float(np.mean(ensemble.var(axis=0, ddof=1)))

    discrepancy = fit_discrepancy(
        predictions,
        observed,
        points,
        kernel=kernel,
        noise_variance=noise,
        seed=seed,
    )
    structural = float(np.mean(np.asarray(discrepancy["discrepancy_mean"]) ** 2))
    if noise is None:
        noise = float(discrepancy["noise_variance"])
        noise_source = "fitted nugget of the discrepancy GP (weakest available basis)"

    components = {
        "observation_noise": float(noise),
        "parameter_uncertainty": parameter_variance,
        "structural_discrepancy": structural,
    }
    attributed = float(noise) + structural + (parameter_variance or 0.0)
    unattributed = total - attributed
    fractions = {
        key: (None if value is None or total <= 0 else float(value / total))
        for key, value in components.items()
    }
    fractions["unattributed"] = None if total <= 0 else float(unattributed / total)
    named = {k: v for k, v in components.items() if v is not None}
    dominant = max(named, key=named.get) if named else None
    warnings = list(discrepancy["warnings"])
    if parameter_variance is None:
        warnings.append(
            "No parameter ensemble was supplied, so parameter uncertainty is reported as "
            "unavailable and its share is inside the unattributed bucket, not zero."
        )
    if total > 0 and unattributed < -0.1 * total:
        warnings.append(
            "The attributed components exceed the total error by more than 10 percent, so they "
            "overlap: the same misfit is being counted by more than one source and the split "
            "should not be reported as a partition."
        )
    return {
        "total_mean_squared_error": total,
        "total_rmse": float(math.sqrt(total)),
        "components": components,
        "unattributed": float(unattributed),
        "fractions": fractions,
        "dominant_identified_source": dominant,
        "observation_noise_source": noise_source,
        "parameter_ensemble_size": ensemble_size,
        "structured_residual": discrepancy["diagnosis"]["structured"],
        "diagnosis": discrepancy["diagnosis"],
        "discrepancy_rms": discrepancy["discrepancy_rms"],
        "n_observations": int(residual.size),
        "warnings": warnings,
        "scope": SCOPE
        + " The buckets are not orthogonal and need not sum to the total; the unattributed "
        "remainder carries that overlap and any source not represented at all.",
    }
