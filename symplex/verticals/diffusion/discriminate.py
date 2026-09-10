"""Mechanism discrimination, and an honest account of when it is impossible.

Contagion, homophily and a shared exogenous shock are famously hard to tell
apart, and a study that reports a winner without asking whether its observations
*could* have distinguished the rivals has reported the prior, not the data. This
module therefore treats "these two mechanisms are not separable with what you
have" as a first-class result rather than a failure.

The four public entry points:

- `fit_mechanisms` scores every rival against one observed cascade and returns
  goodness of fit, AIC/BIC and Akaike weights over the candidate set.
- `identifiability_report` asks, before believing any of that, whether the
  available observations can separate the rivals at all, pair by pair.
- `discriminating_observation` turns each non-separable pair into a concrete
  next measurement and predicts how much separation it would buy.
- `retained_ensemble` keeps every mechanism the data cannot reject, and refuses
  to collapse to a single winner when the observations cannot support one.

Nothing here estimates a real diffusion process. Fitting a mechanism to
simulated data demonstrates inference mechanics; model weights are conditional on
the candidate set considered, which is never exhaustive.
"""

from __future__ import annotations

import dataclasses
import itertools
import math

import numpy as np

from symplex.core.contracts import Invalid
from symplex.verticals.diffusion import mechanisms as mech
from symplex.verticals.diffusion.networks import Network

MAX_FIT_CANDIDATES = 8
MAX_GRID_POINTS = 96
MAX_FIT_SIMULATIONS = 6000
MAX_IDENTIFIABILITY_REPLICATES = 120

OBSERVATION_CHANNELS = (
    "adoption_curve",
    "adoption_times",
    "network",
    "exposure_log",
    "community_labels",
    "node_attributes",
)

CHANNEL_MEANING = {
    "adoption_curve": "counts of new adoptions per time step, with no node identity",
    "adoption_times": "the time at which each identified node adopted",
    "network": "the adjacency structure over the same identified nodes",
    "exposure_log": "how many active neighbours each adopter had at the moment it adopted",
    "community_labels": "a community assignment for each node",
    "node_attributes": "a node-level covariate that could drive adoption without influence",
}

CHANNEL_COST = {
    "adoption_curve": "usually already available from any aggregate counter",
    "adoption_times": "requires per-node event logging with resolved identities",
    "network": "requires the interaction graph over the same identities, at the same time",
    "exposure_log": "requires edge-level exposure instrumentation, the most expensive channel",
    "community_labels": "requires a partition, either observed or estimated with its own error",
    "node_attributes": "requires a covariate measured independently of the outcome",
}

FIT_SCOPE = (
    "goodness of fit of a proposed mechanism to one observed cascade under a "
    "simulation-based (synthetic) likelihood over summary statistics; a fitted "
    "mechanism on simulated data demonstrates inference mechanics, not the truth "
    "of any real diffusion process, and every weight is conditional on the "
    "candidate set considered, which is never exhaustive"
)
IDENTIFIABILITY_SCOPE = (
    "separability of simulated statistic distributions under the stated "
    "observations, parameters, seed set and replicate count; a pair reported as "
    "separable here is separable for these parameter values only, and a pair "
    "reported as not separable may still be separable under different parameters, "
    "more replicates or observations not listed; separability is measured by a "
    "linear discriminant over standardized statistics plus the best single-statistic "
    "AUC, so rivals differing only in the dispersion of a statistic are reported as "
    "not separable by this test"
)


# --------------------------------------------------------------------------- #
# Discriminating statistics
# --------------------------------------------------------------------------- #


def _corr(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.size < 3 or first.std() < 1e-12 or second.std() < 1e-12:
        return 0.0
    value = float(np.corrcoef(first, second)[0, 1])
    return value if math.isfinite(value) else 0.0


def _adopted(result):
    return np.isfinite(result.activation_time)


def _non_seed_adopted(result):
    mask = _adopted(result).copy()
    mask[result.seeds] = False
    return mask


def _final_fraction(result, network):
    return result.final_fraction


def _time_to_peak(result, network):
    if result.increments.size == 0 or result.increments.max() <= 0:
        return 0.0
    return float(np.argmax(result.increments) + 1)


def _peak_increment_fraction(result, network):
    if result.increments.size == 0:
        return 0.0
    return float(result.increments.max() / result.nodes)


def _early_growth_rate(result, network):
    curve = result.size_trajectory.astype(np.float64)
    window = max(1, curve.size // 4)
    return float(math.log((curve[window] + 1.0) / (curve[0] + 1.0)) / window)


def _half_time(result, network):
    curve = result.size_trajectory
    if curve[-1] <= 0:
        return 0.0
    return float(int(np.argmax(curve >= curve[-1] / 2.0)))


def _adoption_time_dispersion(result, network):
    times = result.activation_time[_adopted(result)]
    return float(times.std()) if times.size > 1 else 0.0


def _degree_adoption_correlation(result, network):
    mask = _non_seed_adopted(result)
    if mask.sum() < 4:
        return 0.0
    return _corr(network.degree[mask], result.activation_time[mask])


def _neighbour_preceded_fraction(result, network):
    mask = _non_seed_adopted(result)
    if not mask.any():
        return 0.0
    return float(result.neighbour_preceded[mask].mean())


def _edge_time_concordance(result, network):
    if network.edge_count == 0:
        return 0.0
    times = result.activation_time
    left, right = network.edges[:, 0], network.edges[:, 1]
    both = np.isfinite(times[left]) & np.isfinite(times[right])
    if both.sum() < 4:
        return 0.0
    first = np.concatenate([times[left][both], times[right][both]])
    second = np.concatenate([times[right][both], times[left][both]])
    return _corr(first, second)


def _mean_exposures_at_adoption(result, network):
    mask = _non_seed_adopted(result)
    if not mask.any():
        return 0.0
    values = result.exposures_at_adoption[mask]
    return float(np.nanmean(values)) if np.isfinite(values).any() else 0.0


def _exposure_dispersion_at_adoption(result, network):
    mask = _non_seed_adopted(result)
    values = result.exposures_at_adoption[mask]
    values = values[np.isfinite(values)]
    return float(values.std()) if values.size > 1 else 0.0


def _cross_community_lag(result, network):
    if network.communities is None:
        return 0.0
    mask = _adopted(result)
    medians = []
    for label in np.unique(network.communities):
        member = mask & (network.communities == label)
        if member.sum() >= 2:
            medians.append(float(np.median(result.activation_time[member])))
    if len(medians) < 2:
        return 0.0
    return float(max(medians) - min(medians))


def _attribute_adoption_correlation(result, network):
    if network.attributes is None:
        return 0.0
    mask = _non_seed_adopted(result)
    if mask.sum() < 4:
        return 0.0
    return _corr(network.attributes[mask], result.activation_time[mask])


@dataclasses.dataclass(frozen=True)
class Statistic:
    name: str
    requires: tuple
    meaning: str
    compute: object


STATISTICS = (
    Statistic("final_fraction", ("adoption_curve",), "fraction of nodes that ever adopted", _final_fraction),
    Statistic("time_to_peak", ("adoption_curve",), "step of the largest single-step adoption count", _time_to_peak),
    Statistic("peak_increment_fraction", ("adoption_curve",), "largest single-step adoption count as a fraction of nodes", _peak_increment_fraction),
    Statistic("early_growth_rate", ("adoption_curve",), "log growth of the cumulative curve over its first quarter", _early_growth_rate),
    Statistic("half_time", ("adoption_curve",), "step at which half the eventual adopters had adopted", _half_time),
    Statistic("adoption_time_dispersion", ("adoption_times",), "standard deviation of adoption times", _adoption_time_dispersion),
    Statistic("degree_adoption_correlation", ("adoption_times", "network"), "correlation between degree and adoption time; contagion adopts hubs early, an exogenous driver does not", _degree_adoption_correlation),
    Statistic("neighbour_preceded_fraction", ("adoption_times", "network"), "fraction of adopters with at least one neighbour already active", _neighbour_preceded_fraction),
    Statistic("edge_time_concordance", ("adoption_times", "network"), "correlation of adoption times across edges", _edge_time_concordance),
    Statistic("mean_exposures_at_adoption", ("exposure_log",), "average number of active neighbours at the moment of adoption; the reinforcement signature", _mean_exposures_at_adoption),
    Statistic("exposure_dispersion_at_adoption", ("exposure_log",), "spread of the exposure count at adoption", _exposure_dispersion_at_adoption),
    Statistic("cross_community_lag", ("adoption_times", "community_labels"), "spread of median adoption times across communities", _cross_community_lag),
    Statistic("attribute_adoption_correlation", ("adoption_times", "node_attributes"), "correlation between a node covariate and adoption time; the homophily signature", _attribute_adoption_correlation),
)
STATISTIC_BY_NAME = {s.name: s for s in STATISTICS}


def available_statistics(observations, network=None):
    """Statistics computable from a stated observation set, with the reason for exclusions."""
    channels = _channels(observations, network)
    included, excluded = [], []
    for statistic in STATISTICS:
        if set(statistic.requires) <= channels:
            included.append(statistic.name)
        else:
            excluded.append(
                {
                    "statistic": statistic.name,
                    "missing_channels": sorted(set(statistic.requires) - channels),
                    "meaning": statistic.meaning,
                }
            )
    return {
        "observations": sorted(channels),
        "statistics": included,
        "excluded": excluded,
        "scope": "which summary statistics the stated observations permit; a "
        "statistic being computable does not make it informative",
    }


def _channels(observations, network=None):
    if isinstance(observations, str):
        raise Invalid("observations must be a collection of channel names")
    channels = set(observations or ())
    unknown = channels - set(OBSERVATION_CHANNELS)
    if unknown:
        raise Invalid(
            "unsupported_operation: unknown observation channel(s) "
            + ", ".join(sorted(unknown))
        )
    if not channels:
        raise Invalid("At least one observation channel is required")
    if network is not None:
        if "community_labels" in channels and network.communities is None:
            raise Invalid("community_labels requested but this network carries no labels")
        if "node_attributes" in channels and network.attributes is None:
            raise Invalid("node_attributes requested but this network carries no attributes")
    if "adoption_times" in channels:
        channels.add("adoption_curve")
    if "exposure_log" in channels:
        channels.update({"adoption_times", "adoption_curve", "network"})
    return channels


def statistic_vector(result, network, names):
    values = np.array(
        [float(STATISTIC_BY_NAME[name].compute(result, network)) for name in names],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise Invalid("A discriminating statistic produced a nonfinite value")
    return values


def statistic_matrix(results, network, names):
    return np.vstack([statistic_vector(result, network, names) for result in results])


# --------------------------------------------------------------------------- #
# Candidate handling
# --------------------------------------------------------------------------- #


def _normalize_candidates(candidates, limit=MAX_FIT_CANDIDATES):
    if not isinstance(candidates, (list, tuple)) or not candidates:
        raise Invalid("Supply a nonempty list of candidate mechanisms")
    if len(candidates) > limit:
        raise Invalid(f"At most {limit} candidate mechanisms may be compared at once")
    normalized = []
    for candidate in candidates:
        if isinstance(candidate, str):
            entry = {"mechanism": candidate, "parameters": None, "label": candidate}
        elif isinstance(candidate, dict):
            if "mechanism" not in candidate:
                raise Invalid("A candidate dict must name a mechanism")
            entry = {
                "mechanism": candidate["mechanism"],
                "parameters": candidate.get("parameters"),
                "label": candidate.get("label", candidate["mechanism"]),
            }
        else:
            raise Invalid("Candidates must be mechanism names or dicts")
        mechanism = mech.get_mechanism(entry["mechanism"])
        entry["mechanism"] = mechanism.name
        entry["parameters"] = mechanism.parse(entry["parameters"]).model_dump()
        normalized.append(entry)
    labels = [entry["label"] for entry in normalized]
    if len(set(labels)) != len(labels):
        raise Invalid("Candidate labels must be unique")
    return normalized


def _seed_nodes(seeds, network):
    if seeds is None:
        return [int(np.argmax(network.degree))]
    return list(seeds)


# --------------------------------------------------------------------------- #
# Separability
# --------------------------------------------------------------------------- #


def _auc(first, second):
    """Probability that a draw from `first` exceeds a draw from `second`, ties at 0.5."""
    combined = np.concatenate([first, second])
    order = np.argsort(combined, kind="stable")
    ranks = np.empty(combined.size, dtype=np.float64)
    ranks[order] = np.arange(1, combined.size + 1, dtype=np.float64)
    # Average ranks within ties so exact ties score 0.5.
    values, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
    sums = np.zeros(values.size)
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    rank_sum = ranks[: first.size].sum()
    return float((rank_sum - first.size * (first.size + 1) / 2) / (first.size * second.size))


def _diagonal_gaussian_accuracy(sample_a, sample_b):
    """Two-fold cross-validated accuracy of a diagonal Gaussian classifier."""
    folds = []
    for offset in (0, 1):
        train_a = sample_a[offset::2]
        train_b = sample_b[offset::2]
        test_a = sample_a[1 - offset :: 2]
        test_b = sample_b[1 - offset :: 2]
        if min(train_a.shape[0], train_b.shape[0], test_a.shape[0], test_b.shape[0]) < 2:
            continue
        mean_a, mean_b = train_a.mean(0), train_b.mean(0)
        var = 0.5 * (train_a.var(0, ddof=1) + train_b.var(0, ddof=1))
        var = np.maximum(var, 1e-9)

        def score(points, mean):
            return -0.5 * (((points - mean) ** 2) / var).sum(axis=1)

        correct = int((score(test_a, mean_a) > score(test_a, mean_b)).sum())
        correct += int((score(test_b, mean_b) > score(test_b, mean_a)).sum())
        folds.append(correct / (test_a.shape[0] + test_b.shape[0]))
    return float(np.mean(folds)) if folds else 0.5


def separability(sample_a, sample_b, names, accuracy_threshold=0.80, auc_threshold=0.20):
    """Pairwise separability of two statistic samples.

    Reports both the best single statistic (as a two-sample AUC) and a
    cross-validated multivariate accuracy. A pair counts as separable only when
    both clear their thresholds, so a single noisy statistic cannot carry the
    claim. Chance accuracy is 0.5 and its standard error is roughly
    1/sqrt(replicates), which is why small replicate counts should not be used to
    declare separability.
    """
    sample_a = np.atleast_2d(np.asarray(sample_a, dtype=np.float64))
    sample_b = np.atleast_2d(np.asarray(sample_b, dtype=np.float64))
    if sample_a.shape[1] != sample_b.shape[1] or sample_a.shape[1] != len(names):
        raise Invalid("Statistic samples must share one column per named statistic")
    if min(sample_a.shape[0], sample_b.shape[0]) < 4:
        raise Invalid("Separability needs at least four replicates per mechanism")
    combined = np.vstack([sample_a, sample_b])
    centre = combined.mean(0)
    spread = combined.std(0, ddof=1)
    keep = spread > 1e-12
    per_statistic = []
    for index, name in enumerate(names):
        column_a, column_b = sample_a[:, index], sample_b[:, index]
        pooled = math.sqrt(0.5 * (column_a.var(ddof=1) + column_b.var(ddof=1)))
        area = _auc(column_a, column_b) if keep[index] else 0.5
        per_statistic.append(
            {
                "statistic": name,
                "mean_a": float(column_a.mean()),
                "mean_b": float(column_b.mean()),
                "auc": area,
                "separation": abs(area - 0.5) * 2.0,
                "standardized_difference": float(
                    (column_a.mean() - column_b.mean()) / pooled
                )
                if pooled > 1e-12
                else 0.0,
                "constant_across_both": not bool(keep[index]),
            }
        )
    if not keep.any():
        accuracy = 0.5
    else:
        scaled_a = (sample_a[:, keep] - centre[keep]) / spread[keep]
        scaled_b = (sample_b[:, keep] - centre[keep]) / spread[keep]
        accuracy = _diagonal_gaussian_accuracy(scaled_a, scaled_b)
    best = max(per_statistic, key=lambda entry: entry["separation"])
    separable = bool(
        accuracy >= accuracy_threshold and best["separation"] >= auc_threshold
    )
    classifications = sample_a.shape[0] + sample_b.shape[0]
    standard_error = 0.5 / math.sqrt(classifications)
    above_chance = (accuracy - 0.5) / standard_error
    return {
        "separable": separable,
        "cv_accuracy": float(accuracy),
        "chance_accuracy": 0.5,
        "accuracy_standard_error": float(standard_error),
        "above_chance_z": float(above_chance),
        "signal_note": (
            "the classifier is indistinguishable from chance"
            if above_chance < 2.0
            else "the observations carry statistically detectable signal about which "
            "mechanism ran, but below the stated decision bar it is not enough to "
            "assign a mechanism to a single observed cascade"
            if not separable
            else "the observations assign single cascades to the right mechanism at "
            "the stated decision bar"
        ),
        "best_statistic": best["statistic"],
        "best_auc": best["auc"],
        "best_separation": best["separation"],
        "per_statistic": per_statistic,
        "accuracy_threshold": float(accuracy_threshold),
        "auc_threshold": float(auc_threshold),
        "replicates": [int(sample_a.shape[0]), int(sample_b.shape[0])],
    }


def _simulate_candidates(network, candidates, names, seeds, replicates, seed, horizon, **kwargs):
    samples, runs = {}, {}
    for index, candidate in enumerate(candidates):
        results = mech.replicate(
            network,
            candidate["mechanism"],
            params=candidate["parameters"],
            seeds=seeds,
            seed=int(seed) + 1000 * index,
            horizon=horizon,
            replicates=replicates,
            **kwargs,
        )
        runs[candidate["label"]] = results
        samples[candidate["label"]] = statistic_matrix(results, network, names)
    return samples, runs


def identifiability_report(
    network,
    candidates,
    observations=("adoption_curve", "adoption_times", "network"),
    seeds=None,
    replicates=24,
    seed=0,
    horizon=40,
    accuracy_threshold=0.80,
    auc_threshold=0.20,
):
    """Can these mechanisms be told apart at all, given the available observations?

    Simulates every candidate, computes the discriminating statistics the stated
    observations permit, and reports pairwise separability. Reporting that a pair
    is NOT separable is the intended output whenever it is true: it says the study
    cannot answer the mechanism question with the data in hand, and it should be
    read before any fit result.
    """
    if not isinstance(network, Network):
        raise Invalid("Expected a Network")
    if isinstance(replicates, bool) or not isinstance(replicates, (int, np.integer)):
        raise Invalid("replicates must be an integer")
    replicates = int(replicates)
    if not 4 <= replicates <= MAX_IDENTIFIABILITY_REPLICATES:
        raise Invalid(f"replicates must lie in [4, {MAX_IDENTIFIABILITY_REPLICATES}]")
    candidates = _normalize_candidates(candidates)
    if len(candidates) < 2:
        raise Invalid("Identifiability needs at least two candidate mechanisms")
    catalogue = available_statistics(observations, network)
    names = catalogue["statistics"]
    if not names:
        raise Invalid("The stated observations permit no discriminating statistic")
    seeds = _seed_nodes(seeds, network)
    samples, runs = _simulate_candidates(
        network, candidates, names, seeds, replicates, seed, horizon
    )

    pairs = []
    for first, second in itertools.combinations(candidates, 2):
        report = separability(
            samples[first["label"]],
            samples[second["label"]],
            names,
            accuracy_threshold=accuracy_threshold,
            auc_threshold=auc_threshold,
        )
        report["mechanism_a"] = first["label"]
        report["mechanism_b"] = second["label"]
        report["verdict"] = (
            f"separable: {report['best_statistic']} distinguishes them "
            f"(AUC {report['best_auc']:.2f}, cross-validated accuracy {report['cv_accuracy']:.2f})"
            if report["separable"]
            else (
                f"NOT separable under these observations: the best statistic is "
                f"{report['best_statistic']} at AUC {report['best_auc']:.2f} and "
                f"cross-validated accuracy is {report['cv_accuracy']:.2f} against a "
                f"decision bar of {report['accuracy_threshold']:.2f} "
                f"(chance 0.50, z = {report['above_chance_z']:.1f}). "
                + report["signal_note"]
            )
        )
        pairs.append(report)

    non_separable = [(p["mechanism_a"], p["mechanism_b"]) for p in pairs if not p["separable"]]
    summary = {
        label: {
            name: {
                "mean": float(samples[label][:, index].mean()),
                "sd": float(samples[label][:, index].std(ddof=1)),
            }
            for index, name in enumerate(names)
        }
        for label in samples
    }
    return {
        "observations": catalogue["observations"],
        "statistics": names,
        "excluded_statistics": catalogue["excluded"],
        "candidates": candidates,
        "seeds": list(map(int, seeds)),
        "replicates": replicates,
        "horizon": int(horizon),
        "pairs": pairs,
        "separable_pairs": [(p["mechanism_a"], p["mechanism_b"]) for p in pairs if p["separable"]],
        "non_separable_pairs": non_separable,
        "all_separable": not non_separable,
        "any_non_separable": bool(non_separable),
        "mechanism_statistics": summary,
        "samples": samples,
        "runs": runs,
        "verdict": (
            "every candidate pair is separable under these observations"
            if not non_separable
            else f"{len(non_separable)} of {len(pairs)} candidate pairs are NOT "
            "separable under these observations; a mechanism claim distinguishing "
            "them is not supported by this data"
        ),
        "scope": IDENTIFIABILITY_SCOPE,
    }


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #


def _synthetic_log_likelihood(observed, sample):
    """Gaussian synthetic likelihood with a diagonal covariance and a variance floor.

    Treating the statistics as independent overstates the information in a
    correlated statistic set, so the absolute value of this log likelihood is not
    interpretable; only differences between candidates scored on the identical
    statistic set are used.
    """
    mean = sample.mean(0)
    variance = sample.var(0, ddof=1)
    floor = (0.02 * np.abs(mean) + 0.01) ** 2
    variance = np.maximum(variance, floor)
    residual = observed - mean
    return float(
        -0.5 * np.sum(np.log(2 * math.pi * variance) + (residual**2) / variance)
    ), mean, np.sqrt(variance)


def fit_mechanisms(
    observed_cascade,
    network,
    candidates,
    observations=("adoption_curve", "adoption_times", "network"),
    seeds=None,
    replicates=12,
    seed=17,
    horizon=None,
    resolution=3,
    prior=None,
):
    """Fit every rival to one observed cascade and weight them over the candidate set.

    Each candidate is fitted by a bounded grid search that maximizes a
    simulation-based (synthetic) likelihood of the observed summary statistics.
    AIC penalizes the number of parameters actually searched. BIC's effective
    sample size is the number of summary statistics, not the number of nodes, so
    its penalty is weak; both are comparison devices here rather than Bayes
    factors. Posterior weights are Akaike weights under the stated prior over the
    candidate set, and they say nothing about a mechanism that was not proposed.
    """
    if not isinstance(network, Network):
        raise Invalid("Expected a Network")
    candidates = _normalize_candidates(candidates)
    catalogue = available_statistics(observations, network)
    names = catalogue["statistics"]
    if not names:
        raise Invalid("The stated observations permit no discriminating statistic")
    if isinstance(observed_cascade, mech.CascadeResult):
        observed = statistic_vector(observed_cascade, network, names)
        observed_horizon = observed_cascade.horizon
        observed_seeds = observed_cascade.seeds.tolist()
    elif isinstance(observed_cascade, dict) and "statistics" in observed_cascade:
        missing = [name for name in names if name not in observed_cascade["statistics"]]
        if missing:
            raise Invalid("Observed statistics are missing: " + ", ".join(missing))
        observed = np.array([float(observed_cascade["statistics"][n]) for n in names])
        observed_horizon = int(observed_cascade.get("horizon", 40))
        observed_seeds = list(observed_cascade.get("seeds", []))
    else:
        raise Invalid("observed_cascade must be a CascadeResult or a statistics dict")
    horizon = int(horizon if horizon is not None else observed_horizon)
    seeds = _seed_nodes(seeds if seeds is not None else (observed_seeds or None), network)

    grids = {}
    total = 0
    for candidate in candidates:
        grid = mech.parameter_grid(candidate["mechanism"], resolution=resolution)
        if len(grid) > MAX_GRID_POINTS:
            raise Invalid(
                f"Grid for {candidate['mechanism']} exceeds {MAX_GRID_POINTS} points"
            )
        grids[candidate["label"]] = grid
        total += len(grid) * int(replicates)
    if total > MAX_FIT_SIMULATIONS:
        raise Invalid(
            f"Fit envelope exceeded: {total} simulations against a limit of "
            f"{MAX_FIT_SIMULATIONS}; lower resolution or replicates"
        )

    fits = []
    for index, candidate in enumerate(candidates):
        best = None
        axes = sorted(mech.PARAMETER_GRIDS[candidate["mechanism"]])
        free = sum(
            1
            for axis in axes
            if len({point[axis] for point in grids[candidate["label"]]}) > 1
        )
        for position, point in enumerate(grids[candidate["label"]]):
            results = mech.replicate(
                network,
                candidate["mechanism"],
                params=point,
                seeds=seeds,
                seed=int(seed) + 1000 * index + 7 * position,
                horizon=horizon,
                replicates=replicates,
            )
            sample = statistic_matrix(results, network, names)
            value, mean, sigma = _synthetic_log_likelihood(observed, sample)
            if best is None or value > best["log_likelihood"]:
                best = {
                    "log_likelihood": value,
                    "parameters": dict(point),
                    "simulated_mean": mean,
                    "simulated_sd": sigma,
                }
        residual = (observed - best["simulated_mean"]) / best["simulated_sd"]
        worst = int(np.argmax(np.abs(residual)))
        fits.append(
            {
                "label": candidate["label"],
                "mechanism": candidate["mechanism"],
                "parameters": best["parameters"],
                "log_likelihood": best["log_likelihood"],
                "fitted_parameter_count": int(free),
                "grid_points": len(grids[candidate["label"]]),
                "standardized_residuals": {
                    name: float(residual[position]) for position, name in enumerate(names)
                },
                "worst_statistic": names[worst],
                "worst_standardized_residual": float(abs(residual[worst])),
                "simulated_mean": {
                    name: float(best["simulated_mean"][position])
                    for position, name in enumerate(names)
                },
            }
        )

    sample_size = len(names)
    for entry in fits:
        entry["aic"] = 2 * entry["fitted_parameter_count"] - 2 * entry["log_likelihood"]
        entry["bic"] = entry["fitted_parameter_count"] * math.log(sample_size) - 2 * entry["log_likelihood"]
    best_aic = min(entry["aic"] for entry in fits)
    if prior is None:
        prior = {entry["label"]: 1.0 / len(fits) for entry in fits}
    if set(prior) != {entry["label"] for entry in fits} or abs(sum(prior.values()) - 1.0) > 1e-6:
        raise Invalid("prior must be a normalized distribution over the candidate labels")
    weights = {}
    for entry in fits:
        entry["delta_aic"] = entry["aic"] - best_aic
        weights[entry["label"]] = prior[entry["label"]] * math.exp(-0.5 * min(entry["delta_aic"], 700.0))
    total_weight = sum(weights.values())
    for entry in fits:
        entry["posterior_weight"] = weights[entry["label"]] / total_weight if total_weight else 0.0
    fits.sort(key=lambda entry: entry["aic"])
    return {
        "observed_statistics": {name: float(observed[i]) for i, name in enumerate(names)},
        "statistics": names,
        "observations": catalogue["observations"],
        "fits": fits,
        "best_label": fits[0]["label"],
        "best_mechanism": fits[0]["mechanism"],
        "bic_sample_size": sample_size,
        "bic_sample_size_note": "the effective sample size is the number of summary "
        "statistics, not the number of nodes; BIC's penalty is correspondingly weak",
        "weight_basis": "Akaike weights under the stated prior over the candidate set",
        "simulations_run": total,
        "seeds": list(map(int, seeds)),
        "verdict": "best-fitting candidate identified; read identifiability_report "
        "before treating it as the mechanism",
        "scope": FIT_SCOPE,
    }


# --------------------------------------------------------------------------- #
# Choosing the next observation
# --------------------------------------------------------------------------- #


def _pair_runs(report, first, second):
    return report["runs"][first], report["runs"][second]


def _window_proposal(runs_a, runs_b, label_a, label_b):
    curve_a = np.vstack([r.increments for r in runs_a]).astype(np.float64)
    curve_b = np.vstack([r.increments for r in runs_b]).astype(np.float64)
    mean_a, mean_b = curve_a.mean(0), curve_b.mean(0)
    pooled = np.sqrt(0.5 * (curve_a.var(0, ddof=1) + curve_b.var(0, ddof=1))) + 1e-9
    standardized = np.abs(mean_a - mean_b) / pooled
    if standardized.size == 0:
        return None
    width = max(1, standardized.size // 6)
    kernel = np.ones(width)
    smoothed = np.convolve(standardized, kernel, mode="valid") / width
    start = int(np.argmax(smoothed))
    return {
        "action": "instrument_time_window",
        "detail": {
            "start_step": start + 1,
            "end_step": start + width,
            "mean_increment_a": float(mean_a[start : start + width].mean()),
            "mean_increment_b": float(mean_b[start : start + width].mean()),
        },
        "predicted_separability": None,
        "standardized_gap": float(smoothed[start]),
        "expected_observable_change": (
            f"between steps {start + 1} and {start + width} the mean adoption rate "
            f"differs most between {label_a} and {label_b}; measuring only this "
            "window buys most of the available aggregate separation"
        ),
        "cost_note": "a bounded observation window rather than a full time series",
    }


def _node_proposal(runs_a, runs_b, label_a, label_b, budget=10):
    times_a = np.vstack([np.where(np.isfinite(r.activation_time), r.activation_time, r.horizon + 1) for r in runs_a])
    times_b = np.vstack([np.where(np.isfinite(r.activation_time), r.activation_time, r.horizon + 1) for r in runs_b])
    pooled = np.sqrt(0.5 * (times_a.var(0, ddof=1) + times_b.var(0, ddof=1))) + 1e-9
    standardized = np.abs(times_a.mean(0) - times_b.mean(0)) / pooled
    order = np.argsort(-standardized)[:budget]
    return {
        "action": "instrument_node_set",
        "detail": {
            "nodes": [int(node) for node in order],
            "standardized_gap_per_node": [float(standardized[node]) for node in order],
        },
        "predicted_separability": None,
        "standardized_gap": float(standardized[order].mean()) if order.size else 0.0,
        "expected_observable_change": (
            f"these {len(order)} nodes have the largest difference in expected "
            f"adoption time between {label_a} and {label_b}; logging their adoption "
            "times alone reproduces most of the node-level separation"
        ),
        "cost_note": f"per-node logging for {len(order)} nodes rather than the whole graph",
    }


def _randomized_seeding_proposal(network, candidate_a, candidate_b, seeds, replicates, seed, horizon):
    """Does re-randomizing the seed set move who adopts? Transmission says yes."""
    rng = np.random.default_rng(int(seed) + 991)
    alternative = rng.choice(network.n, size=len(seeds), replace=False).tolist()
    sensitivities = {}
    for candidate in (candidate_a, candidate_b):
        overlaps = []
        for index in range(min(replicates, 12)):
            first = mech.simulate(
                network,
                candidate["mechanism"],
                params=candidate["parameters"],
                seeds=seeds,
                seed=int(seed) + 300 + index,
                horizon=horizon,
            )
            second = mech.simulate(
                network,
                candidate["mechanism"],
                params=candidate["parameters"],
                seeds=alternative,
                seed=int(seed) + 300 + index,
                horizon=horizon,
            )
            set_a = np.isfinite(first.activation_time)
            set_b = np.isfinite(second.activation_time)
            union = float((set_a | set_b).sum())
            overlaps.append(float((set_a & set_b).sum() / union) if union else 1.0)
        sensitivities[candidate["label"]] = 1.0 - float(np.mean(overlaps))
    gap = abs(sensitivities[candidate_a["label"]] - sensitivities[candidate_b["label"]])
    return {
        "action": "randomized_seeding_experiment",
        "detail": {
            "original_seeds": [int(node) for node in seeds],
            "alternative_seeds": [int(node) for node in alternative],
            "seed_set_sensitivity": sensitivities,
            "statistic": "1 - Jaccard overlap of the adopter sets under two seed sets, "
            "with the exogenous randomness held fixed",
        },
        "predicted_separability": None,
        "standardized_gap": float(gap),
        "expected_observable_change": (
            "under transmission the adopter set follows the seed set; under a shared "
            "external driver or a trait ordering it does not. Re-randomizing the "
            f"seeds moves the adopter set by {sensitivities[candidate_a['label']]:.2f} "
            f"for {candidate_a['label']} and {sensitivities[candidate_b['label']]:.2f} "
            f"for {candidate_b['label']}"
        ),
        "cost_note": "requires the ability to intervene on seeding, not only to observe",
    }


def discriminating_observation(
    network,
    candidates,
    observations=("adoption_curve",),
    pair=None,
    seeds=None,
    replicates=24,
    seed=0,
    horizon=40,
    accuracy_threshold=0.80,
    auc_threshold=0.20,
    report=None,
):
    """Name the observation that would separate a non-separable pair, and price it.

    This is the "choose the next evidence" step made concrete. For each
    non-separable pair the function evaluates every observation channel that is
    not yet available by recomputing separability with the statistics that
    channel unlocks, and adds three targeted proposals: the time window where the
    rivals' adoption rates diverge most, the node set whose adoption times differ
    most, and a randomized-seeding experiment. Channel proposals carry a predicted
    separability because they are actually recomputed; the targeted proposals
    carry a standardized gap instead, because they change what is measured rather
    than which statistics exist.
    """
    if report is None:
        report = identifiability_report(
            network,
            candidates,
            observations=observations,
            seeds=seeds,
            replicates=replicates,
            seed=seed,
            horizon=horizon,
            accuracy_threshold=accuracy_threshold,
            auc_threshold=auc_threshold,
        )
    candidates = report["candidates"]
    by_label = {entry["label"]: entry for entry in candidates}
    current = set(report["observations"])
    if pair is not None:
        pair = tuple(pair)
        if len(pair) != 2 or any(label not in by_label for label in pair):
            raise Invalid("pair must name two candidate labels")
        targets = [pair]
    else:
        targets = [tuple(p) for p in report["non_separable_pairs"]]
    seeds = report["seeds"]

    findings = []
    for label_a, label_b in targets:
        baseline = next(
            (
                p
                for p in report["pairs"]
                if {p["mechanism_a"], p["mechanism_b"]} == {label_a, label_b}
            ),
            None,
        )
        baseline_accuracy = baseline["cv_accuracy"] if baseline else 0.5
        runs_a, runs_b = _pair_runs(report, label_a, label_b)
        proposals = []
        for channel in OBSERVATION_CHANNELS:
            if channel in current:
                continue
            if channel == "community_labels" and network.communities is None:
                continue
            if channel == "node_attributes" and network.attributes is None:
                continue
            candidate_channels = sorted(current | {channel})
            names = available_statistics(candidate_channels, network)["statistics"]
            added = [name for name in names if name not in report["statistics"]]
            if not added:
                continue
            sample_a = statistic_matrix(runs_a, network, names)
            sample_b = statistic_matrix(runs_b, network, names)
            outcome = separability(
                sample_a,
                sample_b,
                names,
                accuracy_threshold=accuracy_threshold,
                auc_threshold=auc_threshold,
            )
            proposals.append(
                {
                    "action": "add_observation_channel",
                    "detail": {
                        "channel": channel,
                        "meaning": CHANNEL_MEANING[channel],
                        "statistics_unlocked": added,
                        "best_statistic": outcome["best_statistic"],
                        "best_auc": outcome["best_auc"],
                    },
                    "predicted_separability": outcome["cv_accuracy"],
                    "predicted_separable": outcome["separable"],
                    "separability_gain": outcome["cv_accuracy"] - baseline_accuracy,
                    "standardized_gap": outcome["best_separation"],
                    "expected_observable_change": (
                        f"adding {channel} unlocks {', '.join(added)}; cross-validated "
                        f"separability of {label_a} against {label_b} moves from "
                        f"{baseline_accuracy:.2f} to {outcome['cv_accuracy']:.2f}"
                    ),
                    "cost_note": CHANNEL_COST[channel],
                }
            )
        window = _window_proposal(runs_a, runs_b, label_a, label_b)
        if window is not None:
            proposals.append(window)
        proposals.append(_node_proposal(runs_a, runs_b, label_a, label_b))
        proposals.append(
            _randomized_seeding_proposal(
                network, by_label[label_a], by_label[label_b], seeds, replicates, seed, horizon
            )
        )
        proposals.sort(
            key=lambda entry: (
                entry["predicted_separability"] is not None,
                entry["predicted_separability"] if entry["predicted_separability"] is not None else 0.0,
                entry["standardized_gap"],
            ),
            reverse=True,
        )
        resolving = [
            p for p in proposals if p.get("predicted_separable")
        ]
        findings.append(
            {
                "pair": [label_a, label_b],
                "baseline_separability": baseline_accuracy,
                "baseline_separable": bool(baseline["separable"]) if baseline else None,
                "proposals": proposals,
                "resolving_proposals": [p["detail"] for p in resolving],
                "recommended": proposals[0] if proposals else None,
                "resolvable_with_available_channels": bool(resolving),
            }
        )
    return {
        "current_observations": sorted(current),
        "targets": [list(t) for t in targets],
        "findings": findings,
        "proposal_count": sum(len(f["proposals"]) for f in findings),
        "verdict": (
            "no non-separable pair was targeted; nothing to propose"
            if not findings
            else f"{len(findings)} non-separable pair(s); "
            f"{sum(1 for f in findings if f['resolvable_with_available_channels'])} "
            "would be resolved by a channel that can be added"
        ),
        "scope": "predicted separability under the same simulated candidates and "
        "parameters; a predicted gain is a forecast about a measurement that has "
        "not been made, and acquiring the channel may fail, be biased, or arrive "
        "with error that this projection does not model",
    }


# --------------------------------------------------------------------------- #
# Ensemble retention
# --------------------------------------------------------------------------- #


def _components(labels, edges):
    parent = {label: label for label in labels}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for first, second in edges:
        if first in parent and second in parent:
            root_a, root_b = find(first), find(second)
            if root_a != root_b:
                parent[root_b] = root_a
    groups = {}
    for label in labels:
        groups.setdefault(find(label), []).append(label)
    return [sorted(group) for group in groups.values() if len(group) > 1]


def retained_ensemble(fit, identifiability=None, rejection_delta=10.0):
    """Keep every mechanism the data cannot reject, and refuse a premature winner.

    A candidate is rejected only when it fits clearly worse than the best
    (delta AIC above `rejection_delta`) AND the identifiability report says the
    two are separable under the available observations. When a pair is not
    separable, an AIC difference between them is noise about a distinction the
    data cannot make, so both are retained and their weights are averaged within
    the non-separable group instead of letting the noise pick a winner.
    """
    if not isinstance(fit, dict) or "fits" not in fit:
        raise Invalid("fit must be the output of fit_mechanisms")
    if isinstance(rejection_delta, bool) or not isinstance(rejection_delta, (int, float)):
        raise Invalid("rejection_delta must be a number")
    rejection_delta = float(rejection_delta)
    if not 0.0 <= rejection_delta <= 1000.0:
        raise Invalid("rejection_delta must lie in [0, 1000]")
    labels = [entry["label"] for entry in fit["fits"]]
    non_separable = set()
    if identifiability is not None:
        for first, second in identifiability.get("non_separable_pairs", []):
            non_separable.add(frozenset((first, second)))
    best = fit["fits"][0]["label"]

    retained, rejected = [], []
    for entry in fit["fits"]:
        pair = frozenset((best, entry["label"]))
        tied = entry["label"] == best or pair in non_separable
        if entry["delta_aic"] > rejection_delta and not tied:
            rejected.append(
                {
                    "label": entry["label"],
                    "mechanism": entry["mechanism"],
                    "delta_aic": entry["delta_aic"],
                    "reason": f"fits worse than {best} by delta AIC "
                    f"{entry['delta_aic']:.1f} and the pair is separable under the "
                    "available observations, so the difference is interpretable",
                }
            )
        else:
            retained.append(dict(entry))
    if not retained:  # Never return an empty ensemble.
        retained = [dict(fit["fits"][0])]
        rejected = [r for r in rejected if r["label"] != retained[0]["label"]]

    kept = [entry["label"] for entry in retained]
    groups = _components(kept, [tuple(pair) for pair in non_separable])
    total = sum(entry["posterior_weight"] for entry in retained)
    for entry in retained:
        entry["akaike_weight"] = entry["posterior_weight"] / total if total else 1.0 / len(retained)
        entry["identifiability_adjusted_weight"] = entry["akaike_weight"]
    for group in groups:
        members = [entry for entry in retained if entry["label"] in group]
        share = sum(entry["akaike_weight"] for entry in members) / len(members)
        for entry in members:
            entry["identifiability_adjusted_weight"] = share
            entry["weight_note"] = (
                "averaged with " + ", ".join(sorted(set(group) - {entry["label"]}))
                + " because the available observations cannot separate them"
            )
    retained.sort(key=lambda entry: -entry["identifiability_adjusted_weight"])
    collapsed = len(retained) == 1
    return {
        "members": [
            {
                "label": entry["label"],
                "mechanism": entry["mechanism"],
                "parameters": entry["parameters"],
                "weight": entry["identifiability_adjusted_weight"],
                "akaike_weight": entry["akaike_weight"],
                "delta_aic": entry["delta_aic"],
                "log_likelihood": entry["log_likelihood"],
                "weight_note": entry.get("weight_note"),
            }
            for entry in retained
        ],
        "retained_labels": [entry["label"] for entry in retained],
        "rejected": rejected,
        "non_separable_groups": groups,
        "collapsed_to_single_winner": collapsed,
        "rejection_delta": rejection_delta,
        "verdict": (
            f"one candidate retained; every rival was both clearly worse-fitting and "
            "separable from it under the available observations"
            if collapsed
            else f"{len(retained)} candidates retained; the data do not support "
            "collapsing to a single mechanism"
        ),
        "scope": "the retained set is conditional on the candidate mechanisms "
        "proposed, the parameter grids searched, the observations available and "
        "the stated rejection threshold; retention is failure to reject, not "
        "support, and a mechanism nobody proposed cannot appear here",
    }
