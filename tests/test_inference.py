"""Inference layer against problems with known answers; no storage, network or model calls.

Each fixture is a closed-form problem whose correct result is known independently of this
code: ordinary least squares has analytic standard errors, the Ishigami function has published
Sobol indices, a product parameterization is exactly non-identifiable, a point-mass ensemble
has a CRPS equal to its absolute error. Passing these establishes that the numerics are right,
not that any model fitted with them is.
"""

import json
import unittest

import numpy as np

from symplex.core.contracts import Invalid
from symplex.inference import (
    bootstrap_intervals,
    crps_ensemble,
    fisher_information,
    interval_score,
    least_squares_fit,
    local_sensitivity,
    morris_screening,
    parameter_uncertainty,
    pit_coverage,
    posterior_predictive_check,
    practical_identifiability,
    profile_likelihood,
    reliability,
    sobol_indices,
    uncertainty_propagation,
)

WIDE = [(-50.0, 50.0), (-50.0, 50.0)]
POSITIVE = [(0.1, 50.0), (0.1, 50.0)]


def linear_case(seed=7, noise=0.2):
    """y = a*x + b on centered x, where the OLS estimator and its standard errors are exact."""
    x = np.linspace(-5.0, 5.0, 60)
    values = 2.5 * x - 1.0 + np.random.default_rng(seed).normal(0.0, noise, x.size)
    return x, values, (lambda theta: theta[0] * x + theta[1])


def decay_case(seed=11, noise=0.02):
    time = np.linspace(0.0, 5.0, 50)
    values = 3.0 * np.exp(-0.8 * time) + np.random.default_rng(seed).normal(0.0, noise, time.size)
    return time, values, (lambda theta: theta[0] * np.exp(-theta[1] * time))


def product_case(seed=3, noise=0.05):
    """y = (a*b)*x: only the product is identifiable, so a and b trade off exactly."""
    x = np.linspace(1.0, 4.0, 40)
    values = 6.0 * x + np.random.default_rng(seed).normal(0.0, noise, x.size)
    return x, values, (lambda theta: (theta[0] * theta[1]) * x)


def ishigami(theta):
    """Ishigami with a=7, b=0.1; analytic S1 = [.3139, .4424, 0], ST = [.5576, .4424, .2437]."""
    return np.array(
        [
            np.sin(theta[0])
            + 7.0 * np.sin(theta[1]) ** 2
            + 0.1 * theta[2] ** 4 * np.sin(theta[0])
        ]
    )


ISHIGAMI_BOUNDS = [(-np.pi, np.pi)] * 3
ISHIGAMI_FIRST = [0.3139, 0.4424, 0.0]
ISHIGAMI_TOTAL = [0.5576, 0.4424, 0.2437]


def calibrated_forecasts():
    """Ten bins whose observed frequency equals the forecast exactly, so ECE is zero."""
    probabilities, outcomes = [], []
    for center in np.arange(0.05, 1.0, 0.1):
        positives = int(round(center * 100))
        probabilities.extend([float(center)] * 100)
        outcomes.extend([1] * positives + [0] * (100 - positives))
    return np.array(probabilities), np.array(outcomes)


class EstimationTests(unittest.TestCase):
    def test_least_squares_recovers_linear_parameters_and_reports_convergence(self):
        _, values, simulate = linear_case()
        result = least_squares_fit(simulate, values, [0.0, 0.0], WIDE)
        self.assertTrue(result["converged"])
        self.assertAlmostEqual(result["theta_hat"][0], 2.5, delta=0.05)
        self.assertAlmostEqual(result["theta_hat"][1], -1.0, delta=0.1)
        self.assertEqual(result["degrees_of_freedom"], 58)
        self.assertEqual(result["at_bound"], [False, False])
        self.assertLess(result["evaluations"], result["max_evaluations"])
        self.assertIsInstance(result["scope"], str)
        json.dumps(result, allow_nan=False)

    def test_standard_errors_match_the_analytic_ordinary_least_squares_errors(self):
        x, values, simulate = linear_case()
        fit = least_squares_fit(simulate, values, [0.0, 0.0], WIDE)
        information = fisher_information(simulate, fit["theta_hat"])
        uncertainty = parameter_uncertainty(
            information["fim"], scale=fit["reduced_chi_square"], theta=fit["theta_hat"]
        )
        design = np.column_stack([x, np.ones_like(x)])
        beta = np.linalg.lstsq(design, values, rcond=None)[0]
        residual = values - design @ beta
        variance = float(residual @ residual) / (x.size - 2)
        analytic = np.sqrt(np.diag(variance * np.linalg.inv(design.T @ design)))
        for estimated, expected in zip(uncertainty["standard_errors"], analytic):
            self.assertAlmostEqual(estimated / expected, 1.0, places=6)
        for estimated, expected in zip(fit["theta_hat"], beta):
            self.assertAlmostEqual(estimated, float(expected), places=8)
        self.assertEqual(uncertainty["numerical_rank"], 2)
        self.assertEqual(uncertainty["floored_directions"], 0)
        self.assertEqual(len(uncertainty["intervals"]), 2)

    def test_exponential_decay_recovers_amplitude_and_rate(self):
        _, values, simulate = decay_case()
        fit = least_squares_fit(simulate, values, [1.0, 0.2], POSITIVE)
        self.assertTrue(fit["converged"])
        self.assertAlmostEqual(fit["theta_hat"][0], 3.0, delta=0.05)
        self.assertAlmostEqual(fit["theta_hat"][1], 0.8, delta=0.02)
        information = fisher_information(simulate, fit["theta_hat"], bounds=POSITIVE)
        verdict = practical_identifiability(
            information["fim"], theta=fit["theta_hat"], scale=fit["reduced_chi_square"]
        )
        self.assertEqual(verdict["overall"], "identifiable")
        self.assertEqual(verdict["collinear_pairs"], [])

    def test_supplied_sigma_produces_a_reduced_chi_square_near_one(self):
        _, values, simulate = linear_case(noise=0.2)
        fit = least_squares_fit(simulate, values, [0.0, 0.0], WIDE, sigma=0.2)
        self.assertTrue(fit["sigma_supplied"])
        self.assertAlmostEqual(fit["reduced_chi_square"], 1.0, delta=0.35)

    def test_non_convergence_withholds_theta_hat(self):
        _, values, simulate = decay_case()
        fit = least_squares_fit(simulate, values, [1.0, 0.2], POSITIVE, max_nfev=1)
        self.assertFalse(fit["converged"])
        self.assertIsNone(fit["theta_hat"])
        self.assertEqual(len(fit["terminal_theta"]), 2)
        self.assertIn("did NOT reach a convergence criterion", fit["scope"])

    def test_product_model_is_non_identifiable_with_a_near_perfect_correlation(self):
        _, values, simulate = product_case()
        fit = least_squares_fit(simulate, values, [2.0, 3.0], POSITIVE)
        self.assertAlmostEqual(fit["theta_hat"][0] * fit["theta_hat"][1], 6.0, delta=0.05)
        information = fisher_information(simulate, fit["theta_hat"], bounds=POSITIVE)
        self.assertEqual(information["numerical_rank"], 1)
        self.assertIsNone(information["condition_number"])
        verdict = practical_identifiability(
            information["fim"], theta=fit["theta_hat"], scale=fit["reduced_chi_square"]
        )
        self.assertEqual(verdict["overall"], "non_identifiable")
        self.assertEqual(
            [p["verdict"] for p in verdict["parameters"]],
            ["non_identifiable", "non_identifiable"],
        )
        self.assertLess(verdict["correlation"][0][1], -0.99)
        self.assertEqual(len(verdict["collinear_pairs"]), 1)
        self.assertEqual(verdict["collinear_pairs"][0]["indices"], [0, 1])
        json.dumps(verdict, allow_nan=False)

    def test_profile_likelihood_is_flat_for_the_non_identifiable_product(self):
        _, values, simulate = product_case()
        fit = least_squares_fit(simulate, values, [2.0, 3.0], POSITIVE)
        profile = profile_likelihood(
            simulate, values, fit["theta_hat"], 0, np.linspace(1.0, 5.0, 13), POSITIVE
        )
        self.assertTrue(profile["flat"])
        self.assertEqual(profile["identifiability"], "structurally_non_identifiable")
        self.assertIsNone(profile["confidence_interval"]["lower"])
        self.assertIsNone(profile["confidence_interval"]["upper"])
        self.assertEqual(profile["non_converged_points"], 0)
        self.assertLess(profile["max_delta"], profile["threshold"])

    def test_profile_likelihood_closes_around_an_identifiable_parameter(self):
        _, values, simulate = linear_case()
        fit = least_squares_fit(simulate, values, [0.0, 0.0], WIDE)
        center = fit["theta_hat"][0]
        profile = profile_likelihood(
            simulate, values, fit["theta_hat"], 0, np.linspace(center - 0.1, center + 0.1, 21), WIDE
        )
        self.assertEqual(profile["identifiability"], "identifiable")
        self.assertFalse(profile["flat"])
        interval = profile["confidence_interval"]
        self.assertLess(interval["lower"], center)
        self.assertGreater(interval["upper"], center)
        # The profile interval should agree with the Wald interval on a linear model.
        information = fisher_information(simulate, fit["theta_hat"])
        errors = parameter_uncertainty(
            information["fim"], scale=fit["reduced_chi_square"]
        )["standard_errors"]
        self.assertAlmostEqual(interval["upper"] - interval["lower"], 3.92 * errors[0], delta=0.005)

    def test_profile_likelihood_rejects_a_grid_that_does_not_bracket_the_estimate(self):
        _, values, simulate = linear_case()
        with self.assertRaises(Invalid):
            profile_likelihood(
                simulate, values, [2.5, -1.0], 0, np.linspace(10.0, 12.0, 5), WIDE
            )

    def test_bootstrap_intervals_are_seeded_and_cover_the_generating_parameters(self):
        _, values, simulate = decay_case()
        fit = least_squares_fit(simulate, values, [1.0, 0.2], POSITIVE)
        first = bootstrap_intervals(
            simulate, values, fit["theta_hat"], POSITIVE, draws=24, seed=5
        )
        second = bootstrap_intervals(
            simulate, values, fit["theta_hat"], POSITIVE, draws=24, seed=5
        )
        self.assertEqual(first["lower"], second["lower"])
        self.assertEqual(first["upper"], second["upper"])
        self.assertEqual(first["draws_used"], 24)
        self.assertEqual(first["failed_draws"], 0)
        for index, truth in enumerate((3.0, 0.8)):
            self.assertLessEqual(first["lower"][index], fit["theta_hat"][index])
            self.assertGreaterEqual(first["upper"][index], fit["theta_hat"][index])
            # The estimate sits within a few bootstrap standard errors of the truth; the
            # interval is centered on the estimate, so it need not contain the truth itself.
            error = abs(fit["theta_hat"][index] - truth)
            self.assertLess(error, 3.0 * first["bootstrap_standard_errors"][index])
        # A correctly specified residual bootstrap should agree with the asymptotic errors.
        information = fisher_information(simulate, fit["theta_hat"], bounds=POSITIVE)
        asymptotic = parameter_uncertainty(
            information["fim"], scale=fit["reduced_chi_square"]
        )["standard_errors"]
        for bootstrapped, expected in zip(first["bootstrap_standard_errors"], asymptotic):
            self.assertLess(abs(bootstrapped / expected - 1.0), 0.5)
        different = bootstrap_intervals(
            simulate, values, fit["theta_hat"], POSITIVE, draws=24, seed=6
        )
        self.assertNotEqual(first["lower"], different["lower"])

    def test_parameter_uncertainty_flags_floored_directions_on_a_singular_information(self):
        singular = [[4.0, 2.0], [2.0, 1.0]]
        result = parameter_uncertainty(singular, rcond=1e-10)
        self.assertEqual(result["numerical_rank"], 1)
        self.assertEqual(result["floored_directions"], 1)
        self.assertIn("floored at rcond*lambda_max", result["scope"])

    def test_estimation_rejects_malformed_input(self):
        _, values, simulate = linear_case()
        with self.assertRaises(Invalid):
            least_squares_fit("not-callable", values, [0.0, 0.0], WIDE)
        with self.assertRaises(Invalid):
            least_squares_fit(simulate, values, [99.0, 0.0], WIDE)
        with self.assertRaises(Invalid):
            least_squares_fit(simulate, values, [0.0, 0.0], [(5.0, -5.0), (-5.0, 5.0)])
        with self.assertRaises(Invalid):
            least_squares_fit(simulate, values, [0.0, 0.0], WIDE, sigma=0.0)
        with self.assertRaises(Invalid):
            least_squares_fit(simulate, values, [0.0, 0.0], [(-5.0, 5.0)])
        with self.assertRaises(Invalid):
            least_squares_fit(simulate, values[:-1], [0.0, 0.0], WIDE)
        with self.assertRaises(Invalid):
            parameter_uncertainty([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        with self.assertRaises(Invalid):
            practical_identifiability([[1.0, 0.0], [0.0, 1.0]], parameter_names=["a", "a"])


class SensitivityTests(unittest.TestCase):
    def test_local_elasticity_of_a_scale_parameter_is_one(self):
        grid = np.linspace(1.0, 3.0, 5)
        result = local_sensitivity(lambda theta: theta[0] * grid**2, [2.0])
        for row in result["elasticity"]:
            self.assertAlmostEqual(row[0], 1.0, places=8)
        self.assertEqual(result["undefined_entries"], 0)
        self.assertAlmostEqual(result["parameter_summary"][0]["mean_abs_elasticity"], 1.0, places=8)
        json.dumps(result, allow_nan=False)

    def test_local_elasticity_reports_undefined_entries_instead_of_imputing(self):
        grid = np.array([0.0, 1.0, 2.0])
        result = local_sensitivity(lambda theta: theta[0] * grid, [2.0])
        self.assertEqual(result["undefined_entries"], 1)
        self.assertIsNone(result["elasticity"][0][0])
        self.assertAlmostEqual(result["elasticity"][1][0], 1.0, places=8)

    def test_morris_ranks_an_inert_parameter_last(self):
        def model(theta):
            return np.array([3.0 * theta[0] + 0.5 * theta[1] + 0.0 * theta[2]])

        result = morris_screening(model, [(0.0, 1.0)] * 3, trajectories=8, levels=4, seed=2)
        ranks = {p["index"]: p["rank"] for p in result["parameters"]}
        self.assertEqual(ranks[2], 3)
        self.assertEqual(result["ranking"][-1], 2)
        self.assertAlmostEqual(result["parameters"][2]["mu_star"], 0.0, places=12)
        self.assertGreater(result["parameters"][0]["mu_star"], result["parameters"][1]["mu_star"])
        self.assertEqual(result["evaluations"], 8 * 4)
        repeat = morris_screening(model, [(0.0, 1.0)] * 3, trajectories=8, levels=4, seed=2)
        self.assertEqual(result["elementary_effects"], repeat["elementary_effects"])

    def test_morris_requires_even_levels_and_a_finite_box(self):
        def model(theta):
            return np.array([theta[0]])

        with self.assertRaises(Invalid):
            morris_screening(model, [(0.0, 1.0)], trajectories=4, levels=3, seed=0)
        with self.assertRaises(Invalid):
            morris_screening(model, [(0.0, np.inf)], trajectories=4, levels=4, seed=0)

    def test_sobol_indices_match_the_published_ishigami_values(self):
        result = sobol_indices(ishigami, ISHIGAMI_BOUNDS, n=2048, seed=1, bootstrap=60)
        for index, expected in enumerate(ISHIGAMI_FIRST):
            self.assertAlmostEqual(result["first_order"][index], expected, delta=0.02)
        for index, expected in enumerate(ISHIGAMI_TOTAL):
            self.assertAlmostEqual(result["total_order"][index], expected, delta=0.02)
        # The third input acts only through its interaction with the first.
        self.assertLess(result["parameters"][2]["first_order"], 0.02)
        self.assertGreater(result["parameters"][2]["total_order"], 0.2)
        self.assertGreater(result["interaction_share"], 0.2)
        self.assertEqual(result["evaluations"], 2048 * 5)
        self.assertIn("INDEPENDENT", result["scope"])
        lower, upper = result["parameters"][0]["first_order_interval"]
        self.assertLess(lower, result["first_order"][0])
        self.assertGreater(upper, result["first_order"][0])

    def test_sobol_is_deterministic_and_requires_a_power_of_two(self):
        first = sobol_indices(ishigami, ISHIGAMI_BOUNDS, n=64, seed=3, bootstrap=10)
        second = sobol_indices(ishigami, ISHIGAMI_BOUNDS, n=64, seed=3, bootstrap=10)
        self.assertEqual(first["first_order"], second["first_order"])
        self.assertEqual(first["parameters"], second["parameters"])
        with self.assertRaises(Invalid):
            sobol_indices(ishigami, ISHIGAMI_BOUNDS, n=100, seed=3, bootstrap=0)

    def test_screening_requires_a_declared_scalar_for_a_vector_output(self):
        grid = np.linspace(0.0, 1.0, 4)
        with self.assertRaises(Invalid):
            morris_screening(lambda theta: theta[0] * grid, [(0.0, 1.0)], trajectories=2, seed=0)
        result = morris_screening(
            lambda theta: theta[0] * grid,
            [(0.0, 1.0)],
            trajectories=2,
            seed=0,
            aggregate=np.mean,
        )
        self.assertAlmostEqual(result["parameters"][0]["mu_star"], float(grid.mean()), places=8)

    def test_uncertainty_propagation_returns_an_ordered_quantile_fan(self):
        grid = np.linspace(0.0, 2.0, 6)
        samples = np.random.default_rng(0).normal([2.0, 1.0], [0.1, 0.05], size=(60, 2))
        result = uncertainty_propagation(lambda theta: theta[0] * grid + theta[1], samples)
        self.assertEqual(sorted(result["quantiles"]), ["p25", "p5", "p50", "p75", "p95"])
        fan = result["quantiles"]
        for index in range(grid.size):
            self.assertLessEqual(fan["p5"][index], fan["p25"][index])
            self.assertLessEqual(fan["p25"][index], fan["p50"][index])
            self.assertLessEqual(fan["p50"][index], fan["p75"][index])
            self.assertLessEqual(fan["p75"][index], fan["p95"][index])
            self.assertLessEqual(result["minimum"][index], fan["p5"][index])
            self.assertGreaterEqual(result["maximum"][index], fan["p95"][index])
        self.assertEqual(result["draws"], 60)
        self.assertEqual(result["evaluations"], 60)
        self.assertAlmostEqual(result["mean"][0], 1.0, delta=0.05)
        json.dumps(result, allow_nan=False)

    def test_uncertainty_propagation_rejects_unsorted_or_duplicated_quantiles(self):
        samples = np.zeros((3, 1)) + 1.0
        with self.assertRaises(Invalid):
            uncertainty_propagation(lambda theta: theta * 1.0, samples, quantiles=(0.5, 0.25))
        with self.assertRaises(Invalid):
            uncertainty_propagation(lambda theta: theta * 1.0, samples, quantiles=(0.5, 0.5))
        with self.assertRaises(Invalid):
            uncertainty_propagation(lambda theta: theta * 1.0, samples, quantiles=(0.0, 0.5))


class CalibrationTests(unittest.TestCase):
    def test_perfectly_calibrated_probabilities_have_zero_expected_calibration_error(self):
        probabilities, outcomes = calibrated_forecasts()
        result = reliability(probabilities, outcomes, bins=10)
        self.assertAlmostEqual(result["ece"], 0.0, places=12)
        self.assertAlmostEqual(result["mce"], 0.0, places=12)
        self.assertAlmostEqual(result["decomposition_residual"], 0.0, places=12)
        self.assertAlmostEqual(
            result["brier"],
            result["reliability_component"]
            - result["resolution_component"]
            + result["uncertainty_component"],
            places=12,
        )
        self.assertEqual(result["occupied_bins"], 10)
        json.dumps(result, allow_nan=False)

    def test_overconfident_probabilities_have_a_clearly_positive_calibration_error(self):
        probabilities, outcomes = calibrated_forecasts()
        overconfident = np.clip(0.5 + 2.0 * (probabilities - 0.5), 0.01, 0.99)
        result = reliability(overconfident, outcomes, bins=10)
        self.assertGreater(result["ece"], 0.1)
        self.assertGreater(result["mce"], result["ece"])
        baseline = reliability(probabilities, outcomes, bins=10)
        self.assertGreater(result["brier"], baseline["brier"])
        self.assertGreater(result["reliability_component"], baseline["reliability_component"])

    def test_reliability_rejects_non_probabilities_and_non_binary_outcomes(self):
        with self.assertRaises(Invalid):
            reliability([0.5, 1.5], [0, 1])
        with self.assertRaises(Invalid):
            reliability([0.5, 0.5], [0, 2])
        with self.assertRaises(Invalid):
            reliability([0.5, 0.5], [0, 1, 1])
        with self.assertRaises(Invalid):
            reliability([0.5, 0.5], [0, 1], bins=0)

    def test_crps_of_a_point_mass_ensemble_equals_the_absolute_error(self):
        result = crps_ensemble(np.full(8, 3.0), 1.5)
        self.assertAlmostEqual(result["crps"], 1.5, places=12)
        self.assertAlmostEqual(result["ensemble_spread_term"], 0.0, places=12)
        self.assertAlmostEqual(result["mean_absolute_error_term"], 1.5, places=12)

    def test_crps_matches_the_direct_pairwise_definition(self):
        members = np.array([1.0, 2.0, 4.0, 7.0])
        observation = 3.0
        expected = float(
            np.mean(np.abs(members - observation))
            - 0.5 * np.mean(np.abs(members[:, None] - members[None, :]))
        )
        self.assertAlmostEqual(crps_ensemble(members, observation)["crps"], expected, places=12)

    def test_crps_rewards_the_sharper_of_two_unbiased_ensembles(self):
        generator = np.random.default_rng(1)
        truth = generator.normal(0.0, 1.0, 50)
        sharp = truth[None, :] + generator.normal(0.0, 1.0, (40, 50))
        diffuse = truth[None, :] + generator.normal(0.0, 4.0, (40, 50))
        self.assertLess(
            crps_ensemble(sharp, truth)["crps"], crps_ensemble(diffuse, truth)["crps"]
        )

    def test_pit_is_uniform_for_a_correct_ensemble_and_rejected_for_a_biased_one(self):
        generator = np.random.default_rng(4)
        centers = generator.normal(0.0, 2.0, 200)
        observations = centers + generator.normal(0.0, 1.0, 200)
        ensemble = centers[None, :] + generator.normal(0.0, 1.0, (100, 200))
        correct = pit_coverage(ensemble, observations)
        self.assertFalse(correct["uniform_rejected_at_5_percent"])
        self.assertGreater(correct["ks_pvalue"], 0.05)
        self.assertAlmostEqual(correct["pit_mean"], 0.5, delta=0.05)
        for entry in correct["coverage"]:
            self.assertAlmostEqual(entry["empirical"], entry["nominal"], delta=0.06)
        biased = pit_coverage(ensemble + 2.0, observations)
        self.assertTrue(biased["uniform_rejected_at_5_percent"])
        self.assertLess(biased["ks_pvalue"], 0.01)
        self.assertLess(biased["coverage"][3]["empirical"], 0.95)
        json.dumps(correct, allow_nan=False)

    def test_pit_detects_an_overconfident_ensemble_through_coverage(self):
        generator = np.random.default_rng(9)
        observations = generator.normal(0.0, 1.0, 300)
        narrow = generator.normal(0.0, 0.2, (80, 300))
        result = pit_coverage(narrow, observations, levels=(0.9,))
        self.assertTrue(result["uniform_rejected_at_5_percent"])
        self.assertLess(result["coverage"][0]["empirical"], 0.5)

    def test_interval_score_matches_the_winkler_definition(self):
        result = interval_score([1.0], [3.0], [5.0], 0.1)
        self.assertAlmostEqual(result["interval_score"], 2.0 + (2.0 / 0.1) * 2.0, places=12)
        self.assertAlmostEqual(result["sharpness"], 2.0, places=12)
        self.assertAlmostEqual(result["overprediction_penalty"], 40.0, places=12)
        self.assertEqual(result["misses_above"], 1)
        self.assertEqual(result["empirical_coverage"], 0.0)
        covered = interval_score([1.0], [3.0], [2.0], 0.1)
        self.assertAlmostEqual(covered["interval_score"], 2.0, places=12)
        self.assertEqual(covered["empirical_coverage"], 1.0)
        self.assertLess(covered["interval_score"], result["interval_score"])

    def test_interval_score_rejects_inverted_or_misaligned_intervals(self):
        with self.assertRaises(Invalid):
            interval_score([3.0], [1.0], [2.0], 0.1)
        with self.assertRaises(Invalid):
            interval_score([1.0, 2.0], [3.0], [2.0], 0.1)
        with self.assertRaises(Invalid):
            interval_score([1.0], [3.0], [2.0], 1.0)

    def test_posterior_predictive_flags_an_inflated_observed_variance(self):
        generator = np.random.default_rng(21)
        replicates = generator.normal(0.0, 1.0, (200, 60))
        observations = generator.normal(0.0, 3.0, 60)
        result = posterior_predictive_check(replicates, observations)
        flagged = {entry["statistic"]: entry for entry in result["checks"]}
        self.assertTrue(result["discrepancy_detected"])
        self.assertIn("variance", result["flagged_statistics"])
        self.assertLess(flagged["variance"]["bayesian_pvalue"], 0.05)
        self.assertEqual(flagged["variance"]["direction"], "observed_above_replicates")
        self.assertEqual(result["statistic_names"], ["mean", "variance", "max", "lag1_autocorrelation"])
        json.dumps(result, allow_nan=False)

    def test_posterior_predictive_does_not_flag_a_matched_replicate_set(self):
        generator = np.random.default_rng(21)
        replicates = generator.normal(0.0, 1.0, (200, 60))
        observations = generator.normal(0.0, 1.0, 60)
        result = posterior_predictive_check(replicates, observations, statistics=("mean", "variance"))
        self.assertFalse(result["discrepancy_detected"])
        for entry in result["checks"]:
            self.assertGreater(entry["bayesian_pvalue"], 0.05)
            self.assertLess(entry["bayesian_pvalue"], 0.95)

    def test_posterior_predictive_accepts_a_caller_supplied_statistic(self):
        replicates = np.zeros((4, 5)) + np.arange(4)[:, None]
        observations = np.full(5, 1.5)
        result = posterior_predictive_check(
            replicates, observations, statistics={"median": lambda v: float(np.median(v))}
        )
        self.assertEqual(result["statistic_names"], ["median"])
        self.assertAlmostEqual(result["checks"][0]["bayesian_pvalue"], 0.5, places=12)
        with self.assertRaises(Invalid):
            posterior_predictive_check(replicates, observations, statistics=("not_a_statistic",))
        with self.assertRaises(Invalid):
            posterior_predictive_check(replicates, observations[:3])


class BudgetTests(unittest.TestCase):
    """Every routine that calls the simulator must fail closed on its evaluation budget."""

    def test_least_squares_fit_budget(self):
        _, values, simulate = linear_case()
        with self.assertRaises(Invalid):
            least_squares_fit(simulate, values, [0.0, 0.0], WIDE, max_evaluations=2)

    def test_fisher_information_budget(self):
        _, _, simulate = linear_case()
        with self.assertRaises(Invalid):
            fisher_information(simulate, [2.5, -1.0], max_evaluations=3)

    def test_profile_likelihood_budget(self):
        _, values, simulate = linear_case()
        with self.assertRaises(Invalid):
            profile_likelihood(
                simulate,
                values,
                [2.5, -1.0],
                0,
                np.linspace(2.4, 2.6, 5),
                WIDE,
                max_evaluations=2,
            )

    def test_bootstrap_intervals_budget(self):
        _, values, simulate = linear_case()
        with self.assertRaises(Invalid):
            bootstrap_intervals(
                simulate, values, [2.5, -1.0], WIDE, draws=10, seed=0, max_evaluations=3
            )

    def test_local_sensitivity_budget(self):
        _, _, simulate = linear_case()
        with self.assertRaises(Invalid):
            local_sensitivity(simulate, [2.5, -1.0], max_evaluations=4)

    def test_morris_screening_budget(self):
        with self.assertRaises(Invalid):
            morris_screening(
                ishigami, ISHIGAMI_BOUNDS, trajectories=8, levels=4, seed=0, max_evaluations=10
            )

    def test_sobol_indices_budget(self):
        with self.assertRaises(Invalid):
            sobol_indices(ishigami, ISHIGAMI_BOUNDS, n=64, seed=0, max_evaluations=100)

    def test_uncertainty_propagation_budget(self):
        samples = np.zeros((12, 2)) + 1.0
        with self.assertRaises(Invalid):
            uncertainty_propagation(
                lambda theta: theta[0] * np.ones(3), samples, max_evaluations=5
            )

    def test_a_failing_or_nonfinite_simulator_is_an_input_failure(self):
        _, values, simulate = linear_case()

        def broken(theta):
            raise RuntimeError("caller model exploded")

        with self.assertRaises(Invalid):
            least_squares_fit(broken, values, [0.0, 0.0], WIDE)
        with self.assertRaises(Invalid):
            least_squares_fit(
                lambda theta: np.full(values.size, np.nan), values, [0.0, 0.0], WIDE
            )
        with self.assertRaises(Invalid):
            least_squares_fit(lambda theta: simulate(theta)[:-1], values, [0.0, 0.0], WIDE)


class ScopeTests(unittest.TestCase):
    def test_every_public_routine_reports_what_it_does_not_establish(self):
        import symplex.inference as inference

        grid = np.linspace(1.0, 2.0, 4)
        _, values, simulate = linear_case()
        fit = least_squares_fit(simulate, values, [0.0, 0.0], WIDE)
        information = fisher_information(simulate, fit["theta_hat"])
        probabilities, outcomes = calibrated_forecasts()
        ensemble = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
        results = {
            "least_squares_fit": fit,
            "fisher_information": information,
            "parameter_uncertainty": parameter_uncertainty(information["fim"]),
            "practical_identifiability": practical_identifiability(information["fim"]),
            "profile_likelihood": profile_likelihood(
                simulate, values, fit["theta_hat"], 0, np.linspace(2.4, 2.7, 5), WIDE
            ),
            "bootstrap_intervals": bootstrap_intervals(
                simulate, values, fit["theta_hat"], WIDE, draws=3, seed=0
            ),
            "local_sensitivity": local_sensitivity(lambda theta: theta[0] * grid, [2.0]),
            "morris_screening": morris_screening(
                ishigami, ISHIGAMI_BOUNDS, trajectories=3, levels=4, seed=0
            ),
            "sobol_indices": sobol_indices(
                ishigami, ISHIGAMI_BOUNDS, n=32, seed=0, bootstrap=5
            ),
            "uncertainty_propagation": uncertainty_propagation(
                lambda theta: theta[0] * grid, np.array([[1.0], [2.0], [3.0]])
            ),
            "reliability": reliability(probabilities, outcomes),
            "crps_ensemble": crps_ensemble(ensemble, [0.5, 1.5]),
            "pit_coverage": pit_coverage(ensemble, [0.5, 1.5]),
            "interval_score": interval_score([0.0], [1.0], [0.5], 0.1),
            "posterior_predictive_check": posterior_predictive_check(
                np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]), [1.5, 2.5, 3.5], statistics=("mean",)
            ),
        }
        self.assertEqual(sorted(results), sorted(inference.__all__))
        for name, result in results.items():
            with self.subTest(routine=name):
                self.assertIsInstance(result, dict)
                self.assertIsInstance(result.get("scope"), str)
                self.assertGreater(len(result["scope"]), 80)
                json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
