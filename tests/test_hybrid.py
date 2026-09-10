"""Deterministic checks of the hybrid/neurosymbolic modelling layer.

Every fixture is a small simulated system with a known answer, so a failure localizes to
the implementation rather than to a data source. Nothing here establishes that any of these
methods works on real measurements.
"""

import time
import unittest

import numpy as np

from symplex.core.contracts import Invalid
from symplex.hybrid import (
    GPEmulator,
    NeuralSurrogate,
    TrainingSupport,
    approximation_error,
    decompose_error,
    discrepancy_diagnosis,
    estimate_derivatives,
    fit_discrepancy,
    fit_sindy,
    fit_ude,
    kernel_matrix,
    pareto_sweep,
    penalty_terms,
    polynomial_library,
    project,
    project_trajectory,
    render_equation,
    simulate_ude,
    stability_selection,
    stlsq,
    symbolic_recovery,
    validate_constraints,
    violation_report,
    weak_form_sindy,
)


def rk4(rhs, x0, times):
    states = np.empty((times.size, np.size(x0)))
    states[0] = x0
    for index in range(times.size - 1):
        step = times[index + 1] - times[index]
        current = states[index]
        k1 = rhs(current)
        k2 = rhs(current + step / 2 * k1)
        k3 = rhs(current + step / 2 * k2)
        k4 = rhs(current + step * k3)
        states[index + 1] = current + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return states


def lorenz_states(dt=0.002, horizon=8.0):
    def rhs(x):
        return np.array(
            [10.0 * (x[1] - x[0]), x[0] * (28.0 - x[2]) - x[1], x[0] * x[1] - 8.0 / 3.0 * x[2]]
        )

    times = np.arange(0.0, horizon, dt)
    return times, rk4(rhs, np.array([-8.0, 7.0, 27.0]), times)


def oscillator_states(dt=0.01, horizon=15.0, stiffness=2.0, damping=0.3):
    def rhs(x):
        return np.array([x[1], -stiffness * x[0] - damping * x[1]])

    times = np.arange(0.0, horizon, dt)
    return times, rk4(rhs, np.array([1.0, 0.0]), times)


LOTKA = dict(growth=1.1, predation=0.4, death=0.4, conversion=0.1)


def lotka_volterra(_t, state, _theta=None):
    state = np.atleast_2d(state)
    prey, predator = state[:, 0], state[:, 1]
    return np.column_stack(
        [
            LOTKA["growth"] * prey - LOTKA["predation"] * prey * predator,
            -LOTKA["death"] * predator + LOTKA["conversion"] * prey * predator,
        ]
    )


def lotka_volterra_without_interaction(_t, state, _theta=None):
    """The known physics with the interaction term deliberately removed."""
    state = np.atleast_2d(state)
    return np.column_stack([LOTKA["growth"] * state[:, 0], -LOTKA["death"] * state[:, 1]])


def coefficient_of(fit, equation_index, term):
    return fit["coefficient_matrix"][fit["term_names"].index(term), equation_index]


class PolynomialLibraryTests(unittest.TestCase):
    def setUp(self):
        self.X = np.random.default_rng(0).uniform(-1.0, 1.0, size=(50, 2))

    def test_names_and_shape(self):
        library = polynomial_library(self.X, degree=2)
        self.assertEqual(library["names"], ["1", "x", "y", "x^2", "x y", "y^2"])
        self.assertEqual(library["features"].shape, (50, 6))
        np.testing.assert_allclose(library["features"][:, 4], self.X[:, 0] * self.X[:, 1])
        self.assertIn("scope", library)

    def test_interactions_can_be_excluded(self):
        library = polynomial_library(self.X, degree=3, include_interactions=False)
        self.assertEqual(library["names"], ["1", "x", "y", "x^2", "y^2", "x^3", "y^3"])

    def test_trig_and_custom_terms(self):
        library = polynomial_library(
            self.X,
            degree=1,
            include_trig=True,
            custom_terms=[("tanh(x)", lambda X: np.tanh(X[:, 0]))],
        )
        self.assertIn("sin(x)", library["names"])
        self.assertIn("cos(y)", library["names"])
        self.assertEqual(library["names"][-1], "tanh(x)")
        np.testing.assert_allclose(library["features"][:, -1], np.tanh(self.X[:, 0]))

    def test_rejects_bad_input(self):
        with self.assertRaises(Invalid):
            polynomial_library(np.array([[1.0, np.nan]]), degree=1)
        with self.assertRaises(Invalid):
            polynomial_library(self.X, degree=99)
        with self.assertRaises(Invalid):
            polynomial_library(self.X, degree=1, custom_terms=[("bad", lambda X: np.zeros(3))])
        with self.assertRaises(Invalid):
            polynomial_library(self.X, degree=1, state_names=["x", "x"])


class SindyLorenzTests(unittest.TestCase):
    """The canonical demonstration: recover the Lorenz system from clean trajectory data."""

    @classmethod
    def setUpClass(cls):
        times, states = lorenz_states()
        cls.fit = fit_sindy(states, t=times, degree=2, threshold=0.1, alpha=1e-8)

    def test_recovers_the_correct_active_terms(self):
        self.assertEqual(self.fit["active_terms"][0], ["x", "y"])
        self.assertEqual(self.fit["active_terms"][1], ["x", "y", "x z"])
        self.assertEqual(self.fit["active_terms"][2], ["z", "x y"])
        self.assertEqual(self.fit["total_active_terms"], 7)

    def test_recovers_the_correct_coefficients(self):
        self.assertAlmostEqual(coefficient_of(self.fit, 0, "x"), -10.0, delta=0.02)
        self.assertAlmostEqual(coefficient_of(self.fit, 0, "y"), 10.0, delta=0.02)
        self.assertAlmostEqual(coefficient_of(self.fit, 1, "x"), 28.0, delta=0.05)
        self.assertAlmostEqual(coefficient_of(self.fit, 1, "y"), -1.0, delta=0.02)
        self.assertAlmostEqual(coefficient_of(self.fit, 1, "x z"), -1.0, delta=0.02)
        self.assertAlmostEqual(coefficient_of(self.fit, 2, "z"), -8.0 / 3.0, delta=0.02)
        self.assertAlmostEqual(coefficient_of(self.fit, 2, "x y"), 1.0, delta=0.02)

    def test_renders_readable_equations_and_reports_scope(self):
        self.assertTrue(self.fit["equations"][0].startswith("dx/dt = "))
        self.assertIn(" x y", self.fit["equations"][2])
        self.assertTrue(self.fit["converged"])
        self.assertGreater(min(self.fit["r_squared"]), 0.999)
        self.assertIn("Not a derivation", self.fit["scope"])

    def test_rendering_is_independent_of_the_fit(self):
        rendered = render_equation([-0.1, 0.0, 0.002], ["x", "y", "x y"], lhs="dx/dt")
        self.assertEqual(rendered["equation"], "dx/dt = -0.100 x + 0.002 x y")
        self.assertEqual(rendered["n_active_terms"], 2)
        self.assertEqual(render_equation([0.0, 0.0], ["x", "y"])["equation"], "dx/dt = 0")


class SindyOscillatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.times, cls.states = oscillator_states()
        cls.noisy = cls.states + np.random.default_rng(4).normal(0.0, 0.02, cls.states.shape)

    def test_recovers_linear_damped_oscillator(self):
        fit = fit_sindy(self.states, t=self.times, degree=3, threshold=0.05)
        self.assertEqual(fit["active_terms"], [["y"], ["x", "y"]])
        self.assertAlmostEqual(coefficient_of(fit, 0, "y"), 1.0, delta=1e-3)
        self.assertAlmostEqual(coefficient_of(fit, 1, "x"), -2.0, delta=1e-3)
        self.assertAlmostEqual(coefficient_of(fit, 1, "y"), -0.3, delta=1e-3)

    def test_smoothed_derivatives_beat_naive_differencing_under_noise(self):
        truth = np.column_stack([self.states[:, 1], -2.0 * self.states[:, 0] - 0.3 * self.states[:, 1]])
        naive = estimate_derivatives(self.noisy, t=self.times, method="finite_difference")
        smooth = estimate_derivatives(
            self.noisy, t=self.times, method="savitzky_golay", window=61, polyorder=3
        )
        trim = smooth["suggested_edge_trim"]
        naive_error = np.sqrt(np.mean((naive["derivatives"][trim:-trim] - truth[trim:-trim]) ** 2))
        smooth_error = np.sqrt(np.mean((smooth["derivatives"][trim:-trim] - truth[trim:-trim]) ** 2))
        self.assertLess(smooth_error, 0.2 * naive_error)
        self.assertIn("Endpoint derivatives", " ".join(naive["notes"]))

    def test_stability_selection_separates_true_terms_from_noise_fitted_ones(self):
        report = stability_selection(
            self.noisy,
            t=self.times,
            degree=3,
            threshold=0.05,
            derivative_method="savitzky_golay",
            window=61,
            polyorder=3,
            n_bootstrap=25,
            sample_fraction=0.6,
            seed=11,
        )
        probabilities = {
            (entry["equation"], entry["term"]): entry["inclusion_probability"]
            for entry in report["terms"]
        }
        for key in (("dx/dt", "y"), ("dy/dt", "x"), ("dy/dt", "y")):
            self.assertGreaterEqual(probabilities[key], 0.9, key)
        for key in (("dx/dt", "1"), ("dx/dt", "x"), ("dx/dt", "x^2"), ("dy/dt", "x y"), ("dy/dt", "y^3")):
            self.assertLessEqual(probabilities[key], 0.5, key)
        self.assertIn("Not a posterior probability", report["scope"])
        self.assertEqual(report["n_bootstrap"], 25)

    def test_stability_selection_is_deterministic_under_a_seed(self):
        options = dict(t=self.times, degree=2, threshold=0.05, n_bootstrap=6, seed=3)
        first = stability_selection(self.noisy, **options)
        second = stability_selection(self.noisy, **options)
        self.assertEqual(first["inclusion_probability"], second["inclusion_probability"])

    def test_pareto_sweep_is_monotone_in_active_term_count(self):
        sweep = pareto_sweep(
            self.states, [0.001, 0.01, 0.05, 0.2, 1.0, 5.0], t=self.times, degree=3
        )
        counts = sweep["active_term_counts"]
        self.assertTrue(all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1)), counts)
        self.assertTrue(sweep["monotone_in_active_terms"])
        self.assertEqual(counts[-1], 0)
        self.assertIsNotNone(sweep["suggested_threshold"])

    def test_weak_form_survives_noise_without_differentiating(self):
        fit = weak_form_sindy(
            self.noisy,
            t=self.times,
            degree=3,
            threshold=0.05,
            n_test_functions=120,
            support_points=30,
        )
        self.assertEqual(fit["formulation"], "weak")
        self.assertEqual(fit["active_terms"], [["y"], ["x", "y"]])
        self.assertAlmostEqual(coefficient_of(fit, 1, "x"), -2.0, delta=0.05)
        self.assertIn("correlated", " ".join(fit["warnings"]))


class SparseRegressionTests(unittest.TestCase):
    def test_stlsq_reports_support_and_convergence(self):
        rng = np.random.default_rng(2)
        Theta = rng.normal(size=(200, 5))
        target = 3.0 * Theta[:, 1] - 2.0 * Theta[:, 3]
        result = stlsq(Theta, target, threshold=0.1, alpha=1e-8)
        self.assertEqual(result["n_active"], [2])
        self.assertTrue(result["converged"])
        np.testing.assert_allclose(
            result["coefficient_matrix"][:, 0], [0.0, 3.0, 0.0, -2.0, 0.0], atol=1e-8
        )

    def test_stlsq_rejects_mismatched_shapes_and_bad_parameters(self):
        Theta = np.ones((10, 2))
        with self.assertRaises(Invalid):
            stlsq(Theta, np.ones(9))
        with self.assertRaises(Invalid):
            stlsq(Theta, np.ones(10), alpha=-1.0)
        with self.assertRaises(Invalid):
            stlsq(Theta, np.ones(10), max_iter=0)

    def test_total_variation_derivative_is_available_and_bounded(self):
        times = np.arange(0.0, 6.0, 0.02)
        signal = np.sin(times).reshape(-1, 1)
        noisy = signal + np.random.default_rng(1).normal(0.0, 0.02, signal.shape)
        result = estimate_derivatives(noisy, t=times, method="total_variation", tv_alpha=0.05, tv_iterations=15)
        error = np.sqrt(np.mean((result["derivatives"][:, 0] - np.cos(times)) ** 2))
        self.assertLess(error, 0.2)
        with self.assertRaises(Invalid):
            estimate_derivatives(np.zeros((2000, 1)), dt=0.01, method="total_variation")


class GaussianProcessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_x = np.linspace(0.0, 1.0, 25).reshape(-1, 1)
        cls.train_y = np.sin(2.0 * np.pi * cls.train_x).ravel()
        cls.emulator = GPEmulator(kernel="matern52", seed=1, max_iterations=100)
        cls.info = cls.emulator.fit(cls.train_x, cls.train_y)

    def test_interpolates_a_known_smooth_function(self):
        query = np.linspace(0.02, 0.98, 60).reshape(-1, 1)
        prediction = self.emulator.predict(query)
        error = np.abs(prediction["mean"] - np.sin(2.0 * np.pi * query).ravel()).max()
        self.assertLess(error, 5e-3)
        self.assertTrue(self.info["converged"])
        self.assertIn("scope", self.info)

    def test_variance_is_small_at_training_points_and_large_far_away(self):
        at_training = self.emulator.predict(self.train_x)["variance"].max()
        midway = self.emulator.predict(np.array([[0.5]]))["variance"][0]
        far = self.emulator.predict(np.array([[8.0]]))["variance"][0]
        self.assertLess(at_training, 1e-4)
        self.assertLess(at_training, midway + 1e-12)
        self.assertGreater(far, 100.0 * max(midway, 1e-12))
        self.assertGreater(far, 0.1)

    def test_prediction_carries_the_support_verdict(self):
        prediction = self.emulator.predict(np.array([[0.5], [8.0]]))
        self.assertEqual(prediction["support_status"], ["interior", "extrapolation"])
        self.assertEqual(prediction["n_extrapolating"], 1)
        self.assertIn("unsupported", prediction["scope"])

    def test_both_kernels_fit_and_the_matrix_helper_agrees(self):
        rbf = GPEmulator(kernel="rbf", seed=2, max_iterations=60)
        report = rbf.fit(self.train_x, self.train_y)
        self.assertLess(report["train_rmse"], 1e-3)
        matrix = kernel_matrix(self.train_x[:3], self.train_x[:3], kernel="rbf", length_scale=0.5)
        np.testing.assert_allclose(np.diag(matrix["matrix"]), np.ones(3))

    def test_enforces_the_cubic_cost_cap_and_input_checks(self):
        with self.assertRaises(Invalid):
            GPEmulator().fit(np.zeros((401, 1)), np.zeros(401))
        with self.assertRaises(Invalid):
            GPEmulator().fit(self.train_x, self.train_y[:-1])
        with self.assertRaises(Invalid):
            GPEmulator(kernel="cubic")
        with self.assertRaises(Invalid):
            GPEmulator().predict(self.train_x)


class TrainingSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        axis = np.linspace(0.0, 1.0, 6)
        cls.grid = np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
        cls.support = TrainingSupport(cls.grid)

    def test_classifies_interior_boundary_and_extrapolation(self):
        self.assertEqual(self.support.method, "convex_hull")
        self.assertEqual(self.support.in_support([0.5, 0.5])["status"], "interior")
        self.assertEqual(self.support.in_support([1.0, 0.5])["status"], "boundary")
        extrapolated = self.support.in_support([3.0, 3.0])
        self.assertEqual(extrapolated["status"], "extrapolation")
        self.assertGreater(extrapolated["distance_outside"], 1.0)
        self.assertLess(self.support.in_support([0.5, 0.5])["signed_distance"], 0.0)

    def test_batch_classification_and_summary(self):
        report = self.support.classify(np.array([[0.5, 0.5], [1.0, 0.5], [3.0, 3.0]]))
        self.assertEqual(report["counts"], {"interior": 1, "boundary": 1, "extrapolation": 1})
        self.assertIn("does not certify accuracy", report["scope"])
        summary = self.support.summary()
        self.assertEqual(summary["n_training_points"], 36)
        self.assertEqual(summary["lower"], [0.0, 0.0])

    def test_high_dimensional_support_falls_back_to_the_bounding_box(self):
        points = np.random.default_rng(0).uniform(0.0, 1.0, size=(40, 8))
        support = TrainingSupport(points)
        self.assertEqual(support.method, "bounding_box")
        self.assertIn("outer approximation", support.summary()["note"])
        self.assertEqual(support.in_support(np.full(8, 5.0))["status"], "extrapolation")

    def test_dimension_mismatch_is_rejected(self):
        with self.assertRaises(Invalid):
            self.support.in_support([0.5, 0.5, 0.5])


class NeuralSurrogateTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.X = rng.uniform(-1.0, 1.0, size=(40, 2))
        self.y = (self.X[:, 0] * self.X[:, 1]).reshape(-1, 1)

    def test_analytic_gradient_matches_finite_differences(self):
        for activation in ("tanh", "relu"):
            with self.subTest(activation=activation):
                network = NeuralSurrogate(
                    hidden=(5, 4), activation=activation, weight_decay=1e-3, seed=3
                )
                check = network.gradient_check(self.X, self.y, epsilon=1e-6, tolerance=1e-6)
                self.assertLess(check["max_relative_difference"], 1e-6)
                self.assertTrue(check["passed"])
                self.assertEqual(check["n_parameters_checked"], check["n_parameters"])
                self.assertGreater(check["analytic_gradient_norm"], 0.0)

    def test_training_reduces_error_and_reports_support(self):
        network = NeuralSurrogate(hidden=(12,), seed=5, max_iterations=400)
        report = network.fit(self.X, self.y)
        self.assertLess(report["train_rmse"], 0.05)
        self.assertEqual(report["architecture"], [2, 12, 1])
        prediction = network.predict(np.array([[0.0, 0.0], [9.0, 9.0]]))
        self.assertEqual(prediction["support_status"][1], "extrapolation")
        self.assertIn("scope", prediction)

    def test_budget_exhaustion_is_reported_rather_than_hidden(self):
        network = NeuralSurrogate(hidden=(12,), seed=5, max_iterations=2)
        report = network.fit(self.X, self.y)
        self.assertFalse(report["converged"])
        self.assertTrue(any("budget" in w for w in report["warnings"]))

    def test_parameter_cap_and_input_checks(self):
        with self.assertRaises(Invalid):
            NeuralSurrogate(hidden=(200, 200, 200)).fit(
                np.zeros((10, 32)), np.zeros((10, 1))
            )
        with self.assertRaises(Invalid):
            NeuralSurrogate(activation="swish")
        with self.assertRaises(Invalid):
            NeuralSurrogate().fit(self.X, self.y[:-1])


class ApproximationErrorTests(unittest.TestCase):
    def test_error_outside_the_hull_exceeds_error_inside(self):
        def target(X):
            return np.sin(3.0 * X[:, 0]) + 0.5 * X[:, 0] ** 2

        rng = np.random.default_rng(9)
        train_x = rng.uniform(0.0, 1.0, size=(60, 1))
        network = NeuralSurrogate(hidden=(20,), seed=5, max_iterations=500)
        network.fit(train_x, target(train_x))
        test_x = np.concatenate(
            [rng.uniform(0.1, 0.9, size=(40, 1)), rng.uniform(1.5, 3.0, size=(40, 1))]
        )
        report = approximation_error(
            network.forward(test_x).ravel(), target(test_x), inputs=test_x, support=network.support
        )
        inside = report["by_support"]["interior"]
        outside = report["by_support"]["extrapolation"]
        self.assertEqual(inside["n"], 40)
        self.assertEqual(outside["n"], 40)
        self.assertGreater(outside["rmse"], 10.0 * inside["rmse"])
        self.assertGreater(report["extrapolation_error_ratio"], 10.0)
        self.assertIn("not an upper bound on extrapolation error", report["scope"])

    def test_aggregate_only_when_no_support_is_supplied(self):
        report = approximation_error([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])
        self.assertIsNone(report["by_support"])
        self.assertAlmostEqual(report["max_absolute_error"], 1.0)
        with self.assertRaises(Invalid):
            approximation_error([1.0, 2.0], [1.0])


class UniversalDifferentialEquationTests(unittest.TestCase):
    """The flagship: learn the removed Lotka-Volterra interaction and read it back out."""

    @classmethod
    def setUpClass(cls):
        cls.times = np.arange(0.0, 8.0, 0.05)
        cls.states = simulate_ude(cls.times, np.array([2.0, 1.0]), lotka_volterra)["trajectory"]
        started = time.perf_counter()
        cls.fit = fit_ude(
            cls.times,
            cls.states,
            lotka_volterra_without_interaction,
            hidden=(6,),
            seed=1,
            n_segments=3,
            refine_iterations=20,
            collocation_iterations=250,
            state_names=["x", "y"],
        )
        cls.elapsed = time.perf_counter() - started

    def test_reference_trajectory_is_a_bounded_oscillation(self):
        self.assertGreater(self.states.min(), 0.0)
        self.assertLess(self.states.max(), 50.0)
        # The flagship fit is the heaviest case in this suite; keep it inside a stated budget.
        self.assertLess(self.elapsed, 60.0)

    def test_closure_substantially_reduces_trajectory_error(self):
        self.assertGreater(self.fit["trajectory_rmse_known_only"], 100.0)
        self.assertLess(self.fit["trajectory_rmse_hybrid"], 0.05)
        self.assertGreater(self.fit["error_reduction_fraction"], 0.99)
        self.assertFalse(self.fit["hybrid_diverged"])

    def test_multiple_shooting_ran_and_reported_its_continuity_residual(self):
        shooting = self.fit["multiple_shooting"]
        self.assertTrue(shooting["performed"])
        self.assertEqual(shooting["n_segments"], 3)
        self.assertLessEqual(shooting["objective_after"], shooting["objective_before"])
        self.assertLess(shooting["continuity_rmse"], 0.05)
        self.assertLessEqual(shooting["n_free_parameters"], 400)

    def test_symbolic_recovery_returns_the_interaction_term(self):
        recovered = symbolic_recovery(self.fit, degree=2, threshold=0.02)
        self.assertEqual(recovered["target_names"], ["g_x", "g_y"])
        for index in (0, 1):
            self.assertIn("x y", recovered["active_terms"][index])
            self.assertEqual(recovered["dominant_terms"][index][0]["term"], "x y")
        prey = recovered["coefficient_matrix"][recovered["term_names"].index("x y"), 0]
        predator = recovered["coefficient_matrix"][recovered["term_names"].index("x y"), 1]
        self.assertAlmostEqual(prey, -LOTKA["predation"], delta=0.05)
        self.assertAlmostEqual(predator, LOTKA["conversion"], delta=0.02)
        self.assertGreater(min(recovered["symbolic_r_squared"]), 0.99)
        self.assertIn("hypothesis", recovered["scope"])
        self.assertIn("not used in the fit", recovered["discriminating_test"])
        self.assertIn("f_known_x", recovered["composed_equations"][0])

    def test_learned_term_magnitude_is_reported_against_the_mechanism(self):
        contribution = self.fit["contribution"]
        self.assertEqual([entry["state"] for entry in contribution["per_state"]], ["x", "y"])
        self.assertIsNotNone(contribution["max_ratio"])
        self.assertTrue(contribution["closure_dominates"])
        self.assertTrue(
            any("driven by the learned term" in w for w in self.fit["warnings"])
        )

    def test_stability_selection_of_the_recovered_closure_is_available(self):
        recovered = symbolic_recovery(
            self.fit, degree=2, threshold=0.02, stability=True, n_bootstrap=10, seed=5
        )
        probabilities = {
            (entry["equation"], entry["term"]): entry["inclusion_probability"]
            for entry in recovered["stability"]["terms"]
        }
        self.assertGreaterEqual(probabilities[("dg_x/dt", "x y")], 0.9)

    def test_simulation_rejects_a_nonmonotone_grid_and_reports_divergence(self):
        with self.assertRaises(Invalid):
            simulate_ude(np.array([0.0, 1.0, 0.5]), np.array([1.0, 1.0]), lotka_volterra)
        blowup = simulate_ude(
            np.linspace(0.0, 200.0, 400),
            np.array([1.0]),
            lambda _t, x, _theta=None: np.atleast_2d(x) ** 3,
        )
        self.assertTrue(blowup["diverged"])
        self.assertIsNotNone(blowup["diverged_at_index"])

    def test_refinement_is_skipped_with_a_reason_for_a_nonautonomous_mechanism(self):
        fit = fit_ude(
            self.times,
            self.states,
            lotka_volterra_without_interaction,
            hidden=(4,),
            seed=1,
            collocation_iterations=40,
            autonomous=False,
            state_names=["x", "y"],
        )
        self.assertFalse(fit["multiple_shooting"]["performed"])
        self.assertIn("time-invariant", fit["multiple_shooting"]["reason"])

    def test_bad_input_is_rejected(self):
        with self.assertRaises(Invalid):
            fit_ude(self.times, self.states[:-1], lotka_volterra_without_interaction)
        with self.assertRaises(Invalid):
            fit_ude(np.geomspace(1.0, 5.0, 40), self.states[:40], lotka_volterra_without_interaction)
        with self.assertRaises(Invalid):
            symbolic_recovery({"closure": None, "closure_targets": [0], "weights": None})


class ConstraintTests(unittest.TestCase):
    def setUp(self):
        self.constraints = [
            {"kind": "conservation", "id": "mass", "indices": [0, 1, 2], "total": 1.0},
            {"kind": "non_negativity", "id": "positive", "indices": [0, 1, 2]},
        ]

    def test_projection_restores_conservation_exactly(self):
        result = project([0.5, -0.1, 0.4], self.constraints)
        self.assertAlmostEqual(sum(result["state"]), 1.0, places=12)
        self.assertGreaterEqual(min(result["state"]), 0.0)
        self.assertTrue(result["converged"])
        self.assertAlmostEqual(result["worst_violation_after"], 0.0, places=12)
        self.assertGreater(result["projection_magnitude"], 0.0)
        self.assertIsNotNone(result["relative_projection_magnitude"])

    def test_projection_magnitude_is_zero_for_a_feasible_state(self):
        result = project([0.2, 0.3, 0.5], self.constraints)
        self.assertAlmostEqual(result["projection_magnitude"], 0.0, places=12)

    def test_clipping_redistributes_mass_proportionally(self):
        result = project([0.5, -0.1, 0.4], self.constraints)
        self.assertAlmostEqual(result["state"][1], 0.0, places=12)
        self.assertAlmostEqual(result["state"][0] / result["state"][2], 0.5 / 0.4, places=9)

    def test_bounds_symmetry_and_stoichiometry_projections(self):
        bounded = project([2.0, -1.0], [{"kind": "bounds", "indices": [0, 1], "lower": 0.0, "upper": 1.0}])
        self.assertEqual(bounded["state"], [1.0, 0.0])
        symmetric = project([1.0, 3.0], [{"kind": "symmetry", "pairs": [[0, 1]]}])
        self.assertEqual(symmetric["state"], [2.0, 2.0])
        stoichiometric = project(
            [1.0, 1.0, 1.0],
            [{"kind": "stoichiometry", "matrix": [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]], "totals": [1.0, 1.0]}],
        )
        state = np.asarray(stoichiometric["state"])
        np.testing.assert_allclose([state[0] + state[1], state[1] + state[2]], [1.0, 1.0], atol=1e-12)

    def test_monotonicity_violation_is_detected_and_reported(self):
        trajectory = np.array([[0.0, 1.0], [1.0, 1.0], [0.5, 1.0], [2.0, 1.0]])
        declaration = [{"kind": "monotonicity", "id": "cumulative", "index": 0, "direction": "increasing"}]
        report = violation_report(trajectory, declaration)
        entry = report["constraints"][0]
        self.assertTrue(report["any_violation"])
        self.assertEqual(report["worst_constraint"], "cumulative")
        self.assertAlmostEqual(entry["max_violation"], 0.5)
        self.assertEqual(entry["violating_steps"], 1)
        self.assertEqual(entry["first_violating_step"], 1)
        repaired = project_trajectory(trajectory, declaration)
        np.testing.assert_allclose(repaired["trajectory"][:, 0], [0.0, 1.0, 1.0, 2.0])
        self.assertAlmostEqual(repaired["monotonicity_adjustment"], 0.5)
        self.assertFalse(violation_report(repaired["trajectory"], declaration)["any_violation"])

    def test_dimensional_declarations_are_carried_but_not_checked(self):
        declaration = [
            {"kind": "dimensional", "id": "units", "indices": [0], "dimension": "mass", "unit": "kg"}
        ]
        validated = validate_constraints(declaration, 2)
        self.assertEqual(validated["deferred"][0]["id"], "units")
        report = violation_report(np.zeros((3, 2)), declaration)
        self.assertEqual(report["n_deferred"], 1)
        self.assertFalse(report["constraints"][0]["checked"])

    def test_penalty_terms_expose_soft_residuals(self):
        penalties = penalty_terms(self.constraints, 3)
        self.assertAlmostEqual(penalties["penalty"]([0.2, 0.3, 0.5]), 0.0, places=12)
        self.assertGreater(penalties["penalty"]([0.5, -0.1, 0.4]), 0.0)
        residual = penalties["residual"]([0.5, -0.1, 0.4])
        self.assertGreater(residual.max(), 0.0)
        self.assertTrue(np.all(residual >= 0.0))
        weighted = penalty_terms(self.constraints, 3, weights={"mass": 4.0})
        self.assertAlmostEqual(
            weighted["penalty"]([0.5, 0.0, 0.4]), 4.0 * penalties["penalty"]([0.5, 0.0, 0.4])
        )

    def test_invalid_declarations_are_rejected(self):
        with self.assertRaises(Invalid):
            validate_constraints([{"kind": "unknown"}], 2)
        with self.assertRaises(Invalid):
            validate_constraints([{"kind": "conservation", "indices": [0, 5], "total": 1.0}], 2)
        with self.assertRaises(Invalid):
            validate_constraints([{"kind": "bounds", "indices": [0], "lower": 1.0, "upper": 0.0}], 2)
        with self.assertRaises(Invalid):
            validate_constraints(
                [{"kind": "conservation", "id": "a", "indices": [0], "total": 1.0}] * 2, 2
            )
        with self.assertRaises(Invalid):
            project([0.5, 0.5], [{"kind": "monotonicity", "index": 9, "direction": "increasing"}])


class DiscrepancyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(11)
        cls.inputs = np.linspace(0.0, 1.0, 120).reshape(-1, 1)
        axis = cls.inputs.ravel()
        cls.truth = np.sin(3.0 * axis) + 0.5 * axis**2
        cls.biased = np.sin(3.0 * axis)  # the quadratic mechanism is missing
        cls.noise_sd = 0.02
        cls.observations = cls.truth + rng.normal(0.0, cls.noise_sd, axis.size)

    def test_biased_model_yields_a_structured_residual_diagnosis(self):
        diagnosis = discrepancy_diagnosis(self.observations - self.biased, inputs=self.inputs)
        self.assertEqual(diagnosis["verdict"], "structured")
        self.assertLess(diagnosis["ljung_box_test"]["p_value"], 0.01)
        pointer = diagnosis["missing_mechanism_pointer"]
        self.assertIsNotNone(pointer)
        self.assertGreater(pointer["region_lower"], 0.5)
        self.assertIn("above the model", pointer["direction"])
        self.assertIn("missing mechanism", diagnosis["interpretation"])

    def test_correct_model_with_gaussian_noise_yields_an_unstructured_diagnosis(self):
        diagnosis = discrepancy_diagnosis(self.observations - self.truth, inputs=self.inputs)
        self.assertEqual(diagnosis["verdict"], "unstructured")
        self.assertGreater(diagnosis["runs_test"]["p_value"], 0.01)
        self.assertGreater(diagnosis["ljung_box_test"]["p_value"], 0.01)
        self.assertIsNone(diagnosis["missing_mechanism_pointer"])
        self.assertIn("not evidence that the model is correct", diagnosis["interpretation"])

    def test_fitted_discrepancy_separates_signal_from_noise(self):
        report = fit_discrepancy(
            self.biased, self.observations, self.inputs, noise_variance=self.noise_sd**2, seed=3
        )
        self.assertTrue(report["noise_variance_supplied"])
        self.assertAlmostEqual(report["noise_variance"], self.noise_sd**2, places=6)
        self.assertGreater(report["structured_fraction"], 0.9)
        self.assertLess(report["rmse_after"], 0.2 * report["rmse_before"])
        self.assertTrue(report["diagnosis"]["structured"])
        self.assertIn("in-sample", report["scope"])

    def test_error_decomposition_names_every_bucket(self):
        rng = np.random.default_rng(21)
        ensemble = self.truth[None, :] + rng.normal(0.0, 0.01, size=(20, self.truth.size))
        report = decompose_error(
            self.biased,
            self.observations,
            self.inputs,
            parameter_ensemble=ensemble,
            noise_variance=self.noise_sd**2,
            seed=3,
        )
        self.assertEqual(report["dominant_identified_source"], "structural_discrepancy")
        self.assertAlmostEqual(report["components"]["observation_noise"], self.noise_sd**2)
        self.assertGreater(report["components"]["parameter_uncertainty"], 0.0)
        self.assertLess(abs(report["unattributed"]), 0.2 * report["total_mean_squared_error"])
        self.assertTrue(report["structured_residual"])
        self.assertIn("not orthogonal", report["scope"])

    def test_missing_ensemble_is_reported_as_unavailable_not_zero(self):
        report = decompose_error(self.biased, self.observations, self.inputs, seed=3)
        self.assertIsNone(report["components"]["parameter_uncertainty"])
        self.assertIsNone(report["fractions"]["parameter_uncertainty"])
        self.assertTrue(any("not zero" in w for w in report["warnings"]))

    def test_replicate_groups_supply_an_observation_noise_estimate(self):
        rng = np.random.default_rng(31)
        inputs = np.repeat(np.linspace(0.0, 1.0, 30), 3).reshape(-1, 1)
        groups = np.repeat(np.arange(30), 3)
        truth = np.sin(3.0 * inputs.ravel())
        observations = truth + rng.normal(0.0, 0.05, truth.size)
        report = decompose_error(truth, observations, inputs, replicate_groups=groups, seed=3)
        self.assertIn("within-replicate", report["observation_noise_source"])
        self.assertAlmostEqual(report["components"]["observation_noise"], 0.05**2, delta=0.002)

    def test_bad_input_is_rejected(self):
        with self.assertRaises(Invalid):
            fit_discrepancy(self.biased, self.observations[:-1], self.inputs)
        with self.assertRaises(Invalid):
            discrepancy_diagnosis(self.observations, inputs=self.inputs, alpha=1.5)
        with self.assertRaises(Invalid):
            decompose_error(
                self.biased, self.observations, self.inputs, parameter_ensemble=np.zeros((2, 3))
            )


class ScopeDisciplineTests(unittest.TestCase):
    """Every public result must state what it does not establish."""

    def test_every_public_entry_point_returns_a_scope(self):
        rng = np.random.default_rng(0)
        times, states = oscillator_states(dt=0.05, horizon=4.0)
        inputs = np.linspace(0.0, 1.0, 24).reshape(-1, 1)
        target = np.sin(3.0 * inputs.ravel())
        emulator = GPEmulator(seed=1, max_iterations=40)
        network = NeuralSurrogate(hidden=(4,), seed=1, max_iterations=20)
        support = TrainingSupport(inputs)
        constraints = [{"kind": "non_negativity", "id": "positive", "indices": [0, 1]}]
        results = {
            "polynomial_library": polynomial_library(states, degree=2),
            "stlsq": stlsq(np.ones((20, 2)), np.ones(20)),
            "render_equation": render_equation([1.0], ["x"]),
            "estimate_derivatives": estimate_derivatives(states, t=times),
            "fit_sindy": fit_sindy(states, t=times, degree=2),
            "stability_selection": stability_selection(states, t=times, degree=2, n_bootstrap=3),
            "pareto_sweep": pareto_sweep(states, [0.05, 0.5], t=times, degree=2),
            "weak_form_sindy": weak_form_sindy(states, t=times, degree=2, support_points=5),
            "kernel_matrix": kernel_matrix(inputs, inputs),
            "gp_fit": emulator.fit(inputs, target),
            "gp_predict": emulator.predict(inputs),
            "support_classify": support.classify(inputs),
            "support_in_support": support.in_support([0.5]),
            "support_summary": support.summary(),
            "mlp_gradient_check": network.gradient_check(inputs, target),
            "mlp_fit": network.fit(inputs, target),
            "mlp_predict": network.predict(inputs),
            "approximation_error": approximation_error(target, target),
            "validate_constraints": validate_constraints(constraints, 2),
            "project": project([1.0, -1.0], constraints),
            "project_trajectory": project_trajectory(np.ones((3, 2)), constraints),
            "violation_report": violation_report(np.ones((3, 2)), constraints),
            "penalty_terms": penalty_terms(constraints, 2),
            "discrepancy_diagnosis": discrepancy_diagnosis(rng.normal(size=40)),
            "fit_discrepancy": fit_discrepancy(target, target + 0.01, inputs, seed=1),
            "decompose_error": decompose_error(target, target + 0.01, inputs, seed=1),
            "simulate_ude": simulate_ude(times, states[0], lambda _t, x, _th=None: np.atleast_2d(x) * 0.0),
        }
        for name, result in results.items():
            with self.subTest(entry_point=name):
                self.assertIsInstance(result, dict, name)
                self.assertIsInstance(result.get("scope"), str, name)
                self.assertGreater(len(result["scope"]), 40, name)


if __name__ == "__main__":
    unittest.main()
