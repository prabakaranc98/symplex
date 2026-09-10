"""Analysis diagnostics checked against systems with known analytic answers.

Every fixture here is synthetic and deterministic. Passing tests establish that the
implementations reproduce textbook results for those fixtures; they establish nothing
about any real system.
"""

import math
import unittest

import numpy as np
from scipy.integrate import solve_ivp

from symplex.core.contracts import Invalid
from symplex.modeling.analysis import (
    attractor_embedding,
    behavior_descriptor,
    bifurcation_scan,
    descriptor_key,
    early_warning,
    feedback_loops,
    find_equilibria,
    jacobian,
    loop_dominance,
    lyapunov_max,
    lyapunov_rosenstein,
    network_metrics,
    percolation_threshold,
    regime_segments,
    stability,
)


def linear_field(matrix):
    matrix = np.asarray(matrix, dtype=float)
    return lambda t, x: matrix @ x


def damped_oscillator(zeta, omega):
    return np.array([[0.0, 1.0], [-omega * omega, -2.0 * zeta * omega]])


def lotka_volterra(a=1.5, b=1.0, c=1.0, d=3.0):
    return lambda t, x: np.array([x[0] * (a - b * x[1]), x[1] * (c * x[0] - d)])


def star_graph(leaves=4):
    size = leaves + 1
    matrix = np.zeros((size, size))
    for leaf in range(1, size):
        matrix[0, leaf] = 1.0
        matrix[leaf, 0] = 1.0
    return matrix


def signed_three_node():
    """a -| b -> a is balancing; b -> c -> b is reinforcing; a -> b -> c -| a is balancing."""
    matrix = np.zeros((3, 3))
    matrix[0, 1] = 2.0
    matrix[1, 0] = -0.5
    matrix[1, 2] = 3.0
    matrix[2, 1] = 0.5
    matrix[2, 0] = -1.0
    return matrix


def logistic_series(count=2500, burn_in=500, x0=0.2):
    value, values = x0, []
    for _ in range(count + burn_in):
        value = 4.0 * value * (1.0 - value)
        values.append(value)
    return np.array(values[burn_in:])


class StabilityTests(unittest.TestCase):
    def test_diagonal_linear_system_matches_analytic_eigenvalues(self):
        field = linear_field([[-1.0, 0.0], [0.0, -3.0]])
        derivative = jacobian(field, [0.0, 0.0])
        self.assertAlmostEqual(derivative["matrix"][0][0], -1.0, places=8)
        self.assertAlmostEqual(derivative["matrix"][1][1], -3.0, places=8)
        self.assertAlmostEqual(derivative["matrix"][0][1], 0.0, places=8)
        result = stability(derivative)
        self.assertEqual(result["classification"], "stable_node")
        self.assertTrue(result["is_stable"])
        self.assertFalse(result["oscillatory"])
        self.assertAlmostEqual(result["spectral_abscissa"], -1.0, places=8)
        self.assertAlmostEqual(result["trace"], -4.0, places=8)
        self.assertAlmostEqual(result["determinant"], 3.0, places=8)
        # The dominant mode is the slow -1 direction, carried entirely by the first state.
        self.assertAlmostEqual(result["dominant_participation"][0], 1.0, places=8)
        self.assertAlmostEqual(result["dominant_participation"][1], 0.0, places=8)
        self.assertAlmostEqual(result["eigenvalues"][0]["time_constant"], 1.0, places=6)

    def test_damped_oscillator_is_a_stable_focus_with_the_damped_period(self):
        zeta, omega = 0.15, 2.0
        result = stability(damped_oscillator(zeta, omega))
        self.assertEqual(result["classification"], "stable_focus")
        self.assertTrue(result["oscillatory"])
        expected_period = 2.0 * math.pi / (omega * math.sqrt(1.0 - zeta**2))
        self.assertAlmostEqual(result["eigenvalues"][0]["period"], expected_period, places=8)
        for ratio in result["damping_ratios"]:
            self.assertAlmostEqual(ratio, zeta, places=8)
        self.assertAlmostEqual(result["spectral_abscissa"], -zeta * omega, places=8)

    def test_saddle_unstable_and_marginal_spectra_are_named_conservatively(self):
        self.assertEqual(stability([[1.0, 0.0], [0.0, -2.0]])["classification"], "saddle")
        self.assertEqual(stability([[1.0, 0.0], [0.0, 2.0]])["classification"], "unstable_node")
        self.assertEqual(
            stability([[0.5, -2.0], [2.0, 0.5]])["classification"], "unstable_focus"
        )
        marginal = stability([[0.0, -1.0], [1.0, 0.0]])
        self.assertEqual(marginal["classification"], "center_marginal")
        self.assertFalse(marginal["is_hyperbolic"])
        self.assertFalse(marginal["is_stable"])

    def test_jacobian_rejects_a_nonfinite_state_and_a_wrong_width_field(self):
        with self.assertRaises(Invalid):
            jacobian(linear_field([[1.0]]), [float("nan")])
        with self.assertRaises(Invalid):
            jacobian(lambda t, x: np.array([1.0, 2.0]), [0.0])


class EquilibriumTests(unittest.TestCase):
    def test_lotka_volterra_interior_fixed_point_is_marginal_and_the_origin_is_a_saddle(self):
        a, b, c, d = 1.5, 1.0, 1.0, 3.0
        result = find_equilibria(
            lotka_volterra(a, b, c, d), [[0.05, 0.05], [3.5, 1.2]], eps=1e-5
        )
        self.assertEqual(result["status"], "located")
        self.assertEqual(result["equilibrium_count"], 2)
        origin, interior = result["equilibria"]
        self.assertAlmostEqual(origin["state"][0], 0.0, places=8)
        self.assertEqual(origin["stability"]["classification"], "saddle")
        self.assertAlmostEqual(interior["state"][0], d / c, places=8)
        self.assertAlmostEqual(interior["state"][1], a / b, places=8)
        self.assertEqual(interior["stability"]["classification"], "center_marginal")
        # A center has a purely imaginary pair; its period is 2*pi/sqrt(a*d).
        self.assertAlmostEqual(
            interior["stability"]["eigenvalues"][0]["period"],
            2.0 * math.pi / math.sqrt(a * d),
            places=5,
        )

    def test_duplicate_starts_collapse_to_one_equilibrium(self):
        result = find_equilibria(
            linear_field([[-1.0, 0.0], [0.0, -2.0]]), [[1.0, 1.0], [-4.0, 3.0], [0.2, 0.2]]
        )
        self.assertEqual(result["equilibrium_count"], 1)
        self.assertEqual(result["equilibria"][0]["guess_indices"], [0, 1, 2])
        self.assertEqual(result["converged_start_count"], 3)
        self.assertEqual(result["failures"], [])

    def test_no_real_root_is_reported_as_nonconvergence_not_as_absence(self):
        result = find_equilibria(lambda t, x: np.array([1.0 + x[0] ** 2]), [[0.0], [5.0]])
        self.assertEqual(result["status"], "not_converged")
        self.assertEqual(result["equilibria"], [])
        self.assertEqual(result["failure_count"], 2)
        self.assertFalse(result["absence_established"])
        for failure in result["failures"]:
            self.assertEqual(failure["status"], "not_converged")
            self.assertIsNotNone(failure["residual_norm"])

    def test_oversized_and_malformed_starts_are_rejected(self):
        field = linear_field([[-1.0]])
        with self.assertRaises(Invalid):
            find_equilibria(field, np.zeros((201, 1)))
        with self.assertRaises(Invalid):
            find_equilibria(field, [[float("inf")]])
        with self.assertRaises(Invalid):
            find_equilibria(field, [[0.0]], dimension=2)


class FeedbackLoopTests(unittest.TestCase):
    def test_signed_three_node_graph_has_known_polarity_and_gain(self):
        result = feedback_loops(signed_three_node(), ["price", "supply", "capacity"], max_len=3)
        loops = {entry["signature"]: entry for entry in result["loops"]}
        self.assertEqual(len(loops), 3)
        balancing = loops["price->supply->price"]
        self.assertEqual(balancing["polarity"], "balancing")
        self.assertAlmostEqual(balancing["gain"], -1.0, places=12)
        self.assertEqual(balancing["negative_edge_count"], 1)
        reinforcing = loops["supply->capacity->supply"]
        self.assertEqual(reinforcing["polarity"], "reinforcing")
        self.assertAlmostEqual(reinforcing["gain"], 1.5, places=12)
        triple = loops["price->supply->capacity->price"]
        self.assertEqual(triple["polarity"], "balancing")
        self.assertAlmostEqual(triple["gain"], -6.0, places=12)
        self.assertEqual(result["reinforcing_count"], 1)
        self.assertEqual(result["balancing_count"], 2)
        self.assertEqual(result["self_loop_count"], 0)

    def test_max_len_bounds_enumeration_and_self_loops_are_counted(self):
        pairs_only = feedback_loops(signed_three_node(), max_len=2)
        self.assertEqual(pairs_only["loop_count"], 2)
        self.assertEqual(pairs_only["longest_loop"], 2)
        matrix = signed_three_node()
        matrix[0, 0] = -0.25
        with_self = feedback_loops(matrix, max_len=3)
        self.assertEqual(with_self["self_loop_count"], 1)
        loop = next(entry for entry in with_self["loops"] if entry["length"] == 1)
        self.assertEqual(loop["polarity"], "balancing")
        self.assertAlmostEqual(loop["gain"], -0.25, places=12)

    def test_dominance_ranks_by_absolute_gain(self):
        loops = feedback_loops(signed_three_node(), ["a", "b", "c"], max_len=3)
        ranked = loop_dominance(loops)
        self.assertEqual(ranked["ranking"][0]["nodes"], ["a", "b", "c"])
        self.assertAlmostEqual(ranked["ranking"][0]["abs_gain"], 6.0, places=12)
        self.assertEqual(ranked["ranking"][0]["polarity"], "balancing")
        gains = [entry["abs_gain"] for entry in ranked["ranking"]]
        self.assertEqual(gains, sorted(gains, reverse=True))
        self.assertAlmostEqual(sum(entry["gain_share"] for entry in ranked["ranking"]), 1.0, places=12)

    def test_time_varying_jacobians_split_dominance_into_phases(self):
        adjacency = np.zeros((3, 3))
        adjacency[0, 1] = adjacency[1, 0] = 1.0
        adjacency[1, 2] = adjacency[2, 1] = 1.0
        loops = feedback_loops(adjacency, ["x", "y", "z"], max_len=2)
        self.assertEqual(loops["loop_count"], 2)
        early = np.zeros((3, 3))
        early[1, 0] = early[0, 1] = 2.0
        early[2, 1] = early[1, 2] = 0.5
        late = np.zeros((3, 3))
        late[1, 0] = late[0, 1] = 0.1
        late[2, 1] = late[1, 2] = 3.0
        sequence = np.stack([early, early, late])
        ranked = loop_dominance(loops, jacobian_sequence=sequence, times=[0.0, 1.0, 2.0])
        self.assertTrue(ranked["time_varying"])
        self.assertEqual(ranked["phase_count"], 2)
        first, second = ranked["phases"]
        self.assertEqual(first["nodes"], ["x", "y"])
        self.assertEqual(first["end_index"], 1)
        self.assertAlmostEqual(first["mean_abs_gain"], 4.0, places=12)
        self.assertEqual(second["nodes"], ["y", "z"])
        self.assertEqual(second["start_time"], 2.0)
        self.assertAlmostEqual(second["mean_abs_gain"], 9.0, places=12)
        self.assertEqual(second["polarity_from_jacobian"], "reinforcing")

    def test_loop_inputs_are_bounded_and_validated(self):
        with self.assertRaises(Invalid):
            feedback_loops(np.zeros((3, 3)), ["a", "b"], max_len=2)
        with self.assertRaises(Invalid):
            feedback_loops(np.zeros((3, 3)), max_len=99)
        with self.assertRaises(Invalid):
            feedback_loops(np.zeros((3, 4)))
        dense = np.ones((8, 8))
        with self.assertRaises(Invalid):
            feedback_loops(dense, max_len=8, max_cycles=10)
        with self.assertRaises(Invalid):
            loop_dominance([{"indices": [0, 1]}])


class EarlyWarningTests(unittest.TestCase):
    def rising_autocorrelation_series(self, count=600, seed=11):
        generator = np.random.default_rng(seed)
        values = np.zeros(count)
        coefficients = np.linspace(0.15, 0.97, count)
        for step in range(1, count):
            values[step] = coefficients[step] * values[step - 1] + generator.normal(0.0, 0.1)
        return values

    def test_approach_to_a_fold_shows_rising_autocorrelation_and_variance(self):
        result = early_warning(self.rising_autocorrelation_series(), window=100)
        autocorrelation = result["trend"]["lag1_autocorrelation"]
        variance = result["trend"]["variance"]
        self.assertGreater(autocorrelation["kendall_tau"], 0.5)
        self.assertLess(autocorrelation["p_value"], 0.01)
        self.assertGreater(variance["kendall_tau"], 0.4)
        self.assertTrue(result["slowing_down_consistent"])
        self.assertIn("lag1_autocorrelation", result["rising_indicators"])
        self.assertEqual(result["window_count"], 501)
        self.assertEqual(result["undefined_indicator_count"], 0)
        self.assertGreater(result["lag1_autocorrelation"][-1], result["lag1_autocorrelation"][0])

    def test_stationary_noise_does_not_show_a_consistent_trend(self):
        generator = np.random.default_rng(5)
        result = early_warning(generator.normal(0.0, 1.0, 600), window=100)
        self.assertLess(abs(result["trend"]["lag1_autocorrelation"]["kendall_tau"]), 0.9)
        self.assertFalse(
            result["slowing_down_consistent"]
            and result["trend"]["variance"]["p_value"] < 1e-40
            and result["trend"]["lag1_autocorrelation"]["p_value"] < 1e-40
        )

    def test_window_and_series_bounds_are_enforced(self):
        series = self.rising_autocorrelation_series(count=60)
        with self.assertRaises(Invalid):
            early_warning(series, window=59)
        with self.assertRaises(Invalid):
            early_warning(series, window=3, min_window=4)
        with self.assertRaises(Invalid):
            early_warning(series, window=20, detrend="quadratic")
        with self.assertRaises(Invalid):
            early_warning(np.array([1.0, float("nan"), 3.0, 4.0]))


class BifurcationTests(unittest.TestCase):
    def test_saddle_node_normal_form_is_detected_near_zero(self):
        def factory(r):
            return lambda t, x: np.array([r - x[0] ** 2])

        result = bifurcation_scan(
            factory,
            np.linspace(-0.5, 0.5, 41),
            [[0.6], [-0.6], [0.05], [-0.05]],
            parameter_name="r",
        )
        self.assertTrue(result["bifurcation_detected"])
        self.assertIn("saddle_node", result["labels_detected"])
        folds = [event for event in result["events"] if event["label"] == "saddle_node"]
        self.assertTrue(any(abs(event["parameter_estimate"]) < 0.05 for event in folds))
        # Below the fold no equilibrium is located; above it there are two.
        below = next(point for point in result["points"] if point["parameter"] < -0.2)
        above = next(point for point in result["points"] if point["parameter"] > 0.2)
        self.assertEqual(below["equilibrium_count"], 0)
        self.assertEqual(above["equilibrium_count"], 2)
        classes = {entry["classification"] for entry in above["equilibria"]}
        self.assertEqual(classes, {"stable_node", "unstable_node"})
        self.assertEqual(result["label_confidence"], "heuristic")
        self.assertLess(min(result["parameter_values_without_located_equilibria"]), 0.0)

    def test_hopf_normal_form_crossing_is_labelled_from_a_complex_pair(self):
        def factory(mu):
            def field(t, state):
                x, y = state
                radius = x * x + y * y
                return np.array([mu * x - y - x * radius, x + mu * y - y * radius])

            return field

        result = bifurcation_scan(factory, np.linspace(-0.4, 0.4, 17), [[0.0, 0.0], [0.1, 0.1]])
        crossings = [
            event for event in result["events"] if event["type"] == "spectral_abscissa_crossing"
        ]
        self.assertEqual(len(crossings), 1)
        self.assertEqual(crossings[0]["label"], "hopf")
        self.assertLess(abs(crossings[0]["parameter_estimate"]), 0.05)
        self.assertEqual(crossings[0]["direction"], "destabilizing")
        self.assertGreater(abs(crossings[0]["crossing_eigenvalue"]["imaginary"]), 0.5)

    def test_real_eigenvalue_crossing_is_labelled_transcritical_or_pitchfork(self):
        def factory(r):
            return lambda t, x: np.array([r * x[0] - x[0] ** 3])

        result = bifurcation_scan(factory, np.linspace(-0.5, 0.5, 21), [[0.0], [0.5], [-0.5]])
        crossings = [
            event for event in result["events"] if event["type"] == "spectral_abscissa_crossing"
        ]
        self.assertTrue(crossings)
        self.assertEqual(crossings[0]["label"], "transcritical_or_pitchfork")
        self.assertLess(abs(crossings[0]["parameter_estimate"]), 0.06)

    def test_sweep_inputs_are_bounded(self):
        def factory(r):
            return lambda t, x: np.array([r - x[0]])

        with self.assertRaises(Invalid):
            bifurcation_scan(factory, [0.0, 0.0, 1.0], [[0.0]])
        with self.assertRaises(Invalid):
            bifurcation_scan(factory, np.linspace(0, 1, 401), [[0.0]])
        with self.assertRaises(Invalid):
            bifurcation_scan(lambda r: "not a field", [0.0, 1.0], [[0.0]])


class RegimeSegmentTests(unittest.TestCase):
    def test_single_mean_shift_is_found_at_the_true_index(self):
        generator = np.random.default_rng(3)
        series = np.concatenate([generator.normal(0.0, 1.0, 200), generator.normal(5.0, 1.0, 200)])
        result = regime_segments(series)
        self.assertEqual(result["change_points"], [200])
        self.assertEqual(result["segment_count"], 2)
        self.assertLess(abs(result["segments"][0]["mean"]), 0.5)
        self.assertGreater(result["segments"][1]["mean"], 4.5)

    def test_variance_shift_is_found_and_homogeneous_noise_is_not_split(self):
        generator = np.random.default_rng(17)
        shifted = np.concatenate(
            [generator.normal(0.0, 0.5, 200), generator.normal(0.0, 3.0, 200)]
        )
        detected = regime_segments(shifted)
        self.assertLessEqual(detected["segment_count"], 3)
        self.assertTrue(any(abs(point - 200) < 10 for point in detected["change_points"]))
        homogeneous = generator.normal(0.0, 1.0, 400)
        self.assertEqual(regime_segments(homogeneous)["change_points"], [])

    def test_segmentation_is_deterministic_and_bounded(self):
        generator = np.random.default_rng(3)
        series = np.concatenate([generator.normal(0.0, 1.0, 150), generator.normal(3.0, 1.0, 150)])
        self.assertEqual(regime_segments(series), regime_segments(series))
        self.assertLessEqual(regime_segments(series, max_segments=2)["segment_count"], 2)
        with self.assertRaises(Invalid):
            regime_segments(series, statistic="median")
        with self.assertRaises(Invalid):
            regime_segments(series, penalty=-1.0)


class LyapunovTests(unittest.TestCase):
    def test_logistic_map_recovers_log_two(self):
        result = lyapunov_max(
            lambda t, x: 4.0 * x * (1.0 - x), [0.2], discrete=True, steps=4000, transient=200
        )
        self.assertEqual(result["status"], "estimated")
        self.assertTrue(result["positive"])
        self.assertAlmostEqual(result["lambda_max"], math.log(2.0), places=2)
        self.assertTrue(result["settled"])
        self.assertAlmostEqual(result["predictability_horizon"], 1.0 / result["lambda_max"], places=9)

    def test_stable_linear_field_has_a_negative_exponent_and_no_horizon(self):
        result = lyapunov_max(linear_field([[-1.0, 0.0], [0.0, -3.0]]), [1.0, 1.0], steps=3000)
        self.assertLess(result["lambda_max"], 0.0)
        self.assertAlmostEqual(result["lambda_max"], -1.0, places=1)
        self.assertFalse(result["positive"])
        self.assertIsNone(result["predictability_horizon"])

    def test_unbounded_orbit_is_reported_as_divergence_not_as_an_exponent(self):
        result = lyapunov_max(
            lambda t, x: np.array([x[0] ** 3 + 1.0]), [10.0], dt=0.01, steps=5000
        )
        self.assertEqual(result["status"], "diverged")
        self.assertIsNone(result["lambda_max"])
        self.assertIsNone(result["predictability_horizon"])

    def test_rosenstein_estimator_on_the_logistic_series_and_on_a_periodic_series(self):
        chaotic = lyapunov_rosenstein(logistic_series(1200, 300), dim=2, lag=1, horizon=8, fit_range=(0, 4))
        self.assertGreater(chaotic["lambda_max"], 0.4)
        self.assertLess(chaotic["lambda_max"], 1.0)
        self.assertGreater(chaotic["fit_r_squared"], 0.9)
        self.assertIsNotNone(chaotic["predictability_horizon"])
        times = np.linspace(0.0, 120.0, 1500)
        periodic = lyapunov_rosenstein(
            np.sin(times), dim=3, lag=10, dt=float(times[1] - times[0]), horizon=20
        )
        self.assertLess(abs(periodic["lambda_max"]), 0.05)
        # A periodic orbit loses no predictability inside the record it was measured on.
        horizon = periodic["predictability_horizon"]
        self.assertTrue(horizon is None or horizon > 120.0)

    def test_integration_envelope_is_bounded(self):
        field = linear_field([[-1.0]])
        with self.assertRaises(Invalid):
            lyapunov_max(field, [1.0], steps=5)
        with self.assertRaises(Invalid):
            lyapunov_max(field, [1.0], steps=400_000, transient=200_000)
        with self.assertRaises(Invalid):
            lyapunov_max(field, [1.0], delta=1.0)


class EmbeddingTests(unittest.TestCase):
    def test_sine_wave_suggests_a_two_dimensional_embedding(self):
        times = np.linspace(0.0, 100.0, 1500)
        result = attractor_embedding(np.sin(times), max_lag=32, max_dim=6)
        self.assertEqual(result["suggested_dim"], 2)
        self.assertGreater(result["suggested_lag"], 1)
        self.assertEqual(result["embedding"].shape[1], result["dim"])
        self.assertEqual(
            result["embedding"].shape[0],
            1500 - (result["dim"] - 1) * result["lag"],
        )
        self.assertLess(result["false_nearest_neighbour_fraction"][2], 0.01)
        self.assertIn("mutual information", result["lag_basis"])

    def test_logistic_map_at_lag_one_needs_one_dimension(self):
        result = attractor_embedding(logistic_series(1200, 300), lag=1, max_dim=4)
        self.assertEqual(result["lag"], 1)
        self.assertEqual(result["suggested_dim"], 1)
        self.assertLess(result["false_nearest_neighbour_fraction"][1], 0.01)

    def test_embedding_inputs_are_validated(self):
        with self.assertRaises(Invalid):
            attractor_embedding(np.linspace(0.0, 1.0, 10))
        with self.assertRaises(Invalid):
            attractor_embedding(np.sin(np.linspace(0.0, 10.0, 200)), dim=400)


class NetworkTests(unittest.TestCase):
    def test_star_graph_has_the_known_centrality_ordering(self):
        result = network_metrics(star_graph(4), ["hub", "a", "b", "c", "d"])
        nodes = {entry["label"]: entry for entry in result["nodes"]}
        self.assertEqual(nodes["hub"]["degree"], 4)
        self.assertEqual(nodes["a"]["degree"], 1)
        self.assertAlmostEqual(nodes["hub"]["betweenness"], 1.0, places=9)
        for leaf in ("a", "b", "c", "d"):
            self.assertAlmostEqual(nodes[leaf]["betweenness"], 0.0, places=9)
            self.assertAlmostEqual(
                nodes["hub"]["eigenvector_centrality"] / nodes[leaf]["eigenvector_centrality"],
                2.0,
                places=5,
            )
        self.assertTrue(result["eigenvector_centrality_converged"])
        self.assertAlmostEqual(result["spectral_radius_estimate"], 2.0, places=5)
        self.assertAlmostEqual(result["average_clustering"], 0.0, places=9)
        self.assertEqual(result["weak_component_count"], 1)
        self.assertTrue(result["symmetric"])

    def test_triangle_clustering_and_disconnected_components(self):
        triangle = np.array(
            [
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        result = network_metrics(triangle)
        nodes = result["nodes"]
        for index in range(3):
            self.assertAlmostEqual(nodes[index]["clustering"], 1.0, places=9)
        self.assertAlmostEqual(nodes[3]["clustering"], 0.0, places=9)
        self.assertAlmostEqual(result["transitivity"], 1.0, places=9)
        self.assertEqual(result["weak_component_count"], 2)
        self.assertEqual(result["largest_weak_component_size"], 3)

    def test_directed_chain_has_distinct_strong_components_and_reported_convergence(self):
        chain = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
        result = network_metrics(chain, ["source", "middle", "sink"])
        self.assertFalse(result["symmetric"])
        self.assertEqual(result["strong_component_count"], 3)
        self.assertEqual(result["weak_component_count"], 1)
        nodes = {entry["label"]: entry for entry in result["nodes"]}
        self.assertEqual(nodes["source"]["out_degree"], 1)
        self.assertEqual(nodes["source"]["in_degree"], 0)
        self.assertGreater(nodes["middle"]["betweenness"], 0.0)

    def test_percolation_estimates_on_a_star_and_on_an_edgeless_graph(self):
        result = percolation_threshold(star_graph(4))
        # <k> = 1.6, <k^2> = 4.0 so the mean-field threshold is 1.6 / 2.4.
        self.assertAlmostEqual(result["mean_field_threshold"], 1.6 / 2.4, places=9)
        self.assertAlmostEqual(result["spectral_threshold"], 0.5, places=6)
        self.assertAlmostEqual(result["spectral_radius"], 2.0, places=6)
        self.assertTrue(result["supercritical_by_molloy_reed"])
        empty = percolation_threshold(np.zeros((4, 4)))
        self.assertIsNone(empty["mean_field_threshold"])
        self.assertIsNone(empty["spectral_threshold"])
        self.assertEqual(empty["mean_field_status"], "graph has no edges")

    def test_network_inputs_are_bounded(self):
        with self.assertRaises(Invalid):
            network_metrics(np.zeros((129, 129)))
        with self.assertRaises(Invalid):
            network_metrics(np.zeros((3, 3)), ["a", "a", "b"])
        with self.assertRaises(Invalid):
            percolation_threshold(np.zeros((3, 2)))


class BehaviorDescriptorTests(unittest.TestCase):
    def monotone_trajectory(self):
        times = np.linspace(0.0, 20.0, 400)
        return times, np.exp(-0.5 * times).reshape(-1, 1)

    def oscillatory_trajectory(self, zeta=0.1, omega=2.0):
        times = np.linspace(0.0, 20.0, 400)
        damped = omega * math.sqrt(1.0 - zeta**2)
        return times, (np.exp(-zeta * omega * times) * np.cos(damped * times)).reshape(-1, 1)

    def test_same_input_gives_an_identical_key(self):
        times, states = self.oscillatory_trajectory()
        first = behavior_descriptor(times, states)
        second = behavior_descriptor(times.copy(), states.copy())
        self.assertEqual(first, second)
        self.assertEqual(descriptor_key(first)["key"], descriptor_key(second)["key"])
        self.assertTrue(first["deterministic"])

    def test_oscillatory_and_monotone_trajectories_land_in_different_niches(self):
        monotone = behavior_descriptor(*self.monotone_trajectory())
        oscillatory = behavior_descriptor(*self.oscillatory_trajectory())
        self.assertFalse(monotone["oscillatory"])
        self.assertTrue(monotone["monotone"])
        self.assertTrue(oscillatory["oscillatory"])
        self.assertFalse(oscillatory["monotone"])
        self.assertNotEqual(descriptor_key(monotone)["key"], descriptor_key(oscillatory)["key"])
        self.assertNotEqual(
            descriptor_key(monotone, "fine")["key"], descriptor_key(oscillatory, "fine")["key"]
        )

    def test_damped_oscillation_period_matches_the_analytic_damped_period(self):
        zeta, omega = 0.1, 2.0
        times, states = self.oscillatory_trajectory(zeta, omega)
        result = behavior_descriptor(times, states)
        expected = 2.0 * math.pi / (omega * math.sqrt(1.0 - zeta**2))
        self.assertAlmostEqual(result["dominant_period"] / expected, 1.0, places=2)
        self.assertEqual(result["stability_class"], "converged")
        self.assertTrue(result["settled"])
        self.assertGreater(result["overshoot_fraction"], 0.0)
        self.assertGreater(result["zero_crossings"], 4)

    def test_lotka_volterra_orbit_is_a_sustained_two_dimensional_oscillation(self):
        solution = solve_ivp(
            lotka_volterra(),
            (0.0, 30.0),
            [4.0, 2.0],
            t_eval=np.linspace(0.0, 30.0, 900),
            rtol=1e-10,
            atol=1e-12,
        )
        result = behavior_descriptor(solution.t, solution.y.T)
        self.assertEqual(result["stability_class"], "sustained_oscillation")
        self.assertTrue(result["oscillatory"])
        self.assertFalse(result["settled"])
        # A closed orbit keeps its amplitude and uses both states.
        self.assertAlmostEqual(result["amplitude_ratio"], 1.0, places=2)
        self.assertGreater(result["effective_dimension"], 1.5)
        self.assertAlmostEqual(
            result["dominant_period"], 2.0 * math.pi / math.sqrt(1.5 * 3.0), delta=0.3
        )

    def test_growing_and_settling_trajectories_are_separated(self):
        times = np.linspace(0.0, 10.0, 300)
        growing = behavior_descriptor(times, np.exp(0.9 * times).reshape(-1, 1))
        self.assertEqual(growing["stability_class"], "divergent")
        settling = behavior_descriptor(times, (1.0 - np.exp(-3.0 * times)).reshape(-1, 1))
        self.assertEqual(settling["stability_class"], "converged")
        self.assertLess(settling["settling_fraction"], 0.5)
        self.assertGreater(settling["transient_energy_ratio"], 0.9)
        self.assertNotEqual(
            descriptor_key(growing)["key"], descriptor_key(settling)["key"]
        )

    def test_nonfinite_states_are_truncated_and_reported_as_divergence(self):
        times = np.linspace(0.0, 10.0, 200)
        states = np.exp(0.9 * times).reshape(-1, 1)
        states[150:] = np.inf
        result = behavior_descriptor(times, states)
        self.assertEqual(result["stability_class"], "divergent")
        self.assertTrue(result["truncated_at_nonfinite_state"])
        self.assertEqual(result["sample_count"], 150)

    def test_descriptor_key_uses_fixed_bins_and_rejects_foreign_input(self):
        descriptor = behavior_descriptor(*self.monotone_trajectory())
        coarse = descriptor_key(descriptor)
        self.assertEqual(coarse["cell_count"], 810)
        self.assertEqual(len(coarse["coordinates"]), len(coarse["axis_names"]))
        self.assertTrue(coarse["key"].startswith("analysis-v1/coarse/"))
        self.assertNotEqual(coarse["key"], descriptor_key(descriptor, "fine")["key"])
        with self.assertRaises(Invalid):
            descriptor_key({"features": {}})
        with self.assertRaises(Invalid):
            descriptor_key(descriptor, "medium")
        with self.assertRaises(Invalid):
            descriptor_key(dict(descriptor, stability_class="unknown"))

    def test_trajectory_inputs_are_validated(self):
        times = np.linspace(0.0, 1.0, 20)
        with self.assertRaises(Invalid):
            behavior_descriptor(times, np.zeros((19, 1)))
        with self.assertRaises(Invalid):
            behavior_descriptor(times[::-1], np.zeros((20, 1)))
        with self.assertRaises(Invalid):
            behavior_descriptor(np.linspace(0.0, 1.0, 4), np.zeros((4, 1)))
        with self.assertRaises(Invalid):
            behavior_descriptor(times, np.zeros((20, 33)))


class ScopeDisciplineTests(unittest.TestCase):
    def test_every_entry_point_states_what_it_does_not_establish(self):
        times = np.linspace(0.0, 20.0, 300)
        trajectory = np.exp(-0.5 * times).reshape(-1, 1)
        field = linear_field([[-1.0, 0.0], [0.0, -2.0]])
        loops = feedback_loops(signed_three_node(), max_len=3)
        descriptor = behavior_descriptor(times, trajectory)
        generator = np.random.default_rng(2)
        results = [
            jacobian(field, [0.0, 0.0]),
            stability(damped_oscillator(0.2, 1.0)),
            find_equilibria(field, [[1.0, 1.0]]),
            loops,
            loop_dominance(loops),
            early_warning(generator.normal(0.0, 1.0, 200), window=40),
            bifurcation_scan(
                lambda r: (lambda t, x: np.array([r - x[0] ** 2])),
                np.linspace(-0.2, 0.2, 9),
                [[0.4], [-0.4]],
            ),
            regime_segments(generator.normal(0.0, 1.0, 200)),
            lyapunov_max(field, [1.0, 1.0], steps=500),
            lyapunov_rosenstein(np.sin(np.linspace(0.0, 60.0, 600)), dim=3, lag=8, horizon=10),
            attractor_embedding(np.sin(np.linspace(0.0, 60.0, 600)), max_lag=16, max_dim=4),
            network_metrics(star_graph(3)),
            percolation_threshold(star_graph(3)),
            descriptor,
            descriptor_key(descriptor),
        ]
        for result in results:
            self.assertIsInstance(result, dict)
            self.assertIsInstance(result.get("scope"), str)
            self.assertGreater(len(result["scope"]), 60)
            self.assertNotIn("proves", result["scope"].lower())


if __name__ == "__main__":
    unittest.main()
