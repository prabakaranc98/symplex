"""Deterministic checks of the dimensional stock-and-flow core; no model calls, no network."""

import math
import tempfile
import unittest

from pydantic import ValidationError

from symplex.connectors.compute import save_blob
from symplex.core.contracts import Invalid
from symplex.evaluation.numerics import NumericCheck, execute_numeric_checks
from symplex.infrastructure.storage import Store
from symplex.modeling.systems import (
    ALLOWED_FUNCTIONS,
    MAX_REPLICATES,
    MAX_STEPS,
    SystemModel,
    analyze_dimensions,
    compile_expression,
    compile_model,
    format_dimension,
    numeric_check_specs,
    parse_dimension,
    simulate,
    trajectory_csv,
)

MEANING = "Synthetic software fixture; no measured quantity is established."


def stock(identifier, initial, dimension="mass", unit="kg", non_negative=True, noise_sd=0.0):
    return {
        "id": identifier,
        "name": identifier,
        "unit": unit,
        "dimension": dimension,
        "initial": initial,
        "non_negative": non_negative,
        "noise_sd": noise_sd,
        "meaning": MEANING,
    }


def parameter(identifier, value, dimension="time^-1", unit="per_hour", lower=0.0, upper=100.0):
    return {
        "id": identifier,
        "name": identifier,
        "value": value,
        "unit": unit,
        "dimension": dimension,
        "lower": lower,
        "upper": upper,
        "meaning": MEANING,
    }


def flow(identifier, source, target, rate, dimension="mass*time^-1", delay_id=None):
    return {
        "id": identifier,
        "name": identifier,
        "from_stock": source,
        "to_stock": target,
        "rate": rate,
        "unit": "kg_per_hour",
        "dimension": dimension,
        "delay_id": delay_id,
        "meaning": MEANING,
    }


def scenario(identifier, intervention_ids=()):
    return {
        "id": identifier,
        "name": identifier,
        "intervention_ids": list(intervention_ids),
        "interpretation": MEANING,
    }


def shell(**updates):
    model = {
        "title": "Two-tank drainage fixture",
        "boundary": "A closed pair of tanks with no exchange across the boundary.",
        "time_unit": "hour",
        "time_dimension": "time",
        "horizon": 10.0,
        "stocks": [stock("tank_a", 10.0), stock("tank_b", 0.0)],
        "flows": [flow("drain", "tank_a", "tank_b", "k * tank_a")],
        "auxiliaries": [],
        "parameters": [parameter("k", 0.3)],
        "couplings": [],
        "delays": [],
        "interventions": [],
        "conservation_groups": [
            {
                "id": "total_mass",
                "name": "total mass",
                "stock_ids": ["tank_a", "tank_b"],
                "tolerance": 1e-9,
                "meaning": MEANING,
            }
        ],
        "scenarios": [scenario("baseline")],
        "assumptions": ["First-order drainage between two well-mixed tanks."],
        "unmodeled": ["Evaporation, temperature and tank geometry."],
    }
    model.update(updates)
    return model


def epidemic():
    """A conserved three-compartment fixture with a nonlinear rate and an auxiliary."""
    return {
        "title": "Three-compartment transfer fixture",
        "boundary": "A fixed population with no birth, death or migration.",
        "time_unit": "day",
        "time_dimension": "time",
        "horizon": 40.0,
        "stocks": [
            stock("susceptible", 990.0, dimension="person", unit="person"),
            stock("infected", 10.0, dimension="person", unit="person"),
            stock("recovered", 0.0, dimension="person", unit="person"),
        ],
        "flows": [
            flow(
                "infection",
                "susceptible",
                "infected",
                "beta * susceptible * prevalence",
                dimension="person*time^-1",
            ),
            flow("recovery", "infected", "recovered", "gamma * infected", dimension="person*time^-1"),
        ],
        "auxiliaries": [
            {
                "id": "prevalence",
                "name": "prevalence",
                "expression": "infected / population",
                "unit": "fraction",
                "dimension": "1",
                "meaning": MEANING,
            }
        ],
        "parameters": [
            parameter("beta", 0.35, unit="per_day"),
            parameter("gamma", 0.1, unit="per_day"),
            parameter("population", 1000.0, dimension="person", unit="person", upper=100000.0),
        ],
        "couplings": [],
        "delays": [],
        "interventions": [
            {
                "id": "distancing",
                "name": "contact reduction",
                "target_kind": "parameter",
                "target_id": "beta",
                "mode": "step",
                "time": 10.0,
                "duration": None,
                "value": 0.15,
                "absolute": True,
                "meaning": MEANING,
            }
        ],
        "conservation_groups": [
            {
                "id": "population_total",
                "name": "population",
                "stock_ids": ["susceptible", "infected", "recovered"],
                "tolerance": 1e-9,
                "meaning": MEANING,
            }
        ],
        "scenarios": [scenario("baseline"), scenario("distanced", ["distancing"])],
        "assumptions": ["Homogeneous mixing at a fixed population size."],
        "unmodeled": ["Demography, waning immunity and reporting delay."],
    }


class DimensionAlgebraTests(unittest.TestCase):
    def test_products_quotients_and_rational_exponents_round_trip(self):
        self.assertEqual(parse_dimension("1"), {})
        self.assertEqual(parse_dimension("dimensionless"), {})
        self.assertEqual(
            format_dimension(parse_dimension("mass*time^-1")), "mass*time^-1"
        )
        self.assertEqual(format_dimension(parse_dimension("mass/time")), "mass*time^-1")
        self.assertEqual(format_dimension(parse_dimension("1/time")), "time^-1")
        self.assertEqual(format_dimension(parse_dimension("length^2*length^-2")), "1")
        self.assertEqual(format_dimension(parse_dimension("length^(1/2)")), "length^(1/2)")
        self.assertEqual(format_dimension(parse_dimension("length^(-1/2)*mass")), "length^(-1/2)*mass")

    def test_malformed_dimension_strings_are_rejected(self):
        for text in ("", "Mass", "mass**time", "mass^", "mass^x", "mass time", "mass^99", "mass^(1/2", "mass)"):
            with self.assertRaises(Invalid):
                parse_dimension(text)


class ExpressionSecurityTests(unittest.TestCase):
    NAMES = {"t", "k", "tank_a"}

    def test_arithmetic_and_allowlisted_functions_evaluate(self):
        expression = compile_expression("clip(k * tank_a + exp(-t), 0, 100)", self.NAMES)
        value = expression.evaluate({"t": 0.0, "k": 2.0, "tank_a": 3.0})
        self.assertAlmostEqual(value, 7.0)
        self.assertEqual(expression.names, frozenset({"k", "tank_a", "t"}))
        self.assertIn("clip", ALLOWED_FUNCTIONS)

    def test_comparisons_and_where_are_numeric(self):
        expression = compile_expression("where(t < 1, k, 2 * k)", self.NAMES)
        self.assertAlmostEqual(expression.evaluate({"t": 0.0, "k": 5.0, "tank_a": 0.0}), 5.0)
        self.assertAlmostEqual(expression.evaluate({"t": 2.0, "k": 5.0, "tank_a": 0.0}), 10.0)

    def test_execution_bearing_syntax_is_rejected(self):
        hostile = [
            "__import__('os')",
            "__import__",
            "os.system('rm -rf /')",
            "k.__class__",
            "k.real",
            "tank_a[0]",
            "(1).__class__",
            "lambda: 1",
            "[i for i in (1, 2)]",
            "(i for i in (1, 2))",
            "{i for i in (1, 2)}",
            "{'a': 1}",
            "eval('1+1')",
            "exec('x=1')",
            "open('secret')",
            "print(k)",
            "globals()",
            "k if k > 0 else 0",
            "not k",
            "(k := 1)",
            "'abc'",
            "True",
            "None",
            "k; k",
        ]
        for text in hostile:
            with self.subTest(expression=text):
                with self.assertRaises(Invalid):
                    compile_expression(text, self.NAMES)

    def test_unknown_identifiers_and_dunder_names_are_rejected(self):
        for text in ("unknown_symbol", "k + secret", "__builtins__"):
            with self.assertRaises(Invalid):
                compile_expression(text, self.NAMES)

    def test_arithmetic_faults_raise_invalid_rather_than_propagating(self):
        for text, bindings in (
            ("k / (t - t)", {"t": 1.0, "k": 1.0, "tank_a": 0.0}),
            ("log(t)", {"t": 0.0, "k": 1.0, "tank_a": 0.0}),
            ("sqrt(-k)", {"t": 0.0, "k": 1.0, "tank_a": 0.0}),
            ("exp(k)", {"t": 0.0, "k": 1e4, "tank_a": 0.0}),
            ("k ** 1000", {"t": 0.0, "k": 2.0, "tank_a": 0.0}),
        ):
            with self.subTest(expression=text):
                with self.assertRaises(Invalid):
                    compile_expression(text, self.NAMES).evaluate(bindings)

    def test_function_arity_is_enforced(self):
        with self.assertRaises(Invalid):
            compile_expression("clip(k)", self.NAMES)
        with self.assertRaises(Invalid):
            compile_expression("exp(k, t)", self.NAMES)

    def test_the_contract_layer_also_refuses_execution_bearing_characters(self):
        for rate in ("tank_a[0]", "'abc'", "{'a': 1}", "k # comment", "k\\", "k@k"):
            with self.subTest(rate=rate):
                with self.assertRaises(ValidationError):
                    SystemModel.model_validate(shell(flows=[flow("drain", "tank_a", "tank_b", rate)]))

    def test_attribute_access_survives_the_character_filter_and_dies_at_compilation(self):
        model = shell(flows=[flow("drain", "tank_a", "tank_b", "k.real * tank_a")])
        SystemModel.model_validate(model)
        with self.assertRaises(Invalid):
            compile_model(model)


class ContractStructureTests(unittest.TestCase):
    def test_a_well_formed_model_validates(self):
        parsed = SystemModel.parse(shell())
        self.assertEqual(parsed["stocks"][0]["id"], "tank_a")
        self.assertIn("properties", SystemModel.json_schema())

    def test_identifier_collisions_and_dangling_references_are_rejected(self):
        cases = [
            shell(parameters=[parameter("tank_a", 0.3)]),
            shell(flows=[flow("drain", "tank_a", "missing", "k * tank_a")]),
            shell(flows=[flow("drain", "tank_a", "tank_a", "k * tank_a")]),
            shell(flows=[flow("drain", None, None, "k * tank_a")]),
            shell(conservation_groups=[
                {"id": "g", "name": "g", "stock_ids": ["tank_a", "missing"], "tolerance": 0.0, "meaning": MEANING}
            ]),
            shell(scenarios=[scenario("first")]),
            shell(parameters=[parameter("t", 0.3)]),
            shell(delays=[{"id": "lag", "flow_id": "drain", "kind": "fixed_lag", "duration": 1.0, "meaning": MEANING}]),
        ]
        for case in cases:
            with self.subTest(title=str(case["flows"])[:40]):
                with self.assertRaises(ValidationError):
                    SystemModel.model_validate(case)

    def test_parameter_bounds_and_intervention_shape_are_enforced(self):
        with self.assertRaises(ValidationError):
            SystemModel.model_validate(shell(parameters=[parameter("k", 5.0, lower=0.0, upper=1.0)]))
        with self.assertRaises(ValidationError):
            SystemModel.model_validate(
                shell(
                    interventions=[
                        {
                            "id": "bad",
                            "name": "bad",
                            "target_kind": "parameter",
                            "target_id": "k",
                            "mode": "pulse",
                            "time": 1.0,
                            "duration": None,
                            "value": 1.0,
                            "absolute": True,
                            "meaning": MEANING,
                        }
                    ],
                    scenarios=[scenario("baseline"), scenario("alt", ["bad"])],
                )
            )

    def test_declared_envelopes_bound_the_model(self):
        with self.assertRaises(ValidationError):
            SystemModel.model_validate(shell(stocks=[stock(f"s{i}", 1.0) for i in range(25)]))


class DimensionalAnalysisTests(unittest.TestCase):
    def test_a_consistent_model_reports_no_findings(self):
        report = analyze_dimensions(epidemic())
        self.assertTrue(report["consistent"])
        self.assertEqual(report["findings"], [])
        self.assertEqual(report["flows"][0]["inferred"], "person*time^-1")
        self.assertEqual(report["auxiliaries"][0]["inferred"], "1")

    def test_a_rate_expression_of_the_wrong_dimension_is_rejected(self):
        model = shell(flows=[flow("drain", "tank_a", "tank_b", "k * tank_a * tank_a")])
        report = analyze_dimensions(model)
        self.assertFalse(report["consistent"])
        self.assertIn("mass^2*time^-1", report["findings"][0])
        with self.assertRaises(Invalid):
            compile_model(model)
        with self.assertRaises(Invalid):
            simulate(model)

    def test_a_declared_flow_dimension_that_the_stocks_forbid_is_rejected(self):
        model = shell(flows=[flow("drain", "tank_a", "tank_b", "k * tank_a", dimension="mass")])
        with self.assertRaises(Invalid):
            compile_model(model)

    def test_incompatible_endpoint_dimensions_are_rejected(self):
        model = shell(stocks=[stock("tank_a", 10.0), stock("tank_b", 0.0, dimension="volume", unit="litre")])
        with self.assertRaises(Invalid):
            compile_model(model)

    def test_adding_terms_of_different_dimensions_is_rejected(self):
        model = shell(flows=[flow("drain", "tank_a", "tank_b", "k * (tank_a + k)")])
        with self.assertRaises(Invalid):
            compile_model(model)

    def test_a_dimensional_argument_to_exp_is_rejected(self):
        model = shell(flows=[flow("drain", "tank_a", "tank_b", "k * exp(tank_a)")])
        with self.assertRaises(Invalid):
            compile_model(model)


class IntegrationAccuracyTests(unittest.TestCase):
    def test_rk4_matches_the_analytic_exponential_decay(self):
        result = simulate(shell(), steps=200)
        analytic = 10.0 * math.exp(-0.3 * 10.0)
        self.assertLess(abs(result["scenarios"]["baseline"]["final_mean"]["tank_a"] - analytic), 1e-8)
        trajectory = result["scenarios"]["baseline"]["mean_trajectory"]["tank_a"]
        for position, moment in enumerate(result["times"]):
            self.assertLess(abs(trajectory[position] - 10.0 * math.exp(-0.3 * moment)), 1e-8)

    def test_step_refinement_reports_a_small_discretisation_difference(self):
        result = simulate(shell(), steps=100)
        refinement = result["diagnostics"]["step_refinement"]["baseline"]
        self.assertEqual(refinement["method"], "step_halving")
        self.assertEqual(refinement["control"], {"steps": 200})
        self.assertLess(refinement["max_absolute_difference"], 1e-6)
        self.assertGreater(refinement["max_absolute_difference"], 0.0)

    def test_a_coarse_grid_reports_a_visibly_larger_refinement_difference(self):
        coarse = simulate(shell(), steps=4)["diagnostics"]["step_refinement"]["baseline"]
        fine = simulate(shell(), steps=64)["diagnostics"]["step_refinement"]["baseline"]
        self.assertGreater(coarse["max_absolute_difference"], fine["max_absolute_difference"])

    def test_the_adaptive_integrator_agrees_with_the_fixed_step_integrator(self):
        fixed = simulate(shell(), steps=200)["scenarios"]["baseline"]["final_mean"]["tank_a"]
        adaptive = simulate(shell(), steps=50, integrator="rk45")["scenarios"]["baseline"]["final_mean"]["tank_a"]
        self.assertLess(abs(fixed - adaptive), 1e-6)

    def test_the_adaptive_integrator_refines_by_tightening_its_tolerance(self):
        refinement = simulate(shell(), steps=50, integrator="rk45")["diagnostics"]["step_refinement"]["baseline"]
        self.assertEqual(refinement["method"], "tolerance_tightening")
        self.assertEqual(refinement["control"], {"rtol": 1e-10, "atol": 1e-12})
        self.assertLess(refinement["max_absolute_difference"], 1e-5)

    def test_refinement_can_be_switched_off(self):
        result = simulate(shell(), steps=50, refine=False)
        self.assertEqual(result["diagnostics"]["step_refinement"], {"status": "skipped", "reason": "not requested"})

    def test_repeated_runs_of_one_model_are_identical(self):
        first, second = simulate(shell(), steps=50), simulate(shell(), steps=50)
        self.assertEqual(first["scenarios"], second["scenarios"])
        self.assertEqual(first["model_digest"], second["model_digest"])


class ConservationTests(unittest.TestCase):
    def test_a_closed_two_stock_system_conserves_its_total(self):
        result = simulate(shell(), steps=200)
        group = result["diagnostics"]["conservation"][0]
        self.assertTrue(group["structurally_closed"])
        self.assertTrue(group["closed"])
        self.assertTrue(group["within_tolerance"])
        self.assertLess(group["max_absolute_residual"], 1e-12)
        self.assertEqual(group["initial_total"], 10.0)

    def test_a_closed_three_compartment_system_conserves_its_total(self):
        result = simulate(epidemic(), steps=200)
        group = result["diagnostics"]["conservation"][0]
        self.assertLess(group["max_absolute_residual"], 1e-9)
        self.assertLess(group["max_relative_residual"], 1e-12)
        self.assertTrue(group["per_scenario"]["distanced"]["closed"])

    def test_a_boundary_flow_opens_the_group_and_the_ledger_still_balances(self):
        model = shell(
            flows=[
                flow("drain", "tank_a", "tank_b", "k * tank_a"),
                flow("leak", "tank_b", None, "k * tank_b"),
            ]
        )
        result = simulate(model, steps=200)
        group = result["diagnostics"]["conservation"][0]
        self.assertFalse(group["structurally_closed"])
        self.assertLess(group["max_absolute_residual"], 1e-9)
        self.assertLess(result["scenarios"]["baseline"]["final_mean"]["tank_b"], 10.0)

    def test_a_first_order_delay_keeps_material_in_an_accounted_pipeline(self):
        model = shell(
            flows=[flow("drain", "tank_a", "tank_b", "k * tank_a", delay_id="lag")],
            delays=[{"id": "lag", "flow_id": "drain", "kind": "first_order", "duration": 1.5, "meaning": MEANING}],
        )
        result = simulate(model, steps=200)
        group = result["diagnostics"]["conservation"][0]
        self.assertEqual(group["internal_pipeline_states"], 1)
        self.assertLess(group["max_absolute_residual"], 1e-9)
        undelayed = simulate(shell(), steps=200)["scenarios"]["baseline"]["final_mean"]["tank_b"]
        self.assertLess(result["scenarios"]["baseline"]["final_mean"]["tank_b"], undelayed)

    def test_a_fixed_lag_conserves_the_pair_and_refuses_the_adaptive_integrator(self):
        model = shell(
            flows=[flow("drain", "tank_a", "tank_b", "k * tank_a", delay_id="lag")],
            delays=[{"id": "lag", "flow_id": "drain", "kind": "fixed_lag", "duration": 1.0, "meaning": MEANING}],
        )
        result = simulate(model, steps=200)
        self.assertLess(result["diagnostics"]["conservation"][0]["max_absolute_residual"], 1e-9)
        with self.assertRaises(Invalid):
            simulate(model, steps=200, integrator="rk45")
        with self.assertRaises(Invalid):
            simulate(model, steps=8)

    def test_a_gradient_coupling_equalises_two_compartments_without_loss(self):
        model = shell(
            stocks=[stock("tank_a", 10.0), stock("tank_b", 0.0)],
            flows=[],
            couplings=[
                {
                    "id": "mix",
                    "source": "tank_a",
                    "target": "tank_b",
                    "kind": "gradient",
                    "rate_constant": 0.8,
                    "unit": "per_hour",
                    "mechanism": "diffusive exchange across a shared wall",
                }
            ],
        )
        result = simulate(model, steps=200)
        final = result["scenarios"]["baseline"]["final_mean"]
        self.assertAlmostEqual(final["tank_a"], 5.0, places=5)
        self.assertAlmostEqual(final["tank_b"], 5.0, places=5)
        self.assertLess(result["diagnostics"]["conservation"][0]["max_absolute_residual"], 1e-12)
        self.assertEqual(result["diagnostics"]["network"]["connected_components"], 1)

    def test_non_negativity_violations_are_reported_rather_than_clamped(self):
        model = shell(
            parameters=[parameter("k", 0.3), parameter("bleed", 4.0)],
            flows=[
                flow("drain", "tank_a", "tank_b", "k * tank_a"),
                flow("overdraw", "tank_a", None, "bleed * tank_b"),
            ],
        )
        result = simulate(model, steps=200)
        report = {entry["id"]: entry for entry in result["diagnostics"]["non_negativity"]}
        self.assertTrue(report["tank_a"]["violated"])
        self.assertGreater(report["tank_a"]["violation"], 0.0)
        self.assertLess(min(result["scenarios"]["baseline"]["mean_trajectory"]["tank_a"]), 0.0)


class InterventionTests(unittest.TestCase):
    def test_a_parameter_step_changes_the_trajectory(self):
        result = simulate(epidemic(), steps=200)
        baseline = result["scenarios"]["baseline"]["final_mean"]
        distanced = result["scenarios"]["distanced"]["final_mean"]
        self.assertGreater(distanced["susceptible"], baseline["susceptible"] + 1.0)
        self.assertLess(distanced["recovered"], baseline["recovered"] - 1.0)
        self.assertAlmostEqual(
            result["scenarios"]["distanced"]["paired_difference"]["recovered"],
            distanced["recovered"] - baseline["recovered"],
            places=9,
        )
        self.assertEqual(
            result["scenarios"]["baseline"]["paired_difference"],
            {"susceptible": 0.0, "infected": 0.0, "recovered": 0.0},
        )

    def test_stock_pulse_and_step_interventions_inject_accounted_material(self):
        model = shell(
            interventions=[
                {
                    "id": "top_up",
                    "name": "top up",
                    "target_kind": "stock",
                    "target_id": "tank_a",
                    "mode": "pulse",
                    "time": 1.0,
                    "duration": 2.0,
                    "value": 6.0,
                    "absolute": False,
                    "meaning": MEANING,
                },
                {
                    "id": "dump",
                    "name": "dump",
                    "target_kind": "stock",
                    "target_id": "tank_b",
                    "mode": "step",
                    "time": 4.0,
                    "duration": None,
                    "value": 3.0,
                    "absolute": False,
                    "meaning": MEANING,
                },
            ],
            scenarios=[scenario("baseline"), scenario("supplied", ["top_up", "dump"])],
        )
        result = simulate(model, steps=400)
        baseline = result["scenarios"]["baseline"]["final_mean"]
        supplied = result["scenarios"]["supplied"]["final_mean"]
        total_baseline = baseline["tank_a"] + baseline["tank_b"]
        total_supplied = supplied["tank_a"] + supplied["tank_b"]
        self.assertAlmostEqual(total_supplied - total_baseline, 9.0, places=6)
        group = result["diagnostics"]["conservation"][0]
        self.assertFalse(group["per_scenario"]["supplied"]["closed"])
        self.assertTrue(group["per_scenario"]["baseline"]["closed"])
        self.assertLess(group["max_absolute_residual"], 1e-9)

    def test_a_ramp_intervention_moves_the_parameter_between_its_endpoints(self):
        model = shell(
            interventions=[
                {
                    "id": "slow_down",
                    "name": "slow down",
                    "target_kind": "parameter",
                    "target_id": "k",
                    "mode": "ramp",
                    "time": 2.0,
                    "duration": 4.0,
                    "value": 0.05,
                    "absolute": True,
                    "meaning": MEANING,
                }
            ],
            scenarios=[scenario("baseline"), scenario("ramped", ["slow_down"])],
        )
        result = simulate(model, steps=400)
        self.assertGreater(
            result["scenarios"]["ramped"]["final_mean"]["tank_a"],
            result["scenarios"]["baseline"]["final_mean"]["tank_a"],
        )
        adaptive = simulate(model, steps=400, integrator="rk45")
        self.assertAlmostEqual(
            adaptive["scenarios"]["ramped"]["final_mean"]["tank_a"],
            result["scenarios"]["ramped"]["final_mean"]["tank_a"],
            places=5,
        )


class StochasticTests(unittest.TestCase):
    def noisy(self):
        model = epidemic()
        for entry in model["stocks"]:
            entry["noise_sd"] = 0.5
        return model

    def test_common_random_numbers_pair_scenarios_exactly(self):
        model = self.noisy()
        model["interventions"][0]["time"] = 40.0
        result = simulate(
            model, steps=100, replicates=6, seed=7, stochastic=True, integrator="euler_maruyama"
        )
        baseline = result["scenarios"]["baseline"]["final_by_replicate"]
        inert = result["scenarios"]["distanced"]["final_by_replicate"]
        self.assertEqual(baseline, inert)
        self.assertEqual(
            result["scenarios"]["distanced"]["paired_difference"],
            {"susceptible": 0.0, "infected": 0.0, "recovered": 0.0},
        )
        self.assertGreater(len(set(baseline["infected"])), 1)

    def test_an_active_intervention_moves_every_paired_replicate_in_one_direction(self):
        result = simulate(
            self.noisy(), steps=100, replicates=6, seed=7, stochastic=True, integrator="euler_maruyama"
        )
        differences = [
            distanced - baseline
            for distanced, baseline in zip(
                result["scenarios"]["distanced"]["final_by_replicate"]["susceptible"],
                result["scenarios"]["baseline"]["final_by_replicate"]["susceptible"],
            )
        ]
        self.assertTrue(all(difference > 0 for difference in differences))
        self.assertEqual(result["scenarios"]["distanced"]["fraction_above_baseline"]["susceptible"], 1.0)

    def test_stochastic_runs_replay_exactly_and_report_a_skipped_refinement(self):
        first = simulate(self.noisy(), steps=60, replicates=4, stochastic=True, integrator="euler_maruyama")
        second = simulate(self.noisy(), steps=60, replicates=4, stochastic=True, integrator="euler_maruyama")
        self.assertEqual(first["scenarios"], second["scenarios"])
        self.assertEqual(first["diagnostics"]["step_refinement"]["status"], "skipped")
        self.assertLess(first["diagnostics"]["conservation"][0]["max_absolute_residual"], 1e-9)
        self.assertFalse(first["diagnostics"]["conservation"][0]["closed"])


class EnvelopeTests(unittest.TestCase):
    def test_simulation_envelope_violations_raise_invalid(self):
        cases = [
            {"steps": 0},
            {"steps": MAX_STEPS + 1},
            {"steps": 10.0},
            {"steps": True},
            {"replicates": 0},
            {"replicates": MAX_REPLICATES + 1},
            {"replicates": 4},
            {"integrator": "midpoint"},
            {"stochastic": True},
            {"stochastic": True, "integrator": "rk4"},
            {"seed": -1},
            {"horizon": 0.0},
            {"horizon": float("inf")},
            {"rtol": 1.0},
            {"atol": 10.0},
        ]
        for case in cases:
            with self.subTest(**case):
                arguments = {"steps": 20}
                arguments.update(case)
                with self.assertRaises(Invalid):
                    simulate(shell(), **arguments)

    def test_the_total_state_sample_ceiling_is_enforced(self):
        with self.assertRaises(Invalid):
            simulate(shell(), steps=MAX_STEPS, replicates=MAX_REPLICATES, stochastic=True, integrator="euler_maruyama")

    def test_an_overflowing_rate_expression_is_reported_rather_than_propagated(self):
        model = shell(
            parameters=[parameter("k", 80.0)],
            flows=[flow("drain", "tank_a", "tank_b", "k * exp(k * t) * tank_a")],
        )
        self.assertTrue(analyze_dimensions(model)["consistent"])
        with self.assertRaises(Invalid):
            simulate(model, steps=50)

    def test_a_division_by_zero_during_integration_is_reported(self):
        model = shell(
            parameters=[parameter("k", 0.3), parameter("tau", 2.0, dimension="time", unit="hour")],
            flows=[flow("drain", "tank_a", "tank_b", "k * tank_a * tau / t")],
        )
        with self.assertRaises(Invalid):
            simulate(model, steps=50)


class CsvIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name)
        self.problem = self.store.put("workspace_problem", {"question": "Synthetic fixture"})
        self.run = self.store.put("compute_run", {"calls": [{"status": "completed"}]}, self.problem)

    def emit(self, result):
        text = trajectory_csv(result)
        return text, save_blob(
            self.store, text.encode(), "trajectory.csv", self.problem, "generated", self.run
        )

    def test_the_emitted_csv_is_tidy_and_carries_one_column_per_stock(self):
        result = simulate(epidemic(), steps=20)
        text, _ = self.emit(result)
        lines = text.splitlines()
        self.assertEqual(lines[0], "time,scenario,replicate,susceptible,infected,recovered")
        self.assertEqual(len(lines), 1 + 2 * 21)
        self.assertEqual(lines[1].split(",")[:3], ["0.0", "baseline", "0"])
        self.assertAlmostEqual(sum(float(v) for v in lines[1].split(",")[3:]), 1000.0)

    def test_generated_check_specs_validate_and_pass_the_host_numerical_checks(self):
        result = simulate(epidemic(), steps=20)
        text, ident = self.emit(result)
        specs = numeric_check_specs(result)
        self.assertEqual(
            [spec["id"] for spec in specs],
            [
                "finite_states",
                "non_negative_stocks",
                "conserved_population_total_baseline",
                "conserved_population_total_distanced",
            ],
        )
        checks = [NumericCheck.model_validate(spec) for spec in specs]
        outcome = execute_numeric_checks(self.store, self.problem, self.run, [ident], checks)
        self.assertTrue(outcome["all_passed"], outcome["checks"])
        self.assertEqual(outcome["status"], "checked")
        self.assertEqual(outcome["csv_rows_read"], 2 * 21)
        conserved = next(c for c in outcome["checks"] if c["id"] == "conserved_population_total_baseline")
        self.assertLess(conserved["observed_worst_violation"], 1e-9)
        self.assertFalse(outcome["independently_validated"])

    def test_a_monotonic_host_check_holds_for_a_one_way_stock(self):
        result = simulate(epidemic(), steps=20)
        _, ident = self.emit(result)
        check = NumericCheck.model_validate(
            {
                "id": "recovered_increases",
                "operation": "monotonic",
                "csv_filename": "trajectory.csv",
                "columns": ["recovered"],
                "group_columns": ["scenario", "replicate"],
                "time_column": "time",
                "row_filters": [],
                "lower": None,
                "upper": None,
                "reference_value": None,
                "tolerance": 0.0,
                "direction": "increasing",
                "units": "person",
                "rationale": "The recovered compartment has no outflow in this fixture.",
            }
        )
        outcome = execute_numeric_checks(self.store, self.problem, self.run, [ident], [check])
        self.assertTrue(outcome["all_passed"])

    def test_a_host_check_fails_when_the_declared_conservation_is_violated(self):
        model = epidemic()
        model["flows"].append(
            flow("import", None, "susceptible", "gamma * population", dimension="person*time^-1")
        )
        result = simulate(model, steps=20)
        _, ident = self.emit(result)
        check = NumericCheck.model_validate(
            {
                "id": "population_holds",
                "operation": "sum_conservation",
                "csv_filename": "trajectory.csv",
                "columns": ["susceptible", "infected", "recovered"],
                "group_columns": ["scenario", "replicate"],
                "time_column": "time",
                "row_filters": [],
                "lower": None,
                "upper": None,
                "reference_value": 1000.0,
                "tolerance": 1e-9,
                "direction": None,
                "units": "person",
                "rationale": "The declared population total is checked against the emitted trajectory.",
            }
        )
        outcome = execute_numeric_checks(self.store, self.problem, self.run, [ident], [check])
        self.assertFalse(outcome["all_passed"])
        self.assertEqual(outcome["status"], "failed")
        self.assertGreater(outcome["checks"][0]["observed_worst_violation"], 1.0)
        self.assertFalse(result["diagnostics"]["conservation"][0]["structurally_closed"])

    def test_csv_emission_refuses_an_oversized_trajectory(self):
        result = simulate(shell(), steps=200)
        result["trajectories"] = result["trajectories"] * 600
        with self.assertRaises(Invalid):
            trajectory_csv(result)
        with self.assertRaises(Invalid):
            trajectory_csv({"nothing": True})


class ReportedScopeTests(unittest.TestCase):
    def test_the_result_states_what_it_does_not_establish(self):
        result = simulate(shell(), steps=20)
        self.assertIn("do not establish", result["scope"].replace("they ", ""))
        self.assertFalse(result["empirically_validated"])
        self.assertFalse(result["independently_validated"])
        self.assertTrue(result["synthetic"])
        self.assertEqual(result["verdict"], "inconclusive")
        self.assertEqual(result["protocol"], "dimensional-stock-flow-v1")
        self.assertEqual(len(result["model_digest"]), 64)
        self.assertIn("no agent-authored Python is executed", result["diagnostics"]["expressions"]["execution"])
        self.assertIn("not", result["diagnostics"]["dimensions"]["scope"])

    def test_a_changed_declaration_changes_the_model_digest(self):
        first = simulate(shell(), steps=20)["model_digest"]
        second = simulate(shell(parameters=[parameter("k", 0.31)]), steps=20)["model_digest"]
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
