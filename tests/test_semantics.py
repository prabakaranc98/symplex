"""Deterministic checks of the semantic layer. Synthetic fixtures; no domain claim is tested.

Every assertion here is about internal consistency of declarations. Nothing in this file
establishes that a dimension, a causal arrow or an entity match is scientifically correct.
"""

import unittest

from pydantic import ValidationError
from rdflib import Graph, Namespace

from symplex.core.contracts import Invalid
from symplex.semantics.alignment import (
    align_entities,
    align_time,
    join_risk_report,
)
from symplex.semantics.causal import (
    CausalGraph,
    acyclic_check,
    ancestors,
    backdoor_sets,
    d_separation,
    descendants,
    frontdoor_check,
    identifiable,
    instrument_check,
    testable_implications,
)
from symplex.semantics.dimensions import (
    BASE_DIMENSIONS,
    Dimension,
    Unit,
    check_expression_dimensions,
    check_unit_definition,
    check_unit_definitions,
    convert,
    parse_dimension,
    registry,
    unit,
)
from symplex.semantics.ontology import (
    Assumption,
    Claim,
    Entity,
    Measurement,
    Mechanism,
    OntologyGraph,
    Quantity,
    Source,
    consistency_report,
    from_rdf,
    graph_summary,
    project_from_complex_system,
    relation_vocabulary,
    to_rdf,
)

try:  # `unittest discover -s tests` imports test modules as top-level names.
    from tests.test_complex_system import system_spec
except ImportError:  # pragma: no cover - depends on how the suite was started
    from test_complex_system import system_spec


def measurement(ident="channel", quantity="temperature_q", status="available"):
    return Measurement(
        id=ident,
        quantity_id=quantity,
        name="Thermistor channel",
        instrument="Bench thermistor, synthetic fixture",
        procedure="Read at the declared interval; no calibration is claimed.",
        modality="sensor",
        time_support="Hourly instantaneous samples on UTC.",
        spatial_support="One bench location; no spatial extent is claimed.",
        missingness="Not characterized in this fixture.",
        uncertainty="Not characterized in this fixture.",
        unit_id="k",
        status=status,
    )


def quantity(
    ident="temperature_q", entity="bench", dimension="temperature", unit_id="k"
):
    return Quantity(
        id=ident,
        entity_id=entity,
        name="Bench temperature",
        dimension=dimension,
        unit_id=unit_id,
        observability="observed",
        meaning="Synthetic fixture quantity.",
    )


def entity(ident="bench"):
    return Entity(id=ident, name="Test bench", meaning="Synthetic fixture entity.")


def source(ident="src_fixture"):
    return Source(
        id=ident,
        citation="Synthetic fixture artifact; not a real source.",
        source_kind="artifact",
        locator="fixture://1",
    )


def claim(ident, statement="A synthetic fixture claim."):
    return Claim(
        id=ident,
        statement=statement,
        disconfirmation="Refuted if the fixture stops being a fixture.",
        status="proposed",
    )


class DimensionAlgebraTests(unittest.TestCase):
    def test_parsing_round_trips_through_the_canonical_form(self):
        cases = [
            ("mass*length^2*time^-2", "mass*length^2*time^-2"),
            ("length/time", "length*time^-1"),
            ("dimensionless", "dimensionless"),
            ("1", "dimensionless"),
            ("  Mass * LENGTH ", "mass*length"),
            ("mass/(length*time^2)", "mass*length^-1*time^-2"),
            ("(length/time)^2", "length^2*time^-2"),
            ("currency/count", "currency*count^-1"),
            ("length^0.5", "length^0.5"),
            ("length*length^-1", "dimensionless"),
        ]
        for text, canonical in cases:
            with self.subTest(text=text):
                report = parse_dimension(text)
                self.assertEqual(report["canonical"], canonical)
                self.assertTrue(report["round_trips"])
                self.assertEqual(Dimension.parse(canonical), Dimension.parse(text))
                self.assertIn("scope", report)
        self.assertEqual(len(parse_dimension("mass")["vector"]), len(BASE_DIMENSIONS))

    def test_currency_and_count_are_distinct_pseudo_dimensions(self):
        self.assertNotEqual(Dimension.parse("currency"), Dimension.parse("count"))
        self.assertNotEqual(Dimension.parse("count"), Dimension.parse("amount"))
        self.assertNotEqual(Dimension.parse("count"), Dimension.parse("dimensionless"))

    def test_algebra_composes_and_cancels(self):
        length, time = Dimension.parse("length"), Dimension.parse("time")
        speed = Dimension.parse("length/time")  # the dimension of m/s
        self.assertEqual(speed * time, length)  # m/s * s == m
        self.assertEqual(length / time, speed)
        self.assertEqual(length**3, Dimension.parse("length^3"))
        self.assertTrue((speed / speed).is_dimensionless)
        self.assertEqual(
            unit("m/s").dimension * unit("s").dimension, unit("m").dimension
        )

    def test_unregistered_and_malformed_dimensions_are_refused(self):
        for text in [
            "",
            "gribble",
            "mass**",
            "mass^",
            "mass*/time",
            "2*mass",
            "mass^99",
            "(mass",
        ]:
            with self.subTest(text=text):
                with self.assertRaises(Invalid):
                    Dimension.parse(text)

    def test_unit_definition_table_reports_equivalent_spellings(self):
        report = check_unit_definitions(
            [
                {
                    "id": "mps",
                    "symbol": "m/s",
                    "dimension": "length/time",
                    "scale_to_canonical": 1.0,
                    "offset_to_canonical": 0.0,
                },
                {
                    "id": "kph",
                    "symbol": "km/h",
                    "dimension": "length*time^-1",
                    "scale_to_canonical": 0.2777777777777778,
                    "offset_to_canonical": 0.0,
                },
            ]
        )
        self.assertEqual(report["status"], "checked")
        self.assertEqual(report["distinct_dimensions"], ["length*time^-1"])
        codes = [f["code"] for f in report["findings"]]
        self.assertIn("equivalent_dimension_written_differently", codes)
        self.assertIn("scope", report)

    def test_unparseable_declared_dimension_is_rejected_not_ignored(self):
        report = check_unit_definitions(
            [
                {
                    "id": "vibes",
                    "symbol": "v",
                    "dimension": "vibes per fortnight",
                    "scale_to_canonical": 1.0,
                    "offset_to_canonical": 0.0,
                }
            ]
        )
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["rejected"][0]["id"], "vibes")
        with self.assertRaises(Invalid):
            check_unit_definition({"id": "x"})

    def test_registry_is_declared_incomplete(self):
        report = registry()
        self.assertFalse(report["complete"])
        self.assertTrue(report["unit_count"] > 20)
        self.assertIn("scope", report)


class UnitConversionTests(unittest.TestCase):
    def test_celsius_to_kelvin_applies_its_offset(self):
        report = convert(25.0, "degC", "K")
        self.assertAlmostEqual(report["value"], 298.15, places=9)
        self.assertAlmostEqual(convert(0.0, "degC", "K")["value"], 273.15, places=9)
        self.assertAlmostEqual(convert(273.15, "K", "degC")["value"], 0.0, places=9)
        self.assertAlmostEqual(convert(212.0, "degF", "degC")["value"], 100.0, places=7)
        self.assertTrue(report["affine"])
        self.assertIn("scope", report)

    def test_kilometres_per_hour_to_metres_per_second(self):
        self.assertAlmostEqual(convert(36.0, "km/h", "m/s")["value"], 10.0, places=9)
        self.assertAlmostEqual(convert(1.0, "km", "m")["value"], 1000.0, places=9)
        self.assertAlmostEqual(convert(2.0, "h", "s")["value"], 7200.0, places=9)

    def test_conversion_between_different_dimensions_is_refused(self):
        with self.assertRaisesRegex(Invalid, "different dimensions"):
            convert(1.0, "m", "s")
        with self.assertRaisesRegex(Invalid, "different dimensions"):
            convert(1.0, "USD", "person")

    def test_affine_units_refuse_multiplicative_composition(self):
        with self.assertRaisesRegex(Invalid, "affine unit"):
            unit("degC") * unit("m")
        with self.assertRaisesRegex(Invalid, "affine unit"):
            unit("m") * unit("degC")
        with self.assertRaisesRegex(Invalid, "affine unit"):
            unit("degC") / unit("s")
        with self.assertRaisesRegex(Invalid, "affine unit"):
            unit("degF") ** 2
        composed = unit("m") / unit("s")
        self.assertEqual(composed.dimension, Dimension.parse("length/time"))
        self.assertFalse(composed.is_affine)

    def test_unit_contract_refuses_nonpositive_or_nonfinite_scale(self):
        for scale in (0.0, -1.0, float("inf")):
            with self.subTest(scale=scale):
                with self.assertRaises(Invalid):
                    Unit("bad", "b", Dimension.parse("length"), scale, 0.0)
        with self.assertRaises(Invalid):
            unit("furlong_per_fortnight")


class ExpressionDimensionTests(unittest.TestCase):
    ENV = {
        "m": "mass",
        "t": "time",
        "d": "length",
        "v": "length/time",
        "ratio": "dimensionless",
    }

    def test_products_and_quotients_accumulate_exponents(self):
        report = check_expression_dimensions("m * d**2 / t**2", self.ENV)
        self.assertEqual(report["dimension"], "mass*length^2*time^-2")
        self.assertEqual(report["symbols"], ["d", "m", "t"])
        self.assertFalse(report["executed"])
        self.assertIn("scope", report)
        self.assertEqual(
            check_expression_dimensions("v * t", self.ENV)["dimension"], "length"
        )
        self.assertTrue(
            check_expression_dimensions("d / (v * t)", self.ENV)["dimensionless"]
        )

    def test_adding_unlike_dimensions_raises(self):
        with self.assertRaisesRegex(Invalid, "add or subtract unlike dimensions"):
            check_expression_dimensions("m + t", self.ENV)
        with self.assertRaisesRegex(Invalid, "add or subtract unlike dimensions"):
            check_expression_dimensions("d - t", self.ENV)
        with self.assertRaisesRegex(Invalid, "compare unlike dimensions"):
            check_expression_dimensions("d > t", self.ENV)
        self.assertEqual(
            check_expression_dimensions("d + v * t", self.ENV)["dimension"], "length"
        )

    def test_transcendental_functions_require_dimensionless_arguments(self):
        with self.assertRaisesRegex(Invalid, "requires a dimensionless argument"):
            check_expression_dimensions("exp(5 * d)", self.ENV)
        with self.assertRaisesRegex(Invalid, "requires a dimensionless argument"):
            check_expression_dimensions("log(m)", self.ENV)
        with self.assertRaisesRegex(Invalid, "requires a dimensionless argument"):
            check_expression_dimensions("tanh(t)", self.ENV)
        self.assertTrue(
            check_expression_dimensions("exp(-t / (2 * t))", self.ENV)["dimensionless"]
        )
        self.assertEqual(
            check_expression_dimensions("sqrt(d**2)", self.ENV)["dimension"], "length"
        )

    def test_exponents_and_unknown_symbols_and_syntax_are_bounded(self):
        with self.assertRaisesRegex(Invalid, "exponent must be dimensionless"):
            check_expression_dimensions("ratio ** t", self.ENV)
        with self.assertRaisesRegex(Invalid, "dimensionless base"):
            check_expression_dimensions("d ** ratio", self.ENV)
        with self.assertRaisesRegex(Invalid, "no declared dimension"):
            check_expression_dimensions("d + unknown", self.ENV)
        with self.assertRaises(Invalid):
            check_expression_dimensions("d % t", self.ENV)
        with self.assertRaises(Invalid):
            check_expression_dimensions("__import__('os').system('true')", self.ENV)
        with self.assertRaises(Invalid):
            check_expression_dimensions("d +", self.ENV)

    def test_environment_accepts_units_and_dimension_objects(self):
        report = check_expression_dimensions(
            "speed * span",
            {"speed": unit("m/s"), "span": Dimension.parse("time")},
        )
        self.assertEqual(report["dimension"], "length")
        self.assertEqual(report["symbol_dimensions"]["speed"], "length*time^-1")


class DSeparationTests(unittest.TestCase):
    def chain(self):
        return CausalGraph.from_pairs([("x", "m"), ("m", "y")])

    def fork(self):
        return CausalGraph.from_pairs([("z", "x"), ("z", "y")])

    def collider(self):
        return CausalGraph.from_pairs([("x", "c"), ("y", "c"), ("c", "d")])

    def test_chain_is_blocked_only_by_the_mediator(self):
        graph = self.chain()
        self.assertFalse(d_separation(graph, "x", "y")["d_separated"])
        self.assertTrue(d_separation(graph, "x", "y", ["m"])["d_separated"])
        self.assertIn("scope", d_separation(graph, "x", "y"))

    def test_fork_is_blocked_only_by_the_common_cause(self):
        graph = self.fork()
        self.assertFalse(d_separation(graph, "x", "y")["d_separated"])
        self.assertTrue(d_separation(graph, "x", "y", ["z"])["d_separated"])

    def test_collider_is_closed_until_it_or_a_descendant_is_conditioned_on(self):
        graph = self.collider()
        self.assertTrue(d_separation(graph, "x", "y")["d_separated"])
        opened = d_separation(graph, "x", "y", ["c"])
        self.assertFalse(opened["d_separated"])
        self.assertIn("c", opened["colliders_opened"])
        descendant = d_separation(graph, "x", "y", ["d"])
        self.assertFalse(descendant["d_separated"])
        self.assertIn("c", descendant["colliders_opened"])

    def test_conditioning_on_a_latent_node_is_reported(self):
        graph = CausalGraph.from_pairs([("u", "x"), ("u", "y")], latent=["u"])
        report = d_separation(graph, "x", "y", ["u"])
        self.assertTrue(report["d_separated"])
        self.assertEqual(report["conditioned_on_latent"], ["u"])
        self.assertIsNotNone(report["conditioning_warning"])

    def test_ill_posed_queries_and_cycles_are_refused(self):
        graph = self.chain()
        with self.assertRaises(Invalid):
            d_separation(graph, "x", "x")
        with self.assertRaises(Invalid):
            d_separation(graph, "x", "y", ["x"])
        with self.assertRaises(Invalid):
            d_separation(graph, "x", "absent")
        with self.assertRaises(Invalid):
            d_separation(graph, "x", "y", "m")
        with self.assertRaisesRegex(Invalid, "acyclic"):
            CausalGraph.from_pairs([("a", "b"), ("b", "c"), ("c", "a")])

    def test_ancestors_descendants_and_acyclic_report(self):
        graph = self.chain()
        self.assertEqual(ancestors(graph, "y")["ancestors"], ["m", "x"])
        self.assertEqual(descendants(graph, "x")["descendants"], ["m", "y"])
        report = acyclic_check(graph)
        self.assertTrue(report["acyclic"])
        self.assertEqual(report["topological_order"], ["x", "m", "y"])
        raw = {
            "nodes": [
                {"id": n, "latent": False, "meaning": "fixture"} for n in ("a", "b")
            ],
            "edges": [
                {"source": "a", "target": "b", "sign": "positive", "mechanism": "f"},
                {"source": "b", "target": "a", "sign": "positive", "mechanism": "f"},
            ],
        }
        cyclic = acyclic_check(raw)
        self.assertFalse(cyclic["acyclic"])
        self.assertEqual(cyclic["cycle"], ["a", "b", "a"])


class IdentificationTests(unittest.TestCase):
    def confounded_triangle(self):
        return CausalGraph.from_pairs([("z", "x"), ("z", "y"), ("x", "y")])

    def test_backdoor_set_of_the_confounded_triangle_is_the_confounder(self):
        report = backdoor_sets(self.confounded_triangle(), "x", "y")
        self.assertEqual(report["adjustment_sets"], [["z"]])
        self.assertFalse(report["empty_set_sufficient"])
        self.assertTrue(report["satisfied"])
        self.assertIn("ASSUMED graph", report["scope"])

    def test_a_descendant_of_the_treatment_is_never_an_adjustment_candidate(self):
        graph = CausalGraph.from_pairs(
            [("z", "x"), ("z", "y"), ("x", "y"), ("x", "post")]
        )
        report = backdoor_sets(graph, "x", "y")
        self.assertNotIn("post", report["candidate_nodes"])
        self.assertIn("post", report["excluded_as_descendants"])
        self.assertEqual(report["adjustment_sets"], [["z"]])

    def test_randomised_treatment_needs_no_adjustment(self):
        report = backdoor_sets(CausalGraph.from_pairs([("x", "y")]), "x", "y")
        self.assertEqual(report["adjustment_sets"], [[]])
        self.assertTrue(report["empty_set_sufficient"])
        self.assertTrue(
            identifiable(CausalGraph.from_pairs([("x", "y")]), "x", "y")["identifiable"]
        )

    def test_unobserved_confounder_is_not_identifiable_by_either_criterion(self):
        graph = CausalGraph.from_pairs(
            [("u", "x"), ("u", "y"), ("x", "y")], latent=["u"]
        )
        backdoor = backdoor_sets(graph, "x", "y")
        self.assertEqual(backdoor["adjustment_sets"], [])
        self.assertEqual(backdoor["latent_confounders_of_treatment_and_outcome"], ["u"])
        report = identifiable(graph, "x", "y")
        self.assertFalse(report["identifiable"])
        self.assertEqual(report["verdict"], "not_identifiable_by_backdoor_or_frontdoor")
        self.assertIsNone(report["method"])
        self.assertFalse(report["complete_id_algorithm_applied"])

    def test_frontdoor_recovers_the_effect_through_an_observed_mediator(self):
        graph = CausalGraph.from_pairs(
            [("x", "m"), ("m", "y"), ("u", "x"), ("u", "y")], latent=["u"]
        )
        check = frontdoor_check(graph, "x", "y", ["m"])
        self.assertTrue(check["satisfied"])
        self.assertEqual(check["failing_conditions"], [])
        report = identifiable(graph, "x", "y")
        self.assertTrue(report["identifiable"])
        self.assertEqual(report["method"], "frontdoor_adjustment")

    def test_frontdoor_fails_when_the_mediator_is_bypassed_or_unmeasured(self):
        bypassed = CausalGraph.from_pairs(
            [("x", "m"), ("m", "y"), ("x", "y"), ("u", "x"), ("u", "y")], latent=["u"]
        )
        check = frontdoor_check(bypassed, "x", "y", ["m"])
        self.assertFalse(check["satisfied"])
        self.assertIn(
            "mediators_intercept_all_directed_paths", check["failing_conditions"]
        )
        unmeasured = CausalGraph.from_pairs(
            [("x", "m"), ("m", "y"), ("u", "x"), ("u", "y")], latent=["u", "m"]
        )
        self.assertIn(
            "mediators_observed",
            frontdoor_check(unmeasured, "x", "y", ["m"])["failing_conditions"],
        )
        with self.assertRaises(Invalid):
            frontdoor_check(bypassed, "x", "y", [])

    def test_an_instrument_only_buys_stronger_assumptions_not_identification(self):
        graph = CausalGraph.from_pairs(
            [("z", "x"), ("u", "x"), ("u", "y"), ("x", "y")], latent=["u"]
        )
        report = identifiable(graph, "x", "y")
        self.assertEqual(report["verdict"], "requires_stronger_assumptions")
        self.assertFalse(report["identifiable"])
        self.assertEqual(report["candidate_instruments"], ["z"])


class InstrumentTests(unittest.TestCase):
    def valid_graph(self):
        return CausalGraph.from_pairs(
            [("z", "x"), ("u", "x"), ("u", "y"), ("x", "y")], latent=["u"]
        )

    def test_a_valid_instrument_passes_all_three_conditions(self):
        report = instrument_check(self.valid_graph(), "z", "x", "y")
        self.assertTrue(report["valid_instrument"])
        self.assertEqual(report["failing_conditions"], [])
        self.assertFalse(report["point_identified"])
        self.assertIn("scope", report)

    def test_violating_the_exclusion_restriction_fails(self):
        graph = CausalGraph.from_pairs(
            [("z", "x"), ("z", "y"), ("u", "x"), ("u", "y"), ("x", "y")], latent=["u"]
        )
        report = instrument_check(graph, "z", "x", "y")
        self.assertFalse(report["valid_instrument"])
        self.assertIn("exclusion_restriction", report["failing_conditions"])

    def test_an_irrelevant_or_confounded_instrument_fails(self):
        irrelevant = CausalGraph.from_pairs(
            [("z", "w"), ("u", "x"), ("u", "y"), ("x", "y")], latent=["u"]
        )
        self.assertIn(
            "relevance",
            instrument_check(irrelevant, "z", "x", "y")["failing_conditions"],
        )
        confounded = CausalGraph.from_pairs(
            [("z", "x"), ("c", "z"), ("c", "y"), ("x", "y")]
        )
        report = instrument_check(confounded, "z", "x", "y")
        self.assertFalse(report["valid_instrument"])
        self.assertIn("independence_no_shared_confounder", report["failing_conditions"])
        self.assertTrue(
            instrument_check(confounded, "z", "x", "y", ["c"])["valid_instrument"]
        )

    def test_latent_instrument_is_refused_as_unusable(self):
        graph = CausalGraph.from_pairs(
            [("z", "x"), ("u", "x"), ("u", "y"), ("x", "y")], latent=["u", "z"]
        )
        report = instrument_check(graph, "z", "x", "y")
        self.assertIn("instrument_observed", report["failing_conditions"])
        with self.assertRaises(Invalid):
            instrument_check(graph, "x", "x", "y")


class TestableImplicationTests(unittest.TestCase):
    def test_implications_of_a_known_dag_match_the_hand_derived_list(self):
        # a -> b -> c and b -> d. Local Markov basis, derived by hand:
        #   c is independent of its non-descendants {a, d} given its parent {b}
        #   d is independent of its non-descendants {a, c} given its parent {b}
        #   a and b have no non-parent non-descendants.
        graph = CausalGraph.from_pairs([("a", "b"), ("b", "c"), ("b", "d")])
        report = testable_implications(graph)
        self.assertEqual(
            [(s["x"], s["y"], s["given"]) for s in report["implications"]],
            [("a", "c", ["b"]), ("a", "d", ["b"]), ("c", "d", ["b"])],
        )
        self.assertEqual(
            [s["statement"] for s in report["implications"]],
            ["a _||_ c | b", "a _||_ d | b", "c _||_ d | b"],
        )
        self.assertTrue(
            all(s["verified_by_d_separation"] for s in report["implications"])
        )
        self.assertFalse(report["tested_against_data"])
        self.assertIn("scope", report)

    def test_collider_implies_a_single_marginal_independence(self):
        report = testable_implications(CausalGraph.from_pairs([("x", "c"), ("y", "c")]))
        self.assertEqual(
            [(s["x"], s["y"], s["given"]) for s in report["implications"]],
            [("x", "y", [])],
        )

    def test_a_saturated_dag_implies_nothing_testable(self):
        report = testable_implications(
            CausalGraph.from_pairs([("a", "b"), ("a", "c"), ("b", "c")])
        )
        self.assertEqual(report["implications"], [])
        self.assertEqual(report["implication_count"], 0)

    def test_statements_touching_latent_nodes_are_suppressed(self):
        graph = CausalGraph.from_pairs(
            [("a", "b"), ("b", "c"), ("b", "d"), ("u", "a")], latent=["u"]
        )
        report = testable_implications(graph)
        self.assertTrue(report["suppressed_by_latent_nodes"] > 0)
        for statement in report["implications"]:
            self.assertNotIn("u", [statement["x"], statement["y"], *statement["given"]])


class OntologyTypingTests(unittest.TestCase):
    def graph(self):
        graph = OntologyGraph("fixture")
        graph.add_nodes(
            [entity(), quantity(), measurement(), source(), claim("claim_a")]
        )
        graph.add_relations(
            [
                ("bench", "has_quantity", "temperature_q"),
                ("temperature_q", "measured_by", "channel"),
                ("channel", "derived_from", "src_fixture"),
                ("src_fixture", "supports", "claim_a"),
            ]
        )
        return graph

    def test_ill_typed_triples_are_rejected_rather_than_stored(self):
        graph = self.graph()
        for subject, predicate, object_ in [
            ("temperature_q", "has_quantity", "bench"),
            ("bench", "measured_by", "channel"),
            ("bench", "influences", "temperature_q"),
            ("src_fixture", "contradicts", "claim_a"),
            ("bench", "same_as", "temperature_q"),
            ("channel", "supports", "src_fixture"),
        ]:
            with self.subTest(predicate=predicate):
                with self.assertRaisesRegex(Invalid, "does not accept"):
                    graph.add_relation(subject, predicate, object_)
        with self.assertRaisesRegex(Invalid, "vocabulary is closed"):
            graph.add_relation("bench", "causes", "temperature_q")
        with self.assertRaises(Invalid):
            graph.add_relation("bench", "has_quantity", "absent")
        with self.assertRaises(Invalid):
            graph.add_relation("claim_a", "contradicts", "claim_a")
        self.assertEqual(len(graph.relations), 4)

    def test_node_contracts_reject_malformed_nodes(self):
        graph = OntologyGraph("fixture")
        with self.assertRaises(Invalid):
            graph.add_node({"node_kind": "quantity", "id": "q"})
        with self.assertRaises(Invalid):
            graph.add_node({"node_kind": "wormhole", "id": "q"})
        with self.assertRaises(Invalid):
            graph.add_node(
                {
                    **quantity().model_dump(),
                    "node_kind": "quantity",
                    "dimension": "vibes",
                }
            )
        # Constructed directly, the contract surfaces the same refusal as a pydantic error.
        with self.assertRaisesRegex(ValidationError, "Unknown base dimension"):
            quantity(dimension="vibes")
        graph.add_node(entity())
        with self.assertRaisesRegex(Invalid, "Duplicate"):
            graph.add_node(entity())

    def test_consistency_report_finds_a_quantity_with_no_measurement(self):
        graph = OntologyGraph("fixture")
        graph.add_nodes([entity(), quantity()])
        graph.add_relation("bench", "has_quantity", "temperature_q")
        report = consistency_report(graph)
        codes = [f["code"] for f in report["findings"]]
        self.assertIn("quantity_without_measurement_channel", codes)
        self.assertTrue(report["consistent"])
        self.assertIn("scope", report)
        graph.add_node(measurement())
        graph.add_relation("temperature_q", "measured_by", "channel")
        codes = [f["code"] for f in consistency_report(graph)["findings"]]
        self.assertNotIn("quantity_without_measurement_channel", codes)
        self.assertIn("measurement_without_source", codes)

    def test_consistency_report_finds_orphan_entities(self):
        graph = OntologyGraph("fixture")
        graph.add_node(entity("lonely"))
        codes = [f["code"] for f in consistency_report(graph)["findings"]]
        self.assertIn("orphan_entity", codes)

    def test_same_as_component_merging_two_dimensions_is_an_error(self):
        graph = OntologyGraph("fixture")
        graph.add_nodes(
            [
                entity(),
                quantity("temperature_q"),
                quantity("length_q", dimension="length", unit_id="m"),
                quantity("bridge_q", dimension="temperature"),
            ]
        )
        graph.add_relations(
            [
                ("temperature_q", "same_as", "bridge_q"),
                ("bridge_q", "same_as", "length_q"),
            ]
        )
        report = consistency_report(graph)
        finding = next(
            f
            for f in report["findings"]
            if f["code"] == "same_as_merges_different_dimensions"
        )
        self.assertEqual(
            finding["component"], ["bridge_q", "length_q", "temperature_q"]
        )
        self.assertFalse(report["consistent"])
        self.assertEqual(report["status"], "failed")

    def test_contradicting_claims_sharing_a_source_are_reported(self):
        graph = OntologyGraph("fixture")
        graph.add_nodes([source(), claim("claim_a"), claim("claim_b")])
        graph.add_relations(
            [
                ("claim_a", "derived_from", "src_fixture"),
                ("claim_b", "derived_from", "src_fixture"),
                ("claim_a", "contradicts", "claim_b"),
            ]
        )
        finding = next(
            f
            for f in consistency_report(graph)["findings"]
            if f["code"] == "contradicting_claims_share_a_source"
        )
        self.assertEqual(finding["shared_sources"], ["src_fixture"])

    def test_mechanism_endpoints_are_dimensionally_checked(self):
        graph = OntologyGraph("fixture")
        graph.add_nodes(
            [
                entity(),
                quantity("stock_q", dimension="count", unit_id="person"),
                quantity("flow_q", dimension="count/time"),
                quantity("mass_q", dimension="mass", unit_id="kg"),
                Mechanism(
                    id="good_rate",
                    name="Outflow",
                    description="Stock drains at a rate.",
                    input_quantity_ids=["stock_q"],
                    output_quantity_ids=["flow_q"],
                    dimensional_relation="rate_per_time",
                ),
                Mechanism(
                    id="bad_identity",
                    name="Broken identity",
                    description="Claims the output equals the input.",
                    input_quantity_ids=["stock_q"],
                    output_quantity_ids=["mass_q"],
                    dimensional_relation="identity",
                ),
            ]
        )
        codes = [
            (f["code"], f["subject"]) for f in consistency_report(graph)["findings"]
        ]
        self.assertIn(("mechanism_endpoint_dimension_mismatch", "bad_identity"), codes)
        self.assertNotIn(("mechanism_endpoint_dimension_mismatch", "good_rate"), codes)

    def test_quantity_unit_dimension_mismatch_is_an_error(self):
        graph = OntologyGraph("fixture")
        graph.add_nodes([entity(), quantity(dimension="length", unit_id="kg")])
        codes = [f["code"] for f in consistency_report(graph)["findings"]]
        self.assertIn("quantity_unit_dimension_mismatch", codes)

    def test_relation_vocabulary_is_closed_and_documented(self):
        vocabulary = relation_vocabulary()
        self.assertTrue(vocabulary["closed"])
        self.assertEqual(
            sorted(r["predicate"] for r in vocabulary["relations"]),
            [
                "contradicts",
                "derived_from",
                "has_quantity",
                "influences",
                "measured_by",
                "part_of",
                "same_as",
                "supports",
            ],
        )


class RdfProjectionTests(unittest.TestCase):
    def rich_graph(self):
        graph = OntologyGraph("fixture")
        graph.add_nodes(
            [
                entity(),
                quantity("temperature_q"),
                quantity("flow_q", dimension="length^3/time"),
                measurement(),
                source(),
                claim("claim_a"),
                claim("claim_b"),
                Assumption(
                    id="assumption_0",
                    statement="The bench is thermally isolated.",
                    scope_note="Declared, not checked.",
                    status="asserted",
                ),
                Mechanism(
                    id="mech",
                    name="Thermal exchange",
                    description="A proposed exchange mechanism.",
                    input_quantity_ids=["temperature_q"],
                    output_quantity_ids=["flow_q"],
                    dimensional_relation="unspecified",
                ),
            ]
        )
        graph.add_relations(
            [
                ("bench", "has_quantity", "temperature_q"),
                ("bench", "has_quantity", "flow_q"),
                ("temperature_q", "measured_by", "channel"),
                ("temperature_q", "influences", "flow_q"),
                ("channel", "derived_from", "src_fixture"),
                ("claim_a", "contradicts", "claim_b"),
                ("src_fixture", "supports", "claim_a"),
                ("assumption_0", "supports", "claim_b"),
            ]
        )
        return graph

    def test_rdf_round_trip_preserves_the_graph(self):
        graph = self.rich_graph()
        projected = to_rdf(graph)
        self.assertTrue(projected["triple_count"] > len(graph.nodes))
        self.assertFalse(projected["reasoner_applied"])
        self.assertIn("scope", projected)
        restored = from_rdf(projected["turtle"])["graph"]
        self.assertEqual(restored.snapshot(), graph.snapshot())
        self.assertEqual(restored.digest(), graph.digest())
        self.assertEqual(
            graph_summary(restored)["relations_by_predicate"],
            graph_summary(graph)["relations_by_predicate"],
        )
        self.assertEqual(to_rdf(restored)["turtle"], projected["turtle"])

    def test_projection_uses_the_shared_namespace_and_standard_terms(self):
        turtle = to_rdf(self.rich_graph())["turtle"]
        for fragment in [
            "urn:symplex:",
            "http://www.w3.org/ns/sosa/",
            "http://qudt.org/schema/qudt/",
            "http://www.w3.org/ns/prov#",
            "proposed_semantic_graph",
        ]:
            self.assertIn(fragment, turtle)
        # same_as is deliberately not owl:sameAs, so no reasoner can merge on it.
        self.assertNotIn("owl:sameAs", turtle)

    def test_malformed_or_ill_typed_rdf_is_refused_on_read(self):
        with self.assertRaises(Invalid):
            from_rdf("this is not turtle {{{")
        with self.assertRaises(Invalid):
            from_rdf("")
        # An ill-typed triple inserted by hand is refused on read, not silently stored.
        rdf = Graph()
        rdf.parse(data=to_rdf(self.rich_graph())["turtle"], format="turtle")
        symplex = Namespace("urn:symplex:")
        rdf.add(
            (
                symplex["fixture:bench"],
                symplex["measured_by"],
                symplex["fixture:temperature_q"],
            )
        )
        with self.assertRaisesRegex(Invalid, "does not accept"):
            from_rdf(rdf.serialize(format="turtle"))


class ComplexSystemProjectionTests(unittest.TestCase):
    def test_projection_types_the_host_spec_and_parses_its_units(self):
        report = project_from_complex_system(system_spec())
        graph = report["graph"]
        self.assertEqual(report["unit_check"]["status"], "checked")
        self.assertEqual(report["unit_check"]["distinct_dimensions"], ["temperature"])
        self.assertEqual(report["rejected_relations"], [])
        self.assertEqual(
            report["expression_environment"],
            {"temperature": "temperature", "response": "temperature"},
        )
        self.assertFalse(report["empirically_validated"])
        self.assertIn("scope", report)
        kinds = graph_summary(graph)["nodes_by_kind"]
        for kind in ("entity", "quantity", "mechanism", "claim", "assumption"):
            self.assertIn(kind, kinds)
        self.assertEqual(
            sorted(n.id for n in graph.by_kind("quantity")),
            ["quantity_response", "quantity_temperature"],
        )
        self.assertTrue(any(r.predicate == "contradicts" for r in graph.relations))
        self.assertIn("consistency", report)
        self.assertIn("scope", report["consistency"])

    def test_projected_graph_survives_an_rdf_round_trip(self):
        graph = project_from_complex_system(system_spec())["graph"]
        restored = from_rdf(to_rdf(graph)["turtle"])["graph"]
        self.assertEqual(restored.snapshot(), graph.snapshot())

    def test_projection_refuses_a_spec_that_is_not_a_valid_contract(self):
        broken = system_spec()
        broken["units"][0]["dimension"] = "temperature per fortnight"
        with self.assertRaises(Invalid):
            project_from_complex_system(broken)
        with self.assertRaises(Invalid):
            project_from_complex_system({"title": "incomplete"})

    def test_declared_expressions_can_be_checked_against_the_projected_environment(
        self,
    ):
        environment = project_from_complex_system(system_spec())[
            "expression_environment"
        ]
        self.assertEqual(
            check_expression_dimensions("response - temperature", environment)[
                "dimension"
            ],
            "temperature",
        )
        with self.assertRaises(Invalid):
            check_expression_dimensions("exp(temperature)", environment)


class EntityAlignmentTests(unittest.TestCase):
    def test_exact_and_punctuation_only_differences_match(self):
        report = align_entities(
            ["Riverside Weather Station", "Acme Corp."],
            [
                {"id": "b1", "name": "Riverside Weather Station"},
                {"id": "b2", "name": "Acme Corp"},
            ],
        )
        self.assertEqual(
            sorted((m["a"], m["b"]) for m in report["matches"]),
            [("Acme Corp.", "b2"), ("Riverside Weather Station", "b1")],
        )
        self.assertFalse(report["auto_merged"])
        self.assertIn("scope", report)

    def test_ambiguous_matches_are_not_auto_merged(self):
        report = align_entities(
            ["Riverside Weather Station"],
            ["Riverside Weather Station North", "Riverside Weather Station South"],
        )
        self.assertEqual(report["matches"], [])
        self.assertEqual(report["ambiguous_count"], 1)
        self.assertEqual(report["ambiguous"][0]["reason"], "tied_candidates")
        self.assertEqual(report["ambiguous"][0]["resolution"], "human_review_required")
        self.assertEqual(
            report["requires_human_resolution"], ["Riverside Weather Station"]
        )
        self.assertFalse(report["auto_merged"])

    def test_two_left_records_cannot_silently_claim_one_right_record(self):
        report = align_entities(
            [
                {"id": "a1", "name": "Station 12"},
                {"id": "a2", "name": "Station 12"},
            ],
            [{"id": "b1", "name": "Station 12"}],
        )
        self.assertEqual(report["matches"], [])
        self.assertEqual(report["ambiguous_count"], 2)
        for entry in report["ambiguous"]:
            self.assertEqual(
                entry["reason"], "many_left_records_claim_one_right_record"
            )

    def test_unrelated_names_are_left_unmatched(self):
        report = align_entities(["Copper price index"], ["Rainfall at Kew"])
        self.assertEqual(report["matches"], [])
        self.assertEqual(report["ambiguous"], [])
        self.assertEqual(report["unmatched_a"], ["Copper price index"])
        self.assertEqual(report["unmatched_b"], ["Rainfall at Kew"])

    def test_input_contracts_are_bounded(self):
        with self.assertRaises(Invalid):
            align_entities("not a list", ["a"])
        with self.assertRaises(Invalid):
            align_entities([{"name": ""}], ["a"])
        with self.assertRaises(Invalid):
            align_entities([{"id": "x", "name": "a"}, {"id": "x", "name": "b"}], ["a"])


class TimeAlignmentTests(unittest.TestCase):
    def hourly(self, start_hour=0, offset="+00:00", count=24, day="2026-01-01"):
        return [
            f"{day}T{hour:02d}:00:00{offset}"
            for hour in range(start_hour, start_hour + count)
        ]

    def test_a_timezone_shifted_series_is_flagged(self):
        report = align_time(
            {"id": "utc_feed", "timezone": "UTC", "timestamps": self.hourly()},
            {
                "id": "local_feed",
                "timezone": "Asia/Kolkata",
                "timestamps": self.hourly(offset="+05:30"),
            },
        )
        codes = [f["code"] for f in report["findings"]]
        self.assertIn("suspected_clock_shift", codes)
        self.assertIn("declared_timezone_mismatch", codes)
        self.assertEqual(report["constant_offset_seconds"], -19800.0)
        self.assertNotEqual(report["verdict"], "aligned_as_declared")
        self.assertFalse(report["resampled"])
        self.assertFalse(report["validated"])
        self.assertIn("scope", report)

    def test_naive_timestamps_are_an_error_not_an_assumption(self):
        report = align_time(
            {"id": "a", "timestamps": ["2026-01-01T00:00:00", "2026-01-01T01:00:00"]},
            {
                "id": "b",
                "timestamps": [
                    "2026-01-01T00:00:00+00:00",
                    "2026-01-01T01:00:00+00:00",
                ],
            },
        )
        self.assertIn("naive_timestamps", [f["code"] for f in report["findings"]])
        self.assertEqual(report["verdict"], "blocked")
        self.assertFalse(report["resampling_contract"]["applicable"])

    def test_sampling_rate_mismatch_proposes_an_aggregation_contract(self):
        report = align_time(
            {"id": "hourly", "timezone": "UTC", "timestamps": self.hourly()},
            {
                "id": "quarter_hourly",
                "timezone": "UTC",
                "timestamps": [
                    f"2026-01-01T{hour:02d}:{minute:02d}:00+00:00"
                    for hour in range(24)
                    for minute in (0, 15, 30, 45)
                ],
            },
        )
        self.assertIn("sampling_rate_mismatch", [f["code"] for f in report["findings"]])
        contract = report["resampling_contract"]
        self.assertEqual(contract["target_step_seconds"], 3600.0)
        self.assertEqual(contract["target_clock"], "UTC")
        self.assertEqual(contract["method_b"], "aggregate")
        self.assertEqual(contract["time_alignment"], "aggregate")
        self.assertEqual(report["verdict"], "resample_required")

    def test_non_overlapping_support_blocks_the_join(self):
        report = align_time(
            {"id": "a", "timezone": "UTC", "timestamps": self.hourly()},
            {"id": "b", "timezone": "UTC", "timestamps": self.hourly(day="2027-06-01")},
        )
        self.assertIn("no_temporal_overlap", [f["code"] for f in report["findings"]])
        self.assertEqual(report["verdict"], "blocked")
        self.assertIsNone(report["overlap_window"])

    def test_identical_declared_supports_need_no_resampling(self):
        stamps = self.hourly()
        report = align_time(
            {"id": "a", "timezone": "UTC", "clock": "UTC", "timestamps": stamps},
            {"id": "b", "timezone": "UTC", "clock": "UTC", "timestamps": list(stamps)},
        )
        self.assertEqual(report["findings"], [])
        self.assertEqual(report["verdict"], "aligned_as_declared")
        self.assertEqual(report["resampling_contract"]["time_alignment"], "synchronous")

    def test_malformed_series_are_refused(self):
        for series in [
            {"id": "a"},
            {"id": "a", "timestamps": []},
            {"id": "a", "timestamps": ["not a timestamp"]},
            {"id": "a", "timestamps": [float("nan")]},
        ]:
            with self.subTest(series=series):
                with self.assertRaises(Invalid):
                    align_time(series, {"id": "b", "timestamps": [0.0, 3600.0]})


class JoinRiskTests(unittest.TestCase):
    def test_a_many_to_many_join_reports_its_blowup_factor(self):
        left = [{"k": "a", "v": i} for i in range(3)] + [{"k": "b", "v": 9}]
        right = [{"k": "a", "w": i} for i in range(4)]
        report = join_risk_report(left, right, "k", "k")
        self.assertEqual(report["cardinality"], "many_to_many")
        self.assertEqual(report["projected_output_rows"], 12)
        self.assertEqual(report["duplicate_blowup_factor"], 3.0)
        self.assertEqual(report["max_fanout_product"], 12)
        self.assertEqual(report["risk"], "high")
        self.assertIn("many_to_many_cardinality", report["risk_reasons"])
        self.assertFalse(report["validated_join"])
        self.assertIn("scope", report)

    def test_unmatched_fractions_are_reported_on_both_sides(self):
        left = [{"k": "a"}, {"k": "b"}, {"k": "c"}, {"k": "d"}]
        right = [{"k": "a"}, {"k": "z"}]
        report = join_risk_report(left, right, "k", "k")
        self.assertEqual(report["left"]["unmatched_fraction"], 0.75)
        self.assertEqual(report["right"]["unmatched_fraction"], 0.5)
        self.assertEqual(report["matched_keys"], 1)
        self.assertIn("majority_unmatched", report["risk_reasons"])

    def test_a_clean_one_to_one_join_is_low_risk_but_never_validated(self):
        rows = [{"k": "a"}, {"k": "b"}]
        report = join_risk_report(rows, [{"k": "a"}, {"k": "b"}], "k", "k")
        self.assertEqual(report["risk"], "low")
        self.assertEqual(report["cardinality"], "one_to_one")
        self.assertEqual(report["risk_reasons"], [])
        self.assertFalse(report["validated_join"])
        self.assertEqual(report["status"], "risk_assessment")

    def test_case_and_whitespace_near_misses_and_null_keys_raise_risk(self):
        report = join_risk_report(
            [{"k": "Station12"}, {"k": None}],
            [{"k": "station12 "}],
            "k",
            "k",
        )
        self.assertEqual(report["matched_keys"], 0)
        self.assertEqual(report["case_or_whitespace_near_misses"], ["station12 "])
        self.assertEqual(report["risk"], "blocked")
        self.assertEqual(report["left"]["null_or_missing_keys"], 1)

    def test_key_type_mismatch_blocks_the_join(self):
        report = join_risk_report([{"k": 12}], [{"k": "12"}], "k", "k")
        self.assertIn("key_type_mismatch", report["risk_reasons"])
        self.assertEqual(report["risk"], "blocked")

    def test_inputs_are_bounded_and_typed(self):
        with self.assertRaises(Invalid):
            join_risk_report("rows", [{"k": 1}], "k", "k")
        with self.assertRaises(Invalid):
            join_risk_report([{"k": 1}], [1], "k", "k")
        with self.assertRaises(Invalid):
            join_risk_report([{"k": 1}], [{"k": 1}], "", "k")


class ScopeDisciplineTests(unittest.TestCase):
    """Every public report must say what it has not established."""

    def test_every_public_report_carries_a_scope_string(self):
        graph = CausalGraph.from_pairs([("z", "x"), ("z", "y"), ("x", "y")])
        ontology = OntologyGraph("fixture")
        ontology.add_nodes([entity(), quantity(), measurement(), source()])
        ontology.add_relation("bench", "has_quantity", "temperature_q")
        reports = [
            parse_dimension("mass"),
            registry(),
            convert(1.0, "km", "m"),
            check_unit_definition(
                {
                    "id": "m",
                    "symbol": "m",
                    "dimension": "length",
                    "scale_to_canonical": 1.0,
                    "offset_to_canonical": 0.0,
                }
            ),
            check_unit_definitions([]),
            check_expression_dimensions("a * b", {"a": "mass", "b": "length"}),
            acyclic_check(graph),
            ancestors(graph, "y"),
            descendants(graph, "z"),
            d_separation(graph, "x", "y", ["z"]),
            backdoor_sets(graph, "x", "y"),
            frontdoor_check(
                CausalGraph.from_pairs([("x", "m"), ("m", "y")]), "x", "y", ["m"]
            ),
            identifiable(graph, "x", "y"),
            instrument_check(
                CausalGraph.from_pairs(
                    [("z", "x"), ("u", "x"), ("u", "y"), ("x", "y")], latent=["u"]
                ),
                "z",
                "x",
                "y",
            ),
            testable_implications(graph),
            relation_vocabulary(),
            graph_summary(ontology),
            consistency_report(ontology),
            to_rdf(ontology),
            from_rdf(to_rdf(ontology)["turtle"]),
            project_from_complex_system(system_spec()),
            align_entities(["a"], ["a"]),
            align_time(
                {"id": "a", "timestamps": [0.0, 3600.0]},
                {"id": "b", "timestamps": [0.0, 3600.0]},
            ),
            join_risk_report([{"k": 1}], [{"k": 1}], "k", "k"),
        ]
        for report in reports:
            self.assertIsInstance(report, dict)
            self.assertIsInstance(report.get("scope"), str)
            self.assertTrue(len(report["scope"]) > 40, report["scope"])

    def test_causal_reports_name_the_graph_as_the_assumption(self):
        graph = CausalGraph.from_pairs([("z", "x"), ("z", "y"), ("x", "y")])
        self.assertIn("ASSUMED graph", identifiable(graph, "x", "y")["scope"])
        self.assertIn("assumption", acyclic_check(graph)["scope"])
        self.assertIn(
            "not the complete ID algorithm", backdoor_sets(graph, "x", "y")["scope"]
        )


if __name__ == "__main__":
    unittest.main()
