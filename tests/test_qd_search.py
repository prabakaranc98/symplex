"""Search machinery is host-owned: descriptors, replacement rules and refusals are tested.

These tests exercise the archive on the standard Rastrigin quality-diversity
benchmark, the mutation grammar on its own preconditions, and the lineage graph
on a known chain. They check that the search behaves as declared. They establish
nothing about scientific validity, and no assertion here should be read as one.
"""

import math
import random
import unittest

from symplex.agents.lineage import LineageGraph
from symplex.agents.operators import (
    OPERATORS,
    PROTECTED_FIELDS,
    adaptive_operator_selection,
    add_bias_or_missingness_model,
    add_coupling,
    add_lag,
    adjust_resolution,
    apply_operator,
    change_inference_strategy,
    change_observation_model,
    component_compatibility,
    introduce_shared_driver,
    operator_card,
    recombine,
    remove_coupling,
    repair_alignment_error,
    replace_component,
    seed_genome,
    validate_patch,
)
from symplex.agents.qd_archive import (
    SCALARIZATION,
    Archive,
    crowding_distances,
    linear_bins,
    pareto_compare,
    trivial_descriptor,
)
from symplex.core.contracts import Invalid, canonical

RASTRIGIN_LOW, RASTRIGIN_HIGH, RASTRIGIN_DIMENSION = -5.12, 5.12, 6
RASTRIGIN_OBJECTIVES = [
    {
        "name": "rastrigin_loss",
        "direction": "minimize",
        "lower": 0.0,
        "upper": 90.0,
        "weight": 1.0,
        "units": "benchmark loss units",
        "meaning": "Standard Rastrigin objective over the full parameter vector.",
    },
    {
        "name": "parameter_cost",
        "direction": "minimize",
        "lower": 0.0,
        "upper": RASTRIGIN_DIMENSION * 5.12,
        "weight": 1.0,
        "units": "absolute parameter units",
        "meaning": "Stand-in for compute cost: the L1 size of the parameter vector.",
    },
]


def rastrigin(vector):
    return 10 * len(vector) + sum(
        value * value - 10 * math.cos(2 * math.pi * value) for value in vector
    )


def rastrigin_objectives(candidate):
    vector = candidate["genome"]["x"]
    return {
        "rastrigin_loss": rastrigin(vector),
        "parameter_cost": sum(abs(value) for value in vector),
    }


def rastrigin_descriptor(candidate):
    """The standard MAP-Elites Rastrigin behaviour descriptor: the first two genes."""
    vector = candidate["genome"]["x"]
    return (vector[0], vector[1])


def rastrigin_archive(*, strategy="curiosity", front_capacity=4, **kwargs):
    return Archive(
        ["first_gene", "second_gene"],
        [
            linear_bins(RASTRIGIN_LOW, RASTRIGIN_HIGH, 8, name="first_gene"),
            linear_bins(RASTRIGIN_LOW, RASTRIGIN_HIGH, 8, name="second_gene"),
        ],
        RASTRIGIN_OBJECTIVES,
        descriptor_fn=rastrigin_descriptor,
        objective_fn=rastrigin_objectives,
        selection_strategy=strategy,
        front_capacity=front_capacity,
        **kwargs,
    )


def run_rastrigin(seed, *, strategy="curiosity", generations=25, batch=8, sigma=0.6):
    """One fully seeded quality-diversity run; nothing here draws from global state."""
    archive = rastrigin_archive(strategy=strategy)
    rng = random.Random(seed)
    counter = 0
    for _ in range(batch):
        vector = [
            rng.uniform(RASTRIGIN_LOW, RASTRIGIN_HIGH)
            for _ in range(RASTRIGIN_DIMENSION)
        ]
        archive.add({"id": f"c{counter}", "genome": {"x": vector}})
        counter += 1
    series = [archive.advance_generation()]
    for generation in range(generations):
        for offset in range(batch):
            selection = archive.select_parent(
                seed * 10_007 + generation * 101 + offset, strategy=strategy
            )
            step = random.Random(seed * 7919 + generation * 97 + offset)
            vector = [
                max(RASTRIGIN_LOW, min(RASTRIGIN_HIGH, value + step.gauss(0, sigma)))
                for value in selection["selected_genome"]["x"]
            ]
            archive.add(
                {"id": f"c{counter}", "genome": {"x": vector}},
                parent_ids=[selection["selected_candidate_id"]],
                operator="gaussian_step",
            )
            counter += 1
        series.append(archive.advance_generation())
    return archive, series


def scored(identifier, descriptor, loss, cost):
    return {
        "id": identifier,
        "descriptor": descriptor,
        "objectives": {"rastrigin_loss": loss, "parameter_cost": cost},
    }


def declared_archive(**kwargs):
    """A small archive over the trivial declared descriptor, for unit-level checks."""
    kwargs.setdefault("front_capacity", 3)
    return Archive(
        ["first_gene", "second_gene"],
        [linear_bins(0.0, 4.0, 4), linear_bins(0.0, 4.0, 4)],
        RASTRIGIN_OBJECTIVES,
        descriptor_fn=trivial_descriptor,
        **kwargs,
    )


class RastriginBenchmarkTests(unittest.TestCase):
    def test_coverage_and_qd_score_increase_over_generations(self):
        archive, series = run_rastrigin(11)
        self.assertGreater(series[-1]["coverage"], series[0]["coverage"])
        self.assertGreater(series[-1]["qd_score"], series[0]["qd_score"])
        coverages = [record["coverage"] for record in series]
        self.assertEqual(coverages, sorted(coverages))
        self.assertEqual(archive.qd_score()["scalarization"], SCALARIZATION)
        self.assertEqual(
            archive.coverage()["occupied_cells"], len(archive.qd_score()["per_cell"])
        )
        self.assertLessEqual(archive.qd_score()["qd_score"], len(archive.cells))
        improvement = archive.improvement_over_generations()
        self.assertEqual(improvement["generations_recorded"], len(series))
        self.assertIsNotNone(improvement["last_improved_generation"])

    def test_repeated_seed_is_bit_identical(self):
        first, _ = run_rastrigin(11)
        second, _ = run_rastrigin(11)
        self.assertEqual(canonical(first.to_dict()), canonical(second.to_dict()))
        other, _ = run_rastrigin(12)
        self.assertNotEqual(canonical(first.to_dict()), canonical(other.to_dict()))

    def test_state_round_trips_through_plain_json(self):
        archive, _ = run_rastrigin(11, generations=4)
        state = archive.to_dict()
        restored = Archive.from_dict(
            state,
            descriptor_fn=rastrigin_descriptor,
            objective_fn=rastrigin_objectives,
        )
        self.assertEqual(canonical(restored.to_dict()), canonical(state))
        self.assertEqual(restored.coverage()["coverage"], archive.coverage()["coverage"])
        self.assertEqual(restored.qd_score()["qd_score"], archive.qd_score()["qd_score"])

    def test_every_metric_carries_its_scope(self):
        archive, _ = run_rastrigin(11, generations=3)
        metrics = archive.metrics()
        self.assertFalse(metrics["promotion_allowed"])
        for section in ("coverage", "qd_score", "archive_entropy", "novelty"):
            self.assertIn("scope", metrics[section])
            disclaimer = metrics[section]["scope"].lower()
            self.assertTrue("not" in disclaimer or "never" in disclaimer, section)
        self.assertIn("search", metrics["scope"].lower())
        self.assertGreater(archive.archive_entropy()["entropy_bits"], 0.0)
        self.assertGreaterEqual(archive.novelty()["minimum"], 0.0)

    def test_stagnation_is_visible_in_the_generation_series(self):
        archive = rastrigin_archive()
        archive.add({"id": "a", "genome": {"x": [0.0] * RASTRIGIN_DIMENSION}})
        archive.advance_generation()
        for _ in range(3):
            archive.advance_generation()
        improvement = archive.improvement_over_generations()
        self.assertTrue(improvement["stagnant"])
        self.assertEqual(improvement["generations_since_improvement"], 3)


class ParetoArchiveTests(unittest.TestCase):
    def test_new_cell_then_nondominated_then_dominated_rejection(self):
        archive = declared_archive()
        first = archive.add(scored("a", [0.5, 0.5], 10.0, 5.0))
        self.assertEqual(first["outcome"], "new_cell")
        second = archive.add(scored("b", [0.6, 0.6], 20.0, 1.0))
        self.assertEqual(second["outcome"], "nondominated_added")
        self.assertEqual(second["cell_id"], first["cell_id"])
        third = archive.add(scored("c", [0.7, 0.7], 30.0, 9.0))
        self.assertEqual(third["outcome"], "dominated_rejected")
        self.assertFalse(third["accepted"])
        self.assertTrue(third["retained"])
        self.assertIn("dominated", third["reason"])
        retained = archive.retained_rejections(outcome="dominated_rejected")
        self.assertEqual(retained["count"], 1)
        record = retained["records"][0]
        self.assertEqual(record["candidate_id"], "c")
        self.assertEqual(sorted(record["dominated_by"]), ["a", "b"])
        self.assertEqual(record["objectives"]["rastrigin_loss"], 30.0)

    def test_dominating_candidate_displaces_and_the_loser_is_retained(self):
        archive = declared_archive()
        archive.add(scored("a", [0.5, 0.5], 30.0, 9.0))
        outcome = archive.add(scored("b", [0.5, 0.5], 10.0, 5.0))
        self.assertEqual(outcome["outcome"], "dominates_existing")
        self.assertEqual(outcome["displaced_candidate_ids"], ["a"])
        displaced = archive.retained_rejections(outcome="displaced_by_domination")
        self.assertEqual(displaced["count"], 1)
        self.assertEqual(displaced["records"][0]["candidate_id"], "a")
        self.assertIn("dominated by later candidate b", displaced["records"][0]["reason"])

    def test_infeasible_candidate_is_rejected_and_retained_with_its_reason(self):
        archive = declared_archive(
            feasibility_fn=lambda c: ["resource_limit_exceeded"]
            if c.get("genome", {}).get("cost", 0) > 1
            else []
        )
        outcome = archive.add(
            dict(scored("a", [0.5, 0.5], 10.0, 5.0), genome={"cost": 9})
        )
        self.assertEqual(outcome["outcome"], "infeasible")
        self.assertTrue(outcome["retained"])
        self.assertIsNone(outcome["cell_id"])
        record = archive.retained_rejections(outcome="infeasible")["records"][0]
        self.assertEqual(record["violations"], ["resource_limit_exceeded"])
        self.assertEqual(archive.coverage()["occupied_cells"], 0)

    def test_crowding_distance_pruning_keeps_the_extremes_of_the_front(self):
        archive = declared_archive(front_capacity=3)
        archive.add(scored("extreme_low_loss", [0.5, 0.5], 0.0, 10.0))
        archive.add(scored("interior_near", [0.5, 0.5], 1.0, 9.9))
        archive.add(scored("interior_far", [0.5, 0.5], 2.0, 9.8))
        outcome = archive.add(scored("extreme_low_cost", [0.5, 0.5], 10.0, 0.0))
        self.assertEqual(outcome["outcome"], "nondominated_added")
        self.assertEqual(outcome["crowding_pruned_candidate_ids"], ["interior_near"])
        front = {e["candidate_id"] for e in archive.cells[outcome["cell_id"]]["front"]}
        self.assertEqual(len(front), 3)
        self.assertIn("extreme_low_loss", front)
        self.assertIn("extreme_low_cost", front)
        self.assertNotIn("interior_near", front)
        pruned = archive.retained_rejections(outcome="crowding_pruned")["records"]
        self.assertEqual([r["candidate_id"] for r in pruned], ["interior_near"])
        self.assertIn("crowding distance", pruned[0]["reason"])

    def test_crowding_distance_arithmetic_marks_boundaries_as_unbounded(self):
        front = [
            {"candidate_id": "low", "objectives": {"rastrigin_loss": 0.0, "parameter_cost": 10.0}},
            {"candidate_id": "mid", "objectives": {"rastrigin_loss": 5.0, "parameter_cost": 5.0}},
            {"candidate_id": "high", "objectives": {"rastrigin_loss": 10.0, "parameter_cost": 0.0}},
        ]
        report = crowding_distances(front, RASTRIGIN_OBJECTIVES)
        self.assertTrue(math.isinf(report["distances"]["low"]))
        self.assertTrue(math.isinf(report["distances"]["high"]))
        self.assertFalse(math.isinf(report["distances"]["mid"]))
        self.assertIsNone(report["finite_distances"]["low"])

    def test_pareto_compare_reports_the_relation_and_its_rule(self):
        loss_only = {"rastrigin_loss": 1.0, "parameter_cost": 1.0}
        worse = {"rastrigin_loss": 2.0, "parameter_cost": 2.0}
        trade = {"rastrigin_loss": 0.5, "parameter_cost": 5.0}
        self.assertEqual(
            pareto_compare(loss_only, worse, RASTRIGIN_OBJECTIVES)["relation"],
            "a_dominates_b",
        )
        self.assertEqual(
            pareto_compare(worse, loss_only, RASTRIGIN_OBJECTIVES)["relation"],
            "b_dominates_a",
        )
        self.assertEqual(
            pareto_compare(loss_only, trade, RASTRIGIN_OBJECTIVES)["relation"],
            "mutually_nondominated",
        )
        self.assertEqual(
            pareto_compare(loss_only, loss_only, RASTRIGIN_OBJECTIVES)["relation"],
            "identical_objective_vector",
        )


class DescriptorRangeTests(unittest.TestCase):
    def test_out_of_range_descriptors_are_clamped_and_counted(self):
        archive = declared_archive()
        below = archive.add(scored("below", [-99.0, 2.5], 10.0, 5.0))
        above = archive.add(scored("above", [99.0, 2.5], 11.0, 6.0))
        self.assertEqual(below["cell_index"][0], 0)
        self.assertEqual(above["cell_index"][0], 3)
        self.assertEqual(below["clamped"][0]["direction"], "below_low_edge")
        self.assertEqual(above["clamped"][0]["direction"], "above_high_edge")
        report = archive.clamp_report()
        self.assertEqual(report["total_clamps"], 2)
        self.assertEqual(report["per_dimension"]["first_gene"]["below_low_edge"], 1)
        self.assertEqual(report["per_dimension"]["first_gene"]["above_high_edge"], 1)
        self.assertEqual(report["per_dimension"]["second_gene"]["below_low_edge"], 0)
        self.assertEqual(sorted(report["clamped_candidate_ids"]), ["above", "below"])
        self.assertIn("does not cover", report["scope"])

    def test_in_range_descriptors_are_never_counted_as_clamps(self):
        archive = declared_archive()
        archive.add(scored("low_edge", [0.0, 0.0], 10.0, 5.0))
        archive.add(scored("high_edge", [4.0, 4.0], 11.0, 6.0))
        self.assertEqual(archive.clamp_report()["total_clamps"], 0)

    def test_objective_values_outside_declared_bounds_are_clipped_and_counted(self):
        archive = declared_archive()
        archive.add(scored("huge", [1.0, 1.0], 10_000.0, 5.0))
        self.assertEqual(
            archive.clamp_report()["objective_clip_events"]["rastrigin_loss"], 1
        )

    def test_trivial_descriptor_declares_that_it_measures_nothing(self):
        self.assertIn("Declared, not measured", trivial_descriptor.scope)
        with self.assertRaises(Invalid):
            trivial_descriptor({"id": "a"})


class SelectionStrategyTests(unittest.TestCase):
    """Curiosity bookkeeping, checked directly rather than through a noisy outcome.

    A coverage advantage for curiosity over uniform was NOT observed on the
    Rastrigin benchmark or on a constructed dead-end benchmark, so no such
    advantage is asserted here. What is asserted is the mechanism: the recorded
    curiosity accounting, and the resulting concentration of selection budget on
    the cell whose offspring survive.
    """

    def productive_and_dead_archive(self):
        archive = declared_archive()
        archive.add(scored("productive_parent", [0.5, 0.5], 20.0, 8.0))
        archive.add(scored("dead_parent", [3.5, 3.5], 20.0, 8.0))
        return archive

    def test_curiosity_rises_on_accepted_offspring_and_falls_on_rejected(self):
        archive = self.productive_and_dead_archive()
        productive = archive.cell_of_candidate["productive_parent"]
        dead = archive.cell_of_candidate["dead_parent"]
        self.assertEqual(archive.cells[productive]["curiosity"], 1.0)
        archive.add(
            scored("child_accepted", [1.5, 1.5], 5.0, 2.0),
            parent_ids=["productive_parent"],
        )
        self.assertEqual(archive.cells[productive]["curiosity"], 2.0)
        self.assertEqual(archive.cells[productive]["offspring_accepted"], 1)
        rejected = archive.add(
            scored("child_rejected", [3.5, 3.5], 30.0, 9.0), parent_ids=["dead_parent"]
        )
        self.assertEqual(rejected["outcome"], "dominated_rejected")
        self.assertEqual(archive.cells[dead]["curiosity"], 0.5)
        self.assertEqual(archive.cells[dead]["offspring_rejected"], 1)

    def test_curiosity_is_floored_and_never_goes_negative(self):
        archive = self.productive_and_dead_archive()
        dead = archive.cell_of_candidate["dead_parent"]
        for index in range(6):
            archive.add(
                scored(f"dud_{index}", [3.5, 3.5], 30.0 + index, 9.0 + index),
                parent_ids=["dead_parent"],
            )
        self.assertEqual(archive.cells[dead]["curiosity"], 0.0)
        self.assertEqual(archive.cells[dead]["offspring_rejected"], 6)

    def test_curiosity_concentrates_selection_on_the_productive_cell(self):
        archive = self.productive_and_dead_archive()
        for index in range(4):
            archive.add(
                scored(f"good_{index}", [0.5 + index, 0.5], 5.0 - index, 2.0),
                parent_ids=["productive_parent"],
            )
        for index in range(4):
            archive.add(
                scored(f"bad_{index}", [3.5, 3.5], 40.0 + index, 9.0 + index),
                parent_ids=["dead_parent"],
            )
        productive = archive.cell_of_candidate["productive_parent"]
        dead = archive.cell_of_candidate["dead_parent"]
        weights = {
            row["cell_id"]: row["probability"]
            for row in archive.select_parent(1, strategy="curiosity")["weights"]
        }
        self.assertGreater(weights[productive], weights[dead])
        self.assertEqual(archive.cells[dead]["curiosity"], 0.0)
        curious = sum(
            archive.select_parent(seed, strategy="curiosity")["selected_cell_id"]
            == productive
            for seed in range(200)
        )
        uniform = sum(
            archive.select_parent(seed, strategy="uniform")["selected_cell_id"]
            == productive
            for seed in range(200)
        )
        self.assertGreater(curious, uniform)

    def test_every_strategy_is_selectable_deterministic_and_records_its_arithmetic(self):
        for strategy in ("curiosity", "uniform", "least_visited", "novelty"):
            with self.subTest(strategy=strategy):
                archive, _ = run_rastrigin(3, strategy=strategy, generations=4)
                twin, _ = run_rastrigin(3, strategy=strategy, generations=4)
                first = archive.select_parent(99, strategy=strategy)
                repeat = twin.select_parent(99, strategy=strategy)
                self.assertEqual(strategy, first["strategy"])
                self.assertEqual(
                    first["selected_candidate_id"], repeat["selected_candidate_id"]
                )
                self.assertEqual(first["visit_count_after"], repeat["visit_count_after"])
                self.assertIn("score(cell)", first["arithmetic"])
                self.assertAlmostEqual(
                    sum(row["probability"] for row in first["weights"]), 1.0
                )

    def test_least_visited_reproduces_the_earlier_scheduling_rule(self):
        archive = self.productive_and_dead_archive()
        archive.cells[archive.cell_of_candidate["productive_parent"]]["visit_count"] = 5
        chosen = archive.select_parent(4, strategy="least_visited")
        self.assertEqual(chosen["selected_candidate_id"], "dead_parent")

    def test_selection_requires_an_explicit_seed_and_a_nonempty_archive(self):
        archive = self.productive_and_dead_archive()
        with self.assertRaises(Invalid):
            archive.select_parent("seven")
        with self.assertRaises(Invalid):
            archive.select_parent(1, strategy="hill_climb")
        with self.assertRaises(Invalid):
            declared_archive().select_parent(1)


class ArchiveContractTests(unittest.TestCase):
    def test_bad_configuration_is_refused(self):
        with self.assertRaises(Invalid):
            Archive(["a"] * 13, [linear_bins(0, 1, 2)] * 13, RASTRIGIN_OBJECTIVES)
        with self.assertRaises(Invalid):
            Archive(["a", "a"], [linear_bins(0, 1, 2)] * 2, RASTRIGIN_OBJECTIVES)
        with self.assertRaises(Invalid):
            declared_archive(front_capacity=1)
        with self.assertRaises(Invalid):
            declared_archive(selection_strategy="greedy")
        with self.assertRaises(Invalid):
            Archive(["a"], [[1.0, 0.0]], RASTRIGIN_OBJECTIVES)
        with self.assertRaises(Invalid):
            Archive(
                ["a"],
                [linear_bins(0, 1, 2)],
                [{"name": "x", "direction": "sideways", "lower": 0, "upper": 1}],
            )
        with self.assertRaises(Invalid):
            Archive(
                ["a"],
                [linear_bins(0, 1, 2)],
                [{"name": "x", "direction": "minimize", "lower": 1.0, "upper": 0.0}],
            )

    def test_reachable_cell_override_requires_a_stated_basis(self):
        with self.assertRaises(Invalid):
            declared_archive(reachable_cells=4)
        archive = declared_archive(
            reachable_cells=4,
            reachable_cells_basis="Only four cells are supported by the domain grammar.",
        )
        archive.add(scored("a", [0.5, 0.5], 1.0, 1.0))
        self.assertEqual(archive.coverage()["coverage"], 0.25)
        self.assertEqual(archive.coverage()["total_grid_cells"], 16)

    def test_duplicate_and_malformed_candidates_are_refused(self):
        archive = declared_archive()
        archive.add(scored("a", [0.5, 0.5], 1.0, 1.0))
        with self.assertRaises(Invalid):
            archive.add(scored("a", [0.5, 0.5], 2.0, 2.0))
        with self.assertRaises(Invalid):
            archive.add(scored("b", [0.5], 1.0, 1.0))
        with self.assertRaises(Invalid):
            archive.add({"id": "c", "descriptor": [0.5, 0.5], "objectives": {"x": 1.0}})
        with self.assertRaises(Invalid):
            archive.add(scored("d", [0.5, 0.5], float("nan"), 1.0))


class OperatorGrammarTests(unittest.TestCase):
    def event_indexed_genome(self):
        genome = seed_genome()
        genome["components"][1]["step_seconds"] = None
        genome["components"][1]["time_scale"] = "event_indexed"
        genome["couplings"][0] = {
            "source": "capacity",
            "target": "demand",
            "lag": 0,
            "mechanism": "declared congestion feedback",
        }
        return genome

    def compatible_component(self, ident):
        return {
            "id": ident,
            "kind": "state",
            "meaning": "available service capacity, reimplemented",
            "unit": "unit_per_hour",
            "dimension": "rate",
            "time_scale": "hourly",
            "step_seconds": 3600.0,
            "interface": {"inputs": [], "outputs": ["level"]},
        }

    def test_every_operator_declares_preconditions_and_a_disconfirmation(self):
        for name in OPERATORS:
            with self.subTest(operator=name):
                card = operator_card(name)
                self.assertTrue(card["preconditions"])
                self.assertTrue(card["expected_observable_effect"].strip())
                self.assertTrue(card["what_would_count_against_it"].strip())
                self.assertIn(card["arity"], (1, 2))
        with self.assertRaises(Invalid):
            operator_card("summon_a_better_model")

    def test_add_coupling_preconditions(self):
        genome = seed_genome()
        result = add_coupling(
            genome,
            source="demand",
            target="capacity",
            lag=1,
            mechanism="declared provisioning response",
            parent_id="p0",
        )
        self.assertEqual(len(result["genome"]["couplings"]), 2)
        self.assertEqual(result["parent_ids"], ["p0"])
        self.assertEqual(len(genome["couplings"]), 1)
        with self.assertRaises(Invalid):
            add_coupling(genome, source="ghost", target="capacity", lag=0, mechanism="m")
        with self.assertRaises(Invalid):
            add_coupling(genome, source="capacity", target="demand", lag=0, mechanism="m")
        with self.assertRaises(Invalid):
            add_coupling(genome, source="demand", target="demand", lag=0, mechanism="m")
        with self.assertRaises(Invalid):
            add_coupling(genome, source="demand", target="capacity", lag=99, mechanism="m")
        forbidden = seed_genome()
        forbidden["constraints"] = [
            {"kind": "forbidden_coupling", "source": "demand", "target": "capacity"}
        ]
        with self.assertRaises(Invalid):
            add_coupling(
                forbidden, source="demand", target="capacity", lag=0, mechanism="m"
            )

    def test_remove_coupling_preconditions(self):
        genome = add_coupling(
            seed_genome(),
            source="demand",
            target="capacity",
            lag=1,
            mechanism="declared provisioning response",
        )["genome"]
        result = remove_coupling(genome, source="demand", target="capacity")
        self.assertEqual(len(result["genome"]["couplings"]), 1)
        with self.assertRaises(Invalid):
            remove_coupling(genome, source="demand", target="ghost")
        with self.assertRaises(Invalid):
            remove_coupling(seed_genome(), source="capacity", target="demand")
        required = dict(genome, constraints=[
            {"kind": "required_coupling", "source": "demand", "target": "capacity"}
        ])
        with self.assertRaises(Invalid):
            remove_coupling(required, source="demand", target="capacity")

    def test_add_lag_preconditions_and_undefined_timing(self):
        result = add_lag(seed_genome(), source="capacity", target="demand", lag=2)
        self.assertEqual(result["genome"]["couplings"][0]["lag"], 2)
        self.assertIn("7200.0 seconds", result["detail"])
        with self.assertRaises(Invalid):
            add_lag(seed_genome(), source="capacity", target="demand", lag=0)
        with self.assertRaises(Invalid):
            add_lag(seed_genome(), source="capacity", target="demand", lag=99)
        with self.assertRaises(Invalid):
            add_lag(seed_genome(), source="demand", target="capacity", lag=1)
        with self.assertRaises(Invalid) as caught:
            add_lag(self.event_indexed_genome(), source="capacity", target="demand", lag=1)
        self.assertIn("lag_undefined_without_sampling_interval", str(caught.exception))

    def test_introduce_shared_driver_preconditions(self):
        driver = self.compatible_component("weather")
        driver["step_seconds"] = 1800.0
        result = introduce_shared_driver(
            seed_genome(),
            driver=driver,
            target_ids=["demand", "capacity"],
            mechanism="declared common exogenous forcing",
        )
        self.assertEqual(len(result["genome"]["components"]), 3)
        self.assertEqual(len(result["genome"]["couplings"]), 3)
        with self.assertRaises(Invalid):
            introduce_shared_driver(
                seed_genome(), driver=driver, target_ids=["demand"], mechanism="m"
            )
        with self.assertRaises(Invalid):
            introduce_shared_driver(
                seed_genome(),
                driver=self.compatible_component("demand"),
                target_ids=["demand", "capacity"],
                mechanism="m",
            )
        slow = dict(driver, id="slow_driver", step_seconds=86400.0)
        with self.assertRaises(Invalid) as caught:
            introduce_shared_driver(
                seed_genome(),
                driver=slow,
                target_ids=["demand", "capacity"],
                mechanism="m",
            )
        self.assertIn("driver_sampled_slower_than_target", str(caught.exception))

    def test_bias_and_missingness_preconditions(self):
        model = {"mechanism": "declared reporting delay", "evidence_ids": ["evidence_1"]}
        result = add_bias_or_missingness_model(seed_genome(), kind="bias", model=model)
        self.assertEqual(
            result["genome"]["observation_model"]["bias_model"]["mechanism"],
            "declared reporting delay",
        )
        with self.assertRaises(Invalid):
            add_bias_or_missingness_model(result["genome"], kind="bias", model=model)
        with self.assertRaises(Invalid):
            add_bias_or_missingness_model(seed_genome(), kind="vibes", model=model)
        with self.assertRaises(Invalid):
            add_bias_or_missingness_model(
                seed_genome(),
                kind="missingness",
                model={"mechanism": "m", "evidence_ids": ["unknown_evidence"]},
            )
        with self.assertRaises(Invalid):
            add_bias_or_missingness_model(
                seed_genome(), kind="bias", model={"mechanism": "m", "evidence_ids": []}
            )

    def test_change_observation_model_refuses_a_change_of_support(self):
        result = change_observation_model(seed_genome(), kind="student_t_additive")
        self.assertEqual(result["genome"]["observation_model"]["kind"], "student_t_additive")
        self.assertEqual(result["genome"]["observation_model"]["support"], "real")
        with self.assertRaises(Invalid):
            change_observation_model(seed_genome(), kind="gaussian_additive")
        with self.assertRaises(Invalid):
            change_observation_model(seed_genome(), kind="telepathy")
        with self.assertRaises(Invalid) as caught:
            change_observation_model(seed_genome(), kind="poisson_count")
        self.assertIn("incompatible_observation_support", str(caught.exception))

    def test_replace_component_requires_matching_meaning_units_timing_interface(self):
        result = replace_component(
            seed_genome(),
            component_id="capacity",
            replacement=self.compatible_component("capacity_v2"),
        )
        self.assertIn(
            "capacity_v2", [c["id"] for c in result["genome"]["components"]]
        )
        self.assertEqual(result["genome"]["couplings"][0]["source"], "capacity_v2")
        for field, value in (
            ("unit", "unit_per_day"),
            ("dimension", "count"),
            ("time_scale", "daily"),
            ("step_seconds", 86400.0),
        ):
            with self.subTest(field=field):
                broken = dict(self.compatible_component("capacity_v2"), **{field: value})
                with self.assertRaises(Invalid) as caught:
                    replace_component(
                        seed_genome(), component_id="capacity", replacement=broken
                    )
                self.assertIn("incompatible_replacement", str(caught.exception))
        with self.assertRaises(Invalid):
            replace_component(
                seed_genome(),
                component_id="ghost",
                replacement=self.compatible_component("capacity_v2"),
            )

    def test_adjust_resolution_moves_one_rung_at_a_time(self):
        result = adjust_resolution(seed_genome(), resolution="fine")
        self.assertEqual(result["genome"]["inference"]["resolution"], "fine")
        with self.assertRaises(Invalid):
            adjust_resolution(seed_genome(), resolution="medium")
        with self.assertRaises(Invalid):
            adjust_resolution(seed_genome(), resolution="atomic")
        coarse = seed_genome()
        coarse["inference"]["resolution"] = "coarse"
        with self.assertRaises(Invalid) as caught:
            adjust_resolution(coarse, resolution="fine")
        self.assertIn("resolution_step_too_large", str(caught.exception))

    def test_repair_alignment_error_requires_a_demonstrated_error(self):
        demonstration = {
            "evidence_id": "evidence_1",
            "observed_discrepancy": "Two sources report the same hour under different clocks.",
            "affected_records": 412,
        }
        result = repair_alignment_error(
            seed_genome(),
            component_id="demand",
            demonstration=demonstration,
            correction="Re-key both sources onto UTC hour starts.",
        )
        self.assertEqual(result["genome"]["transforms"][0]["affected_records"], 412)
        with self.assertRaises(Invalid) as caught:
            repair_alignment_error(
                seed_genome(),
                component_id="demand",
                demonstration=dict(demonstration, affected_records=0),
                correction="Re-key both sources onto UTC hour starts.",
            )
        self.assertIn("alignment_error_not_demonstrated", str(caught.exception))
        with self.assertRaises(Invalid):
            repair_alignment_error(
                seed_genome(),
                component_id="demand",
                demonstration=dict(demonstration, evidence_id="unknown_evidence"),
                correction="c",
            )
        with self.assertRaises(Invalid):
            repair_alignment_error(
                seed_genome(),
                component_id="ghost",
                demonstration=demonstration,
                correction="c",
            )

    def test_change_inference_strategy_requires_backend_support(self):
        result = change_inference_strategy(seed_genome(), strategy="map_penalized")
        self.assertEqual(result["genome"]["inference"]["strategy"], "map_penalized")
        with self.assertRaises(Invalid):
            change_inference_strategy(seed_genome(), strategy="maximum_likelihood")
        with self.assertRaises(Invalid):
            change_inference_strategy(seed_genome(), strategy="wishful_thinking")
        with self.assertRaises(Invalid) as caught:
            change_inference_strategy(
                seed_genome(), strategy="markov_chain_monte_carlo"
            )
        self.assertIn("backend_does_not_support_strategy", str(caught.exception))

    def test_apply_operator_dispatches_and_checks_arity(self):
        result = apply_operator(
            "adjust_resolution", seed_genome(), resolution="fine"
        )
        self.assertEqual(result["operator"], "adjust_resolution")
        with self.assertRaises(Invalid):
            apply_operator("adjust_resolution", seed_genome(), seed_genome())
        with self.assertRaises(Invalid):
            apply_operator("teleport", seed_genome())


class RecombinationTests(unittest.TestCase):
    def donor(self, **component_overrides):
        genome = seed_genome()
        component = {
            "id": "weather",
            "kind": "driver",
            "meaning": "declared exogenous weather index",
            "unit": "index_unit",
            "dimension": "index",
            "time_scale": "hourly",
            "step_seconds": 3600.0,
            "interface": {"inputs": [], "outputs": ["level"]},
        }
        component.update(component_overrides)
        genome["components"].append(component)
        genome["couplings"].append(
            {
                "source": "weather",
                "target": "demand",
                "lag": 1,
                "mechanism": "declared weather sensitivity",
            }
        )
        genome["evidence_ids"] = ["evidence_1", "evidence_2"]
        return genome

    def test_recombine_merges_only_compatible_components(self):
        result = recombine(
            seed_genome(),
            self.donor(),
            component_ids=["weather"],
            parent_a_id="a",
            parent_b_id="b",
        )
        self.assertEqual(result["parent_ids"], ["a", "b"])
        self.assertIn("weather", [c["id"] for c in result["genome"]["components"]])
        self.assertEqual(len(result["genome"]["couplings"]), 2)
        self.assertEqual(
            result["genome"]["evidence_ids"], ["evidence_1", "evidence_2"]
        )

    def test_recombine_refuses_incompatible_units(self):
        donor = self.donor()
        donor["components"][1]["unit"] = "unit_per_day"
        with self.assertRaises(Invalid) as caught:
            recombine(seed_genome(), donor, component_ids=["capacity"])
        self.assertIn("incompatible_recombination", str(caught.exception))
        self.assertIn("unit", str(caught.exception))

    def test_recombine_refuses_incompatible_timing(self):
        donor = self.donor()
        donor["components"][1]["step_seconds"] = 60.0
        with self.assertRaises(Invalid) as caught:
            recombine(seed_genome(), donor, component_ids=["capacity"])
        self.assertIn("step_seconds", str(caught.exception))
        donor = self.donor()
        donor["components"][1]["time_scale"] = "per_minute"
        with self.assertRaises(Invalid) as caught:
            recombine(seed_genome(), donor, component_ids=["capacity"])
        self.assertIn("time_scale", str(caught.exception))

    def test_recombine_refuses_incompatible_interfaces_and_meaning(self):
        donor = self.donor()
        donor["components"][1]["interface"] = {"inputs": ["level"], "outputs": []}
        with self.assertRaises(Invalid) as caught:
            recombine(seed_genome(), donor, component_ids=["capacity"])
        self.assertIn("interface", str(caught.exception))
        donor = self.donor()
        donor["components"][1]["dimension"] = "count"
        with self.assertRaises(Invalid) as caught:
            recombine(seed_genome(), donor, component_ids=["capacity"])
        self.assertIn("dimension", str(caught.exception))

    def test_recombine_refuses_parents_that_observe_different_supports(self):
        donor = self.donor()
        donor["observation_model"] = {
            "kind": "poisson_count",
            "support": "count",
            "bias_model": None,
            "missingness_model": None,
        }
        with self.assertRaises(Invalid) as caught:
            recombine(seed_genome(), donor, component_ids=["weather"])
        self.assertIn("incompatible_observation_support", str(caught.exception))

    def test_recombine_requires_named_donor_components(self):
        with self.assertRaises(Invalid):
            recombine(seed_genome(), self.donor(), component_ids=[])
        with self.assertRaises(Invalid):
            recombine(seed_genome(), self.donor(), component_ids=["ghost"])
        with self.assertRaises(Invalid):
            recombine(
                seed_genome(), self.donor(), component_ids=["weather", "weather"]
            )

    def test_component_compatibility_reports_every_reason(self):
        left = seed_genome()["components"][0]
        right = dict(left, unit="unit_per_day", time_scale="daily")
        report = component_compatibility(left, right)
        self.assertFalse(report["compatible"])
        self.assertEqual(len(report["incompatibilities"]), 2)
        self.assertTrue(component_compatibility(left, dict(left))["compatible"])
        self.assertIn("no unit conversion is attempted", report["rule"])


class PatchDenylistTests(unittest.TestCase):
    def test_a_permitted_patch_is_accepted(self):
        parent = seed_genome()
        patch = {"inference": dict(parent["inference"], resolution="fine")}
        result = validate_patch(parent, patch)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["changed_fields"], ["inference"])
        self.assertEqual(result["child"]["inference"]["resolution"], "fine")
        self.assertIn("not that the edit is scientifically sensible", result["scope"])

    def test_the_evaluator_cannot_be_modified(self):
        with self.assertRaises(Invalid) as caught:
            validate_patch(seed_genome(), {"evaluator": "my_own_scorer"})
        self.assertIn("protected_field_not_editable: evaluator", str(caught.exception))

    def test_the_objective_cannot_be_modified(self):
        with self.assertRaises(Invalid) as caught:
            validate_patch(seed_genome(), {"objective": "maximise_my_score"})
        self.assertIn("protected_field_not_editable: objective", str(caught.exception))

    def test_execution_permissions_cannot_be_modified(self):
        with self.assertRaises(Invalid) as caught:
            validate_patch(seed_genome(), {"permissions": ["network", "subprocess"]})
        self.assertIn("protected_field_not_editable: permissions", str(caught.exception))

    def test_budget_enforcement_cannot_be_modified(self):
        with self.assertRaises(Invalid) as caught:
            validate_patch(seed_genome(), {"budget": {"model_calls": 10_000}})
        self.assertIn("protected_field_not_editable: budget", str(caught.exception))

    def test_every_declared_protected_field_is_rejected_by_name(self):
        for field in PROTECTED_FIELDS:
            with self.subTest(field=field), self.assertRaises(Invalid) as caught:
                validate_patch(seed_genome(), {field: "anything"})
            self.assertIn(field, str(caught.exception))

    def test_protected_fields_are_rejected_at_any_nesting_depth(self):
        parent = seed_genome()
        patch = {
            "inference": dict(parent["inference"], resolution="fine", evaluator="mine")
        }
        with self.assertRaises(Invalid) as caught:
            validate_patch(parent, patch)
        self.assertIn("protected_field_not_editable", str(caught.exception))
        nested = {"constraints": [{"kind": "note", "audit_records": ["deleted"]}]}
        with self.assertRaises(Invalid):
            validate_patch(parent, nested)

    def test_fields_outside_the_grammar_and_empty_patches_are_rejected(self):
        parent = seed_genome()
        with self.assertRaises(Invalid) as caught:
            validate_patch(parent, {"secret_backdoor": 1})
        self.assertIn("field_outside_mutation_grammar", str(caught.exception))
        with self.assertRaises(Invalid):
            validate_patch(parent, {"template": "some_other_template"})
        with self.assertRaises(Invalid):
            validate_patch(parent, {})
        with self.assertRaises(Invalid):
            validate_patch(parent, {"inference": parent["inference"]})
        with self.assertRaises(Invalid):
            validate_patch(parent, {"components": []})


class BanditTests(unittest.TestCase):
    def history(self, entries):
        return [{"operator": name, "outcome": outcome} for name, outcome in entries]

    def test_no_history_returns_the_documented_uniform_prior(self):
        result = adaptive_operator_selection([], 7)
        self.assertEqual(result["basis"], "uniform_prior_no_history")
        self.assertIn("uniform prior", result["note"])
        self.assertEqual(len(result["prior"]), len(OPERATORS))
        self.assertAlmostEqual(sum(result["prior"].values()), 1.0)
        for share in result["prior"].values():
            self.assertAlmostEqual(share, 1.0 / len(OPERATORS))
        self.assertIn(result["selected"], OPERATORS)
        self.assertEqual(result["history_records"], 0)
        self.assertTrue(all(row["status"] == "untried" for row in result["arms"]))
        self.assertEqual(
            adaptive_operator_selection([], 7)["selected"],
            adaptive_operator_selection([], 7)["selected"],
        )

    def test_ucb1_prefers_the_historically_successful_operator(self):
        history = self.history(
            [("add_coupling", "new_cell")] * 5
            + [("add_lag", "dominated_rejected")] * 5
        )
        result = adaptive_operator_selection(
            history, 3, operators=("add_coupling", "add_lag")
        )
        self.assertEqual(result["selected"], "add_coupling")
        arms = {row["operator"]: row for row in result["arms"]}
        self.assertEqual(arms["add_coupling"]["plays"], 5.0)
        self.assertEqual(arms["add_coupling"]["mean_reward"], 1.0)
        self.assertEqual(arms["add_lag"]["mean_reward"], 0.0)
        self.assertAlmostEqual(
            arms["add_coupling"]["exploration_bonus"],
            math.sqrt(2.0) * math.sqrt(math.log(10.0) / 5.0),
        )
        self.assertAlmostEqual(
            arms["add_coupling"]["score"] - arms["add_lag"]["score"], 1.0
        )
        self.assertIn("sqrt(ln(N)/plays)", result["arithmetic"])

    def test_untried_operators_are_played_before_exploiting(self):
        history = self.history([("add_coupling", "new_cell")] * 5)
        result = adaptive_operator_selection(
            history, 3, operators=("add_coupling", "add_lag")
        )
        self.assertEqual(result["basis"], "untried_operators_first")
        self.assertEqual(result["selected"], "add_lag")
        self.assertEqual(result["untried"], ["add_lag"])

    def test_credit_assignment_discounts_operators_that_stopped_working(self):
        history = self.history(
            [("add_lag", "new_cell")] * 8
            + [("add_coupling", "dominated_rejected")] * 8
            + [("add_lag", "dominated_rejected")] * 8
            + [("add_coupling", "new_cell")] * 8
        )
        arms = ("add_coupling", "add_lag")
        under_ucb1 = adaptive_operator_selection(history, 5, operators=arms)
        rows = {row["operator"]: row for row in under_ucb1["arms"]}
        self.assertAlmostEqual(rows["add_coupling"]["mean_reward"], 0.5)
        self.assertAlmostEqual(rows["add_lag"]["mean_reward"], 0.5)
        under_credit = adaptive_operator_selection(
            history, 5, strategy="credit", operators=arms
        )
        credited = {row["operator"]: row for row in under_credit["arms"]}
        self.assertEqual(under_credit["selected"], "add_coupling")
        self.assertGreater(
            credited["add_coupling"]["mean_reward"], credited["add_lag"]["mean_reward"]
        )
        self.assertIn("**i", under_credit["arithmetic"])

    def test_bad_bandit_input_is_refused(self):
        with self.assertRaises(Invalid):
            adaptive_operator_selection([], "seed")
        with self.assertRaises(Invalid):
            adaptive_operator_selection([], 1, strategy="thompson")
        with self.assertRaises(Invalid):
            adaptive_operator_selection([{"operator": "hack", "outcome": "new_cell"}], 1)
        with self.assertRaises(Invalid):
            adaptive_operator_selection(["not_a_record"], 1)
        with self.assertRaises(Invalid):
            adaptive_operator_selection([], 1, operators=("not_an_operator",))
        with self.assertRaises(Invalid):
            adaptive_operator_selection(
                [{"operator": "add_lag", "outcome": "new_cell"}] * 10_001, 1
            )


class LineageTests(unittest.TestCase):
    CHAIN = [
        ("g1", "add_coupling"),
        ("g2", "add_lag"),
        ("g3", "change_observation_model"),
        ("g4", "adjust_resolution"),
    ]

    def chain(self):
        graph = LineageGraph()
        graph.add_seed("seed_baseline", generation=0, note="Competent baseline template.")
        parent = "seed_baseline"
        for index, (identifier, operator) in enumerate(self.CHAIN, start=1):
            graph.add_child(
                identifier, [parent], operator, "new_cell", generation=index
            )
            parent = identifier
        return graph

    def test_innovation_path_returns_the_exact_operator_sequence(self):
        graph = self.chain()
        path = graph.innovation_path("g4")
        self.assertEqual(path["operators"], [op for _, op in self.CHAIN])
        self.assertEqual(
            path["node_ids"], ["seed_baseline", "g1", "g2", "g3", "g4"]
        )
        self.assertEqual(path["depth"], 4)
        self.assertEqual(path["seed_id"], "seed_baseline")
        self.assertTrue(path["path_is_unique"])
        self.assertEqual(path["generations"], [0, 1, 2, 3, 4])
        self.assertEqual(graph.lineage_depth("g4")["depth"], 4)
        self.assertEqual(graph.lineage_depth("seed_baseline")["depth"], 0)

    def test_ancestry_and_descendants_agree(self):
        graph = self.chain()
        ancestry = graph.ancestry("g4")
        self.assertEqual(
            sorted(ancestry["ancestor_ids"]), ["g1", "g2", "g3", "seed_baseline"]
        )
        self.assertEqual(ancestry["immediate_parent_ids"], ["g3"])
        self.assertEqual(ancestry["seed_ids"], ["seed_baseline"])
        descendants = graph.descendants("seed_baseline")
        self.assertEqual(descendants["descendant_ids"], ["g1", "g2", "g3", "g4"])
        self.assertEqual(descendants["immediate_child_ids"], ["g1"])
        self.assertEqual(graph.descendants("g4")["descendant_count"], 0)

    def test_recombination_produces_two_parent_edges(self):
        graph = self.chain()
        graph.add_seed("seed_alternate", generation=0, note="A distinct template.")
        graph.add_child(
            "b1", ["seed_alternate"], "introduce_shared_driver", "new_cell", generation=1
        )
        merged = graph.add_child(
            "merged", ["g4", "b1"], "recombine", "dominates_existing", generation=5
        )
        self.assertEqual(merged["parent_count"], 2)
        self.assertEqual(merged["origin"], "recombination")
        self.assertEqual(
            sorted(e["parent_id"] for e in merged["edges"]), ["b1", "g4"]
        )
        self.assertTrue(all(e["operator"] == "recombine" for e in merged["edges"]))
        path = graph.innovation_path("merged")
        self.assertEqual(path["operators"][-1], "recombine")
        self.assertEqual(path["depth"], 5)
        self.assertFalse(path["path_is_unique"])
        self.assertEqual(path["merge_points"], 1)
        ancestry = graph.ancestry("merged")
        self.assertIn("b1", ancestry["ancestor_ids"])
        self.assertIn("g1", ancestry["ancestor_ids"])
        self.assertEqual(
            sorted(ancestry["seed_ids"]), ["seed_alternate", "seed_baseline"]
        )

    def test_graph_json_is_well_formed(self):
        graph = self.chain()
        graph.add_seed("seed_alternate", generation=0)
        graph.add_child(
            "b1", ["seed_alternate"], "add_coupling", "dominated_rejected", generation=1
        )
        graph.add_child("merged", ["g4", "b1"], "recombine", "new_cell", generation=5)
        payload = graph.to_graph_json()
        self.assertEqual(set(payload) >= {"nodes", "edges", "scope"}, True)
        identifiers = {node["id"] for node in payload["nodes"]}
        self.assertEqual(payload["node_count"], len(payload["nodes"]))
        self.assertEqual(payload["edge_count"], len(payload["edges"]))
        self.assertEqual(len(identifiers), len(payload["nodes"]))
        for edge in payload["edges"]:
            self.assertIn(edge["source"], identifiers)
            self.assertIn(edge["target"], identifiers)
            self.assertIn(edge["operator"], OPERATORS)
        merged = next(n for n in payload["nodes"] if n["id"] == "merged")
        self.assertEqual(merged["parent_count"], 2)
        self.assertEqual(merged["depth"], 5)
        rejected = next(n for n in payload["nodes"] if n["id"] == "b1")
        self.assertFalse(rejected["accepted"])
        self.assertIn("not a chain of evidence", payload["scope"])

    def test_state_round_trips_through_plain_json(self):
        graph = self.chain()
        restored = LineageGraph.from_dict(graph.to_dict())
        self.assertEqual(canonical(restored.to_dict()), canonical(graph.to_dict()))
        self.assertEqual(
            restored.innovation_path("g4")["operators"],
            graph.innovation_path("g4")["operators"],
        )

    def test_stagnation_report_names_generations_branches_and_operators(self):
        graph = self.chain()
        graph.add_seed("seed_alternate", generation=0)
        for index in range(3):
            graph.add_child(
                f"dud_{index}",
                ["seed_alternate"],
                "adjust_resolution",
                "dominated_rejected",
                generation=5 + index,
            )
        report = graph.stagnation_report(current_generation=9, window=3, min_attempts=2)
        self.assertEqual(report["last_improved_generation"], 4)
        self.assertEqual(report["generations_since_last_improvement"], 5)
        self.assertIn("adjust_resolution", report["exhausted_operators"])
        self.assertNotIn("add_coupling", report["exhausted_operators"])
        branches = {row["seed_id"]: row for row in report["branch_productivity"]}
        self.assertEqual(branches["seed_baseline"]["accepted"], 4)
        self.assertEqual(branches["seed_alternate"]["accepted"], 0)
        self.assertEqual(branches["seed_alternate"]["acceptance_rate"], 0.0)
        operators = {row["operator"]: row for row in report["operator_productivity"]}
        self.assertEqual(operators["adjust_resolution"]["attempts"], 4)
        self.assertEqual(operators["adjust_resolution"]["accepted"], 1)
        self.assertIn("not evidence that the mechanism", report["scope"])

    def test_malformed_lineage_is_refused(self):
        graph = self.chain()
        with self.assertRaises(Invalid):
            graph.add_seed("g1")
        with self.assertRaises(Invalid):
            graph.add_child("g5", ["ghost"], "add_lag", "new_cell", generation=5)
        with self.assertRaises(Invalid):
            graph.add_child("g5", [], "add_lag", "new_cell", generation=5)
        with self.assertRaises(Invalid):
            graph.add_edge("g4", "g1", "add_lag", "new_cell")
        with self.assertRaises(Invalid):
            graph.add_edge("seed_baseline", "g1", "add_lag", "new_cell")
        with self.assertRaises(Invalid):
            graph.add_child("g6", ["g4"], "add_lag", "teleported", generation=6)
        with self.assertRaises(Invalid):
            graph.ancestry("ghost")
        with self.assertRaises(Invalid):
            graph.stagnation_report(window=0)


class EndToEndSearchTests(unittest.TestCase):
    """One small search wiring archive, operators and lineage together."""

    def test_grammar_archive_and_lineage_agree_on_one_seeded_search(self):
        archive = Archive(
            ["coupling_count", "resolution_rung"],
            [linear_bins(0.0, 4.0, 4), linear_bins(0.0, 3.0, 3)],
            RASTRIGIN_OBJECTIVES,
            descriptor_fn=lambda c: (
                float(len(c["genome"]["couplings"])),
                float(("coarse", "medium", "fine").index(c["genome"]["inference"]["resolution"])),
            ),
            objective_fn=lambda c: {
                "rastrigin_loss": float(20 - 3 * len(c["genome"]["couplings"])),
                "parameter_cost": float(len(c["genome"]["components"])),
            },
            selection_strategy="curiosity",
            front_capacity=2,
        )
        graph = LineageGraph()
        base = seed_genome()
        archive.add({"id": "seed", "genome": base})
        graph.add_seed("seed", generation=0)
        history = []
        plan = [
            ("add_coupling", {"source": "demand", "target": "capacity", "lag": 1,
                              "mechanism": "declared provisioning response"}),
            ("adjust_resolution", {"resolution": "fine"}),
        ]
        parent_genome, parent_id = base, "seed"
        for generation, (name, arguments) in enumerate(plan, start=1):
            choice = adaptive_operator_selection(history, generation)
            self.assertIn(choice["selected"], OPERATORS)
            produced = apply_operator(name, parent_genome, parent_id=parent_id, **arguments)
            validate_patch(parent_genome, produced["patch"])
            child_id = f"child_{generation}"
            outcome = archive.add(
                {"id": child_id, "genome": produced["genome"]},
                parent_ids=[parent_id],
                operator=name,
            )
            graph.add_child(
                child_id, [parent_id], name, outcome["outcome"], generation=generation,
                cell_id=outcome["cell_id"],
            )
            history.append({"operator": name, "outcome": outcome["outcome"]})
            archive.advance_generation()
            parent_genome, parent_id = produced["genome"], child_id
        self.assertEqual(archive.coverage()["occupied_cells"], 3)
        self.assertEqual(
            graph.innovation_path("child_2")["operators"],
            ["add_coupling", "adjust_resolution"],
        )
        self.assertEqual(
            [record["outcome"] for record in history], ["new_cell", "new_cell"]
        )
        informed = adaptive_operator_selection(
            history, 5, operators=("add_coupling", "adjust_resolution")
        )
        self.assertEqual(informed["basis"], "ucb1")
        self.assertFalse(archive.metrics()["promotion_allowed"])


if __name__ == "__main__":
    unittest.main()
