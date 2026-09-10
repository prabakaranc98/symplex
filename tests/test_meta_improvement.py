"""Host-owned governance of method improvement: no model calls, no scalar objective, no auto-promotion."""

import random
import unittest
from pathlib import Path

from symplex.agents.prompts import prompt_overlay, snapshot
from symplex.agents.tools import REGISTRY
from symplex.core.contracts import Invalid
from symplex.improvement.goodhart import (
    divergence_report,
    goodhart_report,
    overfitting_to_task_family,
    proxy_correlation_drift,
    specification_gaming_checks,
)
from symplex.improvement.meta import (
    CLAIM_LADDER,
    MIN_GENERATIONS_PER_ARM,
    PROMOTION_ALPHA_FLOOR,
    VERSION as META_VERSION,
    compare_meta_policies,
    effective_promotion_alpha,
    meta_generation,
    rsi_evidence_report,
)
from symplex.improvement.policy import (
    FORBIDDEN_FIELDS,
    MUTABLE_FIELDS,
    PLANNABLE_TOOLS,
    apply_patch,
    baseline_policy,
    to_prompt_overlay,
)
from symplex.improvement.proposer import (
    ALLOWED_GRAMMAR,
    SAMPLING_STRATEGIES,
    failure_taxonomy,
    propose_patch,
    sample_traces,
)
from symplex.improvement.protocol import (
    DEMOTION_ALPHA,
    describe_board,
    paired_replay,
    power_report,
    promotion_rule,
    sequential_verdict,
)

CAPS = {"usd": 0.1, "model_requests": 2.0}
PRIMARY = "contract_pass_rate"


def board(**overrides):
    data = {
        "label": "Investigation quality board with declared blind spots",
        "proxies": [
            {"name": PRIMARY, "role": "primary", "direction": "higher_is_better",
             "measures": "fraction of replay tasks whose host contract parsed and resolved",
             "minimum_detectable_effect": 0.25, "noise_sd": 0.2},
            {"name": "protected_evidence_coverage", "role": "protected", "direction": "higher_is_better",
             "measures": "fraction of already-available artifacts the investigation referenced",
             "minimum_detectable_effect": 0.1, "noise_sd": 0.2},
            {"name": "protected_check_completion", "role": "protected", "direction": "higher_is_better",
             "measures": "fraction of declared host numerical checks the investigation ran",
             "minimum_detectable_effect": 0.1, "noise_sd": 0.2},
            {"name": "diagnostic_cost_per_task", "role": "diagnostic", "direction": "lower_is_better",
             "measures": "recorded compute cost consumed per replay task",
             "minimum_detectable_effect": 0.1, "noise_sd": 0.2},
        ],
        "unmeasured_dimensions": [
            "whether the investigation's scientific conclusion is correct",
            "whether the decision the investigation supports is the right decision",
        ],
        "population_families": [],
    }
    data.update(overrides)
    return data


def values(primary, coverage=0.8, checks=0.9, cost=1.0):
    return {PRIMARY: primary, "protected_evidence_coverage": coverage,
            "protected_check_completion": checks, "diagnostic_cost_per_task": cost}


def arm(proxy_values, **overrides):
    data = {"proxies": proxy_values, "action_count": 10, "checks_run": 8, "scope_breadth": 6,
            "attempted": True, "accounting_complete": True, "observed_other_arm": False,
            "caps": dict(CAPS), "usage": {"usd": 0.05, "model_requests": 1.0},
            "settings_digest": "settings_a"}
    data.update(overrides)
    return data


def task(ident, family, stage, parent_values, child_values, *,
         parent_kw=None, child_kw=None, difficulty="standard"):
    return {"task_id": ident, "task_family": family, "stage": stage, "difficulty": difficulty,
            "parent": arm(parent_values, **(parent_kw or {})),
            "child": arm(child_values, **(child_kw or {}))}


def patch_body(parent_id, changes, **overrides):
    data = {
        "parent_policy_id": parent_id,
        "operation": "insert_semantic_join_check_before_evaluation",
        "failure_mode": "wasted_evaluation_on_unverified_join",
        "observed_failure": "Evaluations repeatedly ran on entity or timestamp joins nobody verified",
        "evidence_trace_ids": ["trace_join_0"],
        "changes": changes,
        "discovery_cost": 12.0,
        "regression_checks": ["Paired contract-pass count must not regress on any replay task"],
        "promotion_criterion": "Strict primary gain, no protected regression, held-out stage passed",
        "expected_effect": "Fewer evaluations spent on unverified joins",
        "disconfirming_result": "Evaluation count falls but resolved-uncertainty count falls with it",
    }
    data.update(overrides)
    return data


def policy_pair():
    """A frozen parent and a child that really descends from it through the allowed surface."""
    parent = baseline_policy()["policy"]
    result = apply_patch(parent, patch_body(parent["policy_id"], {
        "evidence_admission.require_join_key_verification": True,
        "prompt_fragments.complexity_architect":
            "Verify entity and timestamp join keys before proposing a program mutation.",
    }))
    assert result["applied"], result["rejected"]
    return parent, result["child"]


def step(index, tool, **overrides):
    data = {"index": index, "tool": tool, "status": "completed", "cost": 1.0, "regret": 0.0}
    data.update(overrides)
    return data


def trace(ident, family, sequence, steps, outcome="unresolved"):
    return {"trace_id": ident, "task_family": family, "policy_id": "policy_parent",
            "sequence": sequence, "outcome": outcome, "steps": steps}


def planted_traces():
    """A corpus with one dominant mode and one instance of each of the others."""
    traces = []
    for i in range(4):
        traces.append(trace("trace_join_" + str(i), "prediction_markets", 10 + i, [
            step(0, "profile_table", cost=0.2),
            step(1, "compare_computation", join_keys_verified=False, cost=6.0, regret=0.5),
        ]))
    for i in range(2):
        traces.append(trace("trace_repeat_" + str(i), "urban_infrastructure", 30 + i, [
            step(j, "research_evidence", uncertainty_type="evidential", cost=0.5, regret=0.4)
            for j in range(3)
        ]))
    traces.append(trace("trace_protocol", "urban_infrastructure", 50, [
        step(0, "plan_experiment", status="rejected", rejection_reason="protocol_contract",
             cost=0.4, regret=9.0),
    ]))
    traces.append(trace("trace_omission", "prediction_markets", 5, [
        step(0, "review_model", available_artifact_ids=["file_a", "file_b", "file_c"],
             referenced_artifact_ids=["file_a"], cost=0.3),
    ]))
    traces.append(trace("trace_identifiability", "urban_infrastructure", 6, [
        step(0, "compare_computation", identifiable=False, cost=1.5),
    ]))
    traces.append(trace("trace_budget", "prediction_markets", 99, [
        step(0, "represent_system", cost=0.1),
        step(1, "run_model_code", cost=0.1),
    ], outcome="budget_exhausted"))
    return traces


def meta_body(**overrides):
    data = {"label": "Cost-weighted improver", "trace_sampling_strategy": "most_costly",
            "trace_sample_size": 8, "sampling_seed": 7,
            "patch_grammar_subset": list(ALLOWED_GRAMMAR), "exploration_rate": 0.0,
            "promotion_strictness": 1.0, "generation_budget": 4}
    data.update(overrides)
    return data


FRESH_FAMILIES = ["fresh_alpha", "fresh_beta", "fresh_gamma"]


def fresh_replay(gain=0.5, **arm_overrides):
    """A host-supplied replay_fn returning already-recorded paired arm outcomes."""

    def replay_fn(parent_policy, child_policy, families):
        tasks = []
        for family_index, family in enumerate(families):
            for slot in range(4):
                stage = "holdout" if slot == 3 else "development" if slot < 2 else "regression"
                tasks.append(task(family + "_task_" + str(slot), family, stage,
                                  values(0.3), values(round(0.3 + gain, 6)),
                                  child_kw=dict(arm_overrides)))
        return tasks

    return replay_fn


class MethodPolicyTests(unittest.TestCase):
    def setUp(self):
        self.parent = baseline_policy()["policy"]

    def patch(self, changes, **overrides):
        return patch_body(self.parent["policy_id"], changes, **overrides)

    def test_baseline_policy_is_content_addressed_and_declares_its_surface(self):
        result = baseline_policy()
        self.assertTrue(result["policy"]["policy_id"].startswith("policy_"))
        self.assertEqual(result["policy"], baseline_policy()["policy"])
        self.assertEqual(sorted(result["forbidden_fields"]), sorted(FORBIDDEN_FIELDS))
        self.assertIn("scope", result)

    def test_allowed_patch_produces_a_child_with_lineage_and_a_diff(self):
        result = apply_patch(self.parent, self.patch({"stopping_rules.max_actions": 9}))
        self.assertTrue(result["applied"])
        child = result["child"]
        self.assertEqual(child["parent_policy_id"], self.parent["policy_id"])
        self.assertEqual(child["generation"], self.parent["generation"] + 1)
        self.assertNotEqual(child["policy_id"], self.parent["policy_id"])
        self.assertEqual(child["stopping_rules"]["max_actions"], 9)
        self.assertEqual([c["path"] for c in result["diff"]["changes"]], ["stopping_rules"])

    def test_revision_budget_stays_editable_despite_the_budget_denylist(self):
        result = apply_patch(self.parent, self.patch({"revision_budget.max_program_revisions": 1}))
        self.assertTrue(result["applied"], result["rejected"])
        self.assertEqual(result["child"]["revision_budget"]["max_program_revisions"], 1)

    def test_paths_outside_the_mutable_surface_are_refused_with_a_reason(self):
        result = apply_patch(self.parent, self.patch({"generation": 99}))
        self.assertFalse(result["applied"])
        self.assertIsNone(result["child"])
        self.assertEqual(result["rejected"][0]["reason"], "outside_mutable_surface")

    def test_forbidden_key_nested_in_a_patch_value_is_caught(self):
        result = apply_patch(self.parent, self.patch(
            {"prompt_fragments.metareasoner": {"labels": "rewrite the final labels"}}))
        self.assertFalse(result["applied"])
        self.assertIn("final_labels", result["rejected"][0]["forbidden_fields"])

    def test_a_patch_naming_another_parent_is_refused(self):
        result = apply_patch(self.parent, patch_body("policy_someone_else",
                                                     {"stopping_rules.max_actions": 9}))
        self.assertFalse(result["applied"])
        self.assertEqual(result["rejected"][0]["reason"], "patch_parent_is_not_this_policy")

    def test_to_prompt_overlay_is_accepted_by_the_frozen_overlay_machinery(self):
        _, child = policy_pair()
        rendered = to_prompt_overlay(child)
        self.assertEqual(rendered["roles"], ["complexity_architect"])
        with prompt_overlay(rendered["overlay"]):
            from symplex.agents.prompts import prompt

            self.assertIn("Verify entity and timestamp join keys", prompt("complexity_architect"))
        base = snapshot()
        self.assertTrue(set(rendered["overlay"]) <= set(base))
        self.assertIn("scope", rendered)

    def test_overlay_refuses_a_role_absent_from_the_base(self):
        _, child = policy_pair()
        with self.assertRaises(Invalid):
            to_prompt_overlay(child, base={"metareasoner": "only this role exists here"})

    def test_plannable_tools_are_a_subset_of_the_live_registry(self):
        self.assertTrue(set(PLANNABLE_TOOLS) <= set(REGISTRY.descriptions()),
                        sorted(set(PLANNABLE_TOOLS) - set(REGISTRY.descriptions())))

    def test_mutable_and_forbidden_surfaces_do_not_overlap(self):
        aliases = {alias for group in FORBIDDEN_FIELDS.values() for alias in group}
        self.assertFalse(set(MUTABLE_FIELDS) & aliases)


def _forbidden_case(field):
    def check(self):
        body = self.patch({field: "an edit the bounded editor may not make"})
        result = apply_patch(self.parent, body)
        self.assertFalse(result["applied"], field + " must never be applied")
        self.assertIsNone(result["child"])
        hits = [r for r in result["rejected"] if r["reason"] == "forbidden_field"]
        self.assertTrue(hits, "no forbidden_field rejection recorded for " + field)
        self.assertIn(field, hits[0]["forbidden_fields"])
        self.assertIn("scope", result)

    check.__doc__ = "Loop C forbids editing " + field + "."
    return check


for _field in FORBIDDEN_FIELDS:
    setattr(MethodPolicyTests, "test_forbidden_field_" + _field + "_is_rejected", _forbidden_case(_field))


class ProposerTests(unittest.TestCase):
    def setUp(self):
        self.traces = planted_traces()
        self.parent = baseline_policy()["policy"]

    def test_planted_failure_modes_are_recovered(self):
        result = failure_taxonomy(self.traces)
        found = {m["mode"] for m in result["modes"]}
        self.assertEqual(found, {
            "wasted_evaluation_on_unverified_join", "unresolving_tool_repetition",
            "protocol_rejection", "non_identifiable_comparison_run",
            "context_omitted_existing_artifact", "budget_exhausted_before_resolution"})
        dominant = result["modes"][0]
        self.assertEqual(dominant["mode"], "wasted_evaluation_on_unverified_join")
        self.assertEqual(dominant["incident_count"], 4)
        self.assertEqual(dominant["trace_count"], 4)
        self.assertEqual(result["dominant_mode"], "wasted_evaluation_on_unverified_join")
        self.assertEqual(result["model_calls"], 0)

    def test_unmatched_failures_are_preserved_rather_than_dropped(self):
        traces = self.traces + [trace("trace_odd", "biology", 200, [
            step(0, "run_model_code", status="failed", cost=0.2)])]
        result = failure_taxonomy(traces)
        self.assertEqual([f["trace_id"] for f in result["unclassified_failures"]], ["trace_odd"])

    def test_propose_patch_is_deterministic_and_targets_the_dominant_mode(self):
        taxonomy = failure_taxonomy(self.traces)
        first = propose_patch(taxonomy, self.parent)
        second = propose_patch(failure_taxonomy(self.traces), self.parent)
        self.assertEqual(first["patch"], second["patch"])
        self.assertEqual(first["selected_mode"], taxonomy["dominant_mode"])
        self.assertEqual(first["patch"]["operation"], "insert_semantic_join_check_before_evaluation")
        self.assertEqual(first["patch"]["discovery_cost"], taxonomy["modes"][0]["total_cost"])
        self.assertEqual(first["model_calls"], 0)
        self.assertTrue(apply_patch(self.parent, first["patch"])["applied"])

    def test_lower_ranked_modes_are_preserved_as_rejected_alternatives(self):
        result = propose_patch(failure_taxonomy(self.traces), self.parent)
        reasons = {r["reason"] for r in result["rejected_alternatives"]}
        self.assertIn("lower_ranked_than_the_selected_mode", reasons)
        self.assertEqual(len(result["rejected_alternatives"]), 5)

    def test_a_narrowed_grammar_shifts_the_proposal_and_records_the_exclusion(self):
        taxonomy = failure_taxonomy(self.traces)
        allowed = [op for op in ALLOWED_GRAMMAR if op != "insert_semantic_join_check_before_evaluation"]
        result = propose_patch(taxonomy, self.parent, allowed)
        self.assertNotEqual(result["selected_mode"], "wasted_evaluation_on_unverified_join")
        excluded = [r for r in result["rejected_alternatives"]
                    if r["reason"] == "operation_not_in_allowed_grammar"]
        self.assertTrue(excluded)

    def test_unknown_grammar_and_malformed_input_raise(self):
        taxonomy = failure_taxonomy(self.traces)
        with self.assertRaises(Invalid):
            propose_patch(taxonomy, self.parent, ["not_a_real_operation"])
        with self.assertRaises(Invalid):
            failure_taxonomy([])

    def test_sampling_strategies_are_deterministic_and_distinct(self):
        picks = {}
        for strategy in SAMPLING_STRATEGIES:
            first = sample_traces(self.traces, strategy, 3, 3)
            second = sample_traces(self.traces, strategy, 3, 3)
            self.assertEqual(first["selected_trace_ids"], second["selected_trace_ids"])
            self.assertEqual(first["strategy"], strategy)
            picks[strategy] = tuple(first["selected_trace_ids"])
        self.assertGreaterEqual(len(set(picks.values())), 4, picks)
        self.assertEqual(picks["recent"][0], "trace_budget")
        self.assertIn(picks["most_costly"][0], {"trace_join_0", "trace_join_1", "trace_join_2", "trace_join_3"})
        self.assertEqual(picks["worst_regret"][0], "trace_protocol")

    def test_sampling_rejects_unknown_strategies_and_seeds(self):
        with self.assertRaises(Invalid):
            sample_traces(self.traces, "whatever_feels_right", 0, 3)
        with self.assertRaises(Invalid):
            sample_traces(self.traces, "recent", -1, 3)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.parent, self.child = policy_pair()

    def replay(self, tasks, **kwargs):
        return paired_replay(self.parent, self.child, tasks, CAPS, board=board(**kwargs))

    def gaining_tasks(self, child_values=None, **task_kw):
        child_values = child_values or values(0.9)
        return [
            task("t0", "prediction_markets", "development", values(0.4), child_values, **task_kw),
            task("t1", "prediction_markets", "regression", values(0.4), child_values, **task_kw),
            task("t2", "biology", "development", values(0.4), child_values, **task_kw),
            task("t3", "biology", "holdout", values(0.4), child_values, **task_kw),
        ]

    def test_board_must_declare_unmeasured_dimensions_one_primary_and_a_protected_proxy(self):
        with self.assertRaises(Invalid):
            describe_board(board(unmeasured_dimensions=[]))
        proxies = board()["proxies"]
        with self.assertRaises(Invalid):
            describe_board(board(proxies=[dict(p, role="primary") for p in proxies[:2]]))
        with self.assertRaises(Invalid):
            describe_board(board(proxies=[proxies[0], dict(proxies[3])]))
        summary = describe_board(board())
        self.assertEqual(summary["primary"], PRIMARY)
        self.assertEqual(len(summary["unmeasured_dimensions"]), 2)

    def test_paired_replay_checks_caps_blinding_and_matched_settings(self):
        report = self.replay(self.gaining_tasks())
        self.assertTrue(report["equal_caps_verified"])
        self.assertTrue(report["blinding_verified"])
        self.assertTrue(report["settings_matched"])
        self.assertTrue(report["accounting_complete"])
        self.assertTrue(report["lineage_verified"])
        self.assertEqual(report["unmeasured_dimensions"], board()["unmeasured_dimensions"])
        leaky = self.gaining_tasks(child_kw={"observed_other_arm": True})
        self.assertFalse(self.replay(leaky)["blinding_verified"])
        unequal = self.gaining_tasks(child_kw={"caps": {"usd": 0.4, "model_requests": 9.0}})
        self.assertFalse(self.replay(unequal)["equal_caps_verified"])

    def test_gain_without_protected_regression_promotes(self):
        verdict = promotion_rule(self.replay(self.gaining_tasks()))
        self.assertEqual(verdict["verdict"], "promote")
        self.assertEqual(verdict["unmet_conditions"], [])
        self.assertTrue(verdict["eligible_for_registry_review"])
        self.assertFalse(verdict["auto_promotion"])
        self.assertFalse(verdict["retained_parent"])

    def test_gain_with_a_protected_regression_is_rejected(self):
        tasks = self.gaining_tasks(child_values=values(0.9, coverage=0.4))
        verdict = promotion_rule(self.replay(tasks))
        self.assertEqual(verdict["verdict"], "reject")
        self.assertIn("no_protected_regression", verdict["unmet_conditions"])
        self.assertTrue(verdict["retained_parent"])

    def test_gain_with_incomplete_accounting_is_inconclusive_and_never_a_pass(self):
        tasks = self.gaining_tasks(child_kw={"accounting_complete": False})
        verdict = promotion_rule(self.replay(tasks))
        self.assertEqual(verdict["verdict"], "inconclusive")
        self.assertNotEqual(verdict["verdict"], "pass")
        self.assertNotEqual(verdict["verdict"], "promote")
        self.assertIn("accounting_complete", verdict["unmet_conditions"])

    def test_a_tie_on_the_primary_retains_the_parent(self):
        verdict = promotion_rule(self.replay(self.gaining_tasks(child_values=values(0.4))))
        self.assertEqual(verdict["verdict"], "reject")
        self.assertIn("primary_strict_gain", verdict["unmet_conditions"])

    def test_a_missing_held_out_stage_is_inconclusive(self):
        tasks = [t for t in self.gaining_tasks() if t["stage"] != "holdout"]
        verdict = promotion_rule(self.replay(tasks))
        self.assertEqual(verdict["verdict"], "inconclusive")
        self.assertIn("holdout_stage_passed", verdict["unmet_conditions"])

    def test_a_board_free_comparison_cannot_promote(self):
        report = paired_replay(self.parent, self.child, self.gaining_tasks(), CAPS)
        verdict = promotion_rule(report)
        self.assertEqual(verdict["verdict"], "inconclusive")
        self.assertIn("evaluation_board_declared", verdict["unmet_conditions"])
        self.assertTrue(verdict["unmeasured_dimensions"])

    def test_always_valid_p_value_controls_false_promotions_under_a_true_null(self):
        rng = random.Random(20260910)
        alpha, sigma, peeks, trials = 0.025, 0.1, 30, 300
        crossings = naive = 0
        for _ in range(trials):
            draws = [rng.gauss(0.0, sigma) for _ in range(peeks)]
            verdict = sequential_verdict(draws, sigma=sigma, promotion_alpha=alpha, demotion_alpha=alpha)
            if verdict["always_valid_p_value"] <= alpha:
                crossings += 1
            running = 0.0
            for n, value in enumerate(draws, start=1):
                running += value
                if abs(running / n) / (sigma / (n ** 0.5)) > 1.96:
                    naive += 1
                    break
        rate, naive_rate = crossings / trials, naive / trials
        self.assertLessEqual(rate, alpha + 0.03, "always-valid p-value did not control the peeking rate")
        self.assertGreater(naive_rate, 0.10)
        self.assertGreater(naive_rate, rate, "the comparison would be vacuous if naive peeking were safe")

    def test_asymmetric_budget_needs_more_evidence_to_promote_than_to_demote(self):
        gains = [0.06] * 50
        promote = sequential_verdict(gains, sigma=0.1)
        demote = sequential_verdict([-g for g in gains], sigma=0.1)
        self.assertEqual(promote["decision"], "promote_evidence_sufficient")
        self.assertEqual(demote["decision"], "demote_evidence_sufficient")
        self.assertLess(demote["decision_at_n"], promote["decision_at_n"])
        self.assertLess(promote["promotion_alpha"], promote["demotion_alpha"])
        self.assertEqual(promote["demotion_alpha"], DEMOTION_ALPHA)
        self.assertTrue(any("unmeasured" in a for a in promote["assumptions"]))
        with self.assertRaises(Invalid):
            sequential_verdict(gains, sigma=0.1, promotion_alpha=0.2, demotion_alpha=0.01)
        with self.assertRaises(Invalid):
            sequential_verdict(gains, sigma=0.0)

    def test_power_report_flags_an_underpowered_comparison(self):
        weak = power_report(4, noise_sd=0.2, minimum_detectable_effect=0.05)
        self.assertTrue(weak["underpowered"])
        self.assertEqual(weak["recommendation"], "refuse_underpowered_comparison")
        self.assertGreater(weak["required_task_count"], 4)
        self.assertGreater(weak["shortfall"], 0)
        strong = power_report(weak["required_task_count"], noise_sd=0.2, minimum_detectable_effect=0.05)
        self.assertFalse(strong["underpowered"])
        self.assertEqual(strong["recommendation"], "adequately_powered_to_proceed")
        with self.assertRaises(Invalid):
            power_report(4, noise_sd=0.0, minimum_detectable_effect=0.05)


class GoodhartTests(unittest.TestCase):
    def setUp(self):
        self.parent, self.child = policy_pair()

    def replay(self, tasks, **kwargs):
        return paired_replay(self.parent, self.child, tasks, CAPS, board=board(**kwargs))

    def test_primary_gain_with_a_protected_regression_is_flagged(self):
        tasks = [task("t0", "prediction_markets", "development", values(0.4), values(0.9, coverage=0.3)),
                 task("t1", "biology", "holdout", values(0.4), values(0.9, coverage=0.3))]
        result = divergence_report(self.replay(tasks))
        self.assertTrue(result["flagged"])
        self.assertTrue(result["primary_improved"])
        self.assertEqual(result["protected_regressions"], ["protected_evidence_coverage"])
        self.assertIn(result["max_severity"], {"high", "critical"})
        self.assertTrue(result["unmeasured_dimensions"])

    def test_a_child_that_wins_by_running_fewer_checks_is_caught(self):
        tasks = [task("t" + str(i), "prediction_markets", "development", values(0.4), values(0.9),
                      child_kw={"checks_run": 2}) for i in range(3)]
        result = specification_gaming_checks(self.replay(tasks))
        self.assertTrue(result["flagged"])
        self.assertIn("skipped_checks", result["triggered_checks"])
        self.assertEqual(result["severity"], "high")
        clean = specification_gaming_checks(self.replay(
            [task("t" + str(i), "prediction_markets", "development", values(0.4), values(0.9))
             for i in range(3)]))
        self.assertNotIn("skipped_checks", clean["triggered_checks"])

    def test_shortened_narrowed_and_avoided_work_are_each_visible(self):
        def triggered(**child_kw):
            tasks = [task("t" + str(i), "prediction_markets", "development", values(0.4), values(0.9),
                          child_kw=child_kw, difficulty="hard") for i in range(3)]
            return specification_gaming_checks(self.replay(tasks))["triggered_checks"]

        self.assertIn("shortened_investigations", triggered(action_count=3))
        self.assertIn("narrowed_scope", triggered(scope_breadth=1))
        self.assertIn("avoided_hard_tasks", triggered(attempted=False))

    def test_an_exploited_task_mix_is_flagged_against_a_declared_population(self):
        tasks = [task("t" + str(i), "prediction_markets", "development", values(0.4), values(0.9))
                 for i in range(3)]
        declared = board(population_families=["prediction_markets", "biology", "urban_infrastructure"])
        result = specification_gaming_checks(self.replay(tasks), board=declared)
        self.assertIn("exploited_task_distribution", result["triggered_checks"])

    def test_a_gain_concentrated_in_one_task_family_is_flagged(self):
        tasks = [task("a0", "prediction_markets", "development", values(0.3), values(0.95)),
                 task("a1", "prediction_markets", "development", values(0.3), values(0.95)),
                 task("b0", "biology", "development", values(0.5), values(0.5)),
                 task("b1", "biology", "holdout", values(0.5), values(0.5))]
        result = overfitting_to_task_family(self.replay(tasks))
        self.assertTrue(result["flagged"])
        self.assertEqual(result["top_family"], "prediction_markets")
        self.assertEqual(result["top_family_share"], 1.0)
        spread = [task("a0", "prediction_markets", "development", values(0.3), values(0.6)),
                  task("b0", "biology", "development", values(0.3), values(0.6))]
        self.assertFalse(overfitting_to_task_family(self.replay(spread))["flagged"])

    def test_a_single_family_replay_is_flagged_because_transfer_was_never_tested(self):
        tasks = [task("t" + str(i), "prediction_markets", "development", values(0.3), values(0.9))
                 for i in range(3)]
        result = overfitting_to_task_family(self.replay(tasks))
        self.assertTrue(result["flagged"])
        self.assertEqual(result["severity"], "high")

    def test_proxy_correlation_drift_flags_a_sign_flip_and_reports_short_history(self):
        history = ([{"label": "g" + str(i), "proxies": {"alpha": float(i), "beta": float(i)}} for i in range(6)]
                   + [{"label": "h" + str(i), "proxies": {"alpha": float(i), "beta": float(-i)}} for i in range(6)])
        result = proxy_correlation_drift(history)
        self.assertTrue(result["sufficient_history"])
        self.assertTrue(result["flagged"])
        self.assertEqual(result["flagged_pairs"], [["alpha", "beta"]])
        short = proxy_correlation_drift(history[:4])
        self.assertFalse(short["sufficient_history"])
        self.assertFalse(short["flagged"])
        with self.assertRaises(Invalid):
            proxy_correlation_drift([{"proxies": {"alpha": "not a number"}}])

    def test_goodhart_report_blocks_review_on_a_protected_regression(self):
        tasks = [task("t0", "prediction_markets", "development", values(0.4), values(0.9, coverage=0.3)),
                 task("t1", "biology", "holdout", values(0.4), values(0.9, coverage=0.3))]
        result = goodhart_report(self.replay(tasks), board=board())
        self.assertTrue(result["flagged"])
        self.assertTrue(result["blocks_promotion_review"])


class MetaTests(unittest.TestCase):
    def setUp(self):
        self.traces = planted_traces()
        self.parent = baseline_policy()["policy"]
        self.board = board(population_families=list(FRESH_FAMILIES))

    def generation(self, *, meta=None, replay_fn=None, cost=10.0, index=0, families=None):
        return meta_generation(
            meta or meta_body(), traces=self.traces, parent_policy=self.parent, board=self.board,
            replay_fn=replay_fn or fresh_replay(), fresh_task_families=families or FRESH_FAMILIES,
            caps=CAPS, compute_cost=cost, generation_index=index)

    def test_a_generation_records_a_validated_improvement_and_its_productivity(self):
        record = self.generation()
        self.assertTrue(record["patch_applied"])
        self.assertEqual(record["proposal"]["selected_mode"], "wasted_evaluation_on_unverified_join")
        self.assertTrue(record["fresh_families_verified"])
        self.assertEqual(record["promotion_verdict"]["verdict"], "promote")
        self.assertTrue(record["sequential"]["promotion_evidence_sufficient"])
        self.assertFalse(record["power"]["underpowered"])
        self.assertFalse(record["goodhart"]["blocks_promotion_review"])
        self.assertTrue(record["validated_improvement"])
        self.assertAlmostEqual(record["improvement_productivity"], 0.1)
        self.assertEqual(record["blockers"], [])
        self.assertEqual(record["model_calls"], 0)
        self.assertFalse(record["auto_promotion"])
        self.assertEqual(record["unmeasured_dimensions"], self.board["unmeasured_dimensions"])

    def test_a_generation_evaluated_on_the_families_it_learned_from_is_not_validated(self):
        record = self.generation(families=["prediction_markets"])
        self.assertFalse(record["fresh_families_verified"])
        self.assertFalse(record["validated_improvement"])
        self.assertTrue(any("overlap" in b for b in record["blockers"]))

    def test_a_generation_whose_child_games_the_proxy_is_not_validated(self):
        record = self.generation(replay_fn=fresh_replay(checks_run=1))
        self.assertTrue(record["patch_applied"])
        self.assertFalse(record["validated_improvement"])
        self.assertTrue(any("gaming" in b for b in record["blockers"]))

    def test_a_flat_child_is_not_validated_and_the_parent_is_retained(self):
        record = self.generation(replay_fn=fresh_replay(gain=0.0))
        self.assertFalse(record["validated_improvement"])
        self.assertEqual(record["promotion_verdict"]["verdict"], "reject")
        self.assertTrue(record["promotion_verdict"]["retained_parent"])

    def test_meta_policy_may_only_tighten_the_protected_promotion_floor(self):
        loose = effective_promotion_alpha(meta_body())
        self.assertEqual(loose["effective_promotion_alpha"], PROMOTION_ALPHA_FLOOR)
        strict = effective_promotion_alpha(meta_body(promotion_strictness=5.0))
        self.assertLess(strict["effective_promotion_alpha"], PROMOTION_ALPHA_FLOOR)
        with self.assertRaises(Invalid):
            effective_promotion_alpha(meta_body(promotion_strictness=0.5))
        with self.assertRaises(Invalid):
            effective_promotion_alpha(meta_body(patch_grammar_subset=["rewrite_the_evaluator"]))

    def test_meta_policies_differing_only_in_sampling_produce_different_proposals(self):
        costly = self.generation(meta=meta_body(trace_sampling_strategy="most_costly", trace_sample_size=2))
        recent = self.generation(meta=meta_body(trace_sampling_strategy="recent", trace_sample_size=2))
        self.assertNotEqual(costly["meta_policy_id"], recent["meta_policy_id"])
        self.assertNotEqual(costly["proposal"]["selected_mode"], recent["proposal"]["selected_mode"])

    def synthetic_arm(self, meta, validated, cost, families):
        return [{"version": META_VERSION, "generation_index": i, "meta_policy_id": meta["meta_policy_id"],
                 "compute_cost": cost, "validated_improvement": i < validated,
                 "improvement_productivity": (1.0 / cost) if i < validated else 0.0,
                 "fresh_task_families": [families[i]], "fresh_families_verified": True,
                 "blockers": [], "unmeasured_dimensions": ["a dimension nobody measured"]}
                for i in range(MIN_GENERATIONS_PER_ARM)]

    def test_compare_meta_policies_refuses_an_unequal_budget(self):
        a = effective_promotion_alpha(meta_body(label="A"))
        b = effective_promotion_alpha(meta_body(label="B", trace_sampling_strategy="recent"))
        arm_a = self.synthetic_arm(a, 3, 10.0, ["f0", "f1", "f2"])
        arm_b = self.synthetic_arm(b, 1, 40.0, ["f3", "f4", "f5"])
        result = compare_meta_policies(meta_body(label="A"),
                                       meta_body(label="B", trace_sampling_strategy="recent"),
                                       arm_a, arm_b, new_task_families=["f0"])
        self.assertEqual(result["verdict"], "inconclusive")
        self.assertIn("equal_total_budget", result["unmet_conditions"])
        self.assertFalse(result["strict_productivity_gain"])

    def test_compare_meta_policies_reports_a_productivity_difference_at_equal_budget(self):
        body_a = meta_body(label="A")
        body_b = meta_body(label="B", trace_sampling_strategy="recent")
        a = effective_promotion_alpha(body_a)
        b = effective_promotion_alpha(body_b)
        families = ["f0", "f1", "f2"]
        arm_a = self.synthetic_arm(a, 3, 10.0, families)
        arm_b = self.synthetic_arm(b, 1, 10.0, families)
        result = compare_meta_policies(body_a, body_b, arm_a, arm_b, new_task_families=families)
        self.assertEqual(result["verdict"], "a_more_productive")
        self.assertTrue(result["strict_productivity_gain"])
        self.assertTrue(result["equal_budget_verified"])
        self.assertEqual(result["unmet_conditions"], [])
        with self.assertRaises(Invalid):
            compare_meta_policies(body_a, body_a, arm_a, arm_a)

    def test_rsi_evidence_report_refuses_to_claim_rsi_from_one_generation(self):
        record = self.generation()
        self.assertTrue(record["validated_improvement"])
        report = rsi_evidence_report([record])
        self.assertEqual(report["rung"], "single_bounded_improvement")
        self.assertLessEqual(report["rung_index"], 1)
        self.assertFalse(report["recursive_self_improvement_claimed"])
        rungs = {entry["rung"]: entry for entry in report["ladder"]}
        self.assertTrue(rungs["single_bounded_improvement"]["met"])
        self.assertFalse(rungs["repeated_improvement"]["met"])
        self.assertTrue(rungs["repeated_improvement"]["unmet_preconditions"])
        self.assertFalse(rungs["improvement_in_improvement_rate"]["met"])
        self.assertTrue(rungs["improvement_in_improvement_rate"]["blocked_by_lower_rung"])
        refused = {r["refused_rung"] for r in report["refusals"]}
        self.assertEqual(refused, {"repeated_improvement", "improvement_in_improvement_rate"})
        self.assertIn("does not establish recursive", report["claim_language"])
        self.assertTrue(report["unmeasured_dimensions"])

        # Even with replications and a meta-policy comparison attached, one generation
        # cannot reach a repetition claim: the ladder cannot be entered above its
        # lowest unmet rung.
        body_a, body_b = meta_body(label="A"), meta_body(label="B", trace_sampling_strategy="recent")
        comparison = compare_meta_policies(
            body_a, body_b,
            self.synthetic_arm(effective_promotion_alpha(body_a), 3, 10.0, ["f0", "f1", "f2"]),
            self.synthetic_arm(effective_promotion_alpha(body_b), 1, 10.0, ["f0", "f1", "f2"]),
            new_task_families=["f0", "f1", "f2"])
        stacked = rsi_evidence_report([record], meta_comparison=comparison,
                                      independent_replications=99)
        self.assertEqual(stacked["rung"], "single_bounded_improvement")
        self.assertFalse({e["rung"]: e for e in stacked["ladder"]}["repeated_improvement"]["met"])

    def test_the_ladder_floor_is_no_evidence_when_nothing_validated(self):
        record = self.generation(replay_fn=fresh_replay(gain=0.0))
        report = rsi_evidence_report([record])
        self.assertEqual(report["rung"], "no_evidence")
        self.assertEqual(report["validated_generation_count"], 0)
        self.assertEqual(len(report["refusals"]), 3)
        self.assertEqual(rsi_evidence_report([])["rung"], "no_evidence")

    def test_rsi_report_rejects_foreign_records_and_impossible_replication_counts(self):
        with self.assertRaises(Invalid):
            rsi_evidence_report([{"version": "something-else"}])
        with self.assertRaises(Invalid):
            rsi_evidence_report([], independent_replications=-1)
        with self.assertRaises(Invalid):
            rsi_evidence_report([], meta_comparison={"version": "not-this-one"})


class HouseRuleTests(unittest.TestCase):
    MODULES = ("__init__", "policy", "proposer", "protocol", "goodhart", "meta")

    def sources(self):
        root = Path(__file__).resolve().parents[1] / "symplex" / "improvement"
        return {name: (root / (name + ".py")).read_text() for name in self.MODULES}

    def test_no_module_can_call_a_model_open_a_socket_or_start_a_process(self):
        banned = ("import subprocess", "import socket", "import urllib", "import requests",
                  "import httpx", ".propose(", "provider", "openai", "os.system", "eval(", "exec(")
        for name, text in self.sources().items():
            for token in banned:
                self.assertNotIn(token, text, name + " must stay host-owned governance code: " + token)

    def test_every_public_function_result_carries_a_scope(self):
        parent, child = policy_pair()
        tasks = [task("t0", "prediction_markets", "development", values(0.4), values(0.9)),
                 task("t1", "biology", "holdout", values(0.4), values(0.9))]
        report = paired_replay(parent, child, tasks, CAPS, board=board())
        results = [
            baseline_policy(), apply_patch(parent, patch_body(parent["policy_id"],
                                                              {"stopping_rules.max_actions": 9})),
            to_prompt_overlay(child), describe_board(board()), report, promotion_rule(report),
            sequential_verdict([0.1, 0.2], sigma=0.1), power_report(4, noise_sd=0.2,
                                                                    minimum_detectable_effect=0.05),
            failure_taxonomy(planted_traces()), sample_traces(planted_traces(), "recent", 0, 3),
            propose_patch(failure_taxonomy(planted_traces()), parent),
            divergence_report(report), specification_gaming_checks(report),
            overfitting_to_task_family(report), goodhart_report(report, board=board()),
            effective_promotion_alpha(meta_body()), rsi_evidence_report([]),
        ]
        for result in results:
            self.assertIsInstance(result, dict)
            self.assertIsInstance(result.get("scope"), str)
            self.assertTrue(result["scope"].strip())

    def test_the_package_exposes_no_installation_or_activation_path(self):
        import symplex.improvement as improvement

        for name in improvement.__all__:
            lowered = name.lower()
            self.assertNotIn("install", lowered)
            self.assertNotIn("activate", lowered)
            self.assertNotIn("deploy", lowered)
        self.assertIn("method_registry", improvement.__doc__)


if __name__ == "__main__":
    unittest.main()
