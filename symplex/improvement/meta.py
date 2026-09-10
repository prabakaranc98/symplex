"""Loop D: revising the method-revision procedure, and refusing to overclaim the result.

Loop C asks whether a method patch improved the method. Loop D asks whether a
*different way of producing method patches* produces better methods, at equal total
cost, on task families neither procedure was tuned on. That is the project's actual
recursive research question, and it is the one most easily faked, so the reporting
side of it is written to refuse rather than to summarize.

``rsi_evidence_report`` encodes a ladder of claims — no_evidence,
single_bounded_improvement, repeated_improvement, improvement_in_improvement_rate —
and walks it in order, stopping at the first rung whose preconditions are unmet. A
rung cannot be skipped, and one generation cannot reach the top no matter what it
measured.

This module runs nothing. The host supplies a deterministic ``replay_fn`` that
returns already-recorded arm outcomes; no model is called, no process is started
and nothing is installed. Activation remains the operator review, scoped canary and
rollback path in ``symplex.infrastructure.method_registry``.
"""

import math
import random
from typing import Literal

from pydantic import Field, model_validator

from symplex.core.contracts import Invalid, digest
from symplex.improvement.goodhart import goodhart_report
from symplex.improvement.policy import MAX_GENERATION, _as_policy, apply_patch
from symplex.improvement.proposer import (
    ALLOWED_GRAMMAR,
    MAX_TRACES,
    SAMPLING_STRATEGIES,
    _as_traces,
    failure_taxonomy,
    propose_patch,
    sample_traces,
)
from symplex.improvement.protocol import (
    DEMOTION_ALPHA,
    PROMOTION_ALPHA,
    _as_board,
    paired_replay,
    power_report,
    promotion_rule,
    sequential_verdict,
)
from symplex.modeling.complex_system import ClosedContract, Identifier, Text

VERSION = "bounded-loop-d-v1"
SCOPE = (
    "One improvement generation, or a comparison of improvement procedures, recorded "
    "with its cost and its refusals. Improvement productivity counts validated "
    "improvements per unit compute on the supplied fresh task families under an "
    "explicitly incomplete evaluator. It is not a measure of intelligence, capability "
    "or general recursive self-improvement, and nothing here promotes or installs a method."
)

MAX_GENERATIONS = 32
MIN_REPEATED_GENERATIONS = 3
MIN_REPEATED_REPLICATIONS = 1
MIN_RSI_REPLICATIONS = 2
MIN_GENERATIONS_PER_ARM = 3
EQUAL_BUDGET_TOLERANCE = 0.05

# The protected floor. `promotion_strictness` is a multiplier that may only make
# promotion harder: the effective alpha is PROMOTION_ALPHA / strictness, so no
# MetaPolicy can loosen the bar it was given. A meta-policy that could relax its own
# promotion threshold would be grading its own homework.
PROMOTION_ALPHA_FLOOR = PROMOTION_ALPHA


class MetaPolicy(ClosedContract):
    """The parameters governing the improvement procedure itself."""

    meta_policy_id: str = Field(default="", max_length=64)
    label: Text
    trace_sampling_strategy: Literal[SAMPLING_STRATEGIES]
    trace_sample_size: int = Field(ge=1, le=MAX_TRACES)
    sampling_seed: int = Field(ge=0, lt=2**32)
    patch_grammar_subset: list[Identifier] = Field(min_length=1, max_length=len(ALLOWED_GRAMMAR))
    exploration_rate: float = Field(ge=0.0, le=1.0)
    promotion_strictness: float = Field(
        ge=1.0, le=20.0,
        description="Multiplier that only tightens promotion. Effective alpha is PROMOTION_ALPHA / strictness; the protected floor cannot be raised.",
    )
    generation_budget: int = Field(ge=1, le=MAX_GENERATIONS)

    @model_validator(mode="after")
    def _closed(self):
        unknown = set(self.patch_grammar_subset) - set(ALLOWED_GRAMMAR)
        if unknown:
            raise ValueError("Unknown grammar operations: " + ", ".join(sorted(unknown)))
        if len(set(self.patch_grammar_subset)) != len(self.patch_grammar_subset):
            raise ValueError("patch_grammar_subset must not repeat an operation")
        self.meta_policy_id = "meta_" + digest(
            {k: v for k, v in self.model_dump().items() if k != "meta_policy_id"})[:24]
        return self


def _as_meta(value):
    if isinstance(value, MetaPolicy):
        return value
    if not isinstance(value, dict):
        raise Invalid("Expected a MetaPolicy or its mapping")
    try:
        return MetaPolicy.model_validate(value)
    except Exception as exc:
        raise Invalid("Invalid meta policy: " + str(exc)[:600]) from None


def effective_promotion_alpha(meta_policy):
    """The promotion alpha this meta-policy actually gets. It can only be tighter than the floor."""
    meta = _as_meta(meta_policy)
    alpha = PROMOTION_ALPHA_FLOOR / meta.promotion_strictness
    return {
        "version": VERSION,
        "meta_policy_id": meta.meta_policy_id,
        "promotion_strictness": meta.promotion_strictness,
        "protected_floor": PROMOTION_ALPHA_FLOOR,
        "effective_promotion_alpha": min(alpha, PROMOTION_ALPHA_FLOOR),
        "demotion_alpha": DEMOTION_ALPHA,
        "scope": SCOPE + " A meta-policy chooses only how much stricter than the floor promotion is; "
                         "the floor itself is host-owned and outside the editable surface.",
    }


def _number(name, value, low=0.0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < low:
        raise Invalid(name + " must be a finite number of at least " + repr(low))
    return float(value)


def meta_generation(meta_policy, *, traces, parent_policy, board, replay_fn,
                    fresh_task_families, caps, compute_cost, generation_index=0, history=None):
    """Run one full Loop C cycle under a given MetaPolicy and record its improvement productivity.

    The host supplies ``replay_fn(parent, child, families) -> list[ReplayTask]``: the
    already-recorded paired arm outcomes for the fresh tasks. This function samples
    traces, derives a taxonomy, proposes a patch from the meta-policy's grammar
    subset, applies it, aggregates the paired replay, and records power, the always-
    valid sequential verdict, the promotion verdict and the Goodhart checks.

    ``validated_improvement`` requires all four to hold together: a promote verdict,
    sufficient sequential evidence at this meta-policy's effective alpha, adequate
    power, and no blocking gaming flag. Failing any one is recorded with its reason,
    never dropped.
    """
    meta = _as_meta(meta_policy)
    parent = _as_policy(parent_policy)
    board = _as_board(board)
    parsed_traces = _as_traces(traces)
    if not callable(replay_fn):
        raise Invalid("replay_fn must be a host-supplied callable returning recorded arm outcomes")
    if not isinstance(fresh_task_families, (list, tuple)) or not fresh_task_families:
        raise Invalid("A generation needs at least one fresh task family")
    if len(fresh_task_families) > 16 or any(not isinstance(f, str) or not f for f in fresh_task_families):
        raise Invalid("Fresh task families must be 1 to 16 nonempty names")
    if type(generation_index) is not int or not 0 <= generation_index <= MAX_GENERATION:
        raise Invalid("generation_index must be an integer from 0 to " + str(MAX_GENERATION))
    compute_cost = _number("compute_cost", compute_cost, low=0.0)
    if compute_cost <= 0:
        raise Invalid("Improvement productivity needs a positive recorded compute cost")

    seed = (meta.sampling_seed + generation_index) % (2**32)
    sample = sample_traces(parsed_traces, meta.trace_sampling_strategy, seed, meta.trace_sample_size)
    selected = [t for t in parsed_traces if t.trace_id in set(sample["selected_trace_ids"])]
    taxonomy = failure_taxonomy(selected)
    trained_families = sorted({t.task_family for t in selected})

    allowed = list(meta.patch_grammar_subset)
    exploration = {"applied": False, "reason": "exploration_rate did not fire for this generation"}
    if meta.exploration_rate > 0 and taxonomy["modes"]:
        rng = random.Random(("symplex-meta-exploration", meta.meta_policy_id, generation_index))
        if rng.random() < meta.exploration_rate and len(taxonomy["modes"]) > 1:
            from symplex.improvement.proposer import PATCH_GRAMMAR

            top = PATCH_GRAMMAR.get(taxonomy["modes"][0]["mode"], {}).get("operation")
            if top in allowed and len(allowed) > 1:
                allowed = [op for op in allowed if op != top]
                exploration = {"applied": True, "reason": "Explored below the dominant mode by withholding "
                                                          + top + " from this generation's grammar",
                               "withheld_operation": top}

    proposal = propose_patch(taxonomy, parent, allowed)
    alpha = effective_promotion_alpha(meta)["effective_promotion_alpha"]
    blockers = []
    record = {
        "version": VERSION, "generation_index": generation_index,
        "meta_policy_id": meta.meta_policy_id, "meta_policy": meta.model_dump(),
        "parent_policy_id": parent.policy_id, "child_policy_id": None,
        "board_id": board.board_id, "effective_promotion_alpha": alpha,
        "trace_sample": {k: sample[k] for k in ("strategy", "seed", "limit", "selected_trace_ids", "reference_mode")},
        "taxonomy": {"dominant_mode": taxonomy["dominant_mode"], "incident_count": taxonomy["incident_count"],
                     "modes": [{k: m[k] for k in ("mode", "incident_count", "trace_count", "total_cost")}
                               for m in taxonomy["modes"]],
                     "unclassified_failure_count": len(taxonomy["unclassified_failures"])},
        "exploration": exploration, "allowed_grammar": sorted(allowed),
        "proposal": {"selected_mode": proposal["selected_mode"], "patch": proposal["patch"],
                     "rejected_alternatives": proposal["rejected_alternatives"]},
        "trained_task_families": trained_families,
        "fresh_task_families": sorted(fresh_task_families),
        "fresh_families_verified": not (set(fresh_task_families) & set(trained_families)),
        "compute_cost": compute_cost, "caps": dict(caps) if isinstance(caps, dict) else caps,
        "patch_applied": False, "replay": None, "power": None, "sequential": None,
        "promotion_verdict": None, "goodhart": None,
        "validated_improvement": False, "validated_improvements": 0,
        "improvement_productivity": 0.0,
        "unmeasured_dimensions": list(board.unmeasured_dimensions),
        "model_calls": 0, "auto_promotion": False,
    }
    if not record["fresh_families_verified"]:
        blockers.append("fresh_task_families overlap the families the patch was learned from")
    if proposal["patch"] is None:
        blockers.append("no patch could be derived from the sampled traces under this grammar subset")
        record.update(outcome="no_patch_proposed", blockers=blockers,
                      scope=SCOPE + " No candidate was produced, so nothing was compared.")
        return record

    applied = apply_patch(parent, proposal["patch"])
    record["patch_result"] = {"applied": applied["applied"], "rejected": applied["rejected"],
                              "diff": (applied["diff"] or {}).get("changes") if applied["diff"] else None}
    if not applied["applied"]:
        blockers.append("the derived patch was refused: "
                        + "; ".join(sorted({r["reason"] for r in applied["rejected"]})))
        record.update(outcome="patch_rejected", blockers=blockers,
                      scope=SCOPE + " The patch left the allowlisted surface and was refused; the parent is retained.")
        return record
    child = _as_policy(applied["child"])
    record.update(patch_applied=True, child_policy_id=child.policy_id, child_policy=applied["child"])

    try:
        tasks = replay_fn(parent.model_dump(), applied["child"], sorted(fresh_task_families))
    except Invalid:
        raise
    except Exception as exc:
        raise Invalid("replay_fn did not return recorded arm outcomes: " + str(exc)[:600]) from None
    report = paired_replay(parent, child, tasks, caps, board=board)
    verdict = promotion_rule(report)
    checks = goodhart_report(report, board=board, history=history)
    primary = report["primary_proxy"]
    row = report["proxies"][primary]
    power = power_report(report["usable_task_count"], noise_sd=row["noise_sd"],
                         minimum_detectable_effect=row["minimum_detectable_effect"], alpha=alpha)
    gains = row["paired_gains"] or [0.0]
    sequential = sequential_verdict(gains, sigma=row["noise_sd"], promotion_alpha=alpha,
                                    demotion_alpha=DEMOTION_ALPHA)

    if verdict["verdict"] != "promote":
        blockers.append("promotion rule returned " + verdict["verdict"] + ": "
                        + ", ".join(verdict["unmet_conditions"]))
    if not sequential["promotion_evidence_sufficient"]:
        blockers.append("always-valid evidence insufficient at alpha=" + repr(alpha)
                        + " (p=" + repr(round(sequential["always_valid_p_value"], 6)) + ")")
    if power["underpowered"]:
        blockers.append("comparison is underpowered: " + power["reason"])
    if checks["blocks_promotion_review"]:
        blockers.append("blocking gaming signal at severity " + str(checks["max_severity"]))
    validated = not blockers
    record.update(
        replay={k: report[k] for k in ("task_count", "usable_task_count", "task_families", "stages",
                                       "equal_caps_verified", "blinding_verified", "settings_matched",
                                       "accounting_complete", "primary_proxy", "protected_proxies",
                                       "diagnostic_proxies", "proxies", "by_task_family", "by_difficulty",
                                       "arm_structure", "holdout_task_count", "excluded_tasks")},
        replay_report=report, power=power, sequential=sequential, promotion_verdict=verdict, goodhart=checks,
        validated_improvement=validated, validated_improvements=int(validated),
        improvement_productivity=(1.0 / compute_cost) if validated else 0.0,
        blockers=blockers,
        outcome="validated_bounded_improvement" if validated else "not_validated",
        scope=SCOPE + (" One validated bounded improvement on fresh tasks; this is a single generation and "
                       "supports no claim beyond that." if validated else
                       " This generation did not produce a validated improvement; the parent is retained "
                       "and the reasons are recorded."),
    )
    return record


def _generations(records, label):
    if not isinstance(records, (list, tuple)):
        raise Invalid(label + " must be a list of meta_generation records")
    if len(records) > MAX_GENERATIONS:
        raise Invalid(label + " may hold at most " + str(MAX_GENERATIONS) + " generations")
    for item in records:
        if not isinstance(item, dict) or item.get("version") != VERSION:
            raise Invalid(label + " must contain meta_generation records from this version")
    return list(records)


def _arm(records, meta_policy):
    meta = _as_meta(meta_policy)
    mismatched = [r["generation_index"] for r in records if r["meta_policy_id"] != meta.meta_policy_id]
    cost = sum(_number("compute_cost", r["compute_cost"]) for r in records)
    validated = sum(int(bool(r.get("validated_improvement"))) for r in records)
    families = sorted({f for r in records for f in r.get("fresh_task_families", [])})
    return {
        "meta_policy_id": meta.meta_policy_id,
        "label": meta.label,
        "generation_count": len(records),
        "total_compute_cost": cost,
        "validated_improvements": validated,
        "improvement_productivity": (validated / cost) if cost > 0 else None,
        "per_generation_productivity": [r.get("improvement_productivity", 0.0) for r in records],
        "fresh_task_families": families,
        "all_generations_used_fresh_families": all(r.get("fresh_families_verified") for r in records),
        "mismatched_generation_indices": mismatched,
        "blockers": sorted({b for r in records for b in (r.get("blockers") or [])}),
    }


def compare_meta_policies(meta_a, meta_b, generations_a, generations_b, *, new_task_families=None):
    """The recursive experiment: do policies produced under A beat those under B, at equal budget, on new families?

    This compares improvement *procedures*, not methods. It answers ``inconclusive``
    unless both arms ran enough generations, spent comparably, and evaluated on task
    families neither procedure learned from. A productivity difference under unequal
    budgets or shared families is not evidence about the procedures.
    """
    records_a = _generations(generations_a, "generations_a")
    records_b = _generations(generations_b, "generations_b")
    arm_a, arm_b = _arm(records_a, meta_a), _arm(records_b, meta_b)
    if arm_a["meta_policy_id"] == arm_b["meta_policy_id"]:
        raise Invalid("compare_meta_policies needs two distinct meta policies")
    conditions = []

    def add(name, met, reason):
        conditions.append({"name": name, "met": bool(met), "reason": reason})

    enough = (arm_a["generation_count"] >= MIN_GENERATIONS_PER_ARM
              and arm_b["generation_count"] >= MIN_GENERATIONS_PER_ARM)
    add("sufficient_generations_per_arm", enough,
        "Each arm needs at least " + str(MIN_GENERATIONS_PER_ARM) + " generations; observed "
        + str(arm_a["generation_count"]) + " and " + str(arm_b["generation_count"]))
    add("generations_match_their_meta_policy",
        not arm_a["mismatched_generation_indices"] and not arm_b["mismatched_generation_indices"],
        "Every generation record must name the meta policy it is attributed to")
    cost_a, cost_b = arm_a["total_compute_cost"], arm_b["total_compute_cost"]
    equal_budget = bool(cost_a > 0 and cost_b > 0
                        and abs(cost_a - cost_b) <= EQUAL_BUDGET_TOLERANCE * max(cost_a, cost_b))
    add("equal_total_budget", equal_budget,
        "Total compute " + repr(round(cost_a, 6)) + " versus " + repr(round(cost_b, 6))
        + " within " + repr(EQUAL_BUDGET_TOLERANCE) + " relative tolerance")
    add("fresh_task_families_in_every_generation",
        arm_a["all_generations_used_fresh_families"] and arm_b["all_generations_used_fresh_families"],
        "Every generation must be evaluated on families it did not learn from")
    if new_task_families is not None:
        if not isinstance(new_task_families, (list, tuple)):
            raise Invalid("new_task_families must be a list of family names")
        declared = set(new_task_families)
        covered = declared <= set(arm_a["fresh_task_families"]) and declared <= set(arm_b["fresh_task_families"])
        add("declared_new_families_covered_by_both_arms", covered,
            "Both arms must be evaluated on the declared new families: " + ", ".join(sorted(declared)))
    else:
        add("declared_new_families_covered_by_both_arms", False,
            "No explicit new-task-family set was declared for this comparison")

    productivity_a, productivity_b = arm_a["improvement_productivity"], arm_b["improvement_productivity"]
    delta = (productivity_a - productivity_b) if None not in (productivity_a, productivity_b) else None
    unmet = [c["name"] for c in conditions if not c["met"]]
    if unmet:
        verdict = "inconclusive"
    elif delta is None or abs(delta) <= 1e-12:
        verdict = "no_difference_established"
    else:
        verdict = "a_more_productive" if delta > 0 else "b_more_productive"
    return {
        "version": VERSION,
        "arm_a": arm_a,
        "arm_b": arm_b,
        "productivity_a": productivity_a,
        "productivity_b": productivity_b,
        "productivity_delta": delta,
        "conditions": conditions,
        "unmet_conditions": unmet,
        "verdict": verdict,
        "strict_productivity_gain": verdict in ("a_more_productive", "b_more_productive"),
        "equal_budget_verified": equal_budget,
        "new_task_families": sorted(new_task_families) if new_task_families else [],
        "scope": SCOPE + " Improvement productivity is validated improvements per unit compute under one "
                         "incomplete evaluator on the families tested. A difference here is evidence about "
                         "these two procedures on these families, not about improvement in general.",
    }


CLAIM_LADDER = (
    {
        "rung": "no_evidence",
        "preconditions": (),
        "language": "The recorded evidence supports no claim of method improvement.",
    },
    {
        "rung": "single_bounded_improvement",
        "preconditions": (
            "at_least_one_validated_generation",
            "accounting_complete_in_a_validated_generation",
            "held_out_stage_passed_in_a_validated_generation",
            "no_protected_regression_in_a_validated_generation",
            "adequately_powered_validated_generation",
            "no_blocking_gaming_signal_in_a_validated_generation",
        ),
        "language": ("One bounded method improvement was validated on fresh tasks under a protected, "
                     "explicitly incomplete evaluator. This supports bounded method improvement only. It "
                     "does not establish repeated improvement, and it does not establish recursive "
                     "self-improvement."),
    },
    {
        "rung": "repeated_improvement",
        "preconditions": (
            "at_least_" + str(MIN_REPEATED_GENERATIONS) + "_validated_generations",
            "validated_generations_span_disjoint_fresh_task_families",
            "at_least_" + str(MIN_REPEATED_REPLICATIONS) + "_independent_replication",
        ),
        "language": ("Method improvements were validated repeatedly across disjoint fresh task families "
                     "with independent replication. The rate at which improvements are found has not been "
                     "shown to change."),
    },
    {
        "rung": "improvement_in_improvement_rate",
        "preconditions": (
            "meta_policy_comparison_recorded",
            "strict_productivity_gain_at_equal_budget",
            "comparison_ran_on_declared_new_task_families",
            "at_least_" + str(MIN_RSI_REPLICATIONS) + "_independent_replications",
            "no_flagged_proxy_correlation_drift",
        ),
        "language": ("Improvement productivity itself rose under a revised improvement procedure at equal "
                     "total budget on new task families, with independent replication. Even this is bounded "
                     "to the declared proxies and the families tested; the board's unmeasured dimensions "
                     "remain unmeasured, and this is not a claim of general recursive self-improvement."),
    },
)


def _precondition_facts(records, comparison, replications, drift):
    validated = [r for r in records if r.get("validated_improvement")]
    families = [frozenset(r.get("fresh_task_families") or []) for r in validated]
    disjoint = len(validated) >= MIN_REPEATED_GENERATIONS and all(
        not (families[i] & families[j]) for i in range(len(families)) for j in range(i + 1, len(families)))

    def any_validated(predicate):
        return any(predicate(r) for r in validated)

    facts = {
        "at_least_one_validated_generation": (
            len(validated) >= 1,
            str(len(validated)) + " validated generation(s) of " + str(len(records)) + " recorded"),
        "accounting_complete_in_a_validated_generation": (
            any_validated(lambda r: (r.get("replay") or {}).get("accounting_complete") is True),
            "Equal caps, matched settings, blinded arms and complete resource accounting"),
        "held_out_stage_passed_in_a_validated_generation": (
            any_validated(lambda r: "holdout_stage_passed" not in
                          ((r.get("promotion_verdict") or {}).get("unmet_conditions") or [])
                          and (r.get("replay") or {}).get("holdout_task_count", 0) > 0),
            "A held-out stage was replayed and passed"),
        "no_protected_regression_in_a_validated_generation": (
            any_validated(lambda r: not ((r.get("goodhart") or {}).get("divergence") or {})
                          .get("protected_regressions")),
            "No protected proxy regressed"),
        "adequately_powered_validated_generation": (
            any_validated(lambda r: (r.get("power") or {}).get("underpowered") is False),
            "The comparison could resolve its declared minimum detectable effect"),
        "no_blocking_gaming_signal_in_a_validated_generation": (
            any_validated(lambda r: (r.get("goodhart") or {}).get("blocks_promotion_review") is False),
            "No high or critical structural gaming signal"),
        "at_least_" + str(MIN_REPEATED_GENERATIONS) + "_validated_generations": (
            len(validated) >= MIN_REPEATED_GENERATIONS,
            str(len(validated)) + " validated generation(s); " + str(MIN_REPEATED_GENERATIONS) + " required"),
        "validated_generations_span_disjoint_fresh_task_families": (
            disjoint, "Validated generations must not reuse each other's fresh task families"),
        "at_least_" + str(MIN_REPEATED_REPLICATIONS) + "_independent_replication": (
            replications >= MIN_REPEATED_REPLICATIONS,
            str(replications) + " independent replication(s) recorded; "
            + str(MIN_REPEATED_REPLICATIONS) + " required"),
        "meta_policy_comparison_recorded": (
            comparison is not None, "A compare_meta_policies result is required for any rate claim"),
        "strict_productivity_gain_at_equal_budget": (
            bool(comparison and comparison.get("strict_productivity_gain")
                 and comparison.get("equal_budget_verified")),
            "A strict improvement-productivity difference at equal total budget"),
        "comparison_ran_on_declared_new_task_families": (
            bool(comparison and comparison.get("new_task_families")
                 and "declared_new_families_covered_by_both_arms" not in (comparison.get("unmet_conditions") or [])),
            "Both arms evaluated on an explicitly declared set of new task families"),
        "at_least_" + str(MIN_RSI_REPLICATIONS) + "_independent_replications": (
            replications >= MIN_RSI_REPLICATIONS,
            str(replications) + " independent replication(s) recorded; "
            + str(MIN_RSI_REPLICATIONS) + " required"),
        "no_flagged_proxy_correlation_drift": (
            drift is not None and not drift.get("flagged") and bool(drift.get("sufficient_history")),
            "A proxy-correlation history long enough to check, with no flagged divergence"),
    }
    return facts, validated


def rsi_evidence_report(generations, *, meta_comparison=None, independent_replications=0,
                        correlation_drift=None):
    """State plainly what the evidence supports, and refuse the rungs it does not reach.

    The ladder is walked in order and stops at the first rung with an unmet
    precondition, so a stronger claim cannot be reached by skipping a weaker one. One
    generation can reach ``single_bounded_improvement`` at most, no matter how large
    its effect: repetition needs repetition, and a claim about the *rate* of
    improvement needs a recorded comparison of improvement procedures plus
    independent replication.
    """
    records = _generations(generations, "generations")
    if type(independent_replications) is not int or not 0 <= independent_replications <= 64:
        raise Invalid("independent_replications must be an integer from 0 to 64")
    if meta_comparison is not None and (not isinstance(meta_comparison, dict)
                                        or meta_comparison.get("version") != VERSION):
        raise Invalid("meta_comparison must be a compare_meta_policies result from this version")
    if correlation_drift is not None and not isinstance(correlation_drift, dict):
        raise Invalid("correlation_drift must be a proxy_correlation_drift result")
    facts, validated = _precondition_facts(records, meta_comparison, independent_replications, correlation_drift)
    ladder, awarded, awarded_index, blocked = [], "no_evidence", 0, False
    for index, rung in enumerate(CLAIM_LADDER):
        preconditions = [{"id": name, "met": bool(facts[name][0]), "reason": facts[name][1]}
                         for name in rung["preconditions"]]
        unmet = [p["id"] for p in preconditions if not p["met"]]
        reachable = not blocked and not unmet
        ladder.append({
            "rung": rung["rung"], "index": index, "preconditions": preconditions,
            "unmet_preconditions": unmet, "met": bool(reachable),
            "blocked_by_lower_rung": bool(blocked),
            "language": rung["language"],
        })
        if reachable:
            awarded, awarded_index = rung["rung"], index
        else:
            blocked = True
    refusals = []
    for entry in ladder[awarded_index + 1:]:
        refusals.append({
            "refused_rung": entry["rung"],
            "unmet_preconditions": entry["unmet_preconditions"],
            "blocked_by_lower_rung": entry["blocked_by_lower_rung"],
            "statement": "This evidence does not support '" + entry["rung"] + "': "
                         + (", ".join(entry["unmet_preconditions"]) if entry["unmet_preconditions"]
                            else "a lower rung on the ladder is unmet"),
        })
    return {
        "version": VERSION,
        "rung": awarded,
        "rung_index": awarded_index,
        "ladder": ladder,
        "claim_language": CLAIM_LADDER[awarded_index]["language"],
        "refusals": refusals,
        "generation_count": len(records),
        "validated_generation_count": len(validated),
        "independent_replications": independent_replications,
        "meta_comparison_recorded": meta_comparison is not None,
        "unmeasured_dimensions": sorted({d for r in records for d in (r.get("unmeasured_dimensions") or [])}),
        "recursive_self_improvement_claimed": False,
        "scope": SCOPE + " The rung is the strongest claim the recorded preconditions permit. It is not a "
                         "summary of how well the system performed, and the ladder cannot be entered above "
                         "its lowest unmet rung.",
    }
