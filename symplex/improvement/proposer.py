"""Evidence-driven method proposals derived from recorded failures, without a model call.

An LLM may later *suggest* a method patch. This module exists so the host can
always derive and validate one without it: the taxonomy is computed structurally
from trace records, and the patch is selected from a typed grammar. Nothing here
calls a model, opens a socket or starts a process.

The trace-sampling strategy is deliberately a first-class, swappable, recorded
parameter, because *which failures the improver learns from* is precisely the
Loop D target in ``symplex.improvement.meta``.
"""

import random
from collections import defaultdict
from typing import Literal

from pydantic import Field

from symplex.core.contracts import Invalid
from symplex.improvement.policy import (
    EVALUATION_TOOLS,
    MAX_PATCH_CHANGES,
    MethodPatch,
    UncertaintyType,
    _as_policy,
)
from symplex.modeling.complex_system import ClosedContract, Identifier, Text

VERSION = "structural-failure-taxonomy-v1"
SCOPE = (
    "Recurring failure modes counted structurally from recorded traces, and one "
    "typed patch selected from a fixed grammar. Counts and costs describe the supplied "
    "traces only. A mode is a host-declared reading of a recorded pattern, not a "
    "diagnosis of cause, and a proposed patch is a hypothesis to be tested under "
    "protected evaluation, never an improvement."
)

MAX_TRACES = 400
MAX_STEPS = 200
MAX_EXEMPLARS = 5

FAILURE_MODES = (
    "wasted_evaluation_on_unverified_join",
    "unresolving_tool_repetition",
    "protocol_rejection",
    "non_identifiable_comparison_run",
    "context_omitted_existing_artifact",
    "budget_exhausted_before_resolution",
)

COMPARISON_TOOLS = ("compare_computation", "run_system_scenarios", "evolve_model")

SAMPLING_STRATEGIES = (
    "recent",
    "most_costly",
    "most_frequent",
    "stratified_by_type",
    "worst_regret",
)


class TraceStep(ClosedContract):
    """One recorded investigation action and the host-observable facts about it."""

    index: int = Field(ge=0, le=MAX_STEPS)
    tool: Identifier
    uncertainty_type: UncertaintyType | None = None
    status: Literal["completed", "rejected", "failed", "skipped"]
    rejection_reason: Literal["protocol_contract", "permission", "budget", "schema", "other"] | None = None
    resolved_uncertainty: bool = False
    identifiable: bool | None = None
    join_keys_verified: bool | None = None
    available_artifact_ids: list[str] = Field(default_factory=list, max_length=32)
    referenced_artifact_ids: list[str] = Field(default_factory=list, max_length=32)
    cost: float = Field(default=0.0, ge=0.0)
    regret: float = Field(
        default=0.0,
        ge=0.0,
        description="Host-recorded cost of the resolution this step delayed or missed. Declared bookkeeping, not a measured counterfactual.",
    )


class InvestigationTrace(ClosedContract):
    trace_id: str = Field(min_length=1, max_length=64)
    task_family: Identifier
    policy_id: str = Field(default="", max_length=64)
    sequence: int = Field(ge=0, le=1_000_000)
    outcome: Literal["resolved", "unresolved", "abandoned", "budget_exhausted"]
    steps: list[TraceStep] = Field(min_length=1, max_length=MAX_STEPS)


def _as_traces(traces):
    if not isinstance(traces, (list, tuple)) or not traces:
        raise Invalid("Failure analysis needs a nonempty list of traces")
    if len(traces) > MAX_TRACES:
        raise Invalid("At most " + str(MAX_TRACES) + " traces may be analysed at once")
    parsed = []
    for item in traces:
        if isinstance(item, InvestigationTrace):
            parsed.append(item)
            continue
        try:
            parsed.append(InvestigationTrace.model_validate(item))
        except Exception as exc:
            raise Invalid("Invalid investigation trace: " + str(exc)[:600]) from None
    ids = [t.trace_id for t in parsed]
    if len(set(ids)) != len(ids):
        raise Invalid("Trace identifiers must be unique")
    return parsed


def _incident(trace, step, mode, detail, *, cost=None, regret=None):
    return {
        "mode": mode,
        "trace_id": trace.trace_id,
        "task_family": trace.task_family,
        "policy_id": trace.policy_id,
        "sequence": trace.sequence,
        "step_index": step.index,
        "tool": step.tool,
        "uncertainty_type": step.uncertainty_type,
        "cost": float(step.cost if cost is None else cost),
        "regret": float(step.regret if regret is None else regret),
        "detail": detail,
    }


def _trace_incidents(trace):
    found, classified = [], set()
    repeats = defaultdict(list)
    for step in trace.steps:
        if step.tool in EVALUATION_TOOLS and step.join_keys_verified is False:
            found.append(_incident(trace, step, "wasted_evaluation_on_unverified_join",
                                   "An evaluation ran on a join whose entity or timestamp keys were not verified"))
            classified.add(step.index)
        if step.tool in COMPARISON_TOOLS and step.identifiable is False and step.status in ("completed", "failed"):
            found.append(_incident(trace, step, "non_identifiable_comparison_run",
                                   "A comparison declared non-identifiable was executed anyway"))
            classified.add(step.index)
        if step.status == "rejected" and step.rejection_reason == "protocol_contract":
            found.append(_incident(trace, step, "protocol_rejection",
                                   "The frozen protocol contract rejected this action"))
            classified.add(step.index)
        omitted = sorted(set(step.available_artifact_ids) - set(step.referenced_artifact_ids))
        if omitted and not step.resolved_uncertainty:
            found.append(_incident(trace, step, "context_omitted_existing_artifact",
                                   "Context omitted " + str(len(omitted)) + " artifact(s) that already existed: "
                                   + ", ".join(omitted[:4])))
            classified.add(step.index)
        if step.uncertainty_type is not None:
            repeats[(step.tool, step.uncertainty_type)].append(step)
    for (tool, kind), steps in sorted(repeats.items()):
        if len(steps) < 2 or any(s.resolved_uncertainty for s in steps):
            continue
        last = steps[-1]
        found.append(_incident(
            trace, last, "unresolving_tool_repetition",
            tool + " was selected " + str(len(steps)) + " times against " + kind
            + " uncertainty and never resolved it",
            cost=sum(s.cost for s in steps), regret=sum(s.regret for s in steps)))
        classified.update(s.index for s in steps)
    if trace.outcome == "budget_exhausted":
        last = trace.steps[-1]
        found.append(_incident(trace, last, "budget_exhausted_before_resolution",
                               "The trace consumed its allocation without resolving the question",
                               cost=sum(s.cost for s in trace.steps),
                               regret=sum(s.regret for s in trace.steps)))
        classified.add(last.index)
    unclassified = [
        {"trace_id": trace.trace_id, "step_index": s.index, "tool": s.tool, "status": s.status,
         "rejection_reason": s.rejection_reason, "cost": float(s.cost)}
        for s in trace.steps
        if s.status in ("rejected", "failed") and s.index not in classified
    ]
    return found, unclassified


def failure_taxonomy(traces):
    """Cluster recorded investigation failures into recurring, structurally defined modes.

    A single step can match more than one mode; the modes are different diagnoses,
    not a partition, and cost is therefore counted once per mode rather than once
    overall. Failures matching no declared mode are preserved under
    ``unclassified_failures`` instead of being dropped.
    """
    parsed = _as_traces(traces)
    incidents, unclassified = [], []
    for trace in parsed:
        found, leftover = _trace_incidents(trace)
        incidents.extend(found)
        unclassified.extend(leftover)
    grouped = defaultdict(list)
    for item in incidents:
        grouped[item["mode"]].append(item)
    modes = []
    for mode in FAILURE_MODES:
        items = grouped.get(mode)
        if not items:
            continue
        tools = sorted({i["tool"] for i in items})
        modes.append({
            "mode": mode,
            "incident_count": len(items),
            "trace_count": len({i["trace_id"] for i in items}),
            "total_cost": round(sum(i["cost"] for i in items), 12),
            "total_regret": round(sum(i["regret"] for i in items), 12),
            "task_families": sorted({i["task_family"] for i in items}),
            "tools": tools,
            "dominant_tool": max(tools, key=lambda t: (sum(i["cost"] for i in items if i["tool"] == t),
                                                       sum(1 for i in items if i["tool"] == t), t)),
            "uncertainty_types": sorted({i["uncertainty_type"] for i in items if i["uncertainty_type"]}),
            "exemplars": sorted(items, key=lambda i: (-i["cost"], i["trace_id"], i["step_index"]))[:MAX_EXEMPLARS],
        })
    # Wasted cost is what Loop C is trying to reduce, so it ranks first; count and
    # name break ties so the ordering is reproducible.
    modes.sort(key=lambda m: (-m["total_cost"], -m["incident_count"], m["mode"]))
    return {
        "version": VERSION,
        "trace_count": len(parsed),
        "incident_count": len(incidents),
        "modes": modes,
        "dominant_mode": modes[0]["mode"] if modes else None,
        "incidents": incidents,
        "unclassified_failures": unclassified,
        "task_families": sorted({t.task_family for t in parsed}),
        "model_calls": 0,
        "scope": SCOPE,
    }


def _tie_keys(trace_ids, seed):
    rng = random.Random(("symplex-trace-sample", int(seed)))
    order = sorted(trace_ids)
    return {ident: rng.random() for ident in order}


def sample_traces(traces, strategy="recent", seed=0, limit=8):
    """Choose which recorded failures the improver is allowed to learn from.

    This selection is itself a method parameter, not a convenience: two improvers
    reading the same corpus through different strategies propose different patches.
    Every strategy is fully deterministic given ``seed``.
    """
    parsed = _as_traces(traces)
    if strategy not in SAMPLING_STRATEGIES:
        raise Invalid("Unknown trace sampling strategy: " + str(strategy))
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise Invalid("Trace sampling needs an integer seed in [0, 2**32)")
    if type(limit) is not int or not 1 <= limit <= MAX_TRACES:
        raise Invalid("Trace sample size must be an integer from 1 to " + str(MAX_TRACES))
    by_id = {t.trace_id: t for t in parsed}
    taxonomy = failure_taxonomy(parsed)
    jitter = _tie_keys(by_id, seed)
    per_trace = defaultdict(list)
    for item in taxonomy["incidents"]:
        per_trace[item["trace_id"]].append(item)
    top_mode = taxonomy["modes"][0]["mode"] if taxonomy["modes"] else None

    def cost(ident):
        return sum(s.cost for s in by_id[ident].steps)

    def regret(ident):
        return sum(s.regret for s in by_id[ident].steps)

    ranked = []
    if strategy == "stratified_by_type":
        modes = [m["mode"] for m in taxonomy["modes"]]
        if modes:
            offset = random.Random(("symplex-stratified", int(seed))).randrange(len(modes))
            modes = modes[offset:] + modes[:offset]
        buckets = {
            mode: sorted({i["trace_id"] for i in taxonomy["incidents"] if i["mode"] == mode},
                         key=lambda ident: (-cost(ident), jitter[ident], ident))
            for mode in modes
        }
        taken, position = [], 0
        while len(taken) < len(by_id) and any(len(v) > position for v in buckets.values()):
            for mode in modes:
                if len(buckets[mode]) > position and buckets[mode][position] not in taken:
                    taken.append(buckets[mode][position])
            position += 1
        rest = sorted(set(by_id) - set(taken), key=lambda ident: (jitter[ident], ident))
        ranked = [{"trace_id": ident, "key": {"stratified_position": i}} for i, ident in enumerate(taken + rest)]
    else:
        keys = {
            "recent": lambda ident: (-by_id[ident].sequence, jitter[ident], ident),
            "most_costly": lambda ident: (-cost(ident), jitter[ident], ident),
            "most_frequent": lambda ident: (
                -sum(1 for i in per_trace[ident] if i["mode"] == top_mode),
                -len(per_trace[ident]), jitter[ident], ident),
            "worst_regret": lambda ident: (-regret(ident), jitter[ident], ident),
        }[strategy]
        for ident in sorted(by_id, key=keys):
            ranked.append({"trace_id": ident, "key": {
                "sequence": by_id[ident].sequence, "cost": round(cost(ident), 12),
                "regret": round(regret(ident), 12), "incident_count": len(per_trace[ident]),
                "dominant_mode_incidents": sum(1 for i in per_trace[ident] if i["mode"] == top_mode)}})
    selected = [r["trace_id"] for r in ranked[:limit]]
    return {
        "version": VERSION,
        "strategy": strategy,
        "seed": seed,
        "limit": limit,
        "selected_trace_ids": selected,
        "ordering": ranked,
        "excluded_trace_ids": [r["trace_id"] for r in ranked[limit:]],
        "reference_mode": top_mode,
        "model_calls": 0,
        "scope": SCOPE + " Sampling decides what evidence the proposer sees; a strategy "
                         "that never surfaces a failure mode makes that mode invisible, not absent.",
    }


def _route(kind, tool, before, rationale):
    return {"uncertainty_type": kind, "check_tool": tool, "before_tools": list(before), "rationale": rationale}


def _join_check(policy, mode):
    routes = [r.model_dump() for r in policy.diagnostic_routes]
    addition = _route("semantic", "profile_table", EVALUATION_TOOLS,
                      "Verify entity and timestamp join keys before spending an evaluation on them.")
    changes = {}
    if policy.evidence_admission.require_join_key_verification is not True:
        changes["evidence_admission.require_join_key_verification"] = True
    if addition not in routes:
        changes["diagnostic_routes"] = routes + [addition]
    return changes


def _demote_tool(policy, mode):
    tool = mode["dominant_tool"]
    changes = {}
    current = policy.action_priors.get(tool)
    target = round(max(0.0, (current if current is not None else 0.5) - 0.25), 6)
    if current != target:
        changes["action_priors." + tool] = target
    limit = max(1, policy.stopping_rules.max_consecutive_unresolving_actions - 1)
    if limit != policy.stopping_rules.max_consecutive_unresolving_actions:
        changes["stopping_rules.max_consecutive_unresolving_actions"] = limit
    return changes


def _tighten_admission(policy, mode):
    changes = {}
    target = min(8, policy.evidence_admission.min_independent_sources + 1)
    if target != policy.evidence_admission.min_independent_sources:
        changes["evidence_admission.min_independent_sources"] = target
    if policy.evidence_admission.require_provenance is not True:
        changes["evidence_admission.require_provenance"] = True
    return changes


def _identifiability_precheck(policy, mode):
    routes = [r.model_dump() for r in policy.diagnostic_routes]
    addition = _route("structural", "plan_experiment", COMPARISON_TOOLS,
                      "Establish that rival mechanisms are discriminable before running the comparison.")
    changes = {}
    if policy.evidence_admission.require_identifiability_check is not True:
        changes["evidence_admission.require_identifiability_check"] = True
    if policy.stopping_rules.stop_on_identifiability_failure is not True:
        changes["stopping_rules.stop_on_identifiability_failure"] = True
    if addition not in routes:
        changes["diagnostic_routes"] = routes + [addition]
    return changes


def _inventory_first(policy, mode):
    order = ["search_artifacts", "inspect_artifact"] + [t for t in policy.tool_order
                                                        if t not in ("search_artifacts", "inspect_artifact")]
    changes = {}
    if order != policy.tool_order:
        changes["tool_order"] = order
    for tool in ("search_artifacts", "inspect_artifact"):
        if policy.action_priors.get(tool) != 0.9:
            changes["action_priors." + tool] = 0.9
    return changes


def _tighten_stopping(policy, mode):
    changes = {}
    actions = max(1, policy.stopping_rules.max_actions - 2)
    if actions != policy.stopping_rules.max_actions:
        changes["stopping_rules.max_actions"] = actions
    floor = round(min(1.0, policy.stopping_rules.min_expected_uncertainty_reduction + 0.05), 6)
    if floor != policy.stopping_rules.min_expected_uncertainty_reduction:
        changes["stopping_rules.min_expected_uncertainty_reduction"] = floor
    return changes


PATCH_GRAMMAR = {
    "wasted_evaluation_on_unverified_join": {
        "operation": "insert_semantic_join_check_before_evaluation",
        "build": _join_check,
        "expected_effect": "Fewer evaluations spent on joins whose entity or timestamp keys were never checked.",
        "disconfirming_result": "Evaluation count falls but resolved-uncertainty count falls with it, or the check itself becomes the dominant cost.",
        "regression_checks": [
            "Paired contract-pass count must not regress on any replay task.",
            "Total actions per task must not exceed the parent's cap.",
            "Semantic checks must not replace the evidence they were meant to protect.",
        ],
        "promotion_criterion": "Strict gain on the primary proxy with no protected-proxy regression on fresh tasks at equal caps, held-out stage passed.",
    },
    "unresolving_tool_repetition": {
        "operation": "demote_unresolving_tool",
        "build": _demote_tool,
        "expected_effect": "A tool that repeatedly fails to resolve its uncertainty type is selected less often.",
        "disconfirming_result": "The demoted tool was the only route to that uncertainty type and unresolved items rise.",
        "regression_checks": [
            "Resolution rate per uncertainty type must not regress.",
            "No uncertainty type may lose its last available resolving tool.",
        ],
        "promotion_criterion": "Strict gain on the primary proxy with no protected-proxy regression on fresh tasks at equal caps, held-out stage passed.",
    },
    "protocol_rejection": {
        "operation": "tighten_evidence_admission",
        "build": _tighten_admission,
        "expected_effect": "Fewer actions rejected by the frozen protocol contract.",
        "disconfirming_result": "Rejections fall only because fewer actions are attempted at all.",
        "regression_checks": [
            "Attempted-action count must not fall while rejections fall.",
            "Admitted-evidence count must not fall below the parent's on matched tasks.",
        ],
        "promotion_criterion": "Strict gain on the primary proxy with no protected-proxy regression on fresh tasks at equal caps, held-out stage passed.",
    },
    "non_identifiable_comparison_run": {
        "operation": "require_identifiability_precheck",
        "build": _identifiability_precheck,
        "expected_effect": "Comparisons that cannot discriminate the rival mechanisms are not executed.",
        "disconfirming_result": "Identifiable comparisons are also refused, and discrimination opportunities are lost.",
        "regression_checks": [
            "Count of executed identifiable comparisons must not regress.",
            "Refusal reasons must be recorded for every skipped comparison.",
        ],
        "promotion_criterion": "Strict gain on the primary proxy with no protected-proxy regression on fresh tasks at equal caps, held-out stage passed.",
    },
    "context_omitted_existing_artifact": {
        "operation": "prioritize_existing_artifact_inventory",
        "build": _inventory_first,
        "expected_effect": "Work that already exists in storage is inventoried before new work is requested.",
        "disconfirming_result": "Inventory calls consume the action budget without changing what is referenced.",
        "regression_checks": [
            "Referenced-artifact coverage must rise, not merely inventory calls.",
            "Action count per task must stay within the parent's cap.",
        ],
        "promotion_criterion": "Strict gain on the primary proxy with no protected-proxy regression on fresh tasks at equal caps, held-out stage passed.",
    },
    "budget_exhausted_before_resolution": {
        "operation": "tighten_stopping_rule",
        "build": _tighten_stopping,
        "expected_effect": "Investigations stop before exhausting an allocation they were not going to resolve.",
        "disconfirming_result": "Resolved investigations are cut short and the resolution rate falls.",
        "regression_checks": [
            "Resolved-task count must not regress.",
            "Stopping must be recorded with its triggering rule.",
        ],
        "promotion_criterion": "Strict gain on the primary proxy with no protected-proxy regression on fresh tasks at equal caps, held-out stage passed.",
    },
}

ALLOWED_GRAMMAR = tuple(sorted(entry["operation"] for entry in PATCH_GRAMMAR.values()))


def propose_patch(taxonomy, parent_policy, allowed_grammar=ALLOWED_GRAMMAR):
    """Select one typed patch from the grammar, deterministically, with no model call.

    Modes are considered in dominance order. Modes that the grammar cannot address,
    that the caller's grammar subset excludes, or that the parent policy already
    satisfies are recorded as rejected alternatives rather than discarded.
    """
    policy = _as_policy(parent_policy)
    if not isinstance(taxonomy, dict) or "modes" not in taxonomy:
        raise Invalid("propose_patch needs a failure_taxonomy result")
    if isinstance(allowed_grammar, str) or not isinstance(allowed_grammar, (list, tuple, set, frozenset)):
        raise Invalid("allowed_grammar must be a collection of grammar operation names")
    allowed = set(allowed_grammar)
    unknown = allowed - set(ALLOWED_GRAMMAR)
    if unknown:
        raise Invalid("Unknown grammar operations: " + ", ".join(sorted(unknown)))
    rejected, patch, chosen = [], None, None
    for mode in taxonomy["modes"]:
        entry = PATCH_GRAMMAR.get(mode["mode"])
        if entry is None:
            rejected.append({"mode": mode["mode"], "reason": "no_grammar_entry_for_this_mode"})
            continue
        if entry["operation"] not in allowed:
            rejected.append({"mode": mode["mode"], "operation": entry["operation"],
                             "reason": "operation_not_in_allowed_grammar"})
            continue
        if patch is not None:
            rejected.append({"mode": mode["mode"], "operation": entry["operation"],
                             "reason": "lower_ranked_than_the_selected_mode",
                             "total_cost": mode["total_cost"]})
            continue
        changes = entry["build"](policy, mode)
        if not changes:
            rejected.append({"mode": mode["mode"], "operation": entry["operation"],
                             "reason": "already_satisfied_by_the_parent_policy"})
            continue
        if len(changes) > MAX_PATCH_CHANGES:
            rejected.append({"mode": mode["mode"], "operation": entry["operation"],
                             "reason": "grammar_entry_exceeds_the_patch_size_bound"})
            continue
        trace_ids = sorted({i["trace_id"] for i in taxonomy.get("incidents", []) if i["mode"] == mode["mode"]})
        patch = MethodPatch(
            parent_policy_id=policy.policy_id,
            operation=entry["operation"],
            failure_mode=mode["mode"],
            observed_failure=(mode["mode"] + ": " + str(mode["incident_count"]) + " incidents across "
                              + str(mode["trace_count"]) + " traces, " + str(round(mode["total_cost"], 6))
                              + " recorded cost, families " + ", ".join(mode["task_families"][:6])),
            evidence_trace_ids=(trace_ids or ["unattributed"])[:64],
            changes=changes,
            discovery_cost=float(mode["total_cost"]),
            regression_checks=list(entry["regression_checks"]),
            promotion_criterion=entry["promotion_criterion"],
            expected_effect=entry["expected_effect"],
            disconfirming_result=entry["disconfirming_result"],
        )
        chosen = mode["mode"]
    return {
        "version": VERSION,
        "parent_policy_id": policy.policy_id,
        "dominant_mode": taxonomy.get("dominant_mode"),
        "selected_mode": chosen,
        "patch": patch.model_dump() if patch else None,
        "rejected_alternatives": rejected,
        "allowed_grammar": sorted(allowed),
        "deterministic": True,
        "model_calls": 0,
        "scope": SCOPE + " A selected patch is a candidate for protected evaluation. "
                         "Selection order reflects recorded cost, not established causal attribution.",
    }
