"""Heuristic allocation prior over the permitted action set. Not decision-theoretic VOI.

What this is: a deterministic, auditable arithmetic that ranks the tools the host
has already permitted, by how much declared uncertainty mass each could plausibly
address, discounted by how often that tool has actually resolved that kind of
uncertainty here, divided by a declared cost, times an exploration bonus for
under-visited search dimensions. Every term is returned separately so a human can
recompute the ranking by hand and override it.

What this is NOT: it is not Bayesian, it has no posterior, no likelihood and no
utility function. "Expected information gain" here is a weighted count of declared
gaps, not an expectation over a probability model, and severity is a host-declared
ordering constant rather than a loss. Calling this VOI would overclaim; the name is
kept only because it is the role the number plays in the planner's context.

Authority: this score is ADVISORY input to the planner's context and nothing else.
It never selects, executes or authorises an action. `solver.limit_reason()`, the
depth profile's permission envelope and the budget remain the only authorities on
what may run, and a high score for a forbidden or capped tool changes none of them.
This module makes no network, model or subprocess calls.
"""

from statistics import median

from symplex.agents.uncertainty import TYPES, ledger_digest
from symplex.core.contracts import Invalid

VERSION = "heuristic-action-allocation-prior-v1"

# The prior used when this problem has no scored history for a (tool, type) pair.
# 0.25 is a deliberate, documented, pessimistic constant: it is not fitted, not
# measured, and not a belief. It exists so an unmeasured pairing does not silently
# score as either certain or worthless, and every use of it is reported.
PRIOR_RESOLUTION_RATE = 0.25
PRIOR_BASIS = (
    "documented flat constant used when no scored outcome exists for this tool and "
    "uncertainty type in this problem; not fitted, not measured, not a belief"
)
LOW_SAMPLE_BELOW = 3
EXPLORATION_WEIGHT = 0.5
MIN_EXPECTED_COST = 0.1
MAX_HISTORY = 200

# Declared relative cost in action-step units, host-owned. These are ordering
# constants for planning only; the budget ledger remains the authority on spend.
DEFAULT_TOOL_COST = 1.0
TOOL_COST_UNITS = {
    "run_p1": 4.0,
    "run_model_code": 3.0,
    "run_research_pilot": 3.0,
    "represent_system": 2.5,
    "evolve_model": 2.5,
    "acquire_bamtwoogle": 2.0,
    "research_evidence": 2.0,
    "search_literature": 2.0,
    "design_solution": 2.0,
    "develop_candidate": 2.0,
    "build_scene": 2.0,
    "run_system_scenarios": 2.0,
    "plan_experiment": 1.5,
    "review_model": 1.5,
    "review_hypotheses": 1.5,
    "synthesize_evidence": 1.5,
    "build_outcome": 1.5,
    "propose_method": 1.5,
    "reframe_problem": 1.5,
    "draft_component": 1.5,
    "ask_user": 1.5,
    "compare_computation": 1.0,
    "fetch_artifact": 1.0,
    "profile_table": 1.0,
    "search_artifacts": 1.0,
    "inspect_system_graph": 0.5,
    "inspect_artifact": 0.5,
    "inspect_context": 0.5,
    "discover_datasets": 0.5,
}

# Mirrors NextStep.search_dimension. A tool is declared to explore one dimension so
# the exploration bonus has something to count; this is a host label, not a claim
# about what the tool discovers.
TOOL_DIMENSION = {
    "reframe_problem": "conceptual",
    "inspect_context": "conceptual",
    "inspect_system_graph": "conceptual",
    "ask_user": "conceptual",
    "represent_system": "structural",
    "review_hypotheses": "structural",
    "review_model": "structural",
    "draft_component": "structural",
    "evolve_model": "structural",
    "build_scene": "structural",
    "run_model_code": "computational",
    "run_system_scenarios": "computational",
    "compare_computation": "computational",
    "profile_table": "computational",
    "inspect_artifact": "computational",
    "plan_experiment": "experimental",
    "run_p1": "experimental",
    "design_solution": "experimental",
    "develop_candidate": "experimental",
    "run_research_pilot": "experimental",
    "build_outcome": "experimental",
    "research_evidence": "evidence",
    "search_literature": "evidence",
    "search_artifacts": "evidence",
    "fetch_artifact": "evidence",
    "discover_datasets": "evidence",
    "synthesize_evidence": "evidence",
    "acquire_bamtwoogle": "evidence",
    "propose_method": "method",
}

# Mirrors solver.REVISION_TOOLS and the research cap enforced by limit_reason(); it
# is used only to make a scarce allowance more expensive here. `limit_reason()`
# remains the sole enforcement point, and `check_tool_tables()` reports drift.
DECLARED_REVISION_TOOLS = frozenset({
    "design_solution", "run_system_scenarios", "run_model_code", "review_model",
    "represent_system", "develop_candidate", "build_outcome", "review_hypotheses",
    "synthesize_evidence", "plan_experiment", "compare_computation", "build_scene",
    "evolve_model", "reframe_problem",
})
DECLARED_RESEARCH_TOOLS = frozenset({"research_evidence", "search_literature"})
NOT_SCORED = frozenset({"deliver"})

ARITHMETIC = {
    "expected_information_gain": "sum over uncertainty types of severity_mass[type] * resolution_rate[tool,type]",
    "expected_cost": "declared_cost_units[tool] * (1 + capped_allowance_used_fraction), floored at " + str(MIN_EXPECTED_COST),
    "exploration_bonus": "1 + %s * (1 - dimension_visits[dim] / max_dimension_visits)" % EXPLORATION_WEIGHT,
    "score": "expected_information_gain / expected_cost * exploration_bonus",
}

SCOPE = (
    "Heuristic allocation prior over already-permitted tools, not a decision-theoretic "
    "value of information: there is no posterior, no likelihood and no utility. Severity "
    "mass is a host-declared ordering weight, resolution rates are small-sample counts "
    "from this problem only, and costs are declared constants. Advisory input to the "
    "planner's context; it never bypasses limit_reason(), the permission envelope or the "
    "budget, and it establishes nothing about scientific value."
)


def _check_ledger(ledger):
    if not isinstance(ledger, dict) or not isinstance(ledger.get("items"), list):
        raise Invalid("Expected an uncertainty ledger dict containing an items list")
    for item in ledger["items"]:
        if not isinstance(item, dict) or item.get("type") not in TYPES or "severity" not in item:
            raise Invalid("Ledger item is missing a known type or severity")
    return ledger


def _normalise_tools(available_tools):
    if isinstance(available_tools, dict):
        names = list(available_tools)
    elif isinstance(available_tools, (list, tuple, set, frozenset)):
        names = list(available_tools)
    else:
        raise Invalid("available_tools must be a dict, list, tuple or set of tool names")
    if any(not isinstance(n, str) or not n for n in names):
        raise Invalid("Tool names must be nonempty strings")
    if len(names) > 200:
        raise Invalid("Refusing to score more than 200 tools")
    return sorted(set(names))


def _normalise_history(history):
    if history is None:
        return []
    if not isinstance(history, (list, tuple)):
        raise Invalid("history must be a list of action history entries or None")
    entries = []
    for entry in list(history)[-MAX_HISTORY:]:
        if not isinstance(entry, dict):
            raise Invalid("Each history entry must be a dict")
        entries.append(entry)
    return entries


def _counts_from_history(history):
    """Attempts and resolutions per (tool, uncertainty type). Only outcomes the host
    could actually score are counted; unscorable attempts never become misses."""
    counts = {}
    for entry in history:
        tool, kind = entry.get("tool"), entry.get("uncertainty_type")
        if not tool or kind not in TYPES or entry.get("status") != "scored":
            continue
        bucket = counts.setdefault((tool, kind), {"attempts": 0, "resolutions": 0})
        bucket["attempts"] += 1
        if entry.get("hit") is True:
            bucket["resolutions"] += 1
    return counts


def _dimension_visits(history):
    visits = {}
    for entry in history:
        dimension = entry.get("search_dimension")
        if isinstance(dimension, str) and dimension:
            visits[dimension] = visits.get(dimension, 0) + 1
    return visits


def _rate(counts, tool, kind):
    bucket = counts.get((tool, kind))
    if not bucket or not bucket["attempts"]:
        return {
            "rate": PRIOR_RESOLUTION_RATE, "basis": "prior", "attempts": 0,
            "resolutions": 0, "prior": PRIOR_RESOLUTION_RATE, "prior_basis": PRIOR_BASIS,
            "note": "no scored history for this tool and uncertainty type; the documented prior is used",
        }
    rate = bucket["resolutions"] / bucket["attempts"]
    return {
        "rate": rate, "basis": "measured", "attempts": bucket["attempts"],
        "resolutions": bucket["resolutions"], "prior": PRIOR_RESOLUTION_RATE,
        "prior_basis": PRIOR_BASIS, "low_sample": bucket["attempts"] < LOW_SAMPLE_BELOW,
        "note": "measured from this problem's scored outcomes only; a small sample is a count, not a rate estimate",
    }


def _expected_cost(tool, envelope, history):
    base = TOOL_COST_UNITS.get(tool, DEFAULT_TOOL_COST)
    envelope = envelope if isinstance(envelope, dict) else {}
    used_fraction, cap_name, cap, used = 0.0, None, None, 0
    if tool in DECLARED_REVISION_TOOLS:
        cap_name, cap = "revisions", envelope.get("revisions")
        used = sum(1 for e in history if e.get("tool") in DECLARED_REVISION_TOOLS)
    elif tool in DECLARED_RESEARCH_TOOLS:
        cap_name, cap = "research", envelope.get("research")
        used = sum(1 for e in history if e.get("tool") in DECLARED_RESEARCH_TOOLS)
    if isinstance(cap, (int, float)) and not isinstance(cap, bool) and cap > 0:
        used_fraction = min(1.0, used / float(cap))
    cost = max(MIN_EXPECTED_COST, base * (1.0 + used_fraction))
    return {
        "declared_cost_units": base, "capped_allowance": cap_name,
        "allowance_cap": cap, "allowance_used": used,
        "capped_allowance_used_fraction": used_fraction, "expected_cost": cost,
        "note": "declared planning constants; the budget ledger and limit_reason() remain the authority on what may run",
    }


def _exploration_bonus(tool, visits):
    dimension = TOOL_DIMENSION.get(tool)
    if dimension is None:
        return {"search_dimension": None, "visits": None, "max_visits": None,
                "exploration_bonus": 1.0,
                "note": "no declared search dimension for this tool; no exploration bonus applied"}
    max_visits = max(visits.values()) if visits else 0
    seen = visits.get(dimension, 0)
    if max_visits <= 0:
        bonus = 1.0 + EXPLORATION_WEIGHT
        note = "no recorded visits to any dimension; the bonus is uniform and does not affect ranking"
    else:
        bonus = 1.0 + EXPLORATION_WEIGHT * (1.0 - seen / float(max_visits))
        note = "under-visited dimensions are favoured relative to the most-visited dimension"
    return {"search_dimension": dimension, "visits": seen, "max_visits": max_visits,
            "exploration_weight": EXPLORATION_WEIGHT, "exploration_bonus": bonus, "note": note}


def score_actions(ledger, available_tools, envelope=None, history=None):
    """Rank permitted tools by a fully exposed heuristic score. Advisory only."""
    _check_ledger(ledger)
    tools = _normalise_tools(available_tools)
    history = _normalise_history(history)
    counts = _counts_from_history(history)
    visits = _dimension_visits(history)

    by_type_items = {t: [] for t in TYPES}
    for item in ledger["items"]:
        by_type_items[item["type"]].append(item)

    ranked, unscored, prior_used = [], [], []
    for tool in tools:
        if tool in NOT_SCORED:
            unscored.append({"tool": tool, "reason": "delivery is a host action, not an information-gathering action"})
            continue
        severity_mass, rates, weighted, addressed = {}, {}, {}, []
        for kind in TYPES:
            matching = [i for i in by_type_items[kind] if tool in (i.get("resolvable_by") or [])]
            if not matching:
                continue
            mass = round(sum(i["severity"] for i in matching), 6)
            rate = _rate(counts, tool, kind)
            if rate["basis"] == "prior":
                prior_used.append(tool + "|" + kind)
            severity_mass[kind] = mass
            rates[kind] = rate
            weighted[kind] = mass * rate["rate"]
            addressed.extend(i["id"] for i in matching)
        gain = sum(weighted.values())
        cost = _expected_cost(tool, envelope, history)
        bonus = _exploration_bonus(tool, visits)
        score = gain / cost["expected_cost"] * bonus["exploration_bonus"]
        ranked.append({
            "tool": tool,
            "score": score,
            "terms": {
                "severity_mass_by_type": severity_mass,
                "resolution_rate_by_type": rates,
                "weighted_gain_by_type": weighted,
                "expected_information_gain": gain,
                "expected_cost": cost,
                "exploration": bonus,
            },
            "addressed_item_ids": sorted(set(addressed)),
            "addressed_item_count": len(set(addressed)),
            "arithmetic": ARITHMETIC,
            "note": "no declared uncertainty in this ledger lists this tool as resolvable_by"
                    if not severity_mass else "",
        })
    # Total order: score, then tool name, so repeated calls never reorder ties.
    ranked.sort(key=lambda r: (-round(r["score"], 12), r["tool"]))
    for position, entry in enumerate(ranked, start=1):
        entry["rank"] = position
    return {
        "version": VERSION,
        "ranked": ranked,
        "unscored_tools": unscored,
        "ledger_digest": ledger.get("ledger_digest"),
        "total_severity_mass": ledger.get("total_severity_mass"),
        "history_entries_used": len(history),
        "history_basis": "measured outcomes from this problem" if counts else "no scored history; every rate is the documented prior",
        "prior_used_for": sorted(set(prior_used)),
        "prior_resolution_rate": PRIOR_RESOLUTION_RATE,
        "prior_basis": PRIOR_BASIS,
        "dimension_visits": dict(sorted(visits.items())),
        "arithmetic": ARITHMETIC,
        "authority": "Advisory ranking for the planner's context. It does not select or run anything; "
                     "limit_reason(), the permission envelope and the budget remain the only authorities.",
        "scope": SCOPE,
    }


def resolution_rate(store, problem_id, tool, uncertainty_type, *, max_history=MAX_HISTORY):
    """Measured resolution rate of one tool against one uncertainty type, here.

    Returns the documented prior, and says so, when this problem has no scored
    outcome for the pairing. It never smooths an empty history into a number that
    looks measured.
    """
    if not isinstance(tool, str) or not tool:
        raise Invalid("tool must be a nonempty string")
    if uncertainty_type not in TYPES:
        raise Invalid("Unknown uncertainty type: " + str(uncertainty_type))
    if type(max_history) is not int or not 1 <= max_history <= MAX_HISTORY:
        raise Invalid("max_history must be an integer from 1 to " + str(MAX_HISTORY))
    # Deferred import: calibration_ledger owns the prediction/outcome records and
    # imports ledger_delta from here, so the dependency is resolved at call time.
    from symplex.agents.calibration_ledger import action_history

    history = action_history(store, problem_id, max_history=max_history)["entries"]
    result = dict(_rate(_counts_from_history(history), tool, uncertainty_type))
    result.update({
        "version": VERSION, "tool": tool, "uncertainty_type": uncertainty_type,
        "history_entries_scanned": len(history),
        "history_basis": "no history" if not history else "scored outcomes recorded for this problem",
        "scope": "Resolution counts from this problem's own recorded outcomes. A rate over a handful "
                 "of attempts is a count, not an estimate, and it does not transfer to other problems.",
    })
    return result


def ledger_delta(before, after):
    """Realized change between two ledgers: what closed, what opened, what stayed.

    This is the measurement the metareasoner's forecasts are scored against. It
    measures the recorded ledger only; a resolved item means the declared gap is no
    longer derivable from stored artifacts, not that the underlying question is
    scientifically settled.
    """
    _check_ledger(before)
    _check_ledger(after)
    prior = {i["id"]: i for i in before["items"]}
    current = {i["id"]: i for i in after["items"]}
    resolved = sorted(set(prior) - set(current))
    created = sorted(set(current) - set(prior))
    unchanged = sorted(set(prior) & set(current))
    before_mass = round(sum(i["severity"] for i in prior.values()), 6)
    after_mass = round(sum(i["severity"] for i in current.values()), 6)
    net = round(after_mass - before_mass, 6)
    by_type = {}
    for kind in TYPES:
        by_type[kind] = {
            "resolved": sum(1 for i in resolved if prior[i]["type"] == kind),
            "new": sum(1 for i in created if current[i]["type"] == kind),
            "unchanged": sum(1 for i in unchanged if current[i]["type"] == kind),
            "net_severity_change": round(
                sum(current[i]["severity"] for i in current if current[i]["type"] == kind)
                - sum(prior[i]["severity"] for i in prior if prior[i]["type"] == kind), 6),
        }
    return {
        "version": VERSION,
        "resolved_ids": resolved,
        "new_ids": created,
        "unchanged_ids": unchanged,
        "resolved_count": len(resolved),
        "new_count": len(created),
        "unchanged_count": len(unchanged),
        "resolved_severity_mass": round(sum(prior[i]["severity"] for i in resolved), 6),
        "new_severity_mass": round(sum(current[i]["severity"] for i in created), 6),
        "before_severity_mass": before_mass,
        "after_severity_mass": after_mass,
        "net_severity_change": net,
        "direction": "reduced" if net < 0 else "increased" if net > 0 else "unchanged",
        "by_type": by_type,
        "before_digest": before.get("ledger_digest") or ledger_digest(before),
        "after_digest": after.get("ledger_digest") or ledger_digest(after),
        "scope": "Change in the derived ledger between two points. Resolution means the declared gap is "
                 "no longer derivable from stored artifacts; it is not scientific settlement, and a new "
                 "item may simply mean the record became more explicit.",
    }


def cost_ratio_summary(ratios):
    """Mean and median of predicted-versus-actual cost ratios, or an explicit empty."""
    values = [float(r) for r in ratios if isinstance(r, (int, float)) and not isinstance(r, bool)]
    if not values:
        return {"count": 0, "mean_ratio": None, "median_ratio": None,
                "basis": "no scorable cost pairs", "scope": "No cost comparison is established."}
    return {
        "count": len(values),
        "mean_ratio": round(sum(values) / len(values), 6),
        "median_ratio": round(float(median(values)), 6),
        "basis": "measured",
        "scope": "Ratio of recorded actual cost to predicted cost; declared planning units, not currency.",
    }


def check_tool_tables():
    """Report tools registered by the host that these declared tables do not cover.

    Drift here changes only the ranking, never enforcement, but an uncovered tool
    is silently scored at the default cost with no exploration bonus, so it should
    be visible rather than discovered later.
    """
    try:
        from symplex.agents.tools import REGISTRY
        registered = set(REGISTRY.descriptions())
    except Exception as exc:  # pragma: no cover - reported, never raised
        return {"version": VERSION, "available": False, "reason": str(exc)[:300],
                "scope": "The tool registry could not be inspected; no drift check was performed."}
    scored = registered - set(NOT_SCORED)
    return {
        "version": VERSION,
        "available": True,
        "registered_tool_count": len(registered),
        "missing_dimension": sorted(scored - set(TOOL_DIMENSION)),
        "missing_cost": sorted(scored - set(TOOL_COST_UNITS)),
        "unknown_declared_tools": sorted((set(TOOL_DIMENSION) | set(TOOL_COST_UNITS)) - registered),
        "revision_tool_drift": sorted(DECLARED_REVISION_TOOLS - registered),
        "scope": "Coverage of host-registered tools by this module's declared planning tables. "
                 "Drift affects ranking only; limit_reason() and the permission envelope are unaffected.",
    }
