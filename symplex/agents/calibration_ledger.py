"""Measure whether the metareasoner's forecasts came true, instead of assuming so.

Loop B of the specification asks for a repeated comparison of predicted usefulness
against realized improvement, and says those traces are the input to method
revision. This module is that comparison. At action selection it persists what the
planner predicted; after the action it measures, deterministically and from stored
artifacts only, what actually changed; and it aggregates the two into a hit rate,
a cost-prediction error and a named recurring failure mode.

Three honesty rules hold throughout:

* A prediction the host cannot measure is `unscorable`, never a miss. A prediction
  with no recorded outcome yet is `pending`, never a miss. Only `scored`
  predictions enter any rate, and every denominator is reported next to it.
* "Realized change" means the derived uncertainty ledger stopped containing the
  items the prediction named. It does not mean the underlying scientific question
  was settled, and it cannot check whether the free-text `expected_change` is what
  actually happened - the host has no way to read that text.
* No network, model or subprocess calls. Everything here is arithmetic over
  records the host already wrote.
"""

from symplex.agents.uncertainty import TYPES, uncertainty_ledger
from symplex.agents.voi import MAX_HISTORY, cost_ratio_summary, ledger_delta
from symplex.core.contracts import Invalid, nonempty, number

VERSION = "metareasoner-calibration-ledger-v1"

PREDICTION_KIND = "action_prediction"
OUTCOME_KIND = "calibration_outcome"

# Mirrors NextStep.search_dimension in symplex.agents.solver.
DIMENSIONS = ("conceptual", "structural", "computational", "experimental", "evidence", "method")

# The kinds symplex.agents.improvement.propose_method will accept as feedback_ids.
# Prediction and outcome records are deliberately NOT in this set: they are the
# evidence for a method change, but the existing method-candidate contract admits
# only these evaluation artifacts, and this module does not widen that contract.
METHOD_FEEDBACK_KINDS = ("model_critique", "hypothesis_review", "experiment_comparison", "review")

MAX_CITED_ITEMS = 24
MAX_SOURCE_IDS = 24
DEFAULT_MIN_OCCURRENCES = 2

# The planner asserts expected_change rather than forecasting it with a probability,
# so a scored assertion is treated as a forecast at probability 1.0 and the Brier
# score degenerates to the miss fraction. It is reported anyway, because it makes
# the cost of asserting explicit, and the field exists so a later calibrated planner
# can be scored properly without changing the record format.
ASSERTED_PROBABILITY = 1.0

SCOPE = (
    "Host measurement of the metareasoner's own forecasts against the derived "
    "uncertainty ledger and stored artifacts. Hit rates are counts over this "
    "problem's scored predictions only; unscorable and pending predictions are "
    "reported separately and never counted as misses. Nothing here establishes "
    "scientific validity, and a high hit rate means the planner's declared gaps "
    "stopped being derivable, not that the investigation is correct."
)


def _problem(store, problem_id):
    if not isinstance(problem_id, str) or not problem_id:
        raise Invalid("problem_id must be a nonempty string")
    try:
        record = store.get(problem_id)
    except KeyError:
        raise Invalid("Unknown problem record") from None
    if record["kind"] != "workspace_problem" or record["stale"]:
        raise Invalid("Select a current problem statement")
    return record


def _bounded_history(value, name="max_history"):
    if type(value) is not int or not 1 <= value <= MAX_HISTORY:
        raise Invalid(name + " must be an integer from 1 to " + str(MAX_HISTORY))
    return value


def _current(store, problem_id, kind, limit):
    return [r for r in store.list(kind) if r["parent"] == problem_id and not r["stale"]][-limit:]


def _check_status(store, problem_id):
    """Latest pass/fail state of each frozen host numerical check id.

    Read from numerical_verification records first, then from the comparison
    summaries, oldest to newest, so the newest recorded result wins.
    """
    status = {}
    for record in _current(store, problem_id, "numerical_verification", MAX_HISTORY):
        for check in record["data"].get("checks") or []:
            if isinstance(check, dict) and isinstance(check.get("id"), str) and type(check.get("passed")) is bool:
                status[check["id"]] = check["passed"]
    for record in _current(store, problem_id, "experiment_comparison", MAX_HISTORY):
        verification = record["data"].get("numerical_verification") or {}
        for check_id in verification.get("failed_check_ids") or []:
            if isinstance(check_id, str):
                status[check_id] = False
    return status


def _snapshot(store, problem_id, ledger):
    status = _check_status(store, problem_id)
    return {
        "ledger_digest": ledger.get("ledger_digest"),
        "ledger_item_ids": list(ledger.get("item_ids") or [i["id"] for i in ledger["items"]]),
        "total_severity_mass": ledger.get("total_severity_mass"),
        "artifact_kinds": sorted({
            r["kind"] for r in store.list()
            if r["parent"] == problem_id and not r["stale"]
            # This module's own records are bookkeeping, not investigation progress.
            and r["kind"] not in (PREDICTION_KIND, OUTCOME_KIND)
        }),
        "failing_check_ids": sorted(k for k, ok in status.items() if not ok),
        "passing_check_ids": sorted(k for k, ok in status.items() if ok),
    }


def record_prediction(
    store, problem_id, *, tool, search_dimension, uncertainty, expected_change,
    predicted_cost, ledger=None, uncertainty_type=None, uncertainty_item_ids=(),
    job_id=None, step=None, action_artifact_id=None, predicted_probability=ASSERTED_PROBABILITY,
    predicted_resource_use="", selection_basis="",
):
    """Persist what the planner predicted, before the action runs.

    Recording a prediction changes nothing about what the action is permitted to
    do; it only makes the forecast checkable afterwards.
    """
    _problem(store, problem_id)
    nonempty(tool)
    nonempty(uncertainty)
    nonempty(expected_change)
    if search_dimension not in DIMENSIONS:
        raise Invalid("Unknown epistemic search dimension: " + str(search_dimension))
    number(predicted_cost, 0)
    number(predicted_probability, 0, 1)
    if not isinstance(uncertainty_item_ids, (list, tuple)):
        raise Invalid("uncertainty_item_ids must be a list or tuple of ledger item ids")
    cited = sorted({i for i in uncertainty_item_ids if isinstance(i, str) and i})
    if len(cited) > MAX_CITED_ITEMS:
        raise Invalid("A prediction may cite at most " + str(MAX_CITED_ITEMS) + " ledger items")
    if uncertainty_type is not None and uncertainty_type not in TYPES:
        raise Invalid("Unknown uncertainty type: " + str(uncertainty_type))

    ledger = uncertainty_ledger(store, problem_id) if ledger is None else ledger
    if not isinstance(ledger, dict) or not isinstance(ledger.get("items"), list):
        raise Invalid("Expected an uncertainty ledger dict containing an items list")
    known = {i["id"]: i for i in ledger["items"]}
    unknown_ids = [i for i in cited if i not in known]
    matched = [known[i] for i in cited if i in known]
    if uncertainty_type is None and matched:
        # Deterministic: the most severe cited item, then its id, names the type.
        uncertainty_type = sorted(matched, key=lambda i: (-i["severity"], i["id"]))[0]["type"]
    sources = sorted({s for i in matched for s in i.get("derived_from") or []})[:MAX_SOURCE_IDS]

    data = {
        "version": VERSION,
        "problem_id": problem_id,
        "job_id": job_id,
        "step": step,
        "tool": tool,
        "search_dimension": search_dimension,
        "uncertainty_addressed": uncertainty,
        "uncertainty_type": uncertainty_type,
        "uncertainty_item_ids": cited,
        "uncertainty_item_ids_not_in_ledger": unknown_ids,
        "uncertainty_source_ids": sources,
        "expected_change": expected_change,
        "predicted_cost": float(predicted_cost),
        "predicted_cost_units": "declared planning units as supplied by the caller; not currency",
        # Free text the planner wrote about resources. The host cannot score prose, so
        # it is retained verbatim and never converted into a number that looks measured.
        "predicted_resource_use": str(predicted_resource_use)[:600],
        "item_selection_basis": str(selection_basis)[:600] or "supplied by the caller",
        "predicted_probability": float(predicted_probability),
        "probability_basis": "asserted by the planner unless the caller supplied a calibrated value",
        "action_artifact_id": action_artifact_id,
        "before": _snapshot(store, problem_id, ledger),
        "scope": "Forecast recorded before execution. It is a prediction to be scored, never a result, "
                 "and it grants no permission and reserves no resource.",
    }
    prediction_id = store.put(PREDICTION_KIND, data, problem_id)
    return {
        "version": VERSION,
        "prediction_id": prediction_id,
        "tool": tool,
        "search_dimension": search_dimension,
        "uncertainty_type": uncertainty_type,
        "cited_item_count": len(cited),
        "scorable": bool(cited) and not unknown_ids,
        "unscorable_reason": "" if cited else "the prediction cited no ledger item, so no resolution can be measured",
        "scope": data["scope"],
    }


def score_outcome(store, problem_id, prediction_id, *, ledger_after=None, actual_cost=None, note=""):
    """Measure what actually happened after the predicted action completed."""
    _problem(store, problem_id)
    if not isinstance(prediction_id, str) or not prediction_id:
        raise Invalid("prediction_id must be a nonempty string")
    try:
        prediction = store.get(prediction_id)
    except KeyError:
        raise Invalid("Unknown prediction record") from None
    if prediction["kind"] != PREDICTION_KIND or prediction["parent"] != problem_id or prediction["stale"]:
        raise Invalid("Select a current prediction belonging to this problem")
    if actual_cost is not None:
        number(actual_cost, 0)
    data = prediction["data"]
    before = data.get("before") or {}

    ledger_after = uncertainty_ledger(store, problem_id) if ledger_after is None else ledger_after
    if not isinstance(ledger_after, dict) or not isinstance(ledger_after.get("items"), list):
        raise Invalid("Expected an uncertainty ledger dict containing an items list")
    after = _snapshot(store, problem_id, ledger_after)

    before_ids = set(before.get("ledger_item_ids") or [])
    after_ids = set(after["ledger_item_ids"])
    delta = ledger_delta(
        {"items": [{"id": i, "type": "decision", "severity": 0.0} for i in sorted(before_ids)]},
        {"items": [{"id": i, "type": "decision", "severity": 0.0} for i in sorted(after_ids)]},
    ) if before.get("ledger_item_ids") is not None else None

    cited = list(data.get("uncertainty_item_ids") or [])
    resolved_cited = sorted(i for i in cited if i not in after_ids)
    still_open = sorted(i for i in cited if i in after_ids)

    if not cited:
        status, hit, reason = "unscorable", None, "the prediction cited no ledger item, so resolution cannot be measured"
    elif before.get("ledger_digest") is None:
        status, hit, reason = "unscorable", None, "no ledger snapshot was recorded at prediction time"
    elif any(i not in before_ids for i in cited):
        status, hit, reason = "unscorable", None, "a cited item was not open in the ledger at prediction time"
    else:
        status, hit, reason = "scored", bool(resolved_cited), ""

    new_kinds = sorted(set(after["artifact_kinds"]) - set(before.get("artifact_kinds") or []))
    repaired = sorted(set(before.get("failing_check_ids") or []) & set(after["passing_check_ids"]))
    broken = sorted(set(before.get("passing_check_ids") or []) & set(after["failing_check_ids"]))

    predicted_cost = data.get("predicted_cost")
    if actual_cost is None:
        cost = {"status": "unscorable", "reason": "no actual cost was supplied for this action"}
    elif not isinstance(predicted_cost, (int, float)) or predicted_cost <= 0:
        # The recorded actual is kept so the gap is visible; no ratio is invented.
        cost = {"status": "unscorable", "actual_cost": float(actual_cost),
                "predicted_resource_use": data.get("predicted_resource_use", ""),
                "reason": "the predicted cost was not a positive number, so no ratio is established"}
    else:
        ratio = float(actual_cost) / float(predicted_cost)
        cost = {
            "status": "scored", "predicted_cost": float(predicted_cost),
            "actual_cost": float(actual_cost), "ratio": round(ratio, 6),
            "direction": "underestimated" if ratio > 1.0 else "overestimated" if ratio < 1.0 else "exact",
        }

    outcome = {
        "version": VERSION,
        "problem_id": problem_id,
        "prediction_id": prediction_id,
        "tool": data.get("tool"),
        "search_dimension": data.get("search_dimension"),
        "uncertainty_type": data.get("uncertainty_type"),
        "status": status,
        "hit": hit,
        "unscorable_reason": reason,
        "cited_item_ids": sorted(cited),
        "resolved_cited_item_ids": resolved_cited,
        "still_open_cited_item_ids": still_open,
        "hit_definition": "at least one cited ledger item is no longer derivable from stored artifacts",
        "ledger_before_digest": before.get("ledger_digest"),
        "ledger_after_digest": after["ledger_digest"],
        "ledger_item_delta": {
            "resolved_count": delta["resolved_count"], "new_count": delta["new_count"],
            "unchanged_count": delta["unchanged_count"],
        } if delta else None,
        "severity_mass_before": before.get("total_severity_mass"),
        "severity_mass_after": after["total_severity_mass"],
        "secondary_observations": {
            "new_artifact_kinds": new_kinds,
            "host_checks_repaired": repaired,
            "host_checks_broken": broken,
            "note": "recorded alongside the scored result; these are not the hit criterion",
        },
        "cost": cost,
        "predicted_probability": data.get("predicted_probability", ASSERTED_PROBABILITY),
        "operator_note": str(note)[:600],
        "scope": "Deterministic measurement of recorded state before and after one action. A hit means the "
                 "cited declared gap stopped being derivable, not that the question was scientifically settled; "
                 "the free-text expected_change itself cannot be checked by the host.",
    }
    outcome_id = store.put(OUTCOME_KIND, outcome, problem_id)
    return dict(outcome, outcome_id=outcome_id)


def action_history(store, problem_id, *, max_history=MAX_HISTORY):
    """Flat, bounded prediction-and-outcome history: the input to every rate here."""
    _problem(store, problem_id)
    _bounded_history(max_history)
    predictions = _current(store, problem_id, PREDICTION_KIND, max_history)
    outcomes = {}
    for record in _current(store, problem_id, OUTCOME_KIND, max_history * 2):
        outcomes[record["data"].get("prediction_id")] = record
    entries = []
    for prediction in predictions:
        data = prediction["data"]
        outcome = outcomes.get(prediction["id"])
        result = outcome["data"] if outcome else None
        entries.append({
            "prediction_id": prediction["id"],
            "outcome_id": outcome["id"] if outcome else None,
            "tool": data.get("tool"),
            "search_dimension": data.get("search_dimension"),
            "uncertainty_type": data.get("uncertainty_type"),
            "status": result["status"] if result else "pending",
            "hit": result["hit"] if result else None,
            "unscorable_reason": result["unscorable_reason"] if result else "",
            "predicted_cost": data.get("predicted_cost"),
            "predicted_probability": data.get("predicted_probability", ASSERTED_PROBABILITY),
            "cost": result["cost"] if result else {"status": "pending", "reason": "no outcome recorded yet"},
            "source_ids": data.get("uncertainty_source_ids") or [],
            "cited_item_ids": data.get("uncertainty_item_ids") or [],
        })
    return {
        "version": VERSION,
        "problem_id": problem_id,
        "entries": entries,
        "predictions_scanned": len(predictions),
        "history_basis": "no history" if not entries else "recorded predictions for this problem",
        "scope": "Predictions and their measured outcomes for this problem only. Pending entries have no "
                 "outcome yet and unscorable entries could not be measured; neither is a miss.",
    }


def _group(entries, key):
    grouped = {}
    for entry in entries:
        name = entry.get(key)
        if not isinstance(name, str) or not name:
            continue
        bucket = grouped.setdefault(name, {"attempts": 0, "scored": 0, "hits": 0, "misses": 0,
                                           "unscorable": 0, "pending": 0})
        bucket["attempts"] += 1
        if entry["status"] == "scored":
            bucket["scored"] += 1
            bucket["hits" if entry["hit"] else "misses"] += 1
        elif entry["status"] == "unscorable":
            bucket["unscorable"] += 1
        else:
            bucket["pending"] += 1
    for bucket in grouped.values():
        bucket["hit_rate"] = round(bucket["hits"] / bucket["scored"], 6) if bucket["scored"] else None
        bucket["basis"] = "measured" if bucket["scored"] else "no scored history"
        bucket["denominator"] = "scored predictions only; unscorable and pending are excluded, never counted as misses"
    return dict(sorted(grouped.items()))


def calibration_report(store, problem_id, *, max_history=MAX_HISTORY):
    """Is the metareasoner any good? Per dimension, per tool, and at estimating cost."""
    history = action_history(store, problem_id, max_history=max_history)
    entries = history["entries"]
    scored = [e for e in entries if e["status"] == "scored"]
    unscorable = [e for e in entries if e["status"] == "unscorable"]
    pending = [e for e in entries if e["status"] == "pending"]

    reasons = {}
    for entry in unscorable:
        reason = entry["unscorable_reason"] or "unstated"
        reasons[reason] = reasons.get(reason, 0) + 1

    cost_scored = [e for e in entries if (e["cost"] or {}).get("status") == "scored"]
    ratios = [e["cost"]["ratio"] for e in cost_scored]
    summary = cost_ratio_summary(ratios)
    summary.update({
        "underestimates": sum(1 for e in cost_scored if e["cost"]["direction"] == "underestimated"),
        "overestimates": sum(1 for e in cost_scored if e["cost"]["direction"] == "overestimated"),
        "exact": sum(1 for e in cost_scored if e["cost"]["direction"] == "exact"),
        "unscorable": len(entries) - len(cost_scored),
        "ratio_definition": "actual cost divided by predicted cost; above 1.0 means the planner underestimated",
    })

    if scored:
        brier_value = round(
            sum((float(e["predicted_probability"]) - (1.0 if e["hit"] else 0.0)) ** 2 for e in scored) / len(scored), 6)
        brier = {
            "scored": len(scored), "score": brier_value,
            "forecast_probability_source": "planner assertions are scored at probability "
                                           + str(ASSERTED_PROBABILITY) + " unless the caller supplied a calibrated value",
            "degenerate": all(e["predicted_probability"] == ASSERTED_PROBABILITY for e in scored),
            "scope": "With asserted probabilities this equals the miss fraction and measures nothing beyond the hit "
                     "rate. It is reported to make the cost of asserting explicit, not as a calibration curve.",
        }
    else:
        brier = {"scored": 0, "score": None, "basis": "no scored predictions",
                 "scope": "No Brier score is established; no prediction could be scored."}

    return {
        "version": VERSION,
        "problem_id": problem_id,
        "predictions_total": len(entries),
        "scored": len(scored),
        "unscorable": len(unscorable),
        "pending": len(pending),
        "hits": sum(1 for e in scored if e["hit"]),
        "misses": sum(1 for e in scored if not e["hit"]),
        "hit_rate": round(sum(1 for e in scored if e["hit"]) / len(scored), 6) if scored else None,
        "hit_rate_basis": "measured over scored predictions only" if scored else "no scored predictions",
        "unscorable_reasons": dict(sorted(reasons.items())),
        "by_search_dimension": _group(entries, "search_dimension"),
        "by_tool": _group(entries, "tool"),
        "cost_error": summary,
        "brier": brier,
        "counting_rule": "scored, unscorable and pending are disjoint and always reported together; an unscorable "
                         "or pending prediction is never counted as a miss",
        "scope": SCOPE,
    }


def improvement_signal(store, problem_id, *, min_occurrences=DEFAULT_MIN_OCCURRENCES, max_history=MAX_HISTORY):
    """Name the specific recurring failure mode a method revision should target.

    The output is evidence for `symplex.agents.improvement.propose_method`, not a
    method change. That contract admits only evaluation artifacts as feedback_ids,
    so the prediction and outcome records are returned separately as measurement
    provenance and the admissible feedback ids are listed explicitly.
    """
    if type(min_occurrences) is not int or not 1 <= min_occurrences <= 50:
        raise Invalid("min_occurrences must be an integer from 1 to 50")
    history = action_history(store, problem_id, max_history=max_history)
    entries = history["entries"]

    signals = []
    pairs = {}
    for entry in entries:
        tool, kind = entry["tool"], entry["uncertainty_type"]
        if entry["status"] != "scored" or not tool or kind not in TYPES:
            continue
        bucket = pairs.setdefault((tool, kind), {"attempts": 0, "hits": 0, "entries": []})
        bucket["attempts"] += 1
        bucket["hits"] += 1 if entry["hit"] else 0
        bucket["entries"].append(entry)
    for (tool, kind), bucket in sorted(pairs.items()):
        if bucket["attempts"] < min_occurrences or bucket["hits"]:
            continue
        signals.append({
            "kind": "unresolved_pairing",
            "failure_mode": "%s uncertainties are repeatedly addressed with `%s` and never resolved "
                            "(0 of %d scored attempts resolved a cited ledger item)" % (kind, tool, bucket["attempts"]),
            "tool": tool,
            "uncertainty_type": kind,
            "search_dimension": sorted({e["search_dimension"] for e in bucket["entries"] if e["search_dimension"]}),
            "occurrences": bucket["attempts"],
            "hits": 0,
            "supporting_artifact_ids": sorted(
                {e["prediction_id"] for e in bucket["entries"]} | {e["outcome_id"] for e in bucket["entries"] if e["outcome_id"]}),
            "cited_source_ids": sorted({s for e in bucket["entries"] for s in e["source_ids"]}),
        })

    by_tool = {}
    for entry in entries:
        cost = entry["cost"] or {}
        if cost.get("status") != "scored" or not entry["tool"]:
            continue
        bucket = by_tool.setdefault(entry["tool"], {"under": [], "over": [], "exact": []})
        bucket[{"underestimated": "under", "overestimated": "over", "exact": "exact"}[cost["direction"]]].append(entry)
    for tool, bucket in sorted(by_tool.items()):
        if len(bucket["under"]) < min_occurrences or len(bucket["under"]) <= len(bucket["over"]):
            continue
        signals.append({
            "kind": "cost_underestimation",
            "failure_mode": "`%s` costs are repeatedly underestimated (%d of %d scored cost comparisons ran over)"
                            % (tool, len(bucket["under"]), sum(len(v) for v in bucket.values())),
            "tool": tool,
            "uncertainty_type": None,
            "search_dimension": sorted({e["search_dimension"] for e in bucket["under"] if e["search_dimension"]}),
            "occurrences": len(bucket["under"]),
            "hits": None,
            "supporting_artifact_ids": sorted(
                {e["prediction_id"] for e in bucket["under"]} | {e["outcome_id"] for e in bucket["under"] if e["outcome_id"]}),
            "cited_source_ids": sorted({s for e in bucket["under"] for s in e["source_ids"]}),
        })

    unscorable = [e for e in entries if e["status"] == "unscorable"]
    if len(unscorable) >= min_occurrences:
        signals.append({
            "kind": "unscorable_predictions",
            "failure_mode": "%d predictions could not be scored at all because the planner did not name a ledger "
                            "item its action would close; the forecast loop cannot measure them" % len(unscorable),
            "tool": None,
            "uncertainty_type": None,
            "search_dimension": sorted({e["search_dimension"] for e in unscorable if e["search_dimension"]}),
            "occurrences": len(unscorable),
            "hits": None,
            "supporting_artifact_ids": sorted(
                {e["prediction_id"] for e in unscorable} | {e["outcome_id"] for e in unscorable if e["outcome_id"]}),
            "cited_source_ids": sorted({s for e in unscorable for s in e["source_ids"]}),
        })

    for index, signal in enumerate(signals):
        signal["id"] = "signal_%02d_%s" % (index + 1, signal["kind"])
    supporting = sorted({i for s in signals for i in s["supporting_artifact_ids"]})
    cited_sources = sorted({i for s in signals for i in s["cited_source_ids"]})

    admissible, considered = [], []
    for kind in METHOD_FEEDBACK_KINDS:
        for record in _current(store, problem_id, kind, 8):
            considered.append(record["id"])
            if record["id"] in cited_sources:
                admissible.append(record["id"])
    eligible = sorted(set(admissible)) or sorted(set(considered))
    return {
        "version": VERSION,
        "problem_id": problem_id,
        "signals": signals,
        "signal_count": len(signals),
        "supporting_artifact_ids": supporting,
        "measurement_provenance_ids": supporting,
        "eligible_feedback_ids": eligible,
        "eligible_feedback_basis": (
            "uncertainty sources cited by the failing predictions" if admissible
            else "current evaluation artifacts for this problem" if eligible
            else "none: no admissible evaluation artifact exists, so propose_method will refuse"),
        "method_candidate_inputs": {
            "role": "metareasoner",
            "observed_failure": signals[0]["failure_mode"] if signals else "",
            "feedback_ids": eligible,
            "note": "propose_method admits feedback_ids only from " + ", ".join(METHOD_FEEDBACK_KINDS)
                    + "; prediction and outcome records are measurement provenance and are cited separately.",
        },
        "predictions_scanned": history["predictions_scanned"],
        "basis": "measured" if signals else "no recurring failure mode met the occurrence threshold",
        "min_occurrences": min_occurrences,
        "scope": "Recurring patterns in this problem's own scored forecasts. A named failure mode is a reason to "
                 "propose a method change and evaluate it, never a reason to install one, and it does not "
                 "generalise beyond this problem.",
    }
