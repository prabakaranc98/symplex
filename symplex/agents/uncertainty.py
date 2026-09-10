"""Deterministic, typed inventory of what an investigation has not established.

Every item is read out of a stored artifact and cites the artifact ids it came
from. Nothing here is generated, inferred by a model, or guessed: this module
makes no network, model or subprocess calls and returns the same ledger for the
same stored records.

What an item establishes: a declared gap exists in the recorded contracts. What
it does not establish: that the gap is real in the world, that it is the most
important gap, that the assigned type is the scientifically correct reading, or
that the listed `resolvable_by` tools would in fact resolve it. Severity is a
host-declared ordering constant, not a measured probability or a cost of being
wrong. The classification is a host reading of agent-authored contracts.
"""

from symplex.core.contracts import Invalid, digest

VERSION = "derived-uncertainty-ledger-v1"

# The right next action differs per type, which is the only reason the ledger is
# typed at all. These labels describe the shape of the gap, never its importance.
TYPES = (
    "parametric",     # a quantity is declared but cannot be pinned down
    "structural",     # rival mechanisms are declared and not discriminated
    "observational",  # a declared state has no usable measurement channel
    "semantic",       # entity, unit, clock or join meaning is unresolved
    "numerical",      # a host check failed, or no host check result exists
    "evidential",     # a claim has no source, or recorded sources disagree
    "decision",       # endpoints, constraints or execution requirements undefined
)

MAX_ITEMS = 240
MAX_RECORDS_PER_KIND = 40
MAX_STATEMENT_CHARS = 600

# Declared plausibility only. This table says which tools a human would normally
# reach for; it is not a capability proof, and it is not permission to run them.
# `voi.score_actions` scores only tools the host has actually made available.
RESOLVING_TOOLS = {
    "parametric": ("run_model_code", "run_system_scenarios", "compare_computation", "research_evidence"),
    "structural": ("plan_experiment", "review_hypotheses", "compare_computation", "run_model_code"),
    "observational": ("discover_datasets", "profile_table", "research_evidence", "acquire_bamtwoogle", "ask_user"),
    "semantic": ("inspect_context", "profile_table", "inspect_system_graph", "represent_system", "ask_user"),
    "numerical": ("run_model_code", "compare_computation", "inspect_artifact"),
    "evidential": ("research_evidence", "search_literature", "synthesize_evidence", "search_artifacts", "fetch_artifact"),
    "decision": ("ask_user", "reframe_problem", "represent_system", "plan_experiment", "build_outcome"),
}

# Host-declared ordering weights in [0, 1]. Chosen so that a failed frozen host
# check outranks an agent-declared narrative gap, because the first is measured
# and the second is asserted. These are not probabilities.
SEVERITY = {
    "identifiability_gap": 0.6,
    "undiscriminated_rivals": 0.8,
    "hypothesis_not_identifiable": 0.8,
    "hypothesis_needs_reformulation": 0.5,
    "hypothesis_coverage_gap": 0.5,
    "observation_channel_unavailable": 0.6,
    "state_without_observation_channel": {
        "observed": 0.7, "latent": 0.6, "exogenous": 0.4, "decision": 0.3,
    },
    "event_indexed_time_scale": 0.4,
    "event_mapping_coupling": 0.5,
    "multiple_declared_clocks": 0.6,
    "no_decision_constraints": 0.5,
    "endpoint_without_available_channel": 0.7,
    "evidence_disagreement": 0.7,
    "evidence_gap": 0.5,
    "failed_host_check": 0.9,
    "host_checks_unavailable": 0.6,
    "unsupported_claim": 0.6,
    "external_validation_needed": 0.5,
    "blocked_protocol": 0.7,
}

SOURCE_KINDS = (
    "complex_system",
    "hypothesis_review",
    "evidence_synthesis",
    "experiment_comparison",
    "experiment_protocol",
    "model_critique",
    "decision_brief",
)

SCOPE = (
    "Derived inventory of declared gaps in stored artifacts. Item types, severities "
    "and resolvable_by tools are host-declared readings of agent-authored contracts, "
    "not measured quantities, probabilities, or scientific judgements. An empty ledger "
    "means no declared gap was found in the recorded artifacts, never that the "
    "investigation is complete or correct."
)


def _current(store, problem_id, kind, limit=MAX_RECORDS_PER_KIND):
    """Current, problem-scoped records of one kind, oldest first, bounded."""
    return [
        r for r in store.list(kind) if r["parent"] == problem_id and not r["stale"]
    ][-limit:]


def _text(value):
    return " ".join(str(value).split())[:MAX_STATEMENT_CHARS]


def _add(items, *, kind, rule, reference, statement, severity, blocking_for, derived_from, detail=None):
    """Record one uncertainty. Provenance is mandatory: an item without a stored
    artifact id would be a fabrication, so it is refused rather than emitted."""
    if kind not in TYPES:
        raise Invalid("Unknown uncertainty type: " + str(kind))
    statement = _text(statement)
    if not statement:
        raise Invalid("An uncertainty item requires a statement")
    sources = sorted({s for s in derived_from if isinstance(s, str) and s})
    if not sources:
        raise Invalid("Every ledger item must cite at least one stored artifact id")
    # Identity is the stable gap, not the artifact revision that revealed it, so a
    # re-proposed system does not make every open item look newly created.
    ident = "unc_" + digest(
        {"type": kind, "rule": rule, "reference": reference, "statement": statement}
    )[:20]
    existing = items.get(ident)
    if existing is not None:
        existing["derived_from"] = sorted(set(existing["derived_from"]) | set(sources))
        existing["severity"] = max(existing["severity"], round(float(severity), 4))
        return existing
    items[ident] = {
        "id": ident,
        "type": kind,
        "rule": rule,
        "reference": reference,
        "statement": statement,
        "blocking_for": blocking_for,
        "resolvable_by": list(RESOLVING_TOOLS[kind]),
        "severity": round(float(severity), 4),
        "derived_from": sources,
        "detail": detail or {},
    }
    return items[ident]


def _blocking_for_state(decision, state_id):
    targets = ["endpoint:" + e["id"] for e in decision.get("endpoints", []) if e.get("state_id") == state_id]
    targets += ["constraint:" + c["id"] for c in decision.get("constraints", []) if state_id in (c.get("state_ids") or [])]
    targets += [
        "alternative:" + a["id"]
        for a in decision.get("alternatives", [])
        if any(i.get("state_id") == state_id for i in (a.get("interventions") or []))
    ]
    return "; ".join(sorted(set(targets))) if targets else "decision_frame"


def _blocking_for_hypotheses(system_data, hypothesis_ids):
    wanted = set(hypothesis_ids)
    targets = [
        "experiment:" + e["id"]
        for e in system_data.get("experiments", [])
        if wanted & set(e.get("hypothesis_ids") or [])
    ]
    return "; ".join(sorted(set(targets))) if targets else "decision_frame"


def _discriminating_comparisons(store, problem_id, system_data):
    """Hypothesis sets covered by a completed comparison whose frozen host checks
    all passed and which met its own declared comparison. This is a record of
    completed procedure, not evidence that a mechanism was ruled out."""
    protocols = {r["id"]: r for r in _current(store, problem_id, "experiment_protocol")}
    experiments = {e["id"]: e for e in system_data.get("experiments", [])}
    covered = []
    for comparison in _current(store, problem_id, "experiment_comparison"):
        data = comparison["data"]
        verification = data.get("numerical_verification") or {}
        if verification.get("all_passed") is not True:
            continue
        if not any(c.get("status") == "meets_declared_comparison" for c in (data.get("comparisons") or [])):
            continue
        protocol = protocols.get(data.get("protocol_id"))
        if protocol is None:
            continue
        experiment = experiments.get(protocol["data"].get("experiment_id"))
        if experiment is None:
            continue
        covered.append((comparison["id"], frozenset(experiment.get("hypothesis_ids") or [])))
    return covered


def _mine_system(store, problem_id, system, items):
    data = system["data"]
    sid = system["id"]
    decision = data.get("decision") or {}
    validation = data.get("validation") or {}

    for index, gap in enumerate(validation.get("identifiability_gaps") or []):
        # Declared identifiability gaps say a quantity cannot be pinned down from
        # available data, which is a parametric gap even when the cause is design.
        _add(items, kind="parametric", rule="identifiability_gap",
             reference="validation.identifiability_gaps[%d]" % index,
             statement="Declared identifiability gap: " + _text(gap),
             severity=SEVERITY["identifiability_gap"], blocking_for="decision_frame",
             derived_from=[sid])

    covered = _discriminating_comparisons(store, problem_id, data)
    hypotheses = {h["id"]: h for h in data.get("hypotheses") or []}
    seen_pairs = set()
    for hid, hypothesis in hypotheses.items():
        for rival in hypothesis.get("rivals") or []:
            pair = frozenset((hid, rival))
            if len(pair) != 2 or pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            discriminating = [ident for ident, hyps in covered if pair <= hyps]
            if discriminating:
                continue
            names = sorted(pair)
            _add(items, kind="structural", rule="undiscriminated_rivals",
                 reference="hypotheses:" + "|".join(names),
                 statement="Rival mechanisms '%s' and '%s' are declared rivals and no completed frozen comparison discriminates them." % (names[0], names[1]),
                 severity=SEVERITY["undiscriminated_rivals"],
                 blocking_for=_blocking_for_hypotheses(data, names),
                 derived_from=[sid], detail={"hypothesis_ids": names})

    measured_states = set()
    for channel in data.get("observations") or []:
        if channel.get("status") == "available":
            measured_states.update(channel.get("state_ids") or [])
        else:
            _add(items, kind="observational", rule="observation_channel_unavailable",
                 reference="observations:" + str(channel.get("id")),
                 statement="Observation channel '%s' is declared %s, so the states it covers have no usable measurement." % (channel.get("id"), channel.get("status")),
                 severity=SEVERITY["observation_channel_unavailable"],
                 blocking_for="; ".join(sorted({_blocking_for_state(decision, s) for s in (channel.get("state_ids") or [])})) or "decision_frame",
                 derived_from=[sid],
                 detail={"channel_id": channel.get("id"), "status": channel.get("status"),
                         "state_ids": sorted(channel.get("state_ids") or [])})

    for state in data.get("states") or []:
        if state.get("id") in measured_states:
            continue
        weights = SEVERITY["state_without_observation_channel"]
        _add(items, kind="observational", rule="state_without_observation_channel",
             reference="states:" + str(state.get("id")),
             statement="Declared %s state '%s' has no available observation channel." % (state.get("kind"), state.get("id")),
             severity=weights.get(state.get("kind"), 0.5),
             blocking_for=_blocking_for_state(decision, state.get("id")),
             derived_from=[sid],
             detail={"state_id": state.get("id"), "state_kind": state.get("kind")})

    for scale in (data.get("boundary") or {}).get("scales") or []:
        if scale.get("step_seconds") is None:
            _add(items, kind="semantic", rule="event_indexed_time_scale",
                 reference="boundary.scales:" + str(scale.get("id")),
                 statement="Time scale '%s' declares no sampling interval, so alignment with other scales is an unresolved join." % scale.get("id"),
                 severity=SEVERITY["event_indexed_time_scale"], blocking_for="decision_frame",
                 derived_from=[sid], detail={"scale_id": scale.get("id")})
    clocks = sorted({s.get("clock") for s in (data.get("boundary") or {}).get("scales") or [] if s.get("clock")})
    if len(clocks) > 1:
        _add(items, kind="semantic", rule="multiple_declared_clocks",
             reference="boundary.scales.clocks",
             statement="The system declares %d distinct reference clocks; cross-clock meaning is resolved only by an explicit mapping component." % len(clocks),
             severity=SEVERITY["multiple_declared_clocks"], blocking_for="decision_frame",
             derived_from=[sid], detail={"clocks": clocks})
    for coupling in data.get("couplings") or []:
        if coupling.get("time_alignment") == "event_mapping":
            _add(items, kind="semantic", rule="event_mapping_coupling",
                 reference="couplings:" + str(coupling.get("id")),
                 statement="Coupling '%s' joins two ports by declared event mapping; the join rule is asserted, not checked." % coupling.get("id"),
                 severity=SEVERITY["event_mapping_coupling"], blocking_for="decision_frame",
                 derived_from=[sid], detail={"coupling_id": coupling.get("id")})

    if not (decision.get("constraints") or []):
        _add(items, kind="decision", rule="no_decision_constraints",
             reference="decision.constraints",
             statement="The decision frame declares no constraints, so no alternative can be ruled out on feasibility.",
             severity=SEVERITY["no_decision_constraints"], blocking_for="decision_frame",
             derived_from=[sid])
    for endpoint in decision.get("endpoints") or []:
        if endpoint.get("state_id") in measured_states:
            continue
        _add(items, kind="decision", rule="endpoint_without_available_channel",
             reference="decision.endpoints:" + str(endpoint.get("id")),
             statement="Endpoint '%s' scores state '%s', which has no available observation channel, so the endpoint cannot be evaluated as declared." % (endpoint.get("id"), endpoint.get("state_id")),
             severity=SEVERITY["endpoint_without_available_channel"],
             blocking_for="endpoint:" + str(endpoint.get("id")),
             derived_from=[sid],
             detail={"endpoint_id": endpoint.get("id"), "state_id": endpoint.get("state_id")})


def _mine_hypothesis_reviews(store, problem_id, items):
    for review in _current(store, problem_id, "hypothesis_review"):
        rid = review["id"]
        for assessment in review["data"].get("assessments") or []:
            status = assessment.get("status")
            if status == "not_identifiable_with_current_evidence":
                rule, severity = "hypothesis_not_identifiable", SEVERITY["hypothesis_not_identifiable"]
            elif status == "needs_reformulation":
                rule, severity = "hypothesis_needs_reformulation", SEVERITY["hypothesis_needs_reformulation"]
            else:
                continue
            _add(items, kind="structural", rule=rule,
                 reference="hypothesis:" + str(assessment.get("hypothesis_id")),
                 statement="Reviewed hypothesis '%s' is %s: %s" % (assessment.get("hypothesis_id"), status, _text(assessment.get("issue", ""))),
                 severity=severity, blocking_for="decision_frame", derived_from=[rid],
                 detail={"hypothesis_id": assessment.get("hypothesis_id"), "status": status})
        for index, gap in enumerate(review["data"].get("coverage_gaps") or []):
            _add(items, kind="structural", rule="hypothesis_coverage_gap",
                 reference="coverage_gaps[%d]:%s" % (index, digest(_text(gap))[:8]),
                 statement="Declared hypothesis coverage gap: " + _text(gap),
                 severity=SEVERITY["hypothesis_coverage_gap"], blocking_for="decision_frame",
                 derived_from=[rid])


def _mine_evidence(store, problem_id, items):
    for synthesis in _current(store, problem_id, "evidence_synthesis"):
        sid = synthesis["id"]
        for index, disagreement in enumerate(synthesis["data"].get("disagreements") or []):
            _add(items, kind="evidential", rule="evidence_disagreement",
                 reference="disagreement:" + digest(sorted(disagreement.get("claim_ids") or []) or [str(index)])[:12],
                 statement="Recorded sources disagree: " + _text(disagreement.get("explanation", "")),
                 severity=SEVERITY["evidence_disagreement"], blocking_for="decision_frame",
                 derived_from=[sid],
                 detail={"claim_ids": sorted(disagreement.get("claim_ids") or []),
                         "hypothesis_ids": sorted(disagreement.get("hypothesis_ids") or [])})
        for gap in synthesis["data"].get("gaps") or []:
            _add(items, kind="evidential", rule="evidence_gap",
                 reference="evidence_gap:" + str(gap.get("id")),
                 statement="Declared discriminating evidence gap: " + _text(gap.get("missing_information", "")),
                 severity=SEVERITY["evidence_gap"], blocking_for="decision_frame",
                 derived_from=[sid],
                 detail={"gap_id": gap.get("id"), "hypothesis_ids": sorted(gap.get("hypothesis_ids") or [])})

    for critique in _current(store, problem_id, "model_critique"):
        for index, claim in enumerate(critique["data"].get("unsupported_claims") or []):
            _add(items, kind="evidential", rule="unsupported_claim",
                 reference="unsupported_claim:" + digest(_text(claim))[:12],
                 statement="Review found an unsupported claim: " + _text(claim),
                 severity=SEVERITY["unsupported_claim"], blocking_for="decision_frame",
                 derived_from=[critique["id"]])

    for brief in _current(store, problem_id, "decision_brief"):
        for requirement in brief["data"].get("external_validation_needed") or []:
            _add(items, kind="evidential", rule="external_validation_needed",
                 reference="external_validation:" + digest(_text(requirement))[:12],
                 statement="Decision brief requires external validation: " + _text(requirement),
                 severity=SEVERITY["external_validation_needed"],
                 blocking_for="decision:" + _text(brief["data"].get("title", "brief"))[:80],
                 derived_from=[brief["id"]])


def _mine_numerical(store, problem_id, items):
    for comparison in _current(store, problem_id, "experiment_comparison"):
        cid = comparison["id"]
        verification = comparison["data"].get("numerical_verification") or {}
        for check_id in sorted(verification.get("failed_check_ids") or []):
            _add(items, kind="numerical", rule="failed_host_check",
                 reference="host_check:" + str(check_id),
                 statement="Frozen host numerical check '%s' failed on the recorded execution output." % check_id,
                 severity=SEVERITY["failed_host_check"],
                 blocking_for="experiment_protocol:" + str(comparison["data"].get("protocol_id")),
                 derived_from=[cid],
                 detail={"check_id": check_id, "protocol_id": comparison["data"].get("protocol_id")})
        if verification and verification.get("status") not in ("checked", "failed"):
            _add(items, kind="numerical", rule="host_checks_unavailable",
                 reference="host_checks:" + str(comparison["data"].get("protocol_id")),
                 statement="No host numerical check result is available for this comparison (status %s)." % verification.get("status"),
                 severity=SEVERITY["host_checks_unavailable"],
                 blocking_for="experiment_protocol:" + str(comparison["data"].get("protocol_id")),
                 derived_from=[cid], detail={"status": verification.get("status")})


def _mine_blocked_protocols(store, problem_id, items):
    for protocol in _current(store, problem_id, "experiment_protocol"):
        data = protocol["data"]
        if data.get("execution_readiness") != "blocked" and data.get("status") != "blocked_design":
            continue
        for reason in data.get("blocking_reasons") or []:
            _add(items, kind="decision", rule="blocked_protocol",
                 reference="blocking_reason:" + digest(_text(reason))[:12],
                 statement="Experiment protocol is blocked: " + _text(reason),
                 severity=SEVERITY["blocked_protocol"],
                 blocking_for="experiment:" + str(data.get("experiment_id")),
                 derived_from=[protocol["id"]],
                 detail={"experiment_id": data.get("experiment_id")})


def uncertainty_ledger(store, problem_id, *, max_items=MAX_ITEMS):
    """Typed inventory of declared open uncertainties for one current problem.

    Deterministic: the same stored records always produce the same ledger and the
    same `ledger_digest`, so change between two points in an investigation is
    measurable rather than asserted.
    """
    if not isinstance(problem_id, str) or not problem_id:
        raise Invalid("problem_id must be a nonempty string")
    if type(max_items) is not int or not 1 <= max_items <= MAX_ITEMS:
        raise Invalid("max_items must be an integer from 1 to " + str(MAX_ITEMS))
    try:
        problem = store.get(problem_id)
    except KeyError:
        raise Invalid("Unknown problem record") from None
    if problem["kind"] != "workspace_problem" or problem["stale"]:
        raise Invalid("Select a current problem statement")

    items = {}
    scanned = {}
    systems = _current(store, problem_id, "complex_system")
    scanned["complex_system"] = len(systems)
    if systems:
        # Superseded representations are kept in storage but are not open gaps.
        _mine_system(store, problem_id, systems[-1], items)
    _mine_hypothesis_reviews(store, problem_id, items)
    _mine_evidence(store, problem_id, items)
    _mine_numerical(store, problem_id, items)
    _mine_blocked_protocols(store, problem_id, items)
    for kind in SOURCE_KINDS:
        scanned.setdefault(kind, len(_current(store, problem_id, kind)))

    ordered = sorted(items.values(), key=lambda i: (-i["severity"], i["type"], i["id"]))
    retained, omitted = ordered[:max_items], ordered[max_items:]
    counts, mass = {t: 0 for t in TYPES}, {t: 0.0 for t in TYPES}
    for item in retained:
        counts[item["type"]] += 1
        mass[item["type"]] = round(mass[item["type"]] + item["severity"], 6)
    ledger = {
        "version": VERSION,
        "problem_id": problem_id,
        "problem_digest": problem["digest"],
        "system_id": systems[-1]["id"] if systems else None,
        "items": retained,
        "item_ids": [i["id"] for i in retained],
        "counts_by_type": counts,
        "severity_mass_by_type": mass,
        "total_severity_mass": round(sum(mass.values()), 6),
        "derived_from_artifact_ids": sorted({a for i in retained for a in i["derived_from"]}),
        "sources_scanned": dict(sorted(scanned.items())),
        "truncated": bool(omitted),
        "omitted_item_count": len(omitted),
        "omitted_item_ids": [i["id"] for i in omitted],
        "severity_definition": "Host-declared ordering weight in [0,1]. Not a probability, a cost, or a measured effect size.",
        "resolvable_by_definition": "Declared plausible tools for this gap type. Not a capability proof and not permission to run them.",
        "scope": SCOPE,
    }
    ledger["ledger_digest"] = ledger_digest(ledger)
    return ledger


def ledger_digest(ledger):
    """Stable digest of ledger identity and weight, so drift over time is visible."""
    if not isinstance(ledger, dict) or not isinstance(ledger.get("items"), list):
        raise Invalid("Expected a ledger dict containing an items list")
    return digest({
        "version": VERSION,
        "items": sorted(
            ({"id": i["id"], "type": i["type"], "severity": i["severity"]} for i in ledger["items"]),
            key=lambda i: i["id"],
        ),
    })
