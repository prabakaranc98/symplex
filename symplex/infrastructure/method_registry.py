"""Operator-reviewed, problem-scoped method canaries with immutable rollback history.

The initial evaluator measures complexity-architect software contracts only.
Nothing here grants scientific validity or lets an agent approve its own method.
"""

import json
from contextvars import ContextVar
from functools import wraps
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from symplex.agents.prompts import prompt_overlay, snapshot
from symplex.core.contracts import Invalid, digest
from symplex.infrastructure.execution_scope import execution_lock

STAGES = ("development", "regression", "holdout")
GATE_VERSION = "staged-contract-canary-v1"
SCOPE = "Operator-reviewed instruction canary for named problems. Gates measure software-contract reliability, not scientific accuracy, task utility or general recursive improvement."
_VERSION = ContextVar("symplex_method_version", default=None)


class OperatorInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    actor: str = Field(min_length=1, max_length=120)
    reason: str = Field(min_length=1, max_length=2000)


class ShadowInput(OperatorInput):
    candidate_id: str
    problem_ids: list[str] = Field(min_length=1, max_length=20)


class ReviewInput(ShadowInput):
    decision: Literal["approve", "reject"]
    problem_ids: list[str] = Field(default_factory=list, max_length=20)
    report_ids: dict[str, str] = Field(default_factory=dict)
    execution_failure_limit: int = Field(default=1, ge=1, le=10)
    numerical_failure_limit: int = Field(default=1, ge=1, le=10)


class CanaryInput(OperatorInput):
    review_id: str


class RollbackInput(OperatorInput):
    deployment_id: str


def _record(store, ident, kind):
    try:
        record = store.get(ident)
    except KeyError:
        raise Invalid("Missing " + kind + " artifact") from None
    if record["kind"] != kind or record["stale"]:
        raise Invalid("Select a current " + kind + " artifact")
    return record


def _operator(body):
    if not body.actor.strip() or not body.reason.strip():
        raise Invalid("Operator identity and reason cannot be blank")


def _problems(store, ids):
    if len(ids) != len(set(ids)):
        raise Invalid("Canary problem IDs must be unique")
    for ident in ids:
        _record(store, ident, "workspace_problem")
    return ids


def _sealed(store, key):
    with store.db() as db:
        row = db.execute("SELECT value FROM seals WHERE key=?", (key,)).fetchone()
    if not row:
        raise Invalid("Replay has no protected exposure seal")
    return json.loads(row["value"])


def _verified_report(store, candidate, ident, stage):
    from symplex.evaluation.methods import ADAPTERS, CRITERIA, _task_input
    report = _record(store, ident, "method_evaluation_report")
    protocol = _record(store, report["parent"], "method_evaluation_protocol")
    spec, data = protocol["data"], report["data"]
    if (spec.get("candidate_id") != candidate["id"] or spec.get("candidate_digest") != candidate["digest"]
            or spec.get("stage", "development") != stage or data.get("stage", "development") != stage):
        raise Invalid("Report does not match this candidate and frozen stage")
    adapter = ADAPTERS.get(spec.get("role"))
    if (adapter is None or spec.get("criteria") != CRITERIA
            or spec.get("schema_digest") != digest(adapter.contract.json_schema())):
        raise Invalid("Replay criteria or contract changed")
    if data.get("matched_resources_verified") is not True:
        raise Invalid("Matched resource accounting is not established")
    settings = _record(store, data["settings_id"], "method_replay_settings")
    if settings["parent"] != protocol["id"]:
        raise Invalid("Report settings belong to another protocol")
    pairs = data.get("pairs", [])
    if not pairs or len(pairs) != data.get("task_count"):
        raise Invalid("Report needs complete paired task records")
    task_ids = [p["task_id"] for p in pairs]
    if len(task_ids) != len(set(task_ids)) or _sealed(store, "method_protocol_replay:" + protocol["id"])["task_ids"] != task_ids:
        raise Invalid("Report task set differs from the consumed protocol")
    run_ids, scores = [], {"parent": 0, "child": 0, "regressions": 0}
    for pair in pairs:
        task = _task_input(store, protocol, pair["task_id"])
        if task["fingerprint"] != pair["input_digest"]:
            raise Invalid("Replay inputs changed after evaluation")
        seal = _sealed(store, "method_task_exposure:" + task["fingerprint"])
        if seal != {"protocol_id": protocol["id"], "task_id": pair["task_id"]}:
            raise Invalid("Replay task was not exclusively consumed by this protocol")
        for arm in ("parent", "child"):
            evidence = pair[arm]
            run = _record(store, evidence["run_id"], "method_evaluation_run")
            episode = _record(store, run["parent"], "method_replay_episode")
            exposure = _record(store, episode["parent"], "method_task_exposure")
            if (exposure["parent"] != protocol["id"] or exposure["data"].get("task_id") != pair["task_id"]
                    or run["data"].get("task_id") != pair["task_id"] or run["data"].get("arm") != arm
                    or run["data"].get("settings_id") != settings["id"]
                    or evidence.get("resources_verified") is not True
                    or run["data"].get("resource_failure") is not False
                    or run["data"].get("unsettled_reservation_count") != 0):
                raise Invalid("Paired replay lineage or accounting is invalid")
            expected_prompts = spec["parent_manifest" if arm == "parent" else "child_manifest"]
            actual_prompts = run["data"].get("prompt_manifest", [])
            if {p["name"]: p["digest"] for p in actual_prompts} != {p["name"]: p["digest"] for p in expected_prompts}:
                raise Invalid("Replay prompt versions differ from the frozen arm")
            with store.db() as db:
                rows = db.execute("SELECT name,cap,used FROM scope_budget WHERE scope_id=?", (episode["data"]["scope_id"],)).fetchall()
                pending = db.execute("SELECT 1 FROM reservations r JOIN reservation_scopes s ON r.id=s.reservation_id WHERE s.scope_id=? AND r.settled=0", (episode["data"]["scope_id"],)).fetchone()
            caps = spec["caps_per_task_arm"]
            usage = run["data"].get("usage_or_reservations", {})
            if (pending or {r["name"]: r["cap"] for r in rows} != caps
                    or any(r["used"] > r["cap"] + 1e-9 or abs(r["used"] - usage.get(r["name"], -1)) > 1e-9 for r in rows)
                    or not all(usage.get(k, 0) > 0 for k in ("model_requests", "input_tokens", "output_tokens"))):
                raise Invalid("Replay ledger no longer matches its recorded equal allocation")
            passed = run["data"].get("contract_passed") is True
            if evidence.get("contract_passed") is not passed:
                raise Invalid("Paired result disagrees with its recorded run")
            if passed:
                adapter.validate(adapter.contract.parse(run["data"]["output"]), task["source_ids"])
            scores[arm] += passed
            run_ids.append(run["id"])
        scores["regressions"] += bool(pair["parent"]["contract_passed"] and not pair["child"]["contract_passed"])
    if (set(run_ids) != set(data.get("run_ids", [])) or scores["parent"] != data.get("parent_passes")
            or scores["child"] != data.get("child_passes") or scores["regressions"] != data.get("paired_regressions")):
        raise Invalid("Aggregate counts disagree with the paired evidence")
    compatibility = {k: spec[k] for k in ("candidate_digest", "role", "schema_digest", "parent_manifest", "child_manifest", "caps_per_task_arm", "request_settings")}
    compatibility["actual_settings"] = settings["data"]
    return report, protocol, compatibility


def gates(store, candidate_id, report_ids=None):
    candidate = _record(store, candidate_id, "method_candidate")
    protocols = [r for r in store.list("method_evaluation_protocol")
                 if r["parent"] == candidate_id or r["data"].get("candidate_digest") == candidate["digest"]]
    reports = store.list("method_evaluation_report")
    selected, stages, blockers, versions, verified = {}, {}, [], [], []
    supplied = report_ids or {}
    if set(supplied) - set(STAGES):
        blockers.append("Unknown evaluation stage")
    for stage in STAGES:
        # The first frozen protocol owns this stage; later favorable retries
        # cannot replace an earlier failed or interrupted protected evaluation.
        protocol = next((p for p in protocols if p["data"].get("stage", "development") == stage), None)
        choices = [r for r in reports if protocol and r["parent"] == protocol["id"]]
        ident = supplied.get(stage) or (choices[0]["id"] if choices else None)
        stages[stage] = {"status": "missing", "protocol_id": protocol["id"] if protocol else None, "report_id": ident}
        if not ident:
            blockers.append("Missing " + stage + " report")
            continue
        selected[stage] = ident
        try:
            if not protocol or not choices or ident != choices[0]["id"]:
                raise Invalid("Stage must use its first frozen protocol and completed report")
            report, checked_protocol, compatible = _verified_report(store, candidate, ident, stage)
            data = report["data"]
            stages[stage].update(status="verified", task_count=data["task_count"], parent_passes=data["parent_passes"], child_passes=data["child_passes"], paired_regressions=data["paired_regressions"], matched_resources_verified=True)
            if data["paired_regressions"] or data["child_passes"] < data["parent_passes"]:
                raise Invalid("Paired contract regression recorded")
            if stage == "holdout" and data["child_passes"] <= data["parent_passes"]:
                raise Invalid("Held-out contract gain is not established")
            if verified and checked_protocol["created"] <= verified[-1]["created"]:
                raise Invalid("Freeze each stage after the preceding stage report")
            verified.append(report)
            versions.append(digest(compatible))
        except (Invalid, KeyError, ValueError, TypeError) as exc:
            stages[stage].update(status="blocked", error=str(exc)[:1000])
            blockers.append(stage + ": " + str(exc)[:1000])
    if len(set(versions)) > 1:
        blockers.append("Stage candidate, prompt, schema or resource settings differ")
    return {"candidate_id": candidate_id, "gate_version": GATE_VERSION, "eligible": not blockers, "blockers": blockers,
            "stages": stages, "report_ids": selected, "scope": SCOPE}


def lab_status(store, candidate_id):
    result = gates(store, candidate_id)
    reviews = [r for r in store.list("method_operator_review") if r["parent"] == candidate_id]
    deployments = [r for r in store.list("method_deployment") if r["parent"] == candidate_id]
    rollbacks = [r for r in store.list("method_rollback") if r["data"].get("candidate_id") == candidate_id]
    rolled = {r["data"]["deployment_id"] for r in rollbacks}
    active = [r for r in deployments if r["id"] not in rolled and not r["stale"]]
    state = "canary" if any(r["data"]["mode"] == "canary" for r in active) else "shadow" if active else "approved" if reviews and reviews[-1]["data"]["status"] == "approved" else "rejected" if reviews and reviews[-1]["data"]["status"] == "rejected" else "ready_for_review" if result["eligible"] else "blocked"
    return {**result, "state": state, "reviews": reviews, "deployments": [dict(r, data={**r["data"], "status": "rolled_back" if r["id"] in rolled else "active"}) for r in deployments], "rollbacks": rollbacks,
            "observations": [r for r in store.list("method_canary_observation") if r["data"].get("candidate_id") == candidate_id]}


def operator_review(store, body):
    body = ReviewInput.model_validate(body)
    _operator(body)
    _problems(store, body.problem_ids)
    if body.decision == "approve" and not body.problem_ids:
        raise Invalid("Approval requires an explicit nonempty problem allowlist")
    with execution_lock(store, "method-registry.lock", blocking=True, reentrant=True):
        evaluation = gates(store, body.candidate_id, body.report_ids)
        status = "rejected" if body.decision == "reject" else "approved" if evaluation["eligible"] else "blocked"
        candidate = _record(store, body.candidate_id, "method_candidate")
        data = {**body.model_dump(), "report_ids": evaluation["report_ids"], "candidate_digest": candidate["digest"], "status": status, "gates": evaluation, "actor_basis": "operator_supplied_identity", "scope": SCOPE}
        data["report_digests"] = {}
        for stage, report_id in evaluation["report_ids"].items():
            try:
                data["report_digests"][stage] = store.get(report_id)["digest"]
            except (KeyError, Invalid):
                data["report_digests"][stage] = None
        ident = store.put("method_operator_review", data, body.candidate_id)
        if status == "rejected":
            for deployment in _active_deployments(store):
                if deployment["parent"] == body.candidate_id:
                    _rollback(store, deployment, body.actor, body.reason, "operator_rejection")
        return {"id": ident, "status": status, "gates": evaluation}


def _active_deployments(store):
    rolled = {r["data"]["deployment_id"] for r in store.list("method_rollback")}
    return [r for r in store.list("method_deployment") if r["id"] not in rolled and not r["stale"]]


def shadow(store, body):
    body = ShadowInput.model_validate(body)
    _operator(body)
    _problems(store, body.problem_ids)
    candidate = _record(store, body.candidate_id, "method_candidate")
    ident = store.put("method_deployment", {**body.model_dump(), "candidate_digest": candidate["digest"], "mode": "shadow", "affects_solver": False, "scope": "Recorded candidate designation only; baseline execution is unchanged and no shadow result is implied."}, candidate["id"])
    return {"id": ident, "status": "shadow", "affects_solver": False}


def activate_canary(store, body):
    body = CanaryInput.model_validate(body)
    _operator(body)
    with execution_lock(store, "method-registry.lock", blocking=True, reentrant=True):
        review = _record(store, body.review_id, "method_operator_review")
        candidate_id = review["parent"]
        latest = [r for r in store.list("method_operator_review") if r["parent"] == candidate_id][-1]
        if (review["data"]["status"] != "approved" or latest["id"] != review["id"]
                or review["data"]["gates"].get("gate_version") != GATE_VERSION):
            raise Invalid("Canary requires the latest explicit approved operator review")
        evaluation = gates(store, candidate_id, review["data"]["report_ids"])
        if not evaluation["eligible"] or any(store.get(ident)["digest"] != review["data"]["report_digests"][stage] for stage, ident in evaluation["report_ids"].items()):
            raise Invalid("Reviewed staged evidence no longer passes the fixed gates")
        problems = _problems(store, review["data"]["problem_ids"])
        for existing in _active_deployments(store):
            if existing["data"]["mode"] == "canary" and set(problems) & set(existing["data"]["problem_ids"]):
                raise Invalid("A canary already owns one of the approved problems")
        protocol = store.get(store.get(evaluation["report_ids"]["holdout"])["parent"])
        spec = protocol["data"]
        if {k: digest(v) for k, v in snapshot().items()} != {p["name"]: p["digest"] for p in spec["parent_manifest"]}:
            raise Invalid("Baseline prompts changed since this candidate was evaluated")
        data = {**body.model_dump(), "candidate_id": candidate_id, "candidate_digest": review["data"]["candidate_digest"], "mode": "canary", "problem_ids": problems,
                "review_digest": review["digest"], "report_ids": evaluation["report_ids"], "protocol_id": protocol["id"], "protocol_digest": protocol["digest"],
                "parent_prompts": spec["parent_prompts"], "overrides": spec["child_overrides"],
                "execution_failure_limit": review["data"]["execution_failure_limit"], "numerical_failure_limit": review["data"]["numerical_failure_limit"], "scope": SCOPE}
        ident = store.put("method_deployment", data, candidate_id)
        return {"id": ident, "status": "canary", "problem_ids": problems}


def _rollback(store, deployment, actor, reason, trigger):
    existing = [r for r in store.list("method_rollback") if r["data"]["deployment_id"] == deployment["id"]]
    if existing:
        return existing[-1]["id"]
    return store.put("method_rollback", {"deployment_id": deployment["id"], "candidate_id": deployment["parent"], "actor": actor, "reason": reason, "trigger": trigger, "restores": "baseline_on_new_invocations", "scope": SCOPE}, deployment["id"])


def rollback(store, body):
    body = RollbackInput.model_validate(body)
    _operator(body)
    with execution_lock(store, "method-registry.lock", blocking=True, reentrant=True):
        deployment = _record(store, body.deployment_id, "method_deployment")
        ident = _rollback(store, deployment, body.actor, body.reason, "operator")
        return {"id": ident, "status": "rolled_back", "deployment_id": deployment["id"]}


def method_version():
    return _VERSION.get()


def _selected_canary(store, problem_id):
    active = [r for r in _active_deployments(store) if r["data"]["mode"] == "canary" and problem_id in r["data"]["problem_ids"]]
    if len(active) > 1:
        raise Invalid("Conflicting canary assignments")
    if not active:
        return None
    deployment = active[0]
    data = deployment["data"]
    try:
        candidate = _record(store, deployment["parent"], "method_candidate")
        review = _record(store, data["review_id"], "method_operator_review")
        protocol = _record(store, data["protocol_id"], "method_evaluation_protocol")
        if (candidate["digest"] != data["candidate_digest"] or review["digest"] != data["review_digest"]
                or protocol["digest"] != data["protocol_digest"]
                or review["data"]["gates"].get("gate_version") != GATE_VERSION
                or snapshot() != data["parent_prompts"] or not gates(store, candidate["id"], data["report_ids"])["eligible"]):
            raise Invalid("Canary parent or approved evidence changed")
    except (Invalid, KeyError) as exc:
        _rollback(store, deployment, "host", str(exc), "lineage_or_evidence_changed")
        return None
    return deployment


def observe_canary(store, deployment, problem_id, result, assessment_ids):
    """Only host call sites supply observed job and assessment records."""
    with execution_lock(store, "method-registry.lock", blocking=True, reentrant=True):
        job_id = result.get("id")
        manifests = [r for r in store.list("run_manifest") if r["parent"] == problem_id
                     and r["data"].get("job_id") == job_id
                     and (r["data"].get("method_version") or {}).get("deployment_id") == deployment["id"]]
        if not manifests:
            return
        event_key = manifests[-1]["id"]
        if not job_id or any(r["parent"] == deployment["id"] and r["data"].get("event_key") == event_key for r in store.list("method_canary_observation")):
            return
        failures = [ident for ident in assessment_ids if store.get(ident)["data"].get("status") == "failed"]
        store.put("method_canary_observation", {"candidate_id": deployment["parent"], "problem_id": problem_id, "job_id": job_id, "event_key": event_key, "invocation_manifest_id": event_key,
                  "job_status": result["status"], "execution_failed": result["status"] == "failed", "numerical_failure_ids": failures,
                  "scope": "Job execution failures and failed host assessments only; no scientific correctness score."}, deployment["id"])
        rows = [r["data"] for r in store.list("method_canary_observation") if r["parent"] == deployment["id"]]
        if (len({r["invocation_manifest_id"] for r in rows if r["execution_failed"]}) >= deployment["data"]["execution_failure_limit"]
                or len({ident for r in rows for ident in r["numerical_failure_ids"]}) >= deployment["data"]["numerical_failure_limit"]):
            _rollback(store, deployment, "host", "Declared canary failure threshold reached", "failure_threshold")


def governed_method(function):
    @wraps(function)
    def invoke(store, budget, provider, problem_id, *args, **kwargs):
        with execution_lock(store, "method-registry.lock", blocking=True, reentrant=True):
            deployment = _selected_canary(store, problem_id)
        version = {"deployment_id": deployment["id"], "digest": deployment["digest"], "candidate_id": deployment["parent"], "mode": "canary"} if deployment else None
        token = _VERSION.set(version)
        before = {r["id"] for r in store.list("execution_assessment") if r["parent"] == problem_id}
        try:
            if deployment:
                with prompt_overlay(deployment["data"]["overrides"], base=deployment["data"]["parent_prompts"]):
                    result = function(store, budget, provider, problem_id, *args, **kwargs)
                new_assessments = [r["id"] for r in store.list("execution_assessment") if r["parent"] == problem_id and r["id"] not in before]
                observe_canary(store, deployment, problem_id, result, new_assessments)
                return result
            return function(store, budget, provider, problem_id, *args, **kwargs)
        finally:
            _VERSION.reset(token)
    return invoke
