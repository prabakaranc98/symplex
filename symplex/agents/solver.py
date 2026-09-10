"""Durable problem-solving agent with explicit tools and host-controlled execution."""

import json
from dataclasses import dataclass, field

from symplex.agents.policies import DEPTHS, POLICY_DIGEST
from symplex.agents.planner_context import bounded_planner_context
from symplex.agents.context import working_context, review_context
from symplex.agents.input_state import REFRESH_POLICY, input_delta, snapshot_inputs
from symplex.agents.prompts import frozen_prompts, manifest, prompt
from symplex.agents.tools import REGISTRY, ToolContext
from symplex.core.proposals import DeliveryReview
from symplex.core.contracts import TEXT, TEXTS, Invalid, ProblemDNA, digest, record, schema
from symplex.infrastructure.runner import Cancelled
from symplex.infrastructure.storage import BudgetExhausted
from symplex.infrastructure.execution_scope import scoped_solver
from symplex.infrastructure.method_registry import governed_method, method_version

TOOLS = REGISTRY.descriptions()


@dataclass(frozen=True)
class NextStep:
    tool: str
    target_id: str
    instruction: str
    uncertainty: str
    expected_change: str
    search_dimension: str = "experimental"
    supporting_artifact_ids: list[str] = field(default_factory=list)
    disconfirmation: str = "Not supplied by legacy caller"
    expected_resource_use: str = "Within the remaining host envelope"

    @classmethod
    def parse(cls, data):
        from symplex.core.contracts import closed, nonempty

        closed(data, cls.__annotations__)
        if data["tool"] not in TOOLS:
            raise Invalid("Agent requested an unavailable tool")
        for key in ("instruction", "uncertainty", "expected_change", "disconfirmation", "expected_resource_use"):
            nonempty(data[key])
        if data["search_dimension"] not in ("conceptual", "structural", "computational", "experimental", "evidence", "method"):
            raise Invalid("Unknown epistemic search dimension")
        from symplex.core.contracts import string_list
        string_list(data["supporting_artifact_ids"])
        if len(data["supporting_artifact_ids"]) > 12 or len(set(data["supporting_artifact_ids"])) != len(data["supporting_artifact_ids"]):
            raise Invalid("Provide at most 12 unique supporting artifacts")
        if not isinstance(data["target_id"], str):
            raise Invalid("target_id must be a string")
        return cls(**data)

    @staticmethod
    def json_schema():
        return schema(
            {
                "tool": {"type": "string", "enum": list(TOOLS)},
                "target_id": TEXT,
                "instruction": TEXT,
                "uncertainty": TEXT,
                "expected_change": TEXT,
                "search_dimension": {"type": "string", "enum": ["conceptual", "structural", "computational", "experimental", "evidence", "method"]},
                "supporting_artifact_ids": TEXTS,
                "disconfirmation": TEXT,
                "expected_resource_use": TEXT,
            }
        )



REVISION_TOOLS = {
    "design_solution", "run_system_scenarios", "run_model_code", "review_model",
    "represent_system", "develop_candidate", "build_outcome", "review_hypotheses",
    "synthesize_evidence", "plan_experiment", "compare_computation", "build_scene", "evolve_model",
    "reframe_problem",
}
REPEATABLE_TOOLS = {"research_evidence", "search_literature", "search_artifacts", "inspect_artifact", "fetch_artifact", "ask_user"}


def accepted_revision_count(tool, events):
    return sum(e["tool"] == tool and not (
        isinstance(e.get("result"), dict) and e["result"].get("status") == "rejected"
    ) for e in events)


def limit_reason(tool, events, envelope):
    count = sum(e["tool"] == tool for e in events)
    if tool == "deliver":
        return None
    if tool in REVISION_TOOLS and accepted_revision_count(tool, events) >= envelope["revisions"]:
        return "Accepted model revision cap reached"
    if tool not in REVISION_TOOLS | REPEATABLE_TOOLS and count:
        return "Repeated one-shot action"
    if tool in {"research_evidence", "search_literature"} and sum(
        e["tool"] in {"research_evidence", "search_literature"} for e in events
    ) >= envelope["research"]:
        return "Research action cap reached"
    return None

@scoped_solver
@governed_method
@frozen_prompts
def solve(store, budget, provider, problem_id, cancel=None, depth="balanced"):
    if depth not in DEPTHS:
        raise Invalid("Unknown investigation depth")
    from datetime import datetime, timezone
    runtime_utc = datetime.now(timezone.utc).isoformat()
    envelope = DEPTHS[depth]
    provider.reasoning_effort = envelope.get("reasoning_effort", "medium")
    problem = store.get(problem_id)
    if problem["kind"] != "workspace_problem" or problem["stale"]:
        raise Invalid("Select a current problem statement")
    # Adding context changes entitlement and creates a new investigation, preserving prior records.
    context_ids = [
        r["digest"]
        for r in store.list("context")
        if r["parent"] == problem_id and not r["stale"]
    ]
    context_ids += [
        r["digest"]
        for r in store.list("binary_context")
        if r["parent"] == problem_id and not r["stale"]
    ]
    base_identity = digest(
        {
            "problem": problem_id,
            "revision": problem["digest"],
            "context": context_ids,
            "depth": depth,
            "solver": "v2",
            **({"method_version": method_version()} if method_version() else {}),
        }
    )
    steering_ids = [
        r["id"]
        for r in store.list("steering")
        if r["parent"] == problem_id and not r["stale"]
    ]
    manifests = [
        r
        for r in store.list("run_manifest")
        if r["parent"] == problem_id and r["data"].get("base_identity") == base_identity
    ]
    prior_job = next(
        (
            j
            for j in store.jobs()
            if manifests and j["id"] == manifests[-1]["data"]["job_id"]
        ),
        None,
    )
    if prior_job and prior_job["status"] != "succeeded":
        job, fresh = prior_job, False
    elif (
        prior_job
        and prior_job.get("result_id")
        and store.get(prior_job["result_id"])["data"].get("steering_ids")
        == steering_ids
    ):
        return prior_job
    else:
        job, fresh = store.start_job(
            digest({"base": base_identity, "steering": steering_ids})
        )
    events, completed, result_ids = [], set(), []
    if not fresh:
        if job["status"] not in (
            "failed",
            "cancelled",
            "budget-exhausted",
            "waiting_user",
        ) or not job.get("result_id"):
            return job
        checkpoint_record = store.get(job["result_id"])
        if job["status"] == "waiting_user" and not any(
            r["parent"] == problem_id and r["created"] > checkpoint_record["created"]
            for r in store.list("steering")
        ):
            return job
        checkpoint = store.get(job["result_id"])["data"]
        events = checkpoint.get("completed_actions", [])
        completed = {e["tool"] for e in events}
        result_ids = checkpoint.get("artifact_ids", [])
        store.put(
            "resume",
            {
                "job_id": job["id"],
                "checkpoint_id": job["result_id"],
                "completed_steps": len(events),
            },
            problem_id,
        )
    store.job(job["id"], "running")
    store.put(
        "run_manifest",
        {
            "job_id": job["id"],
            "base_identity": base_identity,
            "budget_scope_id": getattr(budget, "scope_id", None),
            "method_version": method_version(),
            "steering_ids": steering_ids,
            "policy_digest": POLICY_DIGEST,
            "runtime_utc": runtime_utc,
            "prompts": manifest(),
            "depth": depth,
            "tools": list(TOOLS),
        },
        problem_id,
    )
    try:
        initial_inputs = snapshot_inputs(store, problem_id)
        prior_input_acks = [r for r in store.list("planner_input_ack")
                            if r["parent"] == problem_id and r["data"]["job_id"] == job["id"]]
        last_inputs = (store.get(prior_input_acks[-1]["data"]["input_revision_id"])["data"]
                       if prior_input_acks else None)
        prior_dna = [
            r
            for r in store.list("problem_dna")
            if r["parent"] == problem_id and not r["stale"]
        ]
        if not fresh and prior_dna:
            dna = ProblemDNA.parse(prior_dna[-1]["data"])
            prior_basis = [r for r in store.list("problem_dna_basis")
                           if r["parent"] == problem_id and r["data"]["problem_dna_id"] == prior_dna[-1]["id"]]
            dna_input_revision = prior_basis[-1]["data"]["input_revision"] if prior_basis else None
        else:
            dna = provider.propose(
                ProblemDNA,
                {
                    "problem": problem["data"],
                    "supplied_context": working_context(store, problem_id),
                    "runtime_utc": runtime_utc,
                    "instruction": "Establish the user's outcome, decision, actors, assumptions and missing evidence. The user authorizes this investigation and read-only evidence acquisition within the host envelope. A requested illustrative example may use explicit bounded assumptions without requiring approval of every modeling choice. Separate user goals from unverified evidence. Use runtime_utc as the clock, never invent a present cutoff. Keep fields concise; do not repeat general disclaimers in each actor.",
                },
                problem_id,
            )
            dna_id = store.put("problem_dna", record(dna), problem_id)
            dna_input_revision = initial_inputs["revision"]
            store.put("problem_dna_basis", {"problem_dna_id": dna_id,
                      "input_revision": dna_input_revision,
                      "scope": "Available inputs at framing; supplied context contains bounded excerpts."}, problem_id)
        acknowledged = {
            sid
            for r in store.list("steering_ack")
            if r["parent"] == problem_id
            for sid in r["data"]["steering_ids"]
        }
        for step in range(len(events), envelope["steps"]):
            if cancel and cancel.is_set():
                raise Cancelled("Problem solver cancelled between steps")
            current_inputs = snapshot_inputs(store, problem_id)
            input_revision_id = store.put("input_revision", current_inputs, problem_id)
            delta = input_delta(store, last_inputs, current_inputs)
            input_refresh = {
                "input_revision_id": input_revision_id,
                "input_revision": current_inputs["revision"],
                "input_delta": delta,
                "problem_dna_basis": {"input_revision": dna_input_revision,
                                      "basis_known": dna_input_revision is not None,
                                      "predates_current_inputs": dna_input_revision != current_inputs["revision"]},
                "input_refresh_policy": REFRESH_POLICY,
            }
            from symplex.agents.investigation import investigation_state
            search_state = investigation_state(store, problem_id, budget)
            state_id = store.put("investigation_state", search_state, problem_id)
            from symplex.agents.search_policy import build_search_archive
            archive = build_search_archive(store, problem_id)
            archive_id = store.put("search_archive", archive, problem_id)
            imported = [
                {"id": r["id"], "summary": r["data"]["summary"]}
                for r in store.list("imported_dataset")
                if not r["stale"]
            ]
            contexts = [
                {
                    "id": r["id"],
                    "title": r["data"]["title"],
                    "format": r["data"]["format"],
                }
                for r in store.list("context")
                if r["parent"] == problem_id and not r["stale"]
            ]
            steering = [
                {"id": r["id"], **r["data"]}
                for r in store.list("steering")
                if r["parent"] == problem_id and not r["stale"]
            ]
            presented_steering = {s["id"] for s in steering[-8:]} | {
                r["id"] for r in delta["excerpts"] if r["kind"] == "steering"
            }
            new_steering = [s["id"] for s in steering
                            if s["id"] not in acknowledged and s["id"] in presented_steering]
            action = provider.propose(
                NextStep,
                bounded_planner_context({
                    "problem": problem["data"],
                    "problem_dna": record(dna),
                    "runtime_utc": runtime_utc,
                    "user_context": contexts[-20:],
                    "working_context": working_context(store, problem_id),
                    "candidate_archive": search_state["candidates"][-20:],
                    "search_policy": {"archive_id": archive_id,
                        "suggested_parent_id": archive["suggested_parent_id"],
                        "selection_scope": archive["selection_scope"],
                        "evolution_constraint": "The archive hint covers all frozen protocols; evolve_model selects its parent within the latest current frozen protocol.",
                        "selection_rule": archive["selection_rule"],
                        "quality_definition": archive["quality_definition"],
                        "cells": [{k: cell[k] for k in ("id", "descriptor", "elite_package_id", "visit_count", "diagnostic_pass_fraction")} for cell in archive["cells"]][-12:],
                        "excluded_candidates": archive["excluded_candidates"][-8:]},
                    "user_steering": steering[-8:],
                    "completed_actions": events,
                    "available_tools": {t: d for t, d in TOOLS.items() if not limit_reason(t, events, envelope)},
                    "tool_counts": {
                        t: sum(e["tool"] == t for e in events) for t in TOOLS
                    },
                    "accepted_revision_counts": {t: accepted_revision_count(t, events) for t in REVISION_TOOLS},
                    "revision_accounting": "Host-rejected attempts still consume action steps, tool attempts and actual resources, but do not consume the accepted-revision allowance. An executed candidate with failed numerical checks is still a revision.",
                    "imported_datasets": imported,
                    "budget": budget.snapshot(),
                    "steps_remaining": envelope["steps"] - step,
                    "depth": depth,
                    "max_model_revisions": envelope["revisions"],
                    "instruction": prompt("metareasoner"),
                    **input_refresh,
                }),
                problem_id,
            )
            exhausted = limit_reason(action.tool, events, envelope)
            if step == envelope["steps"] - 1:
                exhausted = "Final permitted step"
            if exhausted and action.tool != "deliver":
                store.put(
                    "action_rejection",
                    {
                        "proposal": record(action),
                        "reason": exhausted,
                        "fallback": "Deliver recorded results with remaining dependencies",
                    },
                    problem_id,
                )
                if step < envelope["steps"] - 1:
                    reconsidered = provider.propose(NextStep, bounded_planner_context({
                        "problem": problem["data"], "working_context": working_context(store, problem_id),
                        "rejected_action": record(action), "host_feedback": exhausted,
                        "available_tools": {t: d for t, d in TOOLS.items() if not limit_reason(t, events, envelope)},
                        "budget": budget.snapshot(), "steps_remaining": envelope["steps"] - step,
                        "instruction": "Choose a different useful permitted action, or deliver. This is the one allowed replan after a tool limit; do not repeat the rejected action.",
                        **input_refresh,
                    }), problem_id)
                    if not limit_reason(reconsidered.tool, events, envelope):
                        action, exhausted = reconsidered, None
                if exhausted:
                    action = NextStep(
                        "deliver", "", "Deliver current state: " + exhausted,
                        "Unfinished dependencies", "Make actual results and limitations explicit",
                    )
            store.put("planner_input_ack", {
                "job_id": job["id"], "step": step + 1,
                "input_revision_id": input_revision_id, "input_revision": current_inputs["revision"],
                "delta_excerpt_ids": [r["id"] for r in delta["excerpts"]],
                "chosen_action": record(action),
                "status": "presented_to_planner; not proof of incorporation or validation",
            }, problem_id)
            last_inputs = current_inputs
            if new_steering:
                store.put(
                    "steering_ack",
                    {
                        "steering_ids": new_steering,
                        "job_id": job["id"],
                        "step": step + 1,
                        "chosen_action": record(action),
                    },
                    problem_id,
                )
                acknowledged.update(new_steering)
            aid = store.put(
                "agent_action",
                dict(
                    record(action), step=step + 1, status="running", role="metareasoner", investigation_state_id=state_id
                ),
                problem_id,
            )
            citation_error = None
            try:
                for ident in action.supporting_artifact_ids:
                    cited = store.get(ident)
                    if cited["parent"] != problem_id or cited["stale"] or cited["kind"] in ("dataset", "model_call"):
                        raise Invalid("Action cites unavailable evidence")
            except (Invalid, KeyError):
                citation_error = {
                    "status": "rejected", "error": "Action cites unavailable evidence",
                    "required_response": "Choose available problem-scoped evidence, revise the action, or deliver the recorded gap.",
                }
            if citation_error and step == envelope["steps"] - 1:
                store.put("agent_result", {"action_id": aid, "status": "rejected", "output": citation_error}, aid)
                events.append({"tool": action.tool, "artifact_id": aid, "result": citation_error})
                store.put("action_rejection", {
                    "proposal": record(action), "reason": citation_error["error"],
                    "fallback": "Final-step delivery uses recorded artifacts only; rejected citations and proposed claims are not adopted.",
                }, problem_id)
                action = NextStep("deliver", "", "Deliver recorded results and remaining evidence gaps; the proposed supporting citations were rejected.",
                                  "Unavailable supporting evidence", "Report the verified artifact inventory and limitations")
                aid = store.put("agent_action", dict(record(action), step=step + 1, status="running", role="host_fallback",
                                                     investigation_state_id=state_id), problem_id)
                citation_error = None
            if citation_error:
                output = citation_error
                resource_delta = {key: 0 for key in budget.snapshot()}
            elif action.tool != "deliver":
                before = budget.snapshot()
                try:
                    output = REGISTRY.execute(
                        action.tool,
                        ToolContext(
                            store,
                            budget,
                            provider,
                            problem_id,
                            cancel,
                            result_ids,
                            imported,
                        ),
                        action,
                        envelope["permissions"],
                    )
                    if output.get("problem_dna_id"):
                        reframed = store.get(output["problem_dna_id"])
                        basis = store.get(output["problem_dna_basis_id"])
                        if (reframed["kind"] != "problem_dna" or reframed["parent"] != problem_id
                                or reframed["stale"] or basis["kind"] != "problem_dna_basis"
                                or basis["parent"] != problem_id or basis["stale"]
                                or basis["data"]["problem_dna_id"] != reframed["id"]):
                            raise Invalid("Reframing output has invalid problem or input lineage")
                        dna = ProblemDNA.parse(reframed["data"])
                        dna_input_revision = basis["data"]["input_revision"]
                except Invalid as exc:
                    # A rejected proposal is search feedback, not permission to bypass a guard.
                    # Consumes the same action/tool cap; repeated failures cannot loop forever.
                    output = {"status": "rejected", "error": str(exc)[:2000],
                              "required_response": "Revise the proposal within the unchanged contracts, inspect evidence, or deliver the gap."}
                after = budget.snapshot()
                resource_delta = {k: after[k]["used_or_reserved"] - v["used_or_reserved"] for k, v in before.items()}
            else:
                artifacts = [store.get(i) for i in result_ids]
                current_artifacts = [a for a in artifacts if a["parent"] == problem_id and not a["stale"]]
                adapters = [a for a in current_artifacts if a["kind"] == "adapter_run"]
                review = provider.propose(
                    DeliveryReview,
                    {
                        "problem": problem["data"],
                        "artifact_ids": result_ids,
                        "executed_evidence": review_context(store, problem_id),
                        "auxiliary_adapter_results": [{"id": a["id"], "data": a["data"]} for a in adapters],
                        "user_steering": steering[-8:],
                        "instruction": "Check the proposed deliverable against actual recorded execution. Auxiliary adapter results have their own measured scope and do not establish validation of this problem. Identify unsupported claims and the next required validation. Do not modify scores.",
                        **input_refresh,
                    },
                    problem_id,
                    role="validator",
                )
                rid = store.put("review", review, problem_id)
                executed = [
                    a["id"]
                    for a in current_artifacts
                    if a["kind"] == "decision" and a["data"].get("execution_completed") is True
                    and a["data"].get("problem_id") == problem_id
                    and a["data"].get("synthetic") is False
                ]
                scenarios = [
                    a["id"] for a in current_artifacts if a["kind"] == "scenario_report"
                ]
                computations = [
                    a["id"] for a in current_artifacts if a["kind"] == "compute_package"
                ]
                deliverable = {
                    "title": problem["data"]["question"],
                    "status": "evaluated_within_scope"
                    if executed
                    else "conditional_scenarios_evaluated"
                    if scenarios
                    else "exploratory_computation_executed"
                    if computations
                    else "dependency_gap",
                    "problem_id": problem_id,
                    "input_revision_id": input_revision_id,
                    "input_revision": current_inputs["revision"],
                    "steering_ids": [s["id"] for s in steering],
                    "artifact_ids": result_ids,
                    "evaluated_decision_ids": executed,
                    "scenario_report_ids": scenarios,
                    "compute_package_ids": computations,
                    "auxiliary_adapter_run_ids": [a["id"] for a in adapters],
                    "review_id": rid,
                    "review": review,
                    "next_action": action.instruction,
                    "completed_actions": events,
                    "budget": budget.snapshot(),
                    "scope": "Only current problem-bound, nonsynthetic executed decision artifacts establish measured decision results. Auxiliary adapters retain their own scope. Designs and code drafts remain proposals.",
                }
                did = store.put("deliverable", deliverable, problem_id)
                store.put(
                    "agent_result",
                    {"action_id": aid, "status": "succeeded", "deliverable_id": did},
                    aid,
                )
                store.job(job["id"], "succeeded", did)
                return {"id": job["id"], "status": "succeeded", "result_id": did}
            store.put(
                "agent_result",
                {"action_id": aid, "status": "rejected" if output.get("status") == "rejected" else "succeeded", "output": output,
                 "resource_usage_or_reservations": resource_delta},
                aid,
            )
            events.append({"tool": action.tool, "artifact_id": aid, "result": output})
            # Keep full artifacts in storage and bounded summaries in next-step context.
            if len(json.dumps(events[-1])) > 7000:
                events[-1]["result"] = {
                    "summary": str(output)[:4500],
                    "full_result": aid,
                }
            completed.add(action.tool)
            if output.get("waiting_for_user"):
                checkpoint = store.put(
                    "solver_checkpoint",
                    {
                        "completed_actions": events,
                        "artifact_ids": result_ids,
                        "depth": depth,
                        "question_id": output["question_id"],
                    },
                    problem_id,
                )
                store.job(job["id"], "waiting_user", checkpoint)
                return {
                    "id": job["id"],
                    "status": "waiting_user",
                    "result_id": checkpoint,
                }

        raise BudgetExhausted("Solver exhausted its step cap")
    except Exception as exc:
        status = (
            "cancelled"
            if isinstance(exc, Cancelled)
            else "budget-exhausted"
            if isinstance(exc, BudgetExhausted)
            else "failed"
        )
        fid = store.put(
            "failure",
            {
                "status": status,
                "error": str(exc),
                "completed_actions": events,
                "artifact_ids": result_ids,
                "resume": "Inspect saved checkpoints; this version does not restart a consumed confirmation",
            },
            problem_id,
        )
        store.job(job["id"], status, fid, str(exc))
        return {"id": job["id"], "status": status, "error": str(exc), "result_id": fid}
