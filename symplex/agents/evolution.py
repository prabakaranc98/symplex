"""One host orchestrator: freeze → evolve → investigate → confirm → deliver."""

import difflib
import platform

from symplex.core.contracts import (
    ActionProposal,
    Invalid,
    ProblemDNA,
    ProgramPatch,
    canonical,
    digest,
    record,
)
from symplex.evaluation.metrics import PROTOCOL, PROTOCOL_DIGEST, paired, score
from symplex.evidence.admission import admit, prediction_rows, relations
from symplex.infrastructure.runner import Cancelled, execute
from symplex.infrastructure.storage import BudgetExhausted
from symplex.modeling.templates import DEFAULT_PROGRAM, compile_patch


class Engine:
    def __init__(self, store, budget, provider):
        self.store, self.budget, self.provider = store, budget, provider

    def investigate(
        self, data, cancel=None, assumption="Curator-verified event semantics"
    ):
        summary = admit(data)
        identity = digest(
            {
                "data": summary["digest"],
                "protocol": PROTOCOL_DIGEST,
                "mode": self.provider.mode,
                "assumption": assumption,
                "version": "engine-v1",
            }
        )
        job, fresh = self.store.start_job(identity)
        if not fresh:
            return job
        root = self.store.put(
            "problem",
            {
                "question": "Do linked markets produce more reliable forecasts?",
                "domain": "P1",
                "assumption": assumption,
                "dataset_digest": summary["digest"],
                "synthetic": data["synthetic"],
                "mode": self.provider.mode,
                "job_id": job["id"],
            },
        )
        self.store.job(job["id"], "validated")
        try:
            self.store.job(job["id"], "running")
            if cancel and cancel.is_set():
                raise Cancelled("Cancelled before investigation")
            evidence_id = self.store.put("evidence", summary, root)
            dataset_id = self.store.put("dataset", data, root)
            protocol_id = self.store.put("protocol", PROTOCOL, root)
            training = [r for r in data["rows"] if r["split"] == "train"]
            development = [r for r in data["rows"] if r["split"] == "development"]
            confirmation = [r for r in data["rows"] if r["split"] == "confirmation"]
            # Confirmation observations and labels are never in model context or fitting inputs.
            dna = self.provider.propose(
                ProblemDNA,
                {
                    "summary": summary,
                    "question": "Forecast reliability audit",
                    "evidence_id": evidence_id,
                    "assumption": assumption,
                },
                root,
            )
            self.store.put("problem_dna", record(dna), root)
            diagnostics = relations(prediction_rows(training + development))
            self.store.put(
                "system",
                {
                    "entities": [
                        "event",
                        "contract",
                        "price_observation",
                        "settlement_definition",
                    ],
                    "graph": diagnostics,
                    "boundary": "Point-in-time price proxies; no trader-belief or causal identification",
                    "rivals": [
                        "Exact logical relations improve reliability",
                        "Staleness and incompatible settlement definitions explain apparent incoherence",
                    ],
                    "observation": "Declared timestamped prices",
                    "constraints": [
                        "0 <= p <= 1",
                        "p_high <= p_low for matching greater-than predicates",
                    ],
                },
                root,
            )
            programs, archive = [], {}

            def evaluate(program, name, parent=None, patch=None):
                if len(programs) >= 4:
                    raise BudgetExhausted(
                        "Episode program cap reached (including seeds)"
                    )
                program_id = self.store.put(
                    "program",
                    {
                        "name": name,
                        "genome": program,
                        "parent_program_id": parent,
                        "patch": record(patch) if patch else None,
                        "origin": self.provider.mode if patch else "trusted_seed",
                        "dataset_digest": summary["digest"],
                        "protocol_digest": PROTOCOL_DIGEST,
                    },
                    root,
                )
                try:
                    outputs = execute(
                        self.store,
                        self.budget,
                        program,
                        training,
                        prediction_rows(development),
                        cancel,
                    )
                    run_id = self.store.put(
                        "run",
                        {
                            "status": "succeeded",
                            "split": "development",
                            "program_id": program_id,
                            "dataset_id": dataset_id,
                            "protocol_id": protocol_id,
                            "outputs": outputs,
                            "python": platform.python_version(),
                        },
                        program_id,
                    )
                    metrics = score(development, outputs["predictions"])
                    evaluation_id = self.store.put(
                        "evaluation",
                        {
                            "split": "development",
                            "run_id": run_id,
                            "program_id": program_id,
                            **metrics,
                        },
                        run_id,
                    )
                    candidate = {
                        "id": program_id,
                        "name": name,
                        "genome": program,
                        "development": metrics,
                        "run_id": run_id,
                        "evaluation_id": evaluation_id,
                    }
                    programs.append(candidate)
                    cell = (
                        ("shared_driver" if program["projection"] else "independent")
                        + "/"
                        + (
                            "bias_or_staleness"
                            if program["calibrator"] or program["age_weight"]
                            else "simple"
                        )
                    )
                    if (
                        cell not in archive
                        or metrics["brier"] < archive[cell]["development"]["brier"]
                    ):
                        archive[cell] = candidate
                    return candidate
                except Exception as exc:
                    self.store.put(
                        "run",
                        {
                            "status": "failed",
                            "program_id": program_id,
                            "error": str(exc),
                        },
                        program_id,
                    )
                    raise

            raw = evaluate(dict(DEFAULT_PROGRAM), "Raw price proxy")
            constrained = evaluate(
                dict(DEFAULT_PROGRAM, projection=True), "Validated constraints"
            )
            parent = min(programs, key=lambda p: p["development"]["brier"])
            patch_context = {
                "parent_id": parent["id"],
                "parent_genome": parent["genome"],
                "evidence_ids": [evidence_id],
                "diagnostics": diagnostics,
                "development_metrics": parent["development"],
                "grammar": ["enable_projection", "fit_calibrator", "age_shrinkage"],
                "note": "Change one supported component. value is used only for age_shrinkage; 0 to 1. No arbitrary code.",
            }
            patch = None
            for attempt in range(2):
                try:
                    patch = self.provider.propose(ProgramPatch, patch_context, root)
                    if (
                        patch.parent_id != parent["id"]
                        or not set(patch.evidence_ids).issubset({evidence_id})
                        or not patch.evidence_ids
                    ):
                        raise Invalid("Unknown parent or evidence reference")
                    child = compile_patch(parent["genome"], patch)
                    diff = "\n".join(
                        difflib.unified_diff(
                            canonical(parent["genome"]).splitlines(),
                            canonical(child).splitlines(),
                            fromfile="parent",
                            tofile="child",
                        )
                    )
                    self.store.put(
                        "patch",
                        {
                            **record(patch),
                            "diff": diff,
                            "attempt": attempt + 1,
                            "execution": "typed template configuration; arbitrary generated code unexecuted",
                        },
                        root,
                    )
                    evaluate(child, "Evolved candidate", parent["id"], patch)
                    break
                except (Invalid, TimeoutError) as exc:
                    self.store.put(
                        "failed_mutation",
                        {
                            "attempt": attempt + 1,
                            "error": str(exc),
                            "patch": record(patch) if patch else None,
                        },
                        root,
                    )
                    patch_context["repair_error"] = str(exc)
            selected = min(programs, key=lambda p: p["development"]["brier"])
            action_context = {
                "target_ids": [selected["id"]],
                "selected": selected,
                "diagnostics": diagnostics,
                "available_preconditions": [
                    "definitions_available",
                    "development_available",
                ],
                "available_actions": [
                    "inspect_definitions",
                    "ablate_projection",
                    "mutate_program",
                    "stop",
                ],
                "tool_plan_rule": "Use exactly one tool name matching action_type, or an empty list for stop.",
                "budget": self.budget.snapshot(),
            }
            action = self.provider.propose(ActionProposal, action_context, root)
            if not set(action.target_ids).issubset({p["id"] for p in programs}):
                raise Invalid("Unknown research action target")
            if not set(action.preconditions).issubset(
                {"definitions_available", "development_available"}
            ):
                raise Invalid("Research action preconditions unavailable")
            if action.tool_plan != (
                [] if action.action_type == "stop" else [action.action_type]
            ):
                raise Invalid("Research action tool plan is not executable")
            if cancel and cancel.is_set():
                raise Cancelled("Cancelled before research action")
            action_id = self.store.put("action", record(action), root)
            if action.action_type == "inspect_definitions":
                self.budget.reserve(evidence_requests=1)
                self.store.put(
                    "action_result",
                    {"status": "succeeded", "result": diagnostics},
                    action_id,
                )
            elif action.action_type == "ablate_projection":
                if not selected["genome"]["projection"]:
                    self.store.put(
                        "action_result",
                        {
                            "status": "unsupported_operation",
                            "reason": "Selected program has no projection to ablate",
                        },
                        action_id,
                    )
                else:
                    evaluate(
                        dict(selected["genome"], projection=False),
                        "Projection ablation",
                        selected["id"],
                    )
                    self.store.put(
                        "action_result",
                        {"status": "succeeded", "program_id": programs[-1]["id"]},
                        action_id,
                    )
            elif action.action_type == "mutate_program":
                next_context = dict(
                    patch_context,
                    parent_id=selected["id"],
                    parent_genome=selected["genome"],
                )
                extra_patch = self.provider.propose(ProgramPatch, next_context, root)
                try:
                    if extra_patch.parent_id != selected["id"] or not set(
                        extra_patch.evidence_ids
                    ).issubset({evidence_id}):
                        raise Invalid("Unknown mutation references")
                    evaluate(
                        compile_patch(selected["genome"], extra_patch),
                        "Second generation",
                        selected["id"],
                        extra_patch,
                    )
                    self.store.put(
                        "action_result",
                        {"status": "succeeded", "program_id": programs[-1]["id"]},
                        action_id,
                    )
                except Invalid as exc:
                    self.store.put(
                        "failed_mutation",
                        {"patch": record(extra_patch), "error": str(exc)},
                        action_id,
                    )
            else:
                self.store.put(
                    "action_result",
                    {"status": "succeeded", "result": "Stopped by research policy"},
                    action_id,
                )
            selected = min(programs, key=lambda p: p["development"]["brier"])
            self.store.put(
                "archive",
                {
                    "cells": {cell: p["id"] for cell, p in archive.items()},
                    "shape": [3, 3],
                    "interpretation": "Program diversity; not independent scientific confirmation",
                },
                root,
            )
            self.store.seal(
                "selection:" + identity,
                {"selected": selected["id"], "baseline": raw["id"]},
            )
            # Consume once, before accessing confirmation. Failures do not reopen the audit.
            self.store.seal(
                "confirmation:" + identity,
                {"program": selected["id"], "protocol": PROTOCOL_DIGEST},
            )
            confirm = {}
            for candidate in {
                raw["id"]: raw,
                constrained["id"]: constrained,
                selected["id"]: selected,
            }.values():
                output = execute(
                    self.store,
                    self.budget,
                    candidate["genome"],
                    training,
                    prediction_rows(confirmation),
                    cancel,
                )
                run_id = self.store.put(
                    "run",
                    {
                        "status": "succeeded",
                        "split": "confirmation",
                        "program_id": candidate["id"],
                        "dataset_id": dataset_id,
                        "protocol_id": protocol_id,
                        "outputs": output,
                    },
                    candidate["id"],
                )
                metrics = score(confirmation, output["predictions"])
                eval_id = self.store.put(
                    "evaluation",
                    {
                        "split": "confirmation",
                        "program_id": candidate["id"],
                        "run_id": run_id,
                        **metrics,
                    },
                    run_id,
                )
                confirm[candidate["id"]] = {
                    "metrics": metrics,
                    "evaluation_id": eval_id,
                    "run_id": run_id,
                }
            comparison = paired(
                confirm[raw["id"]]["metrics"], confirm[selected["id"]]["metrics"]
            )
            self.store.put(
                "method",
                {
                    "status": "not_executed",
                    "reason": "No separate policy validation/audit task allocation was supplied",
                    "mutable": ["diagnostic_first"],
                    "protected": [
                        "evaluator",
                        "splits",
                        "labels",
                        "budgets",
                        "permissions",
                    ],
                },
                root,
            )
            decision = {
                "title": "Forecast reliability audit",
                "status": "inconclusive",
                "synthetic": data["synthetic"],
                "mode": self.provider.mode,
                "selected_program_id": selected["id"],
                "baseline_program_id": raw["id"],
                "constrained_program_id": constrained["id"],
                "confirmation": confirm,
                "comparison": comparison,
                "recommended_action": "Inspect mismatched definitions and collect a prospective, independently reviewed forecast archive.",
                "best_alternative": "Retain the raw price proxy while collecting more evidence.",
                "outcome_range": None,
                "resources_required": "Independent event-definition review and prospective observations",
                "assumptions": dna.assumptions,
                "reversal_conditions": [
                    "Prospective loss reverses the observed ranking",
                    "Settlement definitions or first-availability metadata change",
                ],
                "unresolved_evidence": dna.missing_evidence,
                "scope": "Synthetic plumbing only"
                if data["synthetic"]
                else "Retrospective supplied-slice comparison; no trading, causal or broad forecasting claim",
                "budget": self.budget.snapshot(),
                "dataset_id": dataset_id,
                "problem_id": root,
                "programs": programs,
                "execution_completed": True,
            }
            result_id = self.store.put("decision", decision, root)
            self.store.job(job["id"], "succeeded", result_id)
            return {"id": job["id"], "status": "succeeded", "result_id": result_id}
        except Exception as exc:
            status = (
                "cancelled"
                if isinstance(exc, Cancelled)
                else "budget-exhausted"
                if isinstance(exc, BudgetExhausted)
                else "failed"
            )
            self.store.put("failure", {"status": status, "error": str(exc)}, root)
            self.store.job(job["id"], status, error=str(exc))
            return {"id": job["id"], "status": status, "error": str(exc)}
