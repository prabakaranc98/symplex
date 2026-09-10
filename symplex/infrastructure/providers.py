"""Official OpenAI SDK, explicit role routing and persisted usage reservations."""

import json
import os
import time
import re

from symplex.agents.prompts import prompt
from symplex.core.contracts import Invalid, canonical, digest
from symplex.infrastructure.storage import BudgetExhausted
from symplex.infrastructure.limits import MAX_REQUEST_BYTES

# Standard, short-context upper input rate includes cache writes; verified 2026-09-10.
# https://developers.openai.com/api/docs/pricing . No cached-token discount assumed.
RATES = {
    "gpt-6-astra": (12.5, 50),
    "gpt-5.6-sol": (5, 20),
    "gpt-5.6-terra": (2.5, 12),
    "gpt-5.6-luna": (0.25, 1.2),
}
ROLES = {"chat": "gpt-5.6-luna", "heavy": "gpt-6-astra", "validator": "gpt-5.6-sol"}


def safe_error_text(value):
    """Keep useful provider diagnostics without recording credentials or headers."""
    text = str(value)
    secret = os.getenv("OPENAI_API_KEY")
    if secret:
        text = text.replace(secret, "[redacted]")
    return re.sub(r"sk-[A-Za-z0-9_-]+", "[redacted]", text)[:2000]


def api_schema(value):
    """Use the API-supported schema subset; all bounds remain enforced by parse()."""
    if isinstance(value, list):
        return [api_schema(v) for v in value]
    if not isinstance(value, dict):
        return value
    result = {k: api_schema(v) for k, v in value.items()}
    if "$ref" in result:
        if set(result) - {"$ref", "title", "description"}:
            raise Invalid(
                "Unsupported validation keywords beside an API schema reference"
            )
        return {"$ref": result["$ref"]}
    if result.get("type") == "string":
        # String-length bounds are host checks; use supported pattern/format remotely.
        bounds = []
        if "minLength" in result:
            bounds.append(f"at least {result['minLength']} characters")
        if "maxLength" in result:
            bounds.append(f"at most {result['maxLength']} characters")
        if bounds:
            result["description"] = (result.get("description", "") + " Host validation requires " + ", ".join(bounds) + ".").strip()
        for key in ("minLength", "maxLength"):
            result.pop(key, None)
    return result


class OpenAIProvider:
    mode = "live_openai"

    def __init__(self, store, budget, client=None):
        from openai import OpenAI

        self.store, self.budget = store, budget
        self.models = {
            r: os.getenv("OPENAI_" + r.upper() + "_MODEL", m) for r, m in ROLES.items()
        }
        self.client = client or OpenAI(max_retries=0, timeout=180)

    def capability(self):
        results = {}
        for role, model in self.models.items():
            reservation = self.budget.reserve(model_requests=1)
            try:
                self.client.models.retrieve(model)
                results[role] = {"model": model, "status": "accessible"}
            except Exception as e:
                results[role] = {
                    "model": model,
                    "status": "unavailable",
                    "error": type(e).__name__,
                    "http_status": getattr(e, "status_code", None),
                }
            self.budget.settle(reservation, model_requests=1)
        self.store.put("capability", results)
        return results

    def _call(self, role, kwargs, parent=None, search=False, compute=False):
        model = self.models[role]
        if model not in RATES:
            raise Invalid(
                "Configure verified pricing before using a model outside the supported registry"
            )
        kwargs = dict(kwargs)
        output_cap = kwargs.pop("max_output_tokens", 2400)
        kwargs = dict(
            kwargs,
            model=model,
            store=False,
            service_tier="default",
            max_output_tokens=output_cap,
        )
        kwargs.setdefault(
            "reasoning",
            {
                "effort": os.getenv(
                    "OPENAI_" + role.upper() + "_REASONING_EFFORT",
                    os.getenv(
                        "OPENAI_REASONING_EFFORT",
                        getattr(self, "reasoning_effort", "medium")
                        if role == "heavy"
                        else "low",
                    ),
                )
            },
        )
        # Hosted tool returns can add tokens beyond the submitted request. This is
        # an admission estimate, not a guaranteed upper bound on billed usage.
        # Official search_context_size is qualitative and cannot enforce a token cap:
        # https://developers.openai.com/api/docs/guides/tools-web-search#search-context-size
        request_size = len(canonical(kwargs).encode()) + 2048
        if request_size > MAX_REQUEST_BYTES:
            raise Invalid("Context exceeds the bounded investigation envelope")
        input_cap = request_size + (20000 if search or compute else 0)
        input_rate, output_rate = RATES[model]
        compute_reserve = 0.03 * kwargs.get("max_tool_calls", 1) if compute else 0
        max_usd = (
            (input_cap * input_rate + output_cap * output_rate) / 1e6
            + (0.02 if search else 0)
            + compute_reserve
        )
        reserved = self.budget.reserve(
            model_requests=1,
            input_tokens=input_cap,
            output_tokens=output_cap,
            usd=max_usd,
            **({"evidence_requests": 2} if search else {}),
        )
        started = time.monotonic()
        response_artifact_id = None
        phase = "provider_request"
        try:
            # Large typed models can spend several minutes generating a complete
            # representation. Keep a bounded timeout without discarding healthy
            # long-form work at the ordinary chat timeout.
            client = self.client.with_options(timeout=600) if compute else (
                self.client.with_options(timeout=420)
                if output_cap >= 6000 and hasattr(self.client, "with_options")
                else self.client
            )
            response = client.responses.create(**kwargs)
            usage = response.usage
            phase = "response_persistence"
            response_dump = response.model_dump()
            # Save the returned work and exact reported usage before accounting.
            # A local settlement error must never erase a paid provider response.
            response_artifact_id = self.store.put(
                "model_response",
                {
                    "role": role,
                    "model": response.model,
                    "response_id": response.id,
                    "status": response.status,
                    "usage": usage.model_dump() if usage else None,
                    "output": response_dump.get("output", []),
                    "incomplete_details": response_dump.get("incomplete_details"),
                    "reservation": reserved,
                    "request_digest": digest(kwargs),
                    "runtime_seconds": time.monotonic() - started,
                },
                parent,
            )
            phase = "usage_settlement"
            accounting = None
            if usage:
                actual_usd = (
                    usage.input_tokens * input_rate + usage.output_tokens * output_rate
                ) / 1e6
                searches = sum(
                    getattr(x, "type", "") == "web_search_call" for x in response.output
                )
                actual_usd += (
                    searches * 0.01 + compute_reserve
                )  # Retain conservative container charges; no reuse discount.
                accounting = self.budget.settle(
                    reserved,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    usd=actual_usd,
                    **({"evidence_requests": searches} if search else {}),
                )
            else:
                actual_usd = None  # Keep reservation when billing usage is unknown.
            phase = "call_persistence"
            self.store.put(
                "model_call",
                {
                    "role": role,
                    "model": response.model,
                    "response_id": response.id,
                    "response_artifact_id": response_artifact_id,
                    "incomplete_details": response_dump.get("incomplete_details"),
                    "status": response.status,
                    "usage": usage.model_dump() if usage else None,
                    "usd_upper_estimate": actual_usd,
                    "accounting": accounting,
                    "accounting_status": accounting["status"]
                    if accounting
                    else "usage_unavailable_reservation_retained",
                    "reservation_basis": "Submitted request size plus heuristic hosted-tool allowance; actual usage may exceed reservation",
                    "cost_basis": "Reported token counts at configured rates plus conservative hosted-tool estimates; not an invoice",
                    "runtime_seconds": time.monotonic() - started,
                    "request_digest": digest(kwargs),
                    "reasoning_effort": kwargs.get("reasoning", {}).get("effort"),
                    "output": response_dump.get("output", []),
                },
                parent,
            )
            if (
                search
                and response.status == "incomplete"
                and (response_dump.get("incomplete_details") or {}).get("reason")
                == "max_output_tokens"
                and getattr(response, "output_text", "").strip()
            ):
                # Read-only research may retain a clearly labeled partial source note.
                # Structured proposals and compute never take this path.
                return response
            if response.status != "completed":
                phase = "response_completion"
                incomplete = response_dump.get("incomplete_details") or {}
                reason = (
                    incomplete.get("reason", "not_reported")
                    if isinstance(incomplete, dict)
                    else "not_reported"
                )
                raise Invalid(
                    "Model response did not complete: "
                    + str(response.status)
                    + "; reason="
                    + safe_error_text(reason)
                    + "; requested output token limit="
                    + str(output_cap)
                )
            return response
        except Exception as exc:
            # Never persist request headers or exception text containing provider credentials.
            body = getattr(exc, "body", None)
            details = {}
            if isinstance(body, dict):
                error = body.get("error", body)
                if isinstance(error, dict):
                    for key in ("code", "param", "message"):
                        details[key] = safe_error_text(error.get(key, ""))
            if (
                response_artifact_id is None
                and getattr(exc, "status_code", None) == 400
                and details.get("code") == "invalid_function_parameters"
            ):
                # Schema validation rejected this request before model generation.
                try:
                    self.budget.settle(
                        reserved,
                        input_tokens=0,
                        output_tokens=0,
                        usd=0,
                        **({"evidence_requests": 0} if search else {}),
                    )
                    details["accounting"] = (
                        "Generation reservation released after explicit schema rejection; request still counted"
                    )
                except Exception as settlement_error:
                    details["accounting"] = (
                        "Reservation release failed: " + type(settlement_error).__name__
                    )
            failure_id = self.store.put(
                "model_failure",
                {
                    "role": role,
                    "model": model,
                    "error": type(exc).__name__,
                    "http_status": getattr(exc, "status_code", None),
                    "reservation": reserved,
                    "phase": phase,
                    "response_artifact_id": response_artifact_id,
                    "local_error": safe_error_text(exc)
                    if isinstance(exc, (Invalid, BudgetExhausted))
                    else None,
                    "runtime_seconds": time.monotonic() - started,
                    "provider_error": details,
                },
                parent,
            )
            raise Invalid(
                "Model call failed: "
                + type(exc).__name__
                + (
                    ": " + safe_error_text(exc)
                    if isinstance(exc, (Invalid, BudgetExhausted))
                    else ""
                )
                + "; failure artifact "
                + failure_id
            ) from None

    def propose(self, contract, context, parent=None, role="heavy"):
        from symplex.infrastructure.agents_sdk import invoke_proposal

        response = invoke_proposal(self, contract, context, parent, role)
        calls = [x for x in response.output if x.type == "function_call"]
        if len(calls) != 1 or calls[0].name != "submit_proposal":
            raise Invalid("Expected one recognized proposal tool call")
        value = json.loads(calls[0].arguments)
        try:
            return contract.parse(value)
        except (ValueError, Invalid) as exc:
            self.store.put(
                "proposal_rejection",
                {
                    "contract": contract.__name__,
                    "proposal": value,
                    "error": str(exc)[:4000],
                    "response_id": response.id,
                },
                parent,
            )
            # Large contracts use a bounded leaf patch, not another copy of the
            # complete schema, context and generated proposal in one request.
            from symplex.core.repair import ProposalRepair, apply_repair

            if context.get("schema_repair") or contract is ProposalRepair:
                raise
            if len(canonical(value).encode("utf-8")) > 12000:
                patch = self.propose(
                    ProposalRepair,
                    {
                        "schema_repair": True,
                        "contract": contract.__name__,
                        "rejected_proposal": value,
                        "validation_error": str(exc)[:2500],
                        "instruction": "Correct only the stated contract inconsistency with existing-leaf JSON Pointer replacements. Preserve the scientific proposal and source references. Do not invent evidence, change evaluation rules or rewrite containers. The complete original contract will be validated again; if no justified leaf repair exists, return an empty operations list.",
                    },
                    parent,
                    role,
                )
                repaired = apply_repair(value, patch)
                from symplex.core.contracts import record

                repair_id = self.store.put(
                    "proposal_repair",
                    {
                        "contract": contract.__name__,
                        "original_response_id": response.id,
                        "original_digest": digest(value),
                        "repaired_digest": digest(repaired),
                        "patch": record(patch),
                        "scope": "Unfrozen proposal repair; acceptance requires full contract validation",
                    },
                    parent,
                )
                try:
                    parsed = contract.parse(repaired)
                except (ValueError, Invalid) as repaired_error:
                    self.store.put(
                        "proposal_rejection",
                        {
                            "contract": contract.__name__,
                            "proposal": repaired,
                            "error": str(repaired_error)[:4000],
                            "repair_id": repair_id,
                        },
                        parent,
                    )
                    raise
                return parsed
            return self.propose(
                contract,
                {
                    **context,
                    "schema_repair": True,
                    "rejected_proposal": value,
                    "validation_error": str(exc)[:2000],
                    "repair_instruction": "Correct the stated contract failure. Preserve honest assumptions; do not invent evidence or modify validation rules.",
                },
                parent,
                role,
            )

    def chat(self, messages, context, parent=None, research=False):
        kwargs = {
            "instructions": prompt("collaborator"),
            "input": [{"role": "developer", "content": canonical(context)}]
            + messages[-12:],
        }
        if research:
            kwargs.update(
                tools=[{"type": "web_search", "search_context_size": "low"}],
                max_tool_calls=2,
                max_output_tokens=3500,
            )
        response = self._call("heavy" if research else "chat", kwargs, parent, research)
        citations = [
            a.model_dump()
            for item in response.output
            if item.type == "message"
            for content in item.content
            if content.type == "output_text"
            for a in content.annotations
            if a.type == "url_citation"
        ]
        return {
            "text": response.output_text,
            "citations": citations,
            "model": response.model,
            "response_id": response.id,
            "mode": "online_research" if research else "chat",
            "status": response.status,
            "completeness": "complete"
            if response.status == "completed"
            else "partial_due_to_output_limit",
            "scope": "Source-linked research text; not independent claim verification. Partial notes must not be treated as a complete evidence review.",
        }


class RuleProvider:
    """Explicit offline baseline, never advertised as an Astra generation."""

    mode = "rule_based"

    def propose(self, contract, context, parent=None, role="heavy"):
        from symplex.core.contracts import ActionProposal, ProblemDNA, ProgramPatch

        if contract is ProgramPatch:
            return ProgramPatch.parse(
                {
                    "parent_id": context["parent_id"],
                    "operation": "fit_calibrator",
                    "value": 0.0,
                    "evidence_ids": context["evidence_ids"],
                    "expectation": "Training-fitted shrinkage may reduce development Brier loss.",
                    "disconfirmation": "Development Brier is unchanged or higher.",
                }
            )
        if contract is ActionProposal:
            return ActionProposal.parse(
                {
                    "action_type": "inspect_definitions",
                    "target_ids": context["target_ids"],
                    "uncertainty_addressed": "Exact-semantic comparability",
                    "preconditions": ["definitions_available"],
                    "tool_plan": ["inspect_definitions"],
                    "estimated_cost": 0,
                    "expected_observable_change": "List excluded near-matches",
                    "completion_condition": "Relation audit persisted",
                    "stop_condition": "After one bounded follow-up",
                }
            )
        if contract is ProblemDNA:
            return ProblemDNA.parse(
                {
                    "beneficiary": "Forecast analyst",
                    "decision": "Use forecast or request more evidence",
                    "objective": "Compare linked price proxies against resolved binary outcomes",
                    "actors": ["event", "contract", "observation"],
                    "constraints": [
                        "Whole-event chronological splits",
                        "Protected confirmation",
                        "No trades",
                    ],
                    "assumptions": [
                        "Curator-verified settlement metadata",
                        "Prices are probability proxies",
                    ],
                    "missing_evidence": [
                        "Independent relation labels",
                        "Prospective forecast archive",
                    ],
                }
            )
        raise Invalid("Offline provider does not implement this proposal")
