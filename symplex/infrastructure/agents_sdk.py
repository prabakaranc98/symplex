"""One-turn SDK specialists using Symplex's persisted, budgeted Responses gateway.

The SDK owns each specialist run and its proposal tool invocation. Scientific
validation, repair, permissions, steering and investigation state remain host
responsibilities. No SDK client, session or hosted trace exporter makes requests.
"""

import asyncio
import time
from importlib.metadata import version

from agents import (
    Agent,
    FunctionTool,
    Model,
    ModelResponse,
    ModelSettings,
    RunConfig,
    Runner,
    trace,
    custom_span,
)
from agents.model_settings import ModelRetrySettings
from agents.exceptions import AgentsException
from agents.usage import Usage

from symplex.agents.prompts import prompt
from symplex.core.contracts import Invalid, canonical, digest


PROPOSAL_TOOL = "submit_proposal"


class BudgetedProposalModel(Model):
    """A single-use SDK Model; _call remains the sole accounting authority."""

    def __init__(self, provider, parent, role):
        self.provider, self.parent, self.role = provider, parent, role
        self.calls = 0
        self.response = None

    async def get_response(
        self,
        system_instructions,
        input,
        model_settings,
        tools,
        output_schema,
        handoffs,
        tracing,
        *,
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
    ):
        if self.calls:
            raise Invalid("An SDK proposal run permits exactly one model request")
        if (
            handoffs
            or output_schema
            or previous_response_id
            or conversation_id
            or prompt
        ):
            raise Invalid(
                "SDK proposal runs require explicit local context and no handoffs"
            )
        if (
            len(tools) != 1
            or not isinstance(tools[0], FunctionTool)
            or tools[0].name != PROPOSAL_TOOL
        ):
            raise Invalid("SDK proposal runs expose only submit_proposal")
        if model_settings.max_tokens is None or model_settings.max_tokens <= 0:
            raise Invalid("SDK proposal runs require a positive output token limit")
        tool = tools[0]
        kwargs = {
            "instructions": system_instructions,
            "input": input,
            "tools": [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "strict": tool.strict_json_schema,
                    "parameters": tool.params_json_schema,
                }
            ],
            "max_output_tokens": model_settings.max_tokens,
            "tool_choice": {"type": "function", "name": PROPOSAL_TOOL},
            "parallel_tool_calls": False,
        }
        # No SDK retries or second client. Threading keeps independent async role
        # runs responsive; the gateway owns completion accounting even if its
        # awaiting task is cancelled while a request is in flight.
        self.calls += 1
        response = await asyncio.to_thread(
            self.provider._call, self.role, kwargs, self.parent
        )
        self.response = response
        reported = response.usage
        usage = Usage(requests=1)
        if reported is not None:
            usage = Usage(
                requests=1,
                input_tokens=reported.input_tokens,
                output_tokens=reported.output_tokens,
                total_tokens=reported.total_tokens,
                input_tokens_details=getattr(reported, "input_tokens_details", None),
                output_tokens_details=getattr(reported, "output_tokens_details", None),
            )
        return ModelResponse(
            output=response.output,
            usage=usage,
            response_id=response.id,
            raw_usage=reported.model_dump() if reported is not None else None,
        )

    def stream_response(self, *args, **kwargs):
        raise Invalid("Streaming is unavailable for one-turn SDK proposals")


async def invoke_proposal_async(provider, contract, context, parent=None, role="heavy"):
    """Run one real SDK specialist; return its original, already-accounted response.

    submit_proposal only transfers opaque proposal arguments back to the host.
    Contract.parse and any bounded repair stay in OpenAIProvider.propose.
    """
    # Local import permits the provider to import this adapter without a cycle.
    from symplex.infrastructure.providers import api_schema

    submitted = []

    async def submit_proposal(tool_context, arguments):
        submitted.append(tool_context.tool_call_id)
        return arguments

    model = BudgetedProposalModel(provider, parent, role)
    output_limit = getattr(
        contract,
        "output_token_budget",
        6000
        if contract.__name__
        in ("SolutionBlueprint", "SystemSimulation", "CodeProposal")
        else 2400,
    )
    agent = Agent(
        name=f"Symplex {role} {contract.__name__}",
        instructions=prompt("proposer"),
        model=model,
        model_settings=ModelSettings(
            max_tokens=output_limit,
            tool_choice=PROPOSAL_TOOL,
            parallel_tool_calls=False,
            retry=ModelRetrySettings(max_retries=0),
        ),
        tools=[
            FunctionTool(
                name=PROPOSAL_TOOL,
                description="Submit one bounded proposal for host validation. It does not execute itself.",
                params_json_schema=api_schema(contract.json_schema()),
                on_invoke_tool=submit_proposal,
                strict_json_schema=True,
            )
        ],
        tool_use_behavior={"stop_at_tool_names": [PROPOSAL_TOOL]},
    )
    metadata = {
        "sdk": "openai-agents",
        "sdk_version": version("openai-agents"),
        "agent": agent.name,
        "role": role,
        "contract": contract.__name__,
        "model": provider.models[role],
        "context_digest": digest(context),
        "scope_id": getattr(provider.budget, "scope_id", None),
        "max_turns": 1,
        "max_model_requests": 1,
        "max_output_tokens": output_limit,
        "tools": [PROPOSAL_TOOL],
        "external_tracing": False,
        "validation_owner": "host_contract_parser",
    }
    run_id = provider.store.put("sdk_agent_run_started", metadata, parent)
    started = time.monotonic()
    status, error_type, result = "failed", None, None
    try:
        # Explicitly override an enclosing enabled SDK trace as well. A run's
        # tracing_disabled flag alone can inherit its caller's active trace.
        with (
            trace("Symplex local proposal", disabled=True),
            custom_span("Symplex local specialist", disabled=True),
        ):
            result = await Runner.run(
                agent,
                canonical(context),
                max_turns=1,
                run_config=RunConfig(
                    tracing_disabled=True,
                    trace_include_sensitive_data=False,
                    workflow_name="Symplex bounded proposal",
                ),
            )
        if model.response is None:
            raise Invalid("SDK proposal run returned no provider response")
        status = "returned_to_host"
        return model.response
    except BaseException as exc:
        status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
        error_type = type(exc).__name__
        if isinstance(exc, AgentsException):
            raise Invalid("SDK proposal execution failed: " + error_type) from None
        raise
    finally:
        # Deliberately exclude prompts, arguments and exception text. Original
        # provider response/usage artifacts retain the existing local audit trail.
        provider.store.put(
            "sdk_agent_run",
            {
                **metadata,
                "run_id": run_id,
                "status": status,
                "error_type": error_type,
                "gateway_invocations": model.calls,
                "response_id": getattr(model.response, "id", None),
                "usage_reported": model.response is not None
                and model.response.usage is not None,
                "tool_invocations": len(submitted),
                "sdk_result_items": len(result.new_items)
                if result is not None
                else None,
                "runtime_seconds": time.monotonic() - started,
            },
            parent,
        )


def invoke_proposal(provider, contract, context, parent=None, role="heavy"):
    """Synchronous host entry point; async callers use invoke_proposal_async."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            invoke_proposal_async(provider, contract, context, parent, role)
        )
    raise Invalid("Use invoke_proposal_async from an active event loop")
