"""Real Agent/Runner execution through a mocked official Responses transport."""

import asyncio
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI
from agents import ModelSettings
from agents.tracing import get_trace_provider, set_trace_provider, trace, custom_span
from agents.tracing.provider import DefaultTraceProvider
from agents.tracing.processor_interface import TracingProcessor

from symplex.core.contracts import Invalid, schema
from symplex.infrastructure.agents_sdk import (
    BudgetedProposalModel,
    invoke_proposal,
    invoke_proposal_async,
)
from symplex.infrastructure.providers import OpenAIProvider
from symplex.infrastructure.scoped_budget import ScopedBudget
from symplex.infrastructure.storage import Budget, BudgetExhausted, Store


class ExampleProposal:
    output_token_budget = 1234

    @staticmethod
    def json_schema():
        return schema({"label": {"type": "string", "minLength": 2}})

    @staticmethod
    def parse(value):
        raise AssertionError("SDK must leave host semantic validation outside the tool")


class TraceSpy(TracingProcessor):
    def __init__(self):
        self.events = []

    def on_trace_start(self, trace):
        self.events.append("trace_start")

    def on_trace_end(self, trace):
        self.events.append("trace_end")

    def on_span_start(self, span):
        self.events.append("span_start")

    def on_span_end(self, span):
        self.events.append("span_end")

    def shutdown(self):
        pass

    def force_flush(self):
        pass


class AgentsSDKTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.store = Store(folder.name)
        self.parent = self.store.put(
            "workspace_problem", {"question": "SDK transport test"}
        )
        self.budget = Budget(self.store, {"usd": 10})
        env = patch.dict(
            os.environ,
            {
                "OPENAI_HEAVY_MODEL": "gpt-6-astra",
                "OPENAI_VALIDATOR_MODEL": "gpt-5.6-sol",
            },
        )
        env.start()
        self.addCleanup(env.stop)
        self.requests = []

    def provider(
        self,
        *,
        usage=True,
        error=False,
        budget=None,
        input_tokens=100,
        status="completed",
        arguments=None,
        tool_name="submit_proposal",
    ):
        def respond(request):
            self.requests.append(json.loads(request.content))
            if error:
                return httpx.Response(
                    503,
                    json={
                        "error": {
                            "message": "sk-not-for-artifacts",
                            "type": "server_error",
                        }
                    },
                )
            return httpx.Response(
                200,
                json={
                    "id": "resp_" + str(len(self.requests)),
                    "object": "response",
                    "created_at": 123456,
                    "model": self.requests[-1]["model"],
                    "status": status,
                    "incomplete_details": {"reason": "max_output_tokens"}
                    if status == "incomplete"
                    else None,
                    "parallel_tool_calls": False,
                    "tool_choice": {"type": "function", "name": "submit_proposal"},
                    "tools": self.requests[-1]["tools"],
                    "output": [
                        {
                            "id": "fc_1",
                            "call_id": "call_1",
                            "type": "function_call",
                            "name": tool_name,
                            "arguments": arguments.pop(0)
                            if arguments
                            else '{"label":"x"}',
                            "status": "completed",
                        }
                    ]
                    if status == "completed"
                    else [],
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": 50,
                        "total_tokens": input_tokens + 50,
                        "input_tokens_details": {"cached_tokens": 20},
                        "output_tokens_details": {"reasoning_tokens": 10},
                    }
                    if usage
                    else None,
                },
            )

        client = OpenAI(
            api_key="sk-test-local-only",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        )
        self.addCleanup(client.close)
        return OpenAIProvider(self.store, budget or self.budget, client)

    def test_real_runner_invokes_tool_once_and_preserves_budgeted_native_response(self):
        provider = self.provider()
        response = invoke_proposal(
            provider, ExampleProposal, {"input": "fixture"}, self.parent
        )
        self.assertEqual(response.output[0].arguments, '{"label":"x"}')
        self.assertEqual(len(self.requests), 1)
        request = self.requests[0]
        self.assertEqual(request["model"], "gpt-6-astra")
        self.assertEqual(request["max_output_tokens"], 1234)
        self.assertFalse(request["parallel_tool_calls"])
        self.assertFalse(request["store"])
        self.assertEqual(
            request["tool_choice"], {"type": "function", "name": "submit_proposal"}
        )
        self.assertNotIn(
            "minLength", request["tools"][0]["parameters"]["properties"]["label"]
        )
        self.assertEqual(
            self.budget.snapshot()["input_tokens"]["used_or_reserved"], 100
        )
        self.assertEqual(
            self.budget.snapshot()["output_tokens"]["used_or_reserved"], 50
        )
        self.assertEqual(len(self.store.list("model_call")), 1)
        self.assertEqual(len(self.store.list("model_response")), 1)
        run = self.store.list("sdk_agent_run")[0]["data"]
        self.assertEqual(run["status"], "returned_to_host")
        self.assertEqual(run["tool_invocations"], 1)
        self.assertEqual(run["sdk_result_items"], 2)
        self.assertEqual(run["response_id"], response.id)
        self.assertTrue(run["usage_reported"])

    def test_real_sdk_usage_mapping_retains_details(self):
        import symplex.infrastructure.agents_sdk as adapter
        from agents import ModelResponse

        with patch.object(adapter, "ModelResponse", wraps=ModelResponse) as mapping:
            invoke_proposal(self.provider(), ExampleProposal, {}, self.parent)
        usage = mapping.call_args.kwargs["usage"]
        self.assertEqual(
            (
                usage.requests,
                usage.input_tokens,
                usage.output_tokens,
                usage.total_tokens,
            ),
            (1, 100, 50, 150),
        )
        self.assertEqual(usage.input_tokens_details.cached_tokens, 20)
        self.assertEqual(usage.output_tokens_details.reasoning_tokens, 10)

    def test_external_tracing_disabled_and_local_metadata_excludes_context_secrets(
        self,
    ):
        spy = TraceSpy()
        isolated = DefaultTraceProvider()
        isolated.set_processors([spy])
        previous = get_trace_provider()
        set_trace_provider(isolated)
        try:
            with trace("Caller intentionally tracing"), custom_span("Caller span"):
                spy.events.clear()
                invoke_proposal(
                    self.provider(),
                    ExampleProposal,
                    {"secret": "sensitive-fixture"},
                    self.parent,
                )
                self.assertFalse(spy.events)
        finally:
            set_trace_provider(previous)
        for kind in ("sdk_agent_run_started", "sdk_agent_run"):
            value = json.dumps(self.store.list(kind)[0]["data"])
            self.assertNotIn("sensitive-fixture", value)
            self.assertNotIn("sk-test-local-only", value)

    def test_incomplete_without_function_call_is_persisted_billed_and_rejected(self):
        with self.assertRaisesRegex(Invalid, "max_output_tokens"):
            self.provider(status="incomplete").propose(ExampleProposal, {}, self.parent)
        self.assertEqual(len(self.requests), 1)
        raw = self.store.list("model_response")[0]["data"]
        self.assertEqual(raw["status"], "incomplete")
        self.assertEqual(raw["output"], [])
        self.assertEqual(
            self.budget.snapshot()["input_tokens"]["used_or_reserved"], 100
        )
        run = self.store.list("sdk_agent_run")[0]["data"]
        self.assertEqual(run["status"], "failed")
        self.assertEqual(run["tool_invocations"], 0)

    def test_host_parser_repair_is_a_second_separately_accounted_sdk_run(self):
        class HostContract(ExampleProposal):
            @staticmethod
            def parse(value):
                if len(value["label"]) < 2:
                    raise Invalid("Label is too short")
                return value

        provider = self.provider(arguments=['{"label":"x"}', '{"label":"valid"}'])
        result = provider.propose(HostContract, {}, self.parent)
        self.assertEqual(result, {"label": "valid"})
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(len(self.store.list("proposal_rejection")), 1)
        self.assertEqual(len(self.store.list("sdk_agent_run")), 2)
        self.assertEqual(len(self.store.list("model_call")), 2)
        self.assertAlmostEqual(
            self.budget.snapshot()["usd"]["used_or_reserved"], 0.0075
        )

    def test_unknown_tool_cannot_expand_capabilities_or_start_a_second_request(self):
        with self.assertRaisesRegex(Invalid, "SDK proposal execution failed"):
            self.provider(tool_name="execute_anything").propose(
                ExampleProposal, {}, self.parent
            )
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(
            self.store.list("sdk_agent_run")[0]["data"]["tool_invocations"], 0
        )

    def test_failure_is_not_retried_and_conservative_reservation_remains(self):
        with self.assertRaises(Invalid):
            invoke_proposal(self.provider(error=True), ExampleProposal, {}, self.parent)
        self.assertEqual(len(self.requests), 1)
        self.assertGreater(self.budget.snapshot()["usd"]["used_or_reserved"], 0)
        run = self.store.list("sdk_agent_run")[0]["data"]
        self.assertEqual(run["status"], "failed")
        self.assertEqual(run["error_type"], "Invalid")
        self.assertNotIn("sk-not-for-artifacts", json.dumps(run))

    def test_missing_usage_keeps_reservation_and_is_not_reported_as_zero_cost(self):
        invoke_proposal(self.provider(usage=False), ExampleProposal, {}, self.parent)
        self.assertGreater(self.budget.snapshot()["usd"]["used_or_reserved"], 0.00375)
        self.assertFalse(self.store.list("sdk_agent_run")[0]["data"]["usage_reported"])
        self.assertEqual(
            self.store.list("model_call")[0]["data"]["accounting_status"],
            "usage_unavailable_reservation_retained",
        )

    def test_parallel_sdk_roles_preserve_separate_scopes_and_global_actual_usage(self):
        scope_a = ScopedBudget(self.budget, "a", {"usd": 2})
        scope_b = ScopedBudget(self.budget, "b", {"usd": 2})

        async def run():
            return await asyncio.gather(
                invoke_proposal_async(
                    self.provider(budget=scope_a), ExampleProposal, {}, self.parent
                ),
                invoke_proposal_async(
                    self.provider(budget=scope_b),
                    ExampleProposal,
                    {},
                    self.parent,
                    "validator",
                ),
            )

        asyncio.run(run())
        self.assertEqual(len(self.requests), 2)
        self.assertAlmostEqual(
            scope_a.scope_snapshot()["usd"]["used_or_reserved"], 0.00375
        )
        self.assertAlmostEqual(
            scope_b.scope_snapshot()["usd"]["used_or_reserved"], 0.0015
        )
        self.assertAlmostEqual(
            self.budget.snapshot()["usd"]["used_or_reserved"], 0.00525
        )
        self.assertEqual(
            {r["data"]["scope_id"] for r in self.store.list("sdk_agent_run")},
            {"a", "b"},
        )

    def test_exhausted_scope_cannot_bypass_gateway_to_sdk_default_client(self):
        scope = ScopedBudget(self.budget, "exhausted", {"usd": 0.00001})
        with self.assertRaises(BudgetExhausted):
            invoke_proposal(
                self.provider(budget=scope), ExampleProposal, {}, self.parent
            )
        self.assertEqual(self.requests, [])

    def test_actual_overrun_stays_charged_and_blocks_subsequent_sdk_runs(self):
        scope = ScopedBudget(self.budget, "overrun", {"usd": 0.5})
        provider = self.provider(budget=scope, input_tokens=100000)
        invoke_proposal(provider, ExampleProposal, {}, self.parent)
        self.assertGreater(scope.scope_snapshot()["usd"]["used_or_reserved"], 0.5)
        with self.assertRaises(BudgetExhausted):
            invoke_proposal(provider, ExampleProposal, {}, self.parent)
        self.assertEqual(len(self.requests), 1)

    def test_shared_request_envelope_accepts_large_primary_and_blocks_overflow(self):
        from symplex.infrastructure.limits import MAX_REQUEST_BYTES

        provider = self.provider()
        invoke_proposal(
            provider, ExampleProposal, {"primary": "x" * 70000}, self.parent
        )
        self.assertEqual(len(self.requests), 1)
        before = self.budget.snapshot()
        with self.assertRaisesRegex(Invalid, "Context exceeds"):
            invoke_proposal(
                provider,
                ExampleProposal,
                {"primary": "x" * MAX_REQUEST_BYTES},
                self.parent,
            )
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.budget.snapshot(), before)

    def test_model_enforces_single_use_even_if_runner_is_reconfigured(self):
        model = BudgetedProposalModel(self.provider(), self.parent, "heavy")
        model.calls = 1
        with self.assertRaisesRegex(Invalid, "exactly one"):
            asyncio.run(
                model.get_response(None, "", ModelSettings(), [], None, [], None)
            )
        self.assertEqual(self.requests, [])

    def test_sync_entry_point_inside_event_loop_fails_before_starting_work(self):
        async def run():
            with self.assertRaisesRegex(Invalid, "invoke_proposal_async"):
                invoke_proposal(self.provider(), ExampleProposal, {}, self.parent)

        asyncio.run(run())
        self.assertEqual(self.requests, [])
        self.assertFalse(self.store.list("sdk_agent_run_started"))


if __name__ == "__main__":
    unittest.main()
