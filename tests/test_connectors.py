import json
import tempfile
import unittest
from types import SimpleNamespace

from openai.types.responses import ResponseFunctionToolCall

from symplex.agents.tools import Tool, ToolRegistry
from symplex.connectors.registry import CATALOG, HANDLERS, ConnectorInput, execute
from symplex.connectors.retrieval import search
from symplex.connectors.semantics import project
from symplex.connectors.tables import profile
from symplex.core.contracts import Invalid
from symplex.infrastructure.storage import Budget, Store
from symplex.observability.tracing import export_trace, span


class ConnectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name)
        self.budget = Budget(self.store)
        self.problem = self.store.put(
            "workspace_problem", {"question": "Assess protein digestion uncertainty"}
        )

    def test_registry_has_handlers_and_rejects_planned_and_denied(self):
        self.assertEqual(
            {c.id for c in CATALOG if c.status != "planned"}, set(HANDLERS)
        )
        for name in ("remote_mcp", "not_a_connector", "crossref"):
            with self.assertRaises(Invalid):
                execute(
                    self.store,
                    self.budget,
                    None,
                    ConnectorInput(
                        connector=name, problem_id=self.problem, query="test"
                    ),
                    {"artifacts.read"},
                )
        self.assertEqual(
            self.budget.snapshot()["evidence_requests"]["used_or_reserved"], 0
        )

    def test_retrieval_excludes_other_problems_stale_and_protected_labels(self):
        visible = self.store.put(
            "context", {"content": "protein digestion measurements"}, self.problem
        )
        self.store.put(
            "dataset", {"content": "secret protein digestion labels"}, self.problem
        )
        self.store.put(
            "context", {"content": "protein digestion private context"}, "other_problem"
        )
        stale = self.store.put(
            "context", {"content": "protein digestion outdated context"}, self.problem
        )
        self.store.invalidate(stale)
        hits = search(self.store, self.problem, "protein digestion")["hits"]
        self.assertEqual({h["artifact_id"] for h in hits}, {visible})
        self.assertTrue(all(h["digest"] for h in hits))

    def test_duckdb_profiles_current_uploaded_csv_only(self):
        ident = self.store.put(
            "context",
            {"format": "csv", "content": "enzyme,response\nA,0.5\nB,0.7\n"},
            self.problem,
        )
        result = profile(self.store, self.problem, ident)
        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["sample"][0]["enzyme"], "A")
        with self.assertRaises(Invalid):
            profile(self.store, "other_problem", ident)
        self.store.invalidate(ident)
        with self.assertRaises(Invalid):
            profile(self.store, self.problem, ident)

    def test_rdf_preserves_source_and_hypothesis_status(self):
        ident = self.store.put(
            "solution",
            {
                "components": [
                    {"id": "a", "name": "Enzyme", "kind": "model"},
                    {"id": "b", "name": "Absorption", "kind": "model"},
                ],
                "couplings": [
                    {
                        "source": "a",
                        "target": "b",
                        "meaning": "Proposed catalytic response",
                        "units": "normalized",
                        "timing": "lagged",
                    }
                ],
            },
            self.problem,
        )
        value = project(self.store, self.problem)
        self.assertEqual(value["source_id"], ident)
        self.assertEqual(value["claim_status"], "proposed_mechanisms")
        self.assertIn("proposed_mechanism", value["turtle"])
        self.assertEqual(value["edges"][0]["target"], "b")

    def test_nested_traces_record_failures_without_payloads(self):
        with self.assertRaises(ValueError):
            with span(self.store, self.problem, "outer"):
                with span(self.store, self.problem, "inner"):
                    raise ValueError("private-user-document")
        self.store.put(
            "model_call",
            {
                "model": "test",
                "output": ["secret-content"],
                "usage": {"input_tokens": 2},
            },
            self.problem,
        )
        records = export_trace(self.store, self.problem)
        serialized = json.dumps(records)
        self.assertNotIn("private-user-document", serialized)
        self.assertNotIn("secret-content", serialized)
        inner = next(r for r in records if r.get("name") == "inner")
        outer = next(r for r in records if r.get("name") == "outer")
        self.assertEqual(inner["parent_span_id"], outer["span_id"])
        self.assertEqual(inner["status"], "failed")

    def test_tool_dispatch_denied_before_handler(self):
        registry = ToolRegistry()
        called = []
        registry.register(
            Tool(
                "restricted",
                "test",
                lambda *args: called.append(True),
                "network.research",
            )
        )
        with self.assertRaises(Invalid):
            registry.execute("restricted", None, None, {"artifacts.read"})
        self.assertFalse(called)

    def test_connector_result_and_trace_link_to_problem(self):
        self.store.put("context", {"content": "protein digestion"}, self.problem)
        result = execute(
            self.store,
            self.budget,
            None,
            ConnectorInput(
                connector="local_retrieval", problem_id=self.problem, query="protein"
            ),
            {"artifacts.read"},
        )
        self.assertEqual(self.store.get(result["id"])["parent"], self.problem)
        self.assertEqual(len(self.store.list("trace_span")), 1)

    def test_schema_repair_bounded_and_persisted(self):
        from symplex.infrastructure.providers import OpenAIProvider

        class Contract:
            @staticmethod
            def json_schema():
                return {"type": "object"}

            @staticmethod
            def parse(data):
                if data["value"] != "valid":
                    raise Invalid("Invalid candidate")
                return data

        provider = OpenAIProvider(self.store, self.budget, client=object())
        calls = []

        def response(*args, **kwargs):
            calls.append(1)
            return SimpleNamespace(
                id="mock",
                usage=None,
                output=[
                    ResponseFunctionToolCall(
                        id="fc_mock",
                        call_id="call_mock",
                        type="function_call",
                        name="submit_proposal",
                        arguments='{"value":"invalid"}',
                    )
                ],
            )

        provider._call = response
        with self.assertRaises(Invalid):
            provider.propose(Contract, {}, self.problem)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(self.store.list("proposal_rejection")), 2)

    def test_action_cap_delivers_without_executing_repeated_tool(self):
        from symplex.agents.solver import NextStep, solve
        from symplex.core.contracts import ProblemDNA
        from symplex.core.proposals import DeliveryReview
        from symplex.infrastructure.providers import RuleProvider

        class Provider:
            def propose(self, contract, context, parent=None, role="heavy"):
                if contract is ProblemDNA:
                    return RuleProvider().propose(contract, context)
                if contract is NextStep:
                    return NextStep(
                        "inspect_context",
                        "",
                        "Read context",
                        "Missing context",
                        "Inspect once",
                    )
                if contract is DeliveryReview:
                    return {
                        "assessment": "No empirical result",
                        "unsupported_claims": [],
                        "next_validation": "Acquire data",
                    }
                raise AssertionError(contract)

        result = solve(
            self.store, self.budget, Provider(), self.problem, depth="focused"
        )
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(len(self.store.list("action_rejection")), 1)
        self.assertEqual(
            self.store.get(result["result_id"])["data"]["status"], "dependency_gap"
        )

    def test_design_handler_dispatches_to_compiler(self):
        from unittest.mock import patch
        from symplex.agents.tools import REGISTRY, ToolContext

        result_id = self.store.put("solution", {"title": "Compiled"}, self.problem)
        ctx = ToolContext(self.store, self.budget, object(), self.problem, None, [], [])
        with patch(
            "symplex.agents.tools.compile_solution", return_value=result_id
        ) as compiler:
            output = REGISTRY.execute(
                "design_solution", ctx, SimpleNamespace(), {"model.propose"}
            )
        compiler.assert_called_once_with(self.store, ctx.provider, self.problem)
        self.assertEqual(output["solution_id"], result_id)

    def test_new_user_steering_reaches_next_agent_decision(self):
        from symplex.agents.solver import solve, NextStep
        from symplex.infrastructure.providers import RuleProvider
        from symplex.core.contracts import ProblemDNA
        from symplex.core.proposals import DeliveryReview

        store = self.store
        problem = self.problem
        seen = []

        class Provider:
            def propose(self, contract, context, parent=None, role="heavy"):
                if contract is ProblemDNA:
                    return RuleProvider().propose(contract, context)
                if contract is NextStep:
                    seen.append(context["user_steering"])
                    if len(seen) == 1:
                        store.put(
                            "steering",
                            {
                                "instruction": "Prioritize uncertainty over ranking",
                                "kind": "judgment",
                            },
                            problem,
                        )
                        return NextStep(
                            "inspect_context", "", "Inspect", "Missing context", "Read"
                        )
                    return NextStep("deliver", "", "Deliver", "Report limits", "Result")
                if contract is DeliveryReview:
                    return {
                        "assessment": "Scoped",
                        "unsupported_claims": [],
                        "next_validation": "Evidence",
                    }

        self.assertEqual(
            solve(store, self.budget, Provider(), problem)["status"], "succeeded"
        )
        self.assertFalse(seen[0])
        self.assertEqual(
            seen[1][0]["instruction"], "Prioritize uncertainty over ranking"
        )
        self.assertEqual(len(store.list("steering_ack")), 1)

    def test_agent_clarification_pauses_without_repeated_calls_then_resumes(self):
        from symplex.agents.solver import solve, NextStep
        from symplex.infrastructure.providers import RuleProvider
        from symplex.core.contracts import ProblemDNA
        from symplex.core.proposals import DeliveryReview

        calls = []

        class Provider:
            def propose(self, contract, context, parent=None, role="heavy"):
                calls.append(contract)
                if contract is ProblemDNA:
                    return RuleProvider().propose(contract, context)
                if contract is NextStep:
                    return NextStep(
                        "deliver" if context["user_steering"] else "ask_user",
                        "",
                        "What outcome matters?",
                        "Missing objective",
                        "Resolve objective",
                    )
                if contract is DeliveryReview:
                    return {
                        "assessment": "Scoped",
                        "unsupported_claims": [],
                        "next_validation": "Evidence",
                    }

        provider = Provider()
        first = solve(self.store, self.budget, provider, self.problem)
        self.assertEqual(first["status"], "waiting_user")
        count = len(calls)
        self.assertEqual(
            solve(self.store, self.budget, provider, self.problem)["status"],
            "waiting_user",
        )
        self.assertEqual(len(calls), count)
        self.store.put(
            "steering",
            {"instruction": "Prioritize robustness", "kind": "direction"},
            self.problem,
        )
        resumed = solve(self.store, self.budget, provider, self.problem)
        self.assertEqual(resumed["status"], "succeeded")
        self.assertEqual(resumed["id"], first["id"])
