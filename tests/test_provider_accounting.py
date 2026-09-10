"""Actual provider usage survives estimation errors without relaxing budget caps."""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from symplex.core.contracts import Invalid
from symplex.infrastructure.providers import OpenAIProvider
from symplex.infrastructure.storage import Budget, BudgetExhausted, Store


def response(
    input_tokens=40000,
    output_tokens=100,
    searches=3,
    status="completed",
    missing_usage=False,
):
    usage_dict = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    usage = SimpleNamespace(**usage_dict, model_dump=lambda: usage_dict)
    output = [
        {"type": "web_search_call", "id": "search_" + str(i)} for i in range(searches)
    ]
    output.append({"type": "message", "text": "Retained paid research result"})
    return SimpleNamespace(
        id="response_test",
        model="gpt-6-astra",
        status=status,
        usage=None if missing_usage else usage,
        output=[SimpleNamespace(**item) for item in output],
        model_dump=lambda: {
            "output": output,
            "usage": None if missing_usage else usage_dict,
            "incomplete_details": {"reason": "max_output_tokens"}
            if status == "incomplete"
            else None,
        },
    )


class ProviderAccountingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = Store(directory.name)
        self.problem_id = self.store.put(
            "workspace_problem", {"question": "Test accounting only."}
        )
        env = patch.dict(os.environ, {"OPENAI_HEAVY_MODEL": "gpt-6-astra"})
        env.start()
        self.addCleanup(env.stop)

    def provider(self, returned, *, usd=5.0):
        budget = Budget(self.store, {"usd": usd})
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            if isinstance(returned, Exception):
                raise returned
            return returned

        client = SimpleNamespace(responses=SimpleNamespace(create=create))
        return OpenAIProvider(self.store, budget, client), budget, calls

    def call(self, provider):
        return provider._call(
            "heavy",
            {
                "input": "Search for independently verifiable sources.",
                "tools": [{"type": "web_search"}],
                "max_tool_calls": 2,
            },
            self.problem_id,
            search=True,
        )

    def test_partial_research_text_is_retained_but_structured_calls_still_fail(self):
        returned = response(input_tokens=100, status="incomplete")
        returned.output_text = "A partial source-linked note, not a complete review."
        provider, budget, _ = self.provider(returned)
        result = self.call(provider)
        self.assertEqual(result.status, "incomplete")
        self.assertEqual(len(self.store.list("model_call")), 1)
        self.assertEqual(len(self.store.list("model_failure")), 0)
        with self.assertRaisesRegex(Invalid, "incomplete"):
            provider._call("heavy", {"input": "structured proposal"}, self.problem_id)
        self.assertEqual(len(self.store.list("model_response")), 2)

    def test_actual_search_usage_over_reservation_is_recorded_and_output_returned(self):
        returned = response()
        provider, budget, calls = self.provider(returned)
        self.assertIs(self.call(provider), returned)
        self.assertEqual(len(calls), 1)
        raw = self.store.list("model_response")[0]
        call = self.store.list("model_call")[0]
        self.assertLess(raw["created"], call["created"])
        self.assertEqual(raw["data"]["usage"]["input_tokens"], 40000)
        self.assertEqual(
            raw["data"]["output"][-1]["text"], "Retained paid research result"
        )
        self.assertEqual(call["data"]["response_artifact_id"], raw["id"])
        self.assertEqual(call["data"]["accounting_status"], "settled")
        self.assertIn("input_tokens", call["data"]["accounting"]["over_reservation"])
        self.assertEqual(
            call["data"]["accounting"]["over_reservation"]["evidence_requests"], 1
        )
        self.assertEqual(budget.snapshot()["input_tokens"]["used_or_reserved"], 40000)
        self.assertEqual(budget.snapshot()["output_tokens"]["used_or_reserved"], 100)
        self.assertEqual(budget.snapshot()["evidence_requests"]["used_or_reserved"], 3)
        self.assertAlmostEqual(
            budget.snapshot()["usd"]["used_or_reserved"],
            40000 * 12.5 / 1e6 + 100 * 50 / 1e6 + 0.03,
        )
        self.assertFalse(self.store.list("model_failure"))

    def test_actual_cap_overrun_is_honest_and_blocks_every_future_reservation(self):
        provider, budget, calls = self.provider(response(input_tokens=100000), usd=0.5)
        self.call(provider)
        self.assertGreater(budget.snapshot()["usd"]["used_or_reserved"], 0.5)
        self.assertEqual(budget.snapshot()["usd"]["cap"], 0.5)
        self.assertEqual(
            self.store.list("model_call")[0]["data"]["accounting_status"],
            "settled_over_cap",
        )
        with self.assertRaises(BudgetExhausted):
            budget.reserve(worker_seconds=1)
        with self.assertRaises(BudgetExhausted):
            budget.reserve(model_requests=1)
        with self.assertRaises(BudgetExhausted):
            self.call(provider)
        self.assertEqual(len(calls), 1)
        reopened = Budget(Store(self.store.root), {"usd": 100.0})
        self.assertEqual(reopened.snapshot()["usd"]["cap"], 0.5)
        with self.assertRaises(BudgetExhausted):
            reopened.reserve(model_requests=1)

    def test_response_and_usage_persist_before_unexpected_settlement_failure(self):
        provider, budget, _ = self.provider(response())
        with patch.object(
            budget, "settle", side_effect=Invalid("Synthetic settlement failure")
        ):
            with self.assertRaises(Invalid):
                self.call(provider)
        raw = self.store.list("model_response")[0]
        failure = self.store.list("model_failure")[0]["data"]
        self.assertEqual(raw["data"]["usage"]["input_tokens"], 40000)
        self.assertEqual(
            raw["data"]["output"][-1]["text"], "Retained paid research result"
        )
        self.assertEqual(failure["phase"], "usage_settlement")
        self.assertEqual(failure["response_artifact_id"], raw["id"])
        self.assertEqual(failure["local_error"], "Synthetic settlement failure")
        self.assertFalse(self.store.list("model_call"))

    def test_incomplete_provider_response_is_still_persisted_and_billed(self):
        provider, budget, _ = self.provider(response(status="incomplete"))
        with self.assertRaises(Invalid) as raised:
            self.call(provider)
        self.assertIn("reason=max_output_tokens", str(raised.exception))
        self.assertIn("requested output token limit=2400", str(raised.exception))
        self.assertIn("failure artifact model_failure_", str(raised.exception))
        self.assertEqual(
            self.store.list("model_response")[0]["data"]["status"], "incomplete"
        )
        self.assertEqual(
            self.store.list("model_response")[0]["data"]["incomplete_details"],
            {"reason": "max_output_tokens"},
        )
        self.assertEqual(
            self.store.list("model_call")[0]["data"]["status"], "incomplete"
        )
        self.assertEqual(
            self.store.list("model_failure")[0]["data"]["phase"], "response_completion"
        )
        self.assertEqual(budget.snapshot()["input_tokens"]["used_or_reserved"], 40000)

    def test_missing_reported_usage_retains_reservation_and_marks_uncertainty(self):
        provider, budget, _ = self.provider(response(missing_usage=True))
        self.call(provider)
        call = self.store.list("model_call")[0]["data"]
        self.assertIsNone(call["usage"])
        self.assertIsNone(call["usd_upper_estimate"])
        self.assertEqual(
            call["accounting_status"], "usage_unavailable_reservation_retained"
        )
        self.assertGreater(budget.snapshot()["usd"]["used_or_reserved"], 0)
        with self.store.db() as db:
            self.assertEqual(
                db.execute("SELECT settled FROM reservations").fetchone()["settled"], 0
            )

    def test_explicit_pre_generation_schema_failure_releases_only_unspent_resources(
        self,
    ):
        class SchemaRejected(Exception):
            status_code = 400
            body = {
                "error": {
                    "code": "invalid_function_parameters",
                    "message": "Schema rejected before generation",
                    "param": "tools",
                }
            }

        provider, budget, calls = self.provider(SchemaRejected())
        with self.assertRaises(Invalid):
            self.call(provider)
        self.assertEqual(len(calls), 1)
        self.assertEqual(budget.snapshot()["model_requests"]["used_or_reserved"], 1)
        for resource in ("input_tokens", "output_tokens", "usd", "evidence_requests"):
            self.assertEqual(budget.snapshot()[resource]["used_or_reserved"], 0)
        self.assertFalse(self.store.list("model_response"))

    def test_settlement_remains_atomic_single_use_and_auditable(self):
        budget = Budget(self.store, {"usd": 1.0})
        reservation = budget.reserve(usd=0.1, model_requests=1)
        before = budget.snapshot()
        with self.assertRaises(Invalid):
            budget.settle(reservation, usd=0.2, worker_seconds=1)
        self.assertEqual(budget.snapshot(), before)
        with self.assertRaises(Invalid):
            budget.settle(reservation, usd=float("nan"))
        self.assertEqual(budget.snapshot(), before)
        settlement = budget.settle(reservation, usd=0.2)
        self.assertAlmostEqual(settlement["over_reservation"]["usd"], 0.1)
        self.assertEqual(settlement["retained_reservations"], {"model_requests": 1})
        with self.store.db() as db:
            row = db.execute(
                "SELECT data FROM budget_settlements WHERE reservation_id=?",
                (reservation,),
            ).fetchone()
        self.assertEqual(json.loads(row["data"]), settlement)
        with self.assertRaises(Invalid):
            budget.settle(reservation, usd=0)
        self.assertAlmostEqual(budget.snapshot()["usd"]["used_or_reserved"], 0.2)


if __name__ == "__main__":
    unittest.main()
