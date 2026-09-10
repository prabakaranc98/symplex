"""Large output repair stays bounded and must pass the original host contract."""

import json
import tempfile
import unittest
from types import SimpleNamespace

from openai.types.responses import ResponseFunctionToolCall

from symplex.core.contracts import Invalid, canonical
from symplex.infrastructure.providers import OpenAIProvider
from symplex.infrastructure.storage import Budget, Store


class LargeContract:
    @staticmethod
    def json_schema():
        return {"type": "object", "description": "full contract " * 1000}

    @staticmethod
    def parse(value):
        if value["clock"] != "seconds":
            raise Invalid("clock must be seconds")
        return value


class ProviderRepairTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = Store(directory.name)
        self.provider = OpenAIProvider(self.store, Budget(self.store), client=object())
        self.original = {"clock": "unknown", "model": "mechanism " * 1800}
        self.calls = []

    def run_proposal(self, operations):
        outputs = [self.original, {"operations": operations}]

        def call(role, kwargs, parent=None):
            self.calls.append(kwargs)
            value = outputs.pop(0)
            return SimpleNamespace(
                id="response_" + str(len(self.calls)),
                usage=None,
                output=[
                    ResponseFunctionToolCall(
                        id="fc_" + str(len(self.calls)),
                        call_id="call_" + str(len(self.calls)),
                        type="function_call",
                        name="submit_proposal",
                        arguments=canonical(value),
                    )
                ],
            )

        self.provider._call = call
        return self.provider.propose(
            LargeContract, {"working_context": "old evidence " * 1200}
        )

    def test_large_output_uses_one_small_patch_and_full_host_revalidation(self):
        result = self.run_proposal(
            [
                {
                    "path": "/clock",
                    "value_json": '"seconds"',
                    "reason": "Use the declared clock",
                }
            ]
        )
        self.assertEqual(result["clock"], "seconds")
        self.assertEqual(result["model"], self.original["model"])
        self.assertEqual(self.original["clock"], "unknown")
        self.assertEqual(len(self.calls), 2)
        repair = self.calls[1]
        self.assertEqual(repair["max_output_tokens"], 1800)
        self.assertNotIn("working_context", json.loads(repair["input"][0]["content"]))
        self.assertLess(len(canonical(repair).encode()), 30000)
        self.assertEqual(len(self.store.list("proposal_rejection")), 1)
        self.assertEqual(len(self.store.list("proposal_repair")), 1)

    def test_invalid_repair_cannot_trigger_an_unbounded_repair_loop(self):
        with self.assertRaisesRegex(Invalid, "clock must"):
            self.run_proposal(
                [
                    {
                        "path": "/clock",
                        "value_json": '"hours"',
                        "reason": "Invalid proposed clock",
                    }
                ]
            )
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(self.store.list("proposal_rejection")), 2)

    def test_abstention_preserves_rejection_without_claiming_acceptance(self):
        with self.assertRaisesRegex(Invalid, "No justified"):
            self.run_proposal([])
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(self.store.list("proposal_rejection")), 1)
        self.assertEqual(len(self.store.list("proposal_repair")), 0)


if __name__ == "__main__":
    unittest.main()
