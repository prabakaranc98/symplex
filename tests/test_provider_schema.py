"""Remote schema adaptation must not weaken host contracts or expose credentials."""

import copy
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

from symplex.agents.improvement import CandidateDesign, MethodCandidate
from symplex.agents.outcomes import DecisionBrief
from symplex.agents.scientific_cycle import HypothesisReview
from symplex.agents.solver import NextStep
from symplex.core.contracts import (
    ActionProposal,
    Invalid,
    MethodPatch,
    ProblemDNA,
    ProgramPatch,
)
from symplex.core.proposals import CodeProposal, DeliveryReview, Review
from symplex.evaluation.experiments import ExperimentOutput, ExperimentProtocol
from symplex.infrastructure.providers import OpenAIProvider, api_schema
from symplex.infrastructure.storage import Budget, Store
from symplex.modeling.complex_system import ComplexSystemSpec
from symplex.modeling.dynamics import SystemSimulation
from symplex.modeling.solutions import SolutionBlueprint

try:
    from tests.test_complex_system import system_spec
except ModuleNotFoundError:
    from test_complex_system import system_spec


PROPOSAL_CONTRACTS = (
    ActionProposal,
    MethodPatch,
    ProblemDNA,
    ProgramPatch,
    CodeProposal,
    DeliveryReview,
    Review,
    NextStep,
    ComplexSystemSpec,
    SolutionBlueprint,
    SystemSimulation,
    HypothesisReview,
    ExperimentProtocol,
    CandidateDesign,
    MethodCandidate,
    DecisionBrief,
)


class ProviderSchemaTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.store = Store(folder.name)
        self.budget = Budget(self.store, {"usd": 5.0})

    def test_all_exposed_remote_contracts_have_no_ref_siblings(self):
        schemas = [(c.__name__, c.json_schema()) for c in PROPOSAL_CONTRACTS]
        schemas.append(("ExperimentOutput", ExperimentOutput.model_json_schema()))

        def inspect(node):
            if isinstance(node, dict):
                if "$ref" in node:
                    self.assertEqual(set(node), {"$ref"})
                if node.get("type") == "object":
                    self.assertIs(node.get("additionalProperties"), False)
                    self.assertEqual(set(node["properties"]), set(node["required"]))
                if node.get("type") == "string":
                    self.assertNotIn("minLength", node)
                    self.assertNotIn("maxLength", node)
                for value in node.values():
                    inspect(value)
            elif isinstance(node, list):
                for value in node:
                    inspect(value)

        for name, schema in schemas:
            with self.subTest(contract=name):
                original = copy.deepcopy(schema)
                inspect(api_schema(schema))
                self.assertEqual(schema, original)

    def test_only_ref_annotations_and_remote_string_lengths_are_removed(self):
        schema = {
            "$defs": {
                "Label": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 10,
                    "pattern": "^[a-z]+$",
                    "description": "A bounded label.",
                },
            },
            "type": "object",
            "properties": {
                "label": {
                    "$ref": "#/$defs/Label",
                    "title": "Label",
                    "description": "Annotated reference.",
                },
                "quantity": {"type": "number", "minimum": 0, "maximum": 10},
            },
            "required": ["label", "quantity"],
            "additionalProperties": False,
        }
        original = copy.deepcopy(schema)
        adapted = api_schema(schema)
        self.assertEqual(adapted["properties"]["label"], {"$ref": "#/$defs/Label"})
        self.assertEqual(
            adapted["$defs"]["Label"],
            {
                "type": "string",
                "pattern": "^[a-z]+$",
                "description": "A bounded label. Host validation requires at least 1 characters, at most 10 characters.",
            },
        )
        self.assertEqual(
            adapted["properties"]["quantity"], schema["properties"]["quantity"]
        )
        self.assertEqual(adapted["required"], schema["required"])
        self.assertIs(adapted["additionalProperties"], False)
        self.assertEqual(schema, original)

    def test_validation_siblings_are_rejected_instead_of_silently_discarded(self):
        for key, value in (
            ("minimum", 0),
            ("pattern", "^[a-z]+$"),
            ("minLength", 1),
            ("maxItems", 1),
            ("enum", ["only"]),
            ("anyOf", [{"type": "null"}]),
            ("additionalProperties", False),
        ):
            with self.subTest(sibling=key):
                with self.assertRaises(Invalid):
                    api_schema({"$ref": "#/$defs/Target", key: value})

    def test_unsupported_schema_is_rejected_before_network_or_budget_reservation(self):
        requests = []

        def create(**kwargs):
            requests.append(kwargs)
            raise AssertionError("The invalid schema must never reach the client")

        class UnsupportedContract:
            @staticmethod
            def json_schema():
                return {
                    "type": "object",
                    "properties": {"value": {"$ref": "#/$defs/Value", "minimum": 0}},
                    "required": ["value"],
                    "additionalProperties": False,
                    "$defs": {"Value": {"type": "number"}},
                }

        client = SimpleNamespace(responses=SimpleNamespace(create=create))
        before = self.budget.snapshot()
        provider = OpenAIProvider(self.store, self.budget, client=client)
        with self.assertRaises(Invalid):
            provider.propose(UnsupportedContract, {"task": "Do not make a request."})
        self.assertFalse(requests)
        self.assertEqual(self.budget.snapshot(), before)
        self.assertFalse(self.store.list("model_call"))

    def test_host_string_and_numeric_bounds_survive_remote_adaptation(self):
        schema = ComplexSystemSpec.json_schema()
        original = copy.deepcopy(schema)
        api_schema(schema)
        self.assertEqual(ComplexSystemSpec.json_schema(), original)
        for value in ("", "   ", "x" * 1601):
            with self.subTest(title=value[:12]):
                data = system_spec()
                data["title"] = value
                with self.assertRaises(ValidationError):
                    ComplexSystemSpec.parse(data)
        for value in (0.0, -1.0, float("nan"), float("inf"), True, "1.0"):
            with self.subTest(scale=value):
                data = system_spec()
                data["units"][0]["scale_to_canonical"] = value
                with self.assertRaises(ValidationError):
                    ComplexSystemSpec.parse(data)
        data = system_spec()
        data["entities"][0]["id"] = "invalid identifier"
        with self.assertRaises(ValidationError):
            ComplexSystemSpec.parse(data)

    def test_provider_error_preserves_schema_diagnostics_and_redacts_credentials(self):
        # These are synthetic markers. The real environment value is never read or sent.
        synthetic_key = "unit-test-credential-marker"
        synthetic_sk_token = "sk-test_ONLY-fake-123"

        class FakeProviderError(Exception):
            status_code = 400
            body = {
                "error": {
                    "message": "Invalid schema: sibling description. "
                    + synthetic_key
                    + " "
                    + synthetic_sk_token,
                    "param": "tools.0 " + synthetic_sk_token,
                    "code": "invalid_schema " + synthetic_key,
                }
            }

        def create(**kwargs):
            raise FakeProviderError(
                "Exception text must not be persisted: " + synthetic_key
            )

        client = SimpleNamespace(responses=SimpleNamespace(create=create))
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": synthetic_key, "OPENAI_HEAVY_MODEL": "gpt-6-astra"},
        ):
            provider = OpenAIProvider(self.store, self.budget, client=client)
            with self.assertRaises(Invalid) as raised:
                provider._call("heavy", {"input": "Synthetic schema test."})
        failures = self.store.list("model_failure")
        self.assertEqual(len(failures), 1)
        failure = failures[0]["data"]
        stored = json.dumps(failure)
        self.assertNotIn(synthetic_key, stored)
        self.assertNotIn(synthetic_sk_token, stored)
        self.assertNotIn("Exception text must not be persisted", stored)
        self.assertNotIn(synthetic_key, str(raised.exception))
        self.assertNotIn(synthetic_sk_token, str(raised.exception))
        self.assertIn(
            "Invalid schema: sibling description.", failure["provider_error"]["message"]
        )
        self.assertIn("[redacted]", failure["provider_error"]["message"])
        self.assertEqual(failure["http_status"], 400)


if __name__ == "__main__":
    unittest.main()
