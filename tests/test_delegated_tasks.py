"""Planner task focus survives registered tool dispatch into bounded role requests."""

import tempfile
import unittest
from types import SimpleNamespace

from symplex.agents.prompts import prompt
from symplex.agents.tools import REGISTRY, ToolContext
from symplex.infrastructure.storage import Store
from tests.test_complex_system import system_spec
from tests.test_improvement import FakeProposer, brief_spec, method_spec
from tests.test_synthesis import synthesis_result


class DelegatedTaskTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.store = Store(folder.name)
        self.problem = self.store.put("workspace_problem", {"question": "Compare rival lag mechanisms"})
        self.source = self.store.put("context", {
            "content": "Synthetic observation for software-contract testing only.",
            "format": "text", "basis": "user_context",
        }, self.problem)
        self.system_data = system_spec()
        self.system = self.store.put("complex_system", self.system_data, self.problem)
        self.feedback = self.store.put("model_critique", {
            "verdict": "inconclusive", "issue": "References need explicit checking.",
        }, self.problem)

    def run_tool(self, name, provider, instruction, target_id=""):
        ctx = ToolContext(self.store, None, provider, self.problem, lambda: False, [], [])
        output = REGISTRY.execute(name, ctx, SimpleNamespace(
            instruction=instruction, target_id=target_id,
        ), {"model.propose"})
        self.assertEqual(len(ctx.result_ids), 1)
        return self.store.get(ctx.result_ids[0]), output

    def test_registered_roles_receive_exact_task_separately_from_host_instructions(self):
        review = {
            "assessments": [{
                "hypothesis_id": hypothesis["id"], "status": "testable",
                "issue": "Observations remain unvalidated.",
                "discriminating_test": hypothesis["discriminating_test"],
                "confounders": ["Instrument delay"], "evidence_ids": [self.source],
            } for hypothesis in self.system_data["hypotheses"]],
            "coverage_gaps": ["No independent observations"],
            "shared_failure_modes": ["Instrument delay"],
            "next_action": "Obtain independent observations",
        }
        cases = (
            ("review_hypotheses", "hypothesis_critic", review, "review_id", self.system),
            ("synthesize_evidence", "evidence_synthesist", synthesis_result(self.source), "synthesis_id", ""),
            ("propose_method", "method_optimizer", method_spec(self.feedback), "method_candidate_id", ""),
            ("build_outcome", "decision_editor", brief_spec(self.source, self.system), "decision_brief_id", ""),
        )
        original_system = self.store.get(self.system)
        for name, role, result, key, target in cases:
            with self.subTest(tool=name):
                task = f"For {name}, distinguish instrument delay from the proposed memory mechanism.\nPreserve unresolved evidence limits."
                provider = FakeProposer(result)
                record, output = self.run_tool(name, provider, task, target)
                self.assertEqual(output[key], record["id"])
                self.assertEqual(len(provider.calls), 1)
                supplied = provider.calls[0][1]
                self.assertEqual(supplied["requested_task"], task)
                self.assertEqual(supplied["instruction"], prompt(role))
                self.assertEqual(record["data"]["requested_task"], task)
                self.assertEqual(provider.calls[0][2], self.problem)
                self.assertIn("context_selection", supplied)
                if name == "review_hypotheses":
                    self.assertEqual(supplied["system_id"], target)
                    self.assertEqual(supplied["representation"], self.system_data)
                    self.assertEqual(provider.calls[0][3]["role"], "validator")
        self.assertEqual(self.store.get(self.system), original_system)

    def test_synthesis_reuse_requires_the_same_delegated_task(self):
        provider = FakeProposer(synthesis_result(self.source))
        first, _ = self.run_tool("synthesize_evidence", provider, "Assess instrument confounding.")
        repeated, _ = self.run_tool("synthesize_evidence", provider, "Assess instrument confounding.")
        self.assertEqual(first["id"], repeated["id"])
        self.assertEqual(len(provider.calls), 1)
        changed, _ = self.run_tool("synthesize_evidence", provider, "Assess applicability to the new boundary.")
        self.assertNotEqual(first["id"], changed["id"])
        self.assertNotEqual(first["data"]["input_digest"], changed["data"]["input_digest"])
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(self.store.get(first["id"]), first)


if __name__ == "__main__":
    unittest.main()
