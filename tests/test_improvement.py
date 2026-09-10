"""Improvement and outcome contract checks using synthetic artifacts and fake models."""

import copy
import tempfile
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from symplex.agents.improvement import (
    CandidateDesign,
    MethodCandidate,
    develop_candidate,
    propose_method,
)
from symplex.agents.outcomes import DecisionBrief, build_outcome
from symplex.agents.prompts import manifest
from symplex.core.contracts import Invalid
from symplex.infrastructure.storage import Budget, Store

try:
    from tests.test_complex_system import system_spec
except ModuleNotFoundError:
    from test_complex_system import system_spec


class FakeProposer:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def propose(self, contract, context, problem_id, **kwargs):
        self.calls.append((contract, context, problem_id, kwargs))
        return copy.deepcopy(self.result)


def candidate_spec(context_id):
    return {
        "name": "Proposed controlled perturbation",
        "alternative_id": "perturbed",
        "objective": "Discriminate the candidate responses.",
        "design_specification": "Specify a controlled perturbation before collecting data.",
        "changes": [
            {
                "component_id": "source",
                "description": "Change the boundary condition.",
                "expected_observable_change": "Potentially distinguish a response lag.",
                "check_ids": ["instrument_check"],
            }
        ],
        "evidence_ids": [context_id],
        "assumptions": ["A thermal abstraction may be useful."],
        "feasibility_gaps": ["Instrument suitability has not been established."],
        "expected_tradeoffs": ["Additional observations consume resources."],
        "required_experiment_ids": ["lag_test"],
        "acceptance_criterion": "Resolve the rival predictions under the registered observation protocol.",
        "reversal_conditions": ["The required instrument accuracy is unavailable."],
    }


def method_spec(feedback_id):
    return {
        "name": "Require explicit discriminating predictions",
        "changes": [
            {
                "role": "hypothesis_critic",
                "current_failure": "A recorded review found hypotheses sharing predictions.",
                "proposed_instruction": "Request an experiment with distinguishable rival predictions before endorsing testability.",
                "expected_effect": "Identify non-discriminating experiments earlier.",
                "regression_risk": "May over-reject useful exploratory models.",
            }
        ],
        "feedback_ids": [feedback_id],
        "applicable_problem_characteristics": ["Competing system mechanisms."],
        "development_test": "Replay separate development cases under the same budget.",
        "independent_validation_test": "Use unseen cases and a frozen evaluator.",
        "matched_resource_rule": "Same models and resource ceilings for both policies.",
        "acceptance_criterion": "Improve the frozen metric without regression.",
        "rollback_condition": "Withdraw if independent evaluation fails.",
        "transfer_limits": [
            "No improvement has been established across other domains."
        ],
    }


def brief_spec(context_id, system_id):
    return {
        "title": "Conditional experimental-design brief",
        "decision_owner": "Experiment designer",
        "recommendation": "Obtain discriminating observations before deciding between mechanisms.",
        "claims": [
            {
                "statement": "A supplied artifact describes a proposed experimental condition.",
                "basis": "assumption",
                "source_ids": [context_id],
                "limitation": "Artifact presence does not establish a physical mechanism.",
            }
        ],
        "alternatives": [
            {
                "alternative_id": "baseline",
                "assessment": "Reference condition.",
                "benefit_and_tradeoff": "Low intervention cost but limited discrimination.",
                "unresolved_test": "Confirm instrument suitability.",
            },
            {
                "alternative_id": "perturbed",
                "assessment": "Candidate condition.",
                "benefit_and_tradeoff": "Potential information at additional resource cost.",
                "unresolved_test": "Run the frozen comparison.",
            },
        ],
        "next_actions": ["Validate the observation protocol."],
        "reversal_conditions": [
            "Measurements cannot distinguish the rival predictions."
        ],
        "external_validation_needed": [
            "Independent instrument and mechanism validation."
        ],
        "consumable_artifact_ids": [system_id],
    }


class ImprovementTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.store = Store(folder.name)
        self.problem_id = self.store.put(
            "workspace_problem", {"question": "Design a discriminating experiment."}
        )
        self.context_id = self.store.put(
            "context",
            {
                "content": "Synthetic context used only for contract tests.",
                "format": "text",
            },
            self.problem_id,
        )
        self.system = system_spec()
        self.system["observations"][0]["source_artifact_ids"] = [self.context_id]
        self.system["components"][1]["evidence_ids"] = [self.context_id]
        self.system["validation"]["checks"][0]["evidence_ids"] = [self.context_id]
        self.system_id = self.store.put("complex_system", self.system, self.problem_id)

    def test_candidate_is_bound_to_current_system_and_supplied_context(self):
        provider = FakeProposer(candidate_spec(self.context_id))
        ident = develop_candidate(
            self.store, provider, self.problem_id, "Design the declared intervention."
        )
        data = self.store.get(ident)["data"]
        self.assertEqual(data["system_id"], self.system_id)
        self.assertEqual(
            data["system_digest"], self.store.get(self.system_id)["digest"]
        )
        self.assertEqual(data["status"], "proposed")
        self.assertIsNone(data["parent_candidate_id"])
        self.assertIn(self.context_id, {r["id"] for r in data["input_manifest"]})
        self.assertEqual(
            provider.calls[0][1]["task"], "Design the declared intervention."
        )

    def test_unknown_alternatives_components_checks_experiments_and_sources_fail(self):
        changes = [
            lambda c: c.update(alternative_id="invented"),
            lambda c: c["changes"][0].update(component_id="invented"),
            lambda c: c["changes"][0].update(check_ids=["invented"]),
            lambda c: c.update(required_experiment_ids=["invented"]),
            lambda c: c.update(evidence_ids=["invented_artifact"]),
        ]
        for index, mutate in enumerate(changes):
            with self.subTest(mutation=index):
                candidate = candidate_spec(self.context_id)
                mutate(candidate)
                with self.assertRaises(Invalid):
                    develop_candidate(
                        self.store, FakeProposer(candidate), self.problem_id
                    )
        self.assertFalse(self.store.list("candidate_design"))

    def test_existing_evidence_not_in_received_context_is_not_admitted(self):
        with patch("symplex.agents.improvement.working_context", return_value=[]):
            with self.assertRaisesRegex(Invalid, "unavailable evidence"):
                develop_candidate(
                    self.store,
                    FakeProposer(candidate_spec(self.context_id)),
                    self.problem_id,
                )

    def test_revision_preserves_immutable_parent_and_records_predecessor(self):
        first = develop_candidate(
            self.store, FakeProposer(candidate_spec(self.context_id)), self.problem_id
        )
        parent_record = self.store.get(first)
        changed = candidate_spec(self.context_id)
        changed["name"] = "Revised observation plan"
        changed["changes"][0]["description"] = (
            "Use a revised but comparable perturbation design."
        )
        provider = FakeProposer(changed)
        second = develop_candidate(
            self.store, provider, self.problem_id, "Revise the previous design.", first
        )
        self.assertNotEqual(first, second)
        self.assertEqual(self.store.get(first), parent_record)
        self.assertEqual(self.store.get(second)["data"]["parent_candidate_id"], first)
        self.assertEqual(provider.calls[0][1]["predecessor"], parent_record)
        self.assertEqual(
            self.store.get(second)["data"]["system_digest"],
            self.store.get(self.system_id)["digest"],
        )

    def test_stale_and_foreign_candidate_parents_fail_before_model_call(self):
        first = develop_candidate(
            self.store, FakeProposer(candidate_spec(self.context_id)), self.problem_id
        )
        foreign_problem = self.store.put(
            "workspace_problem", {"question": "Other problem"}
        )
        foreign_parent = self.store.put(
            "candidate_design", candidate_spec(self.context_id), foreign_problem
        )
        self.store.invalidate(first)
        for parent in (first, foreign_parent):
            with self.subTest(parent=parent):
                provider = FakeProposer(candidate_spec(self.context_id))
                with self.assertRaises(Invalid):
                    develop_candidate(
                        self.store, provider, self.problem_id, parent_id=parent
                    )
                self.assertFalse(provider.calls)

    def test_latest_system_controls_candidate_reference_admission(self):
        updated = copy.deepcopy(self.system)
        updated["decision"]["alternatives"][1]["id"] = "revised_alternative"
        updated["experiments"][0]["candidate_alternative_ids"] = ["revised_alternative"]
        self.store.put("complex_system", updated, self.problem_id)
        with self.assertRaisesRegex(Invalid, "declared decision alternative"):
            develop_candidate(
                self.store,
                FakeProposer(candidate_spec(self.context_id)),
                self.problem_id,
            )

    def test_candidate_requires_an_experiment_covering_its_alternative(self):
        updated = copy.deepcopy(self.system)
        updated["decision"]["alternatives"].append(
            {
                "id": "untested_alternative",
                "name": "Another intervention",
                "interventions": [],
                "rationale": "A separate decision alternative.",
            }
        )
        self.store.put("complex_system", updated, self.problem_id)
        candidate = candidate_spec(self.context_id)
        candidate["alternative_id"] = "untested_alternative"
        # lag_test is a real experiment but compares only baseline and perturbed.
        with self.assertRaises(Invalid):
            develop_candidate(self.store, FakeProposer(candidate), self.problem_id)
        self.assertFalse(self.store.list("candidate_design"))

    def test_method_proposal_requires_recorded_current_evaluation_before_model_call(
        self,
    ):
        provider = FakeProposer(method_spec("invented_feedback"))
        self.store.put("solution", {"claim": "I improved the system."}, self.problem_id)
        with self.assertRaisesRegex(Invalid, "recorded evaluation feedback"):
            propose_method(self.store, provider, self.problem_id)
        self.assertFalse(provider.calls)
        other = self.store.put("workspace_problem", {"question": "Foreign problem"})
        self.store.put("model_critique", {"verdict": "inconclusive"}, other)
        stale_feedback = self.store.put(
            "model_critique", {"verdict": "inconclusive"}, self.problem_id
        )
        self.store.invalidate(stale_feedback)
        with self.assertRaises(Invalid):
            propose_method(self.store, provider, self.problem_id)
        self.assertFalse(provider.calls)

    def test_method_candidate_never_installs_or_changes_prompts_or_resource_caps(self):
        feedback_id = self.store.put(
            "model_critique",
            {"verdict": "inconclusive", "issue": "Rival predictions overlap."},
            self.problem_id,
        )
        original_feedback = self.store.get(feedback_id)
        prompts_before = manifest()
        budget = Budget(self.store)
        caps_before = budget.snapshot()
        proposal = method_spec(feedback_id)
        provider = FakeProposer(proposal)
        ident = propose_method(self.store, provider, self.problem_id)
        data = self.store.get(ident)["data"]
        self.assertEqual(data["status"], "awaiting_independent_evaluation")
        self.assertIs(data["installed"], False)
        self.assertIs(data["promotion_allowed"], False)
        self.assertEqual(
            data["parent_method"], {r["name"]: r["digest"] for r in prompts_before}
        )
        self.assertEqual(manifest(), prompts_before)
        self.assertEqual(budget.snapshot(), caps_before)
        self.assertEqual(self.store.get(feedback_id), original_feedback)
        self.assertEqual(provider.calls[0][1]["feedback"], [original_feedback])
        self.assertEqual(provider.calls[0][1]["role_versions"], prompts_before)

    def test_method_feedback_must_be_in_the_bounded_supplied_feedback_set(self):
        first = self.store.put(
            "model_critique", {"issue": "Old feedback"}, self.problem_id
        )
        for index in range(8):
            self.store.put(
                "model_critique",
                {"issue": f"Current feedback {index}"},
                self.problem_id,
            )
        for feedback_id in (first, self.context_id, "invented_feedback"):
            with self.subTest(feedback_id=feedback_id):
                with self.assertRaisesRegex(Invalid, "unavailable feedback"):
                    propose_method(
                        self.store,
                        FakeProposer(method_spec(feedback_id)),
                        self.problem_id,
                    )
        self.assertFalse(self.store.list("method_candidate"))

    def test_decision_brief_records_references_without_certifying_claim_basis(self):
        brief = brief_spec(self.context_id, self.system_id)
        brief["claims"][0]["basis"] = "observed"
        provider = FakeProposer(brief)
        ident = build_outcome(self.store, provider, self.problem_id)
        data = self.store.get(ident)["data"]
        self.assertEqual(data["claims"][0]["basis"], "observed")
        self.assertEqual(data["status"], "requires_review")
        self.assertIs(data["independently_validated"], False)
        self.assertEqual(data["system_id"], self.system_id)
        self.assertEqual(
            data["system_digest"], self.store.get(self.system_id)["digest"]
        )
        self.assertIn("citation presence does not establish claim truth", data["scope"])

    def test_outcome_rejects_unsupplied_sources_and_duplicate_or_unknown_alternatives(
        self,
    ):
        changes = [
            lambda b: b["claims"][0].update(source_ids=["invented"]),
            lambda b: b.update(consumable_artifact_ids=["invented"]),
            lambda b: b["alternatives"][1].update(alternative_id="invented"),
            lambda b: b["alternatives"][1].update(alternative_id="baseline"),
        ]
        for index, mutate in enumerate(changes):
            with self.subTest(mutation=index):
                brief = brief_spec(self.context_id, self.system_id)
                mutate(brief)
                with self.assertRaises(Invalid):
                    build_outcome(self.store, FakeProposer(brief), self.problem_id)
        self.assertFalse(self.store.list("decision_brief"))

    def test_outcome_cannot_cite_existing_but_unprovided_artifact(self):
        context = {
            "artifacts": [],
            "executed_files": [],
            "scope": "No evidence supplied.",
        }
        with patch("symplex.agents.outcomes.review_context", return_value=context):
            with self.assertRaisesRegex(Invalid, "not provided to its author"):
                build_outcome(
                    self.store,
                    FakeProposer(brief_spec(self.context_id, self.system_id)),
                    self.problem_id,
                )
            # The current system is separately supplied in full, so its reference is valid.
            brief = brief_spec(self.context_id, self.system_id)
            brief["claims"][0]["source_ids"] = [self.system_id]
            ident = build_outcome(self.store, FakeProposer(brief), self.problem_id)
            self.assertEqual(
                self.store.get(ident)["data"]["consumable_artifact_ids"],
                [self.system_id],
            )

    def test_proposal_schemas_are_closed_required_and_cannot_assert_host_status(self):
        def inspect(node):
            if isinstance(node, dict):
                if node.get("type") == "object":
                    self.assertIs(node.get("additionalProperties"), False)
                    self.assertEqual(set(node["properties"]), set(node["required"]))
                for value in node.values():
                    inspect(value)
            elif isinstance(node, list):
                for value in node:
                    inspect(value)

        cases = [
            (CandidateDesign, candidate_spec(self.context_id), {"status": "executed"}),
            (MethodCandidate, method_spec("feedback"), {"installed": True}),
            (
                DecisionBrief,
                brief_spec(self.context_id, self.system_id),
                {"independently_validated": True},
            ),
        ]
        for contract, value, extra in cases:
            with self.subTest(contract=contract.__name__):
                inspect(contract.json_schema())
                self.assertEqual(contract.parse(value), value)
                with self.assertRaises(ValidationError):
                    contract.parse(dict(value, **extra))
                missing = dict(value)
                missing.pop(next(iter(missing)))
                with self.assertRaises(ValidationError):
                    contract.parse(missing)


if __name__ == "__main__":
    unittest.main()
