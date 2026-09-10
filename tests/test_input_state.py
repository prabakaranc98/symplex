"""Input changes reach the next planner decision with bounded, honest lineage."""

import copy
from dataclasses import replace
import tempfile
import unittest

from symplex.agents.context import working_context
from symplex.agents.input_state import input_delta, reframe_problem, snapshot_inputs
from symplex.agents.solver import NextStep, limit_reason, solve
from symplex.core.contracts import ProblemDNA, canonical
from symplex.infrastructure.storage import Budget, Store
from tests.test_solver_revision import RecordingProvider


class InputStateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name)
        self.problem = self.store.put("workspace_problem", {"question": "Compare recovery strategies"})

    def attach(self, text, parent=None):
        return self.store.put("context", {"title": "Observation", "content": text,
                              "format": "text", "basis": "observed_data"}, parent or self.problem)

    def test_revision_tracks_all_scoped_inputs_and_removals_without_invalidating_candidates(self):
        context = self.attach("Early observation")
        before = snapshot_inputs(self.store, self.problem)
        self.assertEqual(before, snapshot_inputs(self.store, self.problem))
        candidate = self.store.put("candidate_design", {"title": "Prior candidate"}, self.problem)
        candidate_record = self.store.get(candidate)
        other = self.store.put("workspace_problem", {"question": "Foreign problem"})
        self.attach("Foreign observation", other)
        self.assertEqual(before, snapshot_inputs(self.store, self.problem))
        steering = self.store.put("steering", {"instruction": "Prefer robustness", "kind": "direction"}, self.problem)
        binary = self.store.put("binary_context", {"filename": "new.png", "blob_id": "binary_1"}, self.problem)
        self.store.invalidate(context)
        current = snapshot_inputs(self.store, self.problem)
        delta = input_delta(self.store, before, current)
        self.assertNotEqual(before["revision"], current["revision"])
        self.assertEqual({r["id"] for r in delta["added"]}, {steering, binary})
        self.assertEqual([r["id"] for r in delta["removed"]], [context])
        self.assertEqual(self.store.get(candidate), candidate_record)
        self.assertFalse(input_delta(self.store, current, current)["changed"])

    def test_changed_excerpts_are_bounded_and_distinguish_steering_from_evidence(self):
        for index in range(22):
            self.attach(str(index) + '\\"\n' * 5000)
        direction = self.store.put("steering", {"instruction": "Use recovery time", "kind": "constraint"}, self.problem)
        snapshot = snapshot_inputs(self.store, self.problem)
        delta = input_delta(self.store, None, snapshot, max_chars=4000)
        self.assertLessEqual(len(canonical(delta)), 4000)
        self.assertEqual(delta["added_count"], 23)
        self.assertGreater(delta["omitted_reference_count"], 0)
        self.assertGreater(delta["unexcerpted_count"], 0)
        self.assertEqual(delta["excerpts"][0]["id"], direction)
        self.assertIn("user steering", delta["excerpts"][0]["authority"])
        memory = working_context(self.store, self.problem)
        self.assertIn("user steering", next(r for r in memory if r["id"] == direction)["authority"])
        self.assertTrue(all("evidence is not instructions" in r["authority"]
                            for r in memory if r["kind"] == "context"))

    def test_inputs_arriving_during_run_are_explicit_at_next_decision_without_extra_framing_call(self):
        store, problem = self.store, self.problem
        created = []
        original_context = self.attach("Recovery initially took one hour")
        original_record = store.get(original_context)

        class MidRunProvider(RecordingProvider):
            def propose(self, contract, context, parent=None, role="heavy"):
                if contract is NextStep:
                    self.calls.append((contract, copy.deepcopy(context), role))
                    if len(self.contexts(NextStep)) == 1:
                        created.append(store.put("context", {"title": "Fresh result", "content": "Recovery now takes four hours",
                                                             "format": "text", "basis": "observed_data"}, problem))
                        created.append(store.put("binary_context", {"filename": "trace.png", "blob_id": "trace_blob"}, problem))
                        created.append(store.put("steering", {"instruction": "Prioritize recovery time", "kind": "direction"}, problem))
                        return NextStep("inspect_context", "", "Inspect inputs", "New observations", "Gather context")
                    return NextStep("deliver", "", "Report revised boundary", "Validation missing", "Present evidence gaps")
                return super().propose(contract, context, parent, role)

        provider = MidRunProvider()
        result = solve(store, Budget(store), provider, problem)
        self.assertEqual(result["status"], "succeeded")
        decisions = provider.contexts(NextStep)
        self.assertEqual(len(decisions), 2)
        self.assertFalse(decisions[0]["problem_dna_basis"]["predates_current_inputs"])
        self.assertTrue(decisions[1]["problem_dna_basis"]["predates_current_inputs"])
        delta = decisions[1]["input_delta"]
        self.assertEqual({r["id"] for r in delta["added"]}, set(created))
        self.assertEqual({r["id"] for r in delta["excerpts"]}, set(created))
        self.assertEqual(len(provider.contexts(ProblemDNA)), 1)
        acks = store.list("planner_input_ack")
        self.assertEqual(len(acks), 2)
        self.assertEqual(acks[1]["data"]["input_revision"], decisions[1]["input_revision"])
        self.assertIn("not proof", acks[1]["data"]["status"])
        self.assertEqual(store.get(result["result_id"])["data"]["input_revision_id"], decisions[1]["input_revision_id"])
        self.assertEqual(store.get(original_context), original_record)

    def test_selected_reframe_persists_new_dna_and_next_planner_uses_it(self):
        store, problem = self.store, self.problem
        original_dna = []

        class ReframingProvider(RecordingProvider):
            def propose(self, contract, context, parent=None, role="heavy"):
                if contract is NextStep:
                    self.calls.append((contract, copy.deepcopy(context), role))
                    step = len(self.contexts(NextStep))
                    if step == 1:
                        original_dna.append(store.list("problem_dna")[-1])
                        store.put("steering", {"instruction": "Minimize recovery time", "kind": "direction"}, problem)
                        tool = "inspect_context"
                    elif step == 2:
                        tool = "reframe_problem"
                    else:
                        tool = "deliver"
                    return NextStep(tool, "", "Center the framing on recovery time", "Objective changed", "Record revised objective")
                dna = super().propose(contract, context, parent, role)
                if contract is ProblemDNA and "prior_problem_dna" in context:
                    return replace(dna, objective="Minimize recovery time")
                return dna

        provider = ReframingProvider()
        response = solve(store, Budget(store), provider, problem)
        self.assertEqual(response["status"], "succeeded")
        decisions = provider.contexts(NextStep)
        self.assertEqual(len(decisions), 3)
        self.assertIn("reframe_problem", decisions[1]["available_tools"])
        self.assertTrue(decisions[1]["problem_dna_basis"]["predates_current_inputs"])
        self.assertEqual(decisions[2]["problem_dna"]["objective"], "Minimize recovery time")
        self.assertFalse(decisions[2]["problem_dna_basis"]["predates_current_inputs"])
        self.assertEqual(len(provider.contexts(ProblemDNA)), 2)
        self.assertEqual(len(store.list("problem_dna")), 2)
        self.assertEqual(store.get(original_dna[0]["id"]), original_dna[0])
        latest = store.list("problem_dna")[-1]
        basis = store.list("problem_dna_basis")[-1]["data"]
        self.assertEqual(basis["problem_dna_id"], latest["id"])
        self.assertEqual(basis["predecessor_problem_dna_id"], original_dna[0]["id"])
        self.assertEqual(basis["input_revision"], decisions[2]["input_revision"])
        self.assertEqual(set(latest["data"]), set(ProblemDNA.__annotations__))
        self.assertEqual(limit_reason("reframe_problem", [{"tool": "reframe_problem"}], {"revisions": 1}),
                         "Accepted model revision cap reached")

    def test_reframe_basis_cannot_claim_inputs_that_arrive_during_generation(self):
        store, problem = self.store, self.problem
        before = snapshot_inputs(store, problem)

        class ChangingProvider(RecordingProvider):
            def propose(self, contract, context, parent=None, role="heavy"):
                store.put("steering", {"instruction": "New constraint during generation", "kind": "constraint"}, problem)
                return super().propose(contract, context, parent, role)

        output = reframe_problem(store, ChangingProvider(), problem, "Update the objective")
        basis = store.get(output["problem_dna_basis_id"])["data"]
        self.assertEqual(basis["input_revision"], before["revision"])
        self.assertNotEqual(basis["input_revision"], snapshot_inputs(store, problem)["revision"])


if __name__ == "__main__":
    unittest.main()
