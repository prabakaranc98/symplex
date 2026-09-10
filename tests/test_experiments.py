"""Adversarial software checks; fixture scores are not scientific evidence."""

import copy
import json
import tempfile
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from symplex.agents.scientific_cycle import (
    HypothesisReview,
    current_artifact,
    represent_system,
    review_hypotheses,
)
from symplex.connectors.compute import save_blob
from symplex.core.contracts import Invalid
from symplex.evaluation.experiments import (
    EVALUATOR_VERSION,
    ExperimentOutput,
    ExperimentProtocol,
    compare_computation,
    plan_experiment,
)
from symplex.infrastructure.storage import Store

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


def protocol_spec():
    return {
        "execution_readiness": "ready",
        "blocking_reasons": [],
        "experiment_id": "lag_test",
        "baseline_id": "baseline",
        "candidate_ids": ["perturbed"],
        "metrics": [
            {
                "id": "quality",
                "name": "Quality",
                "unit": "declared score",
                "direction": "maximize",
                "minimum_improvement": 1.0,
                "maximum_degradation": 0.0,
            },
            {
                "id": "cost",
                "name": "Cost",
                "unit": "declared resource",
                "direction": "minimize",
                "minimum_improvement": 1.0,
                "maximum_degradation": 0.0,
            },
        ],
        "required_check_ids": ["instrument_check"],
        "numeric_checks": [
            {
                "id": ident,
                "operation": operation,
                "csv_filename": "trajectory.csv",
                "columns": ["quality", "cost"],
                "group_columns": ["alternative"],
                "time_column": None,
                "row_filters": [],
                "lower": 0.0 if operation == "bounds" else None,
                "upper": 100.0 if operation == "bounds" else None,
                "reference_value": None,
                "tolerance": 0.0,
                "direction": None,
                "units": "Synthetic fixture score/resource units",
                "rationale": "Check the fixture's declared finite nonnegative bounded output range.",
            }
            for ident, operation in (
                ("finite_values", "finite"),
                ("fixture_bounds", "bounds"),
            )
        ],
        "comparison_controls": "Same generated-data assumptions and execution budget.",
        "uncertainty_plan": "Report synthetic intervals as unverified maker output.",
        "stopping_rule": "Stop after the frozen alternative set is evaluated once.",
        "evidence_gaps": ["No independent observation or replication."],
    }


def trial(alternative, quality, cost, passed=True):
    return {
        "alternative_id": alternative,
        "metrics": [
            {"metric_id": "quality", "value": quality, "lower": None, "upper": None},
            {"metric_id": "cost", "value": cost, "lower": None, "upper": None},
        ],
        "checks": [
            {
                "check_id": "instrument_check",
                "passed": passed,
                "detail": "Synthetic maker-reported diagnostic.",
            }
        ],
    }


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.store = Store(folder.name)
        self.problem_id = self.store.put(
            "workspace_problem",
            {"question": "Compare hypothetical experimental designs."},
        )
        self.context_id = self.store.put(
            "context",
            {"content": "Synthetic test context.", "format": "text"},
            self.problem_id,
        )
        self.system = system_spec()
        self.system["observations"][0]["source_artifact_ids"] = [self.context_id]
        self.system["components"][1]["evidence_ids"] = [self.context_id]
        self.system["validation"]["checks"][0]["evidence_ids"] = [self.context_id]
        self.system_id = self.store.put("complex_system", self.system, self.problem_id)
        self.protocol_id = plan_experiment(
            self.store, FakeProposer(protocol_spec()), self.problem_id, self.system_id
        )

    def result(self):
        return {
            "protocol_id": self.protocol_id,
            "basis": "synthetic",
            "trials": [trial("baseline", 5.0, 10.0), trial("perturbed", 7.0, 9.0)],
            "data_description": "Synthetic contract data only.",
            "uncertainty_method": "No calibrated uncertainty established.",
            "limitations": ["Maker-reported numbers are not independent evidence."],
        }

    def package(
        self,
        output=None,
        *,
        manifest=None,
        calls=None,
        basis="generated",
        filename="symplex_experiment.json",
        output_run_id=None,
        extra_files=(),
        csv_text=None,
    ):
        protocol = self.store.get(self.protocol_id)
        run_id = self.store.put(
            "compute_run",
            {
                "calls": calls if calls is not None else [{"status": "completed"}],
                "input_manifest": manifest
                if manifest is not None
                else [
                    {k: record[k] for k in ("id", "kind", "digest")}
                    for record in (
                        self.store.get(self.problem_id),
                        self.store.get(protocol["data"]["system_id"]),
                        protocol,
                    )
                ],
            },
            self.problem_id,
        )
        output_data = output if output is not None else self.result()
        raw = json.dumps(output_data).encode()
        file_id = save_blob(
            self.store, raw, filename, self.problem_id, basis, output_run_id or run_id
        )
        if csv_text is None:
            lines = ["alternative,quality,cost"]
            for alternative in output_data["trials"]:
                values = {m["metric_id"]: m["value"] for m in alternative["metrics"]}
                lines.append(
                    f"{alternative['alternative_id']},{values.get('quality', 1)},{values.get('cost', 1)}"
                )
            csv_text = "\n".join(lines) + "\n"
        csv_id = save_blob(
            self.store,
            csv_text.encode(),
            "trajectory.csv",
            self.problem_id,
            "generated",
            run_id,
        )
        package_id = self.store.put(
            "compute_package",
            {"run_id": run_id, "file_ids": [file_id, csv_id, *extra_files]},
            self.problem_id,
        )
        return package_id, run_id, file_id

    def test_protocol_is_frozen_with_system_digest_and_output_contract(self):
        record = self.store.get(self.protocol_id)
        self.assertEqual(record["data"]["status"], "frozen_for_comparison")
        self.assertEqual(record["data"]["system_id"], self.system_id)
        self.assertEqual(
            record["data"]["system_digest"], self.store.get(self.system_id)["digest"]
        )
        self.assertEqual(
            record["data"]["result_contract"], ExperimentOutput.model_json_schema()
        )
        for change in (
            {"candidate_ids": ["invented"]},
            {"baseline_id": "invented"},
            {"experiment_id": "invented"},
            {"required_check_ids": ["invented"]},
            {"numeric_checks": []},
            {"numeric_checks": protocol_spec()["numeric_checks"][:1]},
            {"numeric_checks": protocol_spec()["numeric_checks"][1:]},
        ):
            proposal = dict(protocol_spec(), **change)
            with self.subTest(change=change), self.assertRaises(Invalid):
                plan_experiment(
                    self.store, FakeProposer(proposal), self.problem_id, self.system_id
                )
        self.assertEqual(len(self.store.list("experiment_protocol")), 1)
        self.assertIn("numeric_checks", ExperimentProtocol.json_schema()["required"])

    def test_agent_experiment_focus_reaches_specialist_and_frozen_record(self):
        from types import SimpleNamespace
        from symplex.agents.tools import ToolContext, plan_experiment as tool_plan
        provider = FakeProposer(protocol_spec())
        instruction = "Test the declared perturbation and preserve all failed controls."
        context = ToolContext(self.store, None, provider, self.problem_id, None, [], [])
        result = tool_plan(context, SimpleNamespace(target_id=self.system_id, instruction=instruction))
        self.assertEqual(provider.calls[-1][1]["requested_task"], instruction)
        self.assertEqual(self.store.get(result["protocol_id"])["data"]["requested_task"], instruction)
        self.assertEqual(self.store.get(result["protocol_id"])["data"]["baseline_id"], "baseline")

    def test_readiness_is_explicit_in_new_proposals_and_unknown_for_legacy_loading(self):
        schema = ExperimentProtocol.json_schema()
        for name in ("execution_readiness", "blocking_reasons"):
            self.assertIn(name, schema["required"])
            self.assertNotIn("default", schema["properties"][name])
        self.assertEqual(schema["properties"]["execution_readiness"]["enum"], ["ready", "blocked"])
        legacy = protocol_spec()
        legacy.pop("execution_readiness")
        legacy.pop("blocking_reasons")
        self.assertEqual(ExperimentProtocol.model_validate(legacy).execution_readiness, "unknown")
        with self.assertRaisesRegex(Invalid, "explicit"):
            ExperimentProtocol.parse(legacy)
        for readiness, reasons in (("ready", ["Missing numerical design"]), ("blocked", [])):
            with self.subTest(readiness=readiness), self.assertRaises(ValidationError):
                ExperimentProtocol.parse(dict(protocol_spec(), execution_readiness=readiness, blocking_reasons=reasons))

    def test_blocked_incomplete_design_is_preserved_without_becoming_frozen(self):
        blocked = dict(protocol_spec(), execution_readiness="blocked",
                       blocking_reasons=["Numerical tolerance and observable need definition."],
                       numeric_checks=[], stopping_rule="Do not execute before the unresolved design is specified.")
        ident = plan_experiment(self.store, FakeProposer(blocked), self.problem_id, self.system_id)
        data = self.store.get(ident)["data"]
        self.assertEqual(data["status"], "blocked_design")
        self.assertEqual(data["execution_readiness"], "blocked")
        self.assertEqual(data["blocking_reasons"], blocked["blocking_reasons"])
        self.assertEqual(data["numeric_checks"], [])
        self.protocol_id = ident
        package, _, _ = self.package()
        with self.assertRaisesRegex(Invalid, "frozen"):
            compare_computation(self.store, self.problem_id, package)

    def test_legacy_unknown_readiness_cannot_gain_comparison_eligibility(self):
        legacy = self.store.get(self.protocol_id)["data"]
        legacy.pop("execution_readiness")
        legacy.pop("blocking_reasons")
        self.protocol_id = self.store.put("experiment_protocol", legacy, self.problem_id)
        package, _, output = self.package()
        old = self.store.put("experiment_comparison", {
            "package_id": package, "output_digest": self.store.get(output)["digest"],
            "status": "promising_under_model", "promotion_allowed": False,
        }, self.problem_id)
        before = self.store.get(old)
        comparison = self.store.get(compare_computation(self.store, self.problem_id, package))["data"]
        self.assertEqual(comparison["execution_readiness"], "unknown")
        self.assertEqual(comparison["comparisons"][0]["status"], "not_established")
        self.assertEqual(comparison["pareto_alternative_ids"], [])
        self.assertEqual(comparison["numerical_verification"]["status"], "unavailable")
        self.assertFalse(comparison["promotion_allowed"])
        self.assertEqual(self.store.get(old), before)

    def test_comparison_recomputes_direction_and_is_immutable_on_retry(self):
        package_id, run_id, _ = self.package()
        protocol_before = self.store.get(self.protocol_id)
        comparison_id = compare_computation(self.store, self.problem_id, package_id)
        data = self.store.get(comparison_id)["data"]
        self.assertEqual(
            data["comparisons"][0]["improvement_deltas"], {"quality": 2.0, "cost": 1.0}
        )
        self.assertEqual(data["comparisons"][0]["status"], "meets_declared_comparison")
        self.assertEqual(data["pareto_alternative_ids"], ["perturbed"])
        self.assertFalse(data["promotion_allowed"])
        self.assertFalse(data["independently_validated"])
        self.assertEqual(data["basis"], "synthetic")
        self.assertEqual(data["run_id"], run_id)
        self.assertEqual(data["evaluator_version"], EVALUATOR_VERSION)
        self.assertTrue(data["numerical_verification"]["all_passed"])
        verification = self.store.get(data["numerical_verification_id"])
        self.assertEqual(verification["kind"], "numerical_verification")
        self.assertEqual(verification["data"]["run_id"], run_id)
        self.assertEqual(len(verification["data"]["checks"]), 2)
        self.assertEqual(
            compare_computation(self.store, self.problem_id, package_id), comparison_id
        )
        self.assertEqual(len(self.store.list("experiment_comparison")), 1)
        self.assertEqual(len(self.store.list("numerical_verification")), 1)
        self.assertEqual(self.store.get(self.protocol_id), protocol_before)

    def test_false_maker_pass_flags_cannot_override_host_csv_failure(self):
        package_id, _, _ = self.package(
            csv_text="alternative,quality,cost\nbaseline,5,10\nperturbed,-7,9\n"
        )
        data = self.store.get(
            compare_computation(self.store, self.problem_id, package_id)
        )["data"]
        candidate = data["comparisons"][0]
        self.assertTrue(candidate["maker_diagnostics_passed"])
        self.assertFalse(candidate["host_numerical_checks_passed"])
        self.assertFalse(candidate["diagnostics_passed"])
        self.assertEqual(candidate["status"], "not_established")
        self.assertEqual(data["pareto_alternative_ids"], [])
        self.assertEqual(
            data["numerical_verification"]["failed_check_ids"], ["fixture_bounds"]
        )
        self.assertFalse(data["promotion_allowed"])

    def test_legacy_protocol_no_longer_qualifies_and_old_comparison_is_preserved(self):
        legacy = self.store.get(self.protocol_id)["data"]
        legacy.pop("numeric_checks")
        self.protocol_id = self.store.put(
            "experiment_protocol", legacy, self.problem_id
        )
        parsed = ExperimentProtocol.model_validate(
            {k: v for k, v in legacy.items() if k in ExperimentProtocol.model_fields}
        )
        self.assertEqual(parsed.numeric_checks, [])
        package_id, _, output_id = self.package()
        old = self.store.put(
            "experiment_comparison",
            {
                "package_id": package_id,
                "output_digest": self.store.get(output_id)["digest"],
                "status": "promising_under_model",
            },
            self.problem_id,
        )
        original = self.store.get(old)
        new = compare_computation(self.store, self.problem_id, package_id)
        self.assertNotEqual(new, old)
        self.assertEqual(self.store.get(old), original)
        data = self.store.get(new)["data"]
        self.assertEqual(data["numerical_verification"]["status"], "unavailable")
        self.assertFalse(data["numerical_verification"]["all_passed"])
        self.assertEqual(data["comparisons"][0]["status"], "not_established")
        self.assertEqual(data["pareto_alternative_ids"], [])
        self.assertFalse(data["promotion_allowed"])
        self.assertEqual(
            compare_computation(self.store, self.problem_id, package_id), new
        )

    def test_retry_rechecks_csv_integrity_and_never_returns_stale_verification(self):
        package_id, _, _ = self.package()
        ident = compare_computation(self.store, self.problem_id, package_id)
        verification_id = self.store.get(ident)["data"]["numerical_verification_id"]
        self.store.invalidate(verification_id)
        with self.assertRaises(Invalid):
            compare_computation(self.store, self.problem_id, package_id)

        package_id, _, _ = self.package()
        compare_computation(self.store, self.problem_id, package_id)
        package = self.store.get(package_id)
        csv_record = next(
            self.store.get(i)
            for i in package["data"]["file_ids"]
            if self.store.get(i)["data"]["filename"] == "trajectory.csv"
        )
        (self.store.root / "blobs" / csv_record["data"]["sha256"]).write_bytes(
            b"corrupted"
        )
        with self.assertRaisesRegex(Invalid, "integrity"):
            compare_computation(self.store, self.problem_id, package_id)

    def test_protocol_must_be_in_the_actual_run_input_manifest(self):
        protocol = self.store.get(self.protocol_id)
        for manifest in (
            [],
            [{"id": self.protocol_id, "digest": "wrong"}],
            [{"id": "another_protocol", "digest": protocol["digest"]}],
        ):
            with self.subTest(manifest=manifest):
                package_id, _, _ = self.package(manifest=manifest)
                with self.assertRaisesRegex(Invalid, "frozen and supplied before"):
                    compare_computation(self.store, self.problem_id, package_id)
        self.assertFalse(self.store.list("experiment_comparison"))

    def test_later_protocol_cannot_be_retrofitted_even_with_matching_manifest(self):
        package_id, run_id, _ = self.package()
        # Simulate an invalid source chronology while retaining matching content digests.
        # Metadata setup belongs to this test; no protocol/artifact content is mutated.
        with self.store.db() as db:
            db.execute(
                "UPDATE records SET created=? WHERE id=?",
                (self.store.get(run_id)["created"] + 1, self.protocol_id),
            )
        with self.assertRaisesRegex(Invalid, "frozen and supplied before"):
            compare_computation(self.store, self.problem_id, package_id)

    def test_comparison_requires_matching_problem_and_protocol_system_inputs(self):
        records = [
            self.store.get(ident)
            for ident in (self.problem_id, self.system_id, self.protocol_id)
        ]
        valid = [{k: r[k] for k in ("id", "kind", "digest")} for r in records]
        other_system = self.store.put("complex_system", self.system, self.problem_id)
        for index in (0, 1):
            missing = [r for i, r in enumerate(valid) if i != index]
            wrong_digest = copy.deepcopy(valid)
            wrong_digest[index]["digest"] = "not-the-executed-digest"
            for manifest in (missing, wrong_digest):
                with self.subTest(index=index, manifest=manifest):
                    package, _, _ = self.package(manifest=manifest)
                    with self.assertRaises(Invalid):
                        compare_computation(self.store, self.problem_id, package)
        # Equal contents under another system ID cannot substitute for the frozen ID.
        replaced_system = copy.deepcopy(valid)
        replaced_system[1]["id"] = other_system
        package, _, _ = self.package(manifest=replaced_system)
        with self.assertRaisesRegex(Invalid, "protocol system"):
            compare_computation(self.store, self.problem_id, package)
        self.assertFalse(self.store.list("experiment_comparison"))
        self.assertFalse(self.store.list("numerical_verification"))

    def test_comparison_rejects_unfrozen_or_wrong_system_digest_protocol(self):
        frozen = self.store.get(self.protocol_id)["data"]
        for change in ({"status": "proposed"}, {"system_digest": "different-system"}):
            with self.subTest(change=change):
                self.protocol_id = self.store.put(
                    "experiment_protocol", dict(frozen, **change), self.problem_id
                )
                package, _, _ = self.package()
                with self.assertRaises(Invalid):
                    compare_computation(self.store, self.problem_id, package)
        self.assertFalse(self.store.list("experiment_comparison"))

    def test_comparison_requires_generated_outputs_from_completed_bound_run(self):
        mutations = [
            {"calls": [{"status": "failed"}]},
            {"calls": []},
            {"basis": "user_context"},
            {"filename": "results.json"},
            {"output_run_id": "another_run"},
        ]
        for options in mutations:
            with self.subTest(options=options):
                package_id, _, _ = self.package(**options)
                with self.assertRaises(Invalid):
                    compare_computation(self.store, self.problem_id, package_id)
        self.assertFalse(self.store.list("experiment_comparison"))

    def test_alternative_metric_and_diagnostic_coverage_is_exact(self):
        def duplicate_alternative(d):
            d["trials"][1]["alternative_id"] = "baseline"

        def extra_alternative(d):
            d["trials"].append(trial("invented", 8.0, 9.0))

        mutations = [
            duplicate_alternative,
            extra_alternative,
            lambda d: d["trials"][1]["metrics"].pop(),
            lambda d: d["trials"][1]["metrics"][1].update(metric_id="quality"),
            lambda d: d["trials"][1]["metrics"][1].update(metric_id="invented"),
            lambda d: d["trials"][1]["checks"][0].update(check_id="invented"),
            lambda d: d["trials"][1]["checks"].append(
                copy.deepcopy(d["trials"][1]["checks"][0])
            ),
            lambda d: d["trials"][1].update(checks=[]),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                output = self.result()
                mutate(output)
                package_id, _, _ = self.package(output)
                with self.assertRaises((Invalid, ValidationError)):
                    compare_computation(self.store, self.problem_id, package_id)
        self.assertFalse(self.store.list("experiment_comparison"))

    def test_nonfinite_incomplete_or_invalid_intervals_are_rejected(self):
        for change in (
            {"value": float("nan")},
            {"value": float("inf")},
            {"lower": 0.0},
            {"lower": 8.0, "upper": 10.0},
            {"lower": 1.0, "upper": float("inf")},
            {"value": "7.0"},
            {"value": True},
        ):
            with self.subTest(change=change):
                output = self.result()
                output["trials"][1]["metrics"][0].update(change)
                package_id, _, _ = self.package(output)
                with self.assertRaises(ValidationError):
                    compare_computation(self.store, self.problem_id, package_id)
        self.assertFalse(self.store.list("experiment_comparison"))

    def test_failed_diagnostics_block_candidate_and_baseline_claims(self):
        for failed in ("baseline", "perturbed"):
            with self.subTest(failed=failed):
                output = self.result()
                next(t for t in output["trials"] if t["alternative_id"] == failed)[
                    "checks"
                ][0]["passed"] = False
                package_id, _, _ = self.package(output)
                ident = compare_computation(self.store, self.problem_id, package_id)
                result = self.store.get(ident)["data"]
                self.assertFalse(result["comparisons"][0]["diagnostics_passed"])
                self.assertEqual(result["comparisons"][0]["status"], "not_established")
                self.assertNotIn(failed, result["pareto_alternative_ids"])

    def test_pareto_frontier_preserves_tradeoffs_without_declaring_them_improvements(
        self,
    ):
        # The richer alternative is part of both system design and frozen protocol.
        system = copy.deepcopy(self.system)
        system["decision"]["alternatives"].append(
            {
                "id": "economical",
                "name": "Low cost condition",
                "interventions": [],
                "rationale": "Compare resource tradeoff.",
            }
        )
        system["experiments"][0]["candidate_alternative_ids"].append("economical")
        system_id = self.store.put("complex_system", system, self.problem_id)
        proposal = protocol_spec()
        proposal["candidate_ids"].append("economical")
        self.protocol_id = plan_experiment(
            self.store, FakeProposer(proposal), self.problem_id, system_id
        )
        output = self.result()
        output["trials"] = [
            trial("baseline", 5.0, 10.0),
            trial("perturbed", 7.0, 12.0),
            trial("economical", 4.0, 8.0),
        ]
        package_id, _, _ = self.package(output)
        data = self.store.get(
            compare_computation(self.store, self.problem_id, package_id)
        )["data"]
        self.assertEqual(
            set(data["pareto_alternative_ids"]), {"baseline", "perturbed", "economical"}
        )
        self.assertEqual(
            {c["status"] for c in data["comparisons"]}, {"not_established"}
        )

    def test_stale_or_foreign_artifacts_cannot_supply_comparison_or_review(self):
        package_id, _, _ = self.package()
        other_problem = self.store.put(
            "workspace_problem", {"question": "Unrelated problem"}
        )
        with self.assertRaises(Invalid):
            compare_computation(self.store, other_problem, package_id)
        self.store.invalidate(self.system_id)
        with self.assertRaises(Invalid):
            compare_computation(self.store, self.problem_id, package_id)
        with self.assertRaises(Invalid):
            current_artifact(
                self.store, self.problem_id, "complex_system", self.system_id
            )

    def test_agent_representation_references_only_the_received_context(self):
        provider = FakeProposer(self.system)
        ident = represent_system(
            self.store, provider, self.problem_id, "Use the supplied context."
        )
        data = self.store.get(ident)["data"]
        self.assertEqual(data["status"], "proposed")
        self.assertEqual(data["execution_status"], "not_executed")
        self.assertEqual(data["parent_system_id"], self.system_id)
        self.assertIn(self.context_id, {a["id"] for a in data["input_manifest"]})
        self.assertEqual(provider.calls[0][1]["user_task"], "Use the supplied context.")
        with patch("symplex.agents.scientific_cycle.working_context", return_value=[]):
            with self.assertRaisesRegex(ValueError, "source artifact"):
                represent_system(self.store, FakeProposer(self.system), self.problem_id)

    def test_hypothesis_reviewer_uses_validator_and_covers_every_current_hypothesis(
        self,
    ):
        review = {
            "assessments": [
                {
                    "hypothesis_id": h["id"],
                    "status": "testable",
                    "issue": "Measurements remain unavailable.",
                    "discriminating_test": h["discriminating_test"],
                    "confounders": ["Measurement error."],
                    "evidence_ids": [self.context_id],
                }
                for h in self.system["hypotheses"]
            ],
            "coverage_gaps": ["No observations yet."],
            "shared_failure_modes": ["Both candidates could be misspecified."],
            "next_action": "Obtain discriminating evidence.",
        }
        provider = FakeProposer(review)
        ident = review_hypotheses(self.store, provider, self.problem_id, self.system_id)
        self.assertEqual(provider.calls[0][3]["role"], "validator")
        self.assertEqual(
            self.store.get(ident)["data"]["system_digest"],
            self.store.get(self.system_id)["digest"],
        )
        changes = [
            lambda r: r["assessments"].pop(),
            lambda r: r["assessments"][1].update(hypothesis_id="memoryless"),
            lambda r: r["assessments"][1].update(hypothesis_id="invented"),
            lambda r: r["assessments"][1].update(evidence_ids=["invented_artifact"]),
        ]
        for index, mutate in enumerate(changes):
            with self.subTest(mutation=index):
                changed = copy.deepcopy(review)
                mutate(changed)
                with self.assertRaises(Invalid):
                    review_hypotheses(
                        self.store,
                        FakeProposer(changed),
                        self.problem_id,
                        self.system_id,
                    )
        self.assertEqual(len(self.store.list("hypothesis_review")), 1)

    def test_protocol_and_agent_output_contracts_are_strict(self):
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

        for contract in (ExperimentProtocol, ExperimentOutput, HypothesisReview):
            # The API contract requires new fields; host deserialization also reads
            # immutable legacy protocols that predate numerical checks.
            inspect(
                contract.json_schema()
                if hasattr(contract, "json_schema")
                else contract.model_json_schema()
            )
        for changed in (
            dict(protocol_spec(), candidate_ids=["perturbed", "perturbed"]),
            dict(protocol_spec(), candidate_ids=["baseline"]),
            dict(
                protocol_spec(),
                required_check_ids=["instrument_check", "instrument_check"],
            ),
            dict(protocol_spec(), fabricated_promotion=True),
        ):
            with self.assertRaises(ValidationError):
                ExperimentProtocol.parse(changed)
        output = self.result()
        output["independently_validated"] = True
        with self.assertRaises(ValidationError):
            ExperimentOutput.model_validate(output)


if __name__ == "__main__":
    unittest.main()
