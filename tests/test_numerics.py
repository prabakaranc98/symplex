"""Deterministic CSV diagnostics, with synthetic fixtures and no code execution."""

import copy
import json
import tempfile
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from symplex.connectors.compute import save_blob
from symplex.core.contracts import Invalid
from symplex.evaluation.numerics import NumericCheck, execute_numeric_checks
from symplex.infrastructure.storage import Store


def check_spec(operation="finite", **updates):
    spec = {
        "id": "numerical_check",
        "operation": operation,
        "csv_filename": "trajectory.csv",
        "columns": ["x"],
        "group_columns": [],
        "time_column": None,
        "row_filters": [],
        "lower": 0.0 if operation == "bounds" else None,
        "upper": None,
        "reference_value": 10.0 if operation == "sum_conservation" else None,
        "tolerance": 0.0,
        "direction": "increasing" if operation == "monotonic" else None,
        "units": "synthetic mass units",
        "rationale": "Software fixture only; no measured physical quantity is established.",
    }
    if operation == "monotonic":
        spec["time_column"] = "time"
    spec.update(updates)
    return spec


class NumericCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name)
        self.problem = self.store.put(
            "workspace_problem", {"question": "Synthetic fixture"}
        )
        self.run = self.store.put(
            "compute_run", {"calls": [{"status": "completed"}]}, self.problem
        )

    def csv(self, content, filename="trajectory.csv", **kwargs):
        return save_blob(
            self.store,
            content.encode() if isinstance(content, str) else content,
            filename,
            kwargs.get("problem", self.problem),
            kwargs.get("basis", "generated"),
            kwargs.get("run", self.run),
        )

    def execute(self, ids, checks, **kwargs):
        return execute_numeric_checks(
            self.store,
            kwargs.get("problem", self.problem),
            kwargs.get("run", self.run),
            ids,
            checks,
        )

    def test_finite_checks_recompute_values_and_attach_exact_source_provenance(self):
        ident = self.csv("group,time,x,y\na,0,1,9\na,1,2,8\nb,0,3,7\n")
        checks = [
            check_spec(columns=["x", "y"], group_columns=["group"], time_column="time")
        ]
        result = self.execute([ident], checks)
        self.assertTrue(result["all_passed"])
        self.assertEqual(result["status"], "checked")
        self.assertEqual(result["checks"][0]["comparison_count"], 6)
        self.assertEqual(result["checks"][0]["group_count"], 2)
        source = result["source_manifest"][0]
        self.assertEqual(source["id"], ident)
        self.assertEqual(source["sha256"], self.store.get(ident)["data"]["sha256"])
        self.assertEqual(source["digest"], self.store.get(ident)["digest"])
        self.assertEqual(source["run_id"], self.run)
        self.assertEqual(
            result["scope"],
            "host numerical checks of generated outputs; no empirical validation",
        )
        self.assertFalse(result["independently_validated"])
        self.assertEqual(result, self.execute([ident], checks))
        json.dumps(result, allow_nan=False)

    def test_bounds_ignore_maker_flags_and_report_bounded_failure_examples(self):
        ident = self.csv("x,maker_passed\n" + "-2,true\n" * 7 + "12,true\n")
        result = self.execute(
            [ident], [check_spec("bounds", upper=10.0, tolerance=0.5)]
        )
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["all_passed"])
        checked = result["checks"][0]
        self.assertEqual(checked["violation_count"], 8)
        self.assertEqual(checked["observed_worst_violation"], 2.0)
        self.assertEqual(checked["worst_excess_over_tolerance"], 1.5)
        self.assertEqual(checked["example_count"], 5)
        self.assertEqual(checked["examples"][0]["row"], 2)

    def test_bounds_and_conservation_use_absolute_frozen_tolerance(self):
        ident = self.csv("x,y\n-0.125,10.125\n1,9.125\n")
        checks = [
            check_spec("bounds", id="nonnegative", tolerance=0.125),
            check_spec(
                "sum_conservation", id="mass", columns=["x", "y"], tolerance=0.125
            ),
        ]
        self.assertTrue(self.execute([ident], checks)["all_passed"])
        checks[1]["tolerance"] = 0.0625
        result = self.execute([ident], checks)
        self.assertTrue(result["checks"][0]["passed"])
        self.assertFalse(result["checks"][1]["passed"])
        self.assertEqual(result["checks"][1]["observed_worst_violation"], 0.125)
        self.assertEqual(result["csv_rows_read"], 2)

    def test_monotonic_sorts_time_and_checks_each_group_separately(self):
        ident = self.csv("group,time,x\na,2,3\nb,1,12\na,0,1\nb,0,10\na,1,2\n")
        result = self.execute(
            [ident], [check_spec("monotonic", group_columns=["group"])]
        )
        self.assertTrue(result["all_passed"])
        self.assertEqual(result["checks"][0]["comparison_count"], 3)
        self.assertEqual(result["checks"][0]["group_count"], 2)

    def test_monotonic_detects_accumulated_reversal_and_decreasing_direction(self):
        ident = self.csv("time,x\n0,1\n1,0.875\n2,0.75\n")
        result = self.execute([ident], [check_spec("monotonic", tolerance=0.125)])
        self.assertFalse(result["all_passed"])
        self.assertEqual(result["checks"][0]["observed_worst_violation"], 0.25)
        self.assertEqual(result["checks"][0]["examples"][0]["reference_row"], 2)
        result = self.execute(
            [ident], [check_spec("monotonic", direction="decreasing")]
        )
        self.assertTrue(result["all_passed"])

    def test_filters_are_exact_and_empty_selection_cannot_pass(self):
        ident = self.csv("fixture,x,y\nsmall,2,8\nlarge,3,17\n")
        check = check_spec(
            "sum_conservation",
            columns=["x", "y"],
            row_filters=[{"column": "fixture", "equals": "small"}],
        )
        result = self.execute([ident], [check])
        self.assertTrue(result["all_passed"])
        self.assertEqual(result["checks"][0]["selected_row_count"], 1)
        check["row_filters"][0]["equals"] = "absent"
        with self.assertRaisesRegex(Invalid, "selected no rows"):
            self.execute([ident], [check])
        check["row_filters"] = []
        self.assertFalse(self.execute([ident], [check])["all_passed"])

    def test_malformed_nonfinite_missing_and_duplicate_data_fail_closed(self):
        cases = [
            ("", check_spec()),
            ("x\n", check_spec()),
            ("x,x\n1,1\n", check_spec()),
            ("y\n1\n", check_spec()),
            ("x,y\n1\n", check_spec()),
            ("x\n1,2\n", check_spec()),
            ("x\nNaN\n", check_spec()),
            ("x\nInfinity\n", check_spec()),
            ('x\n""\n', check_spec()),
            ("x\ntrue\n", check_spec()),
            ('x\n"unterminated\n', check_spec()),
            ("time,x\n1,1\n1.0,2\n", check_spec("monotonic")),
            ("time,x\nNaN,1\n2,2\n", check_spec("monotonic")),
            ("time,x\n1,1\n", check_spec("monotonic")),
            (
                "group,time,x\n,1,1\n,2,2\n",
                check_spec("monotonic", group_columns=["group"]),
            ),
        ]
        for content, check in cases:
            with self.subTest(content=content), self.assertRaises(Invalid):
                self.execute([self.csv(content)], [check])

    def test_overflow_does_not_emit_infinite_observations(self):
        ident = self.csv("x,y\n1e308,1e308\n")
        with self.assertRaisesRegex(Invalid, "overflow"):
            self.execute([ident], [check_spec("sum_conservation", columns=["x", "y"])])
        ident = self.csv("x\n1e308\n")
        with self.assertRaisesRegex(Invalid, "overflow"):
            self.execute([ident], [check_spec("bounds", lower=None, upper=-1e308)])

    def test_scope_completion_and_file_run_binding_are_required(self):
        other = self.store.put("workspace_problem", {"question": "Other fixture"})
        failed_run = self.store.put(
            "compute_run", {"calls": [{"status": "failed"}]}, self.problem
        )
        cases = [
            (["file_blob_missing"], {}),
            ([self.csv("x\n1\n", problem=other)], {}),
            ([self.csv("x\n1\n", run=failed_run)], {}),
            ([self.csv("x\n1\n", basis="user_context")], {}),
            ([self.csv("x\n1\n", run=failed_run)], {"run": failed_run}),
        ]
        stale = self.csv("x\n1\n")
        self.store.invalidate(stale)
        cases.append(([stale], {}))
        for ids, args in cases:
            with self.subTest(ids=ids), self.assertRaises(Invalid):
                self.execute(ids, [check_spec()], **args)

    def test_ambiguous_missing_duplicate_check_and_file_selection_rejects(self):
        first, second = self.csv("x\n1\n"), self.csv("x\n2\n")
        cases = [
            ([first, second], [check_spec()]),
            ([first, first], [check_spec()]),
            ([first], [check_spec(csv_filename="missing.csv")]),
            ([first], [check_spec(), check_spec()]),
            ([first], []),
            ([], [check_spec()]),
        ]
        for ids, checks in cases:
            with self.subTest(ids=ids, checks=checks), self.assertRaises(Invalid):
                self.execute(ids, checks)

    def test_checksum_and_full_csv_resource_envelopes_are_enforced(self):
        corrupt = self.csv("x\n1\n")
        record = self.store.get(corrupt)
        (self.store.root / "blobs" / record["data"]["sha256"]).write_bytes(b"corrupted")
        with self.assertRaisesRegex(Invalid, "integrity"):
            self.execute([corrupt], [check_spec()])
        for content, message in (
            (b"x\n" + b"1\n" * 2_500_000, "5 MB"),
            (b"x\n" + b"1\n" * 100_001, "100000-row"),
        ):
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(Invalid, message),
            ):
                self.execute([self.csv(content)], [check_spec()])

    def test_resource_limits_aggregate_distinct_csvs_not_repeated_checks(self):
        first = self.csv("x\n2\n3\n")
        second = self.csv("x\n4\n5\n", filename="second.csv")
        checks = [
            check_spec(id="first"),
            check_spec(id="second", csv_filename="second.csv"),
        ]
        with (
            patch("symplex.evaluation.numerics.MAX_CSV_ROWS", 3),
            self.assertRaisesRegex(Invalid, "row limit"),
        ):
            self.execute([first, second], checks)
        with (
            patch("symplex.evaluation.numerics.MAX_CSV_BYTES", 10),
            self.assertRaisesRegex(Invalid, "CSV limit"),
        ):
            self.execute([first, second], checks)
        unused = self.csv("x\n9\n", filename="unused.csv")
        checks[1]["csv_filename"] = "trajectory.csv"
        result = self.execute([first, unused], checks)
        self.assertEqual(result["csv_rows_read"], 2)
        self.assertEqual([s["id"] for s in result["source_manifest"]], [first])

    def test_contract_is_closed_strict_frozen_and_operation_specific(self):
        valid = NumericCheck.model_validate(check_spec())
        with self.assertRaises(ValidationError):
            valid.tolerance = 1.0
        mutations = [
            {"unknown": True},
            {"tolerance": "0"},
            {"tolerance": True},
            {"tolerance": -1.0},
            {"tolerance": float("nan")},
            {"columns": []},
            {"columns": ["x", "x"]},
            {"group_columns": ["x"]},
            {"time_column": "x"},
            {"csv_filename": "../trajectory.csv"},
            {"csv_filename": "model.py"},
            {"direction": "increasing"},
            {"reference_value": 1.0},
            {"lower": 0.0},
            {"tolerance": 0.1},
        ]
        for changes in mutations:
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                NumericCheck.model_validate({**copy.deepcopy(check_spec()), **changes})
        for check in (
            check_spec("bounds", lower=2.0, upper=1.0),
            check_spec("bounds", lower=None),
            check_spec("sum_conservation", reference_value=None),
            check_spec("monotonic", direction=None),
            check_spec("monotonic", time_column=None),
        ):
            with self.subTest(check=check), self.assertRaises(ValidationError):
                NumericCheck.model_validate(check)


if __name__ == "__main__":
    unittest.main()
