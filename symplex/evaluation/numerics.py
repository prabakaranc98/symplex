"""Host-owned numerical checks of bounded generated CSV; never runs model code."""

import csv
import hashlib
import io
import math
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, model_validator

from symplex.core.contracts import Invalid, digest
from symplex.modeling.complex_system import ClosedContract, Identifier, Text

MAX_CSV_BYTES = 5_000_000
MAX_CSV_ROWS = 100_000
MAX_CHECKS = 32
SCOPE = "host numerical checks of generated outputs; no empirical validation"
Column = Annotated[str, Field(min_length=1, max_length=128)]


class RowFilter(ClosedContract):
    """An exact string comparison, never an expression or executable predicate."""

    model_config = ConfigDict(frozen=True)
    column: Column
    equals: Annotated[str, Field(min_length=1, max_length=256)]


class NumericCheck(ClosedContract):
    model_config = ConfigDict(frozen=True)
    id: Identifier
    operation: Literal["finite", "bounds", "sum_conservation", "monotonic"]
    csv_filename: Annotated[str, Field(min_length=5, max_length=200)]
    columns: list[Column] = Field(min_length=1, max_length=32)
    group_columns: list[Column] = Field(max_length=8)
    time_column: Column | None
    row_filters: list[RowFilter] = Field(max_length=8)
    lower: float | None
    upper: float | None
    reference_value: float | None
    tolerance: float = Field(ge=0)
    direction: Literal["increasing", "decreasing"] | None
    units: Text
    rationale: Text

    @model_validator(mode="after")
    def coherent(self):
        if (
            not self.csv_filename.lower().endswith(".csv")
            or "/" in self.csv_filename
            or "\\" in self.csv_filename
        ):
            raise ValueError("csv_filename must be a CSV basename")
        for names in (
            self.columns,
            self.group_columns,
            [f.column for f in self.row_filters],
        ):
            if len(names) != len(set(names)):
                raise ValueError("Column and filter identifiers must be unique")
        if set(self.columns) & set(self.group_columns):
            raise ValueError("Numeric and group columns must be distinct")
        if self.time_column in set(self.columns) | set(self.group_columns):
            raise ValueError("Time must be distinct from value and group columns")
        if self.operation == "bounds":
            if self.lower is None and self.upper is None:
                raise ValueError("Bounds requires a lower or upper limit")
            if (
                self.lower is not None
                and self.upper is not None
                and self.lower > self.upper
            ):
                raise ValueError("Lower limit exceeds upper limit")
        elif self.lower is not None or self.upper is not None:
            raise ValueError("Only bounds uses lower and upper")
        if (self.reference_value is not None) != (self.operation == "sum_conservation"):
            raise ValueError("Only sum_conservation requires reference_value")
        if self.operation == "monotonic":
            if self.direction is None or self.time_column is None:
                raise ValueError("Monotonic requires direction and numeric time_column")
        elif self.direction is not None:
            raise ValueError("Only monotonic uses direction")
        if self.operation == "finite" and self.tolerance != 0:
            raise ValueError("Finite checks have zero tolerance")
        return self


def _record(store, ident, kind, problem_id=None):
    if not isinstance(ident, str) or not ident or len(ident) > 200:
        raise Invalid("Expected a numerical-check artifact ID")
    try:
        record = store.get(ident)
    except KeyError:
        raise Invalid("Missing numerical-check source: " + str(ident)) from None
    if (
        record["kind"] != kind
        or record["stale"]
        or (problem_id is not None and record["parent"] != problem_id)
    ):
        raise Invalid(
            "Numerical-check sources must be current and belong to this problem"
        )
    return record


def _read_csv(store, record, bytes_remaining, rows_remaining):
    sha = record["data"].get("sha256", "")
    if (
        not isinstance(sha, str)
        or len(sha) != 64
        or any(c not in "0123456789abcdef" for c in sha)
    ):
        raise Invalid("Invalid CSV digest")
    try:
        with (store.root / "blobs" / sha).open("rb") as stream:
            raw = stream.read(bytes_remaining + 1)
    except OSError:
        raise Invalid("CSV content is unavailable") from None
    if len(raw) > bytes_remaining:
        raise Invalid("Numerical checks exceed the total 5 MB CSV limit")
    if hashlib.sha256(raw).hexdigest() != sha or len(raw) != record["data"].get("size"):
        raise Invalid("CSV content integrity failure")
    try:
        reader = csv.reader(
            io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True
        )
        header = next(reader, [])
        if (
            not header
            or len(header) > 256
            or any(not c or c != c.strip() for c in header)
        ):
            raise Invalid("CSV needs nonempty, unambiguous column names")
        if len(header) != len(set(header)):
            raise Invalid("CSV contains duplicate column names")
        rows = []
        for line, values in enumerate(reader, start=2):
            if len(rows) >= rows_remaining:
                raise Invalid("Numerical checks exceed the total 100000-row limit")
            if len(values) != len(header):
                raise Invalid(
                    "CSV contains a missing or extra field at row " + str(line)
                )
            rows.append((line, dict(zip(header, values))))
    except (UnicodeError, csv.Error):
        raise Invalid("Expected well-formed UTF-8 CSV") from None
    if not rows:
        raise Invalid("Numerical checks cannot pass on an empty CSV")
    return header, rows, len(raw)


def _number(value, column, row):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise Invalid(f"Missing or nonnumeric value in {column} at row {row}") from None
    if not math.isfinite(number):
        raise Invalid(f"Nonfinite value in {column} at row {row}")
    return number


def _evaluate(check, header, rows, source):
    needed = set(
        check.columns + check.group_columns + [f.column for f in check.row_filters]
    )
    if check.time_column:
        needed.add(check.time_column)
    if not needed <= set(header):
        raise Invalid(
            "CSV is missing declared columns: "
            + ", ".join(sorted(needed - set(header)))
        )
    selected = [
        (line, row)
        for line, row in rows
        if all(row[f.column] == f.equals for f in check.row_filters)
    ]
    if not selected:
        raise Invalid("Numeric check filters selected no rows: " + check.id)
    groups = {}
    for line, row in selected:
        key = tuple(row[c] for c in check.group_columns)
        if any(not v.strip() for v in key):
            raise Invalid("Missing group value at row " + str(line))
        values = {c: _number(row[c], c, line) for c in check.columns}
        time = (
            _number(row[check.time_column], check.time_column, line)
            if check.time_column
            else None
        )
        groups.setdefault(key, []).append((time, line, values))
    for entries in groups.values():
        if check.time_column:
            times = [r[0] for r in entries]
            if len(times) != len(set(times)):
                raise Invalid("Duplicate group/time rows in check " + check.id)
            entries.sort(key=lambda r: r[0])
        if check.operation == "monotonic" and len(entries) < 2:
            raise Invalid("Monotonic checks require at least two times in every group")

    comparisons = 0
    violations = 0
    worst = 0.0
    examples = []

    def observe(violation, line, key, detail):
        nonlocal comparisons, violations, worst
        if not math.isfinite(violation):
            raise Invalid("Numerical check arithmetic overflow")
        comparisons += 1
        worst = max(worst, violation)
        if violation > check.tolerance:
            violations += 1
            if len(examples) < 5:
                examples.append(
                    {
                        "row": line,
                        "group": dict(zip(check.group_columns, key)),
                        "violation": violation,
                        **detail,
                    }
                )

    for key, entries in groups.items():
        extremes = {}
        for time, line, values in entries:
            if check.operation == "sum_conservation":
                try:
                    total = math.fsum(values.values())
                except OverflowError:
                    raise Invalid("Numerical check arithmetic overflow") from None
                observe(
                    abs(total - check.reference_value),
                    line,
                    key,
                    {"sum": total, "reference_value": check.reference_value},
                )
                continue
            for column, value in values.items():
                if check.operation == "monotonic":
                    if column in extremes:
                        extreme, reference_row = extremes[column]
                        reversal = (
                            extreme - value
                            if check.direction == "increasing"
                            else value - extreme
                        )
                        observe(
                            max(0.0, reversal),
                            line,
                            key,
                            {
                                "column": column,
                                "value": value,
                                "reference_value": extreme,
                                "reference_row": reference_row,
                                "time": time,
                            },
                        )
                        improved = (
                            value > extreme
                            if check.direction == "increasing"
                            else value < extreme
                        )
                        if improved:
                            extremes[column] = (value, line)
                    else:
                        extremes[column] = (value, line)
                elif check.operation == "bounds":
                    violation = max(
                        0.0,
                        check.lower - value if check.lower is not None else 0.0,
                        value - check.upper if check.upper is not None else 0.0,
                    )
                    observe(violation, line, key, {"column": column, "value": value})
                else:
                    observe(0.0, line, key, {"column": column, "value": value})
    return {
        "id": check.id,
        "operation": check.operation,
        "passed": violations == 0,
        "status": "checked" if violations == 0 else "failed",
        "selected_row_count": len(selected),
        "group_count": len(groups),
        "comparison_count": comparisons,
        "violation_count": violations,
        "observed_worst_violation": worst,
        "worst_excess_over_tolerance": max(0.0, worst - check.tolerance),
        "tolerance": check.tolerance,
        "units": check.units,
        "examples": examples,
        "example_count": len(examples),
        "source": source,
        "check_digest": digest(check.model_dump()),
        "monotonic_semantics": "Maximum reversal from any earlier value within each group, sorted by numeric time"
        if check.operation == "monotonic"
        else None,
    }


def execute_numeric_checks(store, problem_id, run_id, file_ids, checks):
    """Recompute frozen absolute checks; malformed inputs fail closed with Invalid.

    Well-formed numerical violations return failed. All selected values and times
    must be finite. Exact row filters are conjunctive; an empty list selects all
    rows. The byte/row ceilings cover distinct CSVs actually read across checks.
    """
    _record(store, problem_id, "workspace_problem")
    run = _record(store, run_id, "compute_run", problem_id)
    if not any(c.get("status") == "completed" for c in run["data"].get("calls", [])):
        raise Invalid("Numerical checks require a completed sandbox execution")
    if not isinstance(checks, list) or not 1 <= len(checks) <= MAX_CHECKS:
        raise Invalid("Supply between 1 and 32 frozen numerical checks")
    parsed = [
        NumericCheck.model_validate(
            c.model_dump() if isinstance(c, NumericCheck) else c
        )
        for c in checks
    ]
    if len({c.id for c in parsed}) != len(parsed):
        raise Invalid("Numeric check IDs must be unique")
    if (
        not isinstance(file_ids, list)
        or not file_ids
        or len(file_ids) > 128
        or any(not isinstance(i, str) for i in file_ids)
        or len(file_ids) != len(set(file_ids))
    ):
        raise Invalid("Supply unique generated file IDs")
    files = {}
    for ident in file_ids:
        record = _record(store, ident, "file_blob", problem_id)
        data = record["data"]
        if data.get("run_id") != run_id or data.get("basis") != "generated":
            raise Invalid("Numeric CSV must be generated by the selected run")
        name = data.get("filename", "")
        if name.lower().endswith(".csv"):
            if name in files:
                raise Invalid("Ambiguous duplicate CSV filename")
            files[name] = record
    wanted = sorted({c.csv_filename for c in parsed})
    if not set(wanted) <= set(files):
        raise Invalid("A frozen numerical-check CSV is missing from this run")
    tables, sources = {}, {}
    total_bytes = total_rows = 0
    for name in wanted:
        record = files[name]
        header, rows, size = _read_csv(
            store, record, MAX_CSV_BYTES - total_bytes, MAX_CSV_ROWS - total_rows
        )
        total_bytes += size
        total_rows += len(rows)
        tables[name] = (header, rows)
        sources[name] = {
            "id": record["id"],
            "digest": record["digest"],
            "sha256": record["data"]["sha256"],
            "filename": name,
            "run_id": run_id,
            "bytes": size,
            "rows": len(rows),
            "basis": "generated output; observation origin is not established by this check",
        }
    results = [
        _evaluate(c, *tables[c.csv_filename], sources[c.csv_filename]) for c in parsed
    ]
    # Catch invalidation during assembly rather than issuing a verdict on stale inputs.
    _record(store, problem_id, "workspace_problem")
    _record(store, run_id, "compute_run", problem_id)
    for source in sources.values():
        _record(store, source["id"], "file_blob", problem_id)
    passed = all(r["passed"] for r in results)
    return {
        "status": "checked" if passed else "failed",
        "all_passed": passed,
        "checks": results,
        "source_manifest": list(sources.values()),
        "checks_digest": digest([c.model_dump() for c in parsed]),
        "run_id": run_id,
        "run_digest": run["digest"],
        "csv_bytes_read": total_bytes,
        "csv_rows_read": total_rows,
        "independently_validated": False,
        "scope": SCOPE,
    }
