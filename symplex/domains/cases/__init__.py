"""Executable synthetic case packs: a declared generator plus a real inference problem.

Nothing in this package is an observation. Every table produced here comes from a
mechanistic generator whose parameters, seed and planted truth are recorded in the same
module. Recovering that planted structure demonstrates inference mechanics -- that the
engine can find structure it was never told. It does not demonstrate that the mechanism
describes any real bike network, market, catchment, community or cell population, and no
number from these packs may be reported as a domain finding.

The truth is deliberately kept out of the agent-visible payload. `truth()` is host state;
`generate()` is what an agent may read. Tests assert that separation for every case.

Pure stdlib, numpy and scipy. Deterministic under an explicit seed. No network, no model
calls, no subprocess, no cost.
"""

import csv
import hashlib
import io
import math
from typing import Annotated, Literal

import numpy as np
from pydantic import ConfigDict, Field, model_validator
from scipy import stats

from symplex.core.contracts import Invalid, canonical, digest
from symplex.evaluation.numerics import NumericCheck
# The same kernel the host runs over generated compute CSV, so demo tables face the
# identical rules. Reused rather than reimplemented; this package never relaxes it.
from symplex.evaluation.numerics import _evaluate as _numeric_kernel
from symplex.modeling.complex_system import ClosedContract, Identifier, Text

SCOPE = (
    "Synthetic observations from the declared generator in this module. Recovering the "
    "planted truth demonstrates inference mechanics only. It establishes nothing "
    "empirical about the real domain, and no output may be reported as a domain result."
)
TRUTH_SCOPE = (
    "Sealed generator truth held by the host and never placed in the agent-visible "
    "payload. Knowing it is not evidence about any real system; it is the answer key to "
    "a simulation this repository wrote."
)
EVAL_SCOPE = (
    "Host-owned scoring against planted generator truth. A score measures recovery of "
    "declared synthetic structure under a declared noise model. It is not empirical "
    "accuracy in the named domain and is not an independent evaluation."
)
NARRATIVE_SCOPE = (
    "Demo script for a synthetic case pack. The decision framing is realistic; the data "
    "is not real. Say so out loud when demonstrating."
)
STATUS = "executable_synthetic_benchmark"

MAX_SEED = 2**31 - 1
MAX_TABLE_ROWS = 20000
MAX_TABLES = 6
ROUND = 4
ORACLE_CONFIDENCE = 0.92
BOOTSTRAP_RESAMPLES = 400
BOOTSTRAP_SEED = 20260910
EPS = 1e-12

CaseId = Annotated[str, Field(pattern=r"^[a-z][a-z0-9]{0,7}$")]
ManifestCase = Annotated[str, Field(pattern=r"^[A-Z][0-9]$")]
CsvName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{2,60}\.csv$")]
Marker = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{2,60}$")]


class CaseEndpoint(ClosedContract):
    """A quantity the decision actually turns on, with its declared unit."""

    model_config = ConfigDict(frozen=True)
    id: Identifier
    name: Text
    unit: Text
    meaning: Text


class CaseMetric(ClosedContract):
    """A host-computed score. `lower_is_better` fixes the comparison direction."""

    model_config = ConfigDict(frozen=True)
    id: Identifier
    name: Text
    lower_is_better: bool
    definition: Text


class CaseSpec(ClosedContract):
    """Case metadata, endpoints and the decision the pack supports.

    `hidden_markers` are answer tokens. They are excluded from `agent_view()` and are
    asserted absent from every agent-visible payload.
    """

    model_config = ConfigDict(frozen=True)
    id: CaseId
    manifest_case: ManifestCase
    domain: Identifier
    title: Text
    status: Literal["executable_synthetic_benchmark"] = STATUS
    decision: Text
    beneficiary: Text
    mechanism_question: Text
    rivals: list[Text] = Field(min_length=2, max_length=6)
    endpoints: list[CaseEndpoint] = Field(min_length=1, max_length=6)
    evidence_channels: list[Text] = Field(min_length=1, max_length=8)
    nuisance: list[Text] = Field(min_length=1, max_length=10)
    primary_metric: CaseMetric
    secondary_metrics: list[CaseMetric] = Field(max_length=6)
    baseline_name: Text
    baseline_rationale: Text
    oracle_rationale: Text
    trivial_name: Text
    hidden_markers: list[Marker] = Field(min_length=1, max_length=24)
    tables: list[CsvName] = Field(min_length=1, max_length=MAX_TABLES)
    knobs: list[Identifier] = Field(max_length=6)
    limits: Text
    scope: Text = SCOPE

    @model_validator(mode="after")
    def coherent(self):
        metric_ids = [self.primary_metric.id] + [m.id for m in self.secondary_metrics]
        if len(metric_ids) != len(set(metric_ids)):
            raise ValueError("Metric identifiers must be unique")
        for names in (self.tables, self.hidden_markers, [e.id for e in self.endpoints]):
            if len(names) != len(set(names)):
                raise ValueError("Table, marker and endpoint identifiers must be unique")
        if self.scope != SCOPE:
            raise ValueError("Every case pack must carry the shared synthetic scope")
        return self

    def agent_view(self):
        """The metadata an agent may read. Answer tokens are removed here, not later."""
        view = self.model_dump()
        view.pop("hidden_markers")
        return view


def check_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
        raise Invalid("Case seeds must be an integer in [0, 2147483647]")
    return int(seed)


def check_knobs(knobs, allowed):
    """Bounded, typed generator knobs. A demo must stay small enough to run in seconds."""
    if not isinstance(knobs, dict):
        raise Invalid("Generator knobs must be a mapping")
    unknown = set(knobs) - set(allowed)
    if unknown:
        raise Invalid("Unknown generator knob(s): " + ", ".join(sorted(unknown)))
    resolved = {}
    for name, (low, high, default) in allowed.items():
        value = knobs.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise Invalid("Generator knob must be a number: " + name)
        if not math.isfinite(float(value)) or not low <= value <= high:
            raise Invalid(f"Generator knob {name} must lie in [{low}, {high}]")
        resolved[name] = int(value) if isinstance(default, int) else float(value)
    return resolved


def stream(seed, label):
    """Independent, reproducible substream per generator stage; stable across runs."""
    if not isinstance(label, str) or not label:
        raise Invalid("Random substreams need a nonempty label")
    key = int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "big")
    return np.random.default_rng(np.random.SeedSequence(entropy=int(seed), spawn_key=(key,)))


def clean(value):
    """Plain JSON types with fixed float precision, so identical seeds give identical bytes."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            raise Invalid("Case packs never emit nonfinite values")
        return round(number, ROUND) + 0.0
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, (int, str)):
        return value
    raise Invalid("Unsupported value type in a case pack payload: " + type(value).__name__)


def cell(value):
    """One CSV cell. `None` becomes an empty field: the missingness is visible, not imputed."""
    if value is None:
        return ""
    if isinstance(value, (np.floating, float)):
        return repr(round(float(value), ROUND) + 0.0)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, str):
        return value
    raise Invalid("Unsupported CSV cell type: " + type(value).__name__)


def table(columns, rows):
    """A tidy, CSV-shaped table: declared columns, bounded rows, explicit missingness."""
    if not isinstance(columns, (list, tuple)) or not columns:
        raise Invalid("A case table needs at least one declared column")
    columns = [str(c) for c in columns]
    if len(columns) != len(set(columns)):
        raise Invalid("Case table columns must be unique")
    if not isinstance(rows, list) or not rows or len(rows) > MAX_TABLE_ROWS:
        raise Invalid(f"Case tables carry 1 to {MAX_TABLE_ROWS} rows")
    records = []
    for row in rows:
        if set(row) != set(columns):
            raise Invalid("Every case-table row must fill exactly the declared columns")
        records.append({c: clean(row[c]) if row[c] is not None else None for c in columns})
    return {"columns": columns, "row_count": len(records), "rows": records}


def to_csv_text(spec_table):
    out = io.StringIO(newline="")
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(spec_table["columns"])
    for row in spec_table["rows"]:
        writer.writerow([cell(row[c]) for c in spec_table["columns"]])
    return out.getvalue()


def run_numeric_check(check, csv_text):
    """Run a frozen `NumericCheck` over a case CSV using the host's own check kernel."""
    parsed = NumericCheck.model_validate(
        check.model_dump() if isinstance(check, NumericCheck) else check
    )
    reader = csv.reader(io.StringIO(csv_text, newline=""), strict=True)
    header = next(reader, [])
    if not header:
        raise Invalid("Case CSV is missing its header row")
    rows = [(line, dict(zip(header, values))) for line, values in enumerate(reader, start=2)]
    if not rows:
        raise Invalid("Case CSV carried no data rows")
    return _numeric_kernel(
        parsed,
        header,
        rows,
        {
            "filename": parsed.csv_filename,
            "basis": "synthetic case-pack table; observation origin is not established",
        },
    )


def envelope(spec, seed, knobs, generator, tables, question, submission, metadata, limitations):
    """The agent-visible payload. `spec.agent_view()` has already dropped answer tokens."""
    if not isinstance(tables, dict) or not 1 <= len(tables) <= MAX_TABLES:
        raise Invalid(f"A case pack publishes 1 to {MAX_TABLES} tables")
    if set(tables) != set(spec.tables):
        raise Invalid("Published tables must match the declared table names")
    payload = {
        "case_id": spec.id,
        "manifest_case": spec.manifest_case,
        "domain": spec.domain,
        "title": spec.title,
        "status": spec.status,
        "scope": SCOPE,
        "synthetic": True,
        "empirically_validated": False,
        "independently_evaluated": False,
        "specification": spec.agent_view(),
        "generator": {
            "name": generator,
            "seed": int(seed),
            "knobs": clean(knobs),
            "declared": "Mechanistic simulation written in this repository; see the module docstring.",
        },
        "decision": spec.decision,
        "question": question,
        "submission": submission,
        "metadata": clean(metadata),
        "tables": tables,
        "known_limitations": list(limitations),
    }
    payload["payload_digest"] = digest(payload)
    return payload


def sealed(spec, seed, knobs, generator, facts, assumptions, support):
    truth_record = {
        "case_id": spec.id,
        "manifest_case": spec.manifest_case,
        "scope": TRUTH_SCOPE,
        "sealed": True,
        "seed": int(seed),
        "generator": {"name": generator, "knobs": clean(knobs)},
        "facts": clean(facts),
        "assumptions": list(assumptions),
        "sampling_support": support,
        "empirically_validated": False,
    }
    truth_record["truth_digest"] = digest(truth_record)
    return truth_record


def predictions_of(kind, spec, values, method, rationale):
    if kind not in ("baseline", "oracle", "trivial"):
        raise Invalid("Reference methods are baseline, oracle or trivial")
    return {
        "case_id": spec.id,
        "kind": kind,
        "method": method,
        "rationale": rationale,
        "scope": SCOPE,
        "predictions": values,
    }


def unwrap(predictions):
    """Accept either a raw prediction mapping or a wrapped reference-method record."""
    if isinstance(predictions, dict) and "predictions" in predictions and "kind" in predictions:
        predictions = predictions["predictions"]
    if not isinstance(predictions, dict) or not predictions:
        raise Invalid("Predictions must be a nonempty mapping")
    return predictions


def require_keys(values, expected, what="prediction"):
    missing = set(expected) - set(values)
    extra = set(values) - set(expected)
    if missing or extra:
        raise Invalid(
            f"{what.capitalize()} coverage mismatch: "
            f"{len(missing)} missing, {len(extra)} unexpected"
        )
    return values


def probability(value, name="probability"):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Invalid(f"Expected a finite {name} in [0, 1]")
    number = float(value)
    if not math.isfinite(number) or not -EPS <= number <= 1 + EPS:
        raise Invalid(f"Expected a finite {name} in [0, 1]")
    return min(1.0, max(0.0, number))


def real(value, name="value"):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Invalid(f"Expected a finite {name}")
    number = float(value)
    if not math.isfinite(number):
        raise Invalid(f"Expected a finite {name}")
    return number


def brier_losses(values, actual):
    return [(probability(values[k]) - float(actual[k])) ** 2 for k in sorted(actual)]


def multiclass_brier_losses(values, actual, classes):
    losses = []
    for key in sorted(actual):
        row = values[key]
        if not isinstance(row, dict) or set(row) != set(classes):
            raise Invalid("Mechanism predictions need one probability per declared class")
        total = sum(probability(row[c]) for c in classes)
        if abs(total - 1.0) > 1e-3:
            raise Invalid("Mechanism probabilities must sum to one")
        losses.append(sum((probability(row[c]) - (1.0 if actual[key] == c else 0.0)) ** 2 for c in classes))
    return losses


def log_losses(values, actual):
    out = []
    for key in sorted(actual):
        p = min(1 - 1e-9, max(1e-9, probability(values[key])))
        y = float(actual[key])
        out.append(-(y * math.log(p) + (1 - y) * math.log(1 - p)))
    return out


def reliability(values, actual, bins=5):
    """Equal-width reliability bins plus expected calibration error over those bins."""
    buckets = [[] for _ in range(bins)]
    for key in sorted(actual):
        p = probability(values[key])
        buckets[min(bins - 1, int(p * bins))].append((p, float(actual[key])))
    total = sum(len(b) for b in buckets)
    rows = [
        {
            "bin": i,
            "count": len(b),
            "mean_prediction": round(sum(p for p, _ in b) / len(b), ROUND) if b else None,
            "observed_rate": round(sum(y for _, y in b) / len(b), ROUND) if b else None,
        }
        for i, b in enumerate(buckets)
    ]
    error = sum(len(b) / total * abs(sum(p - y for p, y in b) / len(b)) for b in buckets if b)
    return rows, round(error, ROUND)


def absolute_errors(values, actual):
    return [abs(real(values[k]) - float(actual[k])) for k in sorted(actual)]


def squared_errors(values, actual):
    return [(real(values[k]) - float(actual[k])) ** 2 for k in sorted(actual)]


def spearman(predicted, actual):
    keys = sorted(actual)
    if len(keys) < 3:
        return None
    a = [real(predicted[k]) for k in keys]
    b = [float(actual[k]) for k in keys]
    if len(set(a)) < 2 or len(set(b)) < 2:
        return None
    return round(float(stats.spearmanr(a, b).statistic), ROUND)


def rank_auc(scores, labels):
    """Mann-Whitney rank AUC; None when a class is absent from the scored units."""
    keys = sorted(labels)
    positives = [real(scores[k]) for k in keys if labels[k]]
    negatives = [real(scores[k]) for k in keys if not labels[k]]
    if not positives or not negatives:
        return None
    ranks = stats.rankdata(positives + negatives)
    n_pos, n_neg = len(positives), len(negatives)
    auc = (float(np.sum(ranks[:n_pos])) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return round(auc, ROUND)


def total_variation(predicted, actual):
    keys = sorted(actual)
    return 0.5 * sum(abs(real(predicted[k]) - float(actual[k])) for k in keys)


def bootstrap(losses, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """Unit-level resampling of the per-unit loss. Descriptive spread, not a test."""
    values = np.asarray([float(x) for x in losses], dtype=float)
    if values.size == 0:
        raise Invalid("Uncertainty needs at least one scored unit")
    generator = np.random.default_rng(seed)
    draws = values[generator.integers(0, values.size, size=(resamples, values.size))].mean(axis=1)
    low, high = np.quantile(draws, [0.1, 0.9])
    return {
        "method": "nonparametric bootstrap over independent scored units",
        "resamples": int(resamples),
        "independent_units": int(values.size),
        "point_estimate": round(float(values.mean()), ROUND),
        "interval_80": [round(float(low), ROUND), round(float(high), ROUND)],
        "caveat": (
            "Spread of the estimate under this one synthetic realisation. It does not "
            "cover generator misspecification, and it is not a significance claim."
        ),
    }


def scored(spec, seed, primary, secondary, losses, units, notes):
    metric = spec.primary_metric
    report = {
        "case_id": spec.id,
        "manifest_case": spec.manifest_case,
        "scope": EVAL_SCOPE,
        "seed": int(seed),
        "primary_metric": metric.id,
        "primary_metric_name": metric.name,
        "primary_score": round(float(primary), 6),
        "lower_is_better": metric.lower_is_better,
        "secondary": clean(secondary),
        "scored_units": int(units),
        "uncertainty": bootstrap(losses),
        "notes": list(notes),
        "empirically_validated": False,
        "independently_evaluated": False,
        "synthetic": True,
    }
    report["result_digest"] = digest(report)
    return report


def story(spec, decision, agent_task, good_answer, trap, two_minutes):
    return {
        "case_id": spec.id,
        "manifest_case": spec.manifest_case,
        "title": spec.title,
        "scope": NARRATIVE_SCOPE,
        "decision": decision,
        "what_the_agent_must_figure_out": agent_task,
        "what_a_good_answer_looks_like": good_answer,
        "where_naive_analysis_fails": trap,
        "two_minute_script": list(two_minutes),
        "must_say_out_loud": (
            "This data is synthetic, from a generator in this repository. A good score "
            "shows the engine recovered structure it was not told. It is not evidence "
            "about the real domain."
        ),
    }


def payload_digest(payload):
    return digest(payload)


def canonical_bytes(value):
    return canonical(clean(value)).encode()


__all__ = [
    "SCOPE",
    "TRUTH_SCOPE",
    "EVAL_SCOPE",
    "NARRATIVE_SCOPE",
    "STATUS",
    "ORACLE_CONFIDENCE",
    "CaseSpec",
    "CaseEndpoint",
    "CaseMetric",
    "NumericCheck",
    "absolute_errors",
    "bootstrap",
    "brier_losses",
    "canonical_bytes",
    "cell",
    "check_knobs",
    "check_seed",
    "clean",
    "envelope",
    "log_losses",
    "multiclass_brier_losses",
    "payload_digest",
    "predictions_of",
    "probability",
    "rank_auc",
    "real",
    "reliability",
    "require_keys",
    "run_numeric_check",
    "scored",
    "sealed",
    "spearman",
    "squared_errors",
    "story",
    "stream",
    "table",
    "to_csv_text",
    "total_variation",
    "unwrap",
]
