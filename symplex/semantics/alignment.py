"""Semantic alignment between evidence and model, before compute is spent on a bad join.

`docs/complex-system-audit.md` names the same failure three times: two markets referring
to different weather stations, an entity join that silently matched the wrong records, a
timestamp join across two clocks. None of those are detected by a schema check and none
of them announce themselves in the output; they produce a plausible number. The three
functions here look for them.

Everything is deterministic: token overlap and edit distance, calendar arithmetic on
declared timestamps, and counting. No embeddings, no model calls, no network. That is a
deliberate limitation, not an oversight: a scoring function you can read is one a reviewer
can argue with, and an ambiguous match is escalated to a person rather than resolved by a
confident-sounding model.

Convention: every public function returns a dict carrying a `scope` string. No function
here validates a join. `join_risk_report` reports risk; the word "validated" appears in
its output only as `False`.
"""

import math
import re
from collections import Counter
from datetime import datetime, timezone

from symplex.core.contracts import Invalid

MAX_ENTITIES = 500
MAX_PAIRS = 250_000
MAX_STRING = 200
MAX_TIMESTAMPS = 100_000
MAX_ROWS = 200_000

MATCH_THRESHOLD = 0.90
AMBIGUOUS_THRESHOLD = 0.60
TIE_MARGIN = 0.02

SCOPE_ENTITIES = (
    "Deterministic surface-string similarity between two name lists. A high score means "
    "the strings look alike, never that the referents are the same thing; two distinct "
    "weather stations can share a name and one station can be spelled six ways. Scores "
    "in the ambiguous band are returned for human resolution and are NOT merged. No "
    "match here is evidence of identity."
)
SCOPE_TIME = (
    "Structural comparison of two declared time supports. It detects clock, resolution "
    "and support mismatch from the declarations and the timestamps given; it cannot "
    "detect a timestamp that is wrong in a way the series is internally consistent "
    "about, and it does not resample anything. The proposed contract is a proposal."
)
SCOPE_JOIN = (
    "Risk assessment of a proposed join from the rows supplied. It counts cardinality, "
    "unmatched fraction and row blowup; it does not establish that matched keys denote "
    "the same entity, and a low-risk verdict is not a validated join. Counts describe "
    "the rows given, which may not be the rows the join will run on."
)

_LEGAL_SUFFIXES = frozenset(
    {
        "inc",
        "incorporated",
        "llc",
        "ltd",
        "limited",
        "corp",
        "corporation",
        "co",
        "company",
        "plc",
        "gmbh",
        "ag",
        "sa",
        "bv",
        "nv",
        "sas",
        "srl",
        "pty",
    }
)
_ARTICLES = frozenset({"the", "of", "and", "a", "an"})
_NON_WORD = re.compile(r"[^0-9a-z]+")


def normalize_name(text) -> str:
    """Lowercase, strip punctuation, collapse whitespace. Value helper, not a report."""
    if not isinstance(text, str):
        raise Invalid("Expected a string entity name")
    lowered = _NON_WORD.sub(" ", text.strip().lower())
    return " ".join(lowered.split())[:MAX_STRING]


def _tokens(text):
    tokens = [
        token
        for token in normalize_name(text).split()
        if token not in _ARTICLES and token not in _LEGAL_SUFFIXES
    ]
    return tokens or normalize_name(text).split()


def _levenshtein(a, b):
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def _similarity(a, b):
    tokens_a, tokens_b = set(_tokens(a)), set(_tokens(b))
    union = tokens_a | tokens_b
    jaccard = len(tokens_a & tokens_b) / len(union) if union else 0.0
    normal_a, normal_b = normalize_name(a), normalize_name(b)
    longest = max(len(normal_a), len(normal_b))
    edit_ratio = 1.0 - _levenshtein(normal_a, normal_b) / longest if longest else 0.0
    return {
        "jaccard": round(jaccard, 6),
        "edit_ratio": round(max(0.0, edit_ratio), 6),
        "score": round(0.5 * jaccard + 0.5 * max(0.0, edit_ratio), 6),
    }


def _names(items, label):
    if not isinstance(items, (list, tuple)):
        raise Invalid(f"Expected a list of {label} names or mappings")
    if len(items) > MAX_ENTITIES:
        raise Invalid(
            f"Entity alignment accepts at most {MAX_ENTITIES} {label} records"
        )
    records = []
    for item in items:
        if isinstance(item, str):
            record = {"id": item, "name": item}
        elif isinstance(item, dict):
            name = item.get("name") or item.get("label") or item.get("id")
            if not isinstance(name, str) or not name.strip():
                raise Invalid(f"Each {label} record needs a nonempty name")
            record = {"id": str(item.get("id", name)), "name": name}
        else:
            raise Invalid(f"Each {label} record is a string or a mapping")
        if len(record["name"]) > MAX_STRING * 4:
            raise Invalid(f"A {label} name exceeds the comparison envelope")
        records.append(record)
    if len({r["id"] for r in records}) != len(records):
        raise Invalid(f"Duplicate {label} record ID")
    return records


def align_entities(a, b) -> dict:
    """Score candidate entity matches between two name lists and refuse to merge the unclear.

    Score is the mean of token Jaccard on normalized tokens and a normalized
    Levenshtein ratio. A pair is a `match` only when it clears MATCH_THRESHOLD, is the
    unique best candidate on both sides, and beats its runner-up by more than TIE_MARGIN.
    Anything scoring above AMBIGUOUS_THRESHOLD that fails one of those tests is returned
    as `ambiguous` for a person to settle. Ambiguous pairs are never merged here, and
    `auto_merged` is always False.
    """
    left, right = _names(a, "left"), _names(b, "right")
    if len(left) * len(right) > MAX_PAIRS:
        raise Invalid(f"Entity alignment exceeds the {MAX_PAIRS}-pair envelope")
    scored = {}
    for record in left:
        ranked = sorted(
            (
                {
                    "b": other["id"],
                    "b_name": other["name"],
                    **_similarity(record["name"], other["name"]),
                }
                for other in right
            ),
            key=lambda entry: (-entry["score"], entry["b"]),
        )
        scored[record["id"]] = ranked

    proposals, ambiguous = {}, []
    for record in left:
        ranked = scored[record["id"]]
        best = ranked[0] if ranked else None
        runner_up = ranked[1] if len(ranked) > 1 else None
        if best is None or best["score"] < AMBIGUOUS_THRESHOLD:
            continue
        tied = (
            runner_up is not None and best["score"] - runner_up["score"] <= TIE_MARGIN
        )
        if best["score"] >= MATCH_THRESHOLD and not tied:
            proposals[record["id"]] = best
            continue
        ambiguous.append(
            {
                "a": record["id"],
                "a_name": record["name"],
                "candidates": [
                    {k: v for k, v in entry.items()}
                    for entry in ranked
                    if entry["score"] >= AMBIGUOUS_THRESHOLD
                ][:5],
                "band": "ambiguous",
                "reason": "tied_candidates" if tied else "below_match_threshold",
                "resolution": "human_review_required",
            }
        )

    claimed = Counter(entry["b"] for entry in proposals.values())
    matches = []
    for left_id, entry in sorted(proposals.items()):
        if claimed[entry["b"]] > 1:
            ambiguous.append(
                {
                    "a": left_id,
                    "a_name": next(r["name"] for r in left if r["id"] == left_id),
                    "candidates": [entry],
                    "band": "ambiguous",
                    "reason": "many_left_records_claim_one_right_record",
                    "resolution": "human_review_required",
                }
            )
            continue
        matches.append(
            {
                "a": left_id,
                "a_name": next(r["name"] for r in left if r["id"] == left_id),
                "b": entry["b"],
                "b_name": entry["b_name"],
                "score": entry["score"],
                "jaccard": entry["jaccard"],
                "edit_ratio": entry["edit_ratio"],
                "band": "match",
            }
        )
    ambiguous.sort(key=lambda entry: entry["a"])
    matched_left = {entry["a"] for entry in matches}
    matched_right = {entry["b"] for entry in matches}
    ambiguous_left = {entry["a"] for entry in ambiguous}
    return {
        "matches": matches,
        "match_count": len(matches),
        "ambiguous": ambiguous,
        "ambiguous_count": len(ambiguous),
        "unmatched_a": sorted(
            r["id"] for r in left if r["id"] not in matched_left | ambiguous_left
        ),
        "unmatched_b": sorted(r["id"] for r in right if r["id"] not in matched_right),
        "auto_merged": False,
        "requires_human_resolution": sorted(entry["a"] for entry in ambiguous),
        "thresholds": {
            "match": MATCH_THRESHOLD,
            "ambiguous": AMBIGUOUS_THRESHOLD,
            "tie_margin": TIE_MARGIN,
        },
        "method": (
            "0.5 * token Jaccard on normalized tokens + 0.5 * normalized Levenshtein "
            "ratio; deterministic, no embeddings and no model calls"
        ),
        "pairs_scored": len(left) * len(right),
        "scope": SCOPE_ENTITIES,
    }


def _timestamps(series, label):
    if not isinstance(series, dict):
        raise Invalid(f"Series {label} must be a mapping with a 'timestamps' list")
    values = series.get("timestamps")
    if not isinstance(values, (list, tuple)) or not values:
        raise Invalid(f"Series {label} needs a nonempty 'timestamps' list")
    if len(values) > MAX_TIMESTAMPS:
        raise Invalid(f"Series {label} exceeds the {MAX_TIMESTAMPS}-timestamp envelope")
    seconds, offsets, naive = [], set(), 0
    for value in values:
        if isinstance(value, bool):
            raise Invalid(f"Series {label} carries a non-timestamp value")
        if isinstance(value, (int, float)):
            if not math.isfinite(value):
                raise Invalid(f"Series {label} carries a nonfinite timestamp")
            seconds.append(float(value))
            offsets.add(0)
            continue
        if not isinstance(value, str):
            raise Invalid(f"Series {label} carries a non-timestamp value")
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            raise Invalid(
                f"Series {label} carries an unparseable ISO 8601 timestamp: {value!r}"
            ) from None
        if parsed.tzinfo is None:
            naive += 1
            parsed = parsed.replace(tzinfo=timezone.utc)
        else:
            offsets.add(int(parsed.utcoffset().total_seconds()))
        seconds.append(parsed.timestamp())
    return {
        "id": str(series.get("id", label)),
        "seconds": seconds,
        "naive_count": naive,
        "utc_offsets": sorted(offsets),
        "declared_timezone": series.get("timezone"),
        "declared_clock": series.get("clock"),
    }


def _profile(parsed):
    seconds = sorted(parsed["seconds"])
    steps = [round(b - a, 6) for a, b in zip(seconds, seconds[1:])]
    modal = Counter(steps).most_common(1)[0][0] if steps else None
    return {
        "id": parsed["id"],
        "count": len(seconds),
        "start": seconds[0],
        "end": seconds[-1],
        "span_seconds": seconds[-1] - seconds[0],
        "modal_step_seconds": modal,
        "regular": bool(steps) and len(set(steps)) == 1,
        "distinct_steps": len(set(steps)),
        "duplicate_timestamps": len(seconds) - len(set(seconds)),
        "naive_count": parsed["naive_count"],
        "utc_offsets": parsed["utc_offsets"],
        "declared_timezone": parsed["declared_timezone"],
        "declared_clock": parsed["declared_clock"],
    }


def align_time(series_a, series_b) -> dict:
    """Compare two declared time supports and propose a resampling contract.

    Detects: naive (offset-free) timestamps, disagreeing declared timezones or clocks,
    mixed UTC offsets inside one series, sampling-rate mismatch, irregular sampling,
    duplicate timestamps, absent or partial support overlap, a constant whole-quarter-hour
    shift between two otherwise identical grids (the timezone bug), and grid phase
    mismatch. The proposed contract uses the alignment vocabulary of `ModelCoupling` in
    `symplex/modeling/complex_system.py` so it can be pasted into a spec.
    """
    parsed_a, parsed_b = _timestamps(series_a, "a"), _timestamps(series_b, "b")
    profile_a, profile_b = _profile(parsed_a), _profile(parsed_b)
    findings = []

    def add(code, severity, detail, **extra):
        findings.append({"code": code, "severity": severity, "detail": detail, **extra})

    for profile in (profile_a, profile_b):
        if profile["naive_count"]:
            add(
                "naive_timestamps",
                "error",
                f"Series {profile['id']} carries {profile['naive_count']} timestamps "
                "with no UTC offset; they were read as UTC for this comparison, which is "
                "an assumption, not a fact.",
                series=profile["id"],
            )
        if len(profile["utc_offsets"]) > 1:
            add(
                "mixed_utc_offsets",
                "warning",
                f"Series {profile['id']} mixes UTC offsets "
                + str(profile["utc_offsets"])
                + "; a daylight-saving boundary inside a series breaks a naive join.",
                series=profile["id"],
            )
        if profile["duplicate_timestamps"]:
            add(
                "duplicate_timestamps",
                "warning",
                f"Series {profile['id']} repeats {profile['duplicate_timestamps']} timestamps.",
                series=profile["id"],
            )
        if not profile["regular"] and profile["count"] >= 2:
            add(
                "irregular_sampling",
                "info",
                f"Series {profile['id']} has {profile['distinct_steps']} distinct step "
                "sizes; an aggregation window cannot be assumed uniform.",
                series=profile["id"],
            )
    if (
        profile_a["declared_timezone"] is not None
        and profile_b["declared_timezone"] is not None
        and profile_a["declared_timezone"] != profile_b["declared_timezone"]
    ):
        add(
            "declared_timezone_mismatch",
            "error",
            "Declared timezones differ: "
            + f"{profile_a['declared_timezone']!r} and {profile_b['declared_timezone']!r}.",
        )
    if (
        profile_a["declared_clock"] is not None
        and profile_b["declared_clock"] is not None
        and profile_a["declared_clock"] != profile_b["declared_clock"]
    ):
        add(
            "declared_clock_mismatch",
            "error",
            "Declared reference clocks differ: "
            + f"{profile_a['declared_clock']!r} and {profile_b['declared_clock']!r}; "
            "an explicit clock-mapping component is required.",
        )
    step_a, step_b = profile_a["modal_step_seconds"], profile_b["modal_step_seconds"]
    if step_a is not None and step_b is not None and step_a != step_b:
        add(
            "sampling_rate_mismatch",
            "warning",
            f"Modal steps differ: {step_a}s and {step_b}s.",
        )

    constant_offset = None
    if profile_a["count"] == profile_b["count"] and step_a == step_b:
        differences = {
            round(b - a, 6)
            for a, b in zip(sorted(parsed_a["seconds"]), sorted(parsed_b["seconds"]))
        }
        if len(differences) == 1:
            constant_offset = differences.pop()
    if constant_offset:
        add(
            "suspected_clock_shift",
            "error" if abs(constant_offset) % 900 == 0 else "warning",
            f"Series {profile_b['id']} sits a constant {constant_offset:.0f}s "
            f"({constant_offset / 3600:.2f}h) from series {profile_a['id']} on an "
            "otherwise identical grid. A whole-quarter-hour constant offset between two "
            "series of the same length is a timezone or clock error far more often than "
            "it is a real lag.",
            constant_offset_seconds=constant_offset,
            whole_quarter_hour=abs(constant_offset) % 900 == 0,
        )

    overlap_start = max(profile_a["start"], profile_b["start"])
    overlap_end = min(profile_a["end"], profile_b["end"])
    overlap = max(0.0, overlap_end - overlap_start)
    if overlap <= 0:
        add(
            "no_temporal_overlap",
            "error",
            "The two declared supports do not overlap; no join over time is possible.",
        )
    else:
        for profile in (profile_a, profile_b):
            covered = (
                overlap / profile["span_seconds"] if profile["span_seconds"] else 0.0
            )
            if covered < 0.999:
                add(
                    "partial_support_overlap",
                    "warning",
                    f"Only {covered:.1%} of series {profile['id']}'s span lies in the "
                    "common window; the remainder must be dropped or declared missing.",
                    series=profile["id"],
                    covered_fraction=round(covered, 6),
                )
    coarse = max(step_a or 0, step_b or 0) or None
    if coarse and not constant_offset:
        phase = round((profile_b["start"] - profile_a["start"]) % coarse, 6)
        if phase not in (0.0, coarse):
            add(
                "grid_phase_mismatch",
                "warning",
                f"The two grids are offset by {phase:.0f}s, which is not a multiple of "
                f"the coarser {coarse:.0f}s step; every bucket boundary falls inside a "
                "sample of the other series.",
                phase_seconds=phase,
            )

    def method(step):
        if coarse is None or step is None or step == coarse:
            return "synchronous"
        return "aggregate"

    errors = sum(1 for f in findings if f["severity"] == "error")
    warnings = sum(1 for f in findings if f["severity"] == "warning")
    verdict = (
        "blocked"
        if errors
        else "resample_required"
        if warnings or (step_a != step_b)
        else "aligned_as_declared"
    )
    return {
        "series_a": profile_a,
        "series_b": profile_b,
        "findings": findings,
        "finding_count": len(findings),
        "severity_counts": {
            "error": errors,
            "warning": warnings,
            "info": len(findings) - errors - warnings,
        },
        "verdict": verdict,
        "constant_offset_seconds": constant_offset,
        "overlap_window": [overlap_start, overlap_end] if overlap > 0 else None,
        "overlap_seconds": overlap,
        "resampling_contract": {
            "target_clock": "UTC",
            "target_step_seconds": coarse,
            "window": [overlap_start, overlap_end] if overlap > 0 else None,
            "boundary": "left_closed_right_open",
            "method_a": method(step_a),
            "method_b": method(step_b),
            "time_alignment": "synchronous"
            if method(step_a) == method(step_b) == "synchronous"
            else "aggregate",
            "unresolved": [f["code"] for f in findings if f["severity"] == "error"],
            "applicable": errors == 0,
        },
        "resampled": False,
        "validated": False,
        "scope": SCOPE_TIME,
    }


def _key_values(rows, key, label):
    if not isinstance(rows, (list, tuple)):
        raise Invalid(f"Expected a list of {label} rows")
    if len(rows) > MAX_ROWS:
        raise Invalid(f"Join risk accepts at most {MAX_ROWS} {label} rows")
    if not isinstance(key, str) or not key:
        raise Invalid(f"Expected a {label} key column name")
    values, blank, types = [], 0, Counter()
    for row in rows:
        if not isinstance(row, dict):
            raise Invalid(f"Each {label} row must be a mapping")
        if key not in row:
            blank += 1
            continue
        value = row[key]
        if value is None or (isinstance(value, str) and not value.strip()):
            blank += 1
            continue
        types[type(value).__name__] += 1
        values.append(value)
    return values, blank, types


def join_risk_report(
    left, right, left_key, right_key, left_name="left", right_name="right"
) -> dict:
    """Report the cardinality, unmatched fraction and row blowup of a proposed join.

    A `risk` verdict is a verdict about the rows supplied, not about the join. Nothing
    here validates a join and `validated_join` is always False: matching keys means the
    key strings agree, which is the assumption the audits repeatedly found to be false.
    """
    left_values, left_blank, left_types = _key_values(left, left_key, left_name)
    right_values, right_blank, right_types = _key_values(right, right_key, right_name)
    left_counts, right_counts = Counter(left_values), Counter(right_values)
    left_rows, right_rows = len(left), len(right)
    matched = set(left_counts) & set(right_counts)
    output_rows = sum(left_counts[key] * right_counts[key] for key in matched)
    left_fanout = max(left_counts.values(), default=0)
    right_fanout = max(right_counts.values(), default=0)
    cardinality = (
        "empty"
        if not left_counts or not right_counts
        else "one_to_one"
        if left_fanout == 1 and right_fanout == 1
        else "one_to_many"
        if left_fanout == 1
        else "many_to_one"
        if right_fanout == 1
        else "many_to_many"
    )
    unmatched_left_rows = sum(
        count for key, count in left_counts.items() if key not in matched
    )
    unmatched_right_rows = sum(
        count for key, count in right_counts.items() if key not in matched
    )

    def normalized(value):
        return value.strip().casefold() if isinstance(value, str) else value

    left_normal = {normalized(key): key for key in left_counts}
    near_miss = sorted(
        str(key)
        for key in right_counts
        if key not in matched and normalized(key) in left_normal
    )[:32]
    type_mismatch = (
        bool(left_types)
        and bool(right_types)
        and not (set(left_types) & set(right_types))
    )
    unmatched_fraction_left = unmatched_left_rows / left_rows if left_rows else 1.0
    unmatched_fraction_right = unmatched_right_rows / right_rows if right_rows else 1.0
    blowup = output_rows / left_rows if left_rows else 0.0
    reasons = []
    if type_mismatch:
        reasons.append("key_type_mismatch")
    if not matched:
        reasons.append("no_matching_keys")
    if cardinality == "many_to_many":
        reasons.append("many_to_many_cardinality")
    if blowup > 2:
        reasons.append("row_blowup_above_2x")
    if max(unmatched_fraction_left, unmatched_fraction_right) > 0.5:
        reasons.append("majority_unmatched")
    if left_blank or right_blank:
        reasons.append("null_or_missing_keys")
    if near_miss:
        reasons.append("case_or_whitespace_near_misses")
    if 0.1 < max(unmatched_fraction_left, unmatched_fraction_right) <= 0.5:
        reasons.append("unmatched_above_10_percent")
    if type_mismatch or not matched:
        risk = "blocked"
    elif {
        "many_to_many_cardinality",
        "row_blowup_above_2x",
        "majority_unmatched",
    } & set(reasons):
        risk = "high"
    elif reasons:
        risk = "elevated"
    else:
        risk = "low"
    return {
        "status": "risk_assessment",
        "risk": risk,
        "risk_reasons": reasons,
        "cardinality": cardinality,
        "left": {
            "name": left_name,
            "key": left_key,
            "rows": left_rows,
            "distinct_keys": len(left_counts),
            "max_rows_per_key": left_fanout,
            "null_or_missing_keys": left_blank,
            "key_types": dict(sorted(left_types.items())),
            "unmatched_rows": unmatched_left_rows,
            "unmatched_fraction": round(unmatched_fraction_left, 6),
        },
        "right": {
            "name": right_name,
            "key": right_key,
            "rows": right_rows,
            "distinct_keys": len(right_counts),
            "max_rows_per_key": right_fanout,
            "null_or_missing_keys": right_blank,
            "key_types": dict(sorted(right_types.items())),
            "unmatched_rows": unmatched_right_rows,
            "unmatched_fraction": round(unmatched_fraction_right, 6),
        },
        "matched_keys": len(matched),
        "projected_output_rows": output_rows,
        "duplicate_blowup_factor": round(blowup, 6),
        "max_fanout_product": left_fanout * right_fanout,
        "case_or_whitespace_near_misses": near_miss,
        "validated_join": False,
        "verdict_meaning": (
            "Risk of a silently wrong join given these rows. A 'low' verdict means no "
            "counted hazard was found, not that the join is correct."
        ),
        "scope": SCOPE_JOIN,
    }
