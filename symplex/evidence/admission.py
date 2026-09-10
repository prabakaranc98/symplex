"""P1 prepared-pack admission and point-in-time semantic reconciliation."""

from datetime import datetime

from symplex.core.contracts import Invalid, closed, digest, nonempty, number


def timestamp(value):
    nonempty(value)
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise Invalid("Invalid ISO timestamp") from exc
    if result.tzinfo is None:
        raise Invalid("Timestamps require an explicit UTC offset")
    return result


ROW_KEYS = (
    "id",
    "event_group",
    "split",
    "contract",
    "predicate",
    "threshold",
    "unit",
    "entity",
    "endpoint",
    "event_time",
    "timezone",
    "rules_available_at",
    "source_span",
    "source_id",
    "observed_at",
    "available_at",
    "cutoff",
    "resolved_at",
    "price",
    "label",
)
PREDICTION_KEYS = tuple(
    k for k in ROW_KEYS if k not in ("label", "resolved_at", "split")
)


def admit(data):
    closed(data, ("domain", "synthetic", "provenance", "rows"))
    if data["domain"] != "P1":
        raise Invalid(
            "unsupported_operation: only the prepared P1 adapter is executable"
        )
    if type(data["synthetic"]) is not bool:
        raise Invalid("synthetic must be explicitly true or false")
    provenance = data["provenance"]
    closed(
        provenance,
        ("source", "license", "retrieved_at", "raw_sha256", "availability_basis"),
    )
    for value in provenance.values():
        nonempty(value)
    if len(provenance["raw_sha256"]) != 64 or any(
        x not in "0123456789abcdef" for x in provenance["raw_sha256"]
    ):
        raise Invalid("raw_sha256 must identify the original archive snapshot")
    timestamp(provenance["retrieved_at"])
    rows = data["rows"]
    if not isinstance(rows, list) or not 6 <= len(rows) <= 10000:
        raise Invalid("Prepared slices require 6–10000 rows")
    seen, groups = set(), {}
    for row in rows:
        closed(row, ROW_KEYS)
        for key in ROW_KEYS:
            if key not in ("price", "threshold", "label"):
                nonempty(row[key])
        if row["id"] in seen:
            raise Invalid("Duplicate observation id")
        seen.add(row["id"])
        if row["split"] not in ("train", "development", "confirmation"):
            raise Invalid("Unknown protected split")
        if row["predicate"] != "greater_than":
            raise Invalid(
                "Unsupported settlement predicate; this adapter supports strict thresholds"
            )
        if row["unit"] not in ("F", "C", "count", "USD", "percent"):
            raise Invalid("Unknown or incompatible units")
        number(row["threshold"])
        number(row["price"], 0, 1)
        if type(row["label"]) is not int or row["label"] not in (0, 1):
            raise Invalid("Only actual binary resolutions are admitted")
        if row["source_span"] not in row["contract"]:
            raise Invalid(
                "Settlement source span is not present in original definition"
            )
        times = {
            k: timestamp(row[k])
            for k in (
                "observed_at",
                "available_at",
                "cutoff",
                "resolved_at",
                "rules_available_at",
                "event_time",
            )
        }
        if not (
            times["observed_at"]
            <= times["available_at"]
            <= times["cutoff"]
            < times["resolved_at"]
        ):
            raise Invalid(
                "Post-cutoff or unresolved-availability evidence is ineligible"
            )
        if (
            times["rules_available_at"] > times["cutoff"]
            or times["cutoff"] >= times["event_time"]
        ):
            raise Invalid("Rules unavailable at cutoff or event already observed")
        group = groups.setdefault(row["event_group"], [])
        if group and (
            group[0]["split"] != row["split"]
            or timestamp(group[0]["cutoff"]) != times["cutoff"]
        ):
            raise Invalid("Whole event groups must share a split and snapshot cutoff")
        group.append(row)
    subsets = {
        split: [r for r in rows if r["split"] == split]
        for split in ("train", "development", "confirmation")
    }
    if any(len({r["event_group"] for r in subset}) < 2 for subset in subsets.values()):
        raise Invalid("Each split requires at least two complete event groups")
    for earlier, later in (("train", "development"), ("development", "confirmation")):
        if max(timestamp(r["resolved_at"]) for r in subsets[earlier]) >= min(
            timestamp(r["cutoff"]) for r in subsets[later]
        ):
            raise Invalid(
                "Labels used for fitting/selection were unavailable before the next split"
            )
    return {
        "digest": digest(data),
        "source_digest": provenance["raw_sha256"],
        "rows": len(rows),
        "groups": len(groups),
        "split_groups": {
            k: len({r["event_group"] for r in v}) for k, v in subsets.items()
        },
        "synthetic": data["synthetic"],
        "provenance": provenance,
        "availability": "Declared by prepared-pack curator; original archive must be independently reviewed",
    }


def prediction_rows(rows):
    return [{k: r[k] for k in PREDICTION_KEYS} for r in rows]


def relation_key(row):
    return (
        row["event_group"],
        row["entity"],
        row["endpoint"],
        timestamp(row["event_time"]).isoformat(),
        row["timezone"],
        row["unit"],
        row["predicate"],
        timestamp(row["cutoff"]).isoformat(),
    )


def relations(rows):
    """Only curator-supplied exact semantics; no free-text guess becomes a fact."""
    groups = {}
    for r in rows:
        groups.setdefault(relation_key(r), []).append(r)
    accepted, rejected = [], []
    for group in groups.values():
        ordered = sorted(group, key=lambda r: r["threshold"])
        for a, b in zip(ordered, ordered[1:]):
            if a["threshold"] < b["threshold"]:
                accepted.append(
                    {
                        "lower": a["id"],
                        "higher": b["id"],
                        "relation": "p_higher <= p_lower",
                        "source_ids": [a["source_id"], b["source_id"]],
                        "source_spans": [a["source_span"], b["source_span"]],
                    }
                )
    by_event = {}
    for row in rows:
        by_event.setdefault(row["event_group"], []).append(row)
    # A linear-size diagnostic of exact-semantic near-matches, not a false-assertion estimate.
    for event_rows in by_event.values():
        reference = event_rows[0]
        for row in event_rows[1:]:
            if relation_key(reference) != relation_key(row):
                rejected.append(
                    {
                        "ids": [reference["id"], row["id"]],
                        "reason": "Entity, endpoint, unit, event time, timezone or cutoff differs",
                    }
                )
    return {
        "accepted": accepted,
        "rejected_near_matches": rejected,
        "false_relation_rate": None,
        "limitation": "Relations rely on curator-verified predicate metadata; no independent relation gold set supplied",
    }
