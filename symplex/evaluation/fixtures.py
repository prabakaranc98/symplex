"""Deterministic SYNTHETIC plumbing fixture, never an empirical market archive."""

from datetime import UTC, datetime, timedelta

from symplex.core.contracts import digest


def synthetic_pack():
    rows = []
    for group in range(12):
        start = datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=group * 3)
        actual = (82, 88, 94, 97)[group % 4]
        for i, threshold in enumerate((85, 90, 95)):
            text = (
                "SYNTHETIC: Daily maximum at TEST station greater than "
                + str(threshold)
                + " F."
            )
            rows.append(
                {
                    "id": "g%d-t%d" % (group, threshold),
                    "event_group": "g%d" % group,
                    "split": "train"
                    if group < 6
                    else "development"
                    if group < 9
                    else "confirmation",
                    "contract": text,
                    "predicate": "greater_than",
                    "threshold": threshold,
                    "unit": "F",
                    "entity": "TEST",
                    "endpoint": "daily_maximum",
                    "event_time": (start + timedelta(days=1)).isoformat(),
                    "timezone": "UTC",
                    "rules_available_at": start.isoformat(),
                    "source_span": text,
                    "source_id": "fixture",
                    "observed_at": start.isoformat(),
                    "available_at": start.isoformat(),
                    "cutoff": (start + timedelta(hours=1)).isoformat(),
                    "resolved_at": (start + timedelta(days=2)).isoformat(),
                    "price": (0.45, 0.7, 0.3)[i],
                    "label": int(actual > threshold),
                }
            )
    return {
        "domain": "P1",
        "synthetic": True,
        "provenance": {
            "source": "Built-in synthetic fixture; not market data",
            "license": "Project license",
            "retrieved_at": "2026-09-10T00:00:00Z",
            "raw_sha256": digest(rows),
            "availability_basis": "Constructed fixture timestamps",
        },
        "rows": rows,
    }
