"""Host-owned fixed evaluator; candidates cannot supply scores or split allocation."""

import math

from symplex.core.contracts import Invalid, digest, number
from symplex.evidence.admission import relations

PROTOCOL = {
    "version": "p1-brier-v1",
    "primary": "event_group_mean_brier",
    "selection": "minimum development Brier; retain earlier candidate on ties",
    "secondary": ["log_loss", "ece_5_bins", "coherence_violations", "coverage"],
    "uncertainty": "per-event paired effects; no small-sample significance claim",
}
PROTOCOL_DIGEST = digest(PROTOCOL)


def score(rows, predictions):
    if set(predictions) != {r["id"] for r in rows} or not rows:
        raise Invalid("Prediction coverage or identity mismatch")
    groups, bins = {}, [[] for _ in range(5)]
    log_losses = []
    for row in rows:
        p = number(predictions[row["id"]], 0, 1)
        loss = (p - row["label"]) ** 2
        groups.setdefault(row["event_group"], []).append(loss)
        bins[min(4, int(p * 5))].append((p, row["label"]))
        clipped = min(1 - 1e-12, max(1e-12, p))
        log_losses.append(
            -(
                row["label"] * math.log(clipped)
                + (1 - row["label"]) * math.log(1 - clipped)
            )
        )
    event_scores = {key: sum(values) / len(values) for key, values in groups.items()}
    reliability = [
        {
            "bin": i,
            "count": len(b),
            "mean_prediction": sum(p for p, y in b) / len(b) if b else None,
            "observed_rate": sum(y for p, y in b) / len(b) if b else None,
        }
        for i, b in enumerate(bins)
    ]
    links = relations(rows)["accepted"]
    return {
        "protocol_digest": PROTOCOL_DIGEST,
        "brier": sum(event_scores.values()) / len(event_scores),
        "event_scores": event_scores,
        "log_loss": sum(log_losses) / len(log_losses),
        "ece": sum(
            len(b) / len(rows) * abs(sum(p - y for p, y in b) / len(b))
            for b in bins
            if b
        ),
        "reliability": reliability,
        "coherence_violations": sum(
            predictions[x["higher"]] > predictions[x["lower"]] + 1e-12 for x in links
        ),
        "relation_count": len(links),
        "coverage": 1.0,
        "observations": len(rows),
        "independent_units": len(groups),
        "false_relation_rate": None,
    }


def paired(parent, child):
    if set(parent["event_scores"]) != set(child["event_scores"]):
        raise Invalid("Cannot compare different confirmation event groups")
    effects = {
        k: parent["event_scores"][k] - child["event_scores"][k]
        for k in parent["event_scores"]
    }
    return {
        "parent_minus_child_brier": sum(effects.values()) / len(effects),
        "per_event_effects": effects,
        "interval": None,
        "scope": "Descriptive paired effects on this slice; no significance or causal claim",
    }
