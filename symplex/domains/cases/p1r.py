"""P1r -- logically linked event contracts with a coherence structure and stale quotes.

Generator: threshold contracts on a daily maximum temperature. Each event group carries a
nested ladder -- "exceeds 80", "exceeds 85", "exceeds 90" on the same station, unit,
clock and settlement window -- so the true probabilities must satisfy
P(>90) <= P(>85) <= P(>80). Some groups also carry a deliberate near-match: the same
number on a *different* station or a different clock, which looks joinable and is not.

Quotes are the true probability shrunk toward one half, by a little in liquid groups and
by a lot in stale, low-activity groups whose last quote is hours older than the cutoff.
Quote age is visible in the pack; the shrinkage coefficient is not.

This pack is built to pass `symplex.evidence.admission.admit` and is scored by
`symplex.evaluation.metrics.score`, the existing host-owned P1 evaluator. Nothing here is
a market observation: the dates are in the future and the prices were written by this
module. Recovering the planted probabilities demonstrates inference mechanics only.
"""

import hashlib
from datetime import UTC, datetime, timedelta

import numpy as np
from scipy import stats

from symplex.core.contracts import Invalid, canonical
from symplex.domains.cases import (
    CaseEndpoint,
    CaseMetric,
    CaseSpec,
    NumericCheck,
    check_knobs,
    check_seed,
    envelope,
    predictions_of,
    probability,
    require_keys,
    scored,
    sealed,
    story,
    stream,
    table,
    unwrap,
)
from symplex.evaluation.metrics import PROTOCOL, score
from symplex.evidence.admission import ROW_KEYS, admit, prediction_rows, relations

CASE_ID = "p1r"
GENERATOR = "p1r_linked_threshold_contracts_v1"
DEFAULT_SEED = 7
KNOBS = {"groups": (18, 48, 32), "stale_fraction": (0.0, 0.8, 0.5), "near_match_fraction": (0.0, 0.6, 0.3)}
THRESHOLDS = (80.0, 85.0, 90.0)
STALE_SHRINK = 0.62
LIQUID_SHRINK = 0.08
STALE_QUOTE_AGE_HOURS = 9
LIQUID_QUOTE_AGE_MINUTES = 20
FORECAST_SD = 4.2
EPOCH = datetime(2031, 3, 1, tzinfo=UTC)

SPEC = CaseSpec(
    id=CASE_ID,
    manifest_case="P1",
    domain="prediction_markets",
    title="Do logically linked markets produce more reliable forecasts when modeled together?",
    decision=(
        "Which of a desk's posted probabilities to publish unchanged, which to restate, "
        "and which pairs of contracts may legitimately be compared at all."
    ),
    beneficiary="A forecast desk running a reliability and consistency audit before publication.",
    mechanism_question=(
        "Are apparent incoherences between linked contracts a real pricing error, or an "
        "artefact of quote age in low-activity groups and of near-matches that are not "
        "the same event?"
    ),
    rivals=[
        "Raw quotes are already the best available probability.",
        "Shared logical structure across a threshold ladder improves the estimate.",
        "Apparent violations are explained by stale quotes rather than mispricing.",
        "Apparent violations come from joining contracts whose settlement terms differ.",
    ],
    endpoints=[
        CaseEndpoint(
            id="resolution_probability",
            name="Probability the contract settles YES",
            unit="probability",
            meaning="The published number a desk stands behind.",
        ),
        CaseEndpoint(
            id="coherence",
            name="Coherence violations across a threshold ladder",
            unit="count",
            meaning="Published numbers that assert a higher threshold is more likely than a lower one.",
        ),
    ],
    evidence_channels=[
        "Fitting and development rows with exact settlement text, timestamps and resolutions.",
        "Confirmation rows stripped of resolution, split and settlement time.",
        "An admission receipt from the existing host P1 contract.",
    ],
    nuisance=[
        "Quote age varies by group and is the only visible trace of the liquidity regime.",
        "Deliberate near-matches share a group but differ in station or clock.",
        "Quotes carry independent noise per contract, so the ladder is not always monotone.",
        "Settlement is a strict threshold, so a quote at a boundary is genuinely ambiguous.",
    ],
    primary_metric=CaseMetric(
        id="event_group_mean_brier",
        name="Event-group mean Brier score",
        lower_is_better=True,
        definition="The existing P1 protocol primary: mean over event groups of the within-group mean Brier.",
    ),
    secondary_metrics=[
        CaseMetric(
            id="log_loss",
            name="Mean log loss",
            lower_is_better=True,
            definition="Host log loss over confirmation observations.",
        ),
        CaseMetric(
            id="ece_5_bins",
            name="Expected calibration error",
            lower_is_better=True,
            definition="Five-bin expected calibration error from the host scorer.",
        ),
        CaseMetric(
            id="coherence_violations",
            name="Coherence violations",
            lower_is_better=True,
            definition="Accepted ladder relations where the submitted numbers invert.",
        ),
    ],
    baseline_name="Raw quote proxy",
    baseline_rationale=(
        "Publish the market's own number. This is the honest reference: quotes are "
        "informative, they beat any constant, and a desk that cannot beat them has "
        "added nothing. It carries the stale-group shrinkage straight through."
    ),
    oracle_rationale=(
        "The planted probability that generated each resolution. It is the Bayes-optimal "
        "forecast for this generator and violates no ladder relation; it marks the headroom."
    ),
    trivial_name="Constant one-half",
    hidden_markers=[
        "resolution_by_observation",
        "latent_probability_by_observation",
        "coherence_pairs_true",
        "near_match_pairs_true",
        "liquidity_regime_by_group",
        "quote_shrinkage_by_group",
    ],
    tables=["p1r_fitting_rows.csv", "p1r_confirmation_rows.csv", "p1r_event_groups.csv"],
    knobs=["groups", "stale_fraction", "near_match_fraction"],
    limits=(
        "Coherence is not accuracy, and an inverted pair here is not demonstrated "
        "executable arbitrage. No trade, venue or real archive is involved."
    ),
)

NUMERIC_CHECKS = [
    NumericCheck(
        id="p1r_prices_are_probabilities",
        operation="bounds",
        csv_filename="p1r_confirmation_rows.csv",
        columns=["price"],
        group_columns=["event_group"],
        time_column=None,
        row_filters=[],
        lower=0.0,
        upper=1.0,
        reference_value=None,
        tolerance=0.0,
        direction=None,
        units="probability",
        rationale="A quote proxy outside the unit interval cannot be read as a probability.",
    ),
    NumericCheck(
        id="p1r_thresholds_finite",
        operation="finite",
        csv_filename="p1r_fitting_rows.csv",
        columns=["threshold", "price"],
        group_columns=["event_group"],
        time_column=None,
        row_filters=[],
        lower=None,
        upper=None,
        reference_value=None,
        tolerance=0.0,
        direction=None,
        units="degrees Fahrenheit and probability",
        rationale="The ladder relation is defined by the threshold, so it must always parse.",
    ),
]


def _iso(moment):
    return moment.isoformat().replace("+00:00", "Z")


def _split_for(index, counts):
    if index < counts[0]:
        return "train"
    if index < counts[0] + counts[1]:
        return "development"
    return "confirmation"


def _build(seed, knobs):
    groups = knobs["groups"]
    counts = (
        max(2, int(groups * 0.32)),
        max(2, int(groups * 0.20)),
        groups - max(2, int(groups * 0.32)) - max(2, int(groups * 0.20)),
    )
    if counts[2] < 2:
        raise Invalid("P1r needs at least two confirmation event groups")
    weather = stream(seed, "p1r/weather")
    market = stream(seed, "p1r/market")
    structure = stream(seed, "p1r/structure")
    rows, meta = [], {}
    for g in range(groups):
        group = f"grp{g:03d}"
        split = _split_for(g, counts)
        day = EPOCH + timedelta(days=g)
        event_time = day.replace(hour=23)
        cutoff = event_time - timedelta(hours=6)
        rules_available = day - timedelta(days=30)
        resolved_at = event_time + timedelta(hours=6)
        stale = bool(market.random() < knobs["stale_fraction"])
        shrink = float(
            np.clip(
                (STALE_SHRINK if stale else LIQUID_SHRINK) + market.normal(0, 0.04), 0.0, 0.8
            )
        )
        age = (
            timedelta(hours=STALE_QUOTE_AGE_HOURS)
            if stale
            else timedelta(minutes=LIQUID_QUOTE_AGE_MINUTES)
        )
        observed_at = cutoff - age
        available_at = cutoff - timedelta(minutes=5)
        centre = 84.0 + 9.0 * float(np.sin(g / 7.0)) + float(weather.normal(0, 3.4))
        realised = centre + float(weather.normal(0, FORECAST_SD))
        secondary = realised + float(weather.normal(0, 1.6))
        entries = [(t, "st_ord_primary", "UTC", realised) for t in THRESHOLDS]
        near_match = bool(structure.random() < knobs["near_match_fraction"])
        if near_match:
            if structure.random() < 0.5:
                entries.append((87.0, "st_ord_secondary", "UTC", secondary))
            else:
                entries.append((87.0, "st_ord_primary", "America/Chicago", secondary))
        meta[group] = {
            "split": split,
            "stale": stale,
            "shrink": shrink,
            "quote_age_seconds": int(age.total_seconds()),
            "near_match": near_match,
            "centre": centre,
        }
        for threshold, entity, zone, outcome in entries:
            latent = float(1.0 - stats.norm.cdf((threshold - centre) / FORECAST_SD))
            noise = float(market.normal(0, 0.085 if stale else 0.022))
            price = float(np.clip(0.5 + (latent - 0.5) * (1.0 - shrink) + noise, 0.02, 0.98))
            span = f"exceeds {threshold:g}"
            contract = (
                f"Settles YES if the daily maximum temperature at {entity}, measured in F, "
                f"{span} for the observation day ending {_iso(event_time)} ({zone} settlement clock)."
            )
            ident = f"obs_{g:03d}_{entity}_{zone}_{threshold:g}".replace("/", "_")
            rows.append(
                {
                    "id": ident,
                    "event_group": group,
                    "split": split,
                    "contract": contract,
                    "predicate": "greater_than",
                    "threshold": round(threshold, 4),
                    "unit": "F",
                    "entity": entity,
                    "endpoint": "daily_max_temperature",
                    "event_time": _iso(event_time),
                    "timezone": zone,
                    "rules_available_at": _iso(rules_available),
                    "source_span": span,
                    "source_id": f"synthetic_pack_{group}_{threshold:g}",
                    "observed_at": _iso(observed_at),
                    "available_at": _iso(available_at),
                    "cutoff": _iso(cutoff),
                    "resolved_at": _iso(resolved_at),
                    "price": round(price, 4),
                    "label": int(outcome > threshold),
                }
            )
            meta.setdefault("latent", {})[ident] = round(latent, 6)
    return rows, meta, counts


def _pack(rows):
    body = canonical([{k: r[k] for k in ROW_KEYS} for r in rows])
    return {
        "domain": "P1",
        "synthetic": True,
        "provenance": {
            "source": "symplex.domains.cases.p1r generator; no venue, archive or trade involved",
            "license": "Repository licence; generated content, not licensed market data",
            "retrieved_at": _iso(EPOCH),
            "raw_sha256": hashlib.sha256(body.encode()).hexdigest(),
            "availability_basis": "Point-in-time fields are generator constructions, not venue records",
        },
        "rows": rows,
    }


def generate(seed=DEFAULT_SEED, **knobs):
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    rows, meta, counts = _build(seed, resolved)
    receipt = admit(_pack(rows))
    fitting = [r for r in rows if r["split"] in ("train", "development")]
    confirmation = [r for r in rows if r["split"] == "confirmation"]
    visible = prediction_rows(confirmation)
    fitting_columns = list(ROW_KEYS)
    confirmation_columns = list(visible[0])
    group_rows = [
        {
            "event_group": group,
            "phase": "fitting" if info["split"] in ("train", "development") else "confirmation",
            "quote_age_seconds": info["quote_age_seconds"],
            "contract_count": sum(1 for r in rows if r["event_group"] == group),
        }
        for group, info in meta.items()
        if group != "latent"
    ]
    tables = {
        "p1r_fitting_rows.csv": table(fitting_columns, [{k: r[k] for k in fitting_columns} for r in fitting]),
        "p1r_confirmation_rows.csv": table(
            confirmation_columns, [{k: r[k] for k in confirmation_columns} for r in visible]
        ),
        "p1r_event_groups.csv": table(
            ["event_group", "phase", "quote_age_seconds", "contract_count"], group_rows
        ),
    }
    return envelope(
        SPEC,
        seed,
        resolved,
        GENERATOR,
        tables,
        question=(
            "Publish a probability for every confirmation observation. The fitting rows "
            "carry their resolutions; the confirmation rows do not. Respect any relation "
            "that the settlement text actually supports, and no relation that it does not."
        ),
        submission={
            "keys": "one entry per id in p1r_confirmation_rows.csv",
            "fields": {"probability": "probability in [0, 1]"},
            "example_key": visible[0]["id"],
        },
        metadata={
            "admission_receipt": receipt,
            "protocol": PROTOCOL,
            "split_counts": {"train": counts[0], "development": counts[1], "confirmation": counts[2]},
            "relation_rule": (
                "Two contracts are comparable only when entity, endpoint, unit, predicate, "
                "event time, settlement clock and snapshot cutoff all match exactly."
            ),
            "clock": "All timestamps carry an explicit UTC offset; the settlement clock is a separate field.",
            "not_observed": [
                "order book depth, trade counts or any activity measure",
                "the mapping from quote age to quote quality",
                "confirmation resolutions and their settlement times",
            ],
        },
        limitations=[
            "Quotes are price proxies, not calibrated probabilities.",
            "An inverted ladder pair is a consistency finding, not an executable trade.",
            "Future-dated synthetic contracts; no venue or archive is represented.",
        ],
    )


def truth(seed=DEFAULT_SEED, **knobs):
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    rows, meta, _ = _build(seed, resolved)
    confirmation = [r for r in rows if r["split"] == "confirmation"]
    links = relations(confirmation)
    return sealed(
        SPEC,
        seed,
        resolved,
        GENERATOR,
        facts={
            "resolution_by_observation": {r["id"]: r["label"] for r in confirmation},
            "latent_probability_by_observation": {
                r["id"]: meta["latent"][r["id"]] for r in confirmation
            },
            "coherence_pairs_true": [[x["lower"], x["higher"]] for x in links["accepted"]],
            "near_match_pairs_true": [x["ids"] for x in links["rejected_near_matches"]],
            "liquidity_regime_by_group": {
                g: ("stale_low_activity" if info["stale"] else "liquid")
                for g, info in meta.items()
                if g != "latent"
            },
            "quote_shrinkage_by_group": {
                g: round(info["shrink"], 6) for g, info in meta.items() if g != "latent"
            },
        },
        assumptions=[
            "One realised temperature per group settles every ladder contract coherently.",
            "Quote distortion is a shrinkage toward one half plus independent noise.",
            "Quote age is a deterministic function of the group's liquidity regime.",
        ],
        support=(
            f"{resolved['groups']} event groups over consecutive synthetic days; "
            "the last block is the confirmation split."
        ),
    )


def _confirmation(seed, knobs):
    rows, meta, _ = _build(seed, knobs)
    return [r for r in rows if r["split"] == "confirmation"], meta


def baseline(seed=DEFAULT_SEED, **knobs):
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    confirmation, _ = _confirmation(seed, resolved)
    values = {r["id"]: r["price"] for r in confirmation}
    return predictions_of("baseline", SPEC, values, "raw_quote_proxy", SPEC.baseline_rationale)


def oracle(seed=DEFAULT_SEED, **knobs):
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    confirmation, meta = _confirmation(seed, resolved)
    values = {r["id"]: meta["latent"][r["id"]] for r in confirmation}
    return predictions_of("oracle", SPEC, values, "planted_latent_probability", SPEC.oracle_rationale)


def trivial(seed=DEFAULT_SEED, **knobs):
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    confirmation, _ = _confirmation(seed, resolved)
    values = {r["id"]: 0.5 for r in confirmation}
    return predictions_of("trivial", SPEC, values, "constant_one_half", "Publish one half everywhere.")


def evaluate(predictions, seed=DEFAULT_SEED, **knobs):
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    confirmation, _ = _confirmation(seed, resolved)
    expected = {r["id"] for r in confirmation}
    values = require_keys(unwrap(predictions), expected)
    cleaned = {k: probability(v, "resolution probability") for k, v in values.items()}
    host = score(confirmation, cleaned)
    links = relations(confirmation)
    return scored(
        SPEC,
        seed,
        primary=host["brier"],
        secondary={
            "log_loss": host["log_loss"],
            "ece_5_bins": host["ece"],
            "coherence_violations": host["coherence_violations"],
            "accepted_relations": host["relation_count"],
            "rejected_near_matches": len(links["rejected_near_matches"]),
            "reliability_bins": host["reliability"],
            "observations": host["observations"],
            "protocol_digest": host["protocol_digest"],
        },
        losses=list(host["event_scores"].values()),
        units=host["independent_units"],
        notes=[
            "Scored by the existing host P1 evaluator; this pack supplies no scores of its own.",
            "Independent units are event groups, not individual contracts.",
            "Coherence is reported separately and never substitutes for accuracy.",
        ],
    )


def narrative():
    return story(
        SPEC,
        decision=(
            "A desk is about to publish probabilities for a ladder of linked contracts. "
            "Some of the posted quotes contradict each other. Which numbers go out?"
        ),
        agent_task=(
            "Two things at once. First, decide which contracts are actually the same event: "
            "the pack contains near-matches that share a group but differ in station or "
            "settlement clock, and joining them is the error the case is about. Second, "
            "learn from the fitting rows that quote age predicts how far a quote has been "
            "shrunk toward one half, and undo it on the confirmation rows."
        ),
        good_answer=(
            "A Brier score below the raw-quote baseline, zero coherence violations on the "
            "relations the settlement text supports, no relation asserted across the "
            "near-matches, and a statement that coherence is not accuracy."
        ),
        trap=(
            "Publishing the quotes is defensible and scores well. It also inherits every "
            "stale-group distortion, and it inverts ladder pairs, because independent quote "
            "noise does not respect logic."
        ),
        two_minutes=[
            "Show one event group: three contracts, one realised temperature, one ladder.",
            "Show the admission receipt from the existing host P1 contract.",
            "Run the baseline: raw quotes, with a nonzero coherence-violation count.",
            "Show quote_age_seconds and ask what it predicts.",
            "Reveal the oracle: same Brier protocol, zero violations, big gap.",
            "Say the sentence: synthetic contracts, future dates, no venue, no trade.",
        ],
    )
