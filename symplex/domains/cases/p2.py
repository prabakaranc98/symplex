"""P2 -- when low activity and stale quotes make a posted forecast unreliable.

Generator: a book of resolved binary events across several categories. Each event has a
latent probability, a posted probability, a quote age and a traded volume. The posted
number is the latent one shrunk toward a half, and the shrinkage really does grow with
quote age and fall with volume -- that is H1, and it is planted.

H2 is planted too, and it is why the case is not trivial. Category difficulty drives both
sides at once: hard categories have latent probabilities near a half *and* thin books. So
the raw association between low activity and high loss overstates the part that
recalibration can fix. The sealed truth records the split between distortion the desk can
undo and irreducible difficulty it cannot.

Synthetic. Recovering the distortion function demonstrates inference mechanics, not that
liquidity causes calibration in any real venue.
"""

import numpy as np

from symplex.core.contracts import Invalid
from symplex.domains.cases import (
    CaseEndpoint,
    CaseMetric,
    CaseSpec,
    NumericCheck,
    brier_losses,
    check_knobs,
    check_seed,
    envelope,
    log_losses,
    predictions_of,
    probability,
    rank_auc,
    reliability,
    require_keys,
    scored,
    sealed,
    story,
    stream,
    table,
    unwrap,
)

CASE_ID = "p2"
GENERATOR = "p2_activity_dependent_calibration_v1"
DEFAULT_SEED = 7
KNOBS = {"events": (60, 400, 320), "holdout_fraction": (0.2, 0.6, 0.375), "missing_rate": (0.0, 0.4, 0.06)}
AGE_COEFFICIENT = 0.55
VOLUME_COEFFICIENT = 0.20
MAX_SHRINK = 0.88

SPEC = CaseSpec(
    id=CASE_ID,
    manifest_case="P2",
    domain="prediction_markets",
    title="When do low activity and stale observations make forecasts unreliable?",
    decision=(
        "Which posted forecasts to publish as they stand, which to restate, and which to "
        "hold back pending more evidence, under a fixed evidence-gathering budget."
    ),
    beneficiary="A desk that can afford to re-research only a fraction of its book.",
    mechanism_question=(
        "Is the extra loss on thin, stale events something recalibration can remove, or is "
        "it irreducible difficulty that happens to correlate with thin books?"
    ),
    rivals=[
        "Reliability varies with quote age and activity, and is recoverable.",
        "Category difficulty and time to resolution explain the same association.",
        "Pooled recalibration is enough; the activity split adds nothing.",
    ],
    endpoints=[
        CaseEndpoint(
            id="calibrated_probability",
            name="Restated probability",
            unit="probability",
            meaning="The number the desk publishes after review.",
        ),
        CaseEndpoint(
            id="evidence_flag",
            name="Evidence-gathering priority",
            unit="score in [0, 1]",
            meaning="Which events are worth spending the review budget on.",
        ),
    ],
    evidence_channels=[
        "Fitting events with posted probability, quote age, volume, category and resolution.",
        "Held-out events with everything except the resolution.",
        "A category table naming each category's typical book depth.",
    ],
    nuisance=[
        "Volume is suppressed by the venue on very thin books, which is MNAR missingness.",
        "Quote age arrives in minutes from one venue and seconds from another.",
        "Category difficulty drives latent probability and book depth together.",
        "Time to resolution is counted in venue-local calendar days.",
    ],
    primary_metric=CaseMetric(
        id="holdout_brier",
        name="Held-out Brier score",
        lower_is_better=True,
        definition="Mean squared error of the restated probability over held-out events.",
    ),
    secondary_metrics=[
        CaseMetric(
            id="holdout_log_loss",
            name="Held-out log loss",
            lower_is_better=True,
            definition="Mean negative log likelihood over held-out events.",
        ),
        CaseMetric(
            id="calibration_error",
            name="Expected calibration error",
            lower_is_better=True,
            definition="Five-bin expected calibration error of the restated probabilities.",
        ),
        CaseMetric(
            id="flag_rank_auc",
            name="Evidence-flag rank AUC",
            lower_is_better=False,
            definition="Rank AUC of the flag score against events whose planted distortion is above median.",
        ),
        CaseMetric(
            id="thin_book_brier",
            name="Brier on the thinnest activity third",
            lower_is_better=True,
            definition="Held-out Brier restricted to the lowest-volume third, where the distortion is largest.",
        ),
    ],
    baseline_name="Pooled activity-blind recalibration",
    baseline_rationale=(
        "A single logistic recalibration of the posted probability, fitted on the fitting "
        "events and applied everywhere. This is what a competent desk already does; it "
        "removes the average distortion and is blind to which events carry it."
    ),
    oracle_rationale=(
        "The planted latent probability and the planted per-event distortion. Bayes-optimal "
        "for this generator; the gap to the pooled calibrator is the activity-aware headroom."
    ),
    trivial_name="Constant one-half with a random flag order",
    hidden_markers=[
        "resolution_by_event",
        "latent_probability_by_event",
        "distortion_by_event",
        "activity_distortion_coefficients",
        "category_difficulty_true",
        "irreducible_loss_by_regime",
    ],
    tables=["p2_fitting_events.csv", "p2_holdout_events.csv", "p2_categories.csv"],
    knobs=["events", "holdout_fraction", "missing_rate"],
    limits=(
        "A predictive association between activity and reliability does not establish that "
        "adding liquidity would improve accuracy. No venue or archive is represented."
    ),
)

NUMERIC_CHECKS = [
    NumericCheck(
        id="p2_posted_probabilities_bounded",
        operation="bounds",
        csv_filename="p2_holdout_events.csv",
        columns=["posted_probability"],
        group_columns=["category"],
        time_column=None,
        row_filters=[],
        lower=0.0,
        upper=1.0,
        reference_value=None,
        tolerance=0.0,
        direction=None,
        units="probability",
        rationale="A posted number outside the unit interval cannot be recalibrated.",
    ),
    NumericCheck(
        id="p2_volume_nonnegative",
        operation="bounds",
        csv_filename="p2_fitting_events.csv",
        columns=["contract_volume"],
        group_columns=["category"],
        time_column=None,
        row_filters=[{"column": "volume_reported", "equals": "yes"}],
        lower=0.0,
        upper=1000000.0,
        reference_value=None,
        tolerance=0.0,
        direction=None,
        units="contracts",
        rationale="Suppressed volumes must be absent, never encoded as a negative sentinel.",
    ),
]

_CATEGORY_TABLE = (
    ("weather_threshold", 0.18, 6.4, "deep"),
    ("economic_release", 0.42, 5.6, "moderate"),
    ("policy_decision", 0.72, 4.6, "thin"),
    ("sports_series", 0.30, 6.0, "deep"),
)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _build(seed, knobs):
    count = knobs["events"]
    holdout = int(count * knobs["holdout_fraction"])
    if holdout < 8 or count - holdout < 16:
        raise Invalid("P2 needs at least 16 fitting and 8 held-out events")
    draw = stream(seed, "p2/events")
    names = [c[0] for c in _CATEGORY_TABLE]
    index = draw.integers(0, len(names), size=count)
    difficulty = np.array([_CATEGORY_TABLE[i][1] for i in index])
    depth = np.array([_CATEGORY_TABLE[i][2] for i in index])
    # Difficulty pulls the latent probability toward a half and thins the book at once.
    raw = draw.normal(0, 2.3, size=count)
    latent = _sigmoid(raw * (1.0 - 0.55 * difficulty))
    volume = np.exp(depth - 1.7 * difficulty + draw.normal(0, 0.65, size=count))
    age_minutes = np.exp(draw.normal(3.2 + 1.5 * difficulty, 0.85, size=count))
    days_to_resolution = draw.integers(1, 31, size=count)
    z_age = (np.log1p(age_minutes) - 4.0) / 1.2
    z_volume = (np.log(volume) - 5.4) / 1.1
    shrink = MAX_SHRINK * _sigmoid(AGE_COEFFICIENT * z_age * 3.0 - VOLUME_COEFFICIENT * z_volume * 3.0 - 0.4)
    noise = draw.normal(0, 0.03, size=count)
    posted = np.clip(0.5 + (latent - 0.5) * (1.0 - shrink) + noise, 0.01, 0.99)
    resolution = (draw.random(size=count) < latent).astype(int)
    order = np.argsort(days_to_resolution + draw.random(size=count))
    ids = [f"evt_{i:04d}" for i in range(count)]
    split = {ids[j]: ("holdout" if rank >= count - holdout else "fitting") for rank, j in enumerate(order)}
    return {
        "ids": ids,
        "category": [names[i] for i in index],
        "latent": latent,
        "posted": posted,
        "volume": volume,
        "age_minutes": age_minutes,
        "days": days_to_resolution,
        "shrink": shrink,
        "resolution": resolution,
        "split": split,
        "difficulty": difficulty,
    }


def _rows(state, seed, knobs, phase):
    gaps = stream(seed, "p2/suppression")
    rows = []
    for i, ident in enumerate(state["ids"]):
        if state["split"][ident] != phase:
            continue
        volume = float(state["volume"][i])
        suppressed = gaps.random() < knobs["missing_rate"] + (0.3 if volume < 90 else 0.0)
        seconds = i % 4 == 0
        rows.append(
            {
                "event_id": ident,
                "category": state["category"][i],
                "posted_probability": round(float(state["posted"][i]), 4),
                "quote_age": round(float(state["age_minutes"][i]) * (60.0 if seconds else 1.0), 2),
                "quote_age_unit": "seconds" if seconds else "minutes",
                "contract_volume": None if suppressed else round(volume, 1),
                "volume_reported": "no" if suppressed else "yes",
                "days_to_resolution": int(state["days"][i]),
                "resolved_yes": int(state["resolution"][i]) if phase == "fitting" else None,
            }
        )
    if phase == "holdout":
        for row in rows:
            row.pop("resolved_yes")
    return rows


def generate(seed=DEFAULT_SEED, **knobs):
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    state = _build(seed, resolved)
    fitting = _rows(state, seed, resolved, "fitting")
    holdout = _rows(state, seed, resolved, "holdout")
    fitting_columns = [
        "event_id",
        "category",
        "posted_probability",
        "quote_age",
        "quote_age_unit",
        "contract_volume",
        "volume_reported",
        "days_to_resolution",
        "resolved_yes",
    ]
    holdout_columns = [c for c in fitting_columns if c != "resolved_yes"]
    category_rows = [
        {
            "category": name,
            "typical_book_depth": band,
            "settlement_clock": "venue local calendar day",
        }
        for name, _, _, band in _CATEGORY_TABLE
    ]
    tables = {
        "p2_fitting_events.csv": table(fitting_columns, fitting),
        "p2_holdout_events.csv": table(holdout_columns, holdout),
        "p2_categories.csv": table(
            ["category", "typical_book_depth", "settlement_clock"], category_rows
        ),
    }
    return envelope(
        SPEC,
        seed,
        resolved,
        GENERATOR,
        tables,
        question=(
            "For every held-out event give a restated probability, and a priority score for "
            "spending review budget on it. The fitting events carry their outcomes."
        ),
        submission={
            "keys": "one entry per event_id in p2_holdout_events.csv",
            "fields": {
                "probability": "restated probability in [0, 1]",
                "evidence_flag": "review priority in [0, 1]; higher means review first",
            },
            "example_key": holdout[0]["event_id"],
        },
        metadata={
            "fitting_events": len(fitting),
            "held_out_events": len(holdout),
            "units": "Read quote_age_unit per row; one venue reports seconds and the other minutes.",
            "missingness": (
                "The venue suppresses contract_volume on very thin books, so a blank volume "
                "is itself informative and is not missing at random."
            ),
            "not_observed": [
                "order book depth or trade-level records",
                "the mapping from age and activity to quote quality",
                "how much of an event's difficulty is irreducible",
            ],
        },
        limitations=[
            "Held-out outcomes are single binary draws; per-event loss is very noisy.",
            "Association between activity and reliability is not a causal claim.",
            "No venue, archive or trade is represented anywhere in this pack.",
        ],
    )


def truth(seed=DEFAULT_SEED, **knobs):
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    state = _build(seed, resolved)
    keys = [i for i in state["ids"] if state["split"][i] == "holdout"]
    lookup = {ident: i for i, ident in enumerate(state["ids"])}
    thirds = np.quantile([state["volume"][lookup[k]] for k in keys], [1 / 3, 2 / 3])
    regimes = {}
    for key in keys:
        volume = state["volume"][lookup[key]]
        regimes[key] = "thin" if volume <= thirds[0] else ("mid" if volume <= thirds[1] else "deep")
    irreducible = {}
    for band in ("thin", "mid", "deep"):
        members = [k for k in keys if regimes[k] == band]
        latent = np.array([state["latent"][lookup[k]] for k in members])
        irreducible[band] = round(float(np.mean(latent * (1 - latent))), 6) if members else None
    return sealed(
        SPEC,
        seed,
        resolved,
        GENERATOR,
        facts={
            "resolution_by_event": {k: int(state["resolution"][lookup[k]]) for k in keys},
            "latent_probability_by_event": {k: round(float(state["latent"][lookup[k]]), 6) for k in keys},
            "distortion_by_event": {k: round(float(state["shrink"][lookup[k]]), 6) for k in keys},
            "activity_distortion_coefficients": {
                "quote_age": AGE_COEFFICIENT,
                "log_volume": -VOLUME_COEFFICIENT,
                "maximum_shrinkage": MAX_SHRINK,
            },
            "category_difficulty_true": {name: value for name, value, _, _ in _CATEGORY_TABLE},
            "irreducible_loss_by_regime": irreducible,
            "activity_regime_by_event": regimes,
        },
        assumptions=[
            "Outcomes are independent Bernoulli draws from the latent probability.",
            "Distortion is a logistic shrinkage in standardised age and log volume.",
            "Category difficulty is a single scalar acting on both probability and depth.",
        ],
        support=(
            f"{resolved['events']} events across {len(_CATEGORY_TABLE)} categories; the "
            "held-out block is the latest-resolving fraction."
        ),
    )


def _platt(x, y, iterations=25):
    """Two-parameter logistic recalibration by IRLS. Deterministic and tiny."""
    beta = np.zeros(2)
    design = np.column_stack([np.ones_like(x), x])
    for _ in range(iterations):
        p = _sigmoid(design @ beta)
        weights = np.clip(p * (1 - p), 1e-6, None)
        gradient = design.T @ (y - p)
        hessian = design.T @ (design * weights[:, None]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(hessian, gradient)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta


def _state(seed, knobs):
    state = _build(seed, knobs)
    lookup = {ident: i for i, ident in enumerate(state["ids"])}
    fitting = [i for i in state["ids"] if state["split"][i] == "fitting"]
    holdout = [i for i in state["ids"] if state["split"][i] == "holdout"]
    return state, lookup, fitting, holdout


def baseline(seed=DEFAULT_SEED, **knobs):
    """Pooled logistic recalibration fitted on the fitting split; activity plays no part."""
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    state, lookup, fitting, holdout = _state(seed, resolved)
    x = _logit(np.array([state["posted"][lookup[k]] for k in fitting]))
    y = np.array([state["resolution"][lookup[k]] for k in fitting], dtype=float)
    beta = _platt(x, y)
    volumes = np.array([state["volume"][lookup[k]] for k in holdout])
    span = float(np.log(volumes).max() - np.log(volumes).min()) or 1.0
    values = {}
    for key in holdout:
        restated = float(_sigmoid(beta[0] + beta[1] * _logit(state["posted"][lookup[key]])))
        thinness = float((np.log(volumes).max() - np.log(state["volume"][lookup[key]])) / span)
        values[key] = {
            "probability": round(restated, 6),
            "evidence_flag": round(float(np.clip(thinness, 0.0, 1.0)), 6),
        }
    return predictions_of(
        "baseline", SPEC, values, "pooled_platt_recalibration", SPEC.baseline_rationale
    )


def oracle(seed=DEFAULT_SEED, **knobs):
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    facts = truth(seed, **resolved)["facts"]
    distortion = facts["distortion_by_event"]
    span = max(distortion.values()) or 1.0
    values = {
        key: {
            "probability": facts["latent_probability_by_event"][key],
            "evidence_flag": round(float(distortion[key] / span), 6),
        }
        for key in distortion
    }
    return predictions_of("oracle", SPEC, values, "planted_latent_probability", SPEC.oracle_rationale)


def trivial(seed=DEFAULT_SEED, **knobs):
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    _, _, _, holdout = _state(seed, resolved)
    values = {k: {"probability": 0.5, "evidence_flag": 0.5} for k in holdout}
    return predictions_of(
        "trivial", SPEC, values, "constant_one_half", "Publish one half and review in arbitrary order."
    )


def evaluate(predictions, seed=DEFAULT_SEED, **knobs):
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    facts = truth(seed, **resolved)["facts"]
    actual = facts["resolution_by_event"]
    values = require_keys(unwrap(predictions), actual)
    probabilities, flags = {}, {}
    for key, row in values.items():
        if not isinstance(row, dict) or set(row) != {"probability", "evidence_flag"}:
            raise Invalid("Each P2 entry needs exactly probability and evidence_flag")
        probabilities[key] = probability(row["probability"], "restated probability")
        flags[key] = probability(row["evidence_flag"], "evidence flag")
    losses = brier_losses(probabilities, actual)
    distortion = facts["distortion_by_event"]
    median = float(np.median(list(distortion.values())))
    above = {k: int(distortion[k] > median) for k in distortion}
    regimes = facts["activity_regime_by_event"]
    thin = {k: v for k, v in actual.items() if regimes[k] == "thin"}
    bins, calibration = reliability(probabilities, actual)
    by_regime = {}
    for band in ("thin", "mid", "deep"):
        members = {k: v for k, v in actual.items() if regimes[k] == band}
        by_regime[band] = (
            round(float(np.mean(brier_losses({k: probabilities[k] for k in members}, members))), 6)
            if members
            else None
        )
    return scored(
        SPEC,
        seed,
        primary=float(np.mean(losses)),
        secondary={
            "holdout_log_loss": float(np.mean(log_losses(probabilities, actual))),
            "calibration_error": calibration,
            "flag_rank_auc": rank_auc(flags, above),
            "thin_book_brier": by_regime["thin"],
            "brier_by_activity_regime": by_regime,
            "irreducible_loss_by_regime": facts["irreducible_loss_by_regime"],
            "reliability_bins": bins,
        },
        losses=losses,
        units=len(actual),
        notes=[
            "Compare each regime's Brier against the planted irreducible loss for that regime.",
            "A gap that closes to the irreducible floor is recalibration; the floor is difficulty.",
        ],
    )


def narrative():
    return story(
        SPEC,
        decision=(
            "A desk can re-research forty of three hundred posted forecasts before "
            "publication. Which forty, and what should the other numbers say?"
        ),
        agent_task=(
            "Separate two things that look identical in the data. Thin, stale events really "
            "are distorted, and recalibration can recover that. Thin books also concentrate "
            "in categories whose events are genuinely near a coin flip, and no recalibration "
            "recovers anything there. The pack seals the irreducible loss per activity band "
            "so the split can be scored."
        ),
        good_answer=(
            "A held-out Brier below the pooled recalibrator, most of the gain landing in the "
            "thin-book third, a review queue that ranks by planted distortion rather than by "
            "raw thinness, and an explicit refusal to claim that adding liquidity would help."
        ),
        trap=(
            "Flagging every thin-book event looks right and scores poorly: it spends the "
            "review budget on events that were always going to be near one half."
        ),
        two_minutes=[
            "Show held-out loss rising as volume falls; the obvious story.",
            "Run the pooled recalibrator; it improves the average and not the thin third.",
            "Reveal the sealed irreducible loss per band; most of that gap was never fixable.",
            "Show the oracle: it beats the pooled calibrator exactly in the thin third.",
            "Show the flag AUC gap between naive thinness and planted distortion.",
            "Say the sentence: synthetic book, recovered distortion, no venue, no causal claim.",
        ],
    )
