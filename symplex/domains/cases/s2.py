"""S2 -- moderation workload under coupled community feedback with a delay.

Generator: twenty communities emitting cross-community links week by week. Each community
has its own baseline rate of negative links. A sparse, directed coupling matrix adds a
delayed excitation term: a burst in one community raises its children's rate three weeks
later. A platform-wide event index, published and visible, loads on every community at
once and is the rival explanation for the same correlation.

The raw link log is the thing an operator actually has: one row per link, with duplicate
rows where the ingest ran twice, whole community-weeks missing where the logger fell over,
and two regional shards whose week labels are one week apart. Aggregating that log naively
produces the wrong series before any model is fitted.

Synthetic. Recovering the coupling demonstrates inference mechanics, not a claim about
moderation load on any real platform.
"""

import numpy as np
from scipy import stats

from symplex.core.contracts import Invalid
from symplex.domains.cases import (
    CaseEndpoint,
    CaseMetric,
    CaseSpec,
    NumericCheck,
    absolute_errors,
    check_knobs,
    check_seed,
    envelope,
    predictions_of,
    rank_auc,
    real,
    require_keys,
    scored,
    sealed,
    story,
    stream,
    table,
    unwrap,
)

CASE_ID = "s2"
GENERATOR = "s2_delayed_community_excitation_v1"
DEFAULT_SEED = 7
KNOBS = {"communities": (8, 30, 20), "weeks": (30, 70, 52), "horizon": (2, 4, 3)}
COUPLING_DELAY_WEEKS = 3
DISPERSION = 4.0
DUPLICATE_RATE = 0.07
SHARD_WEEK_OFFSET = 1

SPEC = CaseSpec(
    id=CASE_ID,
    manifest_case="S2",
    domain="social_simulation",
    title="Can feedback between communities improve forecasts of moderation workload?",
    decision=(
        "How many reviewer hours to roster for each community over the next three weeks, "
        "and whether a burst in one community is a reason to staff up in another."
    ),
    beneficiary="A moderation operations lead rostering a fixed pool of reviewers.",
    mechanism_question=(
        "Does a burst propagate between specific communities with a delay, or does a "
        "platform-wide event raise every community at once?"
    ),
    rivals=[
        "Persistent per-community rates and recent activity explain the counts.",
        "A sparse directed coupling adds delayed excitation between specific communities.",
        "A shared platform-wide event index explains the same co-movement.",
        "Duplicate ingest rows and logger outages explain the apparent bursts.",
    ],
    endpoints=[
        CaseEndpoint(
            id="expected_count",
            name="Expected negative links per community-week",
            unit="links per week",
            meaning="Converted directly into rostered reviewer hours.",
        ),
        CaseEndpoint(
            id="interval",
            name="Eighty per cent forecast interval",
            unit="links per week",
            meaning="Decides how much surge capacity to hold back.",
        ),
        CaseEndpoint(
            id="coupling",
            name="Directed coupling score between community pairs",
            unit="score",
            meaning="Whether a burst in one community justifies staffing another.",
        ),
    ],
    evidence_channels=[
        "A raw cross-community link log with labels, post identifiers and duplicates.",
        "A logging coverage table saying which community-weeks were captured at all.",
        "A published platform-wide event index that extends into the forecast weeks.",
        "A community table with shard, region and joining week.",
    ],
    nuisance=[
        "Some link rows are ingested twice under the same post identifier.",
        "Whole community-weeks are missing, and outages are likelier in the busiest weeks.",
        "Two regional shards label their weeks one week apart.",
        "Labels come from a mix of crowdsourcing and a classifier, and are proxies only.",
    ],
    primary_metric=CaseMetric(
        id="count_mae",
        name="Held-out count mean absolute error",
        lower_is_better=True,
        definition="Mean absolute error of expected negative links over held-out community-weeks.",
    ),
    secondary_metrics=[
        CaseMetric(
            id="poisson_deviance",
            name="Mean Poisson deviance",
            lower_is_better=True,
            definition="Mean unit Poisson deviance of the expected counts against realised counts.",
        ),
        CaseMetric(
            id="coupling_rank_auc",
            name="Coupling recovery rank AUC",
            lower_is_better=False,
            definition="Rank AUC of submitted pair scores against the planted directed couplings.",
        ),
        CaseMetric(
            id="interval_coverage_80",
            name="Eighty per cent interval coverage",
            lower_is_better=False,
            definition="Share of held-out community-weeks whose realised count fell inside the interval.",
        ),
        CaseMetric(
            id="busy_community_mae",
            name="Mean absolute error in the busiest third",
            lower_is_better=True,
            definition="Held-out error restricted to the third of communities with the highest baseline load.",
        ),
    ],
    baseline_name="Recency-weighted per-community rate with a negative-binomial interval",
    baseline_rationale=(
        "The rostering rule an operations lead already uses: an exponentially weighted mean "
        "of each community's recent captured weeks, carried flat across the horizon, with a "
        "negative-binomial interval around it. It uses coverage correctly and ignores only "
        "the cross-community structure."
    ),
    oracle_rationale=(
        "The planted intensity: true baselines, true couplings at the true delay, and the "
        "true event loading, with the negative-binomial interval that generated the counts."
    ),
    trivial_name="Platform-wide mean count for every community",
    hidden_markers=[
        "coupling_matrix_true",
        "coupling_delay_weeks",
        "baseline_rate_by_community",
        "event_loading_by_community",
        "counts_by_community_week",
        "shard_week_offset",
    ],
    tables=["s2_links.csv", "s2_coverage.csv", "s2_event_index.csv", "s2_communities.csv"],
    knobs=["communities", "weeks", "horizon"],
    limits=(
        "Negative links do not measure harassment, polarisation or psychological state. "
        "A recovered coupling here is a recovered simulator parameter, nothing more."
    ),
)

NUMERIC_CHECKS = [
    NumericCheck(
        id="s2_link_labels",
        operation="bounds",
        csv_filename="s2_links.csv",
        columns=["label"],
        group_columns=["source_community"],
        time_column=None,
        row_filters=[],
        lower=-1.0,
        upper=1.0,
        reference_value=None,
        tolerance=0.0,
        direction=None,
        units="signed link label",
        rationale="The label is the only thing separating moderation load from ordinary traffic.",
    ),
    NumericCheck(
        id="s2_event_index_finite",
        operation="finite",
        csv_filename="s2_event_index.csv",
        columns=["event_index"],
        group_columns=[],
        time_column=None,
        row_filters=[],
        lower=None,
        upper=None,
        reference_value=None,
        tolerance=0.0,
        direction=None,
        units="dimensionless index, one is the quiet level",
        rationale="The shared driver must be readable for the common-cause rival to be testable.",
    ),
]


def _simulate(seed, knobs):
    count, weeks, horizon = knobs["communities"], knobs["weeks"], knobs["horizon"]
    if horizon > COUPLING_DELAY_WEEKS:
        raise Invalid("S2 keeps the horizon within the coupling delay so all lags are observed")
    setup = stream(seed, "s2/setup")
    counts_rng = stream(seed, "s2/counts")
    baseline_rate = setup.uniform(1.6, 9.5, size=count)
    loading = setup.uniform(0.25, 1.25, size=count)
    coupling = np.zeros((count, count))
    for target in range(count):
        parents = setup.choice(
            [j for j in range(count) if j != target],
            size=int(setup.integers(0, 4)),
            replace=False,
        )
        for parent in parents:
            coupling[target, parent] = float(setup.uniform(0.10, 0.26))
        total = coupling[target].sum()
        if total > 0.55:
            coupling[target] *= 0.55 / total
    event = np.ones(weeks + horizon + 2)
    level = 0.0
    for t in range(event.size):
        level = 0.7 * level + counts_rng.normal(0, 0.16)
        if counts_rng.random() < 0.07:
            level += counts_rng.uniform(0.5, 1.4)
        event[t] = 1.0 + max(-0.5, level)
    negative = np.zeros((count, weeks + horizon), dtype=int)
    positive = np.zeros((count, weeks + horizon), dtype=int)
    intensity = np.zeros((count, weeks + horizon))
    for t in range(weeks + horizon):
        lagged = negative[:, t - COUPLING_DELAY_WEEKS] if t >= COUPLING_DELAY_WEEKS else np.zeros(count)
        rate = baseline_rate + coupling @ lagged + loading * baseline_rate * (event[t] - 1.0)
        rate = np.clip(rate, 0.25, 90.0)
        intensity[:, t] = rate
        probability_success = DISPERSION / (DISPERSION + rate)
        negative[:, t] = counts_rng.negative_binomial(DISPERSION, probability_success)
        positive[:, t] = counts_rng.poisson(np.clip(rate * 1.15, 0.3, 120.0))
    return {
        "count": count,
        "weeks": weeks,
        "horizon": horizon,
        "baseline_rate": baseline_rate,
        "loading": loading,
        "coupling": coupling,
        "event": event,
        "negative": negative,
        "positive": positive,
        "intensity": intensity,
    }


def _coverage(state, seed, knobs):
    gaps = stream(seed, "s2/coverage")
    busy = np.quantile(state["negative"][:, : state["weeks"]], 0.85)
    logged = np.ones((state["count"], state["weeks"]), dtype=bool)
    for i in range(state["count"]):
        for t in range(state["weeks"]):
            drop = 0.04 + (0.2 if state["negative"][i, t] > busy else 0.0)
            logged[i, t] = gaps.random() >= drop
    return logged


def _keys(state):
    return [
        f"c{i:02d}_w{t:02d}"
        for t in range(state["weeks"], state["weeks"] + state["horizon"])
        for i in range(state["count"])
    ]


def generate(seed=DEFAULT_SEED, **knobs):
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    state = _simulate(seed, resolved)
    logged = _coverage(state, seed, resolved)
    links = stream(seed, "s2/links")
    shard = ["shard_east" if i % 2 else "shard_west" for i in range(state["count"])]
    link_rows, coverage_rows, event_rows, community_rows = [], [], [], []
    post = 0
    for t in range(state["weeks"]):
        for i in range(state["count"]):
            coverage_rows.append(
                {
                    "community_id": f"c{i:02d}",
                    "reported_week": t + (SHARD_WEEK_OFFSET if shard[i] == "shard_east" else 0),
                    "logging_status": "captured" if logged[i, t] else "outage",
                }
            )
            if not logged[i, t]:
                continue
            for label, total in ((-1, state["negative"][i, t]), (1, state["positive"][i, t])):
                for _ in range(int(total)):
                    post += 1
                    target = int(links.integers(0, state["count"]))
                    if target == i:
                        target = (i + 1) % state["count"]
                    row = {
                        "reported_week": t + (SHARD_WEEK_OFFSET if shard[i] == "shard_east" else 0),
                        "source_community": f"c{i:02d}",
                        "target_community": f"c{target:02d}",
                        "post_id": f"p{post:06d}",
                        "label": label,
                    }
                    link_rows.append(row)
                    if links.random() < DUPLICATE_RATE:
                        link_rows.append(dict(row))
    for t in range(state["weeks"] + state["horizon"]):
        event_rows.append({"week": t, "event_index": round(float(state["event"][t]), 4)})
    for i in range(state["count"]):
        community_rows.append(
            {
                "community_id": f"c{i:02d}",
                "shard": shard[i],
                "region": "region_1" if i % 3 else "region_2",
                "joined_week": 0,
            }
        )
    tables = {
        "s2_links.csv": table(
            ["reported_week", "source_community", "target_community", "post_id", "label"], link_rows
        ),
        "s2_coverage.csv": table(
            ["community_id", "reported_week", "logging_status"], coverage_rows
        ),
        "s2_event_index.csv": table(["week", "event_index"], event_rows),
        "s2_communities.csv": table(
            ["community_id", "shard", "region", "joined_week"], community_rows
        ),
    }
    return envelope(
        SPEC,
        seed,
        resolved,
        GENERATOR,
        tables,
        question=(
            "Forecast negative-link counts for every community in each of the next three "
            "weeks, with an eighty per cent interval. You may also submit a directed "
            "coupling score for community pairs."
        ),
        submission={
            "keys": "one entry per community and forecast week, formatted cNN_wWW",
            "fields": {
                "expected_count": "expected negative links, at least zero",
                "lower_80": "lower end of the eighty per cent interval",
                "upper_80": "upper end of the eighty per cent interval",
            },
            "optional_key": (
                "coupling_scores maps 'source>target' to a score; higher means a burst in "
                "source raises target later"
            ),
            "example_key": f"c00_w{resolved['weeks']:02d}",
        },
        metadata={
            "observed_weeks": resolved["weeks"],
            "forecast_weeks": list(range(resolved["weeks"], resolved["weeks"] + resolved["horizon"])),
            "event_index_extends_into_forecast": True,
            "clock": (
                "Weeks are reported on each shard's own calendar; the two shards were "
                "commissioned a week apart and the pack does not state the correction."
            ),
            "missingness": (
                "A community-week marked outage has no rows at all in the link log. A zero "
                "count and an outage are different things."
            ),
            "duplicates": (
                "The ingest occasionally wrote a link twice under the same post identifier."
            ),
            "not_observed": [
                "which community pairs are actually coupled, or with what delay",
                "the true rate behind any week's count",
                "anything about the people or content behind a link",
            ],
        },
        limitations=[
            "Outages concentrate in the busiest weeks, so a naive mean is biased low.",
            "Labels are imperfect proxies produced by mixed crowdsourcing and classification.",
            "Forecast improvement is not evidence that one community causes load in another.",
        ],
    )


def truth(seed=DEFAULT_SEED, **knobs):
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    state = _simulate(seed, resolved)
    keys = _keys(state)
    counts = {}
    for key in keys:
        i = int(key[1:3])
        t = int(key.split("_w")[1])
        counts[key] = int(state["negative"][i, t])
    edges = [
        [f"c{parent:02d}", f"c{target:02d}", round(float(state["coupling"][target, parent]), 6)]
        for target in range(state["count"])
        for parent in range(state["count"])
        if state["coupling"][target, parent] > 0
    ]
    return sealed(
        SPEC,
        seed,
        resolved,
        GENERATOR,
        facts={
            "coupling_matrix_true": edges,
            "coupling_delay_weeks": COUPLING_DELAY_WEEKS,
            "baseline_rate_by_community": {
                f"c{i:02d}": round(float(state["baseline_rate"][i]), 6) for i in range(state["count"])
            },
            "event_loading_by_community": {
                f"c{i:02d}": round(float(state["loading"][i]), 6) for i in range(state["count"])
            },
            "counts_by_community_week": counts,
            "shard_week_offset": {"shard_west": 0, "shard_east": SHARD_WEEK_OFFSET},
        },
        assumptions=[
            "Counts are negative binomial with a fixed dispersion around a linear intensity.",
            "Coupling enters at exactly one delay, identical for every pair.",
            "The event index acts multiplicatively on each community's own baseline rate.",
        ],
        support=(
            f"{resolved['communities']} communities over {resolved['weeks']} observed weeks; "
            f"the next {resolved['horizon']} weeks are scored."
        ),
    )


def _interval(rate):
    lower = float(stats.nbinom.ppf(0.1, DISPERSION, DISPERSION / (DISPERSION + max(rate, 0.05))))
    upper = float(stats.nbinom.ppf(0.9, DISPERSION, DISPERSION / (DISPERSION + max(rate, 0.05))))
    return lower, upper


def baseline(seed=DEFAULT_SEED, **knobs):
    """Exponentially weighted mean of each community's captured weeks, carried flat."""
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    state = _simulate(seed, resolved)
    logged = _coverage(state, seed, resolved)
    values = {}
    for i in range(state["count"]):
        weight, total = 0.0, 0.0
        for t in range(state["weeks"]):
            if not logged[i, t]:
                continue
            factor = 0.94 ** (state["weeks"] - 1 - t)
            total += factor * state["negative"][i, t]
            weight += factor
        rate = total / weight if weight else float(np.mean(state["negative"][:, : state["weeks"]]))
        lower, upper = _interval(rate)
        for t in range(state["weeks"], state["weeks"] + state["horizon"]):
            values[f"c{i:02d}_w{t:02d}"] = {
                "expected_count": round(float(rate), 4),
                "lower_80": lower,
                "upper_80": upper,
            }
    volume = state["negative"][:, : state["weeks"]].mean(axis=1)
    values["coupling_scores"] = {
        f"c{a:02d}>c{b:02d}": round(float(volume[a] * volume[b]), 4)
        for a in range(state["count"])
        for b in range(state["count"])
        if a != b
    }
    return predictions_of(
        "baseline", SPEC, values, "recency_weighted_community_rate", SPEC.baseline_rationale
    )


def oracle(seed=DEFAULT_SEED, **knobs):
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    state = _simulate(seed, resolved)
    values = {}
    for t in range(state["weeks"], state["weeks"] + state["horizon"]):
        for i in range(state["count"]):
            rate = float(state["intensity"][i, t])
            lower, upper = _interval(rate)
            values[f"c{i:02d}_w{t:02d}"] = {
                "expected_count": round(rate, 4),
                "lower_80": lower,
                "upper_80": upper,
            }
    values["coupling_scores"] = {
        f"c{parent:02d}>c{target:02d}": round(float(state["coupling"][target, parent]), 6)
        for target in range(state["count"])
        for parent in range(state["count"])
        if parent != target
    }
    return predictions_of("oracle", SPEC, values, "planted_intensity", SPEC.oracle_rationale)


def trivial(seed=DEFAULT_SEED, **knobs):
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    state = _simulate(seed, resolved)
    logged = _coverage(state, seed, resolved)
    observed = state["negative"][:, : state["weeks"]][logged]
    rate = float(np.mean(observed)) if observed.size else 1.0
    lower, upper = _interval(rate)
    values = {
        key: {"expected_count": rate, "lower_80": lower, "upper_80": upper}
        for key in _keys(state)
    }
    return predictions_of(
        "trivial", SPEC, values, "platform_wide_mean", "One platform-wide rate for every community-week."
    )


def evaluate(predictions, seed=DEFAULT_SEED, **knobs):
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    state = _simulate(seed, resolved)
    facts = truth(seed, **resolved)["facts"]
    actual = facts["counts_by_community_week"]
    values = dict(unwrap(predictions))
    coupling = values.pop("coupling_scores", None)
    require_keys(values, actual)
    expected, covered = {}, []
    for key, row in values.items():
        if not isinstance(row, dict) or set(row) != {"expected_count", "lower_80", "upper_80"}:
            raise Invalid("Each S2 entry needs exactly expected_count, lower_80 and upper_80")
        rate = real(row["expected_count"], "expected count")
        if rate < 0:
            raise Invalid("Expected counts cannot be negative")
        low = real(row["lower_80"], "interval lower bound")
        high = real(row["upper_80"], "interval upper bound")
        if low > high:
            raise Invalid("Interval lower bound exceeds its upper bound")
        expected[key] = rate
        covered.append(low <= actual[key] <= high)
    losses = absolute_errors(expected, actual)
    deviance = []
    for key in sorted(actual):
        observed = float(actual[key])
        rate = max(expected[key], 1e-9)
        term = observed * np.log(observed / rate) if observed > 0 else 0.0
        deviance.append(2.0 * (term - (observed - rate)))
    auc = None
    if isinstance(coupling, dict) and coupling:
        planted = {f"{a}>{b}" for a, b, _ in facts["coupling_matrix_true"]}
        labels = {k: int(k in planted) for k in coupling}
        if set(coupling) >= planted:
            auc = rank_auc({k: real(v, "coupling score") for k, v in coupling.items()}, labels)
    order = np.argsort(-state["baseline_rate"])
    busy = {f"c{i:02d}" for i in order[: max(1, state["count"] // 3)]}
    busy_errors = [abs(expected[k] - actual[k]) for k in actual if k[:3] in busy]
    return scored(
        SPEC,
        seed,
        primary=float(np.mean(losses)),
        secondary={
            "poisson_deviance": float(np.mean(deviance)),
            "coupling_rank_auc": auc,
            "interval_coverage_80": float(np.mean(covered)),
            "busy_community_mae": float(np.mean(busy_errors)) if busy_errors else None,
            "planted_couplings": len(facts["coupling_matrix_true"]),
            "mean_realised_count": float(np.mean(list(actual.values()))),
        },
        losses=losses,
        units=len(actual),
        notes=[
            "Community-weeks in the same week share the event index and are not independent.",
            "Coverage is descriptive; the interval was not calibrated on held-out data.",
        ],
    )


def narrative():
    return story(
        SPEC,
        decision=(
            "A moderation lead rosters reviewers three weeks ahead. A burst just hit one "
            "community. Does anyone else need staffing?"
        ),
        agent_task=(
            "Get a clean weekly series out of a raw link log first: drop duplicate post "
            "identifiers, align the two shards' week labels, and treat an outage week as "
            "missing rather than zero. Only then can the real question be asked, which is "
            "whether a delayed cross-community coupling exists on top of a published "
            "platform-wide event index that already explains most of the co-movement."
        ),
        good_answer=(
            "Forecast error below the recency-weighted rate, most of the gain in the busiest "
            "third, coupling scores that rank the planted directed pairs above the rest, and "
            "an explicit refusal to call the coupling causal."
        ),
        trap=(
            "Aggregating the log without deduplicating and without the coverage table inflates "
            "busy weeks and zeroes out outages, which manufactures exactly the bursty "
            "co-movement the case asks about."
        ),
        two_minutes=[
            "Aggregate the log naively; show the spurious spikes and zeros.",
            "Apply the coverage table and dedupe; the series changes shape.",
            "Show the event index explaining most of the co-movement.",
            "Run the recency-weighted baseline and note where it misses.",
            "Reveal the planted coupling matrix and its three-week delay.",
            "Say the sentence: synthetic log, recovered coupling, no causal claim.",
        ],
    )
