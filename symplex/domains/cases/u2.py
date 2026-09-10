"""U2 -- congestion propagation versus a shared daily driver on one corridor.

Generator: eight loop detectors along a single directed corridor, observed over many
five-minute episodes. Every episode contains the same mild commute dip, so lagged
neighbour correlation is positive whatever happened. Each episode is then drawn from one
of two worlds. In `shockwave` an incident at one detector sends a slowdown upstream at a
finite wave speed, so onset lag grows with distance and downstream detectors speed up. In
`common_driver` a single latent demand bump hits every detector at the same instant with
detector-specific loadings, and there is no propagation at all.

The pack plants one decoy that is the entire point: half the detectors come from a second
vendor whose clock runs a fixed three steps late. A constant, distance-independent offset
manufactures apparent propagation in episodes that had none.

Synthetic. Separating the two worlds demonstrates inference mechanics, not a claim about
any real road network.
"""

import numpy as np

from symplex.core.contracts import Invalid
from symplex.domains.cases import (
    ORACLE_CONFIDENCE,
    CaseEndpoint,
    CaseMetric,
    CaseSpec,
    NumericCheck,
    absolute_errors,
    brier_losses,
    check_knobs,
    check_seed,
    envelope,
    log_losses,
    predictions_of,
    probability,
    real,
    reliability,
    require_keys,
    scored,
    sealed,
    story,
    stream,
    table,
    unwrap,
)

CASE_ID = "u2"
GENERATOR = "u2_corridor_shockwave_vs_common_driver_v1"
DEFAULT_SEED = 7
KNOBS = {"episodes": (8, 40, 24), "steps": (48, 96, 72), "missing_rate": (0.0, 0.4, 0.05)}
N_SENSORS = 8
SPACING_KM = 1.4
VENDOR_OFFSET_STEPS = 3
WAVE_SPEED_KM_PER_STEP = 0.55
SHOCKWAVE_PRIOR = 0.5

SPEC = CaseSpec(
    id=CASE_ID,
    manifest_case="U2",
    domain="urban_infrastructure",
    title="Do congestion patterns propagate through the road network or reflect shared daily demand?",
    decision=(
        "Whether tonight's operator playbook should treat a slowdown as a propagating "
        "queue worth upstream intervention, or as corridor-wide demand to be ridden out."
    ),
    beneficiary="A traffic-management centre choosing between ramp metering and doing nothing.",
    mechanism_question=(
        "Does an episode's correlated slowdown come from a queue travelling upstream at a "
        "finite speed, or from one latent driver hitting every detector simultaneously?"
    ),
    rivals=[
        "Per-detector seasonality is sufficient; the correlation is a shared commute profile.",
        "A queue propagates upstream with onset lag proportional to distance.",
        "A shared latent driver loads every detector at once with different amplitudes.",
        "Apparent lag is a clock artefact of one vendor's detectors, not traffic physics.",
    ],
    endpoints=[
        CaseEndpoint(
            id="mechanism",
            name="Probability the episode is a propagating shockwave",
            unit="probability",
            meaning="Drives whether upstream intervention is even applicable.",
        ),
        CaseEndpoint(
            id="origin",
            name="Incident detector",
            unit="detector identifier",
            meaning="Where a crew would be dispatched on a propagating episode.",
        ),
        CaseEndpoint(
            id="onset",
            name="Onset step",
            unit="five-minute steps from episode start",
            meaning="Lead time available before the queue reaches the upstream ramp.",
        ),
    ],
    evidence_channels=[
        "Five-minute detector speeds per episode, with detector dropouts.",
        "Detector table with position along the corridor, direction and vendor.",
        "Episode table with weekday and the absence of any incident feed.",
    ],
    nuisance=[
        "A commute dip is present in both worlds, so naive lagged correlation is always positive.",
        "One vendor's detector clocks run a fixed three steps late.",
        "Dropouts are more likely while a detector is congested, which is MNAR.",
        "Speeds are floored at a physical minimum, so deep slowdowns are compressed.",
    ],
    primary_metric=CaseMetric(
        id="mechanism_brier",
        name="Episode mechanism Brier score",
        lower_is_better=True,
        definition="Mean squared error of the probability that an episode was a shockwave.",
    ),
    secondary_metrics=[
        CaseMetric(
            id="origin_accuracy",
            name="Incident-detector accuracy",
            lower_is_better=False,
            definition="Fraction of true shockwave episodes whose origin detector was named correctly.",
        ),
        CaseMetric(
            id="onset_step_mae",
            name="Onset-step absolute error",
            lower_is_better=True,
            definition="Mean absolute onset error in steps on true shockwave episodes; the lead-time score.",
        ),
        CaseMetric(
            id="mechanism_log_loss",
            name="Mechanism log loss",
            lower_is_better=True,
            definition="Mean negative log likelihood of the mechanism call.",
        ),
    ],
    baseline_name="Directional cross-correlation lag heuristic",
    baseline_rationale=(
        "What a traffic engineer actually does: cross-correlate adjacent detectors, read "
        "off the lag that maximises correlation, and call it propagation when the lags are "
        "consistently nonzero and directional. It is a real method, and it is exactly the "
        "method the planted clock offset defeats."
    ),
    oracle_rationale=(
        "A reference upper bound given the planted mechanism, origin and onset, reported "
        "at declared confidence. It marks the headroom rather than a reachable target."
    ),
    trivial_name="Constant prior probability with a fixed origin guess",
    hidden_markers=[
        "mechanism_by_episode",
        "origin_sensor_by_episode",
        "onset_step_by_episode",
        "wave_speed_by_episode",
        "vendor_clock_offset_steps",
        "shared_loadings_by_episode",
    ],
    tables=["u2_speeds.csv", "u2_sensors.csv", "u2_episodes.csv"],
    knobs=["episodes", "steps", "missing_rate"],
    limits=(
        "A graph's predictive value does not establish causal traffic propagation. This "
        "pack scores separation of two declared simulators, not road behaviour."
    ),
)

NUMERIC_CHECKS = [
    NumericCheck(
        id="u2_speeds_physical",
        operation="bounds",
        csv_filename="u2_speeds.csv",
        columns=["speed_kph"],
        group_columns=["sensor_id"],
        time_column=None,
        row_filters=[{"column": "observed", "equals": "yes"}],
        lower=0.0,
        upper=140.0,
        reference_value=None,
        tolerance=0.0,
        direction=None,
        units="kilometres per hour",
        rationale="Loop-detector speeds outside this envelope would indicate a generator fault.",
    ),
    NumericCheck(
        id="u2_positions_finite",
        operation="finite",
        csv_filename="u2_sensors.csv",
        columns=["position_km"],
        group_columns=[],
        time_column=None,
        row_filters=[],
        lower=None,
        upper=None,
        reference_value=None,
        tolerance=0.0,
        direction=None,
        units="kilometres from the corridor start",
        rationale="Distance is the only thing that separates propagation from a clock offset.",
    ),
]


def _bump(t, width):
    return np.exp(-((t / width) ** 2))


def _simulate(seed, knobs):
    episodes, steps = knobs["episodes"], knobs["steps"]
    setup = stream(seed, "u2/setup")
    free_flow = setup.normal(96.0, 6.0, size=N_SENSORS).clip(78.0, 112.0)
    vendor = ["vendor_a" if s < N_SENSORS // 2 else "vendor_b" for s in range(N_SENSORS)]
    position = np.arange(N_SENSORS) * SPACING_KM
    world = stream(seed, "u2/world")
    noise = stream(seed, "u2/noise")
    grid = np.arange(steps, dtype=float)
    commute = 0.13 * _bump(grid - steps * 0.42, steps * 0.18)
    records, facts = [], {}
    for e in range(episodes):
        name = f"ep{e:02d}"
        is_shock = bool(world.random() < SHOCKWAVE_PRIOR)
        onset = float(world.integers(int(steps * 0.2), int(steps * 0.65)))
        duration = float(world.uniform(steps * 0.10, steps * 0.20))
        amplitude = float(world.uniform(0.34, 0.58))
        effect = np.zeros((N_SENSORS, steps))
        if is_shock:
            origin = int(world.integers(1, N_SENSORS - 2))
            wave = float(world.uniform(0.45, 1.9))
            for s in range(N_SENSORS):
                gap = position[s] - position[origin]
                if gap >= 0:
                    lag = gap / wave
                    decay = math_exp(-gap / 5.5)
                    effect[s] = amplitude * decay * _bump(grid - onset - lag, duration)
                else:
                    effect[s] = -0.30 * amplitude * _bump(grid - onset, duration)
            loadings = None
        else:
            origin, wave = None, None
            loadings = world.uniform(0.55, 1.45, size=N_SENSORS)
            for s in range(N_SENSORS):
                effect[s] = amplitude * loadings[s] * _bump(grid - onset, duration)
        speeds = np.zeros((N_SENSORS, steps))
        for s in range(N_SENSORS):
            clean_speed = free_flow[s] * (1.0 - commute - effect[s]) + noise.normal(0, 2.2, size=steps)
            speeds[s] = np.clip(clean_speed, 9.0, 135.0)
        shifted = np.full((N_SENSORS, steps), np.nan)
        for s in range(N_SENSORS):
            offset = VENDOR_OFFSET_STEPS if vendor[s] == "vendor_b" else 0
            if offset:
                shifted[s, offset:] = speeds[s, : steps - offset]
                shifted[s, :offset] = speeds[s, 0]
            else:
                shifted[s] = speeds[s]
        records.append(
            {
                "episode": name,
                "speeds": shifted,
                "weekday": int(e % 7),
                "is_shock": is_shock,
            }
        )
        facts[name] = {
            "mechanism": "shockwave" if is_shock else "common_driver",
            "origin": f"det{origin}" if origin is not None else None,
            "onset": onset,
            "wave": wave,
            "loadings": None if loadings is None else [round(float(x), 4) for x in loadings],
        }
    return {
        "episodes": records,
        "facts": facts,
        "free_flow": free_flow,
        "vendor": vendor,
        "position": position,
        "steps": steps,
    }


def math_exp(x):
    return float(np.exp(x))


def generate(seed=DEFAULT_SEED, **knobs):
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    state = _simulate(seed, resolved)
    gaps = stream(seed, "u2/gaps")
    speed_rows = []
    for record in state["episodes"]:
        for s in range(N_SENSORS):
            for t in range(state["steps"]):
                value = float(record["speeds"][s, t])
                drop = resolved["missing_rate"] + (0.22 if value < 52.0 else 0.0)
                observed = gaps.random() >= drop
                speed_rows.append(
                    {
                        "episode_id": record["episode"],
                        "sensor_id": f"det{s}",
                        "step": t,
                        "speed_kph": round(value, 2) if observed else None,
                        "observed": "yes" if observed else "no",
                    }
                )
    sensor_rows = [
        {
            "sensor_id": f"det{s}",
            "position_km": round(float(state["position"][s]), 2),
            "downstream_neighbour": f"det{s - 1}" if s > 0 else "",
            "upstream_neighbour": f"det{s + 1}" if s < N_SENSORS - 1 else "",
            "vendor": state["vendor"][s],
            "free_flow_kph": round(float(state["free_flow"][s]), 2),
        }
        for s in range(N_SENSORS)
    ]
    episode_rows = [
        {
            "episode_id": record["episode"],
            "weekday": record["weekday"],
            "step_seconds": 300,
            "incident_feed": "unavailable",
        }
        for record in state["episodes"]
    ]
    tables = {
        "u2_speeds.csv": table(
            ["episode_id", "sensor_id", "step", "speed_kph", "observed"], speed_rows
        ),
        "u2_sensors.csv": table(
            [
                "sensor_id",
                "position_km",
                "downstream_neighbour",
                "upstream_neighbour",
                "vendor",
                "free_flow_kph",
            ],
            sensor_rows,
        ),
        "u2_episodes.csv": table(
            ["episode_id", "weekday", "step_seconds", "incident_feed"], episode_rows
        ),
    }
    return envelope(
        SPEC,
        seed,
        resolved,
        GENERATOR,
        tables,
        question=(
            "For every episode, give the probability that the slowdown propagated upstream "
            "from an incident, name the detector it started at, and give the step at which "
            "it started. There is no incident feed to check against."
        ),
        submission={
            "keys": "one entry per episode_id",
            "fields": {
                "p_shockwave": "probability in [0, 1]",
                "origin_sensor": "a detector identifier such as det3",
                "onset_step": "step index from the start of the episode",
            },
            "example_key": "ep00",
        },
        metadata={
            "corridor": "Traffic runs from the highest detector index toward det0.",
            "clock": (
                "Detector timestamps come from two vendors that were commissioned "
                "separately. No synchronisation certificate is supplied with this pack."
            ),
            "prior": (
                "Episodes were drawn independently; the pack does not disclose the mix."
            ),
            "not_observed": [
                "incident reports, tow logs and lane closures",
                "the latent demand series, if there is one",
                "any detector's true clock error",
            ],
        },
        limitations=[
            "Speeds are floored, so the deepest slowdowns are compressed and lags shrink.",
            "Dropouts concentrate exactly in the congested window that carries the signal.",
            "Separating these two simulators is not evidence about real traffic physics.",
        ],
    )


def truth(seed=DEFAULT_SEED, **knobs):
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    state = _simulate(seed, resolved)
    facts = state["facts"]
    return sealed(
        SPEC,
        seed,
        resolved,
        GENERATOR,
        facts={
            "mechanism_by_episode": {k: v["mechanism"] for k, v in facts.items()},
            "origin_sensor_by_episode": {k: v["origin"] for k, v in facts.items()},
            "onset_step_by_episode": {k: v["onset"] for k, v in facts.items()},
            "wave_speed_by_episode": {k: v["wave"] for k, v in facts.items()},
            "vendor_clock_offset_steps": {"vendor_a": 0, "vendor_b": VENDOR_OFFSET_STEPS},
            "shared_loadings_by_episode": {k: v["loadings"] for k, v in facts.items()},
        },
        assumptions=[
            "Both worlds share one commute profile and identical observation noise.",
            "Propagation is a constant-speed upstream wave with exponential amplitude decay.",
            "The common driver has exactly zero lag between detectors.",
        ],
        support=(
            f"{resolved['episodes']} episodes x {N_SENSORS} detectors x "
            f"{resolved['steps']} five-minute steps on one corridor."
        ),
    )


def _series(payload):
    """Rebuild per-episode detector series from the agent-visible table only."""
    by_episode = {}
    for row in payload["tables"]["u2_speeds.csv"]["rows"]:
        if row["observed"] != "yes":
            continue
        key = (row["episode_id"], row["sensor_id"])
        by_episode.setdefault(row["episode_id"], {}).setdefault(row["sensor_id"], {})[
            row["step"]
        ] = row["speed_kph"]
        del key
    return by_episode


def _best_lag(a, b, steps, span=5):
    """Lag in [-span, span] maximising correlation of two gappy, de-meaned series."""
    best, best_lag = -2.0, 0
    for lag in range(-span, span + 1):
        pairs = [
            (a[t], b[t + lag])
            for t in range(steps)
            if t in a and (t + lag) in b
        ]
        if len(pairs) < 12:
            continue
        left = np.array([p[0] for p in pairs])
        right = np.array([p[1] for p in pairs])
        if left.std() < 1e-6 or right.std() < 1e-6:
            continue
        score = float(np.corrcoef(left, right)[0, 1])
        if score > best:
            best, best_lag = score, lag
    return best_lag


def baseline(seed=DEFAULT_SEED, **knobs):
    """Cross-correlation lag heuristic computed only from the agent-visible tables."""
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    payload = generate(seed, **resolved)
    steps = resolved["steps"]
    series = _series(payload)
    values = {}
    for episode, sensors in series.items():
        lags = []
        for s in range(N_SENSORS - 1):
            a, b = sensors.get(f"det{s}"), sensors.get(f"det{s + 1}")
            if a and b:
                lags.append(_best_lag(a, b, steps))
        mean_abs_lag = float(np.mean([abs(x) for x in lags])) if lags else 0.0
        p = 1.0 / (1.0 + math_exp(-(mean_abs_lag - 0.95) * 2.6))
        onsets = {}
        for name, points in sensors.items():
            values_sorted = sorted(points.items())
            level = float(np.median([v for _, v in values_sorted]))
            hit = [t for t, v in values_sorted if v < 0.86 * level]
            onsets[name] = float(hit[0]) if hit else float(steps)
        origin = min(onsets, key=lambda k: onsets[k]) if onsets else "det0"
        values[episode] = {
            "p_shockwave": round(float(np.clip(p, 0.03, 0.97)), 4),
            "origin_sensor": origin,
            "onset_step": round(float(min(onsets.values())) if onsets else 0.0, 4),
        }
    return predictions_of(
        "baseline", SPEC, values, "adjacent_lag_cross_correlation", SPEC.baseline_rationale
    )


def oracle(seed=DEFAULT_SEED, **knobs):
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    facts = truth(seed, **resolved)["facts"]
    mechanism = facts["mechanism_by_episode"]
    values = {}
    for episode, label in mechanism.items():
        shock = label == "shockwave"
        values[episode] = {
            "p_shockwave": ORACLE_CONFIDENCE if shock else 1 - ORACLE_CONFIDENCE,
            "origin_sensor": facts["origin_sensor_by_episode"][episode] or "det0",
            "onset_step": float(facts["onset_step_by_episode"][episode]),
        }
    return predictions_of("oracle", SPEC, values, "planted_truth_upper_bound", SPEC.oracle_rationale)


def trivial(seed=DEFAULT_SEED, **knobs):
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    facts = truth(seed, **resolved)["facts"]["mechanism_by_episode"]
    midpoint = float(resolved["steps"]) / 2.0
    values = {
        episode: {"p_shockwave": SHOCKWAVE_PRIOR, "origin_sensor": "det0", "onset_step": midpoint}
        for episode in facts
    }
    return predictions_of(
        "trivial", SPEC, values, "constant_prior", "Constant prior probability, fixed origin and mid-episode onset."
    )


def evaluate(predictions, seed=DEFAULT_SEED, **knobs):
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    facts = truth(seed, **resolved)["facts"]
    mechanism = facts["mechanism_by_episode"]
    actual = {k: 1 if v == "shockwave" else 0 for k, v in mechanism.items()}
    values = require_keys(unwrap(predictions), actual)
    probabilities, origins, onsets = {}, {}, {}
    for key, row in values.items():
        if not isinstance(row, dict) or set(row) != {"p_shockwave", "origin_sensor", "onset_step"}:
            raise Invalid("Each U2 entry needs exactly p_shockwave, origin_sensor and onset_step")
        probabilities[key] = probability(row["p_shockwave"], "shockwave probability")
        if not isinstance(row["origin_sensor"], str) or not row["origin_sensor"]:
            raise Invalid("origin_sensor must name a detector")
        origins[key] = row["origin_sensor"]
        onsets[key] = real(row["onset_step"], "onset step")
    losses = brier_losses(probabilities, actual)
    shock_keys = [k for k, v in actual.items() if v == 1]
    hits = sum(origins[k] == facts["origin_sensor_by_episode"][k] for k in shock_keys)
    onset_actual = {k: facts["onset_step_by_episode"][k] for k in shock_keys}
    bins, calibration = reliability(probabilities, actual)
    return scored(
        SPEC,
        seed,
        primary=float(np.mean(losses)),
        secondary={
            "origin_accuracy": float(hits / len(shock_keys)) if shock_keys else None,
            "onset_step_mae": float(
                np.mean(absolute_errors({k: onsets[k] for k in shock_keys}, onset_actual))
            )
            if shock_keys
            else None,
            "mechanism_log_loss": float(np.mean(log_losses(probabilities, actual))),
            "mechanism_calibration_error": calibration,
            "reliability_bins": bins,
            "shockwave_episodes": len(shock_keys),
        },
        losses=losses,
        units=len(actual),
        notes=[
            "Episodes are independent draws; detectors within an episode are not.",
            "Origin and onset are scored only where a propagating episode was planted.",
        ],
    )


def narrative():
    return story(
        SPEC,
        decision=(
            "A slowdown is spreading across a corridor at 17:20. Ramp metering upstream "
            "helps a propagating queue and does nothing for corridor-wide demand. Which is it?"
        ),
        agent_task=(
            "Decide, per episode, whether the onset lags between detectors scale with "
            "distance -- propagation -- or are a constant offset that does not. Both worlds "
            "share a commute dip, so any lagged-correlation test returns a positive number "
            "in both. The separating fact is whether lag is proportional to spacing or "
            "aligned with the vendor column."
        ),
        good_answer=(
            "Mechanism probabilities that beat the lag heuristic, an explicit finding that "
            "one vendor's clocks are late by a fixed three steps, and origin detectors named "
            "correctly on the propagating episodes with onset error of about a step."
        ),
        trap=(
            "The cross-correlation heuristic is real practice and it is confidently wrong on "
            "the common-driver episodes that contain vendor-b detectors: it reads a constant "
            "clock offset as a travelling queue."
        ),
        two_minutes=[
            "Plot two episodes side by side; both look like propagation.",
            "Run the baseline lag heuristic; note the confident, wrong calls.",
            "Show the sensors table and ask why lag does not scale with position_km.",
            "Reveal the sealed truth: one is a shockwave, one is a clock artefact.",
            "State the headroom between the heuristic and the oracle bound.",
            "Say the sentence: synthetic corridor, recovered mechanism, not a traffic finding.",
        ],
    )
