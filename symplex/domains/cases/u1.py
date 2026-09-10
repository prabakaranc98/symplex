"""U1 -- bike-station imbalance on a tidal flow network.

Generator: twelve docked stations split between a residential and a commercial zone.
Commuter demand is tidal (residential departures peak in the morning, commercial in the
evening), scaled by a weekday/weekend factor and by a weather multiplier with declared
coefficients. Bikes are a conserved stock: departures are censored by on-hand inventory,
arrivals are censored by free docks and spill to a neighbour, and an undisclosed truck
schedule moves bikes toward a target fill twice a day. A station-day "tips" when a rider
finds no bike or an arriving rider finds no dock.

The planted truth is which station-days tip and at what hour. The agent never sees
inventory, unserved demand or the truck schedule -- only completed flows, a stations
table, a weather feed on a different day-label convention, and a maintenance log that
stops before the held-out days.

Synthetic. Recovering the planted tips demonstrates inference mechanics, not that this
resembles any real bike-share system.
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

CASE_ID = "u1"
GENERATOR = "u1_tidal_flow_network_v1"
DEFAULT_SEED = 7
KNOBS = {"days": (12, 40, 24), "holdout_days": (2, 8, 4), "missing_rate": (0.0, 0.4, 0.08)}
HOURS = tuple(range(5, 23))
N_STATIONS = 12
REBALANCE_HOURS = (12, 16)
REBALANCE_TARGET = 0.5
TEMP_COEFFICIENT = 0.55
PRECIP_COEFFICIENT = 0.9
WEATHER_DAY_OFFSET = 1

SPEC = CaseSpec(
    id=CASE_ID,
    manifest_case="U1",
    domain="urban_infrastructure",
    title="Can network-aware demand estimates identify bike-station imbalances earlier?",
    decision=(
        "Which stations belong on tomorrow's inspection and rebalancing watchlist, and "
        "at roughly what hour each is expected to run out of bikes or docks."
    ),
    beneficiary="An operations planner allocating a fixed number of rebalancing visits.",
    mechanism_question=(
        "Is a station's imbalance risk a fixed property of the station, or does it move "
        "with an observable exogenous driver acting on tidal commute demand?"
    ),
    rivals=[
        "Each station's own calendar pattern is sufficient; imbalance risk is a station constant.",
        "Weather-driven demand scaling shifts which stations tip on which day.",
        "Apparent structure is an artefact of the weather feed's day-label convention.",
        "Apparent structure is an artefact of logging outages that coincide with saturation.",
    ],
    endpoints=[
        CaseEndpoint(
            id="tip_probability",
            name="Probability that a station-day tips",
            unit="probability",
            meaning="A rider finds no bike, or an arriving rider finds no free dock, at any hour.",
        ),
        CaseEndpoint(
            id="tip_hour",
            name="First tipping hour",
            unit="hour of local day",
            meaning="When the watchlist visit should be scheduled, on station-days that tip.",
        ),
        CaseEndpoint(
            id="net_flow",
            name="Daily net flow",
            unit="bikes per station-day",
            meaning="Completed arrivals minus completed departures over the local day.",
        ),
    ],
    evidence_channels=[
        "Hourly completed departures and arrivals per station, with logging gaps.",
        "Station table with dock capacity, zone and adjacency.",
        "Daily weather summaries from a second provider with its own day-label convention.",
        "A maintenance log of tipping events that stops at the end of the fitting window.",
    ],
    nuisance=[
        "Bike stock is conserved, so completed flows are censored by inventory and dock space.",
        "Unserved demand and spilled arrivals are never recorded.",
        "Rebalancing trucks act on a schedule that is not published in the pack.",
        "MCAR logging gaps plus MNAR gaps that are more likely while a station is saturated.",
        "Weather is reported in Celsius by one provider and Fahrenheit by the other.",
        "The weather feed labels each daily window by the UTC day on which it closes.",
    ],
    primary_metric=CaseMetric(
        id="tip_brier",
        name="Held-out station-day tip Brier score",
        lower_is_better=True,
        definition="Mean squared error of the tip probability over held-out station-days.",
    ),
    secondary_metrics=[
        CaseMetric(
            id="tip_hour_mae",
            name="Tipping-hour absolute error",
            lower_is_better=True,
            definition="Mean absolute hour error, restricted to station-days that actually tip.",
        ),
        CaseMetric(
            id="net_flow_mae",
            name="Daily net-flow absolute error",
            lower_is_better=True,
            definition="Mean absolute error of arrivals minus departures per held-out station-day.",
        ),
        CaseMetric(
            id="tip_calibration_error",
            name="Expected calibration error",
            lower_is_better=True,
            definition="Five-bin expected calibration error of the tip probabilities.",
        ),
    ],
    baseline_name="Per-station tip climatology with shrinkage",
    baseline_rationale=(
        "The operator's actual current practice: rank stations by how often the "
        "maintenance log recorded a tip, shrunk toward the fleet rate, and use each "
        "station's historical mean tipping hour and mean net flow. It uses every label "
        "the pack supplies and ignores only the exogenous driver."
    ),
    oracle_rationale=(
        "A reference upper bound given the planted tip labels, true expected net flow "
        "and true tipping hour. It is not reachable by inference; it marks the headroom."
    ),
    trivial_name="Constant 0.5 tip probability, fleet-mean hour and zero net flow",
    hidden_markers=[
        "tip_by_station_day",
        "first_tip_hour_by_station_day",
        "expected_net_flow_by_station_day",
        "unserved_departures_by_station_day",
        "weather_demand_coefficient",
        "weather_day_label_offset",
        "rebalance_schedule",
        "latent_inventory_summary",
    ],
    tables=["u1_flows.csv", "u1_stations.csv", "u1_weather.csv", "u1_tip_log.csv"],
    knobs=["days", "holdout_days", "missing_rate"],
    limits=(
        "Completed-trip data cannot prove a saved trip or a rebalancing benefit. This "
        "pack scores recovery of a planted stock-out process, nothing more."
    ),
)

NUMERIC_CHECKS = [
    NumericCheck(
        id="u1_flow_counts_nonnegative",
        operation="bounds",
        csv_filename="u1_flows.csv",
        columns=["departures", "arrivals"],
        group_columns=["station_id"],
        time_column=None,
        row_filters=[{"column": "logged", "equals": "yes"}],
        lower=0.0,
        upper=400.0,
        reference_value=None,
        tolerance=0.0,
        direction=None,
        units="completed bikes per station-hour",
        rationale="Censored counts are still counts; a negative or absurd flow is a generator fault.",
    ),
    NumericCheck(
        id="u1_capacity_finite",
        operation="finite",
        csv_filename="u1_stations.csv",
        columns=["dock_capacity"],
        group_columns=[],
        time_column=None,
        row_filters=[],
        lower=None,
        upper=None,
        reference_value=None,
        tolerance=0.0,
        direction=None,
        units="docks",
        rationale="Every station must declare a usable finite capacity for the stock constraint.",
    ),
]


def _setup(seed, knobs):
    days = knobs["days"]
    holdout = knobs["holdout_days"]
    if holdout >= days - 6:
        raise Invalid("U1 needs at least six fitting days before the held-out block")
    setup = stream(seed, "u1/setup")
    zones = np.array(["residential"] * (N_STATIONS // 2) + ["commercial"] * (N_STATIONS // 2))
    capacity = setup.integers(14, 31, size=N_STATIONS)
    popularity = setup.uniform(0.4, 2.7, size=N_STATIONS)
    neighbours = [
        sorted(setup.choice([j for j in range(N_STATIONS) if j != i], size=2, replace=False).tolist())
        for i in range(N_STATIONS)
    ]
    weather = stream(seed, "u1/weather")
    temp = 18.0 + 7.0 * np.sin(np.arange(days + 2) / 9.0) + weather.normal(0, 2.5, size=days + 2)
    precip = np.where(weather.random(days + 2) < 0.25, weather.exponential(4.0, size=days + 2), 0.0)
    multiplier = np.clip(
        np.exp(TEMP_COEFFICIENT * (temp - 18.0) / 10.0 - PRECIP_COEFFICIENT * precip / 5.0),
        0.45,
        1.9,
    )
    return {
        "days": days,
        "holdout": holdout,
        "zones": zones,
        "capacity": capacity,
        "popularity": popularity,
        "neighbours": neighbours,
        "temp": temp,
        "precip": precip,
        "multiplier": multiplier,
    }


def _profile(zone, hour, weekend):
    if weekend:
        return 0.55 + 0.45 * np.exp(-((hour - 14.0) ** 2) / 18.0)
    if zone == "residential":
        peak, off = 8.0, 18.0
    else:
        peak, off = 18.0, 8.0
    return 0.25 + 1.6 * np.exp(-((hour - peak) ** 2) / 2.6) + 0.35 * np.exp(-((hour - off) ** 2) / 6.0)


def _simulate(seed, knobs):
    """One deterministic realisation of the flow network, stock constraints included."""
    state = _setup(seed, knobs)
    days, zones, capacity = state["days"], state["zones"], state["capacity"]
    demand_rng = stream(seed, "u1/demand")
    route_rng = stream(seed, "u1/routing")
    inventory = (capacity * 0.5).astype(int)
    flows, tips, unserved, net_flow = [], {}, {}, {}
    for day in range(days):
        weekend = day % 7 in (5, 6)
        weather_index = day + WEATHER_DAY_OFFSET
        scale = state["multiplier"][weather_index] * (0.6 if weekend else 1.0)
        day_departures = np.zeros(N_STATIONS, dtype=int)
        day_arrivals = np.zeros(N_STATIONS, dtype=int)
        day_unserved = np.zeros(N_STATIONS, dtype=int)
        first_tip = {}
        for hour in HOURS:
            rates = np.array(
                [
                    state["popularity"][s] * _profile(zones[s], hour, weekend) * scale * 2.6
                    for s in range(N_STATIONS)
                ]
            )
            demand = demand_rng.poisson(rates)
            served = np.minimum(demand, inventory)
            day_unserved += demand - served
            raw_arrivals = np.zeros(N_STATIONS, dtype=int)
            for s in range(N_STATIONS):
                if served[s] == 0:
                    continue
                weights = np.array(
                    [
                        0.0
                        if t == s
                        else (2.5 if zones[t] != zones[s] else 0.7) * state["popularity"][t]
                        for t in range(N_STATIONS)
                    ]
                )
                weights = weights / weights.sum()
                raw_arrivals += route_rng.multinomial(int(served[s]), weights)
            inventory = inventory - served
            space = capacity - inventory
            accepted = np.minimum(raw_arrivals, space)
            overflow = raw_arrivals - accepted
            for s in np.flatnonzero(overflow):
                remaining = int(overflow[s])
                for neighbour in state["neighbours"][s]:
                    free = int(capacity[neighbour] - inventory[neighbour] - accepted[neighbour])
                    take = max(0, min(remaining, free))
                    accepted[neighbour] += take
                    remaining -= take
                    if remaining == 0:
                        break
            inventory = inventory + accepted
            for s in range(N_STATIONS):
                if s not in first_tip and (demand[s] > 0 and served[s] < demand[s] or overflow[s] > 0):
                    first_tip[s] = hour
            day_departures += served
            day_arrivals += accepted
            flows.append((day, hour, served.copy(), accepted.copy(), inventory.copy()))
            if hour in REBALANCE_HOURS:
                target = (capacity * REBALANCE_TARGET).astype(int)
                move = np.clip(target - inventory, -6, 6)
                inventory = np.clip(inventory + move, 0, capacity)
        for s in range(N_STATIONS):
            key = f"st{s:02d}_d{day:02d}"
            tips[key] = 1 if s in first_tip else 0
            unserved[key] = int(day_unserved[s])
            net_flow[key] = int(day_arrivals[s] - day_departures[s])
            if s in first_tip:
                state.setdefault("tip_hours", {})[key] = int(first_tip[s])
    state["flows"] = flows
    state["tips"] = tips
    state["unserved"] = unserved
    state["net_flow"] = net_flow
    state.setdefault("tip_hours", {})
    return state


def _keys(state):
    days, holdout = state["days"], state["holdout"]
    fit_days = list(range(days - holdout))
    test_days = list(range(days - holdout, days))
    fit = [f"st{s:02d}_d{d:02d}" for d in fit_days for s in range(N_STATIONS)]
    test = [f"st{s:02d}_d{d:02d}" for d in test_days for s in range(N_STATIONS)]
    return fit_days, test_days, fit, test


def generate(seed=DEFAULT_SEED, **knobs):
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    state = _simulate(seed, resolved)
    fit_days, test_days, fit_keys, _ = _keys(state)
    gap_rng = stream(seed, "u1/gaps")
    flow_rows = []
    for day, hour, served, accepted, inventory in state["flows"]:
        for s in range(N_STATIONS):
            saturated = inventory[s] == 0 or inventory[s] >= state["capacity"][s]
            drop = resolved["missing_rate"] + (0.22 if saturated else 0.0)
            logged = gap_rng.random() >= drop
            flow_rows.append(
                {
                    "day_index_local": day,
                    "hour_local": hour,
                    "station_id": f"st{s:02d}",
                    "departures": int(served[s]) if logged else None,
                    "arrivals": int(accepted[s]) if logged else None,
                    "logged": "yes" if logged else "no",
                }
            )
    station_rows = [
        {
            "station_id": f"st{s:02d}",
            "zone": str(state["zones"][s]),
            "dock_capacity": int(state["capacity"][s]),
            "capacity_unit": "docks",
            "neighbour_ids": ";".join(f"st{n:02d}" for n in state["neighbours"][s]),
        }
        for s in range(N_STATIONS)
    ]
    weather_rows = []
    for index in range(state["days"] + 2):
        fahrenheit = index % 3 == 0
        weather_rows.append(
            {
                "day_index_utc": index,
                "temperature": round(
                    float(state["temp"][index] * 9 / 5 + 32 if fahrenheit else state["temp"][index]), 2
                ),
                "temperature_unit": "F" if fahrenheit else "C",
                "precipitation_mm": round(float(state["precip"][index]), 2),
                "provider": "metro_fw" if fahrenheit else "metro_sc",
            }
        )
    log_rows = [
        {
            "day_index_local": int(key.split("_d")[1]),
            "station_id": key.split("_d")[0],
            "tipped": state["tips"][key],
            "logged_hour": state["tip_hours"].get(key),
        }
        for key in fit_keys
    ]
    tables = {
        "u1_flows.csv": table(
            ["day_index_local", "hour_local", "station_id", "departures", "arrivals", "logged"],
            flow_rows,
        ),
        "u1_stations.csv": table(
            ["station_id", "zone", "dock_capacity", "capacity_unit", "neighbour_ids"], station_rows
        ),
        "u1_weather.csv": table(
            ["day_index_utc", "temperature", "temperature_unit", "precipitation_mm", "provider"],
            weather_rows,
        ),
        "u1_tip_log.csv": table(
            ["day_index_local", "station_id", "tipped", "logged_hour"], log_rows
        ),
    }
    return envelope(
        SPEC,
        seed,
        resolved,
        GENERATOR,
        tables,
        question=(
            "For every station on each held-out day, give the probability that the station "
            "runs out of bikes or docks, the hour you would send the visit, and the day's "
            "net flow. The maintenance log covers the fitting days only."
        ),
        submission={
            "keys": "one entry per station and held-out day, formatted stNN_dDD",
            "fields": {
                "p_tip": "probability in [0, 1]",
                "tip_hour": f"hour in [{HOURS[0]}, {HOURS[-1]}]",
                "net_flow": "arrivals minus departures, bikes per day",
            },
            "example_key": f"st00_d{state['days'] - state['holdout']:02d}",
        },
        metadata={
            "fitting_days": fit_days,
            "held_out_days": test_days,
            "local_hours": list(HOURS),
            "clock": (
                "Flow and log rows use local time. The weather provider labels each daily "
                "window by the UTC day on which that window closes."
            ),
            "units": (
                "Temperature arrives in two units; read temperature_unit per row. "
                "Precipitation is millimetres for both providers."
            ),
            "not_observed": [
                "on-hand bikes and free docks at any hour",
                "riders who arrived and found nothing",
                "the truck movements that reset stations during the day",
            ],
        },
        limitations=[
            "Completed flows are censored by the stock constraint; absence of a trip is ambiguous.",
            "Logging gaps are not missing at random near saturation.",
            "No claim is made that these parameters resemble a real bike-share system.",
        ],
    )


def truth(seed=DEFAULT_SEED, **knobs):
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    state = _simulate(seed, resolved)
    _, test_days, _, test_keys = _keys(state)
    return sealed(
        SPEC,
        seed,
        resolved,
        GENERATOR,
        facts={
            "tip_by_station_day": {k: state["tips"][k] for k in test_keys},
            "first_tip_hour_by_station_day": {
                k: state["tip_hours"].get(k) for k in test_keys
            },
            "expected_net_flow_by_station_day": {k: state["net_flow"][k] for k in test_keys},
            "unserved_departures_by_station_day": {k: state["unserved"][k] for k in test_keys},
            "weather_demand_coefficient": {
                "temperature_per_10c": TEMP_COEFFICIENT,
                "precipitation_per_mm": PRECIP_COEFFICIENT / 5.0,
            },
            "weather_day_label_offset": WEATHER_DAY_OFFSET,
            "rebalance_schedule": {
                "hours": list(REBALANCE_HOURS),
                "target_fill_fraction": REBALANCE_TARGET,
                "max_bikes_per_visit": 6,
            },
            "latent_inventory_summary": {
                "held_out_days": test_days,
                "capacity_total": int(sum(int(c) for c in state["capacity"])),
            },
        },
        assumptions=[
            "Poisson arrivals per station-hour with a deterministic tidal profile.",
            "Arrivals spill only to two declared neighbours, then are lost.",
            "Rebalancing is a hard clip toward half capacity at two fixed hours.",
        ],
        support=(
            f"{N_STATIONS} stations x {resolved['days']} days x {len(HOURS)} local hours; "
            f"the last {resolved['holdout_days']} days are scored."
        ),
    )


def _observed_history(state):
    fit_days, _, fit_keys, _ = _keys(state)
    by_station = {}
    for key in fit_keys:
        station = key.split("_d")[0]
        by_station.setdefault(station, []).append(key)
    return by_station


def baseline(seed=DEFAULT_SEED, **knobs):
    """Per-station tip climatology, shrunk toward the fleet rate. No weather term."""
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    state = _simulate(seed, resolved)
    _, _, fit_keys, test_keys = _keys(state)
    fleet_rate = sum(state["tips"][k] for k in fit_keys) / len(fit_keys)
    fleet_hour = float(np.mean([state["tip_hours"][k] for k in fit_keys if k in state["tip_hours"]] or [14.0]))
    history = _observed_history(state)
    values = {}
    for key in test_keys:
        station = key.split("_d")[0]
        past = history[station]
        hits = sum(state["tips"][k] for k in past)
        rate = (hits + 4.0 * fleet_rate) / (len(past) + 4.0)
        hours = [state["tip_hours"][k] for k in past if k in state["tip_hours"]]
        flows = [state["net_flow"][k] for k in past]
        values[key] = {
            "p_tip": round(float(rate), 4),
            "tip_hour": round(float(np.mean(hours)) if hours else fleet_hour, 4),
            "net_flow": round(float(np.mean(flows)), 4),
        }
    return predictions_of(
        "baseline",
        SPEC,
        values,
        "per_station_tip_climatology_with_shrinkage",
        SPEC.baseline_rationale,
    )


def oracle(seed=DEFAULT_SEED, **knobs):
    """Reference upper bound: the planted labels, reported at declared confidence."""
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    state = _simulate(seed, resolved)
    _, _, fit_keys, test_keys = _keys(state)
    fleet_hour = float(np.mean([state["tip_hours"][k] for k in fit_keys if k in state["tip_hours"]] or [14.0]))
    values = {
        key: {
            "p_tip": ORACLE_CONFIDENCE if state["tips"][key] else 1 - ORACLE_CONFIDENCE,
            "tip_hour": float(state["tip_hours"].get(key, fleet_hour)),
            "net_flow": float(state["net_flow"][key]),
        }
        for key in test_keys
    }
    return predictions_of("oracle", SPEC, values, "planted_truth_upper_bound", SPEC.oracle_rationale)


def trivial(seed=DEFAULT_SEED, **knobs):
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    state = _simulate(seed, resolved)
    _, _, _, test_keys = _keys(state)
    midday = float(np.mean(HOURS))
    values = {k: {"p_tip": 0.5, "tip_hour": midday, "net_flow": 0.0} for k in test_keys}
    return predictions_of(
        "trivial", SPEC, values, "constant_predictor", "Constant tip probability and zero net flow."
    )


def evaluate(predictions, seed=DEFAULT_SEED, **knobs):
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    facts = truth(seed, **resolved)["facts"]
    actual_tip = facts["tip_by_station_day"]
    actual_hour = facts["first_tip_hour_by_station_day"]
    actual_flow = facts["expected_net_flow_by_station_day"]
    values = require_keys(unwrap(predictions), actual_tip)
    tip_pred, hour_pred, flow_pred = {}, {}, {}
    for key, row in values.items():
        if not isinstance(row, dict) or set(row) != {"p_tip", "tip_hour", "net_flow"}:
            raise Invalid("Each U1 entry needs exactly p_tip, tip_hour and net_flow")
        tip_pred[key] = probability(row["p_tip"], "tip probability")
        hour_pred[key] = real(row["tip_hour"], "tipping hour")
        flow_pred[key] = real(row["net_flow"], "net flow")
    losses = brier_losses(tip_pred, actual_tip)
    tipping = {k: v for k, v in actual_hour.items() if v is not None}
    hour_errors = absolute_errors({k: hour_pred[k] for k in tipping}, tipping) if tipping else [0.0]
    bins, calibration = reliability(tip_pred, actual_tip)
    return scored(
        SPEC,
        seed,
        primary=float(np.mean(losses)),
        secondary={
            "tip_hour_mae": float(np.mean(hour_errors)),
            "net_flow_mae": float(np.mean(absolute_errors(flow_pred, actual_flow))),
            "tip_calibration_error": calibration,
            "reliability_bins": bins,
            "held_out_tip_rate": float(np.mean(list(actual_tip.values()))),
            "tipping_station_days": len(tipping),
        },
        losses=losses,
        units=len(actual_tip),
        notes=[
            "Station-days within a day share the weather draw, so units are not fully independent.",
            "Tipping-hour error is defined only on station-days that actually tipped.",
        ],
    )


def narrative():
    return story(
        SPEC,
        decision=(
            "An operations planner has four rebalancing visits for tomorrow and twelve "
            "stations. Which four, and at what hour?"
        ),
        agent_task=(
            "Infer a latent stock process from censored flows. Completed departures stop "
            "when a station empties, so the strongest evidence for a stock-out is an "
            "absence, not a number. The pack also plants two decoys: logging gaps cluster "
            "exactly where saturation happens, and the weather feed's day label is offset "
            "from the flow feed's day."
        ),
        good_answer=(
            "A watchlist whose tip probabilities beat per-station climatology because the "
            "exogenous demand driver was correctly aligned and used; hours that land within "
            "an hour or two of the planted tipping hour; and an explicit statement that "
            "unserved demand was never observed."
        ),
        trap=(
            "Ranking stations by historical stock-out frequency is defensible and it is what "
            "the baseline does. It cannot move a station up or down for tomorrow's "
            "conditions, and it silently treats saturation-driven logging gaps as quiet hours."
        ),
        two_minutes=[
            "Show u1_flows.csv: hourly completed departures, with gaps.",
            "Point out the stations table: docks are a hard stock constraint.",
            "Show the weather table and its day-label footnote; this is the whole trick.",
            "Run the baseline: per-station climatology, Brier around 0.2.",
            "Show the sealed truth and the oracle bound, then state the headroom.",
            "Say the sentence: this is synthetic; it demonstrates recovery, not a real result.",
        ],
    )
