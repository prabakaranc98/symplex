"""C1 -- why some catchments recover slowly after drought.

Generator: twelve catchments driven by the same kind of seasonal rainfall and potential
evaporation, each holding a single storage state. Half drain as a linear reservoir,
Q proportional to S. The other half drain through a nonlinear storage-discharge relation,
Q proportional to S raised to a power above one, with a hysteresis term that makes the
recession steeper after a wet antecedent period. Under normal flows the two look almost
identical. They part company in the tail: a nonlinear store empties fast and then refills
slowly, so its recovery after a dry spell takes far longer than a fitted linear reservoir
predicts.

The record stops on the last day of the final dry spell. Rainfall and evaporation continue
past that day, so this is a conditional hydrological simulation with realised forcing, not
a weather-unknown forecast, and the pack says so.

Synthetic. Recovering the mechanism demonstrates inference mechanics, not hydrology.
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
    require_keys,
    scored,
    sealed,
    spearman,
    story,
    stream,
    table,
    unwrap,
)

CASE_ID = "c1"
GENERATOR = "c1_storage_discharge_hysteresis_v1"
DEFAULT_SEED = 7
KNOBS = {"catchments": (6, 20, 12), "days": (400, 900, 640), "missing_rate": (0.0, 0.4, 0.05)}
CUT_FRACTION = 0.8125
S_REFERENCE = 150.0
Q_REFERENCE = 2.1
RECOVERY_FRACTION = 0.7
RECOVERY_CAP = 110
FORCING_DAY_OFFSET = 1

SPEC = CaseSpec(
    id=CASE_ID,
    manifest_case="C1",
    domain="climate_agriculture",
    title="Why do some catchments recover slowly after drought?",
    decision=(
        "How long each catchment will stay below its low-flow trigger after the current "
        "dry spell breaks, and therefore which abstraction licences stay suspended."
    ),
    beneficiary="A water-resource analyst setting restriction end dates catchment by catchment.",
    mechanism_question=(
        "Does discharge fall out of storage linearly, or through a nonlinear store with "
        "hysteresis that turns a short rainfall deficit into a long recovery?"
    ),
    rivals=[
        "Rainfall deficit and seasonality explain recovery; storage structure is irrelevant.",
        "A single linear reservoir per catchment is sufficient.",
        "A nonlinear store with hysteresis produces the long memory.",
        "Gauge failure at high flow and unit mismatch explain the apparent difference.",
    ],
    endpoints=[
        CaseEndpoint(
            id="recovery_days",
            name="Days to recover to the low-flow trigger",
            unit="days after the end of the dry spell",
            meaning="Directly sets the restriction end date.",
        ),
        CaseEndpoint(
            id="storage_mechanism",
            name="Probability the store is nonlinear",
            unit="probability",
            meaning="Determines whether the linear planning model may be used at all.",
        ),
    ],
    evidence_channels=[
        "Daily discharge up to the last day of the current dry spell, with gauge outages.",
        "Daily rainfall and potential evaporation continuing past the end of the record.",
        "A catchment attribute table with area, elevation, reporting unit and conventions.",
    ],
    nuisance=[
        "Discharge is reported in millimetres per day by some gauges and cubic metres per second by others.",
        "Gauges fail preferentially at high flow, so the wettest days are missing.",
        "The forcing product labels each day by the UTC day on which its window closes.",
        "Measurement error is multiplicative, so low flows are noisiest in relative terms.",
    ],
    primary_metric=CaseMetric(
        id="recovery_time_mae",
        name="Recovery-time absolute error",
        lower_is_better=True,
        definition="Mean absolute error in days between predicted and realised recovery.",
    ),
    secondary_metrics=[
        CaseMetric(
            id="mechanism_brier",
            name="Storage-mechanism Brier score",
            lower_is_better=True,
            definition="Mean squared error of the probability that a catchment's store is nonlinear.",
        ),
        CaseMetric(
            id="recovery_time_bias",
            name="Recovery-time bias",
            lower_is_better=True,
            definition="Mean signed error in days; a negative value means restrictions lifted too early.",
        ),
        CaseMetric(
            id="within_five_days_rate",
            name="Fraction within five days",
            lower_is_better=False,
            definition="Share of catchments whose recovery date was called within five days.",
        ),
        CaseMetric(
            id="recovery_rank_correlation",
            name="Recovery-time rank correlation",
            lower_is_better=False,
            definition="Spearman correlation between predicted and realised recovery times.",
        ),
    ],
    baseline_name="Per-catchment linear reservoir fitted to observed recessions",
    baseline_rationale=(
        "The standard operational model. Fit one storage constant per catchment on the "
        "observed recession limbs, then route the realised forcing forward. It uses every "
        "catchment's own history and is wrong only in its functional form."
    ),
    oracle_rationale=(
        "The planted mechanism and its parameters, routed forward through the same realised "
        "forcing. Residual error is the multiplicative gauge noise alone."
    ),
    trivial_name="Midpoint recovery time and a coin-flip mechanism call",
    hidden_markers=[
        "storage_mechanism_by_catchment",
        "recovery_days_by_catchment",
        "storage_parameters_by_catchment",
        "hysteresis_strength_by_catchment",
        "forcing_day_label_offset",
        "baseline_flow_by_catchment",
    ],
    tables=["c1_discharge.csv", "c1_forcing.csv", "c1_attributes.csv"],
    knobs=["catchments", "days", "missing_rate"],
    limits=(
        "Realised forcing is supplied, so this is a conditional simulation and not an "
        "operational forecast. Nothing here establishes hydrology in a real basin."
    ),
)

NUMERIC_CHECKS = [
    NumericCheck(
        id="c1_discharge_nonnegative",
        operation="bounds",
        csv_filename="c1_discharge.csv",
        columns=["discharge"],
        group_columns=["catchment_id"],
        time_column=None,
        row_filters=[{"column": "observed", "equals": "yes"}],
        lower=0.0,
        upper=100000.0,
        reference_value=None,
        tolerance=0.0,
        direction=None,
        units="millimetres per day or cubic metres per second; see the attribute table",
        rationale="Storage cannot produce negative discharge under this generator's mass balance.",
    ),
    NumericCheck(
        id="c1_forcing_nonnegative",
        operation="bounds",
        csv_filename="c1_forcing.csv",
        columns=["precipitation_mm", "potential_et_mm"],
        group_columns=["catchment_id"],
        time_column=None,
        row_filters=[],
        lower=0.0,
        upper=400.0,
        reference_value=None,
        tolerance=0.0,
        direction=None,
        units="millimetres per day",
        rationale="Negative rainfall or evaporation would break the water balance the case rests on.",
    ),
]


def _forcing(seed, count, days):
    rng = stream(seed, "c1/forcing")
    grid = np.arange(days + FORCING_DAY_OFFSET + 1)
    season = 1.0 + 0.55 * np.sin(2 * np.pi * (grid - 40) / 365.0)
    precipitation = np.zeros((count, grid.size))
    evaporation = np.zeros((count, grid.size))
    for c in range(count):
        wet = rng.random(grid.size) < np.clip(0.34 * season, 0.05, 0.7)
        precipitation[c] = wet * rng.gamma(1.5, 5.0, size=grid.size)
        evaporation[c] = np.clip(0.75 + 0.45 * np.sin(2 * np.pi * (grid - 110) / 365.0) + rng.normal(0, 0.12, grid.size), 0.05, 4.0)
    return precipitation, evaporation


def _simulate(seed, knobs):
    count, days = knobs["catchments"], knobs["days"]
    cut = int(days * CUT_FRACTION)
    if cut < 200 or days - cut < RECOVERY_CAP + 10:
        raise Invalid("C1 needs a long fitting record and room for the recovery window")
    setup = stream(seed, "c1/setup")
    noise = stream(seed, "c1/gauge")
    precipitation, evaporation = _forcing(seed, count, days)
    catchments = []
    for c in range(count):
        nonlinear = bool(setup.random() < 0.5)
        exponent = float(setup.uniform(1.55, 2.25)) if nonlinear else 1.0
        hysteresis = float(setup.uniform(0.15, 0.45)) if nonlinear else 0.0
        constant = float(setup.uniform(0.75, 1.35))
        area = float(setup.uniform(90.0, 850.0))
        elevation = float(setup.uniform(120.0, 1600.0))
        dry_start = cut - int(setup.integers(18, 33))
        earlier = sorted(int(x) for x in setup.integers(120, cut - 120, size=2))
        catchments.append(
            {
                "id": f"cat{c:02d}",
                "nonlinear": nonlinear,
                "exponent": exponent,
                "hysteresis": hysteresis,
                "constant": constant,
                "area": area,
                "elevation": elevation,
                "dry_start": dry_start,
                "earlier": earlier,
                "unit": "m3_s" if c % 2 else "mm_day",
            }
        )
    total = precipitation.shape[1]
    for c, info in enumerate(catchments):
        forcing = precipitation[c].copy()
        forcing[info["dry_start"] : cut + 1] = 0.0
        for start in info["earlier"]:
            forcing[start : start + 22] = 0.0
        storage = S_REFERENCE
        memory = 0.0
        flow = np.zeros(total)
        for t in range(total):
            rain = forcing[t + FORCING_DAY_OFFSET] if t + FORCING_DAY_OFFSET < total else 0.0
            memory = 0.94 * memory + 0.06 * (rain - 2.4)
            shape = (max(storage, 0.0) / S_REFERENCE) ** info["exponent"]
            gain = 1.0 + info["hysteresis"] * float(np.clip(memory / 2.5, -0.8, 1.6))
            discharge = Q_REFERENCE * info["constant"] * shape * gain
            discharge = float(min(discharge, max(storage, 0.0)))
            storage = max(0.0, storage + rain - evaporation[c][t] - discharge)
            flow[t] = discharge
            if t == cut:
                info["storage_at_cut"] = storage
                info["memory_at_cut"] = memory
        info["true_flow"] = flow
        info["forcing"] = forcing
        info["observed_flow"] = flow * np.exp(noise.normal(0, 0.09, size=total))
        window = info["observed_flow"][info["dry_start"] - 30 : info["dry_start"]]
        info["reference_flow"] = float(np.mean(window))
        target = RECOVERY_FRACTION * info["reference_flow"]
        recovered = RECOVERY_CAP
        for offset in range(1, RECOVERY_CAP + 1):
            if cut + offset < total and info["observed_flow"][cut + offset] >= target:
                recovered = offset
                break
        info["recovery"] = int(recovered)
    return {"catchments": catchments, "cut": cut, "days": days, "evaporation": evaporation, "total": total}


def _to_unit(value, info):
    if info["unit"] == "mm_day":
        return value
    return value * info["area"] * 1000.0 / 86400.0


def generate(seed=DEFAULT_SEED, **knobs):
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    state = _simulate(seed, resolved)
    gaps = stream(seed, "c1/outages")
    cut = state["cut"]
    discharge_rows, forcing_rows, attribute_rows = [], [], []
    for c, info in enumerate(state["catchments"]):
        high = float(np.quantile(info["observed_flow"][:cut], 0.9))
        for t in range(cut + 1):
            value = float(info["observed_flow"][t])
            drop = resolved["missing_rate"] + (0.25 if value > high else 0.0)
            observed = gaps.random() >= drop
            discharge_rows.append(
                {
                    "catchment_id": info["id"],
                    "day_index_local": t,
                    "discharge": round(_to_unit(value, info), 4) if observed else None,
                    "observed": "yes" if observed else "no",
                }
            )
        for t in range(state["total"]):
            forcing_rows.append(
                {
                    "catchment_id": info["id"],
                    "day_index_utc": t,
                    "precipitation_mm": round(float(info["forcing"][t]), 3),
                    "potential_et_mm": round(float(state["evaporation"][c][t]), 3),
                }
            )
        attribute_rows.append(
            {
                "catchment_id": info["id"],
                "area_km2": round(info["area"], 2),
                "mean_elevation_m": round(info["elevation"], 1),
                "discharge_unit": info["unit"],
                "gauge_operator": "agency_a" if c % 2 else "agency_b",
                "record_ends_day": cut,
            }
        )
    tables = {
        "c1_discharge.csv": table(
            ["catchment_id", "day_index_local", "discharge", "observed"], discharge_rows
        ),
        "c1_forcing.csv": table(
            ["catchment_id", "day_index_utc", "precipitation_mm", "potential_et_mm"], forcing_rows
        ),
        "c1_attributes.csv": table(
            [
                "catchment_id",
                "area_km2",
                "mean_elevation_m",
                "discharge_unit",
                "gauge_operator",
                "record_ends_day",
            ],
            attribute_rows,
        ),
    }
    return envelope(
        SPEC,
        seed,
        resolved,
        GENERATOR,
        tables,
        question=(
            "The discharge record stops on the last day of the current dry spell. For every "
            "catchment, say how many days after that day the gauge first reads at least "
            "seventy per cent of its pre-drought mean, and give the probability that its "
            "store is nonlinear rather than a linear reservoir."
        ),
        submission={
            "keys": "one entry per catchment_id",
            "fields": {
                "recovery_days": f"days after the record end, in [1, {RECOVERY_CAP}]",
                "p_nonlinear": "probability in [0, 1]",
            },
            "example_key": "cat00",
        },
        metadata={
            "record_ends_day": cut,
            "recovery_definition": (
                f"First day after the record end whose reported discharge reaches "
                f"{RECOVERY_FRACTION:g} times the mean of the thirty reported days before "
                f"the dry spell began. Censored at {RECOVERY_CAP} days."
            ),
            "forcing_available_after_record": True,
            "clock": (
                "Discharge days are local water days. The forcing product labels each day "
                "by the UTC day on which its accumulation window closes."
            ),
            "units": "Read discharge_unit per catchment before comparing any two gauges.",
            "not_observed": [
                "catchment storage at any moment",
                "the storage-discharge relation or its exponent",
                "discharge after the record end",
            ],
        },
        limitations=[
            "Realised rainfall is supplied, so this is a conditional simulation.",
            "Outages remove the high flows that most constrain the storage relation.",
            "Nothing here establishes hydrological behaviour in a real catchment.",
        ],
    )


def truth(seed=DEFAULT_SEED, **knobs):
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    state = _simulate(seed, resolved)
    catchments = state["catchments"]
    return sealed(
        SPEC,
        seed,
        resolved,
        GENERATOR,
        facts={
            "storage_mechanism_by_catchment": {
                i["id"]: "nonlinear_storage" if i["nonlinear"] else "linear_reservoir"
                for i in catchments
            },
            "recovery_days_by_catchment": {i["id"]: i["recovery"] for i in catchments},
            "storage_parameters_by_catchment": {
                i["id"]: {
                    "exponent": round(i["exponent"], 6),
                    "constant": round(i["constant"], 6),
                    "reference_storage_mm": S_REFERENCE,
                }
                for i in catchments
            },
            "hysteresis_strength_by_catchment": {
                i["id"]: round(i["hysteresis"], 6) for i in catchments
            },
            "forcing_day_label_offset": FORCING_DAY_OFFSET,
            "baseline_flow_by_catchment": {
                i["id"]: round(i["reference_flow"], 6) for i in catchments
            },
        },
        assumptions=[
            "One lumped storage state per catchment with a daily explicit water balance.",
            "Hysteresis acts through an exponential memory of recent rainfall.",
            "Gauge error is lognormal and independent across days.",
        ],
        support=(
            f"{resolved['catchments']} catchments over {resolved['days']} days; the record "
            f"ends at day {state['cut']} and recovery is censored at {RECOVERY_CAP} days."
        ),
    )


def _fit_linear(info, cut):
    """Storage constant from observed recession limbs, in the catchment's own units."""
    flow = info["observed_flow"][: cut + 1]
    ratios = []
    for t in range(1, cut):
        if info["forcing"][t + FORCING_DAY_OFFSET] < 0.5 and flow[t - 1] > 1e-6 and flow[t] > 1e-6:
            ratios.append(flow[t] / flow[t - 1])
    if not ratios:
        return 30.0
    decay = float(np.clip(np.median(ratios), 0.75, 0.999))
    return float(-1.0 / np.log(decay))


def _route(info, state, storage_constant, exponent, hysteresis, constant, start_storage, memory=0.0):
    cut, total = state["cut"], state["total"]
    evaporation = state["evaporation"][int(info["id"][3:])]
    storage = start_storage
    target = RECOVERY_FRACTION * info["reference_flow"]
    for t in range(cut + 1, total):
        rain = info["forcing"][t + FORCING_DAY_OFFSET] if t + FORCING_DAY_OFFSET < total else 0.0
        memory = 0.94 * memory + 0.06 * (rain - 2.4)
        shape = (max(storage, 0.0) / S_REFERENCE) ** exponent
        gain = 1.0 + hysteresis * float(np.clip(memory / 2.5, -0.8, 1.6))
        discharge = Q_REFERENCE * constant * shape * gain if storage_constant is None else max(storage, 0.0) / storage_constant
        discharge = float(min(discharge, max(storage, 0.0)))
        storage = max(0.0, storage + rain - evaporation[t] - discharge)
        if discharge >= target:
            return min(RECOVERY_CAP, t - cut)
    return RECOVERY_CAP


def baseline(seed=DEFAULT_SEED, **knobs):
    """Linear reservoir per catchment, fitted on recessions and routed forward."""
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    state = _simulate(seed, resolved)
    cut = state["cut"]
    values = {}
    for info in state["catchments"]:
        constant = _fit_linear(info, cut)
        recent = float(np.median(info["observed_flow"][cut - 4 : cut + 1]))
        start = recent * constant
        days = _route(info, state, constant, 1.0, 0.0, 1.0, start)
        values[info["id"]] = {"recovery_days": float(days), "p_nonlinear": 0.5}
    return predictions_of(
        "baseline", SPEC, values, "fitted_linear_reservoir_routing", SPEC.baseline_rationale
    )


def oracle(seed=DEFAULT_SEED, **knobs):
    """Planted mechanism and parameters, routed forward through the same realised forcing."""
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    state = _simulate(seed, resolved)
    cut = state["cut"]
    values = {}
    for info in state["catchments"]:
        days = _route(
            info,
            state,
            None,
            info["exponent"],
            info["hysteresis"],
            info["constant"],
            info["storage_at_cut"],
            info["memory_at_cut"],
        )
        values[info["id"]] = {
            "recovery_days": float(days),
            "p_nonlinear": ORACLE_CONFIDENCE if info["nonlinear"] else 1 - ORACLE_CONFIDENCE,
        }
    return predictions_of("oracle", SPEC, values, "planted_storage_relation", SPEC.oracle_rationale)


def trivial(seed=DEFAULT_SEED, **knobs):
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    state = _simulate(seed, resolved)
    midpoint = float(RECOVERY_CAP) / 2.0
    values = {
        i["id"]: {"recovery_days": midpoint, "p_nonlinear": 0.5} for i in state["catchments"]
    }
    return predictions_of(
        "trivial",
        SPEC,
        values,
        "uninformative_midpoint",
        "The midpoint of the allowed recovery window for every catchment.",
    )


def evaluate(predictions, seed=DEFAULT_SEED, **knobs):
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    facts = truth(seed, **resolved)["facts"]
    actual = facts["recovery_days_by_catchment"]
    mechanism = {
        k: 1 if v == "nonlinear_storage" else 0
        for k, v in facts["storage_mechanism_by_catchment"].items()
    }
    values = require_keys(unwrap(predictions), actual)
    days, calls = {}, {}
    for key, row in values.items():
        if not isinstance(row, dict) or set(row) != {"recovery_days", "p_nonlinear"}:
            raise Invalid("Each C1 entry needs exactly recovery_days and p_nonlinear")
        days[key] = real(row["recovery_days"], "recovery days")
        calls[key] = probability(row["p_nonlinear"], "nonlinear probability")
    losses = absolute_errors(days, actual)
    signed = [days[k] - actual[k] for k in sorted(actual)]
    return scored(
        SPEC,
        seed,
        primary=float(np.mean(losses)),
        secondary={
            "mechanism_brier": float(np.mean(brier_losses(calls, mechanism))),
            "recovery_time_bias": float(np.mean(signed)),
            "within_five_days_rate": float(np.mean([e <= 5 for e in losses])),
            "recovery_rank_correlation": spearman(days, actual),
            "nonlinear_catchments": int(sum(mechanism.values())),
            "censored_at_cap": int(sum(v >= RECOVERY_CAP for v in actual.values())),
        },
        losses=losses,
        units=len(actual),
        notes=[
            "Catchments share one forcing generator, so errors are not fully independent.",
            f"Recovery is censored at {RECOVERY_CAP} days; a censored catchment caps the error.",
        ],
    )


def narrative():
    return story(
        SPEC,
        decision=(
            "Abstraction licences are suspended in twelve catchments. The rain has just "
            "returned. When does each restriction lift?"
        ),
        agent_task=(
            "Decide whether each catchment's recession is a straight line in log flow or a "
            "curve. Under normal conditions the two storage relations are almost the same "
            "curve, and the gauge outages remove exactly the high flows that separate them. "
            "The nonlinear catchments are the ones whose restrictions must stay on much "
            "longer than a fitted linear model says."
        ),
        good_answer=(
            "Recovery dates within a few days, nonlinear catchments identified, and an "
            "explicit statement that realised rainfall was supplied, so this is a "
            "conditional simulation rather than an operational forecast."
        ),
        trap=(
            "A fitted linear reservoir matches the observed record well and then lifts "
            "restrictions early on exactly the catchments that recover slowest. Good fit "
            "in the middle of the distribution, wrong answer in the tail."
        ),
        two_minutes=[
            "Plot two recession limbs on a log axis; they look the same.",
            "Show the gauge-outage column: the separating high flows are missing.",
            "Run the fitted linear reservoir; note where it lifts restrictions early.",
            "Reveal which catchments were planted nonlinear and their exponents.",
            "Show the oracle routing the same rainfall through the true relation.",
            "Say the sentence: synthetic catchments, recovered mechanism, not hydrology.",
        ],
    )
