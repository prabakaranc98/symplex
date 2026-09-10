"""C2 -- when heat and moisture stress combine into regional crop losses.

Generator: forty counties in three regions, twenty-four seasons. Yield carries a county
level, a technology trend, and a weather response. In one region the weather response is
strictly additive: heat costs bushels, moisture deficit costs bushels, and that is all. In
the other two regions a planted interaction term makes hot-and-dry seasons cost more than
the sum of hot and dry, which is what turns a bad season into a correlated regional loss.

The stress that matters is not the whole season. It is a two-month window, and the weather
product labels each accumulation window by the month in which it closes, so the window is
one row off from where it looks. Yields arrive in two units, and the statistical agency
suppresses county-years with the largest losses, which is missingness that hides exactly
the observations the interaction term is estimated from.

Synthetic. Recovering the interaction demonstrates inference mechanics, not agronomy.
"""

import numpy as np
from scipy import stats

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
    predictions_of,
    probability,
    real,
    require_keys,
    scored,
    sealed,
    squared_errors,
    story,
    stream,
    table,
    unwrap,
)

CASE_ID = "c2"
GENERATOR = "c2_compound_heat_moisture_stress_v1"
DEFAULT_SEED = 7
KNOBS = {"counties": (12, 60, 40), "years": (16, 30, 24), "holdout_years": (3, 8, 5)}
MONTHS = 6
STRESS_MONTHS = (3, 4)
MONTH_LABEL_OFFSET = 1
HEAT_COEFFICIENT = -0.62
DEFICIT_COEFFICIENT = -0.21
REGION_INTERACTION = {"north": 0.0, "central": -0.0075, "south": -0.0125}
TREND_PER_YEAR = 1.45
SEVERE_LOSS_DROP = 25.0
BU_ACRE_TO_KG_HA = 62.77

SPEC = CaseSpec(
    id=CASE_ID,
    manifest_case="C2",
    domain="climate_agriculture",
    title="When do heat and moisture stress combine into regional crop losses?",
    decision=(
        "Which regions to reserve capital against for the coming seasons, and how much of "
        "the exposure is correlated across counties rather than diversifiable."
    ),
    beneficiary="A regional crop-risk desk sizing a reserve before the season opens.",
    mechanism_question=(
        "Is the weather response additive, or does a compound hot-and-dry window carry an "
        "interaction term that additive models cannot express?"
    ),
    rivals=[
        "An additive weather response plus county trends is sufficient.",
        "A heat and moisture interaction adds nonlinear loss in compound seasons.",
        "Technology trend and unobserved management explain the apparent effect.",
        "Correlated regional losses reflect shared weather, not county-to-county spread.",
    ],
    endpoints=[
        CaseEndpoint(
            id="county_yield",
            name="County yield in the held-out seasons",
            unit="bushels per acre",
            meaning="The quantity the reserve is sized against.",
        ),
        CaseEndpoint(
            id="severe_loss",
            name="Probability of a severe county loss",
            unit="probability",
            meaning="A loss of at least twenty-five bushels per acre below the county trend.",
        ),
        CaseEndpoint(
            id="interaction",
            name="Regional interaction coefficient",
            unit="bushels per acre per heat-day per millimetre of deficit",
            meaning="Whether compound stress is priced at all.",
        ),
    ],
    evidence_channels=[
        "County-season yields in mixed units, with agency suppression.",
        "Monthly heat and precipitation by county, labelled by window close.",
        "A county table giving region, reporting unit and planted area.",
    ],
    nuisance=[
        "Yields are reported in bushels per acre by some counties and kilograms per hectare by others.",
        "The agency suppresses county-seasons with the largest losses and the smallest areas.",
        "Weather rows are labelled by the month in which the accumulation window closes.",
        "Counties in a region share a season shock, so county-seasons are not independent.",
    ],
    primary_metric=CaseMetric(
        id="yield_rmse",
        name="Held-out county-season yield RMSE",
        lower_is_better=True,
        definition="Root mean squared error in bushels per acre over held-out county-seasons.",
    ),
    secondary_metrics=[
        CaseMetric(
            id="severe_loss_brier",
            name="Severe-loss Brier score",
            lower_is_better=True,
            definition="Mean squared error of the probability of a severe county loss.",
        ),
        CaseMetric(
            id="interaction_recovery_error",
            name="Interaction recovery error",
            lower_is_better=True,
            definition="Mean absolute error of the submitted regional interaction coefficients.",
        ),
        CaseMetric(
            id="compound_season_rmse",
            name="RMSE in compound-stress county-seasons",
            lower_is_better=True,
            definition="Held-out RMSE restricted to county-seasons in the hottest and driest quartile.",
        ),
    ],
    baseline_name="County fixed effects with a trend and an additive season-total weather response",
    baseline_rationale=(
        "The standard county-yield regression: one level per county, a linear technology "
        "trend, and season-total heat and deficit entering additively. It is the rival "
        "hypothesis stated in the case, fitted honestly on the training seasons only."
    ),
    oracle_rationale=(
        "The planted response evaluated on the true stress window with the true "
        "coefficients, noise free. The gap is exactly the compound-stress term."
    ),
    trivial_name="Training-mean yield everywhere",
    hidden_markers=[
        "yield_by_county_season",
        "severe_loss_by_county_season",
        "interaction_by_region_true",
        "stress_window_months_true",
        "weather_response_coefficients",
        "month_label_offset",
    ],
    tables=["c2_yields.csv", "c2_weather.csv", "c2_counties.csv"],
    knobs=["counties", "years", "holdout_years"],
    limits=(
        "County estimates and gridded weather cannot identify a field-level causal effect. "
        "This pack scores recovery of a planted interaction, nothing more."
    ),
)

NUMERIC_CHECKS = [
    NumericCheck(
        id="c2_yields_positive",
        operation="bounds",
        csv_filename="c2_yields.csv",
        columns=["reported_yield"],
        group_columns=["county_id"],
        time_column=None,
        row_filters=[{"column": "suppressed", "equals": "no"}],
        lower=0.0,
        upper=40000.0,
        reference_value=None,
        tolerance=0.0,
        direction=None,
        units="bushels per acre or kilograms per hectare; see the county table",
        rationale="A reported yield must be positive in whichever unit the county files.",
    ),
    NumericCheck(
        id="c2_weather_finite",
        operation="finite",
        csv_filename="c2_weather.csv",
        columns=["heat_days", "precipitation_mm"],
        group_columns=["county_id"],
        time_column=None,
        row_filters=[],
        lower=None,
        upper=None,
        reference_value=None,
        tolerance=0.0,
        direction=None,
        units="days above threshold and millimetres",
        rationale="The interaction is a product of these two columns, so neither may be absent.",
    ),
]

REGIONS = ("north", "central", "south")


def _build(seed, knobs):
    counties, years = knobs["counties"], knobs["years"]
    holdout = knobs["holdout_years"]
    if years - holdout < 10:
        raise Invalid("C2 needs at least ten training seasons")
    setup = stream(seed, "c2/setup")
    weather = stream(seed, "c2/weather")
    noise = stream(seed, "c2/noise")
    region = [REGIONS[i % len(REGIONS)] for i in range(counties)]
    base = setup.uniform(128.0, 178.0, size=counties)
    trend = TREND_PER_YEAR + setup.normal(0, 0.18, size=counties)
    area = setup.uniform(3000.0, 90000.0, size=counties)
    unit = ["kg_ha" if i % 3 == 0 else "bu_acre" for i in range(counties)]
    heat = np.zeros((counties, years, MONTHS))
    precipitation = np.zeros((counties, years, MONTHS))
    for y in range(years):
        for r, name in enumerate(REGIONS):
            shock_heat = weather.normal(0, 3.6)
            shock_wet = weather.normal(0, 26.0)
            for c in range(counties):
                if region[c] != name:
                    continue
                for m in range(MONTHS):
                    seasonal = 4.0 + 7.0 * np.exp(-((m - 3.4) ** 2) / 2.4)
                    heat[c, y, m] = max(0.0, seasonal + shock_heat + weather.normal(0, 1.9))
                    precipitation[c, y, m] = max(
                        0.0, 78.0 - 9.0 * abs(m - 2.5) + shock_wet + weather.normal(0, 14.0)
                    )
    stress_heat = heat[:, :, STRESS_MONTHS[0]] + heat[:, :, STRESS_MONTHS[1]]
    stress_wet = precipitation[:, :, STRESS_MONTHS[0]] + precipitation[:, :, STRESS_MONTHS[1]]
    deficit = np.clip(150.0 - stress_wet, 0.0, None)
    gamma = np.array([REGION_INTERACTION[region[c]] for c in range(counties)])
    grid = np.arange(years)
    expected = (
        base[:, None]
        + trend[:, None] * grid[None, :]
        + HEAT_COEFFICIENT * stress_heat
        + DEFICIT_COEFFICIENT * deficit
        + gamma[:, None] * stress_heat * deficit
    )
    observed = expected + noise.normal(0, 5.5, size=(counties, years))
    reference = base[:, None] + trend[:, None] * grid[None, :]
    severe = (observed < reference - SEVERE_LOSS_DROP).astype(int)
    return {
        "counties": counties,
        "years": years,
        "holdout": holdout,
        "region": region,
        "base": base,
        "trend": trend,
        "area": area,
        "unit": unit,
        "heat": heat,
        "precipitation": precipitation,
        "stress_heat": stress_heat,
        "deficit": deficit,
        "expected": expected,
        "observed": observed,
        "reference": reference,
        "severe": severe,
        "gamma": gamma,
    }


def _keys(state):
    train_years = list(range(state["years"] - state["holdout"]))
    test_years = list(range(state["years"] - state["holdout"], state["years"]))
    test = [f"co{c:02d}_y{y:02d}" for y in test_years for c in range(state["counties"])]
    return train_years, test_years, test


def generate(seed=DEFAULT_SEED, **knobs):
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    state = _build(seed, resolved)
    train_years, test_years, _ = _keys(state)
    suppress = stream(seed, "c2/suppression")
    yield_rows, weather_rows, county_rows = [], [], []
    losses = state["observed"] - state["reference"]
    for c in range(state["counties"]):
        for y in train_years:
            hidden = suppress.random() < (
                0.04 + (0.35 if losses[c, y] < -SEVERE_LOSS_DROP else 0.0) + (0.2 if state["area"][c] < 9000 else 0.0)
            )
            value = float(state["observed"][c, y])
            reported = value * BU_ACRE_TO_KG_HA if state["unit"][c] == "kg_ha" else value
            yield_rows.append(
                {
                    "county_id": f"co{c:02d}",
                    "season": y,
                    "reported_yield": round(reported, 3) if not hidden else None,
                    "suppressed": "yes" if hidden else "no",
                }
            )
        for y in range(state["years"]):
            for m in range(MONTHS):
                weather_rows.append(
                    {
                        "county_id": f"co{c:02d}",
                        "season": y,
                        "window_close_month": m + MONTH_LABEL_OFFSET,
                        "heat_days": round(float(state["heat"][c, y, m]), 3),
                        "precipitation_mm": round(float(state["precipitation"][c, y, m]), 3),
                    }
                )
        county_rows.append(
            {
                "county_id": f"co{c:02d}",
                "region": state["region"][c],
                "planted_area_acres": round(float(state["area"][c]), 1),
                "yield_unit": state["unit"][c],
                "reporting_agency": "region_office_" + state["region"][c],
            }
        )
    tables = {
        "c2_yields.csv": table(
            ["county_id", "season", "reported_yield", "suppressed"], yield_rows
        ),
        "c2_weather.csv": table(
            ["county_id", "season", "window_close_month", "heat_days", "precipitation_mm"],
            weather_rows,
        ),
        "c2_counties.csv": table(
            ["county_id", "region", "planted_area_acres", "yield_unit", "reporting_agency"],
            county_rows,
        ),
    }
    return envelope(
        SPEC,
        seed,
        resolved,
        GENERATOR,
        tables,
        question=(
            "Predict every county's yield in bushels per acre for each held-out season, and "
            "the probability that it falls at least twenty-five bushels below its own trend. "
            "You may also submit one interaction coefficient per region."
        ),
        submission={
            "keys": "one entry per county and held-out season, formatted coNN_yYY",
            "fields": {
                "yield_bu_acre": "yield in bushels per acre",
                "p_severe_loss": "probability in [0, 1]",
            },
            "optional_key": (
                "interaction_by_region maps each region to its heat-by-deficit coefficient"
            ),
            "example_key": f"co00_y{test_years[0]:02d}",
        },
        metadata={
            "training_seasons": train_years,
            "held_out_seasons": test_years,
            "months_per_season": MONTHS,
            "severe_loss_definition": (
                f"Yield below the county's own trend line by at least {SEVERE_LOSS_DROP:g} "
                "bushels per acre."
            ),
            "clock": (
                "Each weather row is labelled by the month in which its accumulation window "
                "closes, not the month it covers."
            ),
            "units": (
                "Read yield_unit per county. One bushel per acre of corn grain is "
                f"{BU_ACRE_TO_KG_HA} kilograms per hectare."
            ),
            "not_observed": [
                "planting dates, cultivar choice and irrigation",
                "the stress window the response actually uses",
                "yields in suppressed county-seasons and in held-out seasons",
            ],
        },
        limitations=[
            "Suppression removes the largest losses, which is where the nonlinearity lives.",
            "Weather for the held-out seasons is supplied, so this is an end-of-season hindcast.",
            "County estimates cannot identify a field-level causal effect.",
        ],
    )


def truth(seed=DEFAULT_SEED, **knobs):
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    state = _build(seed, resolved)
    _, _, test_keys = _keys(state)
    values, severe = {}, {}
    for key in test_keys:
        c = int(key[2:4])
        y = int(key.split("_y")[1])
        values[key] = round(float(state["observed"][c, y]), 4)
        severe[key] = int(state["severe"][c, y])
    return sealed(
        SPEC,
        seed,
        resolved,
        GENERATOR,
        facts={
            "yield_by_county_season": values,
            "severe_loss_by_county_season": severe,
            "interaction_by_region_true": dict(REGION_INTERACTION),
            "stress_window_months_true": list(STRESS_MONTHS),
            "weather_response_coefficients": {
                "heat_days": HEAT_COEFFICIENT,
                "moisture_deficit_mm": DEFICIT_COEFFICIENT,
                "deficit_reference_mm": 150.0,
                "trend_per_season": TREND_PER_YEAR,
            },
            "month_label_offset": MONTH_LABEL_OFFSET,
        },
        assumptions=[
            "One lumped stress window per season with an additive plus product response.",
            "Region-level season shocks make counties in a region dependent.",
            "Observation noise is homoscedastic Gaussian in bushels per acre.",
        ],
        support=(
            f"{resolved['counties']} counties x {resolved['years']} seasons; the last "
            f"{resolved['holdout_years']} seasons are scored."
        ),
    )


def _design(state, county_index, years, season_heat, season_deficit):
    rows = []
    for c, y in zip(county_index, years):
        dummies = np.zeros(state["counties"])
        dummies[c] = 1.0
        rows.append(np.concatenate([dummies, [y, season_heat[c, y], season_deficit[c, y]]]))
    return np.array(rows)


def _season_totals(state):
    heat = state["heat"].sum(axis=2)
    deficit = np.clip(450.0 - state["precipitation"].sum(axis=2), 0.0, None)
    return heat, deficit


def baseline(seed=DEFAULT_SEED, **knobs):
    """County fixed effects, linear trend, additive season-total weather. Training only."""
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    state = _build(seed, resolved)
    train_years, _, test_keys = _keys(state)
    heat, deficit = _season_totals(state)
    county_index = [c for c in range(state["counties"]) for _ in train_years]
    years = [y for _ in range(state["counties"]) for y in train_years]
    design = _design(state, county_index, years, heat, deficit)
    target = np.array([state["observed"][c, y] for c, y in zip(county_index, years)])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    sigma = float(np.std(residual)) or 1.0
    values = {}
    for key in test_keys:
        c = int(key[2:4])
        y = int(key.split("_y")[1])
        row = _design(state, [c], [y], heat, deficit)[0]
        predicted = float(row @ coefficients)
        threshold = float(state["reference"][c, y] - SEVERE_LOSS_DROP)
        values[key] = {
            "yield_bu_acre": round(predicted, 4),
            "p_severe_loss": round(float(stats.norm.cdf((threshold - predicted) / sigma)), 6),
        }
    values["interaction_by_region"] = {name: 0.0 for name in REGIONS}
    return predictions_of(
        "baseline", SPEC, values, "additive_county_trend_regression", SPEC.baseline_rationale
    )


def oracle(seed=DEFAULT_SEED, **knobs):
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    state = _build(seed, resolved)
    _, _, test_keys = _keys(state)
    values = {}
    for key in test_keys:
        c = int(key[2:4])
        y = int(key.split("_y")[1])
        predicted = float(state["expected"][c, y])
        threshold = float(state["reference"][c, y] - SEVERE_LOSS_DROP)
        values[key] = {
            "yield_bu_acre": round(predicted, 4),
            "p_severe_loss": round(float(stats.norm.cdf((threshold - predicted) / 5.5)), 6),
        }
    values["interaction_by_region"] = dict(REGION_INTERACTION)
    return predictions_of("oracle", SPEC, values, "planted_response_surface", SPEC.oracle_rationale)


def trivial(seed=DEFAULT_SEED, **knobs):
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    state = _build(seed, resolved)
    train_years, _, test_keys = _keys(state)
    mean = float(np.mean([state["observed"][c, y] for c in range(state["counties"]) for y in train_years]))
    values = {k: {"yield_bu_acre": mean, "p_severe_loss": 0.5} for k in test_keys}
    values["interaction_by_region"] = {name: 0.0 for name in REGIONS}
    return predictions_of(
        "trivial", SPEC, values, "training_mean_yield", "One training-mean yield for every county-season."
    )


def evaluate(predictions, seed=DEFAULT_SEED, **knobs):
    seed = check_seed(seed)
    resolved = check_knobs(knobs, KNOBS)
    state = _build(seed, resolved)
    facts = truth(seed, **resolved)["facts"]
    actual = facts["yield_by_county_season"]
    severe = facts["severe_loss_by_county_season"]
    values = dict(unwrap(predictions))
    interaction = values.pop("interaction_by_region", None)
    require_keys(values, actual)
    yields, flags = {}, {}
    for key, row in values.items():
        if not isinstance(row, dict) or set(row) != {"yield_bu_acre", "p_severe_loss"}:
            raise Invalid("Each C2 entry needs exactly yield_bu_acre and p_severe_loss")
        yields[key] = real(row["yield_bu_acre"], "yield")
        flags[key] = probability(row["p_severe_loss"], "severe-loss probability")
    losses = squared_errors(yields, actual)
    interaction_error = None
    if isinstance(interaction, dict) and set(interaction) == set(REGIONS):
        interaction_error = float(
            np.mean([abs(real(interaction[r]) - REGION_INTERACTION[r]) for r in REGIONS])
        )
    heat_cut = float(np.quantile(state["stress_heat"], 0.75))
    deficit_cut = float(np.quantile(state["deficit"], 0.75))
    compound = []
    for key in actual:
        c = int(key[2:4])
        y = int(key.split("_y")[1])
        if state["stress_heat"][c, y] >= heat_cut and state["deficit"][c, y] >= deficit_cut:
            compound.append((yields[key] - actual[key]) ** 2)
    by_region = {}
    for name in REGIONS:
        members = [k for k in actual if state["region"][int(k[2:4])] == name]
        by_region[name] = round(
            float(np.sqrt(np.mean([(yields[k] - actual[k]) ** 2 for k in members]))), 4
        )
    return scored(
        SPEC,
        seed,
        primary=float(np.sqrt(np.mean(losses))),
        secondary={
            "severe_loss_brier": float(np.mean(brier_losses(flags, severe))),
            "interaction_recovery_error": interaction_error,
            "compound_season_rmse": float(np.sqrt(np.mean(compound))) if compound else None,
            "rmse_by_region": by_region,
            "compound_county_seasons": len(compound),
            "severe_loss_rate": float(np.mean(list(severe.values()))),
        },
        losses=losses,
        units=len(actual),
        notes=[
            "Counties within a region share a season shock; units are not independent.",
            "Uncertainty is reported on squared error and then reported as a root at the point estimate.",
        ],
    )


def narrative():
    return story(
        SPEC,
        decision=(
            "A crop-risk desk must size a reserve for three regions. Regional losses are "
            "either diversifiable across counties or they are not."
        ),
        agent_task=(
            "Find out whether heat and moisture enter additively or as a product, and in "
            "which regions. Two things make it hard on purpose: the stress window is a "
            "two-month slice whose labels are offset by one, and the agency suppresses the "
            "worst county-seasons, which is exactly where the interaction shows up."
        ),
        good_answer=(
            "Held-out RMSE below the additive regression, most of the gain concentrated in "
            "the hot-and-dry quartile, regional interaction coefficients near the planted "
            "values including a correct zero for the additive region, and a statement that "
            "county aggregates cannot identify a field-level effect."
        ),
        trap=(
            "The additive regression fits the record well and understates compound seasons "
            "in exactly the two regions where the reserve matters. Suppression makes the "
            "residual plot look clean because the worst observations are gone."
        ),
        two_minutes=[
            "Show the additive regression fitting the training seasons well.",
            "Show its residuals against heat times deficit; the pattern is only in two regions.",
            "Show the suppression column and where the missing county-seasons sit.",
            "Reveal the planted interaction coefficients, one of which is exactly zero.",
            "Compare the oracle in the compound-stress quartile against the additive fit.",
            "Say the sentence: synthetic counties, recovered interaction, not agronomy.",
        ],
    )
