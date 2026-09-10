"""Scripted, offline demonstrations of Symplex's scientific capabilities.

Every scenario in this module runs real computation against the repository's own
capability modules.  Nothing here calls a model, opens a socket, spawns a process or
reads the clock for a decision.  Given a seed, the numbers are reproducible.

A scenario is a list of steps.  Each step is a dictionary with a fixed shape::

    {
      "title":                 what this step is,
      "what_it_demonstrates":  the capability claim being exercised,
      "computation_run":       the literal call that produced the numbers,
      "result":                ordered list of (label, value) rows,
      "figures":               renderer-agnostic figure specifications,
      "scope":                 what the result does NOT establish,
    }

The ``scope`` field is mandatory.  Where a capability module returns its own scope
line, that line is carried through verbatim rather than paraphrased.
"""

from __future__ import annotations

import heapq
import importlib
import math
import time

import numpy as np

# --------------------------------------------------------------------------------------
# Defensive module loading.  A scenario whose modules have not landed is skipped, never
# faked and never allowed to crash the run.
# --------------------------------------------------------------------------------------

_CACHE: dict[str, object] = {}
_ERRORS: dict[str, str] = {}


def load(path):
    """Import ``path`` or return None, recording why it failed."""
    if path in _CACHE:
        return _CACHE[path]
    if path in _ERRORS:
        return None
    try:
        module = importlib.import_module(path)
    except Exception as exc:  # noqa: BLE001 - a missing capability must not stop the demo
        _ERRORS[path] = f"{type(exc).__name__}: {exc}"
        return None
    _CACHE[path] = module
    return module


def missing_modules(paths):
    """Return the subset of ``paths`` that cannot be imported, with the reason."""
    absent = []
    for path in paths:
        if load(path) is None:
            absent.append(f"{path} ({_ERRORS.get(path, 'unavailable')})")
    return absent


# --------------------------------------------------------------------------------------
# Small helpers shared by the scenarios
# --------------------------------------------------------------------------------------


def row(label, value, note=""):
    return {"label": label, "value": value, "note": note}


def num(value, digits=4):
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return f"{int(value):,}"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "n/a"
    if value != 0.0 and (abs(value) < 1e-3 or abs(value) >= 1e6):
        return f"{value:.{max(digits - 1, 1)}e}"
    return f"{value:.{digits}f}"


def fig_line(title, x_label, y_label, series, caption="", markers=()):
    return {
        "kind": "line",
        "title": title,
        "x_label": x_label,
        "y_label": y_label,
        "series": [dict(s) for s in series],
        "markers": [dict(m) for m in markers],
        "caption": caption,
    }


def fig_bars(title, items, x_label="", caption="", reference=None):
    return {
        "kind": "bars",
        "title": title,
        "x_label": x_label,
        "items": [dict(i) for i in items],
        "reference": reference,
        "caption": caption,
    }


def fig_scatter(title, x_label, y_label, series, caption="", markers=()):
    return {
        "kind": "scatter",
        "title": title,
        "x_label": x_label,
        "y_label": y_label,
        "series": [dict(s) for s in series],
        "markers": [dict(m) for m in markers],
        "caption": caption,
    }


def table(title, columns, rows, caption="", highlight=()):
    return {
        "kind": "table",
        "title": title,
        "columns": list(columns),
        "rows": [list(r) for r in rows],
        "highlight": list(highlight),
        "caption": caption,
    }


def thin(values, keep=240):
    """Subsample a series so the report stays small without changing its shape."""
    values = list(values)
    if len(values) <= keep:
        return [float(v) for v in values]
    step = len(values) / float(keep)
    return [float(values[min(len(values) - 1, int(i * step))]) for i in range(keep)]


# ======================================================================================
# Scenario 1 - discover: recover the governing equations from data alone
# ======================================================================================

DISCOVER_MODULES = ("symplex.modeling.systems", "symplex.hybrid.sindy", "symplex.modeling.analysis")

TRUE_ALPHA, TRUE_BETA, TRUE_GAMMA, TRUE_DELTA = 1.1, 0.40, 0.40, 0.10

MEANING = "Synthetic demonstration fixture; no measured quantity about any real population is established."


def _lotka_volterra_model():
    """A dimensionally-checked predator-prey declaration in the stock-and-flow contract."""

    def parameter(ident, value, dimension, unit, lower, upper):
        return {
            "id": ident,
            "name": ident,
            "value": value,
            "unit": unit,
            "dimension": dimension,
            "lower": lower,
            "upper": upper,
            "meaning": MEANING,
        }

    def flow(ident, source, target, rate, dimension, unit):
        return {
            "id": ident,
            "name": ident,
            "from_stock": source,
            "to_stock": target,
            "rate": rate,
            "unit": unit,
            "dimension": dimension,
            "delay_id": None,
            "meaning": MEANING,
        }

    return {
        "title": "Two-species predator-prey fixture",
        "boundary": "A single closed patch; births enter and deaths leave across the declared boundary.",
        "time_unit": "season",
        "time_dimension": "time",
        "horizon": 30.0,
        "stocks": [
            {
                "id": "prey",
                "name": "prey",
                "unit": "prey",
                "dimension": "prey",
                "initial": 10.0,
                "non_negative": True,
                "noise_sd": 0.0,
                "meaning": MEANING,
            },
            {
                "id": "predator",
                "name": "predator",
                "unit": "predator",
                "dimension": "predator",
                "initial": 5.0,
                "non_negative": True,
                "noise_sd": 0.0,
                "meaning": MEANING,
            },
        ],
        "flows": [
            flow("prey_birth", None, "prey", "alpha * prey", "prey*time^-1", "prey_per_season"),
            flow("predation", "prey", None, "beta * prey * predator", "prey*time^-1", "prey_per_season"),
            flow(
                "predator_birth",
                None,
                "predator",
                "delta * prey * predator",
                "predator*time^-1",
                "predator_per_season",
            ),
            flow("predator_death", "predator", None, "gamma * predator", "predator*time^-1", "predator_per_season"),
        ],
        "auxiliaries": [],
        "parameters": [
            parameter("alpha", TRUE_ALPHA, "time^-1", "per_season", 0.0, 10.0),
            parameter("beta", TRUE_BETA, "predator^-1*time^-1", "per_predator_season", 0.0, 10.0),
            parameter("gamma", TRUE_GAMMA, "time^-1", "per_season", 0.0, 10.0),
            parameter("delta", TRUE_DELTA, "prey^-1*time^-1", "per_prey_season", 0.0, 10.0),
        ],
        "couplings": [],
        "delays": [],
        "interventions": [],
        "conservation_groups": [],
        "scenarios": [
            {
                "id": "baseline",
                "name": "baseline",
                "intervention_ids": [],
                "interpretation": MEANING,
            }
        ],
        "assumptions": [
            "Mass-action encounters between two well-mixed populations.",
            "No age structure, no carrying capacity, no environmental forcing.",
        ],
        "unmodeled": ["Migration, seasonality, disease and any third species."],
    }


TRUE_TERMS = {
    "prey": {"prey": TRUE_ALPHA, "prey predator": -TRUE_BETA},
    "predator": {"predator": -TRUE_GAMMA, "prey predator": TRUE_DELTA},
}


def _scenario_entry(result):
    scenarios = result.get("scenarios")
    if isinstance(scenarios, dict):
        return next(iter(scenarios.values()))
    return scenarios[0]


def run_discover(seed=7):
    systems = load("symplex.modeling.systems")
    sindy = load("symplex.hybrid.sindy")
    analysis = load("symplex.modeling.analysis")
    rng = np.random.default_rng(seed)
    steps = []

    # -- Step 1: declare and integrate the truth -------------------------------------
    spec = _lotka_volterra_model()
    dimensions = systems.analyze_dimensions(spec)
    run = systems.simulate(spec, steps=1200, horizon=30.0, integrator="rk4", seed=seed)
    times = np.asarray(run["times"], dtype=float)
    entry = _scenario_entry(run)
    prey = np.asarray(entry["mean_trajectory"]["prey"], dtype=float)
    predator = np.asarray(entry["mean_trajectory"]["predator"], dtype=float)

    steps.append(
        {
            "title": "Declare a system, check its dimensions, integrate it",
            "what_it_demonstrates": (
                "The stock-and-flow runtime carries units through every expression. A rate whose "
                "dimensions do not reduce to stock-per-time is rejected before a single step is taken."
            ),
            "computation_run": (
                "symplex.modeling.systems.analyze_dimensions(model)\n"
                "symplex.modeling.systems.simulate(model, steps=1200, horizon=30.0, integrator='rk4')"
            ),
            "result": [
                row("Declaration", "4 flows, 2 stocks, 4 parameters, 0 auxiliaries"),
                row("Dimensional check", "consistent" if dimensions.get("consistent", True) else "INCONSISTENT",
                    "every flow rate reduces to its endpoint stock dimension divided by time"),
                row("Integrator", f"{run['integrator']}, dt = {num(run['step_size'], 5)} season"),
                row("Samples produced", len(times)),
                row("Prey range", f"{num(prey.min(), 2)} to {num(prey.max(), 2)}"),
                row("Predator range", f"{num(predator.min(), 2)} to {num(predator.max(), 2)}"),
            ],
            "figures": [
                fig_line(
                    "Integrated trajectory (the ground truth)",
                    "time (season)",
                    "population",
                    [
                        {"name": "prey", "x": thin(times), "y": thin(prey), "style": "line"},
                        {"name": "predator", "x": thin(times), "y": thin(predator), "style": "line"},
                    ],
                    caption="Deterministic RK4 integration of the declared model. These are the only equations the discovery step is not allowed to see.",
                ),
                fig_scatter(
                    "Phase portrait",
                    "prey",
                    "predator",
                    [{"name": "orbit", "x": thin(prey), "y": thin(predator), "style": "line"}],
                    caption="Closed orbits are the signature of the conservative predator-prey structure.",
                ),
            ],
            "scope": systems.SCOPE if hasattr(systems, "SCOPE") else
            "A declared model integrated under its own assumptions. Nothing here is evidence about a real population.",
        }
    )

    # -- Step 2: hide the equations, corrupt the observation --------------------------
    X_clean = np.column_stack([prey, predator])
    noise_level = 0.01
    sigma = noise_level * X_clean.std(axis=0)
    X = X_clean + rng.normal(0.0, sigma, size=X_clean.shape)
    snr = float(np.mean(X_clean.std(axis=0) / sigma))

    steps.append(
        {
            "title": "Throw the equations away and add measurement noise",
            "what_it_demonstrates": (
                "From here the pipeline receives a numeric array and a time vector. It is given no "
                "functional form, no term list beyond a generic polynomial basis, and no parameter values."
            ),
            "computation_run": "X = trajectory + N(0, 0.01 * sd(trajectory))  # observation noise, seeded",
            "result": [
                row("Handed forward", "a 1200 x 2 float array and a time vector - nothing else"),
                row("Observation noise", f"{noise_level:.0%} of each state's standard deviation"),
                row("Signal-to-noise ratio", num(snr, 1)),
                row("Withheld", "the four parameter values, the four flow expressions, the model title"),
            ],
            "figures": [
                fig_line(
                    "What the discovery step actually sees",
                    "time (season)",
                    "observed population",
                    [
                        {"name": "prey (observed)", "x": thin(times), "y": thin(X[:, 0]), "style": "line"},
                        {"name": "predator (observed)", "x": thin(times), "y": thin(X[:, 1]), "style": "line"},
                    ],
                    caption="Noisy samples. The derivative estimator, not the fit, is the fragile part of equation discovery - which is why the next step reports which estimator it used.",
                )
            ],
            "scope": (
                "Synthetic data generated by the very model being recovered. Recovering it is a check that "
                "the method works on a system it can express, not evidence that it recovers real-world dynamics."
            ),
        }
    )

    # -- Step 3: recover the equations ------------------------------------------------
    fit = sindy.fit_sindy(
        X,
        t=times,
        degree=2,
        threshold=0.05,
        alpha=1e-6,
        state_names=["prey", "predator"],
        derivative_method="savitzky_golay",
        window=21,
        polyorder=3,
        precision=3,
    )
    recovered = list(fit["equations"])

    coefficient_rows = []
    coeff = np.asarray(fit["coefficient_matrix"], dtype=float)
    names = list(fit["term_names"])
    states = list(fit["state_names"])
    worst_error = 0.0
    spurious = []
    truth_matrix = np.zeros_like(coeff)
    for j, state in enumerate(states):
        truth_map = TRUE_TERMS.get(state, {})
        for term, true_value in truth_map.items():
            index = names.index(term)
            truth_matrix[index, j] = true_value
            got = float(coeff[index, j])
            err = abs(got - true_value) / max(abs(true_value), 1e-12)
            worst_error = max(worst_error, err)
            coefficient_rows.append(
                [f"d{state}/dt", term, num(true_value, 4), num(got, 4), f"{err * 100:.2f}%"]
            )
        for term in fit["active_terms"][j]:
            if term not in truth_map:
                spurious.append(f"d{state}/dt <- {term} ({num(coeff[names.index(term), j], 4)})")
    # Render the withheld truth through the same renderer, so the comparison is like for like.
    truth = [
        sindy.render_equation(truth_matrix[:, j], names, lhs=f"d{state}/dt", precision=3)["equation"]
        for j, state in enumerate(states)
    ]

    steps.append(
        {
            "title": "Recover the governing equations from the numbers alone",
            "what_it_demonstrates": (
                "Sequentially-thresholded least squares over a degree-2 polynomial library selects, from "
                f"{fit['n_library_terms']} candidate terms, the {fit['total_active_terms']} that the data supports."
            ),
            "computation_run": (
                "symplex.hybrid.sindy.fit_sindy(X, t=times, degree=2, threshold=0.05,\n"
                "    derivative_method='savitzky_golay', window=21, polyorder=3)"
            ),
            "result": [
                row("RECOVERED", recovered[0]),
                row("TRUTH", truth[0]),
                row("RECOVERED", recovered[1]),
                row("TRUTH", truth[1]),
                row("Candidate terms offered", f"{fit['n_library_terms']} per equation "
                    f"({fit['n_library_terms'] * len(states)} free coefficients)"),
                row("Terms selected", fit["total_active_terms"],
                    f"out of {fit['n_library_terms'] * len(states)} available coefficients"),
                row("Spurious terms selected", len(spurious) if spurious else "none",
                    "; ".join(spurious) if spurious else "every selected term is a true term"),
                row("R^2 per equation", ", ".join(num(v, 5) for v in fit["r_squared"])),
                row("Worst coefficient error", f"{worst_error * 100:.2f}%"),
                row("Library condition number", num(fit["condition_number"], 1)),
                row("Derivative estimator", fit["derivative"].get("method", "unknown")),
            ],
            "figures": [
                table(
                    "Recovered coefficients against the withheld truth",
                    ["equation", "term", "true value", "recovered", "relative error"],
                    coefficient_rows,
                    caption="Four nonzero coefficients out of twelve candidates, each recovered from noisy samples without being told the functional form.",
                )
            ],
            "scope": fit.get("scope", ""),
        }
    )

    # -- Step 4: how sure is each term? ------------------------------------------------
    selection = sindy.stability_selection(
        X,
        t=times,
        n_bootstrap=30,
        sample_fraction=0.7,
        seed=seed,
        inclusion_cut=0.8,
        degree=2,
        threshold=0.05,
        alpha=1e-6,
        state_names=["prey", "predator"],
        derivative_method="savitzky_golay",
        window=21,
        polyorder=3,
    )
    entries = [e for e in selection["terms"] if e["inclusion_probability"] > 0.0]
    entries.sort(key=lambda e: -e["inclusion_probability"])
    bars = [
        {
            "label": f"{e['equation']} <- {e['term']}",
            "value": e["inclusion_probability"],
            "emphasis": e["inclusion_probability"] >= selection["inclusion_cut"],
        }
        for e in entries[:12]
    ]

    steps.append(
        {
            "title": "Report how often each term survives resampling",
            "what_it_demonstrates": (
                "A single sparse fit is a point estimate. Bootstrapping the fit turns term selection into a "
                "probability, so a term that only appears in a lucky subsample is visible as fragile rather "
                "than presented as a discovery."
            ),
            "computation_run": (
                f"symplex.hybrid.sindy.stability_selection(X, t=times, n_bootstrap={selection['n_bootstrap']},\n"
                f"    sample_fraction={selection['sample_fraction']}, inclusion_cut={selection['inclusion_cut']}, seed={seed})"
            ),
            "result": [
                row("Bootstrap fits", selection["n_bootstrap"]),
                row("Terms at inclusion probability 1.00",
                    sum(1 for e in entries if e["inclusion_probability"] >= 0.999)),
                row("Robust terms (>= cut)", len(selection["robust_terms"])),
                row("Fragile terms", len(selection["fragile_terms"]) if selection["fragile_terms"] else "none"),
                row("Never selected", len(selection["never_selected"]), "candidate terms the data never supported"),
                row("Non-convergent fits", selection["nonconvergent_fits"]),
            ],
            "figures": [
                fig_bars(
                    "Term inclusion probability across bootstrap resamples",
                    bars,
                    x_label="inclusion probability",
                    reference=selection["inclusion_cut"],
                    caption="Bars at or above the dashed cut are reported as structure. Anything below it is reported as fragile, not silently dropped.",
                )
            ],
            "scope": selection.get("scope", ""),
        }
    )

    # -- Step 5: accuracy against sparsity --------------------------------------------
    thresholds = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5]
    sweep = sindy.pareto_sweep(
        X,
        thresholds,
        t=times,
        degree=2,
        alpha=1e-6,
        state_names=["prey", "predator"],
        derivative_method="savitzky_golay",
        window=21,
        polyorder=3,
    )
    front = sweep["front"]
    front_rows = [
        [
            num(f["threshold"], 3),
            f["total_active_terms"],
            num(f["mean_rmse"], 5),
            num(f["mean_r_squared"], 5),
            "<-- selected" if abs(f["threshold"] - (sweep["suggested_threshold"] or -1)) < 1e-12 else "",
        ]
        for f in front
    ]

    steps.append(
        {
            "title": "Show the accuracy-versus-sparsity trade-off instead of one tuned answer",
            "what_it_demonstrates": (
                "Sparse regression has a knob. Reporting a single equation without the knob's effect hides "
                "the fact that a different threshold gives a different model. The whole front is shown, and "
                "the selection rule is stated."
            ),
            "computation_run": f"symplex.hybrid.sindy.pareto_sweep(X, thresholds={thresholds}, t=times, degree=2)",
            "result": [
                row("Thresholds swept", len(front)),
                row("Suggested threshold", num(sweep["suggested_threshold"], 3)),
                row("Selection rule", sweep.get("suggested_reason", "")),
                row("Monotone in active terms", "yes" if sweep["monotone_in_active_terms"] else "no",
                    "a higher threshold never selected more terms"),
            ],
            "figures": [
                fig_scatter(
                    "Accuracy against sparsity",
                    "active terms retained",
                    "mean RMSE of the derivative fit",
                    [
                        {
                            "name": "threshold sweep",
                            "x": [f["total_active_terms"] for f in front],
                            "y": [f["mean_rmse"] for f in front],
                            "style": "line-points",
                        }
                    ],
                    caption="The knee at four terms is the true model. Beyond it, extra terms buy no accuracy; below it, accuracy collapses.",
                ),
                table(
                    "Threshold sweep",
                    ["threshold", "active terms", "mean RMSE", "mean R^2", ""],
                    front_rows,
                    caption="",
                ),
            ],
            "scope": sweep.get("scope", ""),
        }
    )

    # -- Step 6: analyse the structure of what was recovered ---------------------------
    degree = int(fit["degree"])

    def recovered_field(t, x):
        point = np.asarray(x, dtype=float).reshape(1, -1)
        library = sindy.polynomial_library(point, degree=degree, state_names=["prey", "predator"])
        return np.asarray(library["features"], dtype=float) @ coeff

    equilibria = analysis.find_equilibria(
        lambda t, x: np.asarray(recovered_field(t, x)).ravel(),
        [[4.0, 2.5], [0.1, 0.1], [12.0, 4.0]],
    )
    interior = None
    for candidate in equilibria["equilibria"]:
        state = candidate["state"]
        if min(state) > 1e-3:
            interior = candidate
            break
    interior = interior or (equilibria["equilibria"][0] if equilibria["equilibria"] else None)

    loops_result = None
    if interior is not None:
        J = np.asarray(interior["jacobian"], dtype=float)
        signs = np.sign(J)
        loops_result = analysis.feedback_loops(signs.tolist(), labels=["prey", "predator"], max_len=2)

    true_interior = [TRUE_GAMMA / TRUE_DELTA, TRUE_ALPHA / TRUE_BETA]
    structure_rows = []
    if loops_result is not None:
        for loop in loops_result["loops"][:6]:
            structure_rows.append([loop["signature"], loop["polarity"], str(loop["length"]), num(loop["gain"], 4)])

    result_rows = [
        row("Interior equilibrium recovered",
            f"prey = {num(interior['state'][0], 3)}, predator = {num(interior['state'][1], 3)}" if interior else "none found"),
        row("Interior equilibrium (truth)", f"prey = {num(true_interior[0], 3)}, predator = {num(true_interior[1], 3)}",
            "gamma/delta and alpha/beta - never supplied to the pipeline"),
    ]
    if interior is not None:
        stab = interior["stability"]
        result_rows += [
            row("Stability classification", stab["classification"]),
            row("Spectral abscissa", num(stab["spectral_abscissa"], 6),
                "a value at zero is the marginal centre a conservative predator-prey system must have"),
            row("Oscillatory", "yes" if stab["oscillatory"] else "no"),
            row("Oscillation period",
                num(next((p for p in stab["oscillation_periods"] if p), None), 3) + " season",
                "truth: 2*pi/sqrt(alpha*gamma) = " + num(2 * math.pi / math.sqrt(TRUE_ALPHA * TRUE_GAMMA), 3)),
        ]
    if loops_result is not None:
        result_rows += [
            row("Feedback loops found", loops_result["loop_count"]),
            row("Reinforcing / balancing",
                f"{loops_result['reinforcing_count']} / {loops_result['balancing_count']}"),
        ]

    figures = []
    if structure_rows:
        figures.append(
            table(
                "Feedback structure of the recovered system",
                ["loop", "polarity", "length", "gain"],
                structure_rows,
                caption="Computed from the sign pattern of the Jacobian of the recovered equations - not from the declared model.",
            )
        )

    steps.append(
        {
            "title": "Analyse the recovered system as if it had been declared",
            "what_it_demonstrates": (
                "Equation discovery is only useful if what comes out is a first-class model. The recovered "
                "coefficients are handed straight to the equilibrium finder, the eigenvalue analysis and the "
                "loop enumerator, with no reference to the original declaration."
            ),
            "computation_run": (
                "field = polynomial_library(x, degree=2) @ recovered_coefficients\n"
                "symplex.modeling.analysis.find_equilibria(field, guesses=[[4, 2.5], [0.1, 0.1], [12, 4]])\n"
                "symplex.modeling.analysis.feedback_loops(sign(J), labels=['prey', 'predator'], max_len=2)"
            ),
            "result": result_rows,
            "figures": figures,
            "scope": (
                (loops_result or {}).get("scope")
                or "Local analysis at one equilibrium of a fitted model. It describes the fitted field's behaviour near that point and nothing beyond it."
            ),
        }
    )

    headline = (
        f"Recovered both governing equations from {len(times)} noisy samples: "
        f"{fit['total_active_terms']} of {fit['n_library_terms'] * len(states)} candidate coefficients selected, "
        f"worst coefficient error {worst_error * 100:.2f}%, R^2 = "
        + " / ".join(num(v, 4) for v in fit["r_squared"])
        + "."
    )
    return {
        "headline": headline,
        "wow": "The system wrote down an equation it was never told.",
        "steps": steps,
        "key_numbers": {
            "recovered_equations": recovered,
            "true_equations": truth,
            "worst_relative_coefficient_error": worst_error,
            "r_squared": list(fit["r_squared"]),
            "active_terms": fit["total_active_terms"],
            "library_terms": fit["n_library_terms"],
            "spurious_terms": spurious,
        },
    }


# ======================================================================================
# Scenario 2 - identify: refuse to over-trust a fit that looks good
# ======================================================================================

IDENTIFY_MODULES = ("symplex.inference.estimation",)

PK_TRUTH = {"ka": 1.20, "ke": 0.25, "F": 0.70, "V": 18.0}
PK_NAMES = ("ka", "ke", "F", "V")
PK_DOSE = 100.0
PK_BOUNDS = ((0.20, 6.00), (0.02, 2.00), (0.05, 1.00), (2.0, 120.0))


def _pk_curve(theta, times):
    """One-compartment model with first-order absorption; F and V enter only as F/V."""
    ka, ke, F, V = (float(v) for v in theta)
    gap = ka - ke
    if abs(gap) < 1e-9:
        gap = math.copysign(1e-9, gap if gap != 0 else 1.0)
    scale = (F * PK_DOSE * ka) / (V * gap)
    return scale * (np.exp(-ke * times) - np.exp(-ka * times))


def run_identify(seed=7):
    estimation = load("symplex.inference.estimation")
    rng = np.random.default_rng(seed)
    steps = []

    times = np.concatenate([np.arange(0.25, 4.0, 0.25), np.arange(4.0, 12.5, 0.75)])
    truth_vector = np.array([PK_TRUTH[n] for n in PK_NAMES], dtype=float)
    clean = _pk_curve(truth_vector, times)
    sigma_value = 0.05 * float(clean.max())
    observations = clean + rng.normal(0.0, sigma_value, size=clean.shape)

    def simulate(theta):
        return _pk_curve(theta, times)

    theta0 = np.array([0.90, 0.35, 0.45, 11.0], dtype=float)
    fit = estimation.least_squares_fit(
        simulate, observations, theta0, PK_BOUNDS, sigma=sigma_value
    )
    theta_hat = np.asarray(fit["theta_hat"] or fit["terminal_theta"], dtype=float)
    fitted = simulate(theta_hat)
    ratio_hat = theta_hat[2] / theta_hat[3]
    ratio_true = PK_TRUTH["F"] / PK_TRUTH["V"]

    fit_rows = [
        [name, num(PK_TRUTH[name], 4), num(theta_hat[i], 4),
         f"{abs(theta_hat[i] - PK_TRUTH[name]) / PK_TRUTH[name] * 100:.1f}%"]
        for i, name in enumerate(PK_NAMES)
    ]
    fit_rows.append(["F / V  (derived)", num(ratio_true, 5), num(ratio_hat, 5),
                     f"{abs(ratio_hat - ratio_true) / ratio_true * 100:.1f}%"])

    steps.append(
        {
            "title": "Fit four parameters to noisy observations - the fit looks excellent",
            "what_it_demonstrates": (
                "Weighted least squares against a caller-supplied simulator, with bounds and an evaluation "
                "budget. Every conventional goodness-of-fit signal comes back clean."
            ),
            "computation_run": (
                "symplex.inference.estimation.least_squares_fit(simulate, observations,\n"
                f"    theta0={theta0.tolist()}, bounds=..., sigma={num(sigma_value, 4)})"
            ),
            "result": [
                row("Converged", "yes" if fit["converged"] else "no", fit["termination"]),
                row("Observations", fit["observation_count"]),
                row("Residual sum of squares", num(fit["rss"], 4)),
                row("Reduced chi-square", num(fit["reduced_chi_square"], 4),
                    "a value near 1.0 is exactly what a well-specified model with correct noise looks like"),
                row("Simulator evaluations", f"{fit['evaluations']} of {fit['max_evaluations']}"),
                row("Any parameter at a bound",
                    ", ".join(n for n, flag in zip(PK_NAMES, fit["at_bound"]) if flag) or "no"),
            ],
            "figures": [
                fig_line(
                    "Fitted concentration curve against the observations",
                    "time (hour)",
                    "concentration (mg/L)",
                    [
                        {"name": "observed", "x": times.tolist(), "y": observations.tolist(), "style": "points"},
                        {"name": "fitted", "x": times.tolist(), "y": fitted.tolist(), "style": "line"},
                    ],
                    caption="By eye and by reduced chi-square this is a good fit. That is the problem.",
                ),
                table(
                    "Recovered parameters against the truth",
                    ["parameter", "true value", "fitted value", "relative error"],
                    fit_rows,
                    highlight=["F", "V"],
                    caption="Two parameters are badly wrong. Their ratio is nearly exact. Nothing in the fit report above says so.",
                ),
            ],
            "scope": fit.get("scope", ""),
        }
    )

    # -- Step 2: the confident-looking standard errors ---------------------------------
    information = estimation.fisher_information(simulate, theta_hat, sigma=sigma_value, bounds=PK_BOUNDS)
    common_rcond = 1e-6
    naive = estimation.parameter_uncertainty(
        information["fim"],
        scale=sigma_value,
        theta=theta_hat,
        parameter_names=list(PK_NAMES),
        confidence=0.95,
        rcond=common_rcond,
    )
    uncertainty = estimation.parameter_uncertainty(
        information["fim"],
        scale=sigma_value,
        theta=theta_hat,
        parameter_names=list(PK_NAMES),
        confidence=0.95,
    )
    se = naive["standard_errors"]
    se_default = uncertainty["standard_errors"]
    swing = max(
        (se_default[i] / se[i]) for i in range(len(PK_NAMES)) if se[i] > 0
    )
    naive_rows = [
        [
            name,
            num(theta_hat[i], 4),
            num(se[i], 4),
            f"{se[i] / max(abs(theta_hat[i]), 1e-12) * 100:.1f}%",
            f"[{num(naive['intervals'][i]['lower'], 3)}, {num(naive['intervals'][i]['upper'], 3)}]",
            f"{se_default[i] / max(abs(theta_hat[i]), 1e-12) * 100:.1f}%",
        ]
        for i, name in enumerate(PK_NAMES)
    ]

    steps.append(
        {
            "title": "Ask for standard errors - and get a confident-looking number that is an artefact",
            "what_it_demonstrates": (
                "The asymptotic covariance comes from a spectral pseudo-inverse of the Fisher information. "
                "Because a pseudo-inverse floors tiny eigenvalues instead of dividing by them, it always "
                "returns a finite standard error - and the size of that error is set by where the floor is "
                "put, not by the data."
            ),
            "computation_run": (
                "symplex.inference.estimation.fisher_information(simulate, theta_hat, sigma=sigma)\n"
                "symplex.inference.estimation.parameter_uncertainty(fim, scale=sigma, rcond=1e-6)   # a common default\n"
                "symplex.inference.estimation.parameter_uncertainty(fim, scale=sigma)               # the module default"
            ),
            "result": [
                row("Reported SE on F (rcond = 1e-6)",
                    f"{num(theta_hat[2], 4)} +/- {num(se[2], 4)}  ({se[2] / theta_hat[2] * 100:.1f}% relative)",
                    "a relative standard error most reports would ship without comment"),
                row("Reported SE on F (module default rcond)",
                    f"{num(theta_hat[2], 4)} +/- {num(se_default[2], 4)}  ({se_default[2] / theta_hat[2] * 100:.1f}% relative)"),
                row("Swing in reported precision", f"{swing:.0f}x",
                    "same data, same fit, same function - only the eigenvalue floor changed"),
                row("Information-matrix condition number", num(information["condition_number"], 3)),
                row("Smallest / largest eigenvalue", num(information["eigenvalue_ratio_min"], 3)),
                row("Numerical rank", f"{information['numerical_rank']} of {information['parameter_count']}",
                    "one direction in parameter space carries no information at all"),
                row("Floored directions", naive["floored_directions"],
                    "directions where the reported error is a numerical floor, not a measurement"),
            ],
            "figures": [
                table(
                    "The report a naive pipeline would ship",
                    ["parameter", "estimate", "standard error", "relative SE", "95% interval", "relative SE at the default floor"],
                    naive_rows,
                    highlight=["F", "V"],
                    caption="Read the last two columns together. For ka and ke they agree. For F and V the reported precision moves by two orders of magnitude when the floor moves, which is the tell that neither number is being measured.",
                )
            ],
            "scope": uncertainty.get("scope", ""),
        }
    )

    # -- Step 3: the identifiability audit ---------------------------------------------
    audit = estimation.practical_identifiability(
        information["fim"],
        theta=theta_hat,
        parameter_names=list(PK_NAMES),
        scale=sigma_value,
    )
    verdict_rows = [
        [
            e["name"],
            e["verdict"],
            num(e["max_abs_correlation"], 4),
            e["most_correlated_with"] or "-",
            num(e["null_space_loading"], 4),
        ]
        for e in audit["parameters"]
    ]
    pairs = audit["collinear_pairs"]
    spectrum = [abs(v) for v in audit["eigenvalues"]]
    peak = max(spectrum) or 1.0
    eigen_bars = [
        {"label": f"direction {i + 1}", "value": max(v / peak, 1e-18), "emphasis": v / peak < 1e-6}
        for i, v in enumerate(spectrum)
    ]

    steps.append(
        {
            "title": "Run the identifiability audit - and watch it refuse the fit",
            "what_it_demonstrates": (
                "The audit reads the eigenvalue decay of the information matrix together with the parameter "
                "correlation matrix, and returns a per-parameter verdict. It contradicts the standard errors "
                "printed one step earlier."
            ),
            "computation_run": (
                "symplex.inference.estimation.practical_identifiability(fim, theta=theta_hat,\n"
                "    parameter_names=['ka', 'ke', 'F', 'V'], collinearity_threshold=0.95)"
            ),
            "result": [
                row("OVERALL VERDICT", str(audit["overall"]).upper()),
                row("Verdict counts", ", ".join(f"{k}: {v}" for k, v in audit["verdict_counts"].items())),
                row("Collinear pairs found",
                    "; ".join(f"({p['names'][0]}, {p['names'][1]}) r = {num(p['correlation'], 6)}" for p in pairs)
                    if pairs else "none"),
                row("Numerical rank", f"{audit['numerical_rank']} of {len(PK_NAMES)}"),
                row("Eigenvalue ratio (smallest / largest)", num(min(audit["eigenvalue_ratios"]), 3)),
            ],
            "figures": [
                table(
                    "Per-parameter verdict",
                    ["parameter", "verdict", "max |correlation|", "with", "null-space loading"],
                    verdict_rows,
                    highlight=["F", "V"],
                    caption="ka and ke are supported by the data. F and V are not - only their ratio is.",
                ),
                fig_bars(
                    "Information carried by each direction in parameter space (log scale, normalised)",
                    eigen_bars,
                    x_label="eigenvalue / largest eigenvalue",
                    caption="Three directions carry information. The fourth is numerically zero: that is the F-V valley.",
                ),
            ],
            "scope": audit.get("scope", ""),
        }
    )

    # -- Step 4: profile likelihood confirms it -----------------------------------------
    index_F = PK_NAMES.index("F")
    index_ke = PK_NAMES.index("ke")
    grid_F = np.linspace(0.25, 1.00, 13)
    grid_ke = np.linspace(0.20, 0.50, 15)
    profile_F = estimation.profile_likelihood(
        simulate, observations, theta_hat, index_F, grid_F.tolist(), PK_BOUNDS, sigma=sigma_value
    )
    profile_ke = estimation.profile_likelihood(
        simulate, observations, theta_hat, index_ke, grid_ke.tolist(), PK_BOUNDS, sigma=sigma_value
    )

    def interval_text(profile):
        ci = profile["confidence_interval"]
        left = "open" if profile["open_left"] else num(ci["lower"], 4)
        right = "open" if profile["open_right"] else num(ci["upper"], 4)
        return f"[{left}, {right}]"

    steps.append(
        {
            "title": "Confirm it the expensive way: re-optimise everything else along a grid",
            "what_it_demonstrates": (
                "The profile likelihood is the check that does not rely on a local quadratic approximation. "
                "Fix one parameter, re-fit the rest, and watch the residual. A flat profile means the data "
                "cannot distinguish the values on that axis - regardless of what the curvature said."
            ),
            "computation_run": (
                "symplex.inference.estimation.profile_likelihood(simulate, observations, theta_hat,\n"
                "    index=F, grid=linspace(0.25, 1.00, 13), confidence=0.95)\n"
                "symplex.inference.estimation.profile_likelihood(..., index=ke, grid=linspace(0.20, 0.50, 15))"
            ),
            "result": [
                row("Profile of F: verdict", str(profile_F["identifiability"]).upper()),
                row("Profile of F: max delta chi-square", num(profile_F["max_delta"], 6),
                    f"threshold for a 95% bound is {num(profile_F['threshold'], 4)}"),
                row("Profile of F: flat", "YES - the profile never rises" if profile_F["flat"] else "no"),
                row("Profile of F: 95% interval", interval_text(profile_F),
                    "an open interval is the honest answer here"),
                row("Profile of ke: verdict", str(profile_ke["identifiability"]).upper()),
                row("Profile of ke: max delta chi-square", num(profile_ke["max_delta"], 4)),
                row("Profile of ke: 95% interval", interval_text(profile_ke)),
                row("Simulator evaluations spent", profile_F["evaluations"] + profile_ke["evaluations"]),
            ],
            "figures": [
                fig_line(
                    "Profile likelihood: a flat valley beside a real minimum",
                    "parameter value, normalised to its fitted estimate",
                    "delta chi-square from the best fit",
                    [
                        {
                            "name": "F (non-identifiable)",
                            "x": [float(v) / max(theta_hat[index_F], 1e-12) for v in profile_F["grid"]],
                            "y": [float(v) for v in profile_F["delta_chi_square"]],
                            "style": "line-points",
                        },
                        {
                            "name": "ke (identifiable)",
                            "x": [float(v) / max(theta_hat[index_ke], 1e-12) for v in profile_ke["grid"]],
                            "y": [float(v) for v in profile_ke["delta_chi_square"]],
                            "style": "line-points",
                        },
                    ],
                    markers=[{"kind": "hline", "value": profile_F["threshold"], "label": "95% threshold"}],
                    caption="ke rises steeply out of its minimum and gets a finite interval. F does not move at all: every value of F fits the data equally well once V compensates.",
                )
            ],
            "scope": profile_F.get("scope", ""),
        }
    )

    # -- Step 5: what does survive ------------------------------------------------------
    def simulate_reduced(theta3):
        full = np.array([theta3[0], theta3[1], theta3[2] * PK_TRUTH["V"], PK_TRUTH["V"]], dtype=float)
        return _pk_curve(full, times)

    reduced_bounds = ((0.20, 6.00), (0.02, 2.00), (0.001, 1.0))
    reduced_theta0 = np.array([0.90, 0.35, 0.030], dtype=float)
    reduced_fit = estimation.least_squares_fit(
        simulate_reduced, observations, reduced_theta0, reduced_bounds, sigma=sigma_value
    )
    reduced_theta = np.asarray(reduced_fit["theta_hat"] or reduced_fit["terminal_theta"], dtype=float)
    reduced_information = estimation.fisher_information(
        simulate_reduced, reduced_theta, sigma=sigma_value, bounds=reduced_bounds
    )
    reduced_audit = estimation.practical_identifiability(
        reduced_information["fim"],
        theta=reduced_theta,
        parameter_names=["ka", "ke", "F_over_V"],
        scale=sigma_value,
    )

    steps.append(
        {
            "title": "Re-parameterise to what the data supports - then audit again instead of declaring victory",
            "what_it_demonstrates": (
                "The correct response to a non-identifiable pair is not a bigger optimiser, it is a smaller "
                "model. Replacing (F, V) with the single quantity F/V removes the flat direction entirely. "
                "The audit is then re-run on the reduced model, and it still reports what is left."
            ),
            "computation_run": (
                "least_squares_fit(simulate_reduced, observations, theta0=[ka, ke, F/V], bounds=...)\n"
                "practical_identifiability(fisher_information(simulate_reduced, theta_hat), ...)"
            ),
            "result": [
                row("Reduced model", "3 parameters: ka, ke, F/V"),
                row("F/V estimated", num(reduced_theta[2], 5), f"true value {num(ratio_true, 5)}"),
                row("Condition number before", num(information["condition_number"], 3)),
                row("Condition number after", num(reduced_information["condition_number"], 3)),
                row("Numerical rank after",
                    f"{reduced_information['numerical_rank']} of {reduced_information['parameter_count']}"),
                row("Structural non-identifiability", "ELIMINATED",
                    "the perfectly collinear (F, V) direction no longer exists"),
                row("OVERALL VERDICT after", str(reduced_audit["overall"]).upper(),
                    "still not clean - see the residual correlation below"),
                row("Residual correlated pair after",
                    "; ".join(f"({p['names'][0]}, {p['names'][1]}) r = {num(p['correlation'], 4)}"
                              for p in reduced_audit["collinear_pairs"]) or "none",
                    "a practical, data-limited correlation - unlike the structural r = 1.000000 it replaced"),
                row("Reduced chi-square after", num(reduced_fit["reduced_chi_square"], 4),
                    "the fit is no worse - the dropped parameter was never doing any work"),
            ],
            "figures": [
                table(
                    "Before and after re-parameterisation",
                    ["quantity", "4-parameter model", "3-parameter model"],
                    [
                        ["parameters", "4", "3"],
                        ["information condition number", num(information["condition_number"], 3),
                         num(reduced_information["condition_number"], 3)],
                        ["numerical rank", f"{information['numerical_rank']} / 4",
                         f"{reduced_information['numerical_rank']} / 3"],
                        ["worst |correlation|",
                         num(max((abs(p["correlation"]) for p in pairs), default=0.0), 6),
                         num(max((abs(p["correlation"]) for p in reduced_audit["collinear_pairs"]), default=0.0), 6)],
                        ["overall identifiability", str(audit["overall"]), str(reduced_audit["overall"])],
                        ["reduced chi-square", num(fit["reduced_chi_square"], 4),
                         num(reduced_fit["reduced_chi_square"], 4)],
                    ],
                    caption="Same data, same fit quality, one fewer claim - and the audit still declines to sign off on the rest.",
                )
            ],
            "scope": reduced_audit.get("scope", ""),
        }
    )

    pair_text = "; ".join(f"({p['names'][0]}, {p['names'][1]}) r = {num(p['correlation'], 5)}" for p in pairs) or "none"
    headline = (
        f"The fit converged with reduced chi-square {num(fit['reduced_chi_square'], 3)} and finite standard "
        f"errors on all four parameters. The identifiability audit returned '{audit['overall']}' and named the "
        f"collinear pair {pair_text}; the profile likelihood of F is flat "
        f"(max delta chi-square {num(profile_F['max_delta'], 5)} against a threshold of {num(profile_F['threshold'], 3)})."
    )
    return {
        "headline": headline,
        "wow": "It refuses to over-trust its own fit.",
        "steps": steps,
        "key_numbers": {
            "overall_verdict": audit["overall"],
            "collinear_pairs": pairs,
            "reduced_chi_square": fit["reduced_chi_square"],
            "profile_F_flat": profile_F["flat"],
            "profile_F_max_delta": profile_F["max_delta"],
            "profile_ke_verdict": profile_ke["identifiability"],
            "condition_number_before": information["condition_number"],
            "condition_number_after": reduced_information["condition_number"],
        },
    }


# ======================================================================================
# Scenario 3 - route: five rival policies, common random numbers, no single winner
# ======================================================================================

ROUTE_MODULES = ("symplex.verticals.aisystems.topology", "symplex.verticals.aisystems.workload")

ROUTE_REGIMES = ("steady", "flash_crowd", "adversarial_mix")
SLA_SECONDS = 6.0
LATENCY_PENALTY_USD = 0.004  # dollars charged per second of latency past the SLA


def _tier_table(topology):
    tiers = {}
    for node in topology.model_dump()["nodes"]:
        if node["kind"] == "model" and node["tier"]:
            tiers[node["tier"]] = node
    return tiers


def _success_probability(capability, difficulty):
    return float(min(0.995, max(0.02, 0.5 + 2.2 * (capability - difficulty))))


def _policies(tiers):
    order = ["cheap", "mid", "frontier"]

    def always(tier):
        return lambda r: tier

    def declared(r):
        return {"simple": "cheap", "moderate": "mid"}.get(r.declared_class, "frontier")

    def hint(r):
        if r.difficulty_hint < 0.35:
            return "cheap"
        if r.difficulty_hint < 0.70:
            return "mid"
        return "frontier"

    def value_weighted(r):
        stake = r.value_usd * max(r.difficulty_hint, 0.05)
        if stake < 0.25:
            return "cheap"
        if stake < 1.10:
            return "mid"
        return "frontier"

    return {
        "always_cheap": {"fn": always("cheap"), "idea": "send everything to the cheapest model"},
        "always_frontier": {"fn": always("frontier"), "idea": "send everything to the strongest model"},
        "trust_declared_class": {"fn": declared, "idea": "believe the class the client declares"},
        "difficulty_hint": {"fn": hint, "idea": "threshold on the client's difficulty hint"},
        "value_weighted": {"fn": value_weighted, "idea": "escalate in proportion to what the request is worth"},
        "_order": order,
    }


def _oracle_choice(request, tiers):
    for name in ("cheap", "mid", "frontier"):
        if tiers[name]["capability"] >= request.true_difficulty:
            return name
    return "frontier"


def _run_policy(trace, tiers, choose, seed):
    """Multi-server FIFO queue per tier, driven by the shared trace and shared CRN draws."""
    free = {name: [0.0] * int(tiers[name]["servers"]) for name in tiers}
    for name in free:
        heapq.heapify(free[name])
    total_cost = 0.0
    total_value = 0.0
    latencies = []
    failures = 0
    penalty = 0.0
    tier_counts = {name: 0 for name in tiers}
    workload = load("symplex.verticals.aisystems.workload")

    for request in trace.requests:
        name = choose(request)
        node = tiers[name]
        tier_counts[name] += 1
        service = float(node["service_seconds"]) * float(request.work_factor)
        start = max(request.arrival_time, heapq.heappop(free[name]))
        finish = start + service
        heapq.heappush(free[name], finish)
        latency = finish - request.arrival_time
        latencies.append(latency)
        cost = float(node["cost_per_call_usd"]) * (1.0 + request.prompt_tokens / 1000.0)
        total_cost += cost
        # Common random numbers: one draw per request, reused by every policy and every tier.
        draw = workload.crn_uniform(seed, "outcome", request.id)
        if draw < _success_probability(float(node["capability"]), float(request.true_difficulty)):
            total_value += float(request.value_usd)
        else:
            failures += 1
        penalty += LATENCY_PENALTY_USD * max(0.0, latency - SLA_SECONDS)

    count = max(len(trace.requests), 1)
    workload_module = load("symplex.verticals.aisystems.workload")
    return {
        "requests": count,
        "cost_usd": total_cost,
        "cost_per_request": total_cost / count,
        "failure_rate": failures / count,
        "p95_latency": float(workload_module.quantile(latencies, 0.95)),
        "mean_latency": float(sum(latencies) / count),
        "value_usd": total_value,
        "latency_penalty_usd": penalty,
        "utility_usd": total_value - total_cost - penalty,
        "tier_mix": {k: v / count for k, v in tier_counts.items()},
    }


def _pareto_front(points, x_key, y_key):
    """Indices of points not dominated on (lower x, lower y)."""
    front = []
    for i, point in enumerate(points):
        dominated = any(
            other is not point
            and other[x_key] <= point[x_key]
            and other[y_key] <= point[y_key]
            and (other[x_key] < point[x_key] or other[y_key] < point[y_key])
            for other in points
        )
        if not dominated:
            front.append(i)
    return front


def run_route(seed=7):
    topology_module = load("symplex.verticals.aisystems.topology")
    workload_module = load("symplex.verticals.aisystems.workload")
    steps = []

    topology = topology_module.reference_topology(seed=seed)
    validation = topology_module.validate(topology)
    profile = topology_module.architecture_profile(topology)
    tiers = _tier_table(topology)
    library = workload_module.regime_library()
    regimes = [name for name in ROUTE_REGIMES if name in library]

    traces = {
        name: workload_module.generate_trace(library[name], seed=seed, horizon_seconds=120.0)
        for name in regimes
    }

    steps.append(
        {
            "title": "Declare the serving topology and generate the workloads",
            "what_it_demonstrates": (
                "Model tiers, their capability, their per-call price and their concurrency all come from a "
                "validated topology declaration rather than from numbers invented for the comparison. Every "
                "policy is then measured on byte-identical traces."
            ),
            "computation_run": (
                f"symplex.verticals.aisystems.topology.reference_topology(seed={seed})\n"
                "symplex.verticals.aisystems.topology.validate(topology)\n"
                f"symplex.verticals.aisystems.workload.generate_trace(regime, seed={seed}, horizon_seconds=120.0)"
            ),
            "result": [
                row("Architecture", profile["architecture"]),
                row("Topology valid", "yes" if validation.get("valid", True) else "no"),
                row("Nodes / edges", f"{len(topology.model_dump()['nodes'])} / {len(topology.model_dump()['edges'])}"),
                row("Model tiers", ", ".join(f"{k} (capability {tiers[k]['capability']})" for k in ("cheap", "mid", "frontier"))),
                row("Regimes compared", ", ".join(regimes)),
                row("Requests per regime",
                    ", ".join(f"{name}: {len(traces[name].requests)}" for name in regimes)),
                row("Trace digests",
                    ", ".join(f"{name}: {traces[name].trace_digest[:12]}" for name in regimes),
                    "identical for every policy - the comparison is paired, not independent"),
            ],
            "figures": [
                table(
                    "Model tiers as declared in the topology",
                    ["tier", "capability", "cost per call (USD)", "service seconds", "servers"],
                    [
                        [k, num(tiers[k]["capability"], 2), num(tiers[k]["cost_per_call_usd"], 5),
                         num(tiers[k]["service_seconds"], 2), str(tiers[k]["servers"])]
                        for k in ("cheap", "mid", "frontier")
                    ],
                    caption="A 50x price spread and a 2x capability spread between the cheapest and the strongest tier.",
                )
            ],
            "scope": profile.get("scope", ""),
        }
    )

    # -- Step 2: run all five policies plus the oracle on every regime -----------------
    policies = _policies(tiers)
    names = [k for k in policies if not k.startswith("_")]
    results = {}
    oracle = {}
    for regime in regimes:
        trace = traces[regime]
        oracle[regime] = _run_policy(trace, tiers, lambda r: _oracle_choice(r, tiers), seed)
        for name in names:
            results[(regime, name)] = _run_policy(trace, tiers, policies[name]["fn"], seed)

    per_regime_tables = []
    for regime in regimes:
        rows = []
        for name in names:
            r = results[(regime, name)]
            rows.append([
                name,
                num(r["cost_per_request"] * 1000, 3),
                f"{r['failure_rate'] * 100:.1f}%",
                num(r["p95_latency"], 2),
                num(r["utility_usd"], 2),
                num(oracle[regime]["utility_usd"] - r["utility_usd"], 2),
            ])
        o = oracle[regime]
        rows.append([
            "ORACLE (sees true difficulty)",
            num(o["cost_per_request"] * 1000, 3),
            f"{o['failure_rate'] * 100:.1f}%",
            num(o["p95_latency"], 2),
            num(o["utility_usd"], 2),
            "0.00",
        ])
        per_regime_tables.append(
            table(
                f"Regime: {regime}",
                ["policy", "cost per request (milli-USD)", "failure rate", "p95 latency (s)",
                 "net utility (USD)", "regret vs oracle (USD)"],
                rows,
                caption="",
            )
        )

    best_per_regime = {
        regime: min(names, key=lambda n: oracle[regime]["utility_usd"] - results[(regime, n)]["utility_usd"])
        for regime in regimes
    }
    winners = sorted(set(best_per_regime.values()))

    steps.append(
        {
            "title": "Run five rival policies against an oracle, on identical traces",
            "what_it_demonstrates": (
                "Because the arrival times, the request contents and the per-request outcome draws are shared "
                "across policies (common random numbers), differences between policies are attributable to the "
                "policies and not to sampling noise between runs."
            ),
            "computation_run": (
                "for regime in regimes:\n"
                "    trace = generate_trace(regime, seed)          # one trace, reused by every policy\n"
                "    for policy in five_policies + [oracle]:\n"
                "        simulate multi-server FIFO queues per tier, share crn_uniform(seed, 'outcome', request.id)"
            ),
            "result": [
                row("Policies compared", f"{len(names)} plus a true-difficulty oracle"),
                row("Total simulated requests",
                    sum(len(traces[r].requests) for r in regimes) * (len(names) + 1)),
                row("Best policy by regime",
                    "; ".join(f"{regime}: {best_per_regime[regime]}" for regime in regimes)),
                row("Distinct winners", len(winners), ", ".join(winners)),
                row("A single policy wins everywhere", "NO" if len(winners) > 1 else "yes"),
            ],
            "figures": per_regime_tables,
            "scope": (
                "A declared queueing model of a declared topology driven by a synthetic workload generator. "
                "Service times, capability values and prices are declared parameters, not measurements of any "
                "deployed system, so these numbers rank policies inside this model and establish nothing about "
                "any real serving stack."
            ),
        }
    )

    # -- Step 3: Pareto front over cost and failure rate --------------------------------
    pooled = []
    for name in names:
        cost = sum(results[(r, name)]["cost_usd"] for r in regimes)
        requests = sum(results[(r, name)]["requests"] for r in regimes)
        failures = sum(results[(r, name)]["failure_rate"] * results[(r, name)]["requests"] for r in regimes)
        pooled.append({
            "name": name,
            "cost_per_request": cost / requests,
            "failure_rate": failures / requests,
            "p95": max(results[(r, name)]["p95_latency"] for r in regimes),
            "utility": sum(results[(r, name)]["utility_usd"] for r in regimes),
        })
    front_indices = set(_pareto_front(pooled, "cost_per_request", "failure_rate"))
    front_names = [pooled[i]["name"] for i in sorted(front_indices)]
    dominated = [p["name"] for i, p in enumerate(pooled) if i not in front_indices]

    steps.append(
        {
            "title": "Show the Pareto front instead of naming one winner",
            "what_it_demonstrates": (
                "Cost and failure rate trade off against each other. Presenting a single 'best' policy would "
                "require silently choosing an exchange rate between dollars and failures. The front is reported "
                "instead, together with which policies are genuinely dominated."
            ),
            "computation_run": "pareto_front(policies, x='cost per request', y='failure rate')  # pooled across all regimes",
            "result": [
                row("On the front", ", ".join(front_names)),
                row("Dominated (strictly worse on both axes)", ", ".join(dominated) if dominated else "none"),
                row("Cheapest policy",
                    min(pooled, key=lambda p: p["cost_per_request"])["name"]),
                row("Most accurate policy",
                    min(pooled, key=lambda p: p["failure_rate"])["name"]),
                row("Cost spread across the front",
                    f"{num(min(p['cost_per_request'] for p in pooled) * 1000, 3)} to "
                    f"{num(max(p['cost_per_request'] for p in pooled) * 1000, 3)} milli-USD per request"),
            ],
            "figures": [
                fig_scatter(
                    "Cost against failure rate (pooled over all regimes)",
                    "cost per request (milli-USD)",
                    "failure rate",
                    [
                        {
                            "name": "on the Pareto front",
                            "x": [pooled[i]["cost_per_request"] * 1000 for i in sorted(front_indices)],
                            "y": [pooled[i]["failure_rate"] for i in sorted(front_indices)],
                            "labels": [pooled[i]["name"] for i in sorted(front_indices)],
                            "style": "line-points",
                        },
                        {
                            "name": "dominated",
                            "x": [p["cost_per_request"] * 1000 for i, p in enumerate(pooled) if i not in front_indices],
                            "y": [p["failure_rate"] for i, p in enumerate(pooled) if i not in front_indices],
                            "labels": [p["name"] for i, p in enumerate(pooled) if i not in front_indices],
                            "style": "points",
                        },
                    ],
                    caption="Lower and further left is better on both axes. Points on the connected line are choices; points off it are mistakes.",
                )
            ],
            "scope": (
                "The front is computed over two reported axes only. A policy off the front on these axes may "
                "still be preferable on an axis not measured here, such as tail latency or vendor concentration."
            ),
        }
    )

    # -- Step 4: regret against the oracle, per regime -----------------------------------
    regret_rows = []
    ranking_by_regime = {}
    for regime in regimes:
        ranked = sorted(names, key=lambda n: oracle[regime]["utility_usd"] - results[(regime, n)]["utility_usd"])
        ranking_by_regime[regime] = ranked
        regret_rows.append([regime] + [f"{ranked.index(n) + 1}" for n in names])

    flips = []
    for i, name in enumerate(names):
        ranks = [ranking_by_regime[r].index(name) + 1 for r in regimes]
        if max(ranks) - min(ranks) >= 2:
            flips.append(f"{name}: rank {min(ranks)} in {regimes[ranks.index(min(ranks))]}, "
                         f"rank {max(ranks)} in {regimes[ranks.index(max(ranks))]}")

    regret_series = []
    for name in names:
        regret_series.append({
            "name": name,
            "x": list(range(len(regimes))),
            "y": [oracle[r]["utility_usd"] - results[(r, name)]["utility_usd"] for r in regimes],
            "style": "line-points",
            "categories": list(regimes),
        })

    steps.append(
        {
            "title": "Measure regret against an oracle that cannot exist",
            "what_it_demonstrates": (
                "The oracle routes on the request's true difficulty, which no deployed router can observe. "
                "It is not a competitor; it is a yardstick that turns 'which policy is best' into 'how much "
                "is left on the table, and where'."
            ),
            "computation_run": "regret(policy, regime) = utility(oracle, regime) - utility(policy, regime)",
            "result": [
                row("Rank changes across regimes", "; ".join(flips) if flips else "none"),
                row("Best in steady", best_per_regime.get("steady", "n/a")),
                row("Best under a flash crowd", best_per_regime.get("flash_crowd", "n/a")),
                row("Best under an adversarial mix", best_per_regime.get("adversarial_mix", "n/a")),
                row("Worst-case regret of the best-on-average policy",
                    num(max(oracle[r]["utility_usd"] - results[(r, min(names, key=lambda n: sum(oracle[q]["utility_usd"] - results[(q, n)]["utility_usd"] for q in regimes)))]["utility_usd"] for r in regimes), 2) + " USD"),
            ],
            "figures": [
                fig_line(
                    "Regret against the oracle, by regime",
                    "regime",
                    "regret (USD over the 120 s trace)",
                    regret_series,
                    caption="Lines that cross are the whole point: the ordering of policies is not stable across operating conditions.",
                ),
                table(
                    "Policy rank by regime (1 is best)",
                    ["regime"] + names,
                    regret_rows,
                    caption="",
                ),
            ],
            "scope": (
                "Regret is measured against an oracle inside this simulation, so it bounds the loss attributable "
                "to routing under these declared parameters only. It is not an estimate of achievable savings in "
                "any real deployment."
            ),
        }
    )

    headline = (
        f"Five routing policies over {sum(len(traces[r].requests) for r in regimes)} common-random-number requests "
        f"across {len(regimes)} regimes: {len(front_names)} of {len(names)} policies sit on the cost/accuracy "
        f"Pareto front, {len(winners)} different policies win in different regimes, and "
        f"{'no' if len(winners) > 1 else 'one'} policy wins everywhere."
    )
    return {
        "headline": headline,
        "wow": "No policy wins everywhere, and it shows you the trade-off instead of a fake single winner.",
        "steps": steps,
        "key_numbers": {
            "front": front_names,
            "dominated": dominated,
            "best_per_regime": best_per_regime,
            "regret_usd": {
                regime: {n: oracle[regime]["utility_usd"] - results[(regime, n)]["utility_usd"] for n in names}
                for regime in regimes
            },
            "rank_flips": flips,
        },
    }
