"""Universal Differential Equations: keep the mechanism, learn only the missing term.

Rackauckas, Ma, Martensen, Warner, Zubov, Supekar, Skinner, Ramadhan and Edelman,
"Universal Differential Equations for Scientific Machine Learning", arXiv:2001.04385, 2020.

The composed right-hand side is

    dx/dt = f_known(t, x, theta) + g_learned(x, w)

where `f_known` is caller-supplied mechanism that is retained unchanged and `g_learned`
is a small numpy network fitted to whatever the mechanism does not account for. This is
the honest hybrid: it does not discard the known physics in order to fit the data, and
the size of the learned term relative to the known term is reported so that a closure
which has quietly learned to override the mechanism is visible rather than hidden inside
a good fit.

`symbolic_recovery` closes the neurosymbolic loop: the fitted closure is passed through
the sparse-regression machinery in `symplex.hybrid.sindy` and returned as a candidate
symbolic expression a scientist can read, argue with and reject. What comes back is a
sparse fit to a network that was itself fitted to residuals of the supplied data. Two
layers of inference sit between it and the system; it is a hypothesis to test, not a
discovered law.

Pure numeric surface: arrays and caller-supplied callables in, dicts out.
"""

import math

import numpy as np
from scipy.optimize import minimize

from symplex.core.contracts import Invalid
from symplex.hybrid.sindy import (
    _check_names,
    default_state_names,
    estimate_derivatives,
    fit_sindy,
    render_equation,
    stability_selection,
)
from symplex.hybrid.surrogate import (
    MAX_HIDDEN_LAYERS,
    MAX_OPTIMIZER_ITERATIONS,
    NeuralSurrogate,
)

MAX_TRAJECTORY_POINTS = 20_000
MAX_STATE_DIMENSION = 12
MAX_SEGMENTS = 32
# Multiple-shooting refinement uses numerical gradients (f_known is caller-supplied and
# not differentiable by this module), so the free-parameter count is capped hard.
MAX_SHOOTING_PARAMETERS = 400
MAX_SYMBOLIC_SAMPLES = 5_000
DIVERGENCE_NORM = 1e12

SCOPE = (
    "A composed model fitted to one trajectory set. The learned closure absorbs every "
    "source of mismatch present in those data - missing mechanism, parameter error, "
    "derivative-estimation bias and observation noise together - and does not attribute "
    "the mismatch to any of them."
)


def _matrix(values, label, max_rows=MAX_TRAJECTORY_POINTS, max_cols=MAX_STATE_DIMENSION):
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.size == 0:
        raise Invalid(f"{label} must be a nonempty 1-D or 2-D numeric array")
    if not np.all(np.isfinite(array)):
        raise Invalid(f"{label} contains nonfinite values")
    if array.shape[0] > max_rows:
        raise Invalid(f"{label} exceeds the {max_rows}-row limit")
    if array.shape[1] > max_cols:
        raise Invalid(f"{label} exceeds the {max_cols}-column limit")
    return array


def _integer(value, label, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise Invalid(f"{label} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise Invalid(f"{label} must lie in [{low}, {high}]")
    return value


def _number(value, label, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise Invalid(f"{label} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise Invalid(f"{label} must be finite")
    if low is not None and value < low:
        raise Invalid(f"{label} must be at least {low}")
    if high is not None and value > high:
        raise Invalid(f"{label} must be at most {high}")
    return value


def _increasing_grid(t, minimum=2):
    times = np.asarray(t, dtype=float).reshape(-1)
    if times.size < minimum or not np.all(np.isfinite(times)):
        raise Invalid(f"t must hold at least {minimum} finite sample times")
    if times.size > MAX_TRAJECTORY_POINTS:
        raise Invalid(f"t exceeds the {MAX_TRAJECTORY_POINTS}-sample limit")
    steps = np.diff(times)
    if np.any(steps <= 0):
        raise Invalid("t must be strictly increasing")
    return times, steps


def _uniform_grid(t):
    times, steps = _increasing_grid(t, minimum=4)
    if not np.allclose(steps, steps[0], rtol=1e-6, atol=1e-12):
        raise Invalid(
            "Multiple-shooting refinement requires a uniform time grid; resample or supply "
            "derivatives and use the collocation stage alone"
        )
    return times, float(steps[0])


def _wrap_known(f_known, theta, n_states, vectorized):
    if not callable(f_known):
        raise Invalid("f_known must be callable as f_known(t, x, theta)")

    def known(time, batch):
        if vectorized:
            value = np.asarray(f_known(time, batch, theta), dtype=float)
        else:
            value = np.stack(
                [np.asarray(f_known(time, row, theta), dtype=float).reshape(-1) for row in batch]
            )
        if value.shape != batch.shape:
            raise Invalid(
                "f_known must return an array shaped like its state argument; set "
                "vectorized=False if it only accepts a single state vector"
            )
        return value

    return known


def _rk4(rhs, times, initial):
    """Fixed-step RK4 over the supplied grid, batched over the leading axis.

    A fixed step is used deliberately: an adaptive solver would make the objective a
    non-smooth function of the parameters and the fit irreproducible across platforms.
    Accuracy is therefore the caller's responsibility through the sampling grid.
    """
    states = np.empty((times.size, *initial.shape), dtype=float)
    states[0] = initial
    diverged_at = None
    for index in range(times.size - 1):
        h = times[index + 1] - times[index]
        current = states[index]
        k1 = rhs(times[index], current)
        k2 = rhs(times[index] + h / 2.0, current + h / 2.0 * k1)
        k3 = rhs(times[index] + h / 2.0, current + h / 2.0 * k2)
        k4 = rhs(times[index] + h, current + h * k3)
        nxt = current + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.all(np.isfinite(nxt)) or np.max(np.abs(nxt)) > DIVERGENCE_NORM:
            diverged_at = index + 1
            states[index + 1 :] = current
            break
        states[index + 1] = nxt
    return states, diverged_at


def simulate_ude(t, x0, f_known, closure=None, weights=None, theta=None, closure_targets=None, vectorized=True):
    """Integrate f_known plus an optional learned closure with fixed-step RK4.

    The grid must increase but need not be uniform; the step of the returned solution is
    the spacing of the supplied times, so truncation error is the caller's choice.
    """
    times, _steps = _increasing_grid(t)
    initial = np.asarray(x0, dtype=float)
    single = initial.ndim == 1
    batch = initial.reshape(1, -1) if single else initial
    if batch.ndim != 2 or batch.size == 0 or not np.all(np.isfinite(batch)):
        raise Invalid("x0 must be a finite state vector or a batch of them")
    n_states = batch.shape[1]
    known = _wrap_known(f_known, theta, n_states, vectorized)
    evaluate = _closure_evaluator(closure, weights, closure_targets, n_states)

    def rhs(time, state):
        return known(time, state) + evaluate(state)

    states, diverged_at = _rk4(rhs, times, batch)
    trajectory = states[:, 0, :] if single else states
    return {
        "times": times,
        "trajectory": trajectory,
        "diverged": diverged_at is not None,
        "diverged_at_index": diverged_at,
        "n_steps": int(times.size - 1),
        "integrator": "fixed-step RK4",
        "scope": (
            "A deterministic integration of the supplied right-hand side. Truncation error is "
            "set by the sampling grid and is not estimated here; divergence is reported rather "
            "than clipped away."
        ),
    }


def _closure_evaluator(closure, weights, closure_targets, n_states):
    if closure is None:
        return lambda state: 0.0
    if not isinstance(closure, NeuralSurrogate):
        raise Invalid("closure must be a NeuralSurrogate")
    targets = list(range(n_states)) if closure_targets is None else list(closure_targets)
    if len(targets) != closure.n_outputs:
        raise Invalid("closure_targets must name one state per closure output")

    def evaluate(state):
        if weights is None:
            values = closure.forward(state)
        else:
            values = closure.forward_with_weights(state, weights)
        full = np.zeros((state.shape[0], n_states))
        full[:, targets] = values
        return full

    return evaluate


def _trajectory_error(times, X_obs, f_known, theta, closure, weights, targets, vectorized):
    result = simulate_ude(
        times,
        X_obs[0],
        f_known,
        closure=closure,
        weights=weights,
        theta=theta,
        closure_targets=targets,
        vectorized=vectorized,
    )
    predicted = result["trajectory"]
    residual = predicted - X_obs
    finite = np.all(np.isfinite(residual), axis=1)
    rmse = float(np.sqrt(np.mean(residual[finite] ** 2))) if finite.any() else None
    return {
        "rmse": rmse,
        "diverged": result["diverged"],
        "diverged_at_index": result["diverged_at_index"],
        "trajectory": predicted,
    }


def fit_ude(
    t,
    X_obs,
    f_known,
    theta=None,
    closure_targets=None,
    hidden=(8,),
    activation="tanh",
    weight_decay=1e-6,
    seed=2026,
    n_segments=4,
    continuity_weight=1.0,
    refine=True,
    collocation_iterations=400,
    refine_iterations=60,
    derivative_method="finite_difference",
    window=None,
    polyorder=3,
    state_names=None,
    vectorized=True,
    autonomous=True,
):
    """Fit the learned closure of a Universal Differential Equation.

    Two stages, both reported separately so that a good final number cannot conceal a
    failed stage:

    1. **Collocation.** Estimate dx/dt from the samples, subtract the known mechanism, and
       regress the closure directly on the residual with the network's analytic gradient.
       This is fast and well conditioned, but it inherits every bias of the derivative
       estimator and never actually integrates the composed model.
    2. **Multiple-shooting refinement.** Split the trajectory into `n_segments` contiguous
       segments, give each its own free initial state, integrate each independently, and
       penalize both the data mismatch and the discontinuity at segment joins. Single
       shooting is not used: on chaotic or stiff systems the sensitivity of the final
       state to the initial state grows exponentially with the horizon, so a single-shot
       objective has a nearly flat basin and enormous cliffs, and gradient-based fitting
       fails long before the model does. Segmenting caps that sensitivity at one segment
       length, at the cost of the continuity residual, which is reported.

    The refinement uses numerical gradients because `f_known` is a caller-supplied callable
    that this module cannot differentiate; the free-parameter count is capped accordingly
    (MAX_SHOOTING_PARAMETERS) and the iteration budget is explicit. If the budget is
    exhausted the result says so instead of presenting the last iterate as converged.

    The returned dict carries the live `NeuralSurrogate` under `closure`, because
    `symbolic_recovery` needs it; `nonserializable_keys` names it so a caller that persists
    the result can drop it mechanically.

    Segments are integrated as one batch, which assumes an autonomous `f_known`. Pass
    `autonomous=False` for a time-dependent mechanism: the collocation stage then evaluates
    the mechanism at each sample's own time and the refinement is skipped with a stated
    reason rather than run on the wrong clock.
    """
    times, step = _uniform_grid(t)
    X_obs = _matrix(X_obs, "X_obs")
    if X_obs.shape[0] != times.size:
        raise Invalid("X_obs must hold one row per sample time")
    n_samples, n_states = X_obs.shape
    names = _check_names(state_names, n_states)
    targets = list(range(n_states)) if closure_targets is None else [
        _integer(i, "closure target", 0, n_states - 1) for i in closure_targets
    ]
    if not targets or len(set(targets)) != len(targets):
        raise Invalid("closure_targets must be unique state indices")
    n_segments = _integer(n_segments, "n_segments", 1, MAX_SEGMENTS)
    continuity_weight = _number(continuity_weight, "continuity_weight", low=0.0, high=1e6)
    refine_iterations = _integer(refine_iterations, "refine_iterations", 1, MAX_OPTIMIZER_ITERATIONS)
    collocation_iterations = _integer(
        collocation_iterations, "collocation_iterations", 1, MAX_OPTIMIZER_ITERATIONS
    )
    if not isinstance(refine, bool) or not isinstance(vectorized, bool):
        raise Invalid("refine and vectorized must be booleans")
    if len(tuple(hidden)) > MAX_HIDDEN_LAYERS:
        raise Invalid(f"hidden may name at most {MAX_HIDDEN_LAYERS} layers")

    if not isinstance(autonomous, bool):
        raise Invalid("autonomous must be a boolean")
    known = _wrap_known(f_known, theta, n_states, vectorized)
    warnings = []

    # Stage 1: collocation on the mechanism residual.
    estimate = estimate_derivatives(
        X_obs, t=times, method=derivative_method, window=window, polyorder=polyorder
    )
    trim = estimate["suggested_edge_trim"]
    states = estimate["smoothed_states"][trim : n_samples - trim]
    derivatives = estimate["derivatives"][trim : n_samples - trim]
    sample_times = times[trim : n_samples - trim]
    if autonomous:
        known_values = known(float(times[trim]), states)
    else:
        known_values = np.vstack(
            [known(float(sample_times[i]), states[i : i + 1]) for i in range(states.shape[0])]
        )
    residual = derivatives - known_values
    closure = NeuralSurrogate(
        hidden=tuple(hidden),
        activation=activation,
        weight_decay=weight_decay,
        seed=seed,
        max_iterations=collocation_iterations,
    )
    collocation = closure.fit(states, residual[:, targets])
    warnings.extend(collocation["warnings"])
    collocation_weights = closure.weights.copy()

    baseline = _trajectory_error(times, X_obs, f_known, theta, None, None, targets, vectorized)
    after_collocation = _trajectory_error(
        times, X_obs, f_known, theta, closure, collocation_weights, targets, vectorized
    )

    # Stage 2: multiple shooting.
    steps_per_segment = (n_samples - 1) // n_segments
    shooting = {
        "performed": False,
        "reason": "",
        "n_segments": n_segments,
        "steps_per_segment": int(steps_per_segment),
        "n_free_parameters": None,
        "converged": None,
        "continuity_rmse": None,
        "data_rmse": None,
        "iterations": None,
        "objective_before": None,
        "objective_after": None,
    }
    final_weights = collocation_weights
    if refine and not autonomous:
        shooting["reason"] = (
            "Batched multiple shooting assumes a time-invariant mechanism; refinement is skipped "
            "for a non-autonomous f_known rather than integrated on the wrong clock."
        )
        warnings.append(shooting["reason"])
    elif refine and steps_per_segment >= 2:
        n_free = closure.n_parameters + n_segments * n_states
        if n_free > MAX_SHOOTING_PARAMETERS:
            shooting["reason"] = (
                f"{n_free} free parameters exceeds the {MAX_SHOOTING_PARAMETERS} limit for a "
                "numerically differentiated multiple-shooting objective; reduce the network or "
                "the number of segments"
            )
            warnings.append(shooting["reason"])
        else:
            used = n_segments * steps_per_segment + 1
            starts = np.arange(n_segments) * steps_per_segment
            index_grid = starts[None, :] + np.arange(steps_per_segment + 1)[:, None]
            observed = X_obs[index_grid]  # (steps+1, segments, states)
            segment_times = times[: steps_per_segment + 1]

            def objective(vector):
                weights = vector[: closure.n_parameters]
                anchors = vector[closure.n_parameters :].reshape(n_segments, n_states)
                evaluate = _closure_evaluator(closure, weights, targets, n_states)

                def rhs(time, state):
                    return known(time, state) + evaluate(state)

                predicted, diverged_at = _rk4(rhs, segment_times, anchors)
                if diverged_at is not None:
                    return 1e12
                data = float(np.mean((predicted - observed) ** 2))
                if n_segments > 1:
                    gap = predicted[-1, :-1, :] - anchors[1:, :]
                    continuity = float(np.mean(gap * gap))
                else:
                    continuity = 0.0
                penalty = weight_decay * float(weights @ weights)
                return data + continuity_weight * continuity + penalty

            start_vector = np.concatenate([collocation_weights, X_obs[starts].reshape(-1)])
            before = objective(start_vector)
            result = minimize(
                objective,
                start_vector,
                method="L-BFGS-B",
                options={"maxiter": refine_iterations},
            )
            budget_hit = int(result.nit) >= refine_iterations
            if float(result.fun) < before:
                final_weights = np.asarray(result.x[: closure.n_parameters], dtype=float)
                anchors = np.asarray(result.x[closure.n_parameters :], dtype=float).reshape(
                    n_segments, n_states
                )
            else:
                anchors = X_obs[starts]
                warnings.append(
                    "Multiple-shooting refinement did not improve on the collocation fit; the "
                    "collocation weights are retained."
                )
            evaluate = _closure_evaluator(closure, final_weights, targets, n_states)
            predicted, _diverged = _rk4(
                lambda time, state: known(time, state) + evaluate(state), segment_times, anchors
            )
            continuity_rmse = (
                float(np.sqrt(np.mean((predicted[-1, :-1, :] - anchors[1:, :]) ** 2)))
                if n_segments > 1
                else 0.0
            )
            shooting.update(
                {
                    "performed": True,
                    "reason": "",
                    "n_free_parameters": int(n_free),
                    "converged": bool(result.success) and not budget_hit,
                    "optimizer_message": str(result.message),
                    "iterations": int(result.nit),
                    "function_evaluations": int(result.nfev),
                    "objective_before": float(before),
                    "objective_after": float(result.fun),
                    "continuity_rmse": continuity_rmse,
                    "data_rmse": float(np.sqrt(np.mean((predicted - observed) ** 2))),
                    "samples_used": int(used),
                    "samples_dropped": int(n_samples - used),
                }
            )
            if budget_hit:
                warnings.append(
                    f"Multiple-shooting refinement stopped at its {refine_iterations}-iteration "
                    "budget; the reported closure is the last iterate, not a converged optimum."
                )
            if continuity_weight > 0 and continuity_rmse > 0:
                scale = float(np.sqrt(np.mean(X_obs**2))) or 1.0
                if continuity_rmse > 0.1 * scale:
                    warnings.append(
                        f"Segment continuity residual {continuity_rmse:.3g} is large relative to "
                        "the state scale; the segments have not been reconciled into one "
                        "trajectory and the fit should not be read as a single solution."
                    )
    elif refine and autonomous:
        shooting["reason"] = "Too few samples per segment to integrate; refinement skipped."
        warnings.append(shooting["reason"])
    elif not refine:
        shooting["reason"] = "Refinement disabled by the caller."

    closure.weights = final_weights
    final = _trajectory_error(
        times, X_obs, f_known, theta, closure, final_weights, targets, vectorized
    )
    contribution = closure_contribution(
        states,
        f_known,
        closure,
        theta=theta,
        closure_targets=targets,
        weights=final_weights,
        state_names=names,
        vectorized=vectorized,
        time=float(times[trim]),
    )
    if not autonomous:
        contribution["scope"] += (
            " The mechanism was evaluated at one time only, so this ratio describes that instant "
            "for a time-dependent f_known."
        )
    reduction = None
    if baseline["rmse"] and final["rmse"] is not None and baseline["rmse"] > 0:
        reduction = float(1.0 - final["rmse"] / baseline["rmse"])
    if contribution["closure_dominates"]:
        warnings.append(
            "The learned closure is larger than the known mechanism on at least one state; the "
            "composed model is being driven by the learned term rather than corrected by it."
        )
    return {
        "closure": closure,
        "closure_targets": targets,
        "weights": final_weights,
        "theta": None if theta is None else [float(v) for v in np.asarray(theta, dtype=float).reshape(-1)],
        "state_names": names,
        "times": times,
        "dt": step,
        "collocation": {
            key: value
            for key, value in collocation.items()
            if key in ("converged", "iterations", "final_objective", "train_rmse", "n_parameters", "architecture", "activation")
        },
        "collocation_residual_rms": [float(v) for v in np.sqrt(np.mean(residual**2, axis=0))],
        "multiple_shooting": shooting,
        "trajectory_rmse_known_only": baseline["rmse"],
        "trajectory_rmse_after_collocation": after_collocation["rmse"],
        "trajectory_rmse_hybrid": final["rmse"],
        "known_only_diverged": baseline["diverged"],
        "hybrid_diverged": final["diverged"],
        "error_reduction_fraction": reduction,
        "contribution": contribution,
        "derivative": {k: v for k, v in estimate.items() if k not in ("derivatives", "smoothed_states")},
        "seed": int(seed),
        "autonomous": bool(autonomous),
        "nonserializable_keys": ["closure"],
        "converged": bool(collocation["converged"] and (shooting["converged"] is not False)),
        "warnings": warnings,
        "scope": SCOPE
        + " Trajectory RMSE is measured against the same data used to fit the closure unless the "
        "caller held some out; it is a goodness-of-fit figure, not a validation.",
    }


def closure_contribution(
    X,
    f_known,
    closure,
    theta=None,
    closure_targets=None,
    weights=None,
    state_names=None,
    vectorized=True,
    time=0.0,
):
    """How large is the learned term next to the mechanism it is supposed to correct?

    A UDE whose closure dwarfs `f_known` has not corrected the mechanism, it has replaced
    it, and any interpretation of the retained physics is then unfounded. Reported per
    state as the ratio of root-mean-square magnitudes over the supplied states, plus the
    fraction of samples where the learned term is the larger of the two.
    """
    X = _matrix(X, "X")
    n_states = X.shape[1]
    names = _check_names(state_names, n_states)
    known = _wrap_known(f_known, theta, n_states, vectorized)
    known_values = known(float(time), X)
    evaluate = _closure_evaluator(closure, weights, closure_targets, n_states)
    learned_values = evaluate(X)
    known_rms = np.sqrt(np.mean(known_values**2, axis=0))
    learned_rms = np.sqrt(np.mean(learned_values**2, axis=0))
    ratios = learned_rms / np.where(known_rms > 1e-12, known_rms, np.nan)
    dominance = np.mean(np.abs(learned_values) > np.abs(known_values), axis=0)
    per_state = []
    for index, name in enumerate(names):
        ratio = ratios[index]
        per_state.append(
            {
                "state": name,
                "known_rms": float(known_rms[index]),
                "learned_rms": float(learned_rms[index]),
                "ratio": None if not np.isfinite(ratio) else float(ratio),
                "fraction_learned_larger": float(dominance[index]),
            }
        )
    finite_ratios = [entry["ratio"] for entry in per_state if entry["ratio"] is not None]
    return {
        "per_state": per_state,
        "max_ratio": max(finite_ratios) if finite_ratios else None,
        "mean_ratio": float(np.mean(finite_ratios)) if finite_ratios else None,
        "closure_dominates": bool(any(r > 1.0 for r in finite_ratios)),
        "n_samples": int(X.shape[0]),
        "scope": (
            "A magnitude comparison over the supplied states at a single evaluation time for "
            "time-dependent mechanisms. A small ratio means the closure is a correction here; "
            "it does not mean the retained mechanism is correct."
        ),
    }


def symbolic_recovery(
    ude_fit,
    X_samples=None,
    degree=2,
    include_trig=False,
    include_interactions=True,
    threshold=0.05,
    alpha=1e-8,
    custom_terms=None,
    max_samples=2_000,
    stability=False,
    n_bootstrap=30,
    seed=2026,
    precision=3,
):
    """Turn the fitted black-box closure back into a candidate symbolic expression.

    The closure is evaluated on states drawn from the fitting trajectory, and those exact,
    noise-free input/output pairs are handed to sequentially thresholded least squares.
    Because the targets are a deterministic function evaluation rather than an estimated
    derivative, the sparse regression here is clean; `symbolic_r_squared` therefore measures
    how well the symbolic form reproduces *the network*, which is a different and weaker
    claim than reproducing the system.

    What this establishes: that the closure the fit chose is well approximated, over the
    states visited by these data, by the returned sparse expression in the chosen library.
    What it does not establish: that the expression is the missing mechanism. The closure
    absorbed noise, derivative bias and parameter error along with any real missing term,
    and any of those can present as a clean symbolic form. Treat the result as a hypothesis
    with a stated discriminating test, and check it against held-out regimes.
    """
    if not isinstance(ude_fit, dict) or "closure" not in ude_fit:
        raise Invalid("symbolic_recovery expects the dict returned by fit_ude")
    closure = ude_fit["closure"]
    if not isinstance(closure, NeuralSurrogate) or not closure.fitted:
        raise Invalid("The UDE fit does not carry a fitted closure")
    targets = list(ude_fit["closure_targets"])
    names = list(ude_fit.get("state_names") or default_state_names(closure.n_dimensions))
    max_samples = _integer(max_samples, "max_samples", 20, MAX_SYMBOLIC_SAMPLES)

    if X_samples is None:
        source = np.asarray(closure.support.X, dtype=float)
    else:
        source = _matrix(X_samples, "X_samples")
    if source.shape[1] != closure.n_dimensions:
        raise Invalid("X_samples must match the closure input dimension")
    if source.shape[0] > max_samples:
        stride = int(np.ceil(source.shape[0] / max_samples))
        source = source[::stride]
    if source.shape[0] < 20:
        raise Invalid("At least 20 sample states are needed for symbolic recovery")

    values = closure.forward_with_weights(source, ude_fit["weights"])
    labels = [f"g_{names[index]}" for index in targets]
    recovered = fit_sindy(
        source,
        dXdt=values,
        degree=degree,
        include_trig=include_trig,
        include_interactions=include_interactions,
        threshold=threshold,
        alpha=alpha,
        state_names=names,
        custom_terms=custom_terms,
        target_names=labels,
        precision=precision,
    )
    magnitude = np.sqrt(np.mean(values**2, axis=0))
    relative = [
        float(recovered["rmse"][i] / magnitude[i]) if magnitude[i] > 1e-12 else None
        for i in range(len(labels))
    ]
    stability_report = None
    if stability:
        stability_report = stability_selection(
            source,
            dXdt=values,
            degree=degree,
            include_trig=include_trig,
            include_interactions=include_interactions,
            threshold=threshold,
            alpha=alpha,
            state_names=names,
            target_names=labels,
            custom_terms=custom_terms,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
    dominant = []
    coefficients = recovered["coefficient_matrix"]
    for index, label in enumerate(labels):
        column = coefficients[:, index]
        order = np.argsort(-np.abs(column))
        dominant.append(
            [
                {"term": recovered["term_names"][j], "coefficient": float(column[j])}
                for j in order
                if column[j] != 0.0
            ][:5]
        )
    equations = [
        render_equation(coefficients[:, i], recovered["term_names"], lhs=label, precision=precision)[
            "equation"
        ]
        for i, label in enumerate(labels)
    ]
    return {
        "equations": equations,
        "composed_equations": [
            f"d{names[targets[i]]}/dt = f_known_{names[targets[i]]}(x) + " + equations[i].split(" = ", 1)[1]
            for i in range(len(labels))
        ],
        "coefficients": recovered["coefficients"],
        "coefficient_matrix": coefficients,
        "term_names": recovered["term_names"],
        "target_names": labels,
        "closure_targets": targets,
        "active_terms": recovered["active_terms"],
        "active_term_count": recovered["active_term_count"],
        "dominant_terms": dominant,
        "symbolic_r_squared": recovered["r_squared"],
        "symbolic_rmse": recovered["rmse"],
        "relative_symbolic_rmse": relative,
        "closure_output_rms": [float(v) for v in magnitude],
        "condition_number": recovered["condition_number"],
        "n_sample_states": int(source.shape[0]),
        "threshold": recovered["threshold"],
        "degree": recovered["degree"],
        "converged": recovered["converged"],
        "stability": stability_report,
        "warnings": recovered["warnings"],
        "discriminating_test": (
            "Substitute the recovered expression for the network, refit any free coefficients on "
            "the same data, then compare both against a trajectory from an initial condition or "
            "parameter regime not used in the fit. A missing mechanism transfers; an absorbed "
            "noise or bias artefact does not."
        ),
        "scope": (
            "A sparse symbolic approximation of a fitted network, inside the chosen candidate "
            "library, over the states these data visited. R-squared here is agreement with the "
            "network, not with the system. This is a hypothesis about the missing mechanism and "
            "is not evidence that the mechanism is what the expression says."
        ),
    }
