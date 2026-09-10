"""Symbolic claims as hard constraints on a learned or composed model.

A symbolic rule is a constraint on admissible states, not a component that produces them.
This module makes such rules executable in three ways: `project` maps a predicted state
onto the feasible set so a learned component cannot emit a state that violates a stated
invariant; `violation_report` measures what a trajectory did violate and by how much; and
`penalty_terms` turns the same declarations into soft residuals for use inside a fitting
objective.

Projection magnitude is reported at every step, and deliberately so. A projection that has
to move the state a long way is not a success - it is the measurement of a learned
component fighting the physics it was supposed to respect, and it belongs in the record
next to the corrected trajectory rather than being silently absorbed by it.

The constraints here are the caller's declarations. Enforcing a stated conservation law
does not establish that the quantity is conserved in the real system, and a satisfied
constraint set is not a validated model.
"""

import math

import numpy as np

from symplex.core.contracts import Invalid

MAX_DIMENSION = 256
MAX_CONSTRAINTS = 64
MAX_PROJECTION_PASSES = 64
MAX_TRAJECTORY_ROWS = 200_000

KINDS = (
    "conservation",
    "non_negativity",
    "bounds",
    "monotonicity",
    "symmetry",
    "stoichiometry",
    "dimensional",
)
# `dimensional` is a declaration hook: unit algebra belongs to the modelling contract, and
# nothing here checks it numerically. It is carried through and reported as deferred.
DEFERRED_KINDS = ("dimensional",)

SCOPE = (
    "Enforcement of constraints the caller declared. Satisfying them is necessary, never "
    "sufficient: a state can respect every stated invariant and still be wrong, and an "
    "invariant that was never declared is not checked."
)


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


def _indices(value, label, dimension):
    if not isinstance(value, (list, tuple)) or not value:
        raise Invalid(f"{label} must be a nonempty list of state indices")
    resolved = [_integer(i, label, 0, dimension - 1) for i in value]
    if len(set(resolved)) != len(resolved):
        raise Invalid(f"{label} must not repeat a state index")
    return resolved


def _state(values, dimension, label="state"):
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size != dimension or not np.all(np.isfinite(array)):
        raise Invalid(f"{label} must hold {dimension} finite values")
    return array


def _limit_vector(value, count, label):
    if value is None:
        return None
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 1:
        array = np.repeat(array, count)
    if array.size != count or not np.all(np.isfinite(array)):
        raise Invalid(f"{label} must be one finite value, or one per constrained index")
    return array


def _normalize(constraint, dimension, position):
    if not isinstance(constraint, dict):
        raise Invalid("Each constraint must be a dict")
    kind = constraint.get("kind")
    if kind not in KINDS:
        raise Invalid("constraint kind must be one of: " + ", ".join(KINDS))
    identifier = constraint.get("id", f"{kind}_{position}")
    if not isinstance(identifier, str) or not identifier.strip() or len(identifier) > 64:
        raise Invalid("constraint id must be a short nonempty string")
    tolerance = _number(constraint.get("tolerance", 0.0), "tolerance", low=0.0)
    normalized = {
        "id": identifier.strip(),
        "kind": kind,
        "tolerance": tolerance,
        "rationale": constraint.get("rationale", ""),
    }
    if not isinstance(normalized["rationale"], str) or len(normalized["rationale"]) > 1600:
        raise Invalid("constraint rationale must be a string of at most 1600 characters")

    if kind == "conservation":
        normalized["indices"] = _indices(constraint.get("indices"), "indices", dimension)
        normalized["total"] = _number(constraint.get("total"), "total")
    elif kind == "non_negativity":
        normalized["indices"] = _indices(constraint.get("indices"), "indices", dimension)
    elif kind == "bounds":
        indices = _indices(constraint.get("indices"), "indices", dimension)
        lower = _limit_vector(constraint.get("lower"), len(indices), "lower")
        upper = _limit_vector(constraint.get("upper"), len(indices), "upper")
        if lower is None and upper is None:
            raise Invalid("A bounds constraint needs a lower or an upper limit")
        if lower is not None and upper is not None and np.any(lower > upper):
            raise Invalid("A bounds constraint has a lower limit above its upper limit")
        normalized.update({"indices": indices, "lower": lower, "upper": upper})
    elif kind == "monotonicity":
        normalized["index"] = _integer(constraint.get("index"), "index", 0, dimension - 1)
        direction = constraint.get("direction")
        if direction not in ("increasing", "decreasing"):
            raise Invalid("monotonicity direction must be increasing or decreasing")
        normalized["direction"] = direction
    elif kind == "symmetry":
        pairs = constraint.get("pairs")
        if not isinstance(pairs, (list, tuple)) or not pairs:
            raise Invalid("symmetry needs a nonempty list of index pairs")
        resolved = []
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise Invalid("Each symmetry pair must hold exactly two state indices")
            left = _integer(pair[0], "symmetry index", 0, dimension - 1)
            right = _integer(pair[1], "symmetry index", 0, dimension - 1)
            if left == right:
                raise Invalid("A symmetry pair must name two distinct states")
            resolved.append((left, right))
        normalized["pairs"] = resolved
    elif kind == "stoichiometry":
        matrix = np.asarray(constraint.get("matrix"), dtype=float)
        if matrix.ndim != 2 or matrix.size == 0 or not np.all(np.isfinite(matrix)):
            raise Invalid("stoichiometry needs a finite 2-D matrix")
        if matrix.shape[1] != dimension:
            raise Invalid("stoichiometry matrix must have one column per state")
        if matrix.shape[0] > dimension:
            raise Invalid("stoichiometry matrix has more invariants than states")
        totals = np.asarray(constraint.get("totals"), dtype=float).reshape(-1)
        if totals.size != matrix.shape[0] or not np.all(np.isfinite(totals)):
            raise Invalid("stoichiometry totals must hold one finite value per matrix row")
        normalized.update({"matrix": matrix, "totals": totals})
    else:  # dimensional
        normalized["indices"] = _indices(constraint.get("indices"), "indices", dimension)
        dimension_name = constraint.get("dimension")
        unit = constraint.get("unit", "")
        if not isinstance(dimension_name, str) or not dimension_name.strip():
            raise Invalid("A dimensional declaration needs a dimension name")
        if not isinstance(unit, str) or len(unit) > 64:
            raise Invalid("A dimensional unit must be a short string")
        normalized.update({"dimension": dimension_name.strip(), "unit": unit.strip()})
    return normalized


def validate_constraints(constraints, dimension):
    """Normalize a declarative constraint list and report which kinds are enforceable."""
    dimension = _integer(dimension, "dimension", 1, MAX_DIMENSION)
    if not isinstance(constraints, (list, tuple)):
        raise Invalid("constraints must be a list")
    if len(constraints) > MAX_CONSTRAINTS:
        raise Invalid(f"At most {MAX_CONSTRAINTS} constraints may be declared")
    normalized = [_normalize(c, dimension, i) for i, c in enumerate(constraints)]
    identifiers = [c["id"] for c in normalized]
    if len(set(identifiers)) != len(identifiers):
        raise Invalid("Constraint ids must be unique")
    counts = {kind: sum(1 for c in normalized if c["kind"] == kind) for kind in KINDS}
    return {
        "constraints": normalized,
        "dimension": dimension,
        "ids": identifiers,
        "counts_by_kind": counts,
        "pointwise_enforceable": [
            c["id"] for c in normalized if c["kind"] not in DEFERRED_KINDS and c["kind"] != "monotonicity"
        ],
        "trajectory_only": [c["id"] for c in normalized if c["kind"] == "monotonicity"],
        "deferred": [
            {"id": c["id"], "kind": c["kind"], "reason": "Declared only; no numeric check is performed here."}
            for c in normalized
            if c["kind"] in DEFERRED_KINDS
        ],
        "scope": SCOPE,
    }


def _state_violations(state, constraint):
    """Nonnegative violation amounts for one constraint at one state."""
    kind = constraint["kind"]
    tolerance = constraint["tolerance"]
    if kind == "conservation":
        gap = abs(float(state[constraint["indices"]].sum()) - constraint["total"])
        return np.array([max(0.0, gap - tolerance)])
    if kind == "non_negativity":
        return np.maximum(0.0, -state[constraint["indices"]] - tolerance)
    if kind == "bounds":
        values = state[constraint["indices"]]
        pieces = []
        if constraint["lower"] is not None:
            pieces.append(np.maximum(0.0, constraint["lower"] - values - tolerance))
        if constraint["upper"] is not None:
            pieces.append(np.maximum(0.0, values - constraint["upper"] - tolerance))
        return np.concatenate(pieces)
    if kind == "symmetry":
        gaps = [abs(state[a] - state[b]) for a, b in constraint["pairs"]]
        return np.maximum(0.0, np.asarray(gaps) - tolerance)
    if kind == "stoichiometry":
        gaps = np.abs(constraint["matrix"] @ state - constraint["totals"])
        return np.maximum(0.0, gaps - tolerance)
    return np.zeros(0)  # monotonicity and dimensional are not pointwise


def _project_once(state, constraint):
    kind = constraint["kind"]
    if kind == "non_negativity":
        state[constraint["indices"]] = np.maximum(state[constraint["indices"]], 0.0)
    elif kind == "bounds":
        indices = constraint["indices"]
        if constraint["lower"] is not None:
            state[indices] = np.maximum(state[indices], constraint["lower"])
        if constraint["upper"] is not None:
            state[indices] = np.minimum(state[indices], constraint["upper"])
    elif kind == "symmetry":
        for left, right in constraint["pairs"]:
            mean = 0.5 * (state[left] + state[right])
            state[left] = mean
            state[right] = mean
    elif kind == "conservation":
        indices = constraint["indices"]
        block = state[indices]
        current = float(block.sum())
        target = constraint["total"]
        if np.all(block >= 0.0) and current > 0.0 and target >= 0.0:
            # Proportional redistribution keeps every component nonnegative.
            state[indices] = block * (target / current)
        else:
            state[indices] = block + (target - current) / len(indices)
    elif kind == "stoichiometry":
        matrix, totals = constraint["matrix"], constraint["totals"]
        gap = totals - matrix @ state
        gram = matrix @ matrix.T
        try:
            correction = matrix.T @ np.linalg.solve(gram, gap)
        except np.linalg.LinAlgError:
            correction = matrix.T @ np.linalg.lstsq(gram, gap, rcond=None)[0]
        state += correction
    return state


def project(state, constraints, max_passes=16, tolerance=1e-12):
    """Project a predicted state onto the declared feasible set.

    Constraints are applied in a fixed order - box limits, then symmetry, then the linear
    invariants - and the sweep repeats until the largest violation stops improving. This is
    alternating projection: for convex constraint sets it converges to a feasible point,
    but not in general to the nearest one, and non-convex declarations are not guaranteed
    to converge at all. Non-convergence is reported.

    Non-negativity combined with a conservation constraint is handled as clipping followed
    by proportional redistribution over the conserved indices, so the mass removed by the
    clip is taken back from the components that have it rather than invented.

    `projection_magnitude` is the diagnostic that matters: a large value means the
    component that produced this state disagrees with the declared physics, and repairing
    the state does not repair that disagreement.
    """
    validated = validate_constraints(constraints, len(np.asarray(state, dtype=float).reshape(-1)))
    declared = validated["constraints"]
    dimension = validated["dimension"]
    original = _state(state, dimension)
    max_passes = _integer(max_passes, "max_passes", 1, MAX_PROJECTION_PASSES)
    tolerance = _number(tolerance, "tolerance", low=0.0)

    order = {"bounds": 0, "non_negativity": 1, "symmetry": 2, "conservation": 3, "stoichiometry": 4}
    active = sorted(
        [c for c in declared if c["kind"] in order],
        key=lambda c: order[c["kind"]],
    )
    current = original.copy()

    def worst(vector):
        values = [
            float(_state_violations(vector, c).max()) if _state_violations(vector, c).size else 0.0
            for c in active
        ]
        return max(values) if values else 0.0

    before = worst(current)
    passes, converged = 0, not active
    previous = before
    for passes in range(1, max_passes + 1):
        for constraint in active:
            current = _project_once(current, constraint)
        residual = worst(current)
        if residual <= tolerance:
            converged = True
            break
        if previous - residual <= 1e-15 * max(1.0, previous):
            break
        previous = residual
    after = worst(current)
    change = current - original
    magnitude = float(np.linalg.norm(change))
    reference = float(np.linalg.norm(original))
    per_constraint = []
    for constraint in declared:
        pre = _state_violations(original, constraint)
        post = _state_violations(current, constraint)
        per_constraint.append(
            {
                "id": constraint["id"],
                "kind": constraint["kind"],
                "violation_before": float(pre.max()) if pre.size else None,
                "violation_after": float(post.max()) if post.size else None,
                "enforced": constraint["kind"] in order,
            }
        )
    return {
        "state": [float(v) for v in current],
        "state_array": current,
        "projection": [float(v) for v in change],
        "projection_magnitude": magnitude,
        "relative_projection_magnitude": float(magnitude / reference) if reference > 0 else None,
        "max_absolute_component_change": float(np.max(np.abs(change))) if change.size else 0.0,
        "worst_violation_before": before,
        "worst_violation_after": after,
        "passes": int(passes),
        "converged": bool(converged or after <= tolerance),
        "per_constraint": per_constraint,
        "not_enforced_pointwise": validated["trajectory_only"],
        "deferred": validated["deferred"],
        "scope": SCOPE
        + " Projection repairs the state, not the model that produced it; the reported "
        "magnitude is the size of that disagreement.",
    }


def project_trajectory(trajectory, constraints, max_passes=16, tolerance=1e-12):
    """Project every row of a trajectory, then enforce monotonicity along time.

    Monotonicity is not a pointwise property, so it is applied after the per-row sweep as
    a running maximum or minimum. That enforcement is one-sided: it can only remove
    reversals by dragging later values, and it therefore biases the trajectory in the
    declared direction. The magnitude of that adjustment is reported separately.
    """
    array = np.asarray(trajectory, dtype=float)
    if array.ndim != 2 or array.size == 0 or not np.all(np.isfinite(array)):
        raise Invalid("trajectory must be a nonempty finite 2-D array")
    if array.shape[0] > MAX_TRAJECTORY_ROWS:
        raise Invalid(f"trajectory exceeds the {MAX_TRAJECTORY_ROWS}-row limit")
    validated = validate_constraints(constraints, array.shape[1])
    rows, magnitudes, converged = [], [], True
    for row in array:
        result = project(row, constraints, max_passes=max_passes, tolerance=tolerance)
        rows.append(result["state_array"])
        magnitudes.append(result["projection_magnitude"])
        converged = converged and result["converged"]
    projected = np.vstack(rows)
    monotone_adjustment = 0.0
    for constraint in validated["constraints"]:
        if constraint["kind"] != "monotonicity":
            continue
        column = projected[:, constraint["index"]].copy()
        enforced = (
            np.maximum.accumulate(column)
            if constraint["direction"] == "increasing"
            else np.minimum.accumulate(column)
        )
        monotone_adjustment = max(monotone_adjustment, float(np.max(np.abs(enforced - column))))
        projected[:, constraint["index"]] = enforced
    return {
        "trajectory": projected,
        "step_projection_magnitude": [float(v) for v in magnitudes],
        "total_projection_magnitude": float(np.linalg.norm(projected - array)),
        "mean_step_projection_magnitude": float(np.mean(magnitudes)),
        "max_step_projection_magnitude": float(np.max(magnitudes)),
        "monotonicity_adjustment": monotone_adjustment,
        "converged": bool(converged),
        "n_steps": int(array.shape[0]),
        "deferred": validated["deferred"],
        "scope": SCOPE
        + " Monotonicity enforcement is a one-sided repair applied after the pointwise sweep "
        "and biases the series in the declared direction.",
    }


def violation_report(trajectory, constraints):
    """Per-constraint maximum and mean violation over a trajectory."""
    array = np.asarray(trajectory, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.size == 0 or not np.all(np.isfinite(array)):
        raise Invalid("trajectory must be a nonempty finite 2-D array")
    if array.shape[0] > MAX_TRAJECTORY_ROWS:
        raise Invalid(f"trajectory exceeds the {MAX_TRAJECTORY_ROWS}-row limit")
    validated = validate_constraints(constraints, array.shape[1])
    entries = []
    for constraint in validated["constraints"]:
        kind = constraint["kind"]
        if kind in DEFERRED_KINDS:
            entries.append(
                {
                    "id": constraint["id"],
                    "kind": kind,
                    "checked": False,
                    "reason": "Declared only; no numeric check is performed here.",
                    "max_violation": None,
                    "mean_violation": None,
                    "violating_steps": None,
                    "first_violating_step": None,
                }
            )
            continue
        if kind == "monotonicity":
            column = array[:, constraint["index"]]
            differences = np.diff(column)
            signed = -differences if constraint["direction"] == "increasing" else differences
            per_step = np.maximum(0.0, signed - constraint["tolerance"])
        else:
            per_step = np.array(
                [float(_state_violations(row, constraint).max() if _state_violations(row, constraint).size else 0.0) for row in array]
            )
        violating = per_step > 0
        entries.append(
            {
                "id": constraint["id"],
                "kind": kind,
                "checked": True,
                "max_violation": float(per_step.max()) if per_step.size else 0.0,
                "mean_violation": float(per_step.mean()) if per_step.size else 0.0,
                "violating_steps": int(violating.sum()),
                "fraction_violating": float(violating.mean()) if per_step.size else 0.0,
                "first_violating_step": int(np.argmax(violating)) if violating.any() else None,
                "tolerance": constraint["tolerance"],
            }
        )
    checked = [e for e in entries if e["checked"]]
    worst = max(checked, key=lambda e: e["max_violation"], default=None)
    return {
        "constraints": entries,
        "any_violation": bool(any(e["max_violation"] > 0 for e in checked)),
        "worst_constraint": None if worst is None else worst["id"],
        "worst_violation": None if worst is None else worst["max_violation"],
        "n_steps": int(array.shape[0]),
        "n_checked": len(checked),
        "n_deferred": len(entries) - len(checked),
        "scope": SCOPE
        + " Violations are measured only at the supplied samples; behaviour between them is "
        "not observed.",
    }


def penalty_terms(constraints, dimension, weights=None):
    """Soft-constraint residuals for use inside a fitting objective.

    Returns callables rather than numbers because the residual depends on the state being
    fitted. A soft penalty trades constraint satisfaction against data fit at whatever
    exchange rate the weights imply; it does not guarantee feasibility, and where
    feasibility is required `project` is the mechanism, not this.
    """
    validated = validate_constraints(constraints, dimension)
    declared = [c for c in validated["constraints"] if c["kind"] not in DEFERRED_KINDS]
    if weights is None:
        weight_map = {c["id"]: 1.0 for c in declared}
    else:
        if not isinstance(weights, dict):
            raise Invalid("weights must be a dict of constraint id to weight")
        weight_map = {}
        for constraint in declared:
            weight_map[constraint["id"]] = _number(
                weights.get(constraint["id"], 1.0), "weight", low=0.0
            )

    def residual(values):
        array = np.asarray(values, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != validated["dimension"]:
            raise Invalid("residual input must have one column per declared state")
        if not np.all(np.isfinite(array)):
            raise Invalid("residual input must be finite")
        pieces = []
        for constraint in declared:
            weight = math.sqrt(weight_map[constraint["id"]])
            if constraint["kind"] == "monotonicity":
                column = array[:, constraint["index"]]
                differences = np.diff(column)
                signed = -differences if constraint["direction"] == "increasing" else differences
                pieces.append(weight * np.maximum(0.0, signed - constraint["tolerance"]))
                continue
            for row in array:
                block = _state_violations(row, constraint)
                if block.size:
                    pieces.append(weight * block)
        return np.concatenate(pieces) if pieces else np.zeros(0)

    def penalty(values):
        block = residual(values)
        return float(block @ block)

    return {
        "residual": residual,
        "penalty": penalty,
        "constraint_ids": [c["id"] for c in declared],
        "weights": {k: float(v) for k, v in weight_map.items()},
        "n_soft_constraints": len(declared),
        "deferred": validated["deferred"],
        "note": (
            "residual(x) returns nonnegative one-sided violations, so equality constraints "
            "contribute |g(x)| - tolerance and inequality constraints contribute zero when "
            "satisfied. The residual length depends on the number of rows supplied."
        ),
        "scope": SCOPE
        + " A soft penalty biases a fit towards feasibility; it does not enforce it, and the "
        "weights are an unvalidated exchange rate against the data term.",
    }
