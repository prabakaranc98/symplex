"""Protected evaluation when the evaluator is incomplete.

AlphaEvolve can hill-climb a scalar because its evaluator is machine-gradeable and
complete for its task. This project's is neither, so a single scalar is refused
here by construction. A board declares several non-redundant proxies with explicit
roles — one ``primary`` that may be optimized, ``protected`` proxies that must not
regress, ``diagnostic`` proxies that are watched and never optimized — and it must
name at least one decision-relevant dimension it does not measure at all. That
declaration is reproduced in every report, because it is the honest difference
between this and a graded benchmark.

Nothing in this module runs an investigation, calls a model or promotes anything.
It validates and aggregates arm records the host recorded elsewhere, and returns a
verdict that is at most *eligible for operator review*. Activation remains the
existing human review, scoped canary and rollback path in
``symplex.infrastructure.method_registry``.
"""

import math
from collections import defaultdict
from statistics import NormalDist
from typing import Literal

from pydantic import Field, model_validator

from symplex.core.contracts import Invalid, digest
from symplex.improvement.policy import _as_policy
from symplex.modeling.complex_system import ClosedContract, Identifier, Text

VERSION = "protected-incomplete-evaluator-v1"
SCOPE = (
    "A paired comparison of two frozen methods under an explicitly incomplete "
    "evaluator. Results describe the declared proxies on the supplied tasks only. "
    "No proxy is the objective, the board's unmeasured dimensions remain unmeasured, "
    "and a favourable verdict makes a candidate eligible for operator review under "
    "the existing stage gates — it never installs, activates or promotes anything."
)

MAX_PROXIES = 12
MAX_TASKS = 512
MAX_PEEKS = 2000
EPSILON = 1e-12

# Promotion is the risky act here: it changes how future investigations are run.
# Rolling a change back is cheap. So this deliberately inverts the clinical
# convention and spends less alpha on promotion than on demotion.
PROMOTION_ALPHA = 0.005
DEMOTION_ALPHA = 0.05


class ProxySpec(ClosedContract):
    name: Identifier
    role: Literal["primary", "protected", "diagnostic"]
    direction: Literal["higher_is_better", "lower_is_better"]
    measures: Text = Field(description="What this proxy stands in for. Two proxies with the same answer here are redundant.")
    minimum_detectable_effect: float = Field(gt=0.0)
    noise_sd: float = Field(gt=0.0, description="Pre-declared paired standard deviation. Estimating it from the comparison being judged voids the sequential guarantee.")


class EvaluationBoard(ClosedContract):
    board_id: str = Field(default="", max_length=64)
    label: Text
    proxies: list[ProxySpec] = Field(min_length=2, max_length=MAX_PROXIES)
    unmeasured_dimensions: list[Text] = Field(
        min_length=1,
        max_length=16,
        description="Decision-relevant dimensions this board does not measure. A board that claims none is claiming a complete evaluator, which this project does not have.",
    )
    population_families: list[Identifier] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def _roles(self):
        names = [p.name for p in self.proxies]
        if len(set(names)) != len(names):
            raise ValueError("Proxy names must be unique")
        measures = [" ".join(p.measures.lower().split()) for p in self.proxies]
        if len(set(measures)) != len(measures):
            raise ValueError("Two proxies measuring the same declared thing are redundant")
        primaries = [p.name for p in self.proxies if p.role == "primary"]
        if len(primaries) != 1:
            raise ValueError("Exactly one proxy may be primary; a board without one invites a scalar objective")
        if not any(p.role == "protected" for p in self.proxies):
            raise ValueError("A board needs at least one protected proxy, or 'must not regress' is vacuous")
        self.board_id = "board_" + digest({k: v for k, v in self.model_dump().items() if k != "board_id"})[:24]
        return self


class ArmRecord(ClosedContract):
    """One frozen arm's recorded outcome on one task. Measured elsewhere; only checked here."""

    proxies: dict[Identifier, float] = Field(max_length=MAX_PROXIES)
    action_count: int = Field(ge=0, le=512)
    checks_run: int = Field(ge=0, le=512)
    scope_breadth: int = Field(ge=0, le=512, description="Distinct evidence items or artifacts the arm actually referenced.")
    attempted: bool
    accounting_complete: bool
    observed_other_arm: bool = False
    caps: dict[Identifier, float] = Field(max_length=16)
    usage: dict[Identifier, float] = Field(max_length=16)
    settings_digest: str = Field(min_length=1, max_length=64)


class ReplayTask(ClosedContract):
    task_id: str = Field(min_length=1, max_length=64)
    task_family: Identifier
    stage: Literal["development", "regression", "holdout"]
    difficulty: Literal["standard", "hard"] = "standard"
    parent: ArmRecord
    child: ArmRecord


def _as_board(board):
    if isinstance(board, EvaluationBoard):
        return board
    if not isinstance(board, dict):
        raise Invalid("Expected an EvaluationBoard or its mapping")
    try:
        return EvaluationBoard.model_validate(board)
    except Exception as exc:
        raise Invalid("Invalid evaluation board: " + str(exc)[:600]) from None


def _as_tasks(tasks):
    if not isinstance(tasks, (list, tuple)) or not tasks:
        raise Invalid("A paired replay needs a nonempty task list")
    if len(tasks) > MAX_TASKS:
        raise Invalid("At most " + str(MAX_TASKS) + " replay tasks per comparison")
    parsed = []
    for item in tasks:
        if isinstance(item, ReplayTask):
            parsed.append(item)
            continue
        try:
            parsed.append(ReplayTask.model_validate(item))
        except Exception as exc:
            raise Invalid("Invalid replay task: " + str(exc)[:600]) from None
    ids = [t.task_id for t in parsed]
    if len(set(ids)) != len(ids):
        raise Invalid("Replay task identifiers must be unique; a repeated task is not fresh evidence")
    return parsed


def describe_board(board):
    board = _as_board(board)
    return {
        "version": VERSION,
        "board_id": board.board_id,
        "label": board.label,
        "primary": next(p.name for p in board.proxies if p.role == "primary"),
        "protected": sorted(p.name for p in board.proxies if p.role == "protected"),
        "diagnostic": sorted(p.name for p in board.proxies if p.role == "diagnostic"),
        "unmeasured_dimensions": list(board.unmeasured_dimensions),
        "scope": SCOPE,
    }


def _gain(spec, parent_value, child_value):
    return (child_value - parent_value) if spec.direction == "higher_is_better" else (parent_value - child_value)


def paired_replay(parent, child, tasks, caps, *, board=None):
    """Aggregate a frozen parent/child comparison on fresh tasks at matched settings and equal caps.

    This mirrors what ``symplex.evaluation.methods.evaluate_method`` records: equal
    per-task/arm caps, identical fixed settings, complete resource accounting and no
    arm seeing the other's outputs. It executes nothing itself; it checks the records.
    """
    parent, child = _as_policy(parent), _as_policy(child)
    parsed = _as_tasks(tasks)
    board = _as_board(board) if board is not None else None
    if not isinstance(caps, dict) or not caps or len(caps) > 16:
        raise Invalid("Equal per-task/arm caps must be a nonempty mapping of at most 16 resources")
    for value in caps.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise Invalid("Every cap must be a finite nonnegative number")
    caps = {str(k): float(v) for k, v in caps.items()}
    specs = {p.name: p for p in board.proxies} if board else {}
    excluded, pairs = [], []
    settings, blinded, accounted, equal_caps = set(), True, True, True
    for task in parsed:
        reasons = []
        for arm_name, arm in (("parent", task.parent), ("child", task.child)):
            if {k: float(v) for k, v in arm.caps.items()} != caps:
                reasons.append(arm_name + "_caps_differ_from_the_declared_equal_allocation")
                equal_caps = False
            if any(arm.usage.get(k, 0.0) > caps[k] + 1e-9 for k in caps):
                reasons.append(arm_name + "_usage_exceeds_its_cap")
                equal_caps = False
            if arm.observed_other_arm:
                reasons.append(arm_name + "_observed_the_other_arm")
                blinded = False
            if not arm.accounting_complete:
                reasons.append(arm_name + "_accounting_incomplete")
                accounted = False
            if specs and set(arm.proxies) != set(specs):
                reasons.append(arm_name + "_proxy_set_differs_from_the_board")
                accounted = False
            settings.add(arm.settings_digest)
        if reasons:
            excluded.append({"task_id": task.task_id, "reasons": sorted(set(reasons))})
        pairs.append({
            "task_id": task.task_id, "task_family": task.task_family, "stage": task.stage,
            "difficulty": task.difficulty, "usable": not reasons,
            "gains": {name: _gain(spec, task.parent.proxies.get(name, 0.0), task.child.proxies.get(name, 0.0))
                      for name, spec in specs.items()},
            "parent": {"action_count": task.parent.action_count, "checks_run": task.parent.checks_run,
                       "scope_breadth": task.parent.scope_breadth, "attempted": task.parent.attempted},
            "child": {"action_count": task.child.action_count, "checks_run": task.child.checks_run,
                      "scope_breadth": task.child.scope_breadth, "attempted": task.child.attempted},
        })
    settings_matched = len(settings) == 1
    usable = [p for p in pairs if p["usable"]]
    proxies = {}
    for name, spec in specs.items():
        deltas = [p["gains"][name] for p in usable]
        parent_values = [t.parent.proxies.get(name, 0.0) for t in parsed]
        child_values = [t.child.proxies.get(name, 0.0) for t in parsed]
        mean = (sum(deltas) / len(deltas)) if deltas else 0.0
        proxies[name] = {
            "role": spec.role, "direction": spec.direction, "measures": spec.measures,
            "minimum_detectable_effect": spec.minimum_detectable_effect, "noise_sd": spec.noise_sd,
            "parent_mean": (sum(parent_values) / len(parent_values)) if parent_values else 0.0,
            "child_mean": (sum(child_values) / len(child_values)) if child_values else 0.0,
            "paired_gains": deltas, "mean_gain": mean, "usable_task_count": len(deltas),
            "improved": mean > EPSILON, "regressed": mean < -EPSILON,
            "exceeds_minimum_detectable_effect": abs(mean) >= spec.minimum_detectable_effect,
        }
    primary = next((n for n, s in specs.items() if s.role == "primary"), None)
    by_family = defaultdict(lambda: {"task_count": 0, "primary_gain_sum": 0.0})
    by_difficulty = defaultdict(lambda: {"task_count": 0, "parent_attempted": 0, "child_attempted": 0})
    for pair in usable:
        row = by_family[pair["task_family"]]
        row["task_count"] += 1
        row["primary_gain_sum"] += pair["gains"].get(primary, 0.0) if primary else 0.0
    for pair in pairs:
        row = by_difficulty[pair["difficulty"]]
        row["task_count"] += 1
        row["parent_attempted"] += bool(pair["parent"]["attempted"])
        row["child_attempted"] += bool(pair["child"]["attempted"])
    holdout = [p for p in usable if p["stage"] == "holdout"]
    holdout_gain = {name: (sum(p["gains"][name] for p in holdout) / len(holdout)) if holdout else None
                    for name in specs}
    structure = {}
    for arm in ("parent", "child"):
        structure[arm] = {
            key: (sum(p[arm][key] for p in usable) / len(usable)) if usable else 0.0
            for key in ("action_count", "checks_run", "scope_breadth")
        }
        structure[arm]["attempted_tasks"] = sum(bool(p[arm]["attempted"]) for p in pairs)
    return {
        "version": VERSION,
        "parent_policy_id": parent.policy_id,
        "child_policy_id": child.policy_id,
        "lineage_verified": child.parent_policy_id == parent.policy_id,
        "board_id": board.board_id if board else None,
        "board_declared": board is not None,
        "task_count": len(parsed),
        "usable_task_count": len(usable),
        "task_ids": [t.task_id for t in parsed],
        "task_families": sorted({t.task_family for t in parsed}),
        "stages": sorted({t.stage for t in parsed}),
        "caps": caps,
        "equal_caps_verified": equal_caps,
        "blinding_verified": blinded,
        "settings_matched": settings_matched,
        "settings_digest": sorted(settings)[0] if settings_matched else None,
        "accounting_complete": bool(accounted and equal_caps and blinded and settings_matched
                                    and not excluded and board is not None),
        "primary_proxy": primary,
        "protected_proxies": sorted(n for n, s in specs.items() if s.role == "protected"),
        "diagnostic_proxies": sorted(n for n, s in specs.items() if s.role == "diagnostic"),
        "proxies": proxies,
        "holdout_task_count": len(holdout),
        "holdout_mean_gains": holdout_gain,
        "by_task_family": {k: dict(v, mean_primary_gain=v["primary_gain_sum"] / v["task_count"])
                           for k, v in sorted(by_family.items())},
        "by_difficulty": {k: dict(v) for k, v in sorted(by_difficulty.items())},
        "arm_structure": structure,
        "pairs": pairs,
        "excluded_tasks": excluded,
        "unmeasured_dimensions": list(board.unmeasured_dimensions) if board else
            ["every dimension: no evaluation board was declared for this comparison"],
        "scope": SCOPE,
    }


def _condition(name, status, reason):
    return {"name": name, "status": status, "reason": reason}


def promotion_rule(report):
    """Promote only on a strict primary gain, no protected regression, complete accounting and a passed held-out stage.

    Incomplete accounting yields ``inconclusive``. It never yields a pass, and it
    never yields a rejection either: you cannot conclude anything from a comparison
    whose bookkeeping does not hold.
    """
    if not isinstance(report, dict) or report.get("version") != VERSION:
        raise Invalid("promotion_rule needs a paired_replay report from this protocol version")
    proxies = report.get("proxies") or {}
    primary = report.get("primary_proxy")
    conditions, verdict = [], None

    if not report.get("board_declared"):
        conditions.append(_condition("evaluation_board_declared", "unmet",
                                     "No board was declared, so no proxy has a role and nothing is protected"))
    else:
        conditions.append(_condition("evaluation_board_declared", "met",
                                     "Board " + str(report.get("board_id")) + " with "
                                     + str(len(report.get("unmeasured_dimensions") or []))
                                     + " declared unmeasured dimension(s)"))

    accounting_problems = []
    for key, message in (("equal_caps_verified", "caps were not equal across arms"),
                         ("blinding_verified", "an arm observed the other arm's outputs"),
                         ("settings_matched", "arms ran under different fixed settings")):
        if not report.get(key):
            accounting_problems.append(message)
    if report.get("excluded_tasks"):
        accounting_problems.append(str(len(report["excluded_tasks"])) + " task(s) were excluded for accounting reasons")
    if not report.get("usable_task_count"):
        accounting_problems.append("no task survived accounting checks")
    accounting_ok = bool(report.get("accounting_complete")) and not accounting_problems
    conditions.append(_condition(
        "accounting_complete", "met" if accounting_ok else "unmet",
        "Equal caps, matched settings, blinded arms and complete resource accounting on every task"
        if accounting_ok else "; ".join(accounting_problems) or "resource accounting was not established"))

    primary_row = proxies.get(primary) if primary else None
    if primary_row is None:
        conditions.append(_condition("primary_strict_gain", "unestablished", "No primary proxy was measured"))
    elif primary_row["mean_gain"] > EPSILON:
        conditions.append(_condition("primary_strict_gain", "met",
                                     "Mean paired gain on " + primary + " is " + repr(round(primary_row["mean_gain"], 12))))
    else:
        conditions.append(_condition("primary_strict_gain", "unmet",
                                     "Mean paired gain on " + str(primary) + " is "
                                     + repr(round(primary_row["mean_gain"], 12)) + "; a tie retains the parent"))

    regressed = sorted(n for n in report.get("protected_proxies") or [] if proxies.get(n, {}).get("regressed"))
    if not report.get("protected_proxies"):
        conditions.append(_condition("no_protected_regression", "unestablished", "No protected proxy was declared"))
    elif regressed:
        conditions.append(_condition("no_protected_regression", "unmet",
                                     "Protected proxies regressed: " + ", ".join(regressed)))
    else:
        conditions.append(_condition("no_protected_regression", "met",
                                     "No protected proxy regressed: " + ", ".join(report["protected_proxies"])))

    holdout_gain = (report.get("holdout_mean_gains") or {}).get(primary) if primary else None
    holdout_regressed = []
    if report.get("holdout_task_count"):
        for name in report.get("protected_proxies") or []:
            value = (report.get("holdout_mean_gains") or {}).get(name)
            if value is not None and value < -EPSILON:
                holdout_regressed.append(name)
    if not report.get("holdout_task_count"):
        conditions.append(_condition("holdout_stage_passed", "unestablished",
                                     "No held-out task was replayed; the held-out stage has not been run"))
    elif holdout_gain is not None and holdout_gain > EPSILON and not holdout_regressed:
        conditions.append(_condition("holdout_stage_passed", "met",
                                     "Held-out mean primary gain " + repr(round(holdout_gain, 12))
                                     + " over " + str(report["holdout_task_count"]) + " task(s)"))
    else:
        conditions.append(_condition("holdout_stage_passed", "unmet",
                                     "Held-out stage did not show a gain without protected regression"
                                     + (": " + ", ".join(holdout_regressed) if holdout_regressed else "")))

    statuses = {c["name"]: c["status"] for c in conditions}
    if statuses["accounting_complete"] != "met" or statuses["evaluation_board_declared"] != "met":
        verdict = "inconclusive"
    elif statuses["no_protected_regression"] == "unmet" or statuses["primary_strict_gain"] == "unmet":
        verdict = "reject"
    elif any(s == "unestablished" for s in statuses.values()):
        verdict = "inconclusive"
    elif statuses["holdout_stage_passed"] == "unmet":
        verdict = "reject"
    else:
        verdict = "promote"
    return {
        "version": VERSION,
        "verdict": verdict,
        "conditions": conditions,
        "unmet_conditions": [c["name"] for c in conditions if c["status"] != "met"],
        "parent_policy_id": report.get("parent_policy_id"),
        "child_policy_id": report.get("child_policy_id"),
        "retained_parent": verdict != "promote",
        "eligible_for_registry_review": verdict == "promote",
        "auto_promotion": False,
        "unmeasured_dimensions": list(report.get("unmeasured_dimensions") or []),
        "scope": SCOPE + " A 'promote' verdict means this candidate may be submitted to the "
                         "operator-reviewed stage gates; it is not an installation, an activation "
                         "or a claim that the method is better on the dimensions nobody measured.",
    }


def _mixture_log_lr(n, mean, null_mean, sigma, tau):
    variance = sigma * sigma
    denominator = variance + n * tau * tau
    return (0.5 * math.log(variance / denominator)
            + (n * n * tau * tau * (mean - null_mean) ** 2) / (2.0 * variance * denominator))


def sequential_verdict(gains, *, sigma, tau=0.05, null_mean=0.0,
                       promotion_alpha=PROMOTION_ALPHA, demotion_alpha=DEMOTION_ALPHA):
    """An always-valid (anytime-valid) mSPRT so early stopping cannot inflate false promotions.

    ``gains`` are signed paired differences with positive meaning the child is
    better. The statistic is the Gaussian-mixture likelihood ratio of Howard,
    Ramdas, McAuliffe and Sekhon (2021):

        L_n = sqrt(s2 / (s2 + n t2)) * exp( n^2 t2 (xbar_n - m0)^2 / (2 s2 (s2 + n t2)) )

    Guarantee. Under H0 (true mean gain equals ``null_mean``) with independent
    observations of variance at most ``sigma**2``, L_n is a nonnegative martingale
    with unit mean, so Ville's inequality gives P(exists n: L_n >= 1/alpha) <= alpha.
    The running minimum of 1/L_n is therefore an always-valid p-value: you may look
    after every task and stop the instant it drops below alpha without inflating the
    error rate above alpha.

    What the guarantee does not cover, stated plainly: it assumes independent paired
    differences and a *pre-declared* variance bound (estimating sigma from the same
    comparison voids it); it is a statement about one proxy in isolation, so running
    it on several proxies without a multiplicity correction does not control the
    family-wise rate; and it says nothing whatever about the board's unmeasured
    dimensions.
    """
    if not isinstance(gains, (list, tuple)) or not gains:
        raise Invalid("A sequential verdict needs a nonempty sequence of paired gains")
    if len(gains) > MAX_PEEKS:
        raise Invalid("At most " + str(MAX_PEEKS) + " observations per sequential verdict")
    values = []
    for value in gains:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise Invalid("Paired gains must be finite numbers")
        values.append(float(value))
    for name, value in (("sigma", sigma), ("tau", tau)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise Invalid(name + " must be a finite positive number declared before the comparison")
    for name, value in (("promotion_alpha", promotion_alpha), ("demotion_alpha", demotion_alpha)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1:
            raise Invalid(name + " must lie strictly between 0 and 1")
    if promotion_alpha > demotion_alpha:
        raise Invalid("Promotion must require at least as much evidence as demotion: "
                      "promotion_alpha may not exceed demotion_alpha")
    if isinstance(null_mean, bool) or not isinstance(null_mean, (int, float)) or not math.isfinite(null_mean):
        raise Invalid("null_mean must be a finite number")
    promotion_boundary = math.log(1.0 / promotion_alpha)
    demotion_boundary = math.log(1.0 / demotion_alpha)
    running, total, decision, decision_at = 1.0, 0.0, "continue", None
    path = []
    for index, value in enumerate(values, start=1):
        total += value
        mean = total / index
        log_lr = _mixture_log_lr(index, mean, null_mean, float(sigma), float(tau))
        p_value = min(1.0, math.exp(-log_lr)) if log_lr > -700 else 1.0
        running = min(running, p_value)
        crossed = None
        if log_lr >= promotion_boundary and mean > null_mean:
            crossed = "promote_evidence_sufficient"
        elif log_lr >= demotion_boundary and mean < null_mean:
            crossed = "demote_evidence_sufficient"
        if crossed and decision == "continue":
            decision, decision_at = crossed, index
        path.append({"n": index, "mean": mean, "log_likelihood_ratio": log_lr,
                     "always_valid_p_value": running, "crossed": crossed})
    return {
        "version": VERSION,
        "test": "mSPRT, Gaussian mixture, always-valid p-value (Howard, Ramdas, McAuliffe, Sekhon 2021)",
        "n": len(values),
        "peeks": len(values),
        "mean_gain": total / len(values),
        "sigma": float(sigma),
        "tau": float(tau),
        "null_mean": float(null_mean),
        "promotion_alpha": float(promotion_alpha),
        "demotion_alpha": float(demotion_alpha),
        "promotion_log_boundary": promotion_boundary,
        "demotion_log_boundary": demotion_boundary,
        "always_valid_p_value": running,
        "promotion_evidence_sufficient": decision == "promote_evidence_sufficient",
        "demotion_evidence_sufficient": decision == "demote_evidence_sufficient",
        "decision": decision,
        "decision_at_n": decision_at,
        "path": path,
        "asymmetry": ("Promotion changes how future investigations are run and is therefore the "
                      "risky act; rollback is cheap. Promotion spends alpha="
                      + repr(promotion_alpha) + " and demotion alpha=" + repr(demotion_alpha)
                      + ", inverting the clinical convention deliberately."),
        "assumptions": [
            "Paired differences are independent across tasks.",
            "sigma is a pre-declared upper bound on the paired standard deviation; an underestimate voids the bound.",
            "tau tunes early versus late power and does not affect validity.",
            "The guarantee covers this one proxy; several proxies need a multiplicity correction.",
            "The guarantee covers only what is measured. It says nothing about the board's unmeasured dimensions.",
        ],
        "scope": SCOPE + " A sufficient sequential verdict bounds the false-promotion rate on one "
                         "measured proxy. It is not itself a promotion decision.",
    }


def power_report(task_count, *, noise_sd, minimum_detectable_effect,
                 alpha=PROMOTION_ALPHA, power=0.8, sequential_inflation=1.0):
    """What effect could this comparison detect at all? Refusing an underpowered run is a feature."""
    if type(task_count) is not int or not 0 <= task_count <= MAX_TASKS:
        raise Invalid("task_count must be an integer from 0 to " + str(MAX_TASKS))
    for name, value in (("noise_sd", noise_sd), ("minimum_detectable_effect", minimum_detectable_effect)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise Invalid(name + " must be a finite positive number")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise Invalid("alpha must lie strictly between 0 and 1")
    if isinstance(power, bool) or not isinstance(power, (int, float)) or not 0 < power < 1:
        raise Invalid("power must lie strictly between 0 and 1")
    if isinstance(sequential_inflation, bool) or not isinstance(sequential_inflation, (int, float)) or sequential_inflation < 1:
        raise Invalid("sequential_inflation is a declared allowance of at least 1.0 for anytime-valid stopping")
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1.0 - alpha / 2.0)
    z_power = normal.inv_cdf(power)
    factor = (z_alpha + z_power) * float(noise_sd) * math.sqrt(float(sequential_inflation))
    detectable = factor / math.sqrt(task_count) if task_count > 0 else None
    required = math.ceil((factor / float(minimum_detectable_effect)) ** 2)
    underpowered = task_count <= 0 or detectable > float(minimum_detectable_effect) + EPSILON
    return {
        "version": VERSION,
        "task_count": task_count,
        "noise_sd": float(noise_sd),
        "minimum_detectable_effect": float(minimum_detectable_effect),
        "alpha": float(alpha),
        "power": float(power),
        "sequential_inflation": float(sequential_inflation),
        "detectable_effect_at_this_task_count": detectable,
        "required_task_count": required,
        "shortfall": max(0, required - task_count),
        "underpowered": bool(underpowered),
        "recommendation": "refuse_underpowered_comparison" if underpowered else "adequately_powered_to_proceed",
        "reason": ("This many tasks can only resolve an effect of "
                   + (repr(round(detectable, 6)) if detectable is not None else "any size")
                   + " against a declared minimum detectable effect of "
                   + repr(float(minimum_detectable_effect)) + "; " + str(required)
                   + " task(s) would be needed") if underpowered else
                  ("Resolves " + repr(round(detectable, 6)) + " at or below the declared minimum detectable effect"),
        "scope": SCOPE + " A power calculation is about the declared noise model only. Adequate power "
                         "does not make a proxy the right proxy, and it does not measure the unmeasured.",
    }
