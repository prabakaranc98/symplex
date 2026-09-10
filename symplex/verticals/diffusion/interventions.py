"""Interventions scored against a retained ensemble, including when they reverse.

A single-mechanism study answers "which intervention is best". That question has
no honest answer when the data could not identify the mechanism. The question
this module answers instead is:

    Under the retained model ensemble, intervention A is robust; intervention B
    has greater upside but reverses under plausible alternatives.

`robustness_across_ensemble` produces exactly that finding. An intervention is
designed once, against one declared design mechanism, and then evaluated under
every retained mechanism, because designing a fresh policy per mechanism would be
comparing different policies rather than testing one policy's robustness.

Cascade size is scored four ways: mean, peak timing, cross-community reach and
the upper-tail CVaR, because a policy selected on the mean can be the worst
available choice in the tail.

Nothing here is a measured intervention effect. Every number is a simulated
outcome under a proposed mechanism on a constructed graph.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from typing import Annotated, Literal

import numpy as np
from pydantic import Field
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from symplex.core.contracts import Invalid
from symplex.modeling.complex_system import Identifier, Text
from symplex.verticals.diffusion import mechanisms as mech
from symplex.verticals.diffusion import networks as nets
from symplex.verticals.diffusion.mechanisms import ParameterContract

MAX_BUDGET = 60
MAX_INTERVENTIONS = 10
MAX_ENSEMBLE_MEMBERS = 8
MAX_EVALUATION_REPLICATES = 120
MAX_ORACLE_REPLICATES = 200

TARGETING_METHODS = (
    "degree",
    "betweenness",
    "kcore",
    "eigenvector",
    "bridge",
    "greedy",
    "random",
    "none",
)

INTERVENTION_KINDS = (
    "none",
    "seed_corrective_information",
    "reduce_bridge_amplification",
    "change_recommendation_exposure",
    "harden_key_dependencies",
    "monitoring_budget",
    "immunize",
    "seed_promotion",
)

DECISION_SCOPE = (
    "simulated intervention outcomes under proposed mechanisms on a constructed "
    "graph; no effect here has been measured, no cost or feasibility constraint "
    "has been modeled, and a ranking is conditional on the retained ensemble, "
    "which is conditional on the candidate mechanisms anyone thought to propose"
)


class Intervention(ParameterContract):
    """One decision alternative, declared before it is scored."""

    id: Identifier
    kind: Literal[INTERVENTION_KINDS]
    budget: Annotated[int, Field(ge=0, le=MAX_BUDGET)] = 0
    targeting: Literal[TARGETING_METHODS] = "none"
    exposure_scale: Annotated[float, Field(ge=0.0, le=20.0)] = 1.0
    detection_delay: Annotated[int, Field(ge=0, le=50)] = 1
    rationale: Text = "declared alternative"

    def describe(self):
        return {
            "id": self.id,
            "kind": self.kind,
            "budget": self.budget,
            "targeting": self.targeting,
            "exposure_scale": self.exposure_scale,
            "detection_delay": self.detection_delay,
            "rationale": self.rationale,
        }


def parse_intervention(value):
    if isinstance(value, Intervention):
        return value
    if not isinstance(value, dict):
        raise Invalid("An intervention must be a dict or an Intervention")
    try:
        return Intervention.model_validate(value)
    except Exception as error:
        raise Invalid(f"Invalid intervention: {error}") from None


@dataclasses.dataclass
class Ensemble:
    """A network, a seed set and the mechanisms the data could not reject."""

    network: nets.Network
    members: list
    seeds: list
    horizon: int = 40
    design_label: str | None = None

    def __post_init__(self):
        if not isinstance(self.network, nets.Network):
            raise Invalid("Expected a Network")
        if not isinstance(self.members, (list, tuple)) or not self.members:
            raise Invalid("An ensemble needs at least one member")
        if len(self.members) > MAX_ENSEMBLE_MEMBERS:
            raise Invalid(f"At most {MAX_ENSEMBLE_MEMBERS} ensemble members")
        normalized = []
        for member in self.members:
            if not isinstance(member, dict) or "mechanism" not in member:
                raise Invalid("Each ensemble member must name a mechanism")
            mechanism = mech.get_mechanism(member["mechanism"])
            normalized.append(
                {
                    "label": member.get("label", mechanism.name),
                    "mechanism": mechanism.name,
                    "parameters": mechanism.parse(member.get("parameters")).model_dump(),
                    "weight": float(member.get("weight", 1.0 / len(self.members))),
                }
            )
        labels = [entry["label"] for entry in normalized]
        if len(set(labels)) != len(labels):
            raise Invalid("Ensemble member labels must be unique")
        total = sum(entry["weight"] for entry in normalized)
        if total <= 0:
            raise Invalid("Ensemble weights must be positive")
        for entry in normalized:
            entry["weight"] /= total
        self.members = normalized
        self.seeds = [int(node) for node in self.seeds]
        if not self.seeds:
            raise Invalid("An ensemble needs at least one seed node")
        self.horizon = int(self.horizon)
        if self.design_label is None:
            self.design_label = max(self.members, key=lambda e: e["weight"])["label"]
        if self.design_label not in labels:
            raise Invalid("design_label must name an ensemble member")

    @property
    def design_member(self):
        return next(m for m in self.members if m["label"] == self.design_label)

    def to_dict(self):
        return {
            "network": self.network.to_dict(),
            "members": [dict(member) for member in self.members],
            "seeds": list(self.seeds),
            "horizon": self.horizon,
            "design_label": self.design_label,
            "design_note": "the intervention is designed against this member and then "
            "evaluated under every member; a policy redesigned per mechanism would "
            "not be one policy",
            "scope": DECISION_SCOPE,
        }


def build_ensemble(network, members, seeds, horizon=40, design_label=None):
    """Assemble an ensemble, accepting `retained_ensemble(...)["members"]` directly."""
    return Ensemble(
        network=network,
        members=list(members),
        seeds=list(seeds),
        horizon=horizon,
        design_label=design_label,
    )


# --------------------------------------------------------------------------- #
# Targeting and influence maximization
# --------------------------------------------------------------------------- #


def _top(scores, budget, exclude=()):
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    excluded = set(int(node) for node in exclude)
    chosen = [int(node) for node in order if int(node) not in excluded]
    return chosen[:budget]


def degree_seeds(network, budget, exclude=()):
    """Highest-degree nodes. Fast, and redundant when hubs share a neighbourhood."""
    return {
        "nodes": _top(network.degree, budget, exclude),
        "method": "degree",
        "scope": "a structural heuristic; degree ignores overlap between the "
        "neighbourhoods of the nodes it picks",
    }


def kcore_seeds(network, budget, exclude=()):
    """Highest core number, tie-broken by degree."""
    core = nets.k_core_decomposition(network)["core_number"].astype(np.float64)
    return {
        "nodes": _top(core + 1e-6 * network.degree, budget, exclude),
        "method": "kcore",
        "core_numbers": core,
        "scope": "a structural heuristic; core membership indicates embedding in a "
        "dense region, not measured influence",
    }


def betweenness_seeds(network, budget, exclude=()):
    """Highest shortest-path betweenness."""
    score = nets.betweenness_centrality(network)["betweenness"]
    return {
        "nodes": _top(score, budget, exclude),
        "method": "betweenness",
        "scope": "a structural heuristic that assumes shortest-path traffic, which "
        "no mechanism in this vertical obeys",
    }


def eigenvector_seeds(network, budget, exclude=()):
    score = nets.eigenvector_centrality(network)["eigenvector"]
    return {
        "nodes": _top(score, budget, exclude),
        "method": "eigenvector",
        "scope": "a structural heuristic tied to the leading adjacency eigenvector",
    }


def bridge_seeds(network, budget, labels=None, exclude=()):
    """Nodes whose removal most reduces cross-community reach."""
    report = nets.bridge_nodes(network, labels, budget=max(1, budget))
    nodes = [node for node in report["nodes"] if node not in set(exclude)][:budget]
    return {
        "nodes": nodes,
        "method": "bridge",
        "cross_fraction_before": report["baseline_cross_fraction"],
        "cross_fraction_after": report["final_cross_fraction"],
        "scope": report["scope"],
    }


def random_seeds(network, budget, seed=0, exclude=()):
    rng = np.random.default_rng(int(seed))
    available = [node for node in range(network.n) if node not in set(exclude)]
    if budget > len(available):
        raise Invalid("Budget exceeds the number of available nodes")
    return {
        "nodes": sorted(int(node) for node in rng.choice(available, budget, replace=False)),
        "method": "random",
        "scope": "a randomized control for the structural heuristics",
    }


def _live_edge_components(network, probability, replicates, seed):
    """Sample percolated graphs once; independent cascade spread is then exact and cheap."""
    rng = np.random.default_rng(int(seed))
    samples = []
    for _ in range(replicates):
        keep = rng.random(network.edge_count) < probability
        matrix = np.zeros((network.n, network.n))
        if keep.any():
            rows = network.edges[keep, 0]
            cols = network.edges[keep, 1]
            matrix[rows, cols] = 1.0
            matrix[cols, rows] = 1.0
        _, labels = connected_components(csr_matrix(matrix), directed=False, return_labels=True)
        sizes = np.bincount(labels)
        samples.append((labels, sizes))
    return samples


def greedy_influence_maximization(
    network,
    mechanism="independent_cascade",
    params=None,
    budget=5,
    seed=0,
    replicates=40,
    candidate_pool=60,
    exclude=(),
):
    """Greedy seed selection with the submodular guarantee stated only where it holds.

    For the independent cascade the expected spread is evaluated exactly on
    sampled live-edge graphs, so this is textbook greedy on a submodular monotone
    objective and inherits the (1 - 1/e) approximation guarantee against the true
    expected spread. For every other mechanism the objective is estimated by
    simulation over a bounded candidate pool, and for complex contagion it is not
    submodular at all, so no guarantee is claimed there.
    """
    mechanism = mech.get_mechanism(mechanism)
    parameters = mechanism.parse(params)
    budget = int(budget)
    if not 1 <= budget <= MAX_BUDGET:
        raise Invalid(f"budget must lie in [1, {MAX_BUDGET}]")
    replicates = int(replicates)
    if not 1 <= replicates <= MAX_ORACLE_REPLICATES:
        raise Invalid(f"replicates must lie in [1, {MAX_ORACLE_REPLICATES}]")
    excluded = set(int(node) for node in exclude)

    if mechanism.name == "independent_cascade":
        samples = _live_edge_components(network, parameters.p, replicates, seed)
        chosen, marginal = [], []
        hit = [set() for _ in samples]
        current = 0.0
        for _ in range(budget):
            best_node, best_gain = None, -1.0
            for node in range(network.n):
                if node in excluded or node in chosen:
                    continue
                gain = 0.0
                for index, (labels, sizes) in enumerate(samples):
                    component = int(labels[node])
                    if component not in hit[index]:
                        gain += float(sizes[component])
                gain /= replicates
                if gain > best_gain + 1e-12:
                    best_node, best_gain = node, gain
            if best_node is None:
                break
            for index, (labels, _) in enumerate(samples):
                hit[index].add(int(labels[best_node]))
            chosen.append(int(best_node))
            marginal.append(float(best_gain))
            current += best_gain
        return {
            "nodes": chosen,
            "method": "greedy",
            "marginal_gains": marginal,
            "expected_spread": current,
            "expected_spread_fraction": current / network.n,
            "oracle": "exact expected spread on sampled live-edge graphs",
            "submodular_objective": True,
            "approximation_guarantee": "(1 - 1/e) ~ 0.632 of the optimal expected "
            "spread for the sampled objective; Monte-Carlo error in the sampled "
            "live-edge graphs is additional and is not certified here",
            "scope": "maximizes simulated expected spread under one mechanism and "
            "one parameter value; it is not a measured influence ranking",
        }

    pool = sorted(
        set(_top(network.degree, candidate_pool, excluded))
        | set(kcore_seeds(network, min(candidate_pool, network.n), excluded)["nodes"][:candidate_pool])
    )
    pool = [node for node in pool if node not in excluded][: max(candidate_pool, budget)]

    def spread(seed_set):
        results = mech.replicate(
            network,
            mechanism,
            params=parameters,
            seeds=seed_set,
            seed=int(seed) + 5000,
            horizon=40,
            replicates=replicates,
        )
        return float(np.mean([r.final_size for r in results]))

    # CELF lazy greedy: identical output to naive greedy on the same oracle.
    chosen, marginal = [], []
    current = 0.0
    queue = [[spread([node]), node, 0] for node in pool]
    queue.sort(key=lambda entry: -entry[0])
    for round_index in range(budget):
        while True:
            top = queue[0]
            if top[2] == round_index:
                break
            top[0] = spread(chosen + [top[1]]) - current
            top[2] = round_index
            queue.sort(key=lambda entry: -entry[0])
        best = queue.pop(0)
        chosen.append(int(best[1]))
        marginal.append(float(best[0]))
        current += best[0]
    return {
        "nodes": chosen,
        "method": "greedy",
        "marginal_gains": marginal,
        "expected_spread": current,
        "expected_spread_fraction": current / network.n,
        "oracle": f"simulated expected spread over {replicates} replicates on a "
        f"candidate pool of {len(pool)} nodes",
        "submodular_objective": bool(mechanism.submodular),
        "approximation_guarantee": (
            "no approximation guarantee: this mechanism's spread function is not "
            "submodular, so greedy can be arbitrarily far from optimal"
            if not mechanism.submodular
            else "(1 - 1/e) applies to the exact objective; the simulated oracle and "
            "the bounded candidate pool both break the guarantee in practice"
        ),
        "scope": "maximizes simulated expected spread under one mechanism and one "
        "parameter value; it is not a measured influence ranking",
    }


def select_targets(
    network,
    budget,
    method,
    mechanism=None,
    params=None,
    labels=None,
    seed=0,
    exclude=(),
    replicates=40,
):
    """Resolve a targeting method to a concrete node set."""
    if method not in TARGETING_METHODS:
        raise Invalid("unsupported_operation: targeting method " + str(method))
    budget = int(budget)
    if budget < 0 or budget > MAX_BUDGET:
        raise Invalid(f"budget must lie in [0, {MAX_BUDGET}]")
    if method == "none" or budget == 0:
        return {"nodes": [], "method": method, "scope": "no nodes targeted"}
    if budget > network.n - len(set(exclude)):
        raise Invalid("Budget exceeds the number of targetable nodes")
    if method == "degree":
        return degree_seeds(network, budget, exclude)
    if method == "kcore":
        return kcore_seeds(network, budget, exclude)
    if method == "betweenness":
        return betweenness_seeds(network, budget, exclude)
    if method == "eigenvector":
        return eigenvector_seeds(network, budget, exclude)
    if method == "bridge":
        return bridge_seeds(network, budget, labels, exclude)
    if method == "random":
        return random_seeds(network, budget, seed, exclude)
    if mechanism is None:
        raise Invalid("Greedy targeting requires a design mechanism")
    return greedy_influence_maximization(
        network,
        mechanism,
        params,
        budget=budget,
        seed=seed,
        replicates=replicates,
        exclude=exclude,
    )


def blocking_strategy(network, budget, method="degree", labels=None, seed=0, protect=()):
    """Immunization / blocking: the node set removed from the transmission process."""
    return select_targets(
        network, budget, method, labels=labels, seed=seed, exclude=protect
    )


# --------------------------------------------------------------------------- #
# Applying an intervention
# --------------------------------------------------------------------------- #


def resolve_intervention(intervention, ensemble, budget=None, seed=0):
    """Turn a declared intervention into a concrete simulation configuration.

    Resolved once against the ensemble's design member, so the same policy is
    carried unchanged into every mechanism it is evaluated under.
    """
    intervention = parse_intervention(intervention)
    network = ensemble.network
    budget = int(intervention.budget if budget is None else budget)
    if budget < 0 or budget > MAX_BUDGET:
        raise Invalid(f"budget must lie in [0, {MAX_BUDGET}]")
    design = ensemble.design_member
    labels = network.communities
    configuration = {
        "network": network,
        "blocked": [],
        "monitored": [],
        "quarantine_delay": None,
        "exposure_scale": float(intervention.exposure_scale),
        "seeds": list(ensemble.seeds),
        "removed_edges": [],
        "targets": [],
        "targeting_detail": None,
        "budget_used": 0,
    }
    if intervention.kind == "none":
        pass
    elif intervention.kind == "reduce_bridge_amplification":
        report = nets.bridge_edges(network, labels, budget=max(1, budget)) if budget else None
        if report is not None:
            edges = report["edges"][:budget]
            configuration["network"] = nets.remove_edges(network, edges)
            configuration["removed_edges"] = [list(map(int, e)) for e in edges]
            configuration["targeting_detail"] = report
            configuration["budget_used"] = len(edges)
    elif intervention.kind == "change_recommendation_exposure":
        configuration["budget_used"] = 0
    elif intervention.kind == "monitoring_budget":
        chosen = select_targets(
            network,
            budget,
            intervention.targeting if intervention.targeting != "none" else "degree",
            mechanism=design["mechanism"],
            params=design["parameters"],
            labels=labels,
            seed=seed,
            exclude=ensemble.seeds,
        )
        configuration["monitored"] = chosen["nodes"]
        configuration["quarantine_delay"] = int(intervention.detection_delay)
        configuration["targets"] = chosen["nodes"]
        configuration["targeting_detail"] = chosen
        configuration["budget_used"] = len(chosen["nodes"])
    elif intervention.kind == "seed_promotion":
        chosen = select_targets(
            network,
            budget,
            intervention.targeting if intervention.targeting != "none" else "degree",
            mechanism=design["mechanism"],
            params=design["parameters"],
            labels=labels,
            seed=seed,
            exclude=ensemble.seeds,
        )
        configuration["seeds"] = sorted(set(ensemble.seeds) | set(chosen["nodes"]))
        configuration["targets"] = chosen["nodes"]
        configuration["targeting_detail"] = chosen
        configuration["budget_used"] = len(chosen["nodes"])
    else:  # seed_corrective_information, harden_key_dependencies, immunize
        default = {
            "seed_corrective_information": "greedy",
            "harden_key_dependencies": "kcore",
            "immunize": "degree",
        }[intervention.kind]
        chosen = select_targets(
            network,
            budget,
            intervention.targeting if intervention.targeting != "none" else default,
            mechanism=design["mechanism"],
            params=design["parameters"],
            labels=labels,
            seed=seed,
            exclude=ensemble.seeds,
        )
        configuration["blocked"] = chosen["nodes"]
        configuration["targets"] = chosen["nodes"]
        configuration["targeting_detail"] = chosen
        configuration["budget_used"] = len(chosen["nodes"])
    if configuration["budget_used"] > budget:
        raise Invalid("Intervention exceeded its declared budget")
    configuration["budget"] = budget
    configuration["intervention"] = intervention
    return configuration


# --------------------------------------------------------------------------- #
# Outcomes
# --------------------------------------------------------------------------- #


def cvar(values, alpha=0.9, upper=True):
    """Conditional value at risk: the mean of the worst tail beyond `alpha`.

    With `upper=True` the tail is the largest values, which is the risk direction
    when a cascade is harmful. CVaR estimated from a small replicate count is
    itself noisy, because it averages only the few draws in the tail.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0:
        raise Invalid("CVaR needs at least one value")
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise Invalid("alpha must lie strictly between 0 and 1")
    ordered = np.sort(values)
    if upper:
        cut = int(math.floor(alpha * values.size))
        tail = ordered[cut:] if cut < values.size else ordered[-1:]
    else:
        cut = int(math.ceil((1.0 - alpha) * values.size))
        tail = ordered[:cut] if cut > 0 else ordered[:1]
    return float(tail.mean())


def _member_outcomes(configuration, member, horizon, seed, replicates, alpha):
    network = configuration["network"]
    results = mech.replicate(
        network,
        member["mechanism"],
        params=member["parameters"],
        seeds=configuration["seeds"],
        seed=int(seed),
        horizon=horizon,
        replicates=replicates,
        blocked=configuration["blocked"],
        monitored=configuration["monitored"],
        quarantine_delay=configuration["quarantine_delay"],
        exposure_scale=configuration["exposure_scale"],
    )
    sizes = np.array([r.final_fraction for r in results])
    peaks = np.array(
        [float(np.argmax(r.increments) + 1) if r.increments.max() > 0 else 0.0 for r in results]
    )
    labels = network.communities
    if labels is not None:
        seed_communities = set(int(labels[node]) for node in configuration["seeds"])
        outside = np.array([labels[node] not in seed_communities for node in range(network.n)])
        reach = np.array(
            [
                float((np.isfinite(r.activation_time) & outside).sum() / max(outside.sum(), 1))
                for r in results
            ]
        )
        communities_reached = np.array(
            [
                float(len(np.unique(labels[np.isfinite(r.activation_time)])))
                for r in results
            ]
        )
    else:
        reach = np.full(replicates, float("nan"))
        communities_reached = np.full(replicates, float("nan"))
    return {
        "label": member["label"],
        "mechanism": member["mechanism"],
        "weight": member["weight"],
        "sizes": sizes,
        "peaks": peaks,
        "cross_community_reach_samples": reach,
        "mean_size": float(sizes.mean()),
        "sd_size": float(sizes.std(ddof=1)) if sizes.size > 1 else 0.0,
        "median_size": float(np.median(sizes)),
        "tail_risk_cvar": cvar(sizes, alpha),
        "lower_tail_cvar": cvar(sizes, alpha, upper=False),
        "mean_peak_step": float(peaks.mean()),
        "peak_delay_note": "peak step is the step of the largest single-step adoption "
        "count; a later peak buys response time even when the final size is unchanged",
        "mean_cross_community_reach": float(np.nanmean(reach)) if labels is not None else None,
        "mean_communities_reached": float(np.nanmean(communities_reached))
        if labels is not None
        else None,
        "replicates": replicates,
    }


def evaluate_intervention(intervention, ensemble, budget=None, seed=0, replicates=24, alpha=0.9):
    """Score one intervention under every retained mechanism.

    Reports mean cascade size, peak timing, cross-community reach and the upper
    tail CVaR, per mechanism and pooled by ensemble weight. The pooled number is a
    weighted average over models the data could not separate; it is a summary, not
    a probability that the intervention achieves that outcome.
    """
    if not isinstance(ensemble, Ensemble):
        raise Invalid("Expected an Ensemble")
    replicates = int(replicates)
    if not 4 <= replicates <= MAX_EVALUATION_REPLICATES:
        raise Invalid(f"replicates must lie in [4, {MAX_EVALUATION_REPLICATES}]")
    configuration = resolve_intervention(intervention, ensemble, budget, seed)
    per_mechanism = [
        _member_outcomes(configuration, member, ensemble.horizon, seed, replicates, alpha)
        for member in ensemble.members
    ]
    weights = np.array([entry["weight"] for entry in per_mechanism])
    pooled = {
        "mean_size": float(np.sum(weights * [e["mean_size"] for e in per_mechanism])),
        "tail_risk_cvar": float(np.sum(weights * [e["tail_risk_cvar"] for e in per_mechanism])),
        "mean_peak_step": float(np.sum(weights * [e["mean_peak_step"] for e in per_mechanism])),
        "worst_case_mean_size": float(max(e["mean_size"] for e in per_mechanism)),
        "worst_case_tail_risk": float(max(e["tail_risk_cvar"] for e in per_mechanism)),
        "spread_across_mechanisms": float(
            max(e["mean_size"] for e in per_mechanism)
            - min(e["mean_size"] for e in per_mechanism)
        ),
    }
    if ensemble.network.communities is not None:
        pooled["mean_cross_community_reach"] = float(
            np.sum(weights * [e["mean_cross_community_reach"] for e in per_mechanism])
        )
    return {
        "intervention": configuration["intervention"].describe(),
        "budget": configuration["budget"],
        "budget_used": configuration["budget_used"],
        "targets": [int(node) for node in configuration["targets"]],
        "removed_edges": configuration["removed_edges"],
        "exposure_scale": configuration["exposure_scale"],
        "design_label": ensemble.design_label,
        "per_mechanism": [
            {k: v for k, v in entry.items() if k not in ("sizes", "peaks", "cross_community_reach_samples")}
            for entry in per_mechanism
        ],
        "samples": {entry["label"]: entry["sizes"] for entry in per_mechanism},
        "pooled": pooled,
        "alpha": float(alpha),
        "cvar_definition": f"mean cascade size over the worst {100 * (1 - alpha):.0f}% "
        "of replicates, estimated from a small sample and therefore noisy",
        "scope": DECISION_SCOPE,
    }


def _objective_sign(objective):
    if objective not in ("minimize_size", "maximize_size", "minimize_tail_risk", "delay_peak"):
        raise Invalid("unsupported_operation: objective " + str(objective))
    return {
        "minimize_size": ("mean_size", -1.0),
        "maximize_size": ("mean_size", 1.0),
        "minimize_tail_risk": ("tail_risk_cvar", -1.0),
        "delay_peak": ("mean_peak_step", 1.0),
    }[objective]


def robustness_across_ensemble(
    interventions,
    ensemble,
    budget=None,
    seed=0,
    objective="minimize_size",
    replicates=24,
    alpha=0.9,
    tolerance_sigma=1.5,
):
    """Score every intervention under every retained mechanism and name the reversals.

    Replicates are paired across interventions and mechanisms by common random
    numbers, so a difference between two interventions is a difference in policy
    rather than in draw. An intervention is called robust when it is the best, or
    statistically tied with the best, under every retained mechanism. A reversal
    is recorded when A beats B by more than the paired noise under one mechanism
    and B beats A by more than the paired noise under another. The reversal, not a
    single winner, is the honest output when one is present.
    """
    if not isinstance(ensemble, Ensemble):
        raise Invalid("Expected an Ensemble")
    if not isinstance(interventions, (list, tuple)) or not 2 <= len(interventions) <= MAX_INTERVENTIONS:
        raise Invalid(f"Supply between 2 and {MAX_INTERVENTIONS} interventions")
    metric, sign = _objective_sign(objective)
    evaluations = {}
    for item in interventions:
        report = evaluate_intervention(item, ensemble, budget, seed, replicates, alpha)
        identifier = report["intervention"]["id"]
        if identifier in evaluations:
            raise Invalid("Intervention identifiers must be unique")
        evaluations[identifier] = report
    names = list(evaluations)
    labels = [member["label"] for member in ensemble.members]
    weights = {member["label"]: member["weight"] for member in ensemble.members}

    scores = {
        name: {
            entry["label"]: sign * entry[metric]
            for entry in evaluations[name]["per_mechanism"]
        }
        for name in names
    }
    samples = {name: evaluations[name]["samples"] for name in names}

    def paired_margin(first, second, label):
        difference = sign * (samples[first][label] - samples[second][label])
        mean = float(difference.mean())
        error = float(difference.std(ddof=1) / math.sqrt(difference.size)) if difference.size > 1 else 0.0
        return mean, error

    per_mechanism_ranking = {
        label: sorted(names, key=lambda name: -scores[name][label]) for label in labels
    }
    best_by_mechanism = {label: per_mechanism_ranking[label][0] for label in labels}

    reversals = []
    for first, second in itertools.combinations(names, 2):
        favours_first, favours_second = [], []
        for label in labels:
            mean, error = paired_margin(first, second, label)
            threshold = tolerance_sigma * error
            if mean > threshold and mean > 0:
                favours_first.append({"mechanism": label, "margin": mean, "standard_error": error})
            elif -mean > threshold and mean < 0:
                favours_second.append({"mechanism": label, "margin": -mean, "standard_error": error})
        if favours_first and favours_second:
            reversals.append(
                {
                    "intervention_a": first,
                    "intervention_b": second,
                    "a_wins_under": favours_first,
                    "b_wins_under": favours_second,
                    "statement": (
                        f"{first} beats {second} under "
                        + ", ".join(e["mechanism"] for e in favours_first)
                        + f" (margin up to {max(e['margin'] for e in favours_first):.3f}) "
                        f"but {second} beats {first} under "
                        + ", ".join(e["mechanism"] for e in favours_second)
                        + f" (margin up to {max(e['margin'] for e in favours_second):.3f})"
                    ),
                }
            )

    robust = []
    for name in names:
        dominated_anywhere = False
        for label in labels:
            leader = best_by_mechanism[label]
            if leader == name:
                continue
            mean, error = paired_margin(leader, name, label)
            if mean > tolerance_sigma * error:
                dominated_anywhere = True
                break
        if not dominated_anywhere:
            robust.append(name)

    weighted = {
        name: float(sum(weights[label] * scores[name][label] for label in labels))
        for name in names
    }
    worst_case = {name: float(min(scores[name][label] for label in labels)) for name in names}
    upside = {name: float(max(scores[name][label] for label in labels)) for name in names}
    weighted_ranking = sorted(names, key=lambda name: -weighted[name])
    worst_case_ranking = sorted(names, key=lambda name: -worst_case[name])

    reversing = sorted({r["intervention_a"] for r in reversals} | {r["intervention_b"] for r in reversals})
    robust_choice = next((name for name in worst_case_ranking if name in robust), None)
    upside_choice = max(names, key=lambda name: upside[name])
    if robust_choice and upside_choice != robust_choice and upside_choice in reversing:
        framing = (
            f"Under the retained model ensemble, intervention {robust_choice} is "
            f"robust; intervention {upside_choice} has greater upside but reverses "
            "under plausible alternatives."
        )
    elif reversals:
        framing = (
            "Under the retained model ensemble, no intervention is robust: "
            + reversals[0]["statement"]
            + ". The choice depends on a mechanism the data has not identified."
        )
    else:
        framing = (
            f"Under the retained model ensemble, intervention {weighted_ranking[0]} "
            "ranks first under every retained mechanism; no ranking reversal was "
            "detected at this replicate count, which is weaker than showing that "
            "none exists."
        )
    return {
        "objective": objective,
        "metric": metric,
        "scores": scores,
        "per_mechanism_ranking": per_mechanism_ranking,
        "best_by_mechanism": best_by_mechanism,
        "weighted_score": weighted,
        "weighted_ranking": weighted_ranking,
        "worst_case_score": worst_case,
        "worst_case_ranking": worst_case_ranking,
        "upside_score": upside,
        "robust_interventions": robust,
        "reversals": reversals,
        "reversal_detected": bool(reversals),
        "ranking_agrees_across_mechanisms": len(set(best_by_mechanism.values())) == 1,
        "framing": framing,
        "evaluations": evaluations,
        "tolerance_sigma": float(tolerance_sigma),
        "replicates": int(replicates),
        "verdict": framing,
        "scope": DECISION_SCOPE
        + "; absence of a detected reversal at this replicate count is not evidence "
        "that no reversal exists, and the tail estimates are the noisiest numbers here",
    }
