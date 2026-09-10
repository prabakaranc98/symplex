"""Composable bounded stochastic influence models for conditional scenario exploration.

This is a dimensionless hypothesis runner, not an automatically validated physical simulator.
The agent proposes mechanisms and assumptions. Host code executes and compares alternatives.
"""

import math
import random

from pydantic import Field, model_validator

from symplex.agents.prompts import prompt
from symplex.core.contracts import Invalid, digest
from symplex.modeling.solutions import Contract


class StateNode(Contract):
    id: str
    name: str
    meaning: str
    initial: float = Field(ge=0, le=1)
    inertia: float = Field(ge=0, le=1)
    drift: float = Field(ge=-0.2, le=0.2)
    noise_sd: float = Field(ge=0, le=0.15)
    evidence_ids: list[str]
    assumption: str


class Influence(Contract):
    source: str
    target: str
    weight: float = Field(ge=-1, le=1)
    lag: int = Field(ge=0, le=4)
    mechanism: str


class Intervention(Contract):
    node: str
    value: float = Field(ge=0, le=1)


class Scenario(Contract):
    id: str
    name: str
    interventions: list[Intervention]
    interpretation: str


class Preference(Contract):
    node: str
    weight: float = Field(ge=-1, le=1)
    rationale: str


class SystemSimulation(Contract):
    title: str
    boundary: str
    time_step: str
    nodes: list[StateNode]
    edges: list[Influence]
    scenarios: list[Scenario] = Field(
        description='First entry MUST have id "baseline" and interventions []; remaining entries describe alternatives.'
    )
    preferences: list[Preference]
    assumptions: list[str]
    unmodeled: list[str]
    disconfirmation: str

    @model_validator(mode="after")
    def conformance(self):
        ids = {n.id for n in self.nodes}
        if not 2 <= len(ids) <= 12 or len(ids) != len(self.nodes):
            raise ValueError("Require 2–12 unique state nodes")
        if not 1 <= len(self.edges) <= 30 or any(
            e.source not in ids or e.target not in ids for e in self.edges
        ):
            raise ValueError("Invalid influence graph")
        if not 2 <= len(self.scenarios) <= 5 or len(
            {s.id for s in self.scenarios}
        ) != len(self.scenarios):
            raise ValueError("Require 2–5 uniquely identified scenarios")
        if self.scenarios[0].id != "baseline" or self.scenarios[0].interventions:
            raise ValueError("First scenario must be baseline with no interventions")
        for scenario in self.scenarios:
            if any(i.node not in ids for i in scenario.interventions) or len(
                {i.node for i in scenario.interventions}
            ) != len(scenario.interventions):
                raise ValueError("Invalid or duplicated intervention target")
        if (
            not self.preferences
            or any(p.node not in ids for p in self.preferences)
            or sum(abs(p.weight) for p in self.preferences) == 0
        ):
            raise ValueError("Explicit nonzero preference weights required")
        return self

    @classmethod
    def parse(cls, data):
        return cls.model_validate(data).model_dump()

    @classmethod
    def json_schema(cls):
        return cls.model_json_schema()


def quantile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    low = int(index)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (
        index - low
    )


def simulate(data, rollouts=64, steps=24, seed=2026):
    spec = SystemSimulation.model_validate(data)
    if not 2 <= rollouts <= 128 or not 1 <= steps <= 48:
        raise Invalid("Simulation envelope exceeded")
    ids = [n.id for n in spec.nodes]
    incoming = {n: [e for e in spec.edges if e.target == n] for n in ids}
    results, scores = {}, {}
    weight_sum = sum(abs(p.weight) for p in spec.preferences)
    for scenario in spec.scenarios:
        utilities, trajectories = [], []
        overrides = {i.node: i.value for i in scenario.interventions}
        for rep in range(rollouts):
            rng = random.Random(
                seed + rep
            )  # Common exogenous innovations across scenarios.
            history = [{n.id: overrides.get(n.id, n.initial) for n in spec.nodes}]
            for step in range(steps):
                current = {}
                for node in spec.nodes:
                    influence = sum(
                        e.weight
                        * (history[max(0, len(history) - 1 - e.lag)][e.source] - 0.5)
                        for e in incoming[node.id]
                    )
                    response = 1 / (1 + math.exp(-4 * (influence + node.drift)))
                    noise = rng.gauss(
                        0, node.noise_sd
                    )  # Draw even when intervened to preserve common streams.
                    value = (
                        node.inertia * history[-1][node.id]
                        + (1 - node.inertia) * response
                        + noise
                    )
                    current[node.id] = overrides.get(node.id, min(1, max(0, value)))
                history.append(current)
            utility = (
                sum(p.weight * history[-1][p.node] for p in spec.preferences)
                / weight_sum
            )
            utilities.append(utility)
            trajectories.append(history)
        results[scenario.id] = {
            "name": scenario.name,
            "interpretation": scenario.interpretation,
            "mean_utility": sum(utilities) / rollouts,
            "p10": quantile(utilities, 0.1),
            "p90": quantile(utilities, 0.9),
            "mean_trajectory": [
                {n: sum(t[step][n] for t in trajectories) / rollouts for n in ids}
                for step in range(steps + 1)
            ],
        }
        scores[scenario.id] = utilities
    baseline = scores[spec.scenarios[0].id]
    for ident, values in scores.items():
        differences = [v - b for v, b in zip(values, baseline)]
        results[ident]["paired_difference"] = sum(differences) / rollouts
        results[ident]["fraction_better_than_baseline"] = (
            sum(d > 0 for d in differences) / rollouts
        )
    return {
        "scenarios": results,
        "rollouts": rollouts,
        "steps": steps,
        "seed": seed,
        "model_digest": digest(data),
        "protocol": "bounded-influence-v1",
        "units": "dimensionless normalized states and preference utility",
        "scope": "Conditional on agent-proposed mechanisms, parameters and preferences. Not calibrated uncertainty, empirical validation or real-world impact.",
        "equation": "x_next = clip(inertia*x + (1-inertia)*sigmoid(4*(sum(weight*(lagged_source-0.5))+drift)) + noise, 0, 1); interventions clamp designated states",
        "verdict": "inconclusive",
        "synthetic": True,
    }


def experiment(store, budget, provider, problem_id, cancel=None):
    from symplex.infrastructure.runner import execute_system

    problem = store.get(problem_id)
    designs = [
        r
        for r in store.list("solution")
        if r["parent"] == problem_id and not r["stale"]
    ]
    evidence = [
        r
        for r in store.list("evidence_note")
        if r["parent"] == problem_id and not r["stale"]
    ]
    spec = provider.propose(
        SystemSimulation,
        {
            "problem": problem["data"],
            "design": designs[-1]["data"] if designs else None,
            "evidence_ids": [r["id"] for r in evidence],
            "evidence_summaries": [
                r["data"].get("text", "")[:2000] for r in evidence[-2:]
            ],
            "instruction": prompt("system_modeler"),
        },
        problem_id,
    )
    known = {r["id"] for r in evidence}
    if any(not set(n["evidence_ids"]).issubset(known) for n in spec["nodes"]):
        raise Invalid("System model cites unknown evidence")
    mid = store.put(
        "system_hypothesis",
        dict(spec, status="proposed", template="bounded-influence-v1"),
        problem_id,
    )
    outputs = execute_system(store, budget, spec, cancel)
    run_id = store.put(
        "scenario_run", {"status": "succeeded", "model_id": mid, **outputs}, problem_id
    )
    # A predeclared structural ablation is evidence about model dependence, not model truth.
    rival = dict(spec, edges=[dict(e, weight=0) for e in spec["edges"]])
    ablation = execute_system(store, budget, rival, cancel)
    aid = store.put(
        "scenario_run",
        {
            "status": "succeeded",
            "model_id": mid,
            "ablation": "all_couplings_removed",
            **ablation,
        },
        problem_id,
    )
    report = {
        "title": spec["title"],
        "status": "conditional_scenario_comparison",
        "model_id": mid,
        "run_id": run_id,
        "ablation_run_id": aid,
        "scenario_results": outputs["scenarios"],
        "assumptions": spec["assumptions"],
        "unmodeled": spec["unmodeled"],
        "disconfirmation": spec["disconfirmation"],
        "scope": outputs["scope"],
        "empirically_validated": False,
        "synthetic": True,
        "problem_id": problem_id,
        "next_action": "Validate mechanisms and parameter ranges with domain evidence before interpreting scenario ranking as a real recommendation.",
    }
    return store.put("scenario_report", report, problem_id)
