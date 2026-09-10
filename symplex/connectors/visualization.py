"""Declarative Vega-Lite exports from executed artifacts; no remote data or expressions."""

from symplex.core.contracts import Invalid


def scenario_chart(store, problem_id, artifact_id):
    r = store.get(artifact_id)
    if r["parent"] != problem_id or r["kind"] != "scenario_report" or r["stale"]:
        raise Invalid("Choose a current scenario report belonging to this problem")
    rows = []
    for scenario, data in r["data"]["scenario_results"].items():
        for step, states in enumerate(data["mean_trajectory"]):
            for node, value in states.items():
                rows.append(
                    {"scenario": scenario, "step": step, "state": node, "value": value}
                )
    return {
        "source_id": r["id"],
        "source_digest": r["digest"],
        "basis": "conditional_simulation",
        "spec": {
            "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
            "title": "Conditional trajectories — assumed mechanisms, not measured impact",
            "data": {"values": rows},
            "mark": "line",
            "encoding": {
                "x": {"field": "step", "type": "quantitative"},
                "y": {
                    "field": "value",
                    "type": "quantitative",
                    "scale": {"domain": [0, 1]},
                },
                "color": {"field": "state", "type": "nominal"},
                "column": {"field": "scenario", "type": "nominal"},
            },
        },
        "scope": r["data"]["scope"],
    }
