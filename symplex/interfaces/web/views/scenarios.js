function scenariosView() {
  const reports = related("scenario_report"),
    r = reports.at(-1),
    model = r
      ? records("system_hypothesis").find((x) => x.id === r.data.model_id)
      : null;
  let content = head(
    "EXPERIMENT / INFER / REVISE",
    "Explore how the system responds.",
    "Compare interventions across stochastic rollouts, inspect feedback, and challenge the assumed mechanisms.",
    current()
      ? btn("Agent-led experiment", "scenarios", selected, "primary")
      : "",
  );
  if (!r)
    return (
      content +
      empty(
        "Test a system hypothesis.",
        "The agent builds a bounded stochastic model, selects alternatives, runs paired scenarios and removes couplings to test structural dependence. These simulations remain conditional until validated against real observations.",
      )
    );
  const d = r.data;
  content += `<div class="notice">Assumption-based simulation. These are computed conditional results, not observed impacts or calibrated forecasts.</div><div class="section-heading"><h3>${esc(d.title)}</h3>${artifact(r)}</div><div class="grid">${Object.entries(
    d.scenario_results,
  )
    .map(
      ([id, s]) =>
        `<article class="card"><span class="eyebrow">${esc(id)}</span><h3>${esc(s.name)}</h3><p>${esc(s.interpretation)}</p><div class="metric"><label>NORMALIZED PREFERENCE UTILITY</label><strong>${s.mean_utility.toFixed(3)}</strong><small>10–90% rollout range: ${s.p10.toFixed(3)} to ${s.p90.toFixed(3)}</small></div><p class="tiny">Paired difference from baseline: ${s.paired_difference.toFixed(3)}. These ranges reflect the specified noise model only.</p></article>`,
    )
    .join("")}</div>`;
  if (model) {
    content += `<div class="section-heading"><h3>What if we changed an assumption?</h3></div><form id="whatif-form" class="card" data-model="${model.id}"><label for="whatif-node">Intervention state</label><select id="whatif-node">${model.data.nodes.map((n) => `<option value="${esc(n.id)}">${esc(n.name)}</option>`).join("")}</select><label for="whatif-value">Normalized value: <output id="whatif-output">0.50</output></label><input type="range" id="whatif-value" min="0" max="1" step="0.01" value="0.5"><button class="primary">Recompute scenarios ↗</button><p class="tiny">Creates a new model version; 64 paired rollouts. No extra model call.</p></form>`;
    content += `<div class="section-heading"><h3>State trajectories</h3><span>Normalized 0–1 · mean over 64 rollouts</span></div><div class="card">${Object.entries(
      d.scenario_results,
    )
      .map(([id, s]) => {
        const colors = [
          "#32654e",
          "#8da660",
          "#b49158",
          "#769ca5",
          "#aa7979",
          "#9290a8",
        ];
        const n = s.mean_trajectory.length;
        return `<h3>${esc(s.name)}</h3><svg viewBox="0 0 600 175" role="img" aria-label="${esc(s.name)} simulated state trajectories"><path d="M25 10V145H580" fill="none" stroke="#dce3d5"/>${model.data.nodes
          .map((node, j) => {
            const points = s.mean_trajectory
              .map(
                (t, i) =>
                  (25 + (i / (n - 1)) * 550).toFixed(2) +
                  "," +
                  (145 - t[node.id] * 125).toFixed(2),
              )
              .join(" ");
            return `<polyline points="${points}" fill="none" stroke="${colors[j % colors.length]}" stroke-width="2"/>`;
          })
          .join(
            "",
          )}<text x="25" y="165" fill="#899780" font-size="9">Initial state</text><text x="520" y="165" fill="#899780" font-size="9">24 steps</text></svg><p class="tiny">${model.data.nodes.map((n) => esc(n.name)).join(" · ")}</p>`;
      })
      .join("")}</div>`;
  }
  return (
    content +
    `<div class="section-heading"><h3>Challenge the model</h3></div><div class="grid"><div class="card"><span class="eyebrow">ASSUMPTIONS</span>${list(d.assumptions)}<br><span class="eyebrow">NOT REPRESENTED</span>${list(d.unmodeled)}</div><div class="card"><span class="eyebrow">DISCONFIRMATION</span><p>${esc(d.disconfirmation)}</p><span class="eyebrow">NEXT MEASUREMENT</span><p>${esc(d.next_action)}</p>${d.ablation_run_id ? btn("Inspect coupling ablation", "artifact", d.ablation_run_id) : ""}</div></div>`
  );
}
