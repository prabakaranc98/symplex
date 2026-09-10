function systemView() {
  const sol = latest("solution");
  if (!sol)
    return (
      head(
        "03 / MODEL",
        "Represent the system and its interactions.",
        "Entities, mechanisms, feedback, constraints, and observations belong in one inspectable model.",
      ) +
      empty(
        "Your system model will appear here.",
        "The agent proposes entities, mechanisms, connected components and tests from the problem and supplied context.",
        current()
          ? btn("Launch solver ↗", "solve", selected, "primary")
          : btn("Define problem ↗", "new", "", "primary"),
      )
    );
  const s = sol.data;
  return (
    head("03 / MODEL", s.title, s.boundary, artifact(sol)) +
    `<div class="notice">Proposed system · ${esc(pretty(s.execution_status))}. Execution and empirical evaluation are recorded separately.</div><div class="grid"><div class="card"><span class="eyebrow">ENTITIES & STATE</span>${list(s.entities)}${list(s.state_variables)}</div><div class="card"><span class="eyebrow">FEEDBACK & HARD CONSTRAINTS</span>${list(s.feedback)}${list(s.hard_constraints)}</div></div><div class="section-heading"><h3>Interconnected subproblems</h3></div><div class="grid">${(s.subproblems || []).map((p) => `<div class="card"><span class="eyebrow">${esc(p.id)}</span><h3>${esc(p.question)}</h3><p>${esc(p.acceptance_check)}</p><span class="tiny">Depends on: ${esc(p.depends_on.join(", ") || "initial evidence")}</span></div>`).join("")}</div><div class="section-heading"><h3>Solution component graph</h3><span>Models + rules + tools + delivery</span></div><div class="graph">${s.components.map((c) => `<article class="component"><div class="card-top"><span class="eyebrow">${esc(c.kind)}</span><small>${esc(c.id)}</small></div><h3>${esc(c.name)}</h3><p>${esc(c.responsibility)}</p><br>${tag(c.readiness, true)}</article>`).join("")}</div><div class="section-heading"><h3>Couplings</h3><span>Meaning, units and time</span></div>${s.couplings.map((c) => `<div class="coupling"><strong>${esc(c.source)} → ${esc(c.target)}</strong> · ${esc(c.meaning)}<br>${esc(c.units)} · ${esc(c.timing)}</div>`).join("")}<div class="section-heading"><h3>Inference & consumption</h3></div><div class="grid"><div class="card"><span class="eyebrow">INFERENCE</span><p>${esc(s.inference_plan)}</p><span class="eyebrow">TRAINING</span><p>${esc(s.training_plan)}</p></div><div class="card"><span class="eyebrow">USABLE OUTPUT</span><h3>${esc(s.consumable)}</h3><p>${esc(s.decision)}</p><span class="eyebrow">SUCCESS CRITERION</span><p>${esc(s.success_criterion)}</p></div></div>`
  );
}
function modelsView() {
  const programs = records("program"),
    codes = related("code_proposal"),
    archive = allLatest("archive");
  return (
    head(
      "03 / COMPOSE",
      "Inspect components and program versions.",
      "A solution can combine scientific models, learned components, explicit rules, tools and decision logic.",
      current() ? btn("Draft a component", "code", selected) : "",
    ) +
    `<div class="section-heading"><h3>Model asset registry</h3><span>Supported capabilities are explicit</span></div><div class="card">${state.assets.map((a) => `<div class="list-row"><div class="row-main"><strong>${esc(a.model || "Hugging Face specialist extension")}</strong><small>${esc(a.provider)} · ${esc(a.limitations)}</small></div>${tag(a.status, a.status === "dependency_gap")}</div>`).join("")}</div><div class="section-heading"><h3>Executable lineage</h3><span>Parent → mutation → run → evaluation</span></div>` +
    (programs.length
      ? `<div class="card">${programs.map((p) => `<div class="list-row"><div class="row-main"><strong>${esc(p.data.name)}</strong><small>${p.data.parent_program_id ? "↳ " + esc(p.data.parent_program_id) : "Seed"} · ${esc(pretty(p.data.origin))}</small></div>${artifact(p)}</div>`).join("")}</div>`
      : empty(
          "No programs evaluated yet.",
          "The solver can evolve the prepared P1 adapter. Hosted code and outputs appear in Modeling studio; unexecuted component drafts appear below.",
        )) +
    `<div class="section-heading"><h3>Prepared P1 program archive</h3><span>Sparse 3 × 3 · descriptor coverage</span></div><div class="archive">${[
      "independent",
      "shared_driver",
      "lagged_graph",
    ]
      .flatMap((c) =>
        ["simple", "bias_or_staleness", "missingness_aware"].map((o) => {
          const id = archive?.data.cells[c + "/" + o];
          return `<div class="archive-cell ${id ? "occupied" : ""}"><strong>${esc(pretty(c))}</strong>${esc(pretty(o))}<br><br>${id ? btn("Inspect elite ↗", "artifact", id, "quiet") : "Unoccupied"}</div>`;
        }),
      )
      .join(
        "",
      )}</div><p class="tiny">Archive cells describe program diversity; they are not independent scientific confirmations.</p><div class="section-heading"><h3>Coding workspace</h3><span>Proposals, dependencies and checks</span></div>` +
    (codes.length
      ? codes
          .map(
            (c) =>
              `<article class="card"><div class="card-top"><h3>${esc(c.data.filename)}</h3>${tag("Unexecuted", true)}</div><p>${esc(c.data.expectation)}</p><pre>${esc(c.data.code)}</pre><div class="actions">${artifact(c)}<a class="button" href="/api/export/${c.id}">Export component ↗</a></div></article>`,
          )
          .join("")
      : empty(
          "No component draft saved yet.",
          "Astra can propose an adapter or model with dependencies and tests. Use Modeling Studio to let the agent build and execute code in the hosted Python sandbox.",
        ))
  );
}
