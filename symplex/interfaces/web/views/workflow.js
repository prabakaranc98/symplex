function componentGraph(solution) {
  const nodes = solution.data.components,
    edges = solution.data.couplings;
  const positions = Object.fromEntries(
    nodes.map((n, i) => [
      n.id,
      {
        x: 360 + 250 * Math.cos((2 * Math.PI * i) / nodes.length - Math.PI / 2),
        y: 235 + 155 * Math.sin((2 * Math.PI * i) / nodes.length - Math.PI / 2),
      },
    ]),
  );
  return `<div class="card graph-card"><span class="eyebrow">PROPOSED COMPONENT DEPENDENCIES · SELECT A NODE</span><svg viewBox="0 0 720 480" role="img" aria-label="Component graph for ${esc(solution.data.title)}"><defs><marker id="graph-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7Z" fill="#8a9f86"/></marker></defs>${edges
    .map((e) => {
      const a = positions[e.source],
        b = positions[e.target];
      if (!a || !b) return "";
      const dx = b.x - a.x,
        dy = b.y - a.y,
        len = Math.hypot(dx, dy) || 1;
      return `<path d="M${a.x} ${a.y} Q360 235 ${b.x - (dx / len) * 42} ${b.y - (dy / len) * 28}" fill="none" stroke="#b7c6ad" stroke-width="1.5" marker-end="url(#graph-arrow)"><title>${esc(e.meaning)} · ${esc(e.units)} · ${esc(e.timing)}</title></path>`;
    })
    .join("")}${nodes
    .map((n) => {
      const p = positions[n.id];
      return `<g role="button" tabindex="0" data-component="${esc(n.id)}" data-solution="${solution.id}" aria-label="Inspect ${esc(n.name)}"><rect x="${p.x - 70}" y="${p.y - 25}" width="140" height="50" rx="9" fill="#f3f6ed" stroke="#638166"/><text x="${p.x}" y="${p.y - 3}" text-anchor="middle" font-size="10" fill="#294e37">${esc(n.name.length > 23 ? n.name.slice(0, 21) + "…" : n.name)}</text><text x="${p.x}" y="${p.y + 13}" text-anchor="middle" font-size="8" fill="#7b8973">${esc(n.kind)}</text></g>`;
    })
    .join(
      "",
    )}</svg><p class="tiny">Edges express agent-proposed connections, not verified causal effects. Select a component to inspect its inputs, outputs and implementation status.</p></div>`;
}
function systemGraphView() {
  const s = latest("solution");
  return (s ? componentGraph(s) : "") + systemView();
}

function agentRosterView() {
  const roles = Array.isArray(state?.agents) ? state.agents : [];
  if (!roles.length) return "";
  return `<div class="section-heading"><h3>Agent roles and their contracts</h3><a href="/api/contracts" target="_blank" rel="noreferrer">Inspect schemas ↗</a></div><p class="tiny">Roles called by the solver, with their configured models, inputs and output contracts.</p><div class="grid">${roles.map((role) => `<article class="card"><div class="card-top"><span class="eyebrow">${esc(pretty(role.model_role))}</span><span class="tiny">${esc(state.models?.[role.model_role] || "Model not configured")}</span></div><h3>${esc(role.name)}</h3><dl><div class="ev-field"><dt>Input</dt><dd>${esc(role.input)}</dd></div><div class="ev-field"><dt>Output contract</dt><dd>${esc(role.output)}</dd></div><div class="ev-field"><dt>Runtime entry</dt><dd class="mono">${esc(role.entry)}</dd></div></dl></article>`).join("")}</div>`;
}

function workflowResourceChange(result) {
  const values = result?.data?.resource_usage_or_reservations;
  const entries = values && typeof values === "object" && !Array.isArray(values)
    ? Object.entries(values)
    : [];
  if (!entries.length)
    return '<p class="tiny">Tool resource change was not recorded in this result.</p>';
  const number = new Intl.NumberFormat(undefined, {
    maximumFractionDigits: 6,
    signDisplay: "exceptZero",
  });
  return `<details class="ev-details"><summary>Recorded resource use or reservation changes</summary><p class="tiny">Ledger change during this tool execution. Values combine usage and outstanding reservations; negative values can reflect released reservations. These are not settled billing amounts.</p>${entries.map(([resource, amount]) => `<div class="list-row"><div class="row-main"><strong>${esc(pretty(resource))}</strong></div><span class="mono">${typeof amount === "number" && Number.isFinite(amount) ? esc(number.format(amount)) + (resource === "usd" ? " USD" : "") : "Not reported"}</span></div>`).join("")}</details>`;
}

function workflowActionCard(action, index, outcome) {
  const data = action.data || {};
  const status = outcome?.data.status || "no completed result";
  const sources = Array.isArray(data.supporting_artifact_ids) ? data.supporting_artifact_ids : [];
  const field = (title, value) => `<div class="ev-field"><dt>${esc(title)}</dt><dd>${esc(value || "Not recorded in this action")}</dd></div>`;
  return `<article class="card workflow-step">
    <span class="step-index">${String(index + 1).padStart(2, "0")}</span>
    <div class="card-top"><span class="eyebrow">${esc(pretty(data.tool))}</span>${tag(status, !outcome)}</div>
    <h3>${esc(data.uncertainty)}</h3><p>${esc(data.instruction)}</p>
    <dl>${field("Search dimension", data.search_dimension ? pretty(data.search_dimension) : "")}${field("Expected change", data.expected_change)}${field("Rejection condition", data.disconfirmation)}${field("Expected resource use", data.expected_resource_use)}</dl>
    ${sources.length ? `<details class="ev-details"><summary>Supporting artifacts cited by the proposed action</summary><div class="ev-artifact-links">${sources.map((id) => artifact({ id }, `Support · ${esc(id)}`)).join("")}</div></details>` : '<p class="tiny">No supporting artifact references were recorded.</p>'}
    ${workflowResourceChange(outcome)}
    <div class="actions ev-space">${artifact(action, "Inspect action")}${outcome ? artifact(outcome, "Inspect result") : ""}${data.investigation_state_id ? artifact({ id: data.investigation_state_id }, "Search state at decision") : ""}</div>
  </article>`;
}

function investigationArchiveView() {
  const snapshot = latest("investigation_state");
  if (!snapshot) return "";
  const candidates = Array.isArray(snapshot.data?.candidates) ? snapshot.data.candidates : [];
  const visible = candidates.slice(-20).reverse();
  return `<div class="section-heading"><h3>Retained possibilities</h3>${artifact(snapshot, "Inspect full archive ↗")}</div><p class="tiny">Latest saved investigation state · showing ${visible.length} of ${candidates.length} candidate entries. Earlier entries remain available in the full archive.</p><p>${esc(snapshot.data?.scope || "Candidate representations, interventions, computations and method proposals remain inspectable alongside their evaluations.")}</p><div class="grid">${visible.map((candidate) => `<article class="card"><div class="card-top"><span class="eyebrow">${esc(pretty(candidate.kind))}</span>${tag(candidate.status || "Status not recorded", true)}</div><h3>${esc(candidate.title || candidate.id)}</h3><div class="actions">${artifact({ id: candidate.id }, "Inspect candidate")}${candidate.parent_id ? artifact({ id: candidate.parent_id }, "Parent candidate") : '<span class="tiny">Parent not recorded</span>'}</div>${Array.isArray(candidate.evaluation_ids) && candidate.evaluation_ids.length ? `<details class="ev-details"><summary>${candidate.evaluation_ids.length} linked ${candidate.evaluation_ids.length === 1 ? "evaluation" : "evaluations"}</summary><div class="ev-artifact-links">${candidate.evaluation_ids.map((id) => artifact({ id }, `Evaluation · ${esc(id)}`)).join("")}</div></details>` : '<p class="tiny ev-space">No linked evaluations in this snapshot.</p>'}</article>`).join("") || empty("No candidates in this snapshot.", "The agent records new representations, designs, computations and method proposals as it investigates.")}</div>`;
}

function workflowView() {
  const actions = related("agent_action"),
    outcomes = records("agent_result"),
    spans = related("trace_span"),
    dna = latest("problem_dna"),
    solution = latest("solution");
  return (
    head(
      "ADAPTIVE INVESTIGATION",
      "Inspect actions and results.",
      "Recorded decision rationale, tool calls, outputs, revisions and resource use for this investigation.",
      current()
        ? `<a class="button" href="/api/traces/${selected}">Export trace JSONL ↗</a>`
        : "",
    ) +
    (dna
      ? `<div class="card"><span class="eyebrow">PROBLEM DNA</span><h2>${esc(dna.data.decision)}</h2><p>${esc(dna.data.objective)}</p><div class="grid"><div><span class="eyebrow">ACTORS & CONSTRAINTS</span>${list(dna.data.actors)}${list(dna.data.constraints)}</div><div><span class="eyebrow">ASSUMPTIONS & GAPS</span>${list(dna.data.assumptions)}${list(dna.data.missing_evidence)}</div></div></div>`
      : "") +
    (solution
      ? `<div class="section-heading"><h3>Competing hypotheses</h3><span>Predictions and disconfirmation</span></div><div class="grid">${solution.data.hypotheses.map((h) => `<article class="card"><span class="eyebrow">${esc(h.id)}</span><h3>${esc(h.mechanism)}</h3><p><b>Prediction</b> · ${esc(h.observable_prediction)}</p><p><b>Could disconfirm</b> · ${esc(h.disconfirmation)}</p><p><b>Test</b> · ${esc(h.test)}</p></article>`).join("")}</div>`
      : "") +
    agentRosterView() +
    investigationArchiveView() +
    `<div class="section-heading"><h3>Actual action sequence</h3><span>Revisits remain visible</span></div><div class="workflow-timeline">${
      actions
        .map((a, i) => {
          const out = outcomes.find((r) => r.data.action_id === a.id);
          return workflowActionCard(a, i, out);
        })
        .join("") ||
      empty(
        "The investigation has not started.",
        "Launch the solver from a problem. Its proposed actions and recorded tool outcomes appear here.",
      )
    }</div>` +
    `<div class="section-heading"><h3>Execution spans</h3><span>Metadata only · no external telemetry</span></div><div class="card">${spans.map((r) => `<div class="list-row"><div class="row-main"><strong>${esc(r.data.name)}</strong><small>${r.data.duration_ms.toFixed(0)} ms · ${esc(r.data.span_id)}</small></div>${tag(r.data.status)}${artifact(r)}</div>`).join("") || "<p>New tool and connector operations emit spans.</p>"}</div>` +
    `<div class="section-heading"><h3>Current role prompts & contracts</h3><a href="/api/contracts" target="_blank">Inspect JSON schemas ↗</a></div><div class="card">${state.prompts.map((p) => `<div class="list-row"><div class="row-main"><strong>${esc(pretty(p.name))}</strong><small class="mono">${esc(p.path)}</small><small>Version digest: ${esc(p.digest.slice(0, 16))}</small></div></div>`).join("")}</div>`
  );
}
function inspectComponent(target) {
  const s = records("solution").find((r) => r.id === target.dataset.solution),
    c = s?.data.components.find((c) => c.id === target.dataset.component);
  if (!c) return;
  $("#artifact-title").textContent = c.name;
  $("#artifact-content").textContent = JSON.stringify(c, null, 2);
  $("#artifact-export").href = "/api/export/" + s.id;
  $("#artifact-dialog").showModal();
}
document.addEventListener("click", (e) => {
  const target = e.target.closest("[data-component]");
  if (target) inspectComponent(target);
});
document.addEventListener("keydown", (e) => {
  if (
    (e.key === "Enter" || e.key === " ") &&
    e.target.matches("[data-component]")
  ) {
    e.preventDefault();
    inspectComponent(e.target);
  }
});
