function connectorsView() {
  const ready = state.connectors.filter((c) => c.status === "ready");
  const contexts = related("context").filter((r) => r.data.format === "csv");
  const reports = related("scenario_report");
  return (
    head(
      "CAPABILITIES & EVIDENCE",
      "Connect the right tools.",
      "Discover data, retrieve evidence, inspect semantics, visualize results, and follow the agent’s work.",
    ) +
    (current()
      ? `<form id="connector-form" class="card"><span class="eyebrow">USE A CONNECTOR · ${esc(current().data.question.slice(0, 70))}</span><label for="connector-choice">Capability</label><select id="connector-choice">${ready.map((c) => `<option value="${c.id}">${esc(c.name)} · ${esc(c.category)}</option>`).join("")}</select><label for="connector-query">Query · needed for search and research</label><input id="connector-query" maxlength="2000" placeholder="A specific question or evidence gap"><label for="connector-artifact">Artifact · needed for table and chart operations</label><select id="connector-artifact"><option value="">Select when needed</option>${[...contexts, ...reports].map((r) => `<option value="${r.id}">${esc(r.data.title || r.id)} · ${esc(r.kind)}</option>`).join("")}</select><button class="primary">Run connector ↗</button><p class="tiny">Results remain linked to this problem. Network and model usage enter the shared ledger.</p></form>`
      : empty(
          "Select a problem to use a connector.",
          "Each operation has a problem scope and a persisted result.",
        )) +
    [...new Set(state.connectors.map((c) => c.category))]
      .map(
        (category) =>
          `<div class="section-heading"><h3>${esc(pretty(category))}</h3></div><div class="grid">${state.connectors
            .filter((c) => c.category === category)
            .map(
              (c) =>
                `<article class="card"><div class="card-top"><span class="eyebrow">${esc(c.package)}</span>${tag(c.status, c.status !== "ready")}</div><h3>${esc(c.name)}</h3><p>${esc(c.limits)}</p><p class="tiny">Operations: ${esc(c.operations.join(" · "))}</p>${urlLink(c.documentation, "SDK & contract documentation ↗")}</article>`,
            )
            .join("")}</div>`,
      )
      .join("") +
    `<div class="section-heading"><h3>Connector results</h3></div>${related(
      "connector_result",
    )
      .slice(-8)
      .reverse()
      .map(
        (r) =>
          `<div class="list-row"><div class="row-main"><strong>${esc(pretty(r.data.connector))}</strong><small>${esc(r.data.result.scope || r.data.result.method || "Persisted operation result")}</small></div>${artifact(r)}</div>`,
      )
      .join("")}`
  );
}

document.addEventListener("submit", async (e) => {
  if (e.target.id !== "connector-form") return;
  e.preventDefault();
  e.submitter.disabled = true;
  try {
    const result = await api("/connectors/execute", {
      connector: $("#connector-choice").value,
      problem_id: selected,
      query: $("#connector-query").value,
      artifact_id: $("#connector-artifact").value,
    });
    await refresh();
    await inspect(result.id);
  } catch (error) {
    toast(error.message);
  } finally {
    if (e.submitter.isConnected) e.submitter.disabled = false;
  }
});
