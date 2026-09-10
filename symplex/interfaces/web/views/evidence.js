function evidenceView() {
  const notes = related("evidence_note"),
    evidence = records("evidence");
  return (
    head(
      "02 / SYNTHESIZE",
      "Sources, observations and assumptions.",
      "Attach context, inspect source records and trace how the agent uses them in its models and comparisons.",
    ) +
    contextPanel() +
    useCaseCards() +
    `<div class="notice">Croissant describes a dataset’s metadata and structure. It does not validate the dataset’s scientific suitability.</div><div class="grid">${state.datasets.map(datasetCard).join("")}</div><div class="section-heading"><h3>Research & evidence ledger</h3><span>Sources and recorded provenance</span></div>` +
    (notes.length || evidence.length
      ? `<div class="card">${[...notes, ...evidence].map((r) => `<div class="list-row"><div class="row-main"><strong>${esc(r.data.provenance?.source || r.data.mode || "Evidence snapshot")}</strong><small>${esc(r.data.text?.slice(0, 160) || r.data.digest || r.id)}</small></div>${artifact(r)}</div>`).join("")}</div>`
      : empty(
          "Evidence arrives as the solver works.",
          "Launch a problem investigation or use “Research the web” in chat to explore sources.",
        )) +
    `<div class="section-heading"><h3>Research architecture brief</h3></div><a class="document-link" href="/api/research-brief" target="_blank">Problem-solving architecture and release scope <span>↗</span></a>`
  );
}
