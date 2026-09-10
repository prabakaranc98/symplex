function qualitySummary() {
  if (!selected) return "";
  const packages = related("compute_package");
  const byPackage = new Map(related("execution_assessment").map(r => [r.data.package_id, r]));
  const assessments = packages.map(p => byPackage.get(p.id)).filter(Boolean);
  const passed = assessments.filter(r => r.data.status === "checked");
  const failed = assessments.filter(r => r.data.status === "failed");
  return `<section class="card quality-summary"><span class="eyebrow">RECORDED CHECKS FOR THIS PROBLEM</span><h2>What the recorded work establishes</h2><div class="quality-levels"><div><strong>${packages.length}</strong><span>Compute packages</span><small>Code and outputs recorded; inspect execution provenance.</small></div><div><strong>${passed.length} / ${packages.length}</strong><span>Host numerical assessments passed</span><small>${failed.length} failed; ${packages.length - passed.length - failed.length} missing or unavailable. Frozen CSV checks have a limited scope.</small></div><div><strong>Not established</strong><span>Independent scientific validity</span><small>Numerical consistency and model critique do not establish external accuracy or causal effects.</small></div></div><div class="actions">${btn("Inspect model outputs", "nav", "lab")}${btn("Compare alternatives", "nav", "evolution")}</div>${assessments.length ? `<details><summary>Assessment records and repair requirements</summary>${assessments.map(r => `<div class="list-row"><div class="row-main"><strong>${esc(pretty(r.data.status))}</strong><small>${esc(r.data.error || r.data.next_requirement)}</small></div>${artifact(r)}</div>`).join("")}</details>` : '<p class="tiny">No post-execution assessment is recorded yet. Legacy runs keep their original scope until reassessed.</p>'}</section>`;
}

function runsView() {
  const decisions = records("decision"),
    p1 = !selected ? decisions.filter((r) => r.data.programs).at(-1) : null,
    rr = !selected ? records("research_run") : [],
    jobIds = new Set(related("run_manifest").map(r => r.data.job_id)),
    jobs = selected ? state.jobs.filter(j => jobIds.has(j.id)) : state.jobs;
  return (
    head(
      "TEST / VERIFY / IMPROVE",
      "Inspect execution and evaluation.",
      "Inspect what actually ran, which checks passed, what remains unsupported and what the agent should investigate next.",
    ) +
    qualitySummary() +
    `<details class="card"><summary>Separate software and research benchmarks</summary><div class="actions">${btn("Run synthetic plumbing check", "fixture")}${btn("Live Astra plumbing check", "fixture-live")}${btn("Research-policy pilot", "research-run")}</div><p class="tiny">Synthetic checks are clearly labeled. The research pilot uses real public questions and does not substitute for a domain evaluation.</p></details>` +
    (p1
      ? `<div class="section-heading"><h3>Latest P1 development comparison</h3>${tag(p1.data.synthetic ? "Synthetic" : "Supplied real-data slice", p1.data.synthetic)}</div><div class="card">${p1.data.programs.map((p) => `<div class="bar-row"><span>${esc(p.name)}</span><meter min="0" max="1" value="${p.development.brier}" aria-label="Brier score ${esc(p.name)}"></meter><strong>${p.development.brier.toFixed(4)}</strong></div>`).join("")}<p class="tiny">Brier loss ↓ · These are development scores. Confirmation was consumed once after freezing selection.</p>${artifact(p1, "Inspect confirmation →")}</div>`
      : "") +
    `<div class="section-heading"><h3>${selected ? "This investigation’s runs" : "Workspace execution history"}</h3><span>Workflow status · numerical checks are summarized above</span></div><div class="card">${jobs.length ? jobs.map(j => {
      const result = state.records.find(r => r.id === j.result_id);
      const status = j.status === "succeeded" ? result?.data.status || "workflow finished" : j.status;
      return `<div class="list-row"><div class="row-main"><strong>${esc(j.id)}</strong><small>${esc(j.error || new Date(j.created * 1000).toLocaleString())}</small>${result?.data.review?.assessment ? `<small>${esc(result.data.review.assessment)}</small>` : ""}</div>${tag(status, !["running", "queued"].includes(status))}${j.result_id ? btn("Inspect", "artifact", j.result_id) : ""}</div>`;
    }).join("") : "<p>No runs recorded for this selection. Launch the solver from your problem workspace.</p>"}</div>` +
    (rr.length
      ? `<div class="section-heading"><h3>Research-policy episodes</h3><span>Same Astra model and per-question caps</span></div><div class="card">${rr.map((r) => `<div class="list-row"><div class="row-main"><strong>${esc(r.data.question)}</strong><small>${esc(r.data.split)} · ${esc(r.data.policy_id)} · exact match ${r.data.metrics.exact_match}</small></div>${artifact(r)}</div>`).join("")}</div>`
      : "") +
    `<div class="section-heading"><h3>Agent activity</h3><span>Action selection & tool outcomes</span></div><div class="card">${
      (selected ? related("agent_action") : records("agent_action"))
        .slice(-12)
        .reverse()
        .map(
          (r) =>
            `<div class="list-row"><div class="row-main"><strong>${esc(pretty(r.data.tool))}</strong><small>${esc(r.data.uncertainty)}</small></div>${artifact(r)}</div>`,
        )
        .join("") ||
      "<p>The solver’s proposed actions will appear here as it works.</p>"
    }</div>`
  );
}
function decisionBriefView(record) {
  const brief = record.data || {}, array = value => Array.isArray(value) ? value : [];
  const system = records("complex_system").find(item => item.id === brief.system_id && item.parent === record.parent);
  const spec = system?.data?.spec || system?.data || {};
  const alternativeName = id => array(spec.decision?.alternatives).find(item => item.id === id)?.name || id;
  const sources = ids => `<div class="actions">${array(ids).map(id => artifact({ id }, "Inspect source ↗")).join("")}</div>`;
  const linked = new Set([...array(brief.consumable_artifact_ids), ...array(brief.claims).flatMap(claim => array(claim.source_ids))]);
  const model = related("compute_package").slice().reverse().find(item => linked.has(item.id) || labPackageFiles(item).some(file => linked.has(file.id)));
  const files = model ? labPackageFiles(model) : [];
  const figures = files.filter(file => ["image/png", "image/jpeg"].includes(file.data.mime));
  const table = model ? labReportParts(model).table : "";
  const consumables = array(brief.consumable_artifact_ids).map(id => records("file_blob").find(file => file.id === id && file.parent === record.parent && !file.stale)).filter(Boolean);
  return `<article class="card decision-brief"><div class="card-top"><span class="eyebrow">CURRENT RECOMMENDATION · AGENT-WRITTEN</span>${tag(brief.status || "Requires review", true)}</div><h2>${esc(brief.title || "Decision brief")}</h2><p class="brief-recommendation">${esc(brief.recommendation)}</p><p class="tiny">Decision owner: ${esc(brief.decision_owner || "Not recorded")}</p><div class="actions"><a class="button" href="/api/export/${encodeURIComponent(record.id)}">Export this brief ↗</a>${btn("Continue investigation ↗", "solve", selected, "primary")}</div></article>
    ${figures.length ? `<div class="section-heading"><h3>The computed result</h3><span>Saved plot from the linked model package</span></div><div class="result-gallery">${labFileCard(figures[0], true)}</div>` : ""}
    ${table ? `<section class="card"><span class="eyebrow">VALUES REPORTED BY THE MODEL</span><div class="result-narrative">${resultText(table, model.data.file_ids)}</div><p class="tiny">Recorded model output. Inspect the numerical checks and interpretation limits before relying on these values.</p></section>` : ""}
    <div class="grid"><article class="card"><span class="eyebrow">WHAT TO DO NEXT</span><h3>Next test or decision</h3>${list(array(brief.next_actions))}</article><article class="card"><span class="eyebrow">LIMITS OF THIS ANSWER</span><h3>What still needs validation</h3>${list(array(brief.external_validation_needed))}${!array(brief.external_validation_needed).length ? '<p>No external validation requirements were recorded. Review the supporting evidence before use.</p>' : ""}</article></div>
    <div class="section-heading"><h3>What the results mean</h3><span>Interpretation, evidence and limits</span></div><div class="ev-stack">${array(brief.claims).map(claim => `<article class="card"><div class="card-top"><span class="eyebrow">RECORDED CLAIM</span>${tag(claim.basis || "Basis unspecified", true)}</div><p>${esc(claim.statement)}</p><p class="tiny"><strong>Limit:</strong> ${esc(claim.limitation)}</p><details><summary>Supporting records</summary>${sources(claim.source_ids)}</details></article>`).join("") || '<p>No supported claims were recorded in this brief.</p>'}</div>
    <details class="card"><summary>Alternatives and conditions that change the recommendation</summary><div class="grid">${array(brief.alternatives).map(alternative => `<article class="card"><span class="eyebrow">${esc(alternativeName(alternative.alternative_id))}</span><h3>${esc(alternative.assessment)}</h3><p>${esc(alternative.benefit_and_tradeoff)}</p><p><strong>Unresolved test:</strong> ${esc(alternative.unresolved_test)}</p></article>`).join("")}</div><h3>Reconsider the recommendation when</h3>${list(array(brief.reversal_conditions))}</details>
    ${consumables.length ? `<details class="card"><summary>Download reusable data, plots and code (${consumables.length})</summary><div class="result-gallery">${consumables.map(file => labFileCard(file)).join("")}</div></details>` : ""}
    <details class="card"><summary>Model assumptions, checks and full execution report</summary>${model ? labResultHighlights(model) : ""}${artifact(record, "Inspect the complete decision record")}${system ? artifact(system, "Inspect the system representation") : ""}</details>`;
}

function solverDeliveryCard(record) {
  const data = record.data || {}, review = data.review || {};
  return `<article class="card"><div class="card-top"><span class="eyebrow">RECORDED INVESTIGATION OUTCOME</span>${tag(data.status || "Status unavailable", data.status === "dependency_gap")}</div><h2>${esc(data.title || current()?.data.question || "Investigation outcome")}</h2><p class="brief-recommendation">${esc(review.assessment || "The result record has no written assessment.")}</p><div class="grid"><div><span class="eyebrow">NEXT TEST</span><p>${esc(review.next_validation || data.next_action || "No next test was recorded.")}</p></div><div><span class="eyebrow">LIMITS</span><p>${esc(data.scope || "Inspect the linked execution and evidence before using this result.")}</p>${review.unsupported_claims?.length ? `<details><summary>Claims the reviewer did not support</summary>${list(review.unsupported_claims)}</details>` : ""}</div></div><div class="actions">${btn("Continue investigation ↗", "solve", selected, "primary")}<a class="button" href="/api/export/${encodeURIComponent(record.id)}">Export result ↗</a></div><details><summary>Complete result record</summary>${artifact(record)}</details></article>`;
}

function deliverablesView() {
  const delivered = (selected ? related("deliverable") : records("deliverable")).filter(record => !record.stale),
    decisions = (selected ? related("decision") : records("decision")).filter(record => !record.stale),
    brief = latest("decision_brief"), model = labResultPackage();
  const latestDelivery = delivered.at(-1);
  return head("YOUR RESULTS", "What the investigation found.", "The current recommendation, computed outputs, remaining limits and next useful test.")
    + (brief ? decisionBriefView(brief) : latestDelivery ? solverDeliveryCard(latestDelivery) : model ? `<section class="card"><span class="eyebrow">RESULTS AVAILABLE · DECISION BRIEF PENDING</span><h2>A computation is ready to inspect.</h2><p>The model produced the outputs below. A decision brief has not yet been recorded for this problem.</p>${btn("Continue to a decision brief ↗", "solve", selected, "primary")}</section>` : decisions.length ? "" : empty("A tested answer is not recorded yet.", "Continue the investigation to collect evidence, test the proposed model and identify what can be concluded.", current() ? btn("Continue investigation ↗", "solve", selected, "primary") : ""))
    + (!brief && model ? labResultHighlights(model) : "")
    + (delivered.length > (brief ? 0 : 1) ? `<details class="card"><summary>Other recorded investigation outcomes (${delivered.length - (brief ? 0 : 1)})</summary>${delivered.slice().reverse().filter(record => brief || record.id !== latestDelivery.id).map(solverDeliveryCard).join("")}</details>` : "")
    + decisions.slice().reverse().map(record => { const data = record.data || {}; return `<article class="card"><div class="card-top"><span class="eyebrow">DEDICATED ADAPTER RESULT</span>${tag(data.synthetic ? "Synthetic check" : data.status || "Recorded result", true)}</div><h2>${esc(data.title)}</h2><p class="brief-recommendation">${esc(data.recommended_action)}</p><p>${esc(data.scope)}</p><details><summary>Measurements, alternatives and assumptions</summary>${Number.isFinite(data.comparison?.parent_minus_child_brier) ? `<p>Paired Brier difference: <strong>${data.comparison.parent_minus_child_brier.toFixed(4)}</strong> (raw baseline minus selected).</p>` : ""}<h3>Best alternative</h3><p>${esc(data.best_alternative)}</p>${list(data.assumptions || [])}<h3>Unresolved evidence</h3>${list(data.unresolved_evidence || [])}${artifact(record, "Inspect adapter result")}</details></article>`; }).join("")
    + `<details class="card"><summary>Supporting model critiques and method records</summary>${[...related("model_critique"), ...related("review"), ...related("method")].map(record => `<div class="list-row"><div class="row-main"><strong>${esc(pretty(record.kind))}</strong><small>${esc(record.data.reason || record.data.assessment || record.data.scope || "")}</small></div>${artifact(record)}</div>`).join("") || '<p>No supporting critique is recorded yet.</p>'}</details>`;
}
function harnessView() {
  return (
    head(
      "THE ENGINE",
      "Runtime configuration and limits.",
      "Inspect model routing, resource limits, runtime modules and host evaluation boundaries.",
      btn("Check model access", "capabilities"),
    ) +
    `<div class="section-heading"><h3>Model routing</h3><span>OpenAI SDK · server-side credentials</span></div><div class="grid">${Object.entries(
      state.models,
    )
      .map(
        ([role, model]) =>
          `<article class="card"><span class="eyebrow">${role === "heavy" ? "SYNTHESIS, MODELING & METAREASONING" : role === "chat" ? "CONVERSATION & CLARIFICATION" : "SEPARATE CLAIM CRITIQUE"}</span><h3>${esc(model)}</h3><p>${role === "heavy" ? "Frames problems, proposes mechanisms, designs components and chooses the next action." : role === "chat" ? "Discusses the question, constraints and available evidence." : "Reviews supplied artifacts and proposes a critique of claims and scope."}</p></article>`,
      )
      .join(
        "",
      )}<article class="card"><span class="eyebrow">NUMERICAL AUTHORITY</span><h3>Host numerical evaluator</h3><p>Recomputes frozen CSV checks and the prepared adapters’ numerical scores. Model critiques remain separate from these results.</p></article></div><div class="section-heading"><h3>Global resource ledger</h3><span>Persists across tasks and restarts</span></div><div class="card">${Object.entries(
      state.budget,
    )
      .map(
        ([k, v]) =>
          `<div class="list-row"><div class="row-main"><strong>${esc(pretty(k))}</strong><small>${k === "usd" ? money(v.used_or_reserved) : v.used_or_reserved.toFixed(1)} / ${k === "usd" ? money(v.cap) : v.cap} used or conservatively reserved</small></div><meter min="0" max="${v.cap || 1}" value="${v.used_or_reserved}"></meter></div>`,
      )
      .join(
        "",
      )}</div><div class="section-heading"><h3>Modules & boundaries</h3><span>One application, replaceable interfaces</span></div><div class="card">${state.modules.map((m) => `<div class="list-row"><div class="row-main"><strong>${esc(m.name)}</strong><small>${esc(m.description)}</small><small class="mono">${esc(m.file)}</small></div>${tag(m.status, m.status !== "implemented")}</div>`).join("")}</div><div class="notice">Local single-user prototype. Trusted numerical templates run in a subprocess; generated code runs in a separate hosted Python sandbox. Specialist cross-domain engines require available backends and domain validation.</div>`
  );
}
