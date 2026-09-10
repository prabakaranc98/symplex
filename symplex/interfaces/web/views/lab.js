function modelTraceability(packageRecord) {
  const assessment = labPackageAssessment(packageRecord);
  const map = assessment?.data.model_traceability;
  if (!map) return "";
  return `<details class="card"><summary>From system mechanisms to saved source · ${esc(map.status)}</summary><p>${esc(map.scope)}</p>${map.status === "linked" ? `<p>${map.implemented_count} components linked to source; ${map.omitted_count} explicitly omitted.</p>${(map.components || []).map(c => `<div class="list-row"><div class="row-main"><strong>${esc(c.component_id)} · ${esc(c.status)}</strong><small>${esc(c.mathematical_description)}</small><small>${c.filename ? `${esc(c.filename)}:${c.line} · ${esc(c.symbol)}` : "Outside this computation"}</small><small>${esc(c.limitation)}</small></div></div>`).join("")}${(map.outputs || []).map(o => `<p class="tiny">${esc(o.state_id)} → ${esc(o.filename)} / ${esc(o.column)} · declared unit ${esc(o.unit_id)}</p>`).join("")}` : `<p>${esc(map.error || map.reason)}</p>`}${artifact(assessment, "Inspect traceability assessment")}</details>`;
}

function labPackageFiles(record) {
  const ids = Array.isArray(record?.data?.file_ids) ? record.data.file_ids : [];
  return records("file_blob").filter(file => ids.includes(file.id) && !file.stale && file.parent === record.parent && file.data.run_id === record.data.run_id && file.data.basis === "generated");
}

function labPackageAssessment(record) {
  if (!record || record.stale) return null;
  const run = records("compute_run").find(item => item.id === record.data.run_id && item.parent === record.parent && !item.stale);
  if (!run) return null;
  return records("execution_assessment").filter(item => !item.stale && item.parent === record.parent
    && item.data.package_id === record.id && item.data.package_digest === record.digest
    && item.data.run_id === run.id && item.data.run_digest === run.digest).at(-1) || null;
}

function labReportParts(record) {
  const report = String(record.data.summary || "").replace(/([^\n#])(?=#{2,6} )/g, "$1\n");
  const lines = report.split("\n"), heading = lines.findIndex(line => /^#{1,6}\s/.test(line));
  const title = heading >= 0 ? lines[heading].replace(/^#+\s*/, "") : record.data.title || "Recorded computation";
  const body = (heading >= 0 ? lines.slice(heading + 1) : lines).join("\n").trim();
  const lead = body.split(/\n\s*\n/).find(block => block.trim() && !block.trim().startsWith("|")) || "The saved files contain this run's outputs.";
  const tableStart = lines.findIndex((line, index) => line.includes("|") && /^\s*\|?\s*:?-{3,}/.test(lines[index + 1] || ""));
  const table = [];
  if (tableStart >= 0) for (const line of lines.slice(tableStart)) {
    if (!line.includes("|") || table.length >= 12) break;
    table.push(line);
  }
  return { title, lead: lead.length > 850 ? lead.slice(0, 850) + "…" : lead, table: table.join("\n"), report };
}

function labFileCard(file, figure = false) {
  const path = `/api/files/${encodeURIComponent(file.id)}`;
  return `<figure class="result-file">${figure ? `<a href="${path}" target="_blank" rel="noreferrer"><img loading="lazy" src="${path}" alt="${esc(file.data.filename)} · generated output from the recorded run"></a>` : ""}<figcaption><strong>${esc(file.data.filename)}</strong><p class="tiny">${Number.isFinite(file.data.size) ? (file.data.size / 1024).toFixed(1) + " KB · " : ""}${figure ? "Generated plot · open to enlarge" : "Recorded run output"}</p><a class="button" href="${path}${figure ? "" : "?download=true"}" ${figure ? 'target="_blank" rel="noreferrer"' : ""}>${figure ? "View full-size plot ↗" : "Download ↗"}</a></figcaption></figure>`;
}

function labResultPackage() {
  const packages = related("compute_package").slice().reverse();
  const briefFiles = new Set(latest("decision_brief")?.data?.consumable_artifact_ids || []);
  return packages.find(record => labPackageFiles(record).some(file => briefFiles.has(file.id)))
    || packages.find(record => labPackageFiles(record).some(file => /\.(png|jpg|jpeg|csv)$/i.test(file.data.filename)))
    || packages[0];
}

function labResultHighlights(record) {
  const parts = labReportParts(record), files = labPackageFiles(record);
  const figures = files.filter(file => ["image/png", "image/jpeg"].includes(file.data.mime));
  const tables = files.filter(file => /\.(csv|json|geojson|pdb|sdf)$/i.test(file.data.filename));
  const source = files.filter(file => !figures.includes(file) && !tables.includes(file));
  const run = records("compute_run").find(item => item.id === record.data.run_id && item.parent === record.parent && !item.stale);
  const protocolRef = (run?.data.input_manifest || []).find(item => item.kind === "experiment_protocol");
  const protocol = records("experiment_protocol").find(item => item.id === protocolRef?.id && item.digest === protocolRef?.digest && item.parent === record.parent && !item.stale);
  const system = records("complex_system").find(item => item.id === protocol?.data.system_id);
  const alternatives = system?.data?.decision?.alternatives || [];
  const name = id => alternatives.find(item => item.id === id)?.name || id;
  const comparison = protocol ? [protocol.data.baseline_id, ...(protocol.data.candidate_ids || [])].filter(Boolean).map(name).join(" / ") : "No linked frozen comparison is recorded for this package.";
  const assessment = labPackageAssessment(record);
  const check = assessment?.data.status === "checked" ? "Passed the frozen host checks" : assessment?.data.status === "failed" ? "Host checks need attention" : "Host checks not established";
  const meaning = assessment?.data.status === "checked" ? "The generated outputs passed the declared numerical checks. This does not establish accuracy in the real system." : assessment?.data.error || assessment?.data.next_requirement || "The report below describes what the model produced. No passing host assessment is recorded for this package.";
  return `<section class="card"><div class="card-top"><span class="eyebrow">RECORDED RESULT · AGENT-WRITTEN REPORT</span><span class="tiny">${esc(new Date(record.created * 1000).toLocaleDateString())}</span></div><h2>${esc(parts.title)}</h2><div class="result-narrative">${resultText(parts.lead, record.data.file_ids)}</div><div class="actions">${btn("Open decision and next test", "nav", "deliverables")}${btn("Compare alternatives", "nav", "evolution")}</div></section>
    ${figures.length ? `<div class="section-heading"><h3>See the computed result</h3><span>Saved output from this run</span></div><div class="result-gallery">${labFileCard(figures[0], true)}</div>${figures.length > 1 ? `<details class="card"><summary>${figures.length - 1} more recorded ${figures.length === 2 ? "plot" : "plots"}</summary><div class="result-gallery">${figures.slice(1).map(file => labFileCard(file, true)).join("")}</div></details>` : ""}` : ""}
    ${parts.table ? `<section class="card"><span class="eyebrow">REPORTED VALUES</span><div class="result-narrative">${resultText(parts.table, record.data.file_ids)}</div><p class="tiny">From the saved model report; host-check status is shown below. The complete report retains all reported rows and qualifications.</p></section>` : ""}
    <div class="grid"><section class="card"><span class="eyebrow">WHAT WAS TESTED</span><h3>${esc(protocol?.data.title || protocol?.data.experiment_id || "Recorded computation")}</h3><p>${esc(comparison)}</p>${protocol ? `<details><summary>Comparison settings and evidence gaps</summary><p>${esc(protocol.data.comparison_controls || "")}</p>${list(protocol.data.evidence_gaps || [])}${artifact(protocol, "Inspect frozen comparison")}</details>` : ""}</section><section class="card"><span class="eyebrow">CHECKS AND LIMITS</span><h3>${esc(check)}</h3><p>${esc(meaning)}</p>${assessment ? artifact(assessment, "Inspect checks and source records") : ""}</section></div>
    ${tables.length ? `<div class="section-heading"><h3>Use the computed data</h3><span>Download the exact saved outputs</span></div><div class="result-gallery">${tables.map(file => labFileCard(file)).join("")}</div>` : ""}
    <details class="card"><summary>Complete model report</summary><div class="result-narrative">${resultText(parts.report, record.data.file_ids)}</div></details>
    <details class="card"><summary>Source code, run records and file provenance</summary><div class="result-gallery">${source.map(file => labFileCard(file)).join("")}</div><div class="actions">${artifact(record, "Package record")}${run ? artifact(run, "Execution record") : ""}</div>${record.data.file_failures?.length ? '<p class="notice">Some outputs were not preserved. Exact failures are recorded in the package.</p>' : ""}</details>${modelTraceability(record)}`;
}

function labView() {
  const packages = related("compute_package"), result = labResultPackage(), uploads = related("binary_context");
  const critique = latest("model_critique"), brief = latest("decision_brief");
  const next = brief?.data.next_actions?.[0] || critique?.data.next_validation;
  return head("MODELS & RESULTS", "See what the model produced.", "Inspect the result, its recorded checks and the next experiment. Plots and tables come from saved execution outputs.", current() ? btn("Continue investigation ↗", "solve", selected, "primary") : "")
    + (result ? labResultHighlights(result) : empty("No model result recorded yet.", "Launch the investigation to build a model, test its consequences and save the outputs.", current() ? btn("Launch investigation ↗", "solve", selected, "primary") : ""))
    + (next ? `<section class="card"><span class="eyebrow">NEXT TEST · RECORDED RECOMMENDATION</span><h3>What to investigate next</h3><p>${esc(next)}</p>${btn("Inspect the decision", "nav", "deliverables")}</section>` : "")
    + (packages.length > 1 ? `<details class="card"><summary>Other computations and earlier versions (${packages.length - 1})</summary>${packages.slice().reverse().filter(record => record.id !== result?.id).map(record => `<details class="card"><summary>${esc(labReportParts(record).title)} · ${esc(new Date(record.created * 1000).toLocaleString())}</summary>${labResultHighlights(record)}</details>`).join("")}</details>` : "")
    + (current() ? `<details class="card" ${packages.length ? "" : "open"}><summary>Run a specific experiment or model revision</summary><form id="compute-form"><label for="compute-instruction">What should this computation test?</label><textarea id="compute-instruction" rows="3" maxlength="2000" required placeholder="Describe a rival explanation, an intervention or the uncertainty to test…"></textarea><button class="primary">Run this experiment ↗</button><p class="tiny">Uses the current problem, context and remaining allocation. Supplied files are sent to the hosted Python runtime for this run.</p></form></details><details class="card"><summary>Add images or source documents</summary><p>Attach diagrams, microscopy images, scans or PDFs. Add tables and text under Evidence.</p><button class="secondary" id="binary-upload">Attach PNG, JPEG or PDF</button><p class="tiny">Up to 2 MB per file.</p>${uploads.map(record => `<div class="list-row"><div class="row-main"><strong>${esc(record.data.title)}</strong><small>User-supplied input</small></div><a href="/api/files/${encodeURIComponent(record.data.blob_id)}?download=true">Download ↗</a></div>`).join("")}</details>` : "")
    + `<details class="card"><summary>Runtime availability and supported extensions</summary>${(state.runtime_capabilities || []).filter(record => related("file_blob").some(file => file.id === record.artifact_id)).map(record => `<h3>Recorded runtime probe</h3><p>${esc(record.report.method)}</p><div class="grid">${Object.entries(record.report.packages || {}).map(([name, pkg]) => `<div class="list-row"><div class="row-main"><strong>${esc(name)}</strong><small>${esc(pkg.version || "Not detected")}</small></div>${tag(pkg.available ? "discovered" : "not available", !pkg.available)}</div>`).join("")}</div>`).join("")}<div class="grid">${(state.simulators || []).map(item => `<article class="card"><span class="eyebrow">${esc(item.family)}</span><h3>${esc(item.name)}</h3><p>${esc(item.use)}</p><p class="tiny">${esc(item.required_validation)}</p>${tag(item.status, true)}<p>${urlLink(item.documentation, "Documentation ↗")}</p></article>`).join("")}</div></details>`;
}
document.addEventListener("click", (e) => {
  if (e.target.id !== "binary-upload") return;
  const input = document.createElement("input");
  input.type = "file";
  input.accept = ".png,.jpg,.jpeg,.pdf";
  input.onchange = async () => {
    const file = input.files[0];
    if (!file) return;
    try {
      if (file.size > 2000000) throw Error("Keep binary artifacts below 2 MB.");
      const content_base64 = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result.split(",")[1]);
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });
      await api("/uploads", {
        problem_id: selected,
        filename: file.name,
        content_base64,
      });
      await refresh();
      toast("Multimodal artifact saved for the next model run.");
    } catch (err) {
      toast(err.message);
    }
  };
  input.click();
});

document.addEventListener("submit", async (e) => {
  if (e.target.id !== "compute-form") return;
  e.preventDefault();
  e.submitter.disabled = true;
  try {
    await api("/compute", {
      id: selected,
      instruction: $("#compute-instruction").value,
    });
    await refresh();
    toast("Hosted computation requested.");
  } catch (error) {
    toast(error.message);
  } finally {
    if (e.submitter.isConnected) e.submitter.disabled = false;
  }
});
