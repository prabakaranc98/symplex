"use strict";

const ovArray = value => Array.isArray(value) ? value : [];
const ovExcerpt = (value, limit = 650) => {
  const text = String(value || "").trim();
  if (text.length <= limit) return text;
  const boundary = text.lastIndexOf(" ", limit);
  return text.slice(0, boundary > limit * .65 ? boundary : limit) + "…";
};
const ovPlain = value => String(value || "").replace(/^#{1,6}\s+/gm, "").replace(/\*\*|`/g, "");
const ovImage = file => ["image/png", "image/jpeg", "image/webp", "image/gif"].includes(file?.data?.mime);

function overviewState(problemId = selected) {
  const items = (state?.records || []).filter(r => r.parent === problemId && !r.stale);
  const all = kind => items.filter(r => r.kind === kind), last = kind => all(kind).at(-1);
  const files = all("file_blob"), packages = all("compute_package");
  const packageFiles = p => p ? labPackageFiles(p) : [];
  // Surface the latest saved data/plot computation before an ancillary runtime probe.
  const computation = packages.filter(p => packageFiles(p).some(f => ovImage(f) || f.data.mime === "text/csv")).at(-1) || packages.at(-1);
  const outputFiles = packageFiles(computation);
  const brief = last("decision_brief"), delivered = last("deliverable"), synthesis = last("evidence_synthesis");
  const report = last("scenario_report"), model = last("complex_system") || last("solution"), dna = last("problem_dna");
  const jobIds = new Set(all("run_manifest").map(r => r.data.job_id));
  const jobs = ovArray(state?.jobs).filter(j => jobIds.has(j.id)).sort((a, b) => b.created - a.created);
  const job = jobs[0], waiting = job?.status === "waiting_user", running = job?.status === "running" || job?.status === "queued";
  const clarification = waiting ? last("clarification_request") : null;
  const gap = delivered?.data.status === "dependency_gap";
  const answer = brief || delivered || computation || synthesis;
  const answerText = brief?.data.recommendation || delivered?.data.review?.assessment || computation?.data.summary || synthesis?.data.summary || "";
  const next = clarification ? [clarification.data.question] : brief?.data.next_actions?.length ? brief.data.next_actions : delivered?.data.review?.next_validation ? [delivered.data.review.next_validation] : synthesis?.data.next_action ? [synthesis.data.next_action] : [];
  const remaining = brief?.data.external_validation_needed?.length ? brief.data.external_validation_needed : synthesis?.data.gaps?.length ? synthesis.data.gaps.map(g => g.missing_information) : ovArray(dna?.data.missing_evidence);
  const status = waiting ? "Your input is needed" : running ? "Investigation running" : gap ? "Evidence gaps recorded" : brief || delivered ? "Conclusion saved" : computation || report ? "Execution saved" : model ? "Working model saved" : dna ? "Question framed" : "Ready to begin";
  const assessment = labPackageAssessment(computation);
  return { all, last, files, packages, computation, outputFiles, brief, delivered, synthesis, report, model, dna, job, waiting, running, clarification, gap, answer, answerText, next, remaining, status, assessment };
}

function flow(info = overviewState()) {
  const sources = [...info.all("context"), ...info.all("binary_context"), ...info.all("evidence_note"), ...info.all("evidence_synthesis")];
  const stages = [
    ["1", "The question", info.dna ? "Framing saved" : "Define the decision", "problem", !!info.dna],
    ["2", "Evidence", sources.length ? `${sources.length} source / synthesis records` : "Add or find relevant inputs", "evidence", !!sources.length],
    ["3", "Working model", info.model ? "Representation saved" : "Build a testable explanation", info.model?.kind === "solution" ? "system" : "complexity", !!info.model],
    ["4", "Test & compare", info.packages.length ? `${info.packages.length} saved computations` : info.report ? "Scenario results saved" : "No execution recorded", "lab", !!(info.computation || info.report)],
    ["5", "Results & decision", info.brief || info.delivered ? "Conclusion and limits saved" : "No conclusion recorded", "deliverables", !!(info.brief || info.delivered)],
  ];
  return `<section class="ov-progress" aria-label="Open a part of this investigation"><div class="ov-section-title"><h2>Follow the investigation</h2><p>Open any step. A saved record does not mean its scientific claims are verified.</p></div><div class="ov-steps">${stages.map(([number, title, detail, destination, saved]) => `<button class="ov-step ${saved ? "has-record" : ""}" data-action="nav" data-id="${destination}"><span class="ov-step-number">${number}</span><strong>${title}</strong><small>${esc(detail)}</small><span class="ov-step-open">Open ↗</span></button>`).join("")}</div></section>`;
}

function useCaseCards() {
  const briefs = ovArray(state?.use_cases);
  if (!briefs.length) return "";
  return `<details class="ov-starters"><summary><span>Start from an example</span><small>${briefs.length} adaptable problem briefs</small></summary><p>Choose a brief, add your context, then launch the agent. Creating a problem does not run an investigation.</p><div class="ov-starter-grid">${briefs.map(brief => `<article><span class="eyebrow">${esc(pretty(brief.id))}</span><h3>${esc(brief.title)}</h3><p>${esc(ovExcerpt(brief.problem, 220))}</p><details><summary>What this example needs</summary><p>${esc(brief.decision_or_estimation_target)}</p>${list(ovArray(brief.required_inputs))}<p class="tiny">${esc(brief.current_scope)}</p></details>${btn("Use this brief ↗", "use-case", brief.id, "secondary")}</article>`).join("")}</div></details>`;
}

function overviewLibrary() {
  return `${useCaseCards()}<details class="ov-starters"><summary><span>Browse domain specifications</span><small>Optional reference material</small></summary><div class="ov-starter-grid">${ovArray(state?.domains).map(domain => `<article><h3>${esc(domain.name)}</h3><p>${esc(domain.cases?.[0]?.title || "Recorded domain specification")}</p>${btn("Read specification ↗", "domain", domain.id, "secondary")}</article>`).join("")}</div></details>`;
}

function overviewControls(problem, info) {
  if (info.waiting) return `<div class="actions">${btn("Answer in conversation ↗", "steer-suggest", "Regarding the open question: ", "primary")}${btn("Review the question", "nav", "problem")}</div>`;
  if (info.running) return `<div class="actions">${btn("View live activity ↗", "nav", "workflow", "primary")}${btn("Add context or direction", "steer-suggest", "Please account for this additional context: ")}</div>`;
  return `<div class="actions">${btn(info.answer || info.model ? "Continue investigation ↗" : "Start investigation ↗", "solve", problem.id, "primary")}<label class="ov-depth">Work budget<select id="solver-depth" aria-label="Investigation action budget"><option value="focused">Focused · 8 actions</option><option value="balanced" selected>Balanced · 16 actions</option><option value="thorough">Thorough · 24 actions</option></select></label>${btn("Add context or direction", "steer-suggest", "Please account for this additional context: ", "secondary")}</div>`;
}

function overviewAnswer(info) {
  const sourceLabel = info.brief ? "Saved decision brief · agent recommendation" : info.delivered ? "Saved deliverable · model review" : info.computation ? "Saved computation · agent-written summary" : "Saved evidence synthesis";
  const preview = ovExcerpt(ovPlain(info.answerText), 1150);
  return `<section class="card ov-answer"><div class="card-top"><span class="eyebrow">WHAT IT FOUND</span>${tag(info.answer ? sourceLabel : "No answer recorded", !info.answer || info.gap)}</div><h2>${esc(info.waiting && !info.answer ? "The agent needs one detail to continue." : info.answer ? info.brief?.data.title || (info.gap ? "The current conclusion is limited by missing evidence." : "The current recorded conclusion") : info.model ? "A working explanation is ready to inspect." : "Your question is ready for investigation.")}</h2>
    ${info.answer ? `<div class="ov-answer-text">${resultText(preview)}</div>${info.answerText.length > 1150 ? `<details class="ov-source-text"><summary>Read the complete saved conclusion</summary>${resultText(info.answerText, info.computation?.data.file_ids || [])}</details>` : ""}<div class="actions">${btn("Open results & next steps ↗", "nav", info.brief || info.delivered ? "deliverables" : info.computation ? "lab" : "evidence", "primary")}${artifact(info.answer, "Inspect source record")}</div>` : `<p>${esc(info.clarification?.data.question || (info.model ? "A model has been proposed. No final answer or executed result has been saved for this problem." : "Start the agent to examine your context, build competing explanations and run useful tests within the selected work budget."))}</p><div class="actions">${info.model ? btn("See the working model ↗", "nav", info.model.kind === "solution" ? "system" : "complexity", "secondary") : btn("Review question & inputs", "nav", "problem", "secondary")}</div>`}
    <p class="ov-answer-scope">${info.computation || info.report ? "Recorded execution can support a conditional model result. It does not by itself establish real-world accuracy." : "A saved interpretation or proposed model is not an executed test. Results retain their stated evidence limits."}</p></section>`;
}

function overviewOutput(info) {
  const images = info.outputFiles.filter(ovImage), files = info.outputFiles;
  if (!info.computation && !info.report) return `<section class="card ov-output-empty"><span class="eyebrow">TEST OUTPUTS</span><h2>No computation is recorded yet.</h2><p>${info.model ? "The working model can guide the next experiment. A model diagram or plan is not an executed result." : "Charts, data and saved code will appear here after a computation."}</p>${btn(info.model ? "Inspect the proposed model ↗" : "See available evidence ↗", "nav", info.model ? "complexity" : "evidence", "secondary")}</section>`;
  const assessment = info.assessment?.data;
  return `<section class="card ov-output"><div class="card-top"><span class="eyebrow">ACTUAL SAVED OUTPUTS</span>${tag(info.computation ? `${files.length} files from this computation` : "Scenario result", true)}</div>${images.length ? `<a class="ov-plot" href="/api/files/${encodeURIComponent(images[0].id)}" target="_blank" rel="noreferrer"><img src="/api/files/${encodeURIComponent(images[0].id)}" alt="${esc(images[0].data.filename)} — recorded model output" loading="lazy" /><span>Open saved plot ↗</span></a>` : `<div class="ov-output-placeholder"><span aria-hidden="true">↗</span><h2>${info.computation ? "Code and results are saved." : "Scenario results are saved."}</h2><p>${info.computation ? "This computation has no saved image preview. Open its files to inspect the output." : "Open the scenario report to inspect its recorded comparisons and assumptions."}</p></div>`}
    <div class="ov-file-list">${files.filter(f => !ovImage(f)).slice(0, 4).map(file => `<a href="/api/files/${encodeURIComponent(file.id)}?download=true"><span>${esc(file.data.filename)}</span><small>${esc(file.data.mime || "Saved file")} · Download ↗</small></a>`).join("")}</div>
    <p class="ov-check-state">${assessment ? assessment.status === "checked" ? "Host checks: passed for this computation. Their declared scope still applies." : assessment.status === "failed" ? "Host checks: failures recorded. Review these before using the output." : "Host checks: not established for this computation." : "No host assessment is linked to this computation. The saved summary may describe maker-reported checks."}</p>
    <div class="actions">${btn("Open all outputs ↗", "nav", info.computation ? "lab" : "scenarios", "secondary")}${info.assessment ? artifact(info.assessment, "Inspect host checks") : artifact(info.computation || info.report, "Inspect run record")}</div><p class="tiny">${info.computation ? "Latest saved computation with data or plots, when available. Other runs remain in Modeling studio." : "Scenario results retain their model assumptions and validation limits."}</p></section>`;
}

function overviewNext(info) {
  const next = info.next.length ? info.next : info.running ? ["The agent is working through the investigation. Follow its current action or add context that changes the question."] : info.computation || info.report ? ["Review the saved model outputs, assumptions and checks before deciding which test to run next."] : info.model ? ["Review the proposed model and supply missing evidence, or continue the agent to choose a test."] : ["Add the decision, useful context and constraints, then start the investigation."];
  const remaining = info.remaining;
  return `<section class="ov-next-grid"><article class="card ov-next"><span class="eyebrow">${info.waiting ? "YOUR INPUT IS NEEDED" : "WHAT HAPPENS NEXT"}</span><h2>${info.waiting ? "Answer the open question" : "The next useful step"}</h2><p>${esc(ovExcerpt(next[0], 550))}</p>${next.length > 1 ? `<details><summary>${next.length - 1} more recorded next steps</summary>${list(next.slice(1))}</details>` : ""}${btn(info.waiting ? "Reply to the agent ↗" : "Discuss or steer this step ↗", "steer-suggest", info.waiting ? "Regarding the open question: " : "For the next investigation step: ", "secondary")}</article><article class="card ov-remaining"><span class="eyebrow">WHAT REMAINS UNRESOLVED</span><h2>${remaining.length ? "Evidence and validation still needed" : "Limits to keep in view"}</h2>${remaining.length ? `<ul>${remaining.slice(0, 2).map(text => `<li>${esc(ovExcerpt(text, 290))}</li>`).join("")}</ul>${remaining.length > 2 ? `<details><summary>See all ${remaining.length} recorded requirements</summary>${list(remaining)}</details>` : ""}` : '<p>No separate list of outstanding requirements is saved yet. Review the source result and its assumptions before relying on it.</p>'}${btn("Review evidence & gaps ↗", "nav", "evidence", "quiet")}</article></section>`;
}

function overviewWelcome() {
  const problems = records("workspace_problem").filter(r => !r.stale).slice().reverse();
  return `<div class="overview-view">${head("YOUR INVESTIGATIONS", "From a question to an inspectable answer.", "Bring the question and context. The agent investigates; you steer the work and review what it finds.", btn("New investigation ↗", "new", "", "primary"))}
    ${problems.length ? `<section class="ov-workspace-list"><div class="ov-section-title"><h2>Open an investigation</h2><p>See its current conclusion, actual outputs and next step.</p></div><div class="ov-problem-grid">${problems.map(problem => { const info = overviewState(problem.id); return `<button class="card ov-problem-card" data-action="select" data-id="${esc(problem.id)}"><span class="eyebrow">${esc(info.status)}</span><h3>${esc(problemLabel(problem))}</h3><p>${esc(ovExcerpt(ovPlain(info.answerText || problem.data.question), 180))}</p><span class="ov-problem-foot">${info.packages.length ? `${info.packages.length} saved computations` : info.model ? "Working model available" : "Question and context"}<b>Open investigation ↗</b></span></button>`; }).join("")}</div></section>` : `<section class="problem-hero"><span class="eyebrow">BEGIN WITH YOUR QUESTION</span><h2>What decision or uncertainty should we work on?</h2><p>Describe the outcome you need and the evidence you have. You can keep adding observations and expert guidance as the investigation develops.</p>${btn("Define the question ↗", "new", "", "primary")}</section>`}
    ${overviewLibrary()}</div>`;
}

function overview() {
  const problem = current();
  if (!problem) return overviewWelcome();
  const info = overviewState();
  return `<div class="overview-view">${head("YOUR INVESTIGATION", problemLabel(problem), "The current answer, saved work and next step — with the evidence limits attached.")}
    <section class="ov-goal"><div class="card-top"><span class="eyebrow">THE QUESTION YOU ASKED</span>${tag(info.status, info.waiting || info.gap || !info.answer)}</div><p>${esc(ovExcerpt(problem.data.question, 190))}</p>${problem.data.question.length > 190 ? `<details><summary>Read the complete question</summary><p>${esc(problem.data.question)}</p></details>` : ""}<div class="ov-launch">${overviewControls(problem, info)}<p>The agent chooses its next actions. You can add evidence or direction at any time.</p></div></section>
    <div class="ov-results-grid">${overviewAnswer(info)}${overviewOutput(info)}</div>${overviewNext(info)}${flow(info)}${overviewLibrary()}</div>`;
}
