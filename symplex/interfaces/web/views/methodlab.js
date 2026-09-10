"use strict";

// Operator controls use host gate responses. No render or refresh activates a method.
const methodLabCache = new Map();
const mlData = value => value?.data || value || {};
const mlArray = value => Array.isArray(value) ? value : [];
const mlCandidates = () => records("method_candidate").filter(r => !r.stale && (!selected || r.parent === selected)).slice().reverse();
const mlStages = [["development", "Development replay"], ["regression", "Regression suite"], ["holdout", "Protected fresh tasks"]];
const mlId = record => record?.id || record?.review_id || record?.deployment_id || "";
const mlCount = value => Number.isInteger(value) && value >= 0 ? String(value) : "Not recorded";

async function methodLabRefresh(candidateId = null) {
  const candidates = candidateId ? mlCandidates().filter(r => r.id === candidateId) : mlCandidates();
  await Promise.all(candidates.map(async candidate => {
    try {
      const result = await api("/method-lab/" + encodeURIComponent(candidate.id));
      methodLabCache.set(candidate.id, { result, fetched: new Date().toISOString() });
    } catch (error) {
      methodLabCache.set(candidate.id, { error: error.message });
    }
  }));
  if (page === "methodlab") renderMain();
}

function mlReportId(data, stage) {
  const item = data.report_ids?.[stage];
  return typeof item === "string" ? item : mlArray(item).at(-1) || mlData(data.stages?.[stage]).report_id || "";
}

function mlStageCard(data, stage, label) {
  const report = mlData(data.stages?.[stage]);
  const summary = mlData(report.summary || report);
  const id = mlReportId(data, stage);
  const present = !!id;
  return `<article class="ml-stage"><div class="card-top"><span class="eyebrow">${esc(label)}</span>${tag(present ? pretty(summary.status || "Report recorded") : "No report", !present || summary.status !== "verified")}</div>
    ${present ? `<dl>${evField("Task pairs", mlCount(summary.task_count))}${evField("Parent / candidate passes", `${mlCount(summary.parent_passes)} / ${mlCount(summary.child_passes)}`)}${evField("Paired regressions", mlCount(summary.paired_regressions))}${evField("Resource accounting", summary.matched_resources_verified === true ? "Verified by host" : "Not established")}</dl>${summary.error ? `<p class="ml-status-error">${esc(summary.error)}</p>` : ""}` : '<p>No completed report is linked to this stage. Missing evidence leaves its gate open.</p>'}
    <div class="actions">${present ? artifact({ id }, "Inspect stage report ↗") : ""}${summary.protocol_id ? artifact({ id: summary.protocol_id }, "Frozen stage") : ""}</div>
  </article>`;
}

function mlIdentityFields() {
  return '<label>Reviewer or operator<input name="actor" required maxlength="120" autocomplete="name" placeholder="Your name" /></label><label>Reason and remaining limitations<textarea name="reason" required maxlength="2000" rows="3" placeholder="Explain the evidence and scope behind this decision."></textarea></label>';
}

function mlProblemFields() {
  const problems = records("workspace_problem").filter(r => !r.stale);
  return `<fieldset><legend>Problems covered by this decision</legend><p class="tiny">A canary can affect only the explicitly selected problems on future invocations.</p><div class="ml-problem-options">${problems.map(problem => `<label class="ml-check"><input type="checkbox" name="problem_ids" value="${esc(problem.id)}" ${problem.id === selected ? "checked" : ""} /><span>${esc(problemLabel(problem))}<small>${esc(problem.id)}</small></span></label>`).join("")}</div></fieldset>`;
}

function mlFormFooter(label, disabled = false) {
  return `<p class="ml-form-message" role="status" aria-live="polite"></p><button type="submit" class="secondary" ${disabled ? "disabled" : ""}>${esc(label)}</button>`;
}

function mlReviewForm(candidate, data) {
  return `<details class="ml-control"><summary>Record a human review</summary><form data-method-form="review" data-candidate="${esc(candidate.id)}">
    <p>Approval records a reviewed problem scope. Activating a canary is a separate action, and the host rechecks the evidence. Rejection applies to the candidate and rolls back any of its active canaries.</p>
    ${mlStages.map(([stage]) => `<input type="hidden" name="report_${stage}" value="${esc(mlReportId(data, stage))}" />`).join("")}
    <label>Decision<select name="decision" required><option value="">Choose a decision</option><option value="approve" ${data.eligible === true ? "" : "disabled"}>Approve for a scoped canary</option><option value="reject">Reject this candidate</option></select></label>
    ${data.eligible === true ? "" : '<p class="tiny">Approval is unavailable while host gate requirements are outstanding.</p>'}
    ${mlIdentityFields()}${mlProblemFields()}
    <div class="ml-limits"><label>Execution failures before rollback<input type="number" name="execution_failure_limit" min="1" max="10" value="1" required /></label><label>Failed host assessments before rollback<input type="number" name="numerical_failure_limit" min="1" max="10" value="1" required /></label></div>
    ${mlFormFooter("Record review")}</form></details>`;
}

function mlShadowForm(candidate) {
  return `<details class="ml-control"><summary>Record a shadow scope</summary><form data-method-form="shadow" data-candidate="${esc(candidate.id)}"><p>A shadow designation preserves the baseline. Recording it does not execute a replay or change live instructions.</p>${mlIdentityFields()}${mlProblemFields()}${mlFormFooter("Record shadow scope")}</form></details>`;
}

function mlCanaryForm(data) {
  const latestReview = mlArray(data.reviews).at(-1);
  const reviews = latestReview && mlData(latestReview).status === "approved" ? [latestReview] : [];
  return `<details class="ml-control"><summary>Activate an approved scoped canary</summary>${reviews.length ? `<form data-method-form="canary"><p>The host revalidates the selected review. A canary changes instructions only for its approved problems and future invocations.</p><label>Approved review<select name="review_id" required><option value="">Choose a reviewed scope</option>${reviews.map(r => `<option value="${esc(mlId(r))}">${esc(mlId(r))} · ${esc(mlData(r).actor || "Operator")}</option>`).join("")}</select></label>${mlIdentityFields()}${mlFormFooter("Activate scoped canary")}</form>` : '<p>No approved review is available. Recorded proposals and replay counts cannot activate themselves.</p>'}</details>`;
}

function mlDeployment(record, rollbacks) {
  const data = mlData(record), id = mlId(record);
  const rolledBack = data.status === "rolled_back" || rollbacks.some(r => mlData(r).deployment_id === id);
  const canary = data.mode === "canary";
  return `<article class="ml-deployment"><div class="card-top"><span class="eyebrow">${esc(canary ? "Scoped canary" : "Shadow designation")}</span>${tag(rolledBack ? "Rolled back" : data.status || "Recorded", rolledBack || !canary)}</div>
    <dl>${evField("Recorded by", data.actor)}${evField("Reason", data.reason)}${evField("Problem scope", mlArray(data.problem_ids).join(" · "))}${evField("Execution failure limit", data.execution_failure_limit)}${evField("Host-assessment failure limit", data.numerical_failure_limit)}</dl>
    <div class="actions">${artifact({ id }, "Inspect deployment record ↗")}</div>
    ${canary && !rolledBack ? `<details class="ml-control"><summary>Roll back this canary</summary><form data-method-form="rollback"><input type="hidden" name="deployment_id" value="${esc(id)}" /><p>Rollback restores the baseline for future invocations. Existing run artifacts remain preserved.</p>${mlIdentityFields()}${mlFormFooter("Roll back canary")}</form></details>` : ""}
  </article>`;
}

function mlRecordedList(items, emptyText) {
  return items.length ? `<div class="ml-record-list">${items.map(record => { const data = mlData(record); return `<div><strong>${esc(pretty(data.status || data.decision || record.kind || "Recorded"))}</strong><p>${esc(data.reason || data.scope || "Inspect the recorded observations and linked results.")}</p>${data.actor ? `<p class="tiny">Recorded by ${esc(data.actor)}</p>` : ""}${data.problem_ids ? `<p class="tiny">Problem scope: ${esc(mlArray(data.problem_ids).join(" · "))}</p>` : ""}${artifact({ id: mlId(record) }, "Inspect record ↗")}</div>`; }).join("")}</div>` : `<p class="ev-unset">${esc(emptyText)}</p>`;
}

function mlObservations(items) {
  if (!items.length) return '<p class="ev-unset">No canary observations are recorded. An absence of observations is not evidence of successful monitoring.</p>';
  return `<div class="ml-record-list">${items.map(record => { const data = mlData(record); return `<div><strong>${esc(pretty(data.job_status || "Job status not recorded"))}</strong><dl>${evField("Job", data.job_id)}${evField("Execution failed", data.execution_failed === true ? "Yes" : data.execution_failed === false ? "No failure recorded" : "Not reported")}${evField("Failed host assessments", mlArray(data.numerical_failure_ids).length ? data.numerical_failure_ids.join(" · ") : "None linked to this observation")}</dl>${artifact(record, "Inspect canary observation ↗")}</div>`; }).join("")}</div>`;
}

function methodLabCandidate(candidate) {
  const cached = methodLabCache.get(candidate.id), data = cached?.result, method = candidate.data || {};
  const reviews = mlArray(data?.reviews), deployments = mlArray(data?.deployments), rollbacks = mlArray(data?.rollbacks);
  return `<article class="card ml-candidate"><div class="card-top"><span class="eyebrow">VERSIONED METHOD CANDIDATE</span>${tag(data ? pretty(data.state || "State not reported") : "Gate status not loaded", data?.eligible !== true)}</div>
    <h2>${esc(method.name || "Method candidate")}</h2><p class="ml-candidate-id">${esc(candidate.id)}</p>
    <div class="ml-proposal">${mlArray(method.changes).map(change => `<div><span class="eyebrow">${esc(pretty(change.role))}</span><h3>${esc(change.current_failure)}</h3><p>${esc(change.proposed_instruction)}</p><p class="tiny">Expected effect: ${esc(change.expected_effect)} · Regression risk: ${esc(change.regression_risk)}</p></div>`).join("")}</div>
    <div class="actions">${artifact(candidate, "Inspect proposed method ↗")}${btn("Refresh gate status", "method-lab-refresh", candidate.id, "quiet")}</div>
    ${cached?.error ? `<p class="ml-status-error" role="status">${esc(cached.error)}</p>` : ""}
    ${data ? `<div class="ml-gate-summary"><strong>${data.eligible === true ? "Host gate requirements met" : "Host gate requirements outstanding"}</strong>${mlArray(data.blockers).length ? list(data.blockers) : '<p>Review the recorded stages and limitations before making an operator decision.</p>'}<small>Gate status fetched ${esc(new Date(cached.fetched).toLocaleString())}. Actions are revalidated by the host.</small></div>
      <div class="ml-stages">${mlStages.map(([key, label]) => mlStageCard(data, key, label)).join("")}</div>
      <div class="ml-controls">${mlReviewForm(candidate, data)}${mlShadowForm(candidate)}${mlCanaryForm(data)}</div>
      <section class="ml-history"><h3>Human reviews</h3>${mlRecordedList(reviews, "No operator review has been recorded.")}</section>
      <section class="ml-history"><h3>Shadow and canary scopes</h3>${deployments.length ? deployments.map(r => mlDeployment(r, rollbacks)).join("") : '<p class="ev-unset">No deployment scope is recorded. Baseline instructions remain in use.</p>'}</section>
      <section class="ml-history"><h3>Monitoring and rollback</h3>${mlObservations(mlArray(data.observations))}${mlRecordedList(rollbacks, "No rollback has been recorded.")}</section>
      <p class="ml-scope">${esc(data.scope || "Method gates currently measure software-contract reliability. Scientific accuracy and decision utility are not established by these counts.")}</p>` : '<p class="ev-unset">Load the host gate record to inspect development, regression and protected fresh-task evidence. No approval or activation is inferred from the proposal.</p>'}
  </article>`;
}

function methodLabView() {
  const candidates = mlCandidates();
  return `<div class="method-lab-view">${head("METHOD LAB", "Improve the investigation method.", "Follow recorded evidence through evaluation, human review, scoped use and rollback. Every gate shows its actual state.", btn("Refresh gates", "method-lab-refresh", "", "secondary"))}
    <ol class="ml-lifecycle" aria-label="Method lifecycle"><li>Propose</li><li>Develop & regress</li><li>Protected fresh tasks</li><li>Human review</li><li>Shadow / canary</li><li>Monitor / roll back</li></ol>
    <div class="ml-boundary"><strong>Current evaluator: complexity-architect contract reliability</strong><p>Passing a schema and available-reference check does not establish scientific truth, useful decisions or general recursive self-improvement. A recorded shadow scope does not run a comparison. Activation requires an explicit operator action.</p></div>
    ${candidates.length ? `<div class="ml-candidates">${candidates.map(methodLabCandidate).join("")}</div>` : empty("No method candidate is available.", "A recorded evaluation failure can support a bounded instruction proposal. Development and protected replay evidence must be collected before a candidate can earn a reviewed scope.", btn("View design & improvement", "nav", "evolution", "quiet"))}</div>`;
}

document.addEventListener("submit", async event => {
  const form = event.target;
  const operation = form.dataset?.methodForm;
  if (!operation) return;
  event.preventDefault();
  if (busy) return;
  const fields = new FormData(form), submit = event.submitter, message = form.querySelector(".ml-form-message");
  const text = name => String(fields.get(name) || "").trim();
  const body = {actor: text("actor"), reason: text("reason")};
  busy = true;
  if (submit) submit.disabled = true;
  try {
    if (!body.actor || !body.reason) throw Error("Record an operator name and a reason.");
    if (["review", "shadow"].includes(operation)) {
      body.candidate_id = form.dataset.candidate;
      body.problem_ids = fields.getAll("problem_ids").map(String);
      if (!body.problem_ids.length && !(operation === "review" && text("decision") === "reject")) throw Error("Select at least one problem for this scope.");
    }
    if (operation === "review") {
      body.decision = text("decision");
      body.report_ids = Object.fromEntries(mlStages.map(([stage]) => [stage, text("report_" + stage)]));
      body.execution_failure_limit = Number(fields.get("execution_failure_limit"));
      body.numerical_failure_limit = Number(fields.get("numerical_failure_limit"));
      if (!body.decision) throw Error("Choose an explicit review decision.");
    } else if (operation === "canary") {
      body.review_id = text("review_id");
      if (!body.review_id) throw Error("Select an approved review.");
    } else if (operation === "rollback") body.deployment_id = text("deployment_id");
    else if (operation !== "shadow") throw Error("Unknown Method Lab action.");
    const result = await api("/method-lab/" + operation, body);
    await refresh(false);
    await methodLabRefresh();
    toast(result.status === "blocked" ? "Review recorded as blocked. Inspect the outstanding gates." : operation === "canary" ? "Scoped canary recorded. Future invocations use the approved scope." : operation === "rollback" ? "Rollback recorded for future invocations." : "Operator decision recorded.");
  } catch (error) {
    if (message) message.textContent = error.message;
    toast(error.message);
  } finally {
    busy = false;
    if (submit?.isConnected) submit.disabled = false;
  }
});
