"use strict";

// Saved designs, method proposals and host-computed comparisons stay separate.
const evArray = (value) => (Array.isArray(value) ? value : []);
const evFinite = (value) => typeof value === "number" && Number.isFinite(value);
const evNumber = (value) =>
  evFinite(value)
    ? new Intl.NumberFormat(undefined, {
        maximumSignificantDigits: 5,
        notation: Math.abs(value) >= 1e6 || (value !== 0 && Math.abs(value) < 1e-4)
          ? "scientific" : "standard",
      }).format(value)
    : "Not reported";
const evField = (label, value) =>
  value == null || value === ""
    ? ""
    : `<div class="ev-field"><dt>${esc(label)}</dt><dd>${esc(value)}</dd></div>`;
const evList = (items, fallback = "Not specified in this proposal.") =>
  evArray(items).length ? list(items) : `<p class="ev-unset">${esc(fallback)}</p>`;
const evLinks = (ids, label = "Inspect") =>
  `<div class="ev-artifact-links">${evArray(ids).map((id) => artifact({ id }, `${esc(label)} · ${esc(id)}`)).join("")}</div>`;
const evSystem = (id) => records("complex_system").find((record) => record.id === id);
const evSpec = (record) => record?.data?.spec || record?.data || {};
const evAlternative = (spec, id) =>
  evArray(spec?.decision?.alternatives).find((alternative) => alternative.id === id)?.name || id || "Unspecified";
const evSection = (id, title, note, contents) =>
  `<section id="ev-${id}" class="ev-section" aria-labelledby="ev-${id}-title"><div class="section-heading"><h3 id="ev-${id}-title">${esc(title)}</h3><span>${esc(note)}</span></div>${contents}</section>`;

function evolutionSummary(system, candidates, methods) {
  const model = evSpec(system);
  return `<div class="ev-layers">
    <article class="card ev-layer"><span class="eyebrow">01 / SYSTEM REPRESENTATION</span><h3>${esc(system ? model.title : "Explain the system")}</h3><p>${esc(system ? model.representation_rationale : "The agent proposes states, mechanisms, observation channels and hypotheses that can be challenged.")}</p><div class="ev-layer-footer">${system ? artifact(system, "Inspect system ↗") : '<span class="ev-unset">No system artifact yet</span>'}</div></article>
    <article class="card ev-layer"><span class="eyebrow">02 / EXPLORATORY INTERVENTIONS</span><h3>Change the intervention.</h3><p>Designs instantiate decision alternatives. Each change names a component, its expected effect and the experiments required for acceptance.</p><div class="ev-layer-footer"><a href="#ev-designs">${candidates.length} saved ${candidates.length === 1 ? "design" : "designs"} ↓</a></div></article>
    <article class="card ev-layer"><span class="eyebrow">03 / INVESTIGATION METHODS</span><h3>Test an instruction change.</h3><p>Method proposals link failures to instruction changes. Method Lab exposes replay gates, human review, scoped canaries and rollback records.</p><div class="ev-layer-footer">${btn("Open Method Lab ↗", "nav", "methodlab", "quiet")}<a href="#ev-methods">${methods.length} saved ${methods.length === 1 ? "proposal" : "proposals"} ↓</a></div></article>
  </div>`;
}

function evolutionCandidate(record) {
  const design = record.data || {};
  const system = evSystem(design.system_id);
  const spec = evSpec(system);
  const componentName = (id) => evArray(spec.components).find((component) => component.id === id)?.name || id;
  return `<article class="card ev-design">
    <div class="card-top"><span class="eyebrow">${esc(evAlternative(spec, design.alternative_id))}</span>${tag(design.status || "Proposed", true)}</div>
    <h3>${esc(design.name || "Candidate design")}</h3><p>${esc(design.objective)}</p>
    <div class="ev-design-spec">${esc(design.design_specification)}</div>
    <div class="ev-changes">${evArray(design.changes).map((change) => `<div class="ev-change"><span class="eyebrow">${esc(componentName(change.component_id))}</span><h4>${esc(change.description)}</h4><dl>${evField("Expected observable change", change.expected_observable_change)}${evField("Required checks", evArray(change.check_ids).join(" · "))}</dl></div>`).join("")}</div>
    <div class="ev-acceptance"><span class="eyebrow">ACCEPTANCE CRITERION</span><p>${esc(design.acceptance_criterion)}</p><span class="tiny">Required experiments: ${esc(evArray(design.required_experiment_ids).join(" · ") || "not specified")}</span></div>
    <div class="grid ev-space"><div><span class="eyebrow">EXPECTED TRADEOFFS</span>${evList(design.expected_tradeoffs)}</div><div><span class="eyebrow">FEASIBILITY GAPS</span>${evList(design.feasibility_gaps, "No feasibility gaps recorded. This does not establish feasibility.")}</div></div>
    <details class="ev-details"><summary>Assumptions, reversal conditions and evidence</summary><div class="grid"><div><span class="eyebrow">ASSUMPTIONS</span>${evList(design.assumptions)}</div><div><span class="eyebrow">REVERSE WHEN</span>${evList(design.reversal_conditions)}</div></div>${evArray(design.evidence_ids).length ? evLinks(design.evidence_ids, "Evidence") : '<p class="ev-unset">No supporting evidence artifacts linked.</p>'}</details>
    <div class="ev-lineage"><span class="tiny">${esc(design.scope || "Exploratory intervention proposal; feasibility and benefit require execution and evidence.")}</span><div class="actions">${artifact(record, "Inspect design ↗")}${design.system_id ? artifact({ id: design.system_id }, "System version") : ""}${design.parent_candidate_id ? artifact({ id: design.parent_candidate_id }, "Parent design") : ""}</div></div>
  </article>`;
}

function evolutionMethod(record) {
  const method = record.data || {};
  const evaluations = related("method_evaluation").filter((evaluation) => !evaluation.stale && evaluation.data?.candidate_id === record.id);
  return `<article class="card ev-method">
    <div class="card-top"><span class="eyebrow">METHOD PROPOSAL</span>${tag("Immutable proposal record", true)}</div>
    <h3>${esc(method.name || "Method candidate")}</h3><p class="ev-method-status">${esc(pretty(method.status || "Awaiting independent evaluation"))}</p>
    <div class="ev-changes">${evArray(method.changes).map((change) => `<div class="ev-change"><span class="eyebrow">${esc(pretty(change.role))}</span><h4>${esc(change.current_failure)}</h4><dl>${evField("Proposed instruction", change.proposed_instruction)}${evField("Expected effect", change.expected_effect)}${evField("Regression risk", change.regression_risk)}</dl></div>`).join("")}</div>
    <div class="grid"><div><span class="eyebrow">DEVELOPMENT TEST PLAN</span><p>${esc(method.development_test)}</p></div><div><span class="eyebrow">PLANNED INDEPENDENT TEST</span><p>${esc(method.independent_validation_test)}</p></div></div>
    <dl>${evField("Matched resource rule", method.matched_resource_rule)}${evField("Acceptance criterion", method.acceptance_criterion)}${evField("Rollback condition", method.rollback_condition)}</dl>
    ${evaluations.length ? evaluations.map(evolutionMethodEvaluation).join("") : '<p class="ev-unset">No paired replay is recorded for this candidate. The supported backend evaluates one complexity-architect instruction change against its frozen parent.</p>'}
    <details class="ev-details"><summary>Applicability, feedback and parent prompt versions</summary><div class="grid"><div><span class="eyebrow">APPLICABLE PROBLEMS</span>${evList(method.applicable_problem_characteristics)}</div><div><span class="eyebrow">TRANSFER LIMITS</span>${evList(method.transfer_limits)}</div></div>${evLinks(method.feedback_ids, "Feedback")}<div class="ev-versions">${Object.entries(method.parent_method || {}).map(([role, digest]) => `<div><span>${esc(pretty(role))}</span><code>${esc(digest)}</code></div>`).join("")}</div></details>
    <div class="ev-lineage"><p>This immutable proposal preserves its creation-time assumptions. Method Lab shows the current evaluation, review and deployment records.</p><div class="actions">${artifact(record, "Inspect method proposal ↗")}${btn("Review Method Lab state ↗", "nav", "methodlab", "quiet")}</div></div>
  </article>`;
}

function evolutionMethodEvaluation(record) {
  const result = record.data || {};
  const count = (value) => Number.isInteger(value) && value >= 0 ? String(value) : "Not recorded";
  const verified = result.matched_resources_verified === true;
  const status = result.status === "contract_reliability_improved_on_replay"
    ? "More contract passes on these tasks"
    : result.status === "no_contract_reliability_improvement"
      ? "No contract reliability improvement"
      : result.status === "inconclusive_resource_accounting"
        ? "Inconclusive resource accounting"
        : pretty(result.status || "Evaluation status unavailable");
  return `<section class="ev-acceptance"><div class="card-top"><span class="eyebrow">PAIRED CONTRACT REPLAY</span>${tag(status, !verified)}</div>
    <dl>${evField("Fresh task pairs", count(result.task_count))}${evField("Parent contract passes", count(result.parent_passes))}${evField("Candidate contract passes", count(result.child_passes))}${evField("Parent-pass / candidate-fail regressions", count(result.paired_regressions))}${evField("Matched resource checks", verified ? "Verified fixed settings, equal caps and recorded usage" : "Not established; inspect accounting")}</dl>
    <p>${esc(result.scope || "Software-contract reliability on supplied tasks only; scientific utility and generalization are not established.")}</p>
    <div class="actions">${artifact(record, "Inspect replay aggregate ↗")}${result.report_id ? artifact({ id: result.report_id }, "Detailed replay record") : ""}<span class="badge warn">No automatic promotion</span></div>
  </section>`;
}

function evolutionMetric(metric, comparison) {
  const estimate = evArray(comparison.intervals).find((value) => value.metric_id === metric.id);
  const delta = comparison.improvement_deltas?.[metric.id];
  const sign = evFinite(delta) && delta > 0 ? "+" : "";
  const interval = evFinite(estimate?.lower) && evFinite(estimate?.upper)
    ? `${evNumber(estimate.lower)} – ${evNumber(estimate.upper)}`
    : "No interval reported";
  return `<div class="ev-metric">
    <div class="ev-metric-name"><strong>${esc(metric.name || metric.id)}</strong><small>${esc(pretty(metric.direction))} · ${esc(metric.unit)}</small></div>
    <div class="ev-estimate"><strong>${evNumber(estimate?.value)}</strong><small>Reported range: ${esc(interval)}</small></div>
    <div class="ev-delta ${evFinite(delta) && delta > 0 ? "positive" : evFinite(delta) && delta < 0 ? "negative" : ""}"><strong>${sign}${evNumber(delta)}</strong><small>reported delta vs baseline</small></div>
  </div>`;
}

function evolutionTradeoffPlot(data, spec) {
  const metrics = evArray(data.metrics);
  if (metrics.length < 2) return "";
  const [xMetric, yMetric] = metrics;
  const points = evArray(data.comparisons).map((comparison) => ({
    id: comparison.candidate_id,
    x: evArray(comparison.intervals).find((value) => value.metric_id === xMetric.id)?.value,
    y: evArray(comparison.intervals).find((value) => value.metric_id === yMetric.id)?.value,
    frontier: evArray(data.pareto_alternative_ids).includes(comparison.candidate_id),
  })).filter((point) => evFinite(point.x) && evFinite(point.y));
  if (points.length < 2) return "";
  const extent = (key) => {
    const values = points.map((point) => point[key]);
    const min = Math.min(...values), max = Math.max(...values);
    const margin = (max - min || Math.abs(max) || 1) * .12;
    return [min - margin, max + margin];
  };
  const [xmin, xmax] = extent("x"), [ymin, ymax] = extent("y");
  if (![xmin, xmax, ymin, ymax, xmax - xmin, ymax - ymin].every(evFinite)) return "";
  const x = (value) => 90 + (value - xmin) / (xmax - xmin) * 470;
  const y = (value) => 245 - (value - ymin) / (ymax - ymin) * 195;
  const ticks = [0, .5, 1].map((fraction) => {
    const xvalue = xmin + (xmax - xmin) * fraction;
    const yvalue = ymin + (ymax - ymin) * fraction;
    return `<path class="ev-gridline" d="M${x(xvalue)} 50V245M90 ${y(yvalue)}H560"/><text x="${x(xvalue)}" y="266" text-anchor="middle">${esc(evNumber(xvalue))}</text><text x="78" y="${y(yvalue) + 3}" text-anchor="end">${esc(evNumber(yvalue))}</text>`;
  }).join("");
  return `<figure class="card ev-tradeoff"><span class="eyebrow">CANDIDATE TRADEOFFS · POINT ESTIMATES</span><svg viewBox="0 0 650 325" role="img" aria-label="Candidate point estimates for ${esc(xMetric.name)} and ${esc(yMetric.name)}. Pareto membership comes from all frozen metrics and reported diagnostics.">${ticks}<path class="ev-axes" d="M90 50V245H560"/>${points.map((point, index) => `<g><circle class="ev-point ${point.frontier ? "frontier" : ""}" cx="${x(point.x)}" cy="${y(point.y)}" r="7"/><text class="ev-point-label" x="${x(point.x) + 11}" y="${y(point.y) - 9}">${index + 1}</text><title>${esc(evAlternative(spec, point.id))}: ${esc(xMetric.name)} ${evNumber(point.x)}, ${esc(yMetric.name)} ${evNumber(point.y)}. ${point.frontier ? "On saved Pareto frontier" : "Outside saved Pareto frontier"}.</title></g>`).join("")}<text x="325" y="304" text-anchor="middle">${esc(xMetric.name)} · ${esc(xMetric.unit)} · ${esc(xMetric.direction)}</text><text transform="translate(20 150) rotate(-90)" text-anchor="middle">${esc(yMetric.name)} · ${esc(yMetric.unit)} · ${esc(yMetric.direction)}</text></svg><figcaption><div class="ev-plot-legend">${points.map((point, index) => `<span><b>${index + 1}</b> ${esc(evAlternative(spec, point.id))}${point.frontier ? " · frontier" : ""}</span>`).join("")}</div><p class="tiny">Two metrics are shown. The saved frontier uses every frozen metric and reported diagnostic. Baseline values and uncertainty are available in the linked output artifact.</p></figcaption></figure>`;
}

function evolutionNumerical(data) {
  const verification = data.numerical_verification || {};
  const passed = verification.status === "checked" && verification.all_passed === true;
  const failed = verification.status === "failed" || (verification.status === "checked" && !passed);
  const status = passed ? "Passed frozen numerical checks" : failed ? "Numerical checks failed" : "Numerical checks unavailable";
  const detail = passed
    ? "The host recomputed the declared checks over generated CSV values. This establishes only those numerical properties under the frozen limits."
    : failed
      ? "The generated outputs did not pass all frozen host checks. This comparison does not establish numerical eligibility."
      : "This record has no established passing host numerical verification. Earlier maker-reported flags are not a substitute.";
  return `<article class="card ev-space"><div class="card-top"><span class="eyebrow">HOST NUMERICAL VERIFICATION</span>${tag(status, !passed)}</div><p>${esc(detail)}</p><dl>${evField("Frozen checks", Number.isInteger(verification.check_count) ? String(verification.check_count) : "Not recorded")}${evField("Failed checks", evArray(verification.failed_check_ids).join(" · "))}</dl><p class="tiny">Independent empirical validation: ${data.independently_validated === true ? "reported separately; inspect the supporting evidence" : "not established"}.</p>${data.numerical_verification_id ? artifact({ id: data.numerical_verification_id }, "Inspect numerical checks and source hashes ↗") : ""}</article>`;
}

function evolutionComparison(record) {
  const data = record.data || {};
  const spec = evSpec(evSystem(data.system_id));
  const metrics = evArray(data.metrics);
  const comparisons = evArray(data.comparisons);
  return [
    `<div class="ev-comparison-scope"><div class="actions">${tag("Maker-declared basis · " + (data.basis || "unspecified"), true)}${tag("Exploratory comparison", true)}</div><p>${esc(data.scope || "Point estimates and reported diagnostics require independent validation before they can support scientific promotion.")}</p><p class="tiny">Positive deltas favor the metric's declared objective; negative deltas indicate degradation. They are not statistical significance or confidence scores.</p></div>`,
    evolutionNumerical(data),
    evolutionTradeoffPlot(data, spec),
    `<div class="grid ev-comparisons">${comparisons.map((comparison) => `<article class="card"><div class="card-top"><span class="eyebrow">CANDIDATE ALTERNATIVE</span>${tag(comparison.status || "Not established", true)}</div><h3>${esc(evAlternative(spec, comparison.candidate_id))}</h3><p>Compared with ${esc(evAlternative(spec, data.baseline_id))}</p><div class="ev-metrics">${metrics.map((metric) => evolutionMetric(metric, comparison)).join("")}</div><p class="tiny ev-diagnostic">Maker-reported diagnostic flags: ${(comparison.maker_diagnostics_passed ?? comparison.diagnostics_passed) === true ? "passed" : (comparison.maker_diagnostics_passed ?? comparison.diagnostics_passed) === false ? "did not all pass" : "not reported"}. These flags remain unverified; the separate host numerical result above controls numerical eligibility.</p></article>`).join("") || empty("No candidate comparison is recorded.", "Inspect the saved artifact to see which outputs were available.")}</div>`,
    `<article class="card ev-frontier ev-space"><span class="eyebrow">SAVED PARETO FRONTIER</span><h3>Alternatives retained for their tradeoffs</h3><div class="ev-frontier-members">${evArray(data.pareto_alternative_ids).map((id) => `<span>${esc(evAlternative(spec, id))}${id === data.baseline_id ? ' <small>baseline</small>' : ""}</span>`).join("") || '<p class="ev-unset">No alternatives retained under the reported diagnostics.</p>'}</div><p>Using the reported point estimates, no eligible alternative is at least as good on every metric and better on one. This retains tradeoffs under the frozen comparison.</p></article>`,
    `<details class="card ev-details ev-space"><summary>Frozen thresholds, uncertainty and limitations</summary><div class="ev-thresholds">${metrics.map((metric) => `<div><strong>${esc(metric.name || metric.id)}</strong><span>${esc(pretty(metric.direction))} · ${esc(metric.unit)}</span><small>Minimum improvement: ${evNumber(metric.minimum_improvement)} · Maximum degradation: ${evNumber(metric.maximum_degradation)}</small></div>`).join("")}</div><dl>${evField("Reported uncertainty method", data.uncertainty_method)}</dl>${evList(data.limitations)}<p class="tiny">Independent validation: ${data.independently_validated === true ? "reported" : "not established"}. Promotion: ${data.promotion_allowed === true ? "reported as enabled" : "disabled"}.</p></details>`,
    `<div class="actions ev-space">${artifact(record, "Inspect comparison ↗")}${data.protocol_id ? artifact({ id: data.protocol_id }, "Frozen protocol") : ""}${data.output_id ? artifact({ id: data.output_id }, "Executed output") : ""}${data.package_id ? artifact({ id: data.package_id }, "Compute package") : ""}</div>`,
  ].join("");
}

function evolutionView() {
  const system = latest("complex_system");
  const candidates = related("candidate_design").slice().reverse();
  const methods = related("method_candidate").slice().reverse();
  const comparison = latest("experiment_comparison");
  const support = ["hypothesis_review", "evidence_synthesis"].flatMap((kind) => related(kind));
  const action = current()
    ? btn("Launch solver ↗", "solve", selected, "primary")
    : btn("Open problem workspace ↗", "nav", "problem", "primary");
  return [
    '<div class="evolution-view">',
    head("DESIGN & IMPROVEMENT", "Compare designs and revisions.", "Inspect proposed system models, interventions, executed comparisons and suggested method changes.", action),
    evolutionSummary(system, candidates, methods),
    evSection("designs", "Exploratory intervention designs", "Versioned proposals tied to the complex system they change", candidates.length
      ? `<div class="ev-stack">${candidates.map(evolutionCandidate).join("")}</div>`
      : empty("No exploratory intervention has been proposed yet.", "The agent can turn a declared decision alternative into a concrete candidate, record the expected changes, and name the evidence and experiments required for acceptance.")),
    evSection("comparisons", "Latest executed comparison", "Frozen metrics, candidate outcomes and unresolved tradeoffs", comparison
      ? evolutionComparison(comparison)
      : empty("No executed comparison is available.", "The agent must freeze a protocol, execute the alternatives and produce a valid result artifact before the host can compare metrics or retain a Pareto frontier.")),
    evSection("methods", "Investigation methods and replay", "Proposals, frozen parent comparisons and contract reliability", methods.length
      ? `<div class="ev-stack">${methods.map(evolutionMethod).join("")}</div>`
      : empty("No method change proposed yet.", "The agent first needs recorded evaluation feedback. A method candidate includes development and independent validation tests, a resource rule, acceptance criteria and a rollback condition. Proposing it does not install it.")),
    support.length ? evSection("support", "Supporting reviews and synthesis", "Trace decisions back to their source artifacts", `<div class="card">${support.map((record) => `<div class="list-row"><div class="row-main"><strong>${esc(pretty(record.kind))}</strong><small class="mono">${esc(record.id)}</small></div>${artifact(record)}</div>`).join("")}</div>`) : "",
    '</div>',
  ].join("");
}
