"use strict";

// This view renders saved semantic contracts. It never creates domain outputs.
const cxArray = (value) => (Array.isArray(value) ? value : []);
const cxText = (value) =>
  value == null
    ? ""
    : typeof value === "object"
      ? JSON.stringify(value)
      : String(value);
const cxId = (value) =>
  "cx-component-" +
  Array.from(String(value ?? ""), (char) => char.codePointAt(0).toString(16)).join("-");
const cxLabel = (collection, id) => {
  const item = cxArray(collection).find((value) => value.id === id);
  return item?.name || item?.symbol || id || "Unspecified";
};
const cxDetail = (label, value) =>
  value == null || value === ""
    ? ""
    : `<div class="cx-detail"><dt>${esc(label)}</dt><dd>${esc(cxText(value))}</dd></div>`;
const cxList = (values, emptyText = "Not specified in this version.") => {
  const items = cxArray(values);
  return items.length
    ? `<ul class="detail-list">${items.map((item) => `<li>${esc(cxText(item))}</li>`).join("")}</ul>`
    : `<p class="cx-missing">${esc(emptyText)}</p>`;
};
const cxSources = (ids) => {
  const sources = cxArray(ids);
  return sources.length
    ? `<div class="cx-sources">${sources.map((id) => artifact({ id }, `Source · ${esc(id)}`)).join("")}</div>`
    : '<span class="cx-missing">No source artifacts linked.</span>';
};
const cxSection = (id, title, note, body) =>
  `<section class="cx-section" id="cx-${id}" aria-labelledby="cx-${id}-title"><div class="section-heading"><h3 id="cx-${id}-title">${esc(title)}</h3><span>${esc(note)}</span></div>${body}</section>`;

function complexityGraph(spec) {
  const components = cxArray(spec.components);
  if (!components.length) return "";
  const columns = components.length > 2 ? 3 : components.length;
  const rows = Math.ceil(components.length / columns);
  const width = columns * 250;
  const height = rows * 146 + 38;
  const positions = new Map(
    components.map((component, index) => [
      component.id,
      { x: (index % columns) * 250 + 125, y: Math.floor(index / columns) * 146 + 74 },
    ]),
  );
  const owners = new Map();
  for (const component of components)
    for (const port of cxArray(component.ports)) owners.set(port.id, component.id);
  const links = cxArray(spec.couplings)
    .map((edge) => {
      const source = positions.get(owners.get(edge.source_port));
      const target = positions.get(owners.get(edge.target_port));
      if (!source || !target) return "";
      const direction = target.x >= source.x ? 1 : -1;
      const x1 = source.x + direction * 99;
      const x2 = target.x - direction * 102;
      const bend = Math.max(32, Math.abs(x2 - x1) / 2);
      const path = source === target
        ? `M${source.x + 55} ${source.y - 31} C${source.x + 120} ${source.y - 93} ${source.x - 120} ${source.y - 93} ${source.x - 55} ${source.y - 34}`
        : `M${x1} ${source.y} C${x1 + direction * bend} ${source.y} ${x2 - direction * bend} ${target.y} ${x2} ${target.y}`;
      return `<path class="cx-edge" d="${path}" marker-end="url(#cx-arrow)"><title>${esc(edge.id)}: ${esc(edge.source_port)} → ${esc(edge.target_port)}. ${esc(edge.mechanism)}</title></path>`;
    })
    .join("");
  const nodes = components
    .map((component) => {
      const position = positions.get(component.id);
      const label = String(component.name || component.id || "Component");
      return `<a href="#${cxId(component.id)}" aria-label="Inspect ${esc(label)}"><rect class="cx-node" x="${position.x - 99}" y="${position.y - 32}" width="198" height="64" rx="7"/><text class="cx-node-title" x="${position.x}" y="${position.y - 3}" text-anchor="middle">${esc(label.length > 29 ? label.slice(0, 27) + "…" : label)}</text><text class="cx-node-kind" x="${position.x}" y="${position.y + 16}" text-anchor="middle">${esc(pretty(component.kind))}</text><title>${esc(label)}. ${esc(component.mechanism)}</title></a>`;
    })
    .join("");
  return `<div class="card cx-graph"><div class="card-top"><span class="eyebrow">TYPED COMPONENT CONNECTIONS</span><a href="#cx-components" class="cx-text-link">Inspect ports ↓</a></div><svg viewBox="0 0 ${width} ${height}" role="group" aria-label="Agent-proposed component connections. Component links lead to their port and mechanism details."><defs><marker id="cx-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7Z" fill="#8a9f86"/></marker></defs>${links}${nodes}</svg><p class="tiny">Layout represents connectivity, not physical position or effect size. Each connection is a proposed mechanism with an explicit data and time interface.</p></div>`;
}

function complexityObservations(spec) {
  const observations = cxArray(spec.observations);
  if (!observations.length)
    return empty("Observation model is missing.", "The agent must specify how available inputs measure the states that matter, including missingness and uncertainty.");
  return `<div class="grid">${observations.map((observation) => `<article class="card"><div class="card-top"><span class="eyebrow">${esc(pretty(observation.modality))}</span>${tag(observation.status || "Unspecified", true)}</div><h3>${esc(observation.name || observation.id)}</h3><dl>${cxDetail("Measures", cxArray(observation.state_ids).map((id) => cxLabel(spec.states, id)).join(" · "))}${cxDetail("Measurement process", observation.measurement_process)}${cxDetail("Missingness", observation.missingness)}${cxDetail("Uncertainty", observation.uncertainty)}</dl><span class="eyebrow">PROVENANCE</span>${cxSources(observation.source_artifact_ids)}</article>`).join("")}</div>`;
}

function complexityComponents(spec) {
  return `<div class="cx-stack">${cxArray(spec.components).map((component) => `<article class="card cx-component" id="${cxId(component.id)}"><div class="card-top"><span class="eyebrow">${esc(pretty(component.kind))}</span><span class="mono">${esc(component.id)}</span></div><h3>${esc(component.name)}</h3><p>${esc(component.mechanism)}</p><div class="cx-ports" role="list" aria-label="Input and output ports">${cxArray(component.ports).map((port) => `<div class="cx-port" role="listitem"><span class="cx-port-direction">${esc(pretty(port.direction))}</span><div><strong>${esc(cxLabel(spec.states, port.state_id))}</strong><small>${esc(port.id)}</small></div><div><strong>${esc(cxLabel(spec.units, port.unit_id))}</strong><small>${esc(cxLabel(spec.boundary?.scales, port.scale_id))}</small></div></div>`).join("") || '<p class="cx-missing">No ports defined.</p>'}</div><details class="cx-details"><summary>Assumptions and supporting artifacts</summary>${cxList(component.assumptions)}${cxSources(component.evidence_ids)}</details></article>`).join("") || empty("No components specified.", "Components should expose a mechanism and typed inputs and outputs before execution.")}</div>`;
}

function complexityCouplings(spec) {
  const ports = new Map();
  for (const component of cxArray(spec.components))
    for (const port of cxArray(component.ports))
      ports.set(port.id, `${component.name || component.id} · ${cxLabel(spec.states, port.state_id)}`);
  const couplings = cxArray(spec.couplings);
  if (!couplings.length) return '<p class="cx-missing">No component couplings specified.</p>';
  return `<div class="cx-stack">${couplings.map((coupling) => `<article class="card cx-coupling"><span class="eyebrow">${esc(coupling.id)}</span><h3>${esc(ports.get(coupling.source_port) || coupling.source_port)} <span aria-label="connects to">→</span> ${esc(ports.get(coupling.target_port) || coupling.target_port)}</h3><p>${esc(coupling.mechanism)}</p><dl>${cxDetail("Ports", `${coupling.source_port} → ${coupling.target_port}`)}${coupling.conversion ? cxDetail("Conversion", `target = source × ${coupling.conversion.scale} + ${coupling.conversion.offset}`) : ""}${cxDetail("Time alignment", pretty(coupling.time_alignment))}${cxDetail("Alignment rationale", coupling.alignment_rationale)}</dl></article>`).join("")}</div>`;
}

function complexityHypotheses(spec) {
  const hypotheses = cxArray(spec.hypotheses);
  if (!hypotheses.length) return empty("Rival explanations are not yet specified.", "A claim needs observable predictions and a test that could reject it.");
  return `<div class="cx-stack">${hypotheses.map((hypothesis) => `<article class="card"><div class="card-top"><span class="eyebrow">${esc(hypothesis.id)}</span><span class="cx-missing">Proposed explanation</span></div><h3>${esc(hypothesis.claim)}</h3><div class="grid"><div><span class="eyebrow">RIVAL EXPLANATIONS</span>${cxList(cxArray(hypothesis.rivals).map((id) => `${id} · ${cxArray(spec.hypotheses).find((rival) => rival.id === id)?.claim || "Unresolved hypothesis reference"}`))}<dl>${cxDetail("Components", cxArray(hypothesis.component_ids).map((id) => cxLabel(spec.components, id)).join(" · "))}</dl></div><div><dl>${cxDetail("Discriminating test", hypothesis.discriminating_test)}${cxDetail("Expected observation", hypothesis.expected_observation)}${cxDetail("Reject when", hypothesis.rejection_condition)}</dl></div></div><details class="cx-details"><summary>Linked evidence</summary>${cxSources(hypothesis.evidence_ids)}</details></article>`).join("")}</div><p class="tiny">These are proposed explanations and tests. Inspect linked comparisons and evidence for their outcomes.</p>`;
}

function complexityValidation(spec) {
  const validation = spec.validation || {};
  return `<div class="grid"><article class="card"><span class="eyebrow">IDENTIFIABILITY</span><h3>What the observations cannot resolve</h3>${cxList(validation.identifiability_gaps)}<dl>${cxDetail("Calibration plan", validation.calibration_plan)}</dl></article><article class="card"><span class="eyebrow">VALIDITY BOUNDARY</span><h3>Where the inference may fail</h3>${cxList(validation.extrapolation_limits)}</article></div><div class="cx-stack cx-space">${cxArray(validation.checks).map((check) => `<article class="card"><div class="card-top"><span class="eyebrow">${esc(check.id)}</span><span class="cx-missing">Required check · not a result</span></div><h3>${esc(check.name)}</h3><dl>${cxDetail("Method", check.method)}${cxDetail("Accept when", check.acceptance_condition)}${cxDetail("Components", cxArray(check.component_ids).map((id) => cxLabel(spec.components, id)).join(" · "))}</dl>${cxSources(check.evidence_ids)}</article>`).join("")}</div>`;
}

function complexityExperiments(spec) {
  const experiments = cxArray(spec.experiments);
  if (!experiments.length)
    return empty("A comparative experiment has not been specified.", "Define a baseline, candidate interventions, observed endpoints and a stopping rule before treating a comparison as evidence.");
  return `<div class="cx-stack">${experiments
    .map((experiment) => `<article class="card">
      <div class="card-top"><span class="eyebrow">${esc(experiment.id)}</span><span class="cx-missing">Experiment design · not a result</span></div>
      <h3>${esc(experiment.name)}</h3>
      <div class="cx-comparison"><div><span class="eyebrow">BASELINE</span><strong>${esc(cxLabel(spec.decision?.alternatives, experiment.baseline_alternative_id))}</strong></div><span class="cx-comparison-arrow" aria-hidden="true">↔</span><div><span class="eyebrow">CANDIDATE ALTERNATIVES</span>${cxList(cxArray(experiment.candidate_alternative_ids).map((id) => cxLabel(spec.decision?.alternatives, id)))}</div></div>
      <dl>${cxDetail("Hypotheses tested", cxArray(experiment.hypothesis_ids).join(" · "))}${cxDetail("Endpoints", cxArray(experiment.endpoint_ids).map((id) => cxLabel(spec.decision?.endpoints, id)).join(" · "))}${cxDetail("Method", experiment.method)}${cxDetail("Comparison controls", experiment.comparison_controls)}${cxDetail("Uncertainty plan", experiment.uncertainty_plan)}${cxDetail("Stopping rule", experiment.stopping_rule)}</dl>
    </article>`)
    .join("")}</div>`;
}

function complexityDecision(spec) {
  const decision = spec.decision || {};
  return `<article class="card cx-decision"><span class="eyebrow">DECISION OWNER · ${esc(decision.beneficiary || "Unspecified")}</span><h2>${esc(decision.question || "Decision not specified")}</h2><div class="cx-stack">${cxArray(decision.endpoints).map((endpoint) => `<div class="cx-endpoint"><span class="eyebrow">${esc(pretty(endpoint.direction))}</span><div><strong>${esc(endpoint.name)}</strong><p>${esc(endpoint.criterion)}</p><span class="tiny">Observable: ${esc(cxLabel(spec.states, endpoint.state_id))}</span></div></div>`).join("")}</div></article><div class="section-heading"><h3>Alternatives to compare</h3><span>Interventions are proposed, not ranked</span></div><div class="grid">${cxArray(decision.alternatives).map((alternative) => `<article class="card"><span class="eyebrow">${esc(alternative.id)}</span><h3>${esc(alternative.name)}</h3><p>${esc(alternative.rationale)}</p><dl>${cxArray(alternative.interventions).map((intervention) => cxDetail(cxLabel(spec.states, intervention.state_id), intervention.change)).join("") || '<div class="cx-missing">No intervention specified in this alternative.</div>'}</dl></article>`).join("")}</div><details class="card cx-details cx-space"><summary>Constraints and acceptance tests</summary>${cxArray(decision.constraints).map((constraint) => `<div class="cx-rule"><strong>${esc(constraint.description)}</strong><dl>${cxDetail("Check", constraint.test)}${cxDetail("States", cxArray(constraint.state_ids).map((id) => cxLabel(spec.states, id)).join(" · "))}</dl></div>`).join("") || '<p class="cx-missing">No constraints specified.</p>'}</details>`;
}

function complexityResilience(spec) {
  if (!spec.resilience) return "";
  return cxSection("resilience", "Disturbance and recovery", "Agent-proposed resilience study", `<div class="grid"><article class="card"><span class="eyebrow">DISTURBANCES</span>${cxArray(spec.resilience.disturbances).map((disturbance) => `<div class="cx-rule"><h3>${esc(disturbance.name)}</h3><dl>${cxDetail("Perturbation", disturbance.perturbation)}${cxDetail("Affected states", cxArray(disturbance.affected_state_ids).map((id) => cxLabel(spec.states, id)).join(" · "))}</dl></div>`).join("")}</article><article class="card"><span class="eyebrow">RECOVERY OBSERVABLES</span>${cxArray(spec.resilience.recovery_observables).map((observable) => `<div class="cx-rule"><h3>${esc(cxLabel(spec.states, observable.state_id))}</h3><dl>${cxDetail("Recovery criterion", observable.criterion)}${cxDetail("Observation window", observable.observation_window)}</dl></div>`).join("")}</article></div>`);
}


function complexityBoundary(spec) {
  const boundary = spec.boundary || {};
  return `<div class="grid"><article class="card"><span class="eyebrow">INSIDE THE MODEL</span>${cxList(boundary.included)}<span class="eyebrow cx-subheading">OUTSIDE THE MODEL</span>${cxList(boundary.excluded)}</article><article class="card"><span class="eyebrow">SCALES & CLOCKS</span>${cxArray(boundary.scales).map((scale) => `<div class="cx-rule"><h3>${esc(scale.name)}</h3><dl>${cxDetail("Clock", scale.clock)}${cxDetail("Step", scale.step_seconds == null ? "No fixed time step specified" : `${scale.step_seconds} seconds`)}</dl></div>`).join("") || '<p class="cx-missing">No scales defined.</p>'}</article></div><details class="card cx-details cx-space"><summary>Entities, state meanings and units</summary><div class="cx-stack">${cxArray(spec.entities).map((entity) => `<div class="cx-rule"><h3>${esc(entity.name)}</h3><p>${esc(entity.meaning)}</p>${cxArray(spec.states).filter((item) => item.entity_id === entity.id).map((item) => `<div class="cx-state"><strong>${esc(item.name)}</strong><span class="tiny">${esc(pretty(item.kind))} · ${esc(cxLabel(spec.units, item.unit_id))} · ${esc(cxLabel(boundary.scales, item.scale_id))}</span><p>${esc(item.meaning)}</p></div>`).join("")}</div>`).join("")}</div>${cxArray(spec.units).length ? `<div class="cx-unit-list"><span class="eyebrow">UNIT DEFINITIONS</span>${cxArray(spec.units).map((unit) => `<div class="cx-rule"><strong>${esc(unit.symbol || unit.id)}</strong><dl>${cxDetail("Dimension", unit.dimension)}${cxDetail("Canonical conversion", `value × ${unit.scale_to_canonical} + ${unit.offset_to_canonical}`)}</dl></div>`).join("")}</div>` : ""}</details>`;
}

function complexityFeedback(spec) {
  return `<div class="grid">${cxArray(spec.feedback).map((loop) => `<article class="card"><div class="card-top"><span class="eyebrow">${esc(loop.id)}</span>${tag(loop.polarity || "Unspecified", true)}</div><h3>${esc(loop.description)}</h3><dl>${cxDetail("Components", cxArray(loop.component_ids).map((id) => cxLabel(spec.components, id)).join(" → "))}${cxDetail("Nonlinearity", loop.nonlinearity)}${cxDetail("Stochasticity", loop.stochasticity)}</dl></article>`).join("")}</div>`;
}

function complexityEvidenceLinks() {
  const kinds = ["hypothesis_review", "experiment_comparison"];
  const artifacts = kinds.flatMap((kind) => related(kind));
  if (!artifacts.length) return "";
  return cxSection(
    "review-artifacts", "Saved reviews and comparisons", "Inspect the actual artifact and its scope",
    `<div class="card">${artifacts.map((record) => `<div class="list-row"><div class="row-main"><strong>${esc(pretty(record.kind))}</strong><small class="mono">${esc(record.id)}</small></div>${artifact(record)}</div>`).join("")}</div>`,
  );
}

function complexitySteering(spec) {
  return `<div class="grid"><article class="card"><span class="eyebrow">ASSUMPTIONS TO CHALLENGE</span>${cxList(spec.assumptions)}</article><article class="card"><span class="eyebrow">UNRESOLVED QUESTIONS</span>${cxList(spec.unresolved_questions)}</article></div><div class="card cx-steering cx-space"><div><h3>Add observations or change the frame.</h3><p>Attach source material, explain a constraint, or challenge a mechanism. The next agent revision can use that context while preserving this version.</p></div><div class="actions">${btn("Add evidence", "nav", "evidence")}${btn("Steer the next investigation", "steer-suggest", "Before the next investigation, I want to revise or challenge the current system representation: ")}${btn("See executed results", "nav", "lab")}</div></div>`;
}

function complexityView() {
  const record = latest("complex_system");
  const create = current()
    ? btn(record ? "Revise system model ↗" : "Propose system model ↗", "complex-system", selected, "primary")
    : btn("Define a problem ↗", "new", "", "primary");
  if (!record) {
    return [
      head("COMPLEX SYSTEM", "Give the problem a model you can inspect.", "The agent connects observations, mechanisms, rival explanations and decisions across the scales the problem requires."),
      empty("No system representation saved yet.", "Start from a problem and available context. Astra will propose typed components, explicit couplings, competing hypotheses and validation requirements. Missing evidence remains visible.", create),
      `<div class="grid cx-space">
        <article class="card"><span class="eyebrow">PERCEPTION → REPRESENTATION</span><h3>Describe how inputs measure the system.</h3><p>Images, documents, measurements and human judgments should map to system states through an explicit observation process.</p></article>
        <article class="card"><span class="eyebrow">REPRESENTATION → DECISION</span><h3>Mechanisms need discriminating tests.</h3><p>Compare proposed interventions through measurable endpoints, rival hypotheses and checks that expose when the model fails.</p></article>
      </div>`,
    ].join("");
  }
  const spec = record.data?.spec || record.data || {};
  const counts = [
    [cxArray(spec.components).length, "components"],
    [cxArray(spec.observations).length, "observation channels"],
    [cxArray(spec.hypotheses).length, "hypotheses"],
    [cxArray(spec.validation?.checks).length, "required checks"],
  ];
  const navigation = [
    ["boundary", "Boundary"], ["observations", "Observations"],
    ["components", "Components"], ["hypotheses", "Hypotheses"],
    ["experiments", "Experiments"], ["validation", "Validation"],
    ["decision", "Decision"],
  ];
  const versions = related("complex_system");
  return [
    '<div class="complexity-view">',
    head("COMPLEX SYSTEM", spec.title || "Complex system specification", "A versioned representation of the problem, its mechanisms, observations and decision requirements.", create),
    `<div class="cx-artifact-strip"><div class="actions">${tag(record.data?.status || "Proposed model", true)}${record.data?.authority ? `<span class="tiny">${esc(cxText(record.data.authority))}</span>` : ""}</div>${artifact(record, "Inspect full contract ↗")}</div>`,
    '<p class="cx-scope-note">This is an agent-authored specification. Linked execution results and validation evidence determine which claims it can support.</p>',
    `<nav class="cx-jump-nav" aria-label="Complex system sections">${navigation.map(([id, label]) => `<a href="#cx-${id}">${label}</a>`).join("")}</nav>`,
    `<div class="cx-counts">${counts.map(([count, label]) => `<div><strong>${count}</strong><span>${label}</span></div>`).join("")}</div>`,
    spec.representation_rationale
      ? `<article class="card cx-representation"><span class="eyebrow">WHY THIS REPRESENTATION</span><p>${esc(spec.representation_rationale)}</p></article>`
      : "",
    complexityGraph(spec),
    cxSection("boundary", "Boundary, entities and scales", "Explicit inclusions and omissions", complexityBoundary(spec)),
    cxSection("observations", "Multimodal observation model", "How each input is proposed to measure system states", complexityObservations(spec)),
    cxSection("components", "Hybrid components and typed ports", "Meaning, units and scale at every interface", complexityComponents(spec)),
    cxSection("couplings", "Couplings and synchronization", "Inspect conversions and temporal assumptions", complexityCouplings(spec)),
    cxArray(spec.feedback).length
      ? cxSection("feedback", "Feedback, nonlinearity and stochasticity", "Mechanisms to challenge", complexityFeedback(spec))
      : "",
    cxSection("hypotheses", "Rival hypotheses and discriminating tests", "What would change the explanation", complexityHypotheses(spec)),
    cxSection("experiments", "Comparative experiment designs", "Controls and stopping criteria before execution", complexityExperiments(spec)),
    cxSection("validation", "Validation and inference limits", "Required checks remain separate from their outcomes", complexityValidation(spec)),
    spec.complexity
      ? cxSection("mechanisms", "Properties of the complex system", "Declared mechanisms and unresolved structure", `<div class="grid">${Object.entries(spec.complexity).map(([key, value]) => `<article class="card"><span class="eyebrow">${esc(pretty(key))}</span><p class="cx-mechanism">${esc(cxText(value))}</p></article>`).join("")}</div>`)
      : "",
    complexityResilience(spec),
    complexityEvidenceLinks(),
    cxSection("decision", "From inference to a decision", "Endpoints, alternatives and constraints", complexityDecision(spec)),
    cxSection("steering", "Your judgment shapes the next investigation", "Additional context stays attached to the problem", complexitySteering(spec)),
    versions.length > 1
      ? `<details class="card cx-details cx-space"><summary>${versions.length} saved specifications</summary><p>Previous specifications are preserved as artifacts. This view shows the latest version for the selected problem.</p><div class="cx-sources">${versions.slice().reverse().map((version) => artifact(version, esc(version.id))).join("")}</div></details>`
      : "",
    '</div>',
  ].join("");
}
