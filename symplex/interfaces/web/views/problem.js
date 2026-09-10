function problemView() {
  const p = current();
  if (!p)
    return (
      head(
        "01 / DEFINE",
        "Start with the decision.",
        "State the question, who needs the result, and how it will be assessed.",
      ) +
      empty(
        "What are we solving?",
        "Describe who needs a decision, what they can change, and the cost of getting it wrong.",
        btn("New problem ↗", "new", "", "primary"),
      )
    );
  const dna = latest("problem_dna"),
    sol = latest("solution");
  return (
    head(
      "01 / DEFINE",
      problemLabel(p),
      "The agent proposes a decision frame, assumptions, evidence gaps and hypotheses to test.",
      btn("Launch solver ↗", "solve", p.id, "primary"),
    ) +
    `<details class="card problem-detail"><summary>Full problem statement</summary><p>${esc(p.data.question)}</p></details>` +
    flow() +
    `<div class="actions">${btn("Frame with Astra", "plan", p.id)}${btn("Design solution", "design", p.id)}${btn("Explore scenarios", "scenarios", p.id)}${artifact(p)}</div>` +
    (dna
      ? `<div class="section-heading"><h3>Problem DNA</h3>${artifact(dna)}</div><div class="grid"><div class="card"><span class="eyebrow">BENEFICIARY & DECISION</span><h3>${esc(dna.data.beneficiary)}</h3><p>${esc(dna.data.decision)}</p><span class="eyebrow">OBJECTIVE</span><p>${esc(dna.data.objective)}</p></div><div class="card"><span class="eyebrow">CONSTRAINTS</span>${list(dna.data.constraints)}<br><span class="eyebrow">ASSUMPTIONS</span>${list(dna.data.assumptions)}</div></div>`
      : `<div class="onboarding"><span class="step-number">01</span><div><h3>Frame the investigation.</h3><p>Launch the solver to frame the objective, identify evidence gaps and choose an investigation.</p></div></div>`) +
    (sol
      ? `<div class="section-heading"><h3>Competing explanations</h3><span>Each one can be challenged</span></div><div class="grid">${sol.data.hypotheses.map((h, i) => `<article class="card"><span class="eyebrow">HYPOTHESIS ${i + 1}</span><h3>${esc(h.mechanism)}</h3><p>${esc(h.observable_prediction)}</p><span class="eyebrow">WHAT WOULD COUNT AGAINST IT</span><p>${esc(h.disconfirmation)}</p><span class="eyebrow">TEST</span><p>${esc(h.test)}</p></article>`).join("")}</div>`
      : "") +
    `<div class="section-heading"><h3>Revise an assumption</h3><span>Creates a version; marks prior descendants stale</span></div><form class="inline-form" id="revision-form"><input id="revision-input" required minlength="3" maxlength="2000" placeholder="An assumption that should change…" aria-label="Revised assumption"><button class="secondary">Create revision</button></form>`
  );
}
