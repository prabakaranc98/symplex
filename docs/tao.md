# Symplex — A problem-solving engine for complex systems

Symplex connects a question to observations, hypotheses, interacting system components, executable models and recorded comparisons. The agent can revisit any of these as results or user inputs change. The question determines which model families and scales belong in the investigation.

## Two interacting layers

The **investigation and execution layer** frames problems, acquires and synthesizes evidence, proposes representations, constructs models, estimates states/parameters, executes simulations and compares results. Its outputs are versioned artifacts and reusable files.

The **reasoning and improvement layer** chooses the next useful operation, challenges representation choices and assumptions, diagnoses failure, allocates bounded computation, and proposes revisions. It may return to framing, semantics or observation design after a failed computation. It is not a fixed sequence of reports. The controller is `agents/solver.py`; actual role boundaries, contracts and tools are exposed by `agents/roster.py` and the in-app Agent workflow.

Versions are tracked at three levels:

- The **system representation** describes mechanisms and behavior. `ComplexSystemSpec` separates observations from latent states, model families, units/clocks, interfaces, feedback, rivals and tests.
- The **candidate solution** describes a proposed change to that system. `CandidateDesign` links an exploratory intervention to specific components, a declared alternative, evidence and experiments that actually include that alternative.
- The **investigation method** describes how the engine chooses representations, experiments and revisions. `MethodCandidate` is a frozen proposal derived from recorded evaluation feedback. It is not an installed prompt change.

## How someone uses it

A user describes a situation, starts from a use-case brief, or supplies documents/data. They specify the decision or estimation target, constraints and what would make a result useful. Astra frames Problem DNA and chooses tools under the remaining resource envelope.

The user can inspect the Complex system view, challenge a state definition or hypothesis, attach evidence, and steer through chat. Astra can synthesize source disagreements, revise the representation, freeze a comparison, write and execute code in hosted Python, and inspect failures. Each decision boundary records the available inputs and highlights additions, changes and removals. An explicit reframe action can revise Problem DNA against those inputs while preserving the prior frame. A clarification reply resumes its waiting checkpoint; new inputs after completion create a new investigation identity.

For an explicit computation revision, a predecessor package supplies its generated files within the input budget. The archive-driven `evolve_model` action additionally requires included Python source and a parent evaluated under the current frozen protocol. Role prompts are frozen for the invocation; a proposed method change cannot edit the instructions governing its own run.

Modeling studio shows real output files and plots. Design & improvement separates representations, interventions and method proposals. Method Lab exposes recorded stage gates, human reviews, scoped canaries, failure observations and rollback decisions. Deliverables shows linked claims, alternative assessments, limitations, reversal conditions, next actions and reusable artifacts. A graph or visual inherits the source artifact's scope; it does not establish validity through appearance.

## Meaning across representations

Intent sets endpoints and constraints. Semantic definitions identify entities, quantities, observations and source claims. Structure identifies mechanisms, interactions, clocks, scales and feedback. Mathematics specifies equations, distributions, rules, observation operators and uncertainty. Computation implements those choices and produces recorded outputs. Evaluation tests numerical behavior, compares predictions or conditions, and assesses the scope of a decision claim.

The present host checks declared references, units/affine conversions, interface direction, clocks, feedback cycles and experimental coverage. An optional `symplex_model_map.json` accounts for every component with a saved Python symbol or an explicit omission, and can bind state outputs to CSV columns and declared units. The host verifies file provenance, parses source symbols without executing them, and checks the declared columns. These links do not prove that the symbol ran or that a semantic concept became the correct equation. General symbolic proof, automatic causal identification and certified semantic-to-equation compilation remain extension work.

## Synthesis science and synthetic data

The question determines which evidence, theory and analytical methods must be assembled. Symplex's synthesis role records claim provenance, measurement limits, source dependence, disagreements and evidence gaps. An anomaly may reflect noise, a regime change or an omitted mechanism; its interpretation remains a question for investigation. This follows the question-led integration described in the [synthesis-science reference](https://nscore.columbia.edu/content/synthesis-science).

Synthetic observations serve separate purposes: exercising a pipeline, testing recovery of known latent structure, stress-testing missingness or noise, and exploring conditional behavior. The generator, seeds, known truth, sampling support and assumptions should be preserved. Synthetic data can test inference mechanics; empirical evidence assesses whether those mechanics describe a real system. Combining multiple datasets or disciplines does not make a generated dataset observed.

## Improvement and evaluation

The host binds each frozen protocol to its system version and execution inputs, then checks generated CSV values against its declared numerical limits. Every collected computation package receives an assessment: checked, failed or unavailable. The result includes source hashes, check outcomes and repair requirements. Summary-metric deltas and nondominated alternatives are computed from maker-reported numbers; those numbers, diagnostic flags, intervals and declared observation origins remain separate from host numerical verification.

The search archive retains evaluated packages in groups defined by declared component kinds, evidence basis and frozen protocol. The fraction of frozen checks passed is its numerical diagnostic score. Scores are compared only within an identical protocol, and visit counts guide parent selection. Evolution selects an eligible predecessor under the current protocol, supplies its source and results, executes the proposed revision and assesses the outputs. This is an implemented program-revision loop; its diagnostic score does not measure scientific accuracy or establish that a revision improved the real system.

The method-replay backend evaluates one `complexity_architect` instruction change against frozen parent prompts and a fixed software contract. Development, regression and protected fresh-task stages retain separate protocols. An operator supplies tasks after freezing; exact repeats, originating feedback and previously investigated problems are rejected. Both arms use matching model settings and equal persistent resource caps, with no access to the other arm's outputs. Detailed tasks and outputs remain under the replay protocol; only aggregate pass counts, regressions and accounting status return to the originating problem. Missing or unsettled accounting makes the comparison inconclusive.

The Method Lab registry verifies the first frozen report for each stage and the ordering of successive stages. Paired regressions block approval, and the held-out stage must show a strict gain in contract passes. A human review records its evidence, reason, problem allowlist and failure limits. It does not activate the candidate. A separate operator action can start a canary only after the host revalidates the latest approved review, stage evidence and baseline prompt version. A shadow designation preserves the baseline and does not itself execute a comparison.

Canaries apply a frozen instruction overlay only to approved problems on future solver invocations; their method version participates in job identity. Recorded job failures and failed host assessments are counted against the reviewed limits. A threshold breach, invalidated lineage, explicit rollback or operator rejection restores the baseline for future invocations. Running invocations retain their frozen instructions. Method Lab presents those actual records without converting absent reports or absent monitoring observations into successful gates.

This replay measures contract parsing and available evidence-reference linkage. It does not assess scientific truth, task utility or semantic novelty. Other role evaluators, automatic promotion and broad operational drift evaluation remain unimplemented. Recursive improvement would additionally require evidence that subsequent revision sequences improve the engine's ability to make further validated improvements. This release does not claim that result.

## Release scope

Recorded assumptions, action rationale, source references and tool outcomes make an investigation inspectable. They are saved artifacts, not access to a model's private chain of thought.

The first release focuses on formulation, synthesis, representation, modeling, fitting, estimation, prediction, simulation, comparisons and bounded exploratory interventions. Optimization, reinforcement learning, game/mechanism design, production deployment and adaptive control are subsequent-release work. A generic hosted Python capability does not establish a validated implementation of those systems.

The app includes graph and spatial views, plots, and interactive 3D replay of recorded state. CSV projections select actual time and coordinate columns with exact row filters; source hashes and run IDs remain linked. Playback steps through recorded frames, and missing samples stay absent. A phase-space view does not infer physical geometry. A scientific city or social simulation additionally needs a behavioral model, an execution runtime, coordinate definitions and domain evaluation; those specialist adapters are not implemented. The six [use-case resource briefs](use-case-resources.md) describe the required inputs and dependencies.
