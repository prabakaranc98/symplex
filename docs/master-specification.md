Downstream of Intelligence

A GPT-6 Astra problem-solving engine: define, synthesize, compose, train, test and deliver

Research, product, modeling, engineering, evaluation and four-hour hackathon specification

Version 2.0 · 10 September 2026 · Five domains · Ten empirical problem statements

This master specification consolidates both supplied markdown references and the current conversation. The later requirements for GPT-6, ten empirical use cases, explicit evolutionary search and end-to-end decision artifacts govern the build. Earlier simulator-only examples remain optional fixtures or later research directions.

Core proposition: A real-world problem-solving engine that turns a user’s objective and heterogeneous evidence into a working, evaluated solution. It can define the problem, select or construct models, compose them with rules and tools, train suitable components, write and integrate code, test alternatives, and deliver a useful forecast, operational workflow, decision aid or experimental proposal. Complex-system modeling, evolutionary synthesis and metareasoning are core mechanisms inside this larger lifecycle.

Scope: Prediction Markets; Urban Infrastructure; Climate and Agriculture; Social Simulation; Biology. This document specifies the engine and its use cases. It does not claim that the software has been implemented, that a hypothesis has been validated, or that a longstanding problem has been solved.

Four-hour goal: one complete investigation with a prepared real-data slice and held-out comparison, one tested research-policy revision if time permits, and five inspectable domain packs containing ten empirical problem specifications. The default is a prediction-market consistency/reliability case. If the market archive is not prepared, use a prepared traffic-network slice. A synthetic simulator can test plumbing and counterfactual assumptions, but cannot replace the promised real-data comparison. Packs are marked specification-only, data-verified, executable or independently evaluated according to actual status.

Reading guide: GPT-6 evolution/metareasoning/RSI is section 1. Model reuse, training and fusion is section 5. The ten empirical use cases are sections 7–11. The full lifecycle and extension contract are section 12; the four-hour schedule is section 15; the Codex handoff is section 20.

1. The central engine: GPT-6 proposal, executable evolution, metareasoning and method revision

This is the project's technical center. GPT-6 constructs and evolves candidate solutions: model assemblies, data pipelines, inference/training recipes, code, tool workflows and decision policies. Scientific programs are one important artifact within that solution. An evaluator checks the resulting task performance; an archive preserves useful alternatives; GPT-6 selects the next investigation from actual evidence; a separate outer process tests changes to that improvement machinery.

The complete SolutionSpec binds a problem version to an evidence pipeline, model/component graph, training and inference plan, tool workflow, decision contract, serving/export interface, resource envelope and immutable evaluator reference. A solution may be a forecast service, an operator watchlist, a crop-risk workflow, a reputation diagnostic or a biological experiment planner. Success is evaluated at the user’s outcome and constraints as well as at intermediate model metrics. Where real action outcomes are not observed, the delivery identifies the remaining validation instead of pretending predicted utility is measured impact.

AlphaEvolve provides a precedent for model-generated programs, automated evaluation and an evolutionary program database. We adopt that general pattern with GPT-6 and add evidence/measurement semantics, system-model synthesis and research-policy evaluation. This is an architectural proposal, not an AlphaEvolve implementation or a demonstrated algorithmic novelty. AlphaEvolve, primary description.

flowchart TD
    P["Problem DNA and evidence"] --> R["GPT-6 metareasoner"]
    R --> G["GPT-6 program proposer"]
    G --> X["Compile, wire and execute"]
    X --> J["Protected numerical evaluator"]
    J --> A["Versioned model archive"]
    A --> R
    J --> D["GPT-6 judgment and decision artifact"]
    A --> M["GPT-6 method-revision proposer"]
    M --> T["Fresh-task policy evaluation"]
    T -->|"retained policy version"| R

The four mechanisms, explicitly

Mechanism

Input to GPT-6

What it produces

Independent authority

Scientific-program evolution

Problem contract, evidence, parent code/spec, diagnostics, archive summaries and permitted edits

A child program or typed patch, changed mechanism, testable expectation and required checks

Compiler, runtime and empirical evaluator determine validity and performance.

Metareasoning

Current candidates, unresolved discrepancies, available evidence/tools, completed actions and remaining budget

Next action: retrieve, reconcile, mutate, fit, ablate, expand computation, compare or stop

Scheduler validates availability, preconditions and resource limits.

Bounded method improvement

Development-task traces and identified process failure

A versioned change to proposal instructions, diagnostic routing, action selection or an allowed helper

Frozen parent/child policies are tested on separate tasks.

Recursive improvement research

Multiple method generations and their measured downstream improvements

A proposed change to how future method improvements are generated or selected

New independent task suites compare improvement productivity under equal total budgets.

The GPT-6 weights remain fixed in the first product. Editable artifacts are scientific programs and a limited part of the external research harness. Claiming model-weight learning would require an actual training pipeline and its own evidence.

Loop A — evolve scientific programs

Candidate genome: template/version; component and coupling graph; data-transformation recipe; observation model; parameter domains; solver/inference plan; supported queries; code; evidence references; parent IDs; and applicable constraints.

Permitted mutations: add/remove an allowed lag or coupling; introduce a shared driver; add a bias or missingness model; change a compatible observation/noise model; replace a component; adjust resolution; repair a demonstrated alignment error; or change inference strategy if the backend supports it. A mutation includes its expected observable effect and the result that would count against it. Recombine only components with compatible meaning, units, timing and interfaces.

Population procedure: seed the archive with a competent baseline and one distinct template. Select an eligible parent, ask GPT-6 for a small evidence-linked patch, apply/compile it, run cheap conformance checks, execute its experiment, score against the same development protocol, and retain eligible candidates. Preserve losing and invalid branches with their reasons. The MVP must execute at least one complete parent–child generation; target two small generations if the measured budget permits.

Fitness contract: minimize the predeclared development loss subject to semantic validity, executable validity and resource limits. For P1 this can be Brier loss; false relation assertions remain a separate eligibility/quality check. Keep calibration, robustness and cost visible rather than hiding all trade-offs in an arbitrary weighted sum. Compare candidates only when target, resolution version, evidence entitlement, cutoff, split and evaluator match. If evidence changes, rescore affected comparators under the same entitlement or explicitly compare quality–cost curves.

Small archive: an initial sparse 3 × 3 grid can use program descriptors: coupling class (independent, shared_driver, lagged_graph) and observation class (simple, bias_or_staleness, missingness_aware). Populate only cells supported by the domain grammar. Keep the best eligible development candidate per cell and preserve lineage. This is a proposed small quality-diversity implementation; occupied cells are not posterior probabilities or independent scientific confirmations. MAP-Elites is the relevant archive/search precedent; descriptor coverage measures diversity, not scientific truth. MAP-Elites primary paper.

Loop B — decide what research to do next

The metareasoner chooses among different kinds of progress, not only which model to sample. For example, a comparison may be failing because two markets refer to different weather stations. Another parameter search will not resolve that; a semantic check can.

Return an ActionProposal with action_type, target_ids, uncertainty_addressed, preconditions, tool_plan, estimated_cost, expected_observable_change, completion_condition, and stop_condition. The rationale is a concise auditable decision justification. Expected usefulness is a forecast to evaluate, not a measured result.

Use a deterministic diagnostic policy as a baseline. Let GPT-6 propose actions in the adaptive arm, then enforce the same tool menu and budget. Repeatedly compare predicted usefulness with realized improvement or ambiguity reduction; those traces are inputs to method revision.

Loop C — test improvements to the research harness

Maintain a method archive separate from the model archive. A proposed MethodPatch names the parent policy, observed failure, changed fields/code, discovery cost, regression checks and promotion criterion.

MVP example: GPT-6 observes that incorrect entity/timestamp joins repeatedly waste model evaluations. It proposes a conditional semantic check before program mutation. Run frozen parent and child research policies on separate development/validation tasks with identical available tools and maximum budgets. Retain the change only if the declared promotion rule is met; otherwise retain the parent. Final audit tasks are used once after selection.

Allowed first edits are investigation order, a diagnostic trigger, or the proposal template within an explicit schema. Later edits can alter an allowlisted proposal/helper module. The editor cannot change final labels, objective, evaluator, task allocation, execution permissions, budget enforcement or audit records.

Loop D — the actual recursive research question

A later version can revise the method-revision procedure itself, such as which failure traces it samples or which allowed policy mutations it proposes. Test whether the revised procedure creates better subsequent policies at equal total cost on new task families. This requires multiple generations and independent replication. One successful Loop C update supports bounded method improvement; it does not establish general RSI.

Darwin Gödel Machine is a related empirical self-modifying-agent precedent. It helps position method evolution, while this project's scientific outcomes and real-data validity remain separate tests. Darwin Gödel Machine.

Minimal control algorithm

The following is orchestration pseudocode, not an implemented runtime:

problem = freeze_problem_and_benchmark(user_question, domain_pack)
evidence = curate_versioned_evidence(problem)
model_archive = seed_and_evaluate_templates(problem, evidence)

while budget.can_start_another_action():
    state = summarize(problem, evidence, model_archive, diagnostics, budget)
    action = gpt6.propose_action(state, schema=ActionProposal)
    scheduler.validate(action, problem.permissions, budget)
    if action.action_type == "stop":
        break
    if action.action_type == "mutate_program":
        parent = model_archive.get(action.parent_id)
        patch = gpt6.propose_program_patch(parent, state, allowed_grammar)
        child = compiler.apply_and_validate(parent, patch)
        result = worker.execute(child, permitted_inputs, limits)
        score = evaluator.score_development_outputs(result)
        model_archive.record(parent, child, patch, result, score)
    else:
        result = tools.execute_validated(action)
        evidence, diagnostics = record_and_update(result)
    budget.record_actual_usage(result)

policy_patch = gpt6.propose_method_patch(development_traces)
policy_candidate = apply_allowed_method_patch(frozen_parent, policy_patch)
policy_report = compare_on_separate_validation_tasks(
    frozen_parent, policy_candidate, equal_resource_caps
)
selected_policy = fixed_promotion_rule(policy_report)
final_report = audit_once(selected_policy, untouched_tasks)
return compose_decision_artifact(real_runs, final_report, evidence)

Errors, cancellations and exhausted budgets are explicit recorded branches in the implementation. GPT-6-produced scores or claimed experiments never bypass the evaluator. A missing tool produces unsupported_operation, not a fictitious result.

GPT-6 integration sketch

Use the Responses API, a configured accessible model, and typed tool calls. The application must implement tool dispatch and validation; ChatGPT's own tools are not automatically present in an API application. Keep proposals small to reuse templates and cached evidence. The sketch intentionally omits application tool schemas and error-handling code:

from openai import OpenAI

client = OpenAI()  # Reads server-side OPENAI_API_KEY.
response = client.responses.create(
    model="gpt-6-astra",
    input=build_proposer_context(problem, parent, evidence, diagnostics),
    tools=registered_application_tool_schemas,
)
# Dispatch recognized tool calls in application code; return tool results;
# continue until a valid typed proposal, explicit stop, or failure is recorded.

Set supported reasoning controls only after the account capability check. For GPT-6 tool use, use Responses; avoid inventing sampling parameters. Count model calls/tokens, evidence requests, fitting time, memory, runtime and known monetary costs separately. Illustrative caps for one small investigation are eight reasoning calls, four evaluated programs including seeds, three research actions and one repair per program; instrument real latency before committing to repeat counts.

Global sprint execution cap: an illustrative ceiling is 64 total model requests, 250,000 aggregate input tokens, 50,000 aggregate output tokens and 15 minutes of cumulative numerical-worker time, with a separately configured monetary ceiling based on current account pricing. These are proposed maximums, not estimates or authorization to spend. The global ledger includes capability probes, failed calls, repairs, seed evaluations, policy development and every parent/child evaluation episode. Per-episode caps never reset this ledger. Stop at the first exhausted limit and report incomplete comparisons.

2. Show Astra solving the whole problem

The intended experience is problem DNA → evidence synthesis → system model → executable experiment → judgment → decision and consumption. Astra should make consequential choices at these transitions, with an observable artifact and an independent check at each one.

Stage

Reasoning work to demonstrate

Artifact and check

Problem DNA

Identify the decision-maker, objective, actors, feedback, constraints, missing information and failure cost; decide whether the current question is identifiable.

A structured problem brief; required facts and assumptions are distinguishable.

Evidence synthesis

Resolve incompatible units, timestamps, entities, endpoints and conflicting claims; identify the next useful dataset.

A tested data join, semantic mapping and provenance ledger.

Model design

Choose a representation; generate competing mechanisms; identify which components need data, rules or uncertainty.

Comparable model specs, assumptions, and predictions that could fail.

Engineering

Generate code, reuse scientific components, wire adapters and debug actual exceptions.

Executed candidate and integration logs.

Investigation

Choose the next experiment, ablation or missing-data check using its potential to change the conclusion.

A valid executed action plus its measured outcome.

Judgment

Explain effect sizes, uncertainty, counterexamples, identification limits and trade-offs.

A verdict whose numerical claims resolve to external results.

Decision

Map supported evidence to a feasible action, a conditional recommendation or a request for more evidence.

A decision card with alternatives, assumptions, expected outcomes and reversal conditions.

Consumption

Express the result as an operator dashboard, scientist report, reusable model, API result or experiment proposal.

The user can inspect evidence, change assumptions and export the artifact.

Learning

Propose one change to the investigation procedure and assess its transfer.

Parent/child comparison on fresh cases, including all costs.

The model's value is not assumed. Compare (A) a competent fixed analytical workflow, (B) the same Astra model in a fixed workflow, and (C) Astra with adaptive synthesis and investigation. For a later model-capability study, compare a simpler model in the same harness under both matched-cost and matched-call conditions. This separates gains due to model capability, tools, orchestration and extra compute. Do not promise that any arm wins.

Model: GPT-6 Astra. Official documentation specifies gpt-6-astra through the Responses API; account access must be verified at implementation time. Use this model for hypothesis/program proposals, investigation decisions and method-revision proposals. Credentials remain server-side. OpenAI model guidance.

Empirical admission rule for use cases: require an identifiable data source, observable target, honest system boundary, runnable baseline, protected comparison and a useful consumption artifact. Source pages below were checked; the datasets have not been downloaded or validated in this task. Access, license, schema and completeness checks are implementation prerequisites.

3. What your references change

Columbia's NSCORE approach connects dataset sourcing, integration and curation, search for useful datasets and attributes, analysis, and provenance/access governance. It explicitly includes semantic reconciliation and on-demand curation. We adopt that evidence infrastructure, then propose an additional complex-systems modeling and experiment-selection loop. NSCORE's page does not itself establish an RSI architecture or validate this proposal. NSCORE approach.

Your supplied Pasted markdown(1).md contributes the modeling perspective: heterogeneous components, interactions, feedback, nonlinear dynamics, adaptation, path dependence, hybrid representations, and evaluation at micro, meso and macro scales. Its assertions of novelty and illustrative numerical confidence values are not treated as research evidence.

The ODD protocol supplies an existing foundation for documenting simulation purpose, entities, state variables, scales, scheduling, initialization, inputs and submodels. Adapt it into the engine's model specification so generated simulations remain reviewable and reproducible. Grimm et al., ODD protocol update.

The unit of research is an executable system hypothesis: a proposed set of components, interaction rules and observations that could generate the phenomenon being investigated. An output may be a supported mechanism, a rejected explanation, a robust intervention, a regime boundary, or a justified request for a missing measurement.

4. What a system model contains

Represent a candidate model as:

[
M=(B,X,G,F,\Pi,\Theta,O,C).
]

Part

Meaning

Why it matters

B

Boundary, time scale, spatial scale and external drivers

Determines which feedbacks are modeled and which are held external.

X

Entities, state variables, stocks and resources

Makes populations and physical quantities explicit.

G

Interaction graph or spatial topology

Specifies who influences whom and where flows occur.

F

Transition mechanisms and update schedule

Encodes dynamics, delays, thresholds and noise.

Pi

Heterogeneous agent policies or adaptation rules

Represents different information, incentives and behavior.

Theta

Parameters and their uncertainty

Separates measured values, inferred values and assumptions.

O

Observation and measurement model

Separates the hidden process from what datasets actually measure.

C

Constraints and invariants

Enforces admissible actions, conservation, units and institutional rules.

A general transition can be written as x[t+1] = F_M(x[t], G[t], a[t], external[t], noise[t]), with a_i[t] ~ pi_i(local_observations, memory, resources) and y[t] ~ O_M(x[t]). Graphs and policies may themselves evolve. Continuous-time, discrete-event and agent-based versions are alternative templates.

Do not hard-code the macro phenomenon into every agent. If the question concerns herding, congestion or cooperation, encode candidate local mechanisms and measure whether those patterns emerge. Reproducing a pattern shows that a mechanism can generate it; other mechanisms may generate the same pattern.

A configurable probabilistic–neural–symbolic composition

Component

Appropriate job

Example

Probabilistic

Measurement noise, uncertain parameters, stochastic arrivals, heterogeneous populations

Distribution over signal reliability or rainfall.

Neural

Extract semantics, estimate a difficult submodel, learn a surrogate or propose code

Contract parsing; learned travel-time residual; optional expression encoder.

Symbolic

Explicit rules, graph relations, equations, constraints and intervention definitions

Settlement predicates; water balance; road capacities; reaction stoichiometry.

Simulation

Evolve coupled states and local actions through time

Information cascade, queue buildup, aquifer depletion, biological recovery.

Inference

Compare explanations and estimate parameters under stated assumptions

Fit shared versus heterogeneous response rules.

Each entity can use a different representation. A pump can follow physical equations, a farmer a bounded decision rule, weather a stochastic process, and a document an LLM extraction pipeline. The model does not need an LLM call for every simulated entity and time step. Reuse numerical rules for fast rollouts and place expensive reasoning at model-design and investigation decisions.

Hard constraints should be enforced in software. Extracted scientific claims remain uncertain proposals with provenance; assigning an arbitrary confidence number does not calibrate them. Neural components are optional when a simpler component is sufficient.

5. Model reuse, training and fusion: GPT-6 plus Hugging Face and scientific components

The engine should explicitly support define → discover → select → use → adapt/train → compose → evaluate → serve → monitor. It chooses whether training is justified by the problem and available data. A frozen pretrained model or a simple fitted statistical model may already be sufficient.

What model fusion can mean

Mode

Concrete assembly

What must be checked

Tool/model orchestration

GPT-6 invokes an accessible specialist forecaster, encoder, biological model, simulator or solver.

Input/output meaning, units, timing, runtime and task-level benefit.

Sequential pipeline

A text model extracts entities; a graph model predicts a state; a solver produces a feasible action.

Propagation of extraction errors and uncertainty; end-to-end evaluation.

Parallel ensemble

Several compatible predictors feed a calibrated combination or routing policy.

Same target/horizon/scale; ensemble weights fit on development data; correlated errors.

Mechanistic plus learned residual

A storage/flow/dynamical model is corrected by a learned component.

Identifiability, physical constraints, stability and performance under shift.

Neuro-symbolic composition

A neural component proposes facts or scores; explicit relations/rules constrain inference or actions.

Uncertain proposals remain distinct from established facts; actual constraint enforcement.

Fine-tuning or adapters

Adapt an accessible open model using task data and a supported training method.

Data availability, model compatibility, compute, leakage, saved checkpoints and held-out improvement.

Distillation

Train a smaller accessible model on validated labels, teacher outputs or successful tool traces.

Source/model terms, trace quality, task fidelity and fresh held-out evaluation.

Weight/adapter merging

Combine compatible accessible checkpoints or supported adapters.

Architecture/base-model/tokenizer compatibility, merge support and independent evaluation.

Hugging Face model cards describe intended use, limitations, training data and evaluation metadata; use them to screen candidate components. Transformers pipelines provide interfaces for supported inference tasks, while PEFT supports selected parameter-efficient adaptation and adapter-merging workflows. These are building blocks, not evidence that arbitrary models are compatible. Model cards, Pipelines, PEFT, adapter merging.

GPT-6 remains an API component in this design. API inference access does not provide parameter files to merge into an open checkpoint. Compose GPT-6 with Hugging Face models through predictions, structured messages, tools, routing or a separately permitted distillation workflow. Do not imply GPT-6 weight merging or GPT-6 fine-tuning is available from ordinary API access.

Component registry and training contract

Each ModelAsset records source/provider, exact model/repository ID and revision, model-card reference, task, input/output schema, preprocessing/tokenizer, target units and horizon, dependencies, compute requirements, permissions/license, training-data provenance where known, supported adaptation methods, local evaluation and limitations. A model with unknown training overlap is not presumed uncontaminated.

The model gateway dispatches to GPT-6 API, a supported hosted/local Hugging Face runtime, a scientific Python model or a simulator. Download or access checks occur before selection is finalized. Model names, metadata and a successful load do not prove scientific suitability.

A TrainingPlan defines objective, training/selection splits, initialization, trainable parameters, optimizer and stopping rule, maximum runtime/memory, checkpoints, validation metrics and rollback. Test observations never train model weights, ensemble weights, calibration, feature selection or a neural residual. If the requested compute or labels are unavailable, produce a plan and dependency gap rather than claiming a trained model.

For the four-hour build, demonstrate one small fitted component such as a calibrator, linear/count model or tiny residual alongside GPT-6. Add an already cached specialist model only if it improves the executable path. Large-model training, arbitrary Hugging Face downloads and LoRA jobs are later extensions unless the exact environment and data are already prepared. The engine exposes their contracts without presenting them as completed capabilities.

Evolve complete solutions

Expand the permitted genome from model mechanisms to the composition graph: replace a compatible component; add/remove a residual; choose an ensemble; change a gate or supported training recipe; select a different solver; modify an adapter; or change a decision rule within the fixed problem constraints. GPT-6 proposes the edit and its expected practical effect. The runner then trains if needed, executes the whole assembly and scores end-to-end results.

A cached component can make execution faster, but its saved work must be accounted for consistently across comparisons. Reusing weights is different from reusing test answers. A pipeline with a better intermediate score can still fail the actual decision objective because of latency, missing coverage, integration errors or constraint violations.

Concrete assemblies across the five domains

Domain

Example composed solution

Useful consumption

Prediction Markets

GPT-6 contract reasoning + temporal probability model + symbolic event relations + fitted calibrator.

Forecast reliability panel and a decision to use an estimate or obtain more evidence.

Urban Infrastructure

GPT-6 problem/model design + flow or graph predictor + capacity rules + scenario optimizer.

Station/sensor watchlist and conditional operational options.

Climate and Agriculture

GPT-6 evidence reconciliation + hydrological/crop component + uncertainty ensemble + optional learned correction.

Basin recovery forecast or regional crop-risk report with scope conditions.

Social Simulation

GPT-6 mechanism synthesis + temporal-network model + explicit behavioral alternatives + workload constraints.

Reputation/moderation workload diagnostic and study proposal.

Biology

GPT-6 experiment reasoning + perturbation/transport model + metadata constraints + uncertainty and acquisition policy.

Ranked experiment or sampling plan grounded in recorded outcomes.

These examples are proposed compositions. Each specialist model must be selected against actual task data; a generic popular checkpoint is not automatically an appropriate forecasting or biological model.

6. What hypotheses the engine generates

Hypothesis type

Example

Distinguishing test

Structural

A concentrated communication network amplifies false market consensus.

Change topology while controlling signal quality.

Mechanistic

Traders copy price movement instead of processing independent evidence.

Break the price-to-belief feedback in the simulator.

Heterogeneity

A small group with high influence drives the outcome.

Redistribute influence while holding its total fixed.

Temporal

Delayed measurements cause a stabilizing policy to oscillate.

Vary delay separately from gain and noise.

Observation

Apparent disagreement comes from incompatible measurement definitions.

Harmonize units, timing and observation models before refitting.

Boundary

A locally beneficial intervention transfers costs to a coupled subsystem.

Expand the model to include the affected stock or population.

Intervention

A rule remains beneficial across several plausible mechanisms.

Evaluate it across a model ensemble and new conditions.

Meta-hypothesis

Checking measurement alignment before fitting complex dynamics saves investigations.

Compare frozen research policies on fresh tasks at equal resource caps.

Generate two or three plausible rivals first. Each record includes supporting and opposing evidence, assumptions, observable predictions, a potential disconfirming result, and an executable test. Hypothesis search may change mechanisms, topology or observation rules; parameter tuning alone explores only one part of the model space.

7. Prediction Markets — real event contracts, observations and resolutions

P1. Do logically linked markets produce more reliable forecasts when modeled together?

Real evidence: contract definitions, event groups, timestamped bid/ask or transaction summaries and final resolution labels from Kalshi. The official documentation provides public market endpoints and an archived-market candlestick route; select a complete small archive before the sprint. Market-data guide, historical candlesticks.

System/problem DNA: related event contracts form a probability-constraint graph; price observations evolve over time under shared information and market activity. Settlement definitions, time zones and measurement locations are part of the model. The useful output is a reliability/consistency audit for a forecast desk.

Rival hypotheses: shared event structure improves estimates; apparent violations are explained by different definitions, stale observations or spreads. Public prices alone do not identify individual trader beliefs or herding.

Model/code/integration: Astra extracts proposed event predicates and matches them to exact source spans. Deterministic checks validate comparisons. Generate a temporal graph estimator or constrained probability projection and connect it to point-in-time observations. For matching events, P(Tmax > 90 F) <= P(Tmax > 85 F); sum-to-one requires a complete mutually exclusive partition.

Experiment and baseline: compare raw price proxies, a fixed manually validated constraint baseline and the proposed method. Reserve complete event/date groups chronologically; fit transformations only on earlier groups. Include deliberate near-matches that should not be joined.

Verification/judgment: compute Brier/log loss against actual resolutions, false relation flags, calibration diagnostics and coverage. Coherence is a separate score and cannot stand in for accuracy. Quotes are price proxies, not guaranteed calibrated probabilities. An apparent midpoint inconsistency is not demonstrated executable arbitrage.

Meta-test and consumption: test whether checking definitions before extra data collection reduces false flags per call. Export a graph of comparable events, forecast differences, confidence limits where justified and unresolved comparisons. Historical LLM outcome memorization remains a limitation; prospective saved forecasts give stronger evidence of forecasting skill. No trades are required.

P2. When do low activity and stale observations make forecasts unreliable?

Real evidence: reuse resolved event families and their timestamped observations, volumes/activity indicators and bid/ask information where provided. Verify field semantics and coverage for the chosen archive; do not infer historical order-book depth from candlestick volume.

System and rivals: forecasts, market activity and information arrival co-evolve across related events. H1—reliability varies with quote age and activity; H2—event difficulty, category and time-to-resolution explain the same association. The application is to flag forecasts requiring additional evidence.

Models/code: hierarchical calibration or a regime-dependent stochastic time-series model with event-family effects; optional nonlinear residual; explicit temporal restrictions. Generate snapshot construction, missingness checks, stratified evaluation and a report.

Experiment: compare raw price, pooled calibration and an activity-aware model on later complete events. Prespecify observation horizons so near-resolution observations do not dominate the result; use event-level resampling and category-held-out sensitivity checks.

Judgment: held-out loss, reliability by activity regime, uncertainty and useful coverage. Predictive association does not prove that increasing liquidity would improve accuracy. An agent-based explanation may be explored but is not identified by these observations alone.

Meta/decision: evaluate whether metadata-quality auditing before model expansion improves held-out forecast reliability under a fixed investigation budget. Deliver a data-quality/reliability panel and a conditional request for additional information, with the reason and potential effect on the decision.

8. Urban Infrastructure — measured flows and traffic networks

U1. Can network-aware demand estimates identify bike-station imbalances earlier?

Real evidence: Citi Bike publishes trip histories with start/end stations and timestamps, plus a live GBFS feed. The data page also describes filtering and removed service trips. Citi Bike system data.

System/problem DNA: bikes and stations are stocks, rides are flows, commuters have recurring demand, and unavailable stations can redirect riders. The observed data measure completed rides; unserved demand and historical rebalancing are not fully observed. The user is an operations planner seeking where and when to inspect or rebalance.

Rivals: each station's local calendar pattern is enough; connected stations' lagged flows improve forecasts; apparent improvements result from future information or changing station IDs.

Models and wiring: compare seasonal per-station counts, a count/state-space model and a network-coupled variant. Astra generates schema reconciliation, station/time aggregation, graph construction, model fitting and prediction code. Add stock/capacity constraints only where corresponding inventory observations exist.

Experiment: fit earlier weeks, select on later weeks and confirm on untouched days, retaining whole-day dependence. Evaluate held-out incoming/outgoing completed trips and net-flow imbalance. Include station-level error and tail imbalances, not only the network mean.

Judgment and use: actual flow prediction can be verified from trip data. Stockout prediction requires matching inventory history or prospective collection. Alternative rebalancing outcomes are model-based until validated prospectively; completed-trip data alone cannot prove saved trips. Produce a station watchlist and a conditional rebalancing simulation with these boundaries visible.

Meta-test: compare investigating temporal alignment and flow imbalance first versus adding a more complex model first, using equal research caps on fresh station/time subsets.

U2. Do congestion patterns propagate through the road network or reflect shared daily demand?

Real evidence: METR-LA/PEMS-BAY provide real traffic time series; the DCRNN authors provide data links, sensor graph information, baseline code and reported forecasting results. Authors' repository, DCRNN paper.

System/problem DNA: sensors observe a coupled road network; local slowdowns and upstream/downstream interactions evolve alongside common commute patterns. The application is an operational congestion forecast with a measured lead time.

Rivals: per-sensor seasonality is sufficient; network-lag structure adds predictive information; a shared latent driver explains the apparent propagation.

Models/code: persistence, seasonal/independent autoregression, graph-lag regression or state-space dynamics; optional neural graph model after the sprint. Astra writes adapters and competing models, reuses sensor graph metadata and defines consistent missing-value handling.

Experiment: use the published split/protocol for any comparison with published scores; otherwise clearly report a new small-slice benchmark. Compare a real graph, a degree-controlled shuffled graph and an independent model. Hold out contiguous time blocks and report separate forecast horizons. A published DCRNN score is a reference, not a result the prototype has reproduced.

Judgment/decision: MAE/RMSE, congestion-alert precision/recall under a prespecified definition, and errors by sensor/regime. A graph's predictive value does not alone establish causal traffic propagation. Export a forecast/alert artifact; road-control interventions need separate simulation and field validation.

Meta-test: does identifying graph-versus-common-driver ambiguity before architecture search improve useful forecasts per run? Record tool and training costs along with predictive quality.

9. Climate and Agriculture — measured hydrology and crop outcomes

C1. Why do some catchments recover slowly after drought?

Real evidence: CAMELS-US includes daily meteorological forcing, streamflow and catchment attributes for 671 catchments minimally affected by human activity. The documented record covers 1980–2014; verify gauge-specific completeness. Dataset/downloads, primary dataset paper.

System/problem DNA: precipitation, snow, soil/storage and drainage interact through stocks, flows and delays. The practical output is a low-flow/recovery forecast for water-resource analysis.

Rivals: rainfall deficits and seasonality explain persistence; nonlinear storage produces hydrological memory; snow processes or forcing/observation errors explain remaining differences.

Models/code: seasonal/lagged-flow baseline, linear reservoir, nonlinear storage alternative and optional neural residual with uncertainty. Astra generates unit/time alignment, inference wrappers and candidate storage/observation mechanisms. Preserve nonnegative storage and mass balance where the model represents them.

Experiment: proposed split 1981–2004 fit, 2005–2009 development and 2010–2014 test, subject to actual coverage. Start with a small packaged basin subset. Freeze drought/recovery definitions before evaluation and use only information available at each forecast time. With realized future weather forcing, report conditional hydrological simulation rather than an operational weather-unknown forecast.

Judgment/meta-test: discharge error, low-flow bias, recovery-time error, interval coverage and basin-wise robustness. Test residual-directed choice of storage/snow hypotheses against a fixed model-search sequence on new basin tasks. These data do not establish farmer behavior, pumping or allocation-policy effects. Consumption: a basin recovery report, conditional forecasts and targeted additional-measurement suggestions.

C2. When do heat and moisture stress combine into regional crop losses?

Real evidence: USDA NASS Quick Stats county corn-grain yield estimates joined to NASA POWER daily weather. Query CORN, GRAIN - YIELD, MEASURED IN BU / ACRE; begin with one state and a proposed 2000–2023 window, verifying coverage and suppression first. USDA Quick Stats, NASA POWER.

System/problem DNA: crop response interacts with heat, moisture, seasonal timing and spatially shared weather; concurrent stress can create correlated county losses. Management and cultivar changes are partly unobserved. The application is regional crop-risk assessment.

Rivals: additive weather response is sufficient; heat–moisture interactions add nonlinear loss; trends and unobserved management explain apparent effects. Regional co-failure may reflect shared drivers rather than county-to-county propagation.

Models/code: county trend, additive weather regression, interaction/spline model and an optional mechanistic stress index. Generate joins, unit normalization, suppression handling and training-only detrending. If actual soil moisture is unavailable, label precipitation-based quantities as moisture proxies. AquaCrop can later add a model layer; it does not supply the observed yield labels.

Experiment: proposed fit 2000–2014, develop 2015–2018 and test 2019–2023; evaluate geographically withheld groups separately. A model using the full growing season's realized weather is an end-of-season hindcast. Earlier operational warnings require truncated weather and appropriate forecasts.

Judgment/meta-test: yield-anomaly error, calibrated severe-loss probabilities and regional co-occurrence, with uncertainty preserving shared-year dependence. Test whether investigation targeted at compound-stress years improves unseen-year reliability versus average-error-only search. County estimates and gridded weather do not identify field-level causal effects. Consumption: a regional risk report, failure conditions and a list of missing management observations.

10. Social Simulation — observable temporal interaction networks

S1. Which network mechanisms predict newly expressed distrust?

Real evidence: Stanford SNAP provides Bitcoin OTC and Alpha timestamped, directed user ratings from −10 to +10. Alpha can serve as a second-platform transfer comparison. Ratings measure expressed trust, not verified fraud or private beliefs. Bitcoin OTC, Bitcoin Alpha.

System/problem DNA: users rate one another, altering a reputation graph that may affect later interactions. Reciprocity, signed triangles, concentration and recency are candidate mechanisms. The application is reputation-system reliability assessment and prioritization of human review.

Rivals: stable rater harshness and recipient reputation explain outcomes; reciprocal/local network structure adds information; recent distrust propagates after controlling for activity and history. Influence is a hypothesis, not an observed causal fact.

Models/code: start with shrunk recipient reputation and regularized logistic/boosted models using historical activity, negativity, degree and recency. Generate a competing signed temporal-graph model, leakage-safe feature builder and evaluator; a stochastic agent model is an optional hypothesis-testing extension.

Experiment: predict whether the next recorded rating is negative, conditional on its rater–recipient pair being known. Split chronological blocks, group equal-time records and compute graph features strictly from the past. Test platform transfer without assuming shared identities. This target does not predict whether a transaction or rating occurs.

Judgment/meta-test: compare held-out log loss, Brier score, negative-class PR-AUC, calibration and cold starts. Test whether analyzing disagreement between reputation-only and network models improves the investigation policy under equal budgets. Consumption: export a reliability report and review queue with reasons; intervention effects on a real reputation system remain unvalidated.

S2. Can feedback between communities improve forecasts of moderation workload?

Real evidence: SNAP's Reddit Hyperlink Network contains timestamped community-to-community links. Its label distinguishes explicitly negative (−1) from neutral or positive (+1); labels combine crowdsourcing and a text classifier. They are imperfect observable proxies. Dataset and label definitions.

System/problem DNA: links between communities may trigger reciprocal linking and bursts of attention. Community activity, stable relationships and common external shocks interact. The measurable target is incoming negative-labeled links over the next seven days for communities selected using training-period activity.

Rivals: persistent community/pair tendencies and activity explain counts; reciprocal excitation adds predictive information; common shocks explain apparent contagion.

Models/code: recency-weighted counts and regularized negative-binomial baselines; a competing temporal-network/self-exciting model with an explicit common-driver alternative. Astra writes deduplication, time-safe features, simulation/inference adapters and forecasts.

Experiment: rolling temporal evaluation with target windows wholly contained in each split. Deduplicate overlapping title/body records by source, target and post ID. Exclude all-period embeddings unless rebuilt only from training data. Compare under the same timeline and target definition.

Judgment/meta-test: predictive count score/deviance, absolute error and interval coverage, stratified by community activity. Test whether examining stable hostility and common drivers before adding feedback improves predictive quality and avoids unsupported causal claims. Consumption: a workload forecast and review-priority artifact; negative links do not directly measure harassment, polarization or psychological state. Policy interventions require additional causal evidence.

11. Biology — measured perturbations and cell-population dynamics

B1. Which combined gene perturbations produce responses that additive effects cannot explain?

Real evidence: Norman et al. 2019, GEO accession GSE133344, with original author code and a standardized scPerturb file NormanWeissman2019_filtered.h5ad. The full file is roughly 699 MB; prepackage a scientifically justified subset for the sprint. Author repository, standardized dataset record.

System/problem DNA: interacting genetic programs produce cell-expression responses. Perturbation pairs can amplify or buffer responses, but observed interaction depends on the measurement scale. The user is a researcher prioritizing informative combinations.

Rivals: additive single-perturbation effects explain combinations; low-rank/nonlinear interactions add information; batch, selection or expression normalization explains the apparent gain.

Models/code: explicitly define response relative to control and its scale; compare additive effects, an interaction model and a neural/graph comparator when available. Astra writes perturbation/control alignment, response aggregation, training code and comparison reports. The official GEARS repository provides a Norman loader and comparator; its documentation notes that reliable combination prediction needs some combination data during training. GEARS.

Experiment: hold out entire perturbation combinations. Separate new pairs of previously seen genes from previously unseen genes; random cell splitting cannot establish either transfer claim. Keep preprocessing and selected genes within training boundaries.

Judgment/meta-test: response-relative error, non-additive residuals and expression-program consistency, avoiding scores dominated by unchanged genes. Cells are not automatically independent biological replicates. Test model-disagreement acquisition against random reveal of a hidden combination pool. This is retrospective replay of real experimental results, not a new wet-lab experiment. Consumption: ranked combinations, evidence strength, uncertainty and a replication proposal; transcriptomic results do not establish clinical effects or feedback topology.

B2. Can a model predict changing cell populations while accounting for both state transitions and growth?

Real evidence: Schiebinger et al. 2019 reprogramming time course; the authors' Waddington-OT tutorial supplies expression/time metadata and held-out-timepoint analysis for 39 time points across 18 days. Author tutorial, Waddington-OT.

System/problem DNA: cell-state distributions change through transitions, proliferation and survival. Different processes can generate similar population snapshots. The application is assessing dynamic models and selecting an informative sampling time.

Rivals: persistence or simple distribution interpolation is sufficient; transport of cell states explains the change; differential growth materially changes the predicted population distribution.

Models/code: previous-distribution and simple interpolation baselines, transport without differential growth, and growth-aware transport. Astra generates a training-only representation, timepoint splits, inference adapters and distribution comparisons. Use the author workflow as a reproducible comparator when setup is available.

Experiment: withhold complete intermediate timepoints and predict their expression distributions and state proportions. Using observations on both sides is interpolation, not forecasting. A separate forward forecast must use earlier observations only. Check sensitivity to growth assumptions and representation choices.

Judgment/meta-test: distribution distance, state-proportion error, performance across withheld times and uncertainty from supported replication. Destructive single-cell sampling does not observe the same individual cells over time; inferred paths are not ground-truth lineages. Test whether revealing model-disagreement timepoints improves model-failure detection versus uniform sampling. Consumption: a dynamic comparison report and sampling proposal, with remaining non-identifiability explicit.

Deferred examples: organismal heat-stress resilience, shared-aquifer governance and pest–predator interventions remain useful research directions, but need suitable outcome evidence and validated couplings before becoming empirical benchmark packs. Their earlier simulated forms are not substitutes for the ten measured-data cases above.

12. Full lifecycle and the artifacts it produces

Stage

Engine behavior

Reviewable result

Frame

Define phenomenon, beneficiary, boundary, scales, practical outcomes and budget.

ProblemSpec

Source

Locate datasets, papers, existing simulators and code with access/provenance.

EvidenceLedger

Reconcile

Align entities, units, time, resolution, endpoints and observation conditions only as needed.

DataContract plus transformation lineage

Synthesize

Connect claims and mechanisms; identify contradictions and missing measurements.

MechanismGraph with evidence links

Hypothesize

Generate alternative structures, behaviors, observations and intervention claims.

HypothesisSet with disconfirmation conditions

Model

Instantiate reusable components and specify all coupling and update rules.

SystemSpec plus model versions

Implement

Write code, integrate adapters, wire experiments, check interfaces and repair bounded failures.

Executable candidate, code diff and run environment

Infer

Calibrate parameters and assess what available observations can distinguish.

Fit results, uncertainty and identifiability notes

Experiment

Run baselines, interventions, ablations, sensitivity tests and held-out cases.

RunRecords, trajectories, metrics and failed attempts

Judge

Assess correctness, empirical fit, mechanism support, robustness and practical trade-offs.

Verdict: supported in scope, unsupported, or inconclusive

Meta-test

Compare alternative investigation policies under matched budgets.

Parent/child policy evaluation

Improve

Retain or reject one versioned change based on separate evaluation.

Policy archive and rollback target

Apply

Export the model, finding, intervention or next measurement; define external validation.

Reproducible report and downstream interface

Monitor

Track new evidence, prediction failures and scope changes after use.

Drift log and justified revision trigger

Evidence synthesis includes data and model semantics as well as literature. Preserve transformation code so the engine can trace a conclusion back through a data join, unit conversion or measurement definition.

A new-use-case contract

Users can bring a new question and evidence. Astra first creates a draft pack, checks missing information and chooses reusable components. A pack becomes executable only after its adapters and baseline work, and it becomes empirically evaluated only after its protected comparison runs. A text prompt alone does not supply a validated simulator or measurement process.

Pack element

Required content

problem_dna

Beneficiary, decision, goals, available actions, constraints, scope, time scales, observable outcomes and failure costs.

evidence_contract

Source references, access/license conditions, entity keys, units, timestamps, observation model, missingness and provenance.

system_spec

Entities, graphs, dynamics, heterogeneous policies, uncertain parameters and hard constraints.

hypotheses

Rival mechanisms, predicted differences, assumptions and disconfirmation conditions.

adapters

Data loading, harmonization, fitting/simulation, interventions and artifact export.

benchmark

Strong baseline, protected splits, metrics, uncertainty unit, practical threshold and comparison budget.

research_actions

Executable data checks, model variations, experiments and stop rule.

decision_contract

Action alternatives, outcome/cost model, feasibility, conditions requiring more evidence, and consumption format.

The reusable design is a specification → executable adapter → independent evaluator → decision artifact pattern. Probabilistic state-space, temporal graph, agent-based, constrained optimization and dynamical-system components form a small model registry. A new domain can combine them without rebuilding the runner, provenance, budget controls, evaluation or reporting.

Maintain separate uncertainties for observed data, unknown parameters, competing model structures and outcome preferences. Robust decisions should be compared across plausible models. If empirical policy outcomes are unavailable, provide scenario-conditioned results and the validation needed before action; do not label simulated expected utility as observed impact.

The decision card includes: recommended action or next measurement; supporting evidence; best alternative; predicted outcome range; resources required; model assumptions; conditions that reverse the recommendation; and a follow-up measurement. The consumption layer selects a dashboard, report, reusable model/API output or experiment proposal based on the user's workflow.

13. Engineering and reasoning harness

Use one orchestrator with logical roles: evidence curator, system modeler, code/experiment builder, critic and metareasoner. These can be different prompts using the same account-accessible model. Durable typed state, actual tool execution and protected evaluation establish the workflow.

The OpenAI Responses API can connect reasoning to application-defined tools; the application validates and executes calls and returns results. Structured Outputs can enforce output shape but do not establish scientific correctness. Keep the model ID configurable and verify account access during setup. Function calling, Structured Outputs.

Interface

Responsibility

load_evidence(pack)

Supply authorized observations and source references.

reconcile_data(spec)

Apply declared semantic/unit/time transformations and record lineage.

build_model(system_spec)

Generate or configure allowed model components.

validate_model(candidate)

Check interfaces, units, finite outputs and domain invariants.

run_experiment(candidate, experiment_spec)

Execute with fixed inputs, seeds and resource limits.

read_results(run_id)

Return actual logs, trajectories, metrics and failure state.

propose_policy_patch(parent, patch)

Submit one allowed, versioned research-policy change.

stop(reason)

Terminate because evidence, budget or capability limits justify stopping.

Define initialize(config, seed), step(state, intervention, rng) and observe(state) for simulator components. Define fit(data, config) and predict(model, inputs) where inference is needed. The fixed evaluator computes metrics from outputs; candidate-written scores are not authoritative.

Keep generated code in an existing isolated execution worker with no credentials or network, controlled inputs, bounded compute and a writable output directory. Keep hidden generator mechanisms, confirmation observations and evaluator logic inaccessible to the candidate. Read-only access still reveals test answers. If isolation is unavailable, execute trusted templates with validated parameters and show arbitrary generated code as unexecuted.

Log model/policy versions, source and data hashes, environment, random seeds, interventions, tool outputs, token usage, runtime, exceptions and concise decision justifications. Replay of saved numerical code is different from reproducing a stochastic LLM generation. Allow one repair attempt per candidate in the sprint and retain both attempts.

A proposed pack configuration:

domain: prediction_markets
problem: linked_event_forecast_consistency
status: specification_only
problem_dna:
  user: forecast_analyst
  decision: use_forecast_or_request_more_evidence
  outcome: resolved_binary_event
system:
  boundary: observed_related_contracts_over_time
  entities: [contract, event, price_observation, source]
  graph: validated_logical_and_event_relations
  state: [price_proxy, observation_age, activity, resolution_rules]
  observation: timestamped_market_snapshots
  latent_hypotheses: [shared_information, temporal_coupling]
  constraints: [probability_bounds, information_cutoff, valid_event_relations]
models:
  baseline: raw_price_and_manually_validated_constraints
  candidate: generated_relation_validator_and_joint_estimator
  neural_component: evidence_extraction_model_design_code_and_judgment
experiments:
  permitted: [inspect_definitions, compare_relation_ablations, test_temporal_regimes]
  primary_metric: brier_score_on_unseen_event_groups
  secondary: [false_relation_rate, calibration, coverage, cost]
budget:
  max_evaluated_programs_including_seeds: 4
  max_research_actions: 3
  max_repairs_per_candidate: 1
meta:
  max_proposed_policy_revisions: 1
  mutable: [investigation_order, ambiguity_trigger]
  protected: [evaluator, hidden_cases, budget_limits, execution_permissions]
consumption:
  artifact: forecast_audit_and_next_evidence_decision

These are design fields, not an already implemented API. Every proposed experiment must exist in the selected environment before the engine can execute it.

14. Evaluations that make claims reviewable

Level

Concrete checks

Software and integration

Valid schema; reproducible initialization; bounded execution; real tool outputs; clean failure states.

Model mechanics

Unit consistency, conservation where appropriate, valid probabilities, sensitivity to time step/update order.

Micro behavior

Agent action distributions, individual response curves, local rule adherence.

Meso structure

Clustering, bottlenecks, source overlap, communities, flow concentration.

Macro behavior

Cascades, persistence, recovery, tail losses, aggregate performance and distributions.

Empirical adequacy

Multiple prespecified patterns and held-out trajectories, not one attractive plot.

Mechanism discrimination

Rival mechanisms, observational equivalence, interventions and ablations.

Robustness

Alternative boundaries, topologies, rules, uncertainty, initial states and external shocks.

Research policy

Conclusion quality at matched budgets; actual cost; failure and abstention rates.

Practical usefulness

Reproducible artifact, time to a valid experiment and decision-relevant trade-offs; measure these rather than assuming them.

Baseline protocol: compare a competent fixed informative experiment sweep, an adaptive investigation policy and, if feasible, a frozen revised policy. Keep available evidence, tools, model families, base model settings and maximum resources identical. Record all model calls, repairs, failed simulations and the cost of developing the new policy.

Two levels of separation: reserve fitting, model-selection and confirmation observations within each scientific task; also reserve development, validation and final audit tasks for research-policy evaluation. Split complete independent episodes, events, studies or region-years as appropriate, not correlated rows or time steps.

Use paired task scenarios and common exogenous random streams across policies. Different actions may create different endogenous trajectories. Propose one revision using development traces, validate it once, freeze the selected policy, then run the final audit once. Audit outcomes cannot guide another patch in this sprint.

A subsequent pilot can use four development, four validation and six audit tasks, drawn from distinct event groups or comparable task units. This is not a promised four-hour workload. In the sprint, use only the matched episode pairs that fit the measured remaining global budget and the evaluation slot, often a very small number; report the actual count and mark the result exploratory. Such a pilot demonstrates integration and gives descriptive evidence; it cannot establish broad generalization or strong significance. Report per-task paired effects and uncertainty only at justified independent units. New seeds test less transfer than new mechanisms or domains.

An external evaluator determines whether a numerical run is valid and computes metrics. The LLM interprets results with scope and caveats, referencing result IDs. Engineering success does not establish a mechanism; synthetic mechanism recovery does not establish real-system truth. Support, contradiction and inconclusive results are all valid outcomes.

For a threshold or tipping-point claim, distinguish a broad transition region from evidence for a formal dynamical bifurcation. Check numerical artifacts, finite-size effects and alternative model structures before using stronger terminology.

15. Every 30 minutes: build, release and test

Prerequisites: one developer with AI coding help, working API access, installed numerical tools, an existing isolated runner, a prepared small real-data slice and a few source notes. Setup consumes the four hours if these are missing. Use a CLI/notebook or simple existing display first.

The default is P1 with actual contract definitions, a small resolved-event archive and matched observations. If that archive is not ready, a prepared METR-LA slice provides a clearer measured comparison. Preserve the same full-lifecycle contract. Synthetic fixtures can test execution and known-mechanism recovery; they are labeled and never substituted silently for the empirical comparison.

Time

Feature and runnable release

Astra capability

Acceptance test / evaluation

00:00–00:30

v0.1: real-data adapter, baseline and five-pack registry. Freeze data slice, source hashes, target, split allocation, metric and resource caps.

Inspect data and map it to a measurable problem.

Source fields and timing are valid; baseline reruns; no target leakage or inaccessible required field.

00:30–01:00

v0.2: problem DNA, system graph and rival hypotheses. Reconcile definitions and generate two testable model specifications.

Problem formulation, synthesis and representation choice.

Each claim has evidence or an assumption label; tests use available observations and tools.

01:00–01:30

v0.3: GPT-6 evolutionary proposer. Select an archive parent, generate one typed structural/observation patch, compile and execute it; allow one repair.

Coding, wiring and debugging from actual failures.

Interface, bounds and finite outputs pass; save code diff and logs; evaluator computes scores.

01:30–02:00

v0.4: evolutionary selection and scientific judgment. Score parent/child using development/selection observations, update the sparse model archive and run a development ablation. Keep scientific confirmation data sealed.

Experimental design, numerical inference and scoped conclusions.

Metrics derive from runs; transformations fit only on training data; inconclusive/negative results are valid.

02:00–02:30

v0.5: GPT-6 metareasoning. Choose an evidence check, a second program generation, regime test or stop action; execute and update the archive.

Metareasoning based on what could change the decision.

A controlled evidence change can alter the action; chosen action executes within budget.

02:30–03:00

v0.6: meta-hypothesis and one policy patch. Change investigation order or trigger using development tasks.

Bounded research-process revision.

Versioned allowed-field patch; no evaluator/data/budget changes; development comparison recorded.

03:00–03:30

v0.7: frozen policy validation/audit. Validate once, select parent/child, freeze and compare on untouched tasks.

Judgment about whether the research process improved.

Freeze each selected scientific program before its confirmation test and freeze the selected policy before final audit; matched caps, actual costs/failures/counts, no retuning. A second pack smoke test is optional.

03:30–04:00

v1.0: decision artifact and reproducible demo. Produce forecast audit, recommended next action, assumptions, alternatives and evidence/code export.

Translation of analysis into useful consumption.

Clean live run; every number resolves to a result; live/replay and executable/specification statuses are accurate.

Allocate roughly 20 minutes per feature, 7 for its gate and 3 for a runnable checkpoint. At 90 minutes, prioritize generated code that runs. At two hours, prioritize an actual measured comparison. At three hours, freeze scope. If time slips, cut autonomous policy revision before sacrificing the evidence-to-decision path.

Fallbacks must preserve claim accuracy: prepared hypotheses are labeled; a deterministic selector is labeled rule-based; a manual policy patch is not autonomous improvement; a trusted template does not establish arbitrary code execution; a recorded run is not live. A second pack's schema/baseline smoke test demonstrates partial portability, not cross-domain scientific transfer.

Defer: broad crawling, five full simulator installations, rich dashboards, large neural training, unrestricted code rewriting, distributed agent services and broad MAP-Elites search. All ten cases can be specified and compared in the product catalog; five full empirically validated integrations are a later milestone.

16. Product surface, persistence and operational behavior

Reader view: the problem, useful conclusion, recommended next action, outcome range where justified, assumptions, alternative and reversal conditions. Technical view: Problem DNA, Evidence, Models, Runs, Evaluation, Method History and Deliverables. Show the same underlying artifacts at different detail levels.

The model view displays parent–child lineage, changed mechanisms, archive cells, scores and costs. A claim opens its evidence/program/run/evaluator chain. Changing an assumption creates a new version and marks affected results stale; it never silently updates an already evaluated forecast.

Use one modular Python application, SQLite metadata, immutable files for larger artifacts and one isolated numerical worker. Add a thin UI over this working backend. Keep storage/provider/worker interfaces replaceable; a distributed queue and service fleet are later choices. Protect budget and evaluator enforcement in the host application.

Metadata object

Minimal identity and behavior

Project/problem

Stable ID plus immutable problem versions and decision contract.

Source/evidence

Raw content hash, source locator, observation/publication/availability/ingestion times and revision links.

Program

Parent IDs, template digest, component graph, typed patch and dependency manifest.

Run

Program/data/evaluator/policy versions, status, resources, artifacts and idempotency key.

Evaluation

Protocol hash, split, actual metrics, uncertainty method and claim scope.

Method

Parent, allowed patch, discovery costs, validation/audit linkage and promotion state.

Deliverable

Referenced results, production timestamp, consumption format and stale/current status.

Availability time is not interchangeable with observation time or ingestion time. If first availability is unknown, record that uncertainty. Corrections preserve original and revised snapshots. Track source families and derived datasets so repeated publications do not count as independent evidence.

Use explicit job states: proposed, validated, queued, running, succeeded, failed, cancelled and budget-exhausted. Scientific verdicts are separately supported_within_scope, contradicted_under_test, inconclusive, or invalid_test. Missing evidence and unsupported operations have their own completion codes. Resume using a run/action idempotency key to avoid duplicate logical work.

Cache by evidence digest, program digest, inference configuration, seed and evaluator context. Invalidate descendants after relevant changes. Warm starts are allowed only when dimensions, meaning and training boundaries remain compatible; record them as part of cost comparisons.

17. Acceptance gates and demonstrated capability matrix

Capability

Required MVP evidence

Later stronger evidence

GPT-6 problem formulation

A validated Problem DNA with a real measurable target.

Better downstream validity versus fixed intake.

Synthesis

One corrected semantic/timing mismatch with preserved lineage.

Reduced independent data-integration error across domains.

Program evolution

A GPT-generated parent–child patch compiled, run, scored and archived.

Better external quality–cost than fixed enumeration/random/beam search.

Metareasoning

GPT selects and executes a valid follow-up based on actual diagnostics.

Better quality per total cost on fresh tasks.

Method improvement

If reached: one versioned policy candidate evaluated against its parent; otherwise show not executed.

Repeated transferred improvements with discovery cost included.

Recursive improvement

Clearly represented improvement-procedure mutation surface.

Multiple generations improve subsequent improvement productivity on independent task families.

Genericity

Five complete pack specs; one working empirical pack.

Two then five empirical adapters pass the same contract and benchmark workflow.

Decision usefulness

A result-linked decision card with alternatives and scope.

Measured user time savings or improved real decisions in a pilot.

Necessary targeted checks: reject incompatible units/backend; exclude post-cutoff data; reject an unknown component; prevent reads of protected outcomes; prevent scoring/objective edits; handle nonfinite predictions and timeouts; stop at budget; preserve a failed mutation; display an inconclusive comparison; reproduce a saved numerical run; invalidate dependent outputs after assumption edits. These verify the engine's claimed behavior, not superficial function coverage.

No paid model calls, new scientific experiments, dataset downloads or application implementation were performed while preparing this document. The cited source pages and architecture were reviewed; actual runtime and empirical results remain build deliverables.

18. The hackathon story and three-minute demo

Pitch: “Downstream of Intelligence turns fragmented real-world evidence into executable models, evolves those models with GPT-6, and tests which investigation or decision should come next.”

The differentiating demonstration is a consequential revision supported by execution. The audience should see the model identify why a plausible approach is wrong, change a specific assumption or mechanism, test the change against a baseline, and produce an appropriately scoped decision. A large agent diagram or a rising score without provenance is insufficient evidence.

Demo time

Show

What it demonstrates

0:00–0:25

Real problem, source snapshot and decision contract.

Practical relevance and testability.

0:25–0:50

GPT-6's Problem DNA and two rival explanations.

Formulation, system structure and uncertainty.

0:50–1:20

Parent code, GPT-6 mutation, compile/run and archive update.

Executable evolutionary reasoning.

1:20–1:50

Actual comparison plus one failure, null result or rejected candidate if present.

Evidence-based judgment rather than a forced win.

1:50–2:15

GPT-6 chooses and runs the next informative action.

Metareasoning changes execution.

2:15–2:40

Proposed method patch and measured parent/child comparison, if executed; otherwise its explicit not-executed status.

Bounded improvement evaluation or an honest account of the remaining work.

2:40–3:00

Decision card, export and another domain's complete specification.

Consumption and reusable design.

Prepare a recorded trace only after a real run; label replay explicitly. Keep live and recorded results distinct. Fill result placeholders from the evaluator at demo time; do not invent improvement percentages. A good demo can retain the parent after a poor mutation while still demonstrating the complete loop.

No plan can guarantee a hackathon win. This design targets concrete judging strengths: useful problem, technically distinct GPT-6 role, real execution, verifiable comparisons, coherent product experience and a plausible route beyond the demo.

19. Subsequent releases and research program

Stage

Outcome

Exit criterion

Four-hour MVP

One real-data lifecycle, one evolved candidate, adaptive action; small method comparison if budget permits.

Clean run with evidence/code/score/decision lineage.

Next working day, assuming prepared data

Complete the second empirical adapter and remove integration shortcuts.

Same contracts operate on two different data/model/evaluator families.

Following week, subject to runtime and data access

Repeated benchmark runs and ablations of templates, semantics, search and metareasoning.

Quality–cost curves, failures and uncertainty on protected tasks.

Domain pilot

One forecaster, operator or researcher uses the exported result.

Measured usefulness and independent review of assumptions.

Multi-generation study

Evolve research methods and optionally their revision procedure.

Fresh-family transfer, total discovery cost, regressions and reproducible generations.

Proposed research hypotheses: structural/observation search improves external quality beyond parameter fitting; archive diversity helps under shift; adaptive evidence/compute allocation improves quality per budget; semantic/provenance checks reduce false conclusions; templates reduce construction cost without losing quality; method revisions transfer; and revising the revision procedure improves future improvement. These require separate ablations and should not collapse into one unexplained benchmark score.

Report novelty only after a proper related-work comparison. NSCORE informs evidence synthesis; ODD informs model documentation; AlphaEvolve and MAP-Elites inform program search/archive design; Darwin Gödel Machine informs empirical self-modification. Their combination alone is not proof of a novel algorithm.

20. Implementation handoff for Codex

Use this brief with the complete specification and a prepared dataset pack:

Build Downstream of Intelligence as a modular GPT-6 Astra real-world problem-solving application. The central mechanism is evolution of executable solution assemblies—models, code, data transformations, training/inference recipes and decision workflows—plus adaptive investigation and separately evaluated research-policy revision. Support a model-asset registry for Hugging Face, API and scientific components; distinguish orchestration, ensembles, neuro-symbolic composition, supported training and compatible adapter merging. Implement the Prediction Markets P1 pack first using a small real resolved-event archive with actual definitions, timestamps and labels; do not replace the empirical comparison with a synthetic result. If the archive is unavailable, record the blocker and use an explicitly selected prepared METR-LA pack for the empirical demonstration.

Use the Responses API with the configured accessible gpt-6-astra model, typed contracts, server-side credentials, SQLite metadata, immutable artifacts and an existing isolated numerical worker. Reuse templates. Implement Problem DNA, evidence/availability lineage, a competent baseline, typed model mutations, parent–child execution and selection, a sparse quality-diversity archive, GPT-selected research actions, budget enforcement, and a separate method archive. Run at least one complete evolutionary generation and one matched parent/child research-policy comparison if time allows. Keep evaluator, final labels, split allocation and budgets outside mutation access.

Provide five domain manifests containing the ten empirical questions in this specification. Enable Run only for working adapters. Expose Problem, Evidence, Models, Runs, Evaluation, Method History and Deliverables through a minimal interface. Numeric displays read actual result records. Decision cards show alternatives, assumptions, reversal conditions and unresolved evidence.

Follow the eight half-hour releases. Save runnable checkpoints, include targeted validation of leakage/units/budget/protected access, and preserve failed or inconclusive branches. Document actual setup, usage, verification, costs and implemented versus planned capabilities. Do not claim RSI, improvement, domain validation or live execution without the corresponding measured evidence.

Suggested module boundaries:

Module

Work

contracts

ProblemDNA, Evidence, SystemSpec, ProgramPatch, ActionProposal, MethodPatch, Run, Verdict, Decision.

providers

GPT-6 request/response handling, schema validation, usage capture and errors.

evidence

Versioned acquisition, semantics, temporal eligibility and transformation lineage.

templates

Compatible model components, compiler, conformance checks and registry.

model_assets

Hugging Face/scientific/API model descriptors, pinned revisions, preprocessing and compatibility.

training

Fitting, supported fine-tuning, ensembles/calibration, checkpoints, compute caps and rollback.

composition

Model/tool/rule graph, compatible fusion, solution-level execution and tests.

evolution

Parent selection, GPT-6 mutations, model archive and comparison context.

metareasoning

State summaries, action proposals, deterministic baseline and stop logic.

methods

Allowed harness revisions, parent/child comparison and method archive.

runner

Isolated execution, resource accounting, artifacts, failure handling and resume.

evaluation

Fixed metrics, protected splits, baselines, uncertainty and audit.

domains

Five domain packs and their data/model/evaluator adapters.

delivery

Workbench views, result-linked decisions, reports and exports.

The final deliverable from the build is a usable engine and evidence of what it executed. This document is the build contract.