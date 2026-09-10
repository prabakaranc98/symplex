# Symplex — complex-system problem solving

Symplex uses Astra to search for useful explanations and representations, make them executable, test their consequences, and choose what to investigate next. The outer loop chooses the investigation; the inner loop does the evidence, modeling and computation. Neither loop establishes scientific truth by itself.

```mermaid
flowchart TB
    U[User: problem, context, judgment, steering] --> D[Problem DNA: decision, boundary, endpoints]
    D --> S[Persistent investigation state]
    subgraph Outer[Outer loop: epistemic and evolutionary search]
      S --> P[Propose: framings, rivals, structures, methods]
      P --> A[Candidate archive: retain alternatives and lineage]
      A --> C[Controller: choose parent, operator and next test]
      C --> G[Host: contracts, permissions, resource limits]
    end
    subgraph Inner[Inner loop: executable scientific investigation]
      G --> R[Semantic and structural complex-system representation]
      R <--> E[Evidence and synthesis: sources, data, artifacts, disagreements]
      R --> H[Hypothesis critique: discrimination and identifiability]
      H --> F[Freeze experiment: baseline, rivals, checks, metrics]
      F --> X[Astra coding and modeling in hosted sandbox]
      X --> O[Recorded code, outputs, provenance and failures]
      O --> L[Source-symbol and state-column linkage]
      O --> V[Host numerical checks and comparative evaluation]
      V --> J[Separate model critique: assumptions and validity gaps]
      J --> S
      E --> S
    end
    J --> B[Decision brief: scoped claims, tradeoffs, uncertainty]
    O --> W[Visualizers: plots, graph, raw files, recorded 3D states]
    B --> U
    W --> U
    S --> M[Method candidate: uninstalled]
    M --> T[Development and regression replay: equal caps]
    T --> Q[Protected fresh-task contract replay]
    Q --> HR[Human review: evidence, problem scope, failure limits]
    HR -->|explicit activation and revalidated gates| K[Scoped instruction canary]
    M --> SH[Shadow designation: baseline unchanged]
    K --> P
    K --> MO[Recorded job and host-assessment failures]
    MO -->|threshold breach or operator action| RB[Rollback: baseline on future invocations]
```

Edges describe implemented responsibilities and conditional transitions; their presence does not mean a candidate has passed them. The method backend supports one complexity-architect instruction change and measures software-contract reliability only. Human review and canary activation are separate explicit actions. A feature being implemented does not mean it is validated for a particular scientific domain.

```mermaid
mindmap
  root((Symplex))
    User workbench
      Problem and context
      Evidence and system view
      Modeling studio
      Comparisons and uncertainty
      Recorded 3D trajectories
      Decision briefs
      Method Lab operator controls
    Reasoning stack
      Problem DNA
      Semantic and structural representations
      Rival hypotheses
      Meta critique
      Synthesis science
      Epistemic search
    Execution stack
      Typed tool registry
      Hosted Python coding
      Explicit predecessor revisions
      Component to source-symbol maps
      Runtime capability probes
      Data and artifact connectors
      Persistent state and traces
    Evaluation stack
      Frozen protocols
      Raw CSV numerical checks
      Baseline and rival comparisons
      Separate model critique
      Paired architect contract replay
      Protected evaluation partitions where implemented
      Missing external validation stays visible
    Evolution scopes
      World model
      Candidate design or intervention
      Investigation method
        Proposal supported
        Fresh-task contract replay for one role
        Staged reports and human review
        Problem-scoped canary overlays
        Failure monitoring and rollback
        No automatic promotion
```

The implementation is arranged by responsibility:

- `agents/`: controller, search policy, scientific-cycle services, role prompts and input context.
- `modeling/`: typed complex-system contracts, scenario runtime, scene contracts and projections.
- `evidence/` and `connectors/`: admission, synthesis, source retrieval, data profiling and hosted compute.
- `evaluation/`: frozen experiments, numerical checks, source linkage, execution assessment and paired method replay.
- `infrastructure/` and `observability/`: SDK routing, accounting, storage, execution boundaries, reviewed method registry, canary monitoring and traces.
- `interfaces/`: API and browser workbench; `tests/`: behavior and failure-path regression tests.

The search design draws on [MAP-Elites](https://arxiv.org/abs/1504.04909), which retains quality across niches, and [AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/), which evaluates explicit executable candidates and uses their results to guide subsequent proposals. Symplex’s generic archive currently uses host numerical-check coverage as a software-quality proxy. It is not a scientific-accuracy score or evidence of superior problem solving.

Three forms of improvement remain distinct: a better representation of the world, a better proposed intervention, and a better reusable investigation method. The architect replay compares frozen parent and candidate prompts on newly supplied tasks using matching settings and equal persistent caps. The registry checks development, regression and held-out reports in order, rejects paired regressions, and requires a held-out contract gain. Incomplete accounting makes a report inconclusive. Detailed replay inputs and outputs stay outside the originating problem's agent context, which receives only aggregate results.

Method Lab shows those recorded gates and exposes explicit review, shadow, canary and rollback controls. Shadow records do not run additional comparisons. An approved canary applies only to its problem allowlist and future invocations; its version participates in solver job identity. Execution failures and failed host assessments trigger rollback at reviewed limits. These controls do not measure scientific utility, semantic novelty or broad operational drift, and no candidate is automatically promoted.

Component maps provide a separate structural check: each component names an existing saved Python symbol or an omission, and mapped state outputs name actual CSV columns. Source digests bind these links to the recorded run. This does not prove that the symbol executed or correctly implements the declared mathematics.
