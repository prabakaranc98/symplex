**The core of Symplex should be an agent that constructs and improves representations of a problem—and uses those representations to develop and test solutions.**

That is the subtle distinction your examples require. The agent must reason about **what the system is, how it behaves, what can be changed, and what would establish that a proposed change works**.

I would define it as:

> **Symplex is a problem-solving environment where Astra turns real-world goals and heterogeneous evidence into executable models, candidate solutions, and experiments—then improves both the solutions and the methods used to develop them.**

**Three things evolve inside the product**

These deserve separate representations, interfaces, and evaluations:

| What evolves                   | Its purpose                                                 | Enzyme-deficiency example                                                              |
| ------------------------------ | ----------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| **The model of the system**    | Explain and predict relevant behavior.                      | How enzyme activity, delivery, digestive conditions, and measurements interact.        |
| **The candidate solution**     | Change the system toward the intended outcome.              | A candidate enzyme, formulation, delivery approach, or combination.                    |
| **The problem-solving method** | Improve how the engine investigates and develops solutions. | How it selects models, searches designs, chooses experiments, and interprets failures. |

A more accurate biological model does not automatically produce a useful product. A promising candidate does not automatically establish a better research method. Symplex needs to connect these three levels while evaluating each appropriately.

**The engine needs several connected representations**

Astra should be able to move between them and check whether each translation preserves the meaning of the problem.

| Representation                      | What it contains                                                             | What makes it useful                                     |
| ----------------------------------- | ---------------------------------------------------------------------------- | -------------------------------------------------------- |
| **Intent and decision**             | Desired outcome, beneficiaries, constraints, alternatives, success criteria. | Establishes what “solving” means.                        |
| **Semantic**                        | Entities, concepts, measurements, definitions, source claims.                | Makes heterogeneous information interpretable together.  |
| **Structural**                      | Mechanisms, dependencies, feedback, compartments, networks, time scales.     | Makes the proposed organization of the system explicit.  |
| **Mathematical**                    | Equations, distributions, constraints, objectives, observation models.       | Specifies consequences that can be calculated or tested. |
| **Computational**                   | Executable components, trained models, simulators, solvers, adapters.        | Produces actual numerical results.                       |
| **Empirical and decision evidence** | Measurements, experiments, diagnostics, comparisons, uncertainty.            | Determines what the model or solution can support.       |

These are working representations that change during investigation. For example, a failed experiment might reveal that the semantic definition of an endpoint was inconsistent across datasets—not that a more elaborate neural model is needed.

**A concrete walkthrough: enzyme-deficiency research**

Start with the user’s request:

> “Help develop a protein-based solution for digestive enzyme deficiency associated with pancreatitis.”

The first useful reasoning step is to identify the intended biological problem. Exocrine pancreatic insufficiency can involve insufficient enzyme production, insufficient delivery, or problems with enzymes mixing or functioning appropriately. Existing treatment includes pancreatic enzyme replacement therapy. Those facts make the mechanism and comparator important starting points. [NIDDK explanation](https://www.niddk.nih.gov/health-information/digestive-diseases/exocrine-pancreatic-insufficiency/definition-facts), [NIDDK treatment overview](https://www.niddk.nih.gov/health-information/digestive-diseases/exocrine-pancreatic-insufficiency/treatment)

**1. Establish the development question.**
Astra drafts a target specifying the population or experimental preparation, measured endpoint, intended improvement, existing comparator, feasible design variables, and constraints. It identifies whether the investigation should pursue a new enzyme, improve a formulation, improve delivery, or compare these approaches.

**2. Construct competing explanations.**
Proposed hypotheses might concern activity, stability, delivery, experimental conditions, or measurement artifacts. These are candidate explanations, with explicit distinguishing observations. Several may operate together.

**3. Assemble the system model.**
The model could connect molecular properties, enzyme activity, formulation behavior, relevant digestive conditions, and assay measurements. Each connection needs a declared meaning and evidence basis.

A neural component might estimate a molecular property or extract an assay measurement. A mechanistic component might describe relevant dynamics. A probabilistic observation model handles measurement variability. Symbolic constraints express admissible designs and consistency requirements.

**4. Acquire and synthesize the necessary evidence.**
The agent connects measurements with their protocols, conditions, uncertainty, and provenance. It checks whether different studies measured comparable quantities. Missing evidence becomes a specific acquisition or experiment task.

Synthetic observations can exercise the inference machinery. Their origin remains explicit.

**5. Construct and compare candidate solutions.**
Astra retrieves existing components, generates configurations or code, and proposes candidate designs within a defined search space. Compatible specialist models can be connected through assessed interfaces.

The resulting artifact is more than a model score: it includes the candidate specification, predicted behavior, assumptions, feasibility questions, and required tests.

**6. Choose the next informative experiment.**
Suppose two candidate explanations fit the current observations but predict different behavior under another experimental condition. The agent proposes a feasible distinguishing test and explains why its outcome could change the design choice.

**7. Deliver and learn.**
The initial usable output could be a reproducible candidate comparison and experimental plan for a research team. Subsequent measured results update the model and design. Laboratory performance and clinical benefit remain separate evidence claims.

**Astra’s harness makes this reasoning operational**

Every substantive proposal should have a contract:

> **Target artifact → proposed change → supporting evidence → expected observable consequence → required computation/test → budget → acceptance criterion.**

For example:

> “Modify the delivery component because these measurements are inconsistent with its current assumption. Refit the affected parameters, compare predictions on the reserved development cases, and retain the change only if the specified checks pass.”

The harness supplies:

* Versioned problem state, evidence, models, designs, and investigation memory.
* Typed tools for retrieval, model construction, code execution, inference, and experimentation.
* Semantic and numerical checks between representations.
* Resource allocation, checkpoints, failure recovery, and bounded execution.
* Evaluation records and feedback that the candidate cannot rewrite.

Astra supports structured outputs and function calling for implementing these contracts. [OpenAI model documentation](https://developers.openai.com/api/docs/models/gpt-6-astra)

**Recursive improvement happens at the method level**

Imagine the engine repeatedly spends compute optimizing candidate proteins before discovering that delivery assumptions dominate the outcome.

It can propose a revised investigation policy: **check decision-sensitive delivery uncertainty before expanding molecular search**.

That revision becomes a frozen engine candidate, tested against the previous version on fresh tasks with comparable resources. If it improves performance within the tested scope, it becomes available to future investigations.

The recursive question is then: **does that improved engine become better at generating and validating further improvements?** That requires evaluating subsequent revision sequences. It is stronger than showing that one candidate improved.

**The same architecture changes its scientific content across applications**

For an airport, the system model connects demand, access, operating capacity, environmental effects, and costs; the candidate solution is a location and development package.

For market creation, it connects participants, unmet needs, incentives, transaction rules, supply response, and adoption; the candidate solution is a market mechanism and launch strategy, tested through appropriate evidence and pilots.

For prediction markets, it connects event definitions, dependencies, observations, and uncertainty; the output is an accountable forecast and an updating policy.

The user enters through conversation, then works with **linked evidence, system models, candidate solutions, experiments, and results**. They can upload context, inspect a mechanism, change an assumption, run a comparison, and see exactly why a recommendation changed.

**The research question worth owning is whether this combination—explicit representations, executable models, adaptive experimentation, and evaluated method improvement—produces better real-world solutions per unit of evidence and computation.**
