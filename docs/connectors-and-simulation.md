# Symplex: agent reasoning and execution surfaces

The controller receives a problem, current problem DNA, user context and steering, available tools, previous outcomes, model criticism and remaining resources. It decides what uncertainty to address next. Connectors expose capabilities; they never choose a canned answer from a problem name.

## Reasoning stack

`agents/prompts/` contains the actual prompts consumed by the runtime, not documentation copies. `core/` defines user and model output contracts. `modeling/solutions.py` defines hypotheses, subproblems, component interfaces, temporal/unit meanings, experiment requirements and the useful final deliverable. `agents/context.py` selects current problem-scoped artifacts; protected evaluation labels are excluded. Run manifests record prompt and execution-policy digests.

The metareasoner can frame, inspect, retrieve, discover literature, research sources, design, execute models, challenge them, request clarification and deliver. A user can add data, assumptions, judgments or steering. An in-flight call keeps its original input; new steering enters the next decision. Clarification pauses persist across process restarts and resume from the same checkpoint after steering.

The existing model-evolution and research-policy loops use narrower, fixed grammars. Those grammars preserve a meaningful evaluator. Changing a model or a method is distinct from claiming improved scientific performance. General prompt self-modification is not enabled without separate promotion evidence.

## Execution stack

The [OpenAI Code Interpreter](https://developers.openai.com/api/docs/guides/tools-code-interpreter) supports Astra writing and running Python in a hosted sandbox and producing files. Symplex uses the official SDK, a 1 GB container and bounded tool calls, records the returned code/tool statuses, then downloads the resulting files before the container expires. Model-generated code never runs on the application host. Inputs are scoped to the selected problem; source hashes and output hashes are retained.

The simulation engineer selects equations and methods from the problem rather than a fixed industry switch. It can implement mechanistic dynamics, stochastic processes, population interactions, symbolic checks, graph reasoning, probabilistic inference and forecasting with available libraries. It must inspect actual availability, represent assumptions explicitly and report missing components. The host validates schemas, artifacts, permissions and numerical-template evaluations; generated diagnostic code remains maker-authored evidence, not an independent verifier.

The first live hosted run generated two compartment-model structures, performed conservation, positivity, limiting-case and solver-accuracy checks, ran sensitivity sweeps, and returned Python, CSV, JSON and PNG files. All inputs were synthetic and explicitly labeled. This demonstrates executable model construction, not protein efficacy or physiological calibration.

## Connected capabilities

- **Data:** supplied text/CSV/JSON/GeoJSON plus PNG/JPEG/PDF artifacts. [DuckDB](https://duckdb.org/docs/stable/clients/python/overview) profiles uploaded CSV through fixed read-only operations. No arbitrary user or model SQL is executed. [Croissant](https://docs.mlcommons.org/croissant/) validates the implemented Google BamTwoogle metadata mapping. Other datasets need a reviewed source mapping, not just a catalog URL.
- **Research:** Astra Responses web search retains citations. [Crossref](https://www.crossref.org/documentation/retrieve-metadata/rest-api/) returns scholarly metadata through a fixed, bounded endpoint. Metadata discovery does not establish full-text access, entailment or scientific truth. Dedicated OpenAI deep-research background jobs remain a planned adapter; current bounded multi-action Astra research is a separate implemented capability. The Codex Deep Research plugin is not automatically an application credential or backend.
- **Semantics:** local TF-IDF retrieval returns artifact IDs, hashes and offsets. This is lexical relevance, not embedding inference or entailment. [RDFLib](https://rdflib.readthedocs.io/en/stable/) projects typed component relationships, units and timing into RDF. Proposed relationships retain their hypothesis status. No arbitrary remote JSON-LD or SPARQL is followed.
- **Visualization:** the workbench renders actual component graphs, hypotheses, action sequences, supplied geometry, stochastic trajectories and generated image files. Scenario exports use declarative [Vega-Lite](https://vega.github.io/vega-lite/docs/spec.html) with inline data. Active generated HTML/JavaScript is not executed in the workbench. Molecular structures and simulator-specific trajectories can be preserved as files; advanced 3D viewers are extension work.
- **Tracing:** [OpenTelemetry](https://opentelemetry.io/docs/languages/python/instrumentation/) spans wrap tools, connectors and hosted computation. Local artifact records are authoritative. JSONL exports omit raw prompts, documents and model output. No external telemetry collector is silently enabled by ambient environment variables.
- **Enterprise access:** remote MCP, S3/Arrow/Parquet and STAC/xarray adapters are declared extension points. They require source-specific scopes, authentication, pagination, licensing and lineage before being marked connected. App/plugin availability in Codex does not automatically grant Symplex access.

## Scientific extension design

Libraries belong to model families, not to hardcoded use cases. The agent selects a family based on mechanism, observability, required inference, time scale, data quality and evaluation criteria.

**Biology and proteins.** [OpenMM](https://docs.openmm.org/latest/userguide/) supports atomistic molecular simulation when a suitable structure, force field and simulation protocol are available. [BioSimulators](https://docs.biosimulations.org/) provides a separate route for standardized biological model execution. Molecular dynamics, biochemical reaction networks and organism-level absorption are different modeling problems; one does not automatically validate another. A future [3Dmol.js](https://3dmol.org/doc/) viewer can render genuine molecular structures and trajectories without fabricating atom coordinates.

**Population, behavior and crowds.** [Mesa](https://mesa.readthedocs.io/stable/getting_started.html) is an agent-based modeling framework. The scientific requirements include heterogeneous behavior, update schedules, interaction structure, random seeds, sensitivity and validation against micro and macro observations. An agent can implement a small explicit population model in numerical Python when the framework is absent; it must report what actually ran.

**Spatial infrastructure.** [SUMO/TraCI](https://sumo.dlr.de/docs/TraCI/Interfacing_TraCI_from_Python.html) provides traffic simulation interfaces. [PySTAC Client](https://pystac-client.readthedocs.io/en/stable/) supports searching standardized earth-observation catalogs. Neither is a general airport-location or land-use decision model. Network geometry, CRS, demand, constraints and counterfactual assumptions need their own contracts.

**Finance and forecasting.** [QuantLib](https://www.quantlib.org/docs.shtml) supports financial instruments and pricing/risk models. Forecast evaluation requires point-in-time information, explicit targets and horizons, calibration diagnostics, temporal splits and stress tests. Simulated price paths do not establish profitable execution. No trading actuator is exposed.

**Probabilistic and symbolic systems.** [PyMC](https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/posterior_predictive.html) supports prior/posterior predictive workflows. SciPy, SymPy and NetworkX support numerical, symbolic and graph components. The agent must separately establish identifiability, convergence, unit consistency, uncertainty propagation and predictive validity.

The live runtime probe discovered NumPy, SciPy, SymPy, NetworkX, PyMC, ArviZ, GeoPandas, Rasterio, xarray, PyTorch CPU and RDKit metadata. It did not find Mesa, OpenMM or QuantLib. This is a timestamped container observation, not a guarantee about every future container or successful imports. The workbench exposes the actual probe file.

## Adding an adapter

1. Define its input/output contract, source provenance, resource limits and failure behavior.
2. Implement one connector handler in `connectors/`; do not import a module path supplied by a model.
3. Register the handler, required permission, SDK and operational status in `connectors/registry.py`.
4. Add a narrowly described agent tool if autonomous use is useful. The tool registry enforces permissions before invoking the handler.
5. Preserve result artifacts and spans. Add a renderer only for validated output types.
6. Test permission denial, malformed inputs, stale/cross-problem references, actual execution and unsupported scientific claims. Only then upgrade availability status.

Production needs durable remote-job recovery, tenant authorization, governed remote sources, specialist compute infrastructure and independent domain evaluation. These are explicit implementation requirements, not properties implied by a sophisticated prompt.
