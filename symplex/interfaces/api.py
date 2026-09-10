"""Local FastAPI workbench. Secrets and protected evaluator stay on the server."""

import json
import os
import secrets
import threading
import uuid
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from symplex.agents.evolution import Engine
from symplex.agents.roster import ROSTER
from symplex.agents.prompts import manifest as prompt_manifest
from symplex.connectors.registry import ConnectorInput
from symplex.connectors.registry import catalog as connector_catalog
from symplex.connectors.registry import execute as connector_execute
from symplex.core.contracts import (
    Invalid,
    ProblemDNA,
    record,
)
from symplex.core.inputs import (
    ChatInput,
    ComputeInput,
    ContextInput,
    ProblemInput,
    ReferenceInput,
    RevisionInput,
    RunInput,
    SceneInput,
    SolveInput,
    SteeringInput,
    UploadInput,
    WhatIfInput,
)
from symplex.core.proposals import CodeProposal, Review
from symplex.domains.catalog import DATASETS, MODULES, domains, use_cases
from symplex.evaluation.fixtures import synthetic_pack
from symplex.evaluation.research import benchmark
from symplex.evidence.acquisition import download_bam
from symplex.evidence.admission import admit
from symplex.infrastructure.providers import ROLES, OpenAIProvider, RuleProvider
from symplex.infrastructure.storage import Budget, BudgetExhausted, Store
from symplex.infrastructure.execution_scope import ExecutionBusy, problem_budget, problem_execution, workspace_problem_id
from symplex.infrastructure.method_registry import CanaryInput, ReviewInput as MethodReviewInput, RollbackInput, ShadowInput
from symplex.modeling.assets import ASSETS, training_plan
from symplex.observability.tracing import export_trace


def create_app(root=".symplex"):
    load_dotenv(Path.cwd() / ".env", override=False)
    store = Store(root)
    budget = Budget(store, {"usd": float(os.getenv("SYMPLEX_MAX_USD", "5"))})
    token = secrets.token_urlsafe(32)
    active = {}
    lock = threading.Lock()
    app = FastAPI(title="Symplex", version="0.1.0")
    app.state.store, app.state.budget = store, budget
    operation_problem = ContextVar("api_operation_problem", default=None)
    operation_allocation = ContextVar("api_operation_allocation", default=budget)

    def current_budget():
        return operation_allocation.get()

    def scoped_operation(select, *, background=False):
        def decorate(function):
            @wraps(function)
            def invoke(body):
                ident = select(body)
                problem_id = workspace_problem_id(store, ident) if ident else None
                allocation = problem_budget(store, budget, problem_id)
                ptoken = operation_problem.set(problem_id)
                btoken = operation_allocation.set(allocation)
                try:
                    if background:
                        return function(body)
                    with problem_execution(store, allocation, problem_id) as effective:
                        operation_allocation.set(effective)
                        return function(body)
                finally:
                    operation_allocation.reset(btoken)
                    operation_problem.reset(ptoken)
            return invoke
        return decorate

    @app.middleware("http")
    async def local_boundary(request: Request, call_next):
        if request.url.hostname not in ("127.0.0.1", "localhost", "testserver"):
            return JSONResponse({"detail": "Local workbench only"}, status_code=403)
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin and origin != str(request.base_url).rstrip("/"):
                return JSONResponse(
                    {"detail": "Cross-origin requests are not allowed"}, status_code=403
                )
            if not secrets.compare_digest(
                request.headers.get("x-symplex-token", ""), token
            ):
                return JSONResponse(
                    {"detail": "Missing workbench session token"}, status_code=403
                )
            size = 0
            chunks = []
            async for chunk in request.stream():
                size += len(chunk)
                if size > 4 * 1024 * 1024:
                    return JSONResponse(
                        {"detail": "Request exceeds 4 MiB"}, status_code=413
                    )
                chunks.append(chunk)
            request._body = b"".join(chunks)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        )
        return response

    @app.exception_handler(Invalid)
    async def invalid_handler(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(BudgetExhausted)
    async def budget_handler(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(ExecutionBusy)
    async def execution_busy_handler(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    def provider():
        if not os.getenv("OPENAI_API_KEY"):
            raise Invalid("Set OPENAI_API_KEY in the server .env to use live AI")
        return OpenAIProvider(store, current_budget())

    def lookup(ident):
        try:
            return store.get(ident)
        except KeyError:
            raise HTTPException(404, "Artifact not found")

    def launch(label, function, *, problem_id=None):
        problem_id = problem_id or operation_problem.get()
        allocation = problem_budget(store, current_budget(), problem_id)
        with lock:
            if any(x["status"] == "running" for x in active.values()):
                raise HTTPException(409, "An investigation is already running")
            task_id = uuid.uuid4().hex
            event = threading.Event()
            active[task_id] = {
                "id": task_id,
                "label": label,
                "status": "running",
                "cancel": event,
            }

        def work():
            ptoken = operation_problem.set(problem_id)
            btoken = operation_allocation.set(allocation)
            try:
                with problem_execution(store, allocation, problem_id) as effective:
                    operation_allocation.set(effective)
                    result = function(event)
                active[task_id].update(
                    status=result.get("status", "succeeded"), result=result
                )
            except Exception as exc:
                active[task_id].update(status="failed", error=str(exc))
                store.put("failure", {"task": task_id, "error": str(exc)})
            finally:
                operation_allocation.reset(btoken)
                operation_problem.reset(ptoken)

        threading.Thread(target=work, daemon=True).start()
        return {"task_id": task_id, "status": "running"}

    @app.get("/api/state")
    def state():
        # Dataset rows and raw provider output are available only through explicit artifact inspection.
        capability_reports = []
        from symplex.connectors.compute import read_blob

        for blob in store.list("file_blob"):
            if blob["data"]["filename"] == "capabilities.json" and not blob["stale"]:
                try:
                    capability_reports.append(
                        {
                            "artifact_id": blob["id"],
                            "created": blob["created"],
                            "report": json.loads(read_blob(store, blob)),
                        }
                    )
                except (ValueError, Invalid):
                    pass
        visible = [
            r for r in store.list() if r["kind"] not in ("dataset", "model_call", "model_response")
        ]
        return {
            "token": token,
            "records": visible,
            "jobs": store.jobs(),
            "budget": budget.snapshot(),
            "active": [
                {k: v for k, v in task.items() if k != "cancel"}
                for task in active.values()
            ],
            "models": {
                r: os.getenv("OPENAI_" + r.upper() + "_MODEL", m)
                for r, m in ROLES.items()
            },
            "key_configured": bool(os.getenv("OPENAI_API_KEY")),
            "domains": domains(),
            "use_cases": use_cases(),
            "datasets": DATASETS,
            "modules": [
                {"name": n, "file": f, "description": d, "status": s}
                for n, f, d, s in MODULES
            ],
            "runtime_capabilities": capability_reports[-1:],
            "agents": ROSTER,
            "simulators": json.loads(
                (Path(__file__).parents[1] / "modeling/extensions.json").read_text()
            ),
            "connectors": connector_catalog(),
            "prompts": prompt_manifest(),
            "assets": ASSETS,
            "training": [training_plan(a) for a in ASSETS],
        }

    @app.get("/api/artifacts/{ident}")
    def artifact(ident: str):
        return lookup(ident)

    @app.get("/api/export/{ident}")
    def export(ident: str):
        value = lookup(ident)
        root_id = value["data"].get("problem_id", ident)
        records = store.list()
        ids = {root_id}
        for _ in range(8):
            ids.update(r["id"] for r in records if r["parent"] in ids)
        exported = {
            "artifact": value,
            "lineage": [r for r in records if r["id"] in ids],
            "notice": "Includes private supplied evidence if imported. Provider credentials are never stored.",
        }
        return Response(
            json.dumps(exported, indent=2),
            media_type="application/json",
            headers={
                "Content-Disposition": 'attachment; filename="symplex-'
                + ident
                + '.json"'
            },
        )

    @app.post("/api/steer")
    def steer(body: SteeringInput):
        problem = lookup(body.problem_id)
        if problem["kind"] != "workspace_problem" or problem["stale"]:
            raise Invalid("Select a current problem")
        ident = store.put(
            "steering",
            {
                "instruction": body.instruction,
                "kind": body.kind,
                "basis": "user_direction"
                if body.kind == "direction"
                else "user_supplied_unverified",
                "application": "next_agent_decision_boundary",
            },
            body.problem_id,
        )
        store.put(
            "message",
            {"role": "user", "text": body.instruction, "steering_id": ident},
            body.problem_id,
        )
        store.put(
            "message",
            {
                "role": "assistant",
                "model": "Symplex runtime",
                "text": "Steering saved. The agent will consider it at its next decision boundary; an in-flight computation keeps its original inputs.",
                "steering_id": ident,
            },
            body.problem_id,
        )
        for job in store.jobs():
            if job["status"] != "waiting_user" or not job.get("result_id"):
                continue
            checkpoint = store.get(job["result_id"])
            if checkpoint["parent"] == body.problem_id and os.getenv("OPENAI_API_KEY"):
                from symplex.agents.solver import solve

                try:
                    # The worker checks again; this preflight returns a truthful queued
                    # response while an external runner owns the case execution.
                    with problem_execution(store, budget, body.problem_id):
                        pass
                except (ExecutionBusy, Invalid):
                    return {"id": ident, "status": "pending_next_decision_boundary", "resume": "queued_for_scope_owner"}
                with lock:
                    running = any(x["status"] == "running" for x in active.values())
                if running:
                    return {"id": ident, "status": "pending_next_decision_boundary", "resume": "queued_for_running_investigation"}
                task = launch(
                    "Resuming with your steering",
                    lambda cancel: solve(
                        store,
                        current_budget(),
                        provider(),
                        body.problem_id,
                        cancel,
                        checkpoint["data"]["depth"],
                    ),
                    problem_id=body.problem_id,
                )
                return {"id": ident, "status": "resuming", **task}
        return {"id": ident, "status": "pending_next_decision_boundary"}

    @app.get("/api/method-lab/{candidate_id}")
    def method_lab(candidate_id: str):
        from symplex.infrastructure.method_registry import lab_status
        return lab_status(store, candidate_id)

    @app.post("/api/method-lab/review")
    def method_review(body: MethodReviewInput):
        from symplex.infrastructure.method_registry import operator_review
        return operator_review(store, body)

    @app.post("/api/method-lab/shadow")
    def method_shadow(body: ShadowInput):
        from symplex.infrastructure.method_registry import shadow
        return shadow(store, body)

    @app.post("/api/method-lab/canary")
    def method_canary(body: CanaryInput):
        from symplex.infrastructure.method_registry import activate_canary
        return activate_canary(store, body)

    @app.post("/api/method-lab/rollback")
    def method_rollback(body: RollbackInput):
        from symplex.infrastructure.method_registry import rollback
        return rollback(store, body)

    @app.post("/api/compute")
    @scoped_operation(lambda body: body.id, background=True)
    def compute(body: ComputeInput):
        from symplex.connectors.compute import run

        lookup(body.id)
        return launch(
            "Astra simulation engineer · hosted Python",
            lambda cancel: {
                "status": "succeeded",
                "result_id": run(
                    store,
                    current_budget(),
                    provider(),
                    body.id,
                    body.instruction,
                    cancel=cancel,
                    predecessor_package_id=body.predecessor_package_id,
                ),
            },
        )

    @app.get("/api/scenes/columns/{ident}")
    def scene_columns(ident: str):
        import csv
        import io
        from symplex.connectors.compute import read_blob
        blob = lookup(ident)
        if blob["kind"] != "file_blob" or blob["stale"] or not blob["data"]["filename"].endswith(".csv") or blob["data"].get("basis") != "generated":
            raise Invalid("Select a generated CSV")
        rows = csv.DictReader(io.StringIO(read_blob(store, blob).decode("utf-8-sig")))
        return {"columns": rows.fieldnames or [], "samples": [row for _, row in zip(range(3), rows)]}

    @app.post("/api/scenes")
    def create_scene(body: SceneInput):
        from symplex.modeling.scenes import scene_from_csv
        lookup(body.id)
        return {"id": scene_from_csv(store, body.id, body.blob_id, body.time_column,
                                     body.x_column, body.y_column, body.z_column,
                                     body.group_column, filters=body.filters)}

    @app.get("/api/scenes/{ident}")
    def get_scene(ident: str):
        from symplex.modeling.scenes import load_scene
        artifact = lookup(ident)
        provenance = artifact
        if artifact["kind"] == "simulation_scene":
            artifact = lookup(artifact["data"]["scene_blob_id"])
        if artifact["kind"] != "file_blob":
            raise Invalid("Select a generated simulation scene")
        scene = load_scene(store, artifact["parent"], artifact["id"])
        return {"scene": scene, "source_id": provenance["id"], "run_id": artifact["data"]["run_id"],
                "scope": "Replay of generated states; scientific validity is established separately"}

    @app.post("/api/compute/recover")
    @scoped_operation(lambda body: body.id)
    def recover_compute(body: ReferenceInput):
        from symplex.connectors.compute import collect_outputs

        run_record = lookup(body.id)
        return {"id": collect_outputs(store, provider(), run_record["parent"], body.id)}

    @app.post("/api/uploads")
    def upload(body: UploadInput):
        import base64

        from symplex.connectors.compute import save_blob

        problem = lookup(body.problem_id)
        if problem["kind"] != "workspace_problem" or problem["stale"]:
            raise Invalid("Select a current problem")
        try:
            raw = base64.b64decode(body.content_base64, validate=True)
        except ValueError:
            raise Invalid("Invalid encoded file")
        if len(raw) > 2000000:
            raise Invalid("Upload exceeds 2 MB")
        extension = Path(body.filename).suffix.lower()
        signatures = {
            ".png": b"\x89PNG\r\n\x1a\n",
            ".jpg": b"\xff\xd8\xff",
            ".jpeg": b"\xff\xd8\xff",
            ".pdf": b"%PDF-",
        }
        if extension not in signatures or not raw.startswith(signatures[extension]):
            raise Invalid(
                "Upload a valid PNG, JPEG or PDF; use context for text and tables"
            )
        bid = save_blob(store, raw, body.filename, body.problem_id, "user_supplied")
        ident = store.put(
            "binary_context",
            {
                "title": body.filename,
                "format": extension[1:],
                "blob_id": bid,
                "basis": "supplied_unverified",
            },
            body.problem_id,
        )
        return {"id": ident, "blob_id": bid}

    @app.get("/api/files/{ident}")
    def file_content(ident: str, download: bool = False):
        from symplex.connectors.compute import read_blob

        blob = lookup(ident)
        if blob["kind"] != "file_blob":
            raise Invalid("Select a file artifact")
        raw = read_blob(store, blob)
        mime = blob["data"]["mime"]
        inline = mime in ("image/png", "image/jpeg") and not download
        return Response(
            raw,
            media_type=mime if inline else "application/octet-stream",
            headers={
                "Content-Disposition": ("inline" if inline else "attachment")
                + '; filename="symplex-'
                + ident
                + Path(blob["data"]["filename"]).suffix
                + '"',
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.post("/api/connectors/execute")
    @scoped_operation(lambda body: body.problem_id)
    def connector(body: ConnectorInput):
        live = (
            provider() if body.connector in ("web_research", "hosted_compute") else None
        )
        return connector_execute(
            store,
            current_budget(),
            live,
            body,
            {"artifacts.read", "network.research", "network.dataset", "compute.hosted"},
        )

    @app.get("/api/traces/{problem_id}")
    def traces(problem_id: str):
        lookup(problem_id)
        events = export_trace(store, problem_id)
        return Response(
            "\n".join(json.dumps(e) for e in events),
            media_type="application/x-ndjson",
            headers={
                "Content-Disposition": 'attachment; filename="symplex-trace.jsonl"'
            },
        )

    @app.get("/api/contracts")
    def contracts():
        from symplex.agents.visualization import SceneProjection
        from symplex.agents.solver import NextStep
        from symplex.modeling.dynamics import SystemSimulation
        from symplex.modeling.solutions import SolutionBlueprint
        from symplex.modeling.complex_system import ComplexSystemSpec
        from symplex.agents.scientific_cycle import HypothesisReview
        from symplex.agents.improvement import CandidateDesign, MethodCandidate
        from symplex.agents.outcomes import DecisionBrief
        from symplex.evidence.synthesis import EvidenceSynthesis
        from symplex.modeling.scenes import SimulationScene
        from symplex.evaluation.experiments import ExperimentProtocol, ExperimentOutput

        return {
            "problem_dna": ProblemDNA.json_schema(),
            "solution": SolutionBlueprint.json_schema(),
            "complex_system": ComplexSystemSpec.json_schema(),
            "hypothesis_review": HypothesisReview.json_schema(),
            "candidate_design": CandidateDesign.json_schema(),
            "method_candidate": MethodCandidate.json_schema(),
            "decision_brief": DecisionBrief.json_schema(),
            "evidence_synthesis": EvidenceSynthesis.json_schema(),
            "simulation_scene": SimulationScene.json_schema(),
            "scene_projection": SceneProjection.json_schema(),
            "experiment_protocol": ExperimentProtocol.json_schema(),
            "experiment_output": ExperimentOutput.model_json_schema(),
            "system_simulation": SystemSimulation.json_schema(),
            "next_action": NextStep.json_schema(),
            "connector_input": ConnectorInput.model_json_schema(),
            "code_proposal": CodeProposal.json_schema(),
        }

    @app.post("/api/problems")
    def new_problem(body: ProblemInput):
        if body.dataset not in {d["id"] for d in DATASETS} | {"general"}:
            raise Invalid("Unknown dataset")
        ident = store.put(
            "workspace_problem",
            {"question": body.question, "dataset": body.dataset, "status": "draft"},
        )
        return {"id": ident}

    @app.post("/api/plan")
    @scoped_operation(lambda body: body.id)
    def plan(body: ReferenceInput):
        from symplex.agents.context import working_context
        problem = lookup(body.id)
        dna = provider().propose(
            ProblemDNA,
            {
                "problem": problem["data"],
                "datasets": DATASETS,
                "supplied_context": working_context(store, body.id),
                "instruction": "Frame this user's problem. Treat unavailable adapters as dependency gaps; distinguish assumptions from facts.",
            },
            body.id,
        )
        return {"id": store.put("problem_dna", record(dna), body.id)}

    @app.post("/api/context")
    def add_context(body: ContextInput):
        lookup(body.problem_id)
        import csv
        import io

        details = {}
        if body.format == "csv":
            rows = list(csv.DictReader(io.StringIO(body.content)))
            details = {
                "rows": len(rows),
                "columns": list(rows[0]) if rows else [],
                "sample": rows[:5],
            }
        elif body.format in ("json", "geojson"):
            try:
                parsed = json.loads(body.content)
            except ValueError:
                raise Invalid("Invalid JSON")
            if body.format == "geojson":
                from symplex.evidence.spatial import validate_geojson

                parsed = validate_geojson(parsed)
            details = {"parsed": parsed}
        ident = store.put(
            "context",
            {
                **body.model_dump(),
                "inspection": details,
                "status": "supplied_unverified",
                "availability": "Unknown unless established by source metadata",
            },
            body.problem_id,
        )
        return {"id": ident}

    @app.post("/api/chat")
    @scoped_operation(lambda body: body.problem_id)
    def chat(body: ChatInput):
        problem = lookup(body.problem_id) if body.problem_id else None
        history = [
            r["data"] for r in store.list("message") if r["parent"] == body.problem_id
        ]
        messages = [
            {"role": x["role"], "content": x["text"]} for x in history[-10:]
        ] + [{"role": "user", "content": body.message}]
        store.put("message", {"role": "user", "text": body.message}, body.problem_id)
        decisions = [
            {"id": r["id"], "title": r["data"]["title"], "scope": r["data"]["scope"]}
            for r in store.list("decision")
            if not r["stale"]
        ]
        result = provider().chat(
            messages,
            {
                "problem": problem["data"] if problem else None,
                "datasets": DATASETS,
                "decisions": decisions,
                "implemented_adapters": [
                    "P1 prepared slice",
                    "BamTwoogle research pilot",
                ],
                "execution_rule": "Chat proposes; user launches experiments with Run. Code drafts are unexecuted.",
            },
            body.problem_id,
            body.research,
        )
        ident = store.put("message", dict(result, role="assistant"), body.problem_id)
        return {"id": ident, **result}

    @app.post("/api/datasets/bamtwoogle")
    def acquire():
        return {"id": download_bam(store, budget)["id"]}

    @app.post("/api/datasets/import")
    def import_data(body: dict):
        summary = admit(body)
        ident = store.put("imported_dataset", {"pack": body, "summary": summary})
        return {"id": ident, "summary": summary}

    @app.post("/api/run")
    def run(body: RunInput):
        if body.kind == "research":
            live = provider()
            return launch(
                "BamTwoogle · live research policy pilot",
                lambda cancel: benchmark(store, budget, live, cancel),
            )
        if body.kind == "fixture":
            data = synthetic_pack()
        else:
            if not body.dataset_id:
                raise Invalid("Import a prepared P1 dataset first")
            data = lookup(body.dataset_id)["data"].get("pack")
            if data is None:
                raise Invalid("Not a prepared P1 dataset")
        proposer = provider() if body.live else RuleProvider()
        return launch(
            ("Synthetic check" if data["synthetic"] else "P1 empirical comparison")
            + " · "
            + proposer.mode,
            lambda cancel: Engine(store, budget, proposer).investigate(data, cancel),
        )

    @app.post("/api/cancel")
    def cancel(body: ReferenceInput):
        if body.id not in active:
            raise HTTPException(404, "Task not found")
        active[body.id]["cancel"].set()
        return {
            "status": "cancellation_requested",
            "note": "Current API request may finish; no further episodes will start",
        }

    @app.post("/api/validate")
    @scoped_operation(lambda body: body.id)
    def validate(body: ReferenceInput):
        decision = lookup(body.id)
        if decision["kind"] != "decision":
            raise Invalid("Select a decision artifact")
        if decision["stale"]:
            raise Invalid("Decision is stale after an assumption change")
        result = provider().propose(
            Review,
            {
                "artifact_id": body.id,
                "decision": decision["data"],
                "instruction": "Independently check whether claims exceed recorded results. Refer only to the supplied artifact ID. This review cannot change numerical scores.",
            },
            body.id,
            role="validator",
        )
        if not set(result["result_ids"]).issubset({body.id}):
            raise Invalid("Reviewer referenced an unavailable result")
        return {"id": store.put("review", result, body.id), **result}

    @app.post("/api/code")
    @scoped_operation(lambda body: body.id)
    def code(body: ReferenceInput):
        problem = lookup(body.id)
        result = provider().propose(
            CodeProposal,
            {
                "problem": problem["data"],
                "instruction": "Draft a small pure Python adapter or model candidate for this problem. It will be saved for inspection, never executed. Name dependencies and meaningful verification checks. No credentials, networking, shell or file writes.",
            },
            body.id,
        )
        return {
            "id": store.put(
                "code_proposal",
                dict(
                    result,
                    status="unexecuted",
                    reason="No isolated arbitrary-code backend configured",
                ),
                body.id,
            )
        }

    @app.post("/api/solutions")
    @scoped_operation(lambda body: body.id)
    def solutions(body: ReferenceInput):
        from symplex.modeling.solutions import design_solution

        lookup(body.id)
        return {"id": design_solution(store, provider(), body.id)}

    @app.post("/api/solve")
    @scoped_operation(lambda body: body.id, background=True)
    def solve_problem(body: SolveInput):
        from symplex.agents.solver import solve

        lookup(body.id)
        return launch(
            "Autonomous problem solver · " + body.depth,
            lambda cancel: solve(store, current_budget(), provider(), body.id, cancel, body.depth),
        )

    @app.post("/api/complex-system")
    @scoped_operation(lambda body: body.id, background=True)
    def complex_system(body: ReferenceInput):
        from symplex.agents.scientific_cycle import represent_system

        lookup(body.id)
        return launch(
            "Astra complexity architect",
            lambda cancel: {
                "status": "succeeded",
                "result_id": represent_system(store, provider(), body.id),
            },
        )

    @app.post("/api/scenarios")
    @scoped_operation(lambda body: body.id, background=True)
    def scenarios(body: ReferenceInput):
        from symplex.modeling.dynamics import experiment

        lookup(body.id)
        return launch(
            "Stochastic system experiment",
            lambda cancel: {
                "status": "succeeded",
                "result_id": experiment(store, current_budget(), provider(), body.id, cancel),
            },
        )

    @app.post("/api/what-if")
    @scoped_operation(lambda body: body.model_id)
    def what_if(body: WhatIfInput):
        from symplex.infrastructure.runner import execute_system
        from symplex.modeling.dynamics import SystemSimulation

        model = lookup(body.model_id)
        if model["kind"] != "system_hypothesis" or model["stale"]:
            raise Invalid("Choose a current system hypothesis")
        spec = {
            k: v for k, v in model["data"].items() if k in SystemSimulation.model_fields
        }
        if body.node_id not in {n["id"] for n in spec["nodes"]}:
            raise Invalid("Unknown state node")
        spec["scenarios"] = [
            spec["scenarios"][0],
            {
                "id": "what_if",
                "name": "User what-if",
                "interventions": [{"node": body.node_id, "value": body.value}],
                "interpretation": "User-selected normalized intervention; not an observed condition",
            },
        ]
        mid = store.put(
            "system_hypothesis",
            dict(
                spec,
                status="proposed",
                template="bounded-influence-v1",
                parent_model_id=body.model_id,
            ),
            model["parent"],
        )
        result = execute_system(store, current_budget(), spec)
        rid = store.put(
            "scenario_run",
            dict(result, status="succeeded", model_id=mid),
            model["parent"],
        )
        return {
            "id": store.put(
                "scenario_report",
                {
                    "title": "What-if: " + model["data"]["title"],
                    "status": "conditional_scenario_comparison",
                    "model_id": mid,
                    "run_id": rid,
                    "ablation_run_id": None,
                    "scenario_results": result["scenarios"],
                    "assumptions": spec["assumptions"],
                    "unmodeled": spec["unmodeled"],
                    "disconfirmation": spec["disconfirmation"],
                    "scope": result["scope"],
                    "empirically_validated": False,
                    "synthetic": True,
                    "problem_id": model["parent"],
                    "next_action": "Validate these assumptions and intervention feasibility with domain evidence.",
                },
                model["parent"],
            )
        }

    @app.post("/api/revise")
    def revise(body: RevisionInput):
        old = lookup(body.id)
        new = store.put(
            old["kind"],
            dict(
                old["data"],
                assumption=body.assumption,
                revises=body.id,
                status="needs_new_evaluation",
            ),
        )
        # Carry supplied inputs forward by value; computed conclusions are invalidated.
        for supplied in store.list():
            if (
                supplied["parent"] == body.id
                and supplied["kind"] in ("context", "binary_context", "steering")
                and not supplied["stale"]
            ):
                store.put(
                    supplied["kind"],
                    dict(supplied["data"], copied_from=supplied["id"]),
                    new,
                )
        store.invalidate(body.id)
        return {"id": new, "status": "prior_descendants_marked_stale"}

    @app.post("/api/capabilities")
    def capabilities():
        return provider().capability()

    @app.get("/api/research-brief")
    def brief():
        return FileResponse(Path(__file__).with_name("web") / "research.html")

    @app.get("/")
    def home():
        return FileResponse(Path(__file__).with_name("web") / "index.html")

    app.mount(
        "/static", StaticFiles(directory=Path(__file__).with_name("web")), name="static"
    )
    return app
