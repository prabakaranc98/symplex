"""Agent-authored code in the hosted OpenAI sandbox; never executed on the application host."""

import hashlib
from pathlib import Path

from symplex.agents.prompts import frozen_prompts, prompt
from symplex.core.contracts import Invalid, canonical, digest
from symplex.infrastructure.runner import Cancelled
from symplex.observability.tracing import span

ALLOWED_EXTENSIONS = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".csv": "text/csv",
    ".json": "application/json",
    ".py": "text/plain",
    ".txt": "text/plain",
    ".md": "text/plain",
    ".pdb": "text/plain",
    ".sdf": "text/plain",
    ".geojson": "application/geo+json",
    ".pdf": "application/pdf",
}


REVISION_EXTENSIONS = {".py", ".json", ".csv"}
MAX_REVISION_FILES = 6
MAX_REVISION_BYTES = 4000000
MAX_INPUT_BYTES = 2000000
AUTO_RESULT_KINDS = ("execution_assessment", "numerical_verification", "experiment_comparison")


def _source(store, ident, problem_id, kind):
    """Resolve references before network activity; a dangling reference is not input."""
    try:
        record = store.get(ident)
    except KeyError:
        raise Invalid("Missing compute source: " + ident) from None
    if record["kind"] != kind or record["parent"] != problem_id or record["stale"]:
        raise Invalid("Compute source must be current and belong to this problem")
    return record


def _revision_inputs(store, problem_id, package_id):
    """An explicit predecessor creates a revision; latest output never selects itself."""
    if package_id is None:
        return None, []
    package = _source(store, package_id, problem_id, "compute_package")
    run_record = _source(store, package["data"]["run_id"], problem_id, "compute_run")
    candidates = []
    for ident in dict.fromkeys(package["data"].get("file_ids", [])):
        blob = _source(store, ident, problem_id, "file_blob")
        if (
            blob["data"].get("run_id") != run_record["id"]
            or blob["data"].get("basis") != "generated"
        ):
            raise Invalid("Predecessor file does not belong to its executed run")
        if Path(blob["data"]["filename"]).suffix.lower() in REVISION_EXTENSIONS:
            candidates.append(blob)
    order = {".py": 0, ".json": 1, ".csv": 2}
    candidates.sort(
        key=lambda r: (order[Path(r["data"]["filename"]).suffix.lower()], r["id"])
    )
    inputs, omitted = [], []
    size = 0
    for blob in candidates:
        raw = read_blob(store, blob)
        if (
            len(raw) > MAX_INPUT_BYTES
            or len(inputs) >= MAX_REVISION_FILES
            or size + len(raw) > MAX_REVISION_BYTES
        ):
            omitted.append(
                {"id": blob["id"], "reason": "Revision input envelope exceeded"}
            )
            continue
        inputs.append(
            {
                "record": blob,
                "raw": raw,
                "filename": "previous_"
                + blob["id"]
                + "_"
                + Path(blob["data"]["filename"]).name,
                "basis": "generated_predecessor",
            }
        )
        size += len(raw)
    if not inputs:
        raise Invalid("Predecessor has no usable generated Python, JSON or CSV inputs")
    diagnostics = []
    for kind in AUTO_RESULT_KINDS:
        records = [r for r in store.list(kind) if r["parent"] == problem_id
                   and not r["stale"] and r["data"].get("package_id") == package_id]
        if records:
            record = records[-1]
            encoded = canonical(record["data"])
            diagnostics.append({**{key: record[key] for key in ("id", "kind", "digest")},
                                "data_excerpt": encoded[:2000], "truncated": len(encoded) > 2000})
    return {
        "package_id": package["id"],
        "package_digest": package["digest"],
        "run_id": run_record["id"],
        "run_digest": run_record["digest"],
        "summary": package["data"].get("summary", "")[:6000],
        "file_ids": [i["record"]["id"] for i in inputs],
        "omitted_files": omitted,
        "diagnostics": diagnostics,
        "authority": "Generated predecessor artifacts; execution does not establish scientific validity",
    }, inputs


@frozen_prompts
def run(
    store,
    budget,
    provider,
    problem_id,
    instruction="",
    cancel=None,
    *,
    predecessor_package_id=None,
):
    problem = store.get(problem_id)
    if problem["kind"] != "workspace_problem" or problem["stale"]:
        raise Invalid("Select a current problem")
    if cancel and cancel.is_set():
        raise Cancelled("Cancelled before hosted compute")
    known = [r for r in store.list() if r["parent"] == problem_id and not r["stale"]]
    from symplex.agents.context import working_context

    # Generated compute output enters a new run only through an explicit predecessor.
    # This prevents collecting an output from changing the identity of its own retry.
    evidence = working_context(
        store, problem_id, excluded_kinds={"compute_package", "compute_run", "artifact_inspection", *AUTO_RESULT_KINDS}
    )
    predecessor, inputs = _revision_inputs(store, problem_id, predecessor_package_id)
    # Full scientific contracts travel as files; context excerpts are only an index.
    for kind in (
        "complex_system",
        "experiment_protocol",
        "candidate_design",
        "evidence_synthesis",
    ):
        contracts = [r for r in known if r["kind"] == kind][-1:]
        for r in contracts:
            raw = canonical(r["data"]).encode()
            if len(raw) > MAX_INPUT_BYTES:
                raise Invalid("Scientific contract exceeds compute input envelope")
            inputs.append(
                {
                    "record": r,
                    "raw": raw,
                    "filename": r["id"] + ".json",
                    "basis": "proposed_contract",
                }
            )
    for r in [a for a in known if a["kind"] in ("context", "binary_context")][-5:]:
        if r["kind"] == "context":
            raw = r["data"]["content"].encode()
            extension = {"csv": ".csv", "json": ".json", "geojson": ".geojson"}.get(
                r["data"]["format"], ".txt"
            )
            item = {
                "record": r,
                "raw": raw,
                "filename": r["id"] + extension,
                "basis": r["data"].get("basis", "user_context"),
            }
        else:
            blob = _source(store, r["data"]["blob_id"], problem_id, "file_blob")
            raw = read_blob(store, blob)
            item = {
                "record": r,
                "blob": blob,
                "raw": raw,
                "filename": r["id"] + "_" + Path(blob["data"]["filename"]).name,
                "basis": blob["data"].get("basis", "user_context"),
            }
        if len(raw) > MAX_INPUT_BYTES:
            raise Invalid("Compute input exceeds 2 MB per artifact")
        inputs.append(item)
    input_manifest = [
        {
            "id": problem["id"],
            "digest": problem["digest"],
            "kind": problem["kind"],
            "channel": "problem",
        },
        *[
            {
                "id": r["id"],
                "digest": r["digest"],
                "kind": r["kind"],
                "channel": "context",
            }
            for r in evidence
        ],
    ]
    if predecessor:
        input_manifest.extend(
            [
                {
                    "id": predecessor["package_id"],
                    "digest": predecessor["package_digest"],
                    "kind": "compute_package",
                    "channel": "predecessor",
                },
                {
                    "id": predecessor["run_id"],
                    "digest": predecessor["run_digest"],
                    "kind": "compute_run",
                    "channel": "predecessor",
                },
            ]
        )
        input_manifest.extend(
            {**{key: record[key] for key in ("id", "kind", "digest")},
             "channel": "predecessor_diagnostic"}
            for record in predecessor["diagnostics"]
        )
    for item in inputs:
        r = item["record"]
        input_manifest.append(
            {
                "id": r["id"],
                "digest": r["digest"],
                "kind": r["kind"],
                "channel": "file",
                "filename": item["filename"],
                "basis": item["basis"],
                "content_sha256": hashlib.sha256(item["raw"]).hexdigest(),
                "bytes": len(item["raw"]),
            }
        )
        if "blob" in item:
            blob = item["blob"]
            input_manifest.append(
                {
                    "id": blob["id"],
                    "digest": blob["digest"],
                    "kind": "file_blob",
                    "channel": "binary_source",
                    "content_sha256": blob["data"]["sha256"],
                }
            )
    input_digest = digest(
        {
            "version": "compute-inputs-v2",
            "instruction": instruction,
            "inputs": input_manifest,
            "predecessor": predecessor,
            "prompt_digest": digest(prompt("simulation_engineer")),
        }
    )
    for previous in store.list("compute_run"):
        if (
            previous["parent"] == problem_id
            and not previous["stale"]
            and previous["data"].get("input_digest") == input_digest
        ):
            return collect_outputs(store, provider, problem_id, previous["id"])
    # Collection and identical retries above never require a new paid execution.
    # A saved design is not executable merely because a protocol record exists.
    protocols = [record for record in known if record["kind"] == "experiment_protocol"]
    if protocols:
        from symplex.evaluation.experiments import require_execution_ready
        require_execution_ready(protocols[-1])
    uploaded = []
    images = []
    try:
        for item in inputs:
            r, raw, filename = item["record"], item["raw"], item["filename"]
            f = provider.client.files.create(file=(filename, raw), purpose="user_data")
            uploaded.append(f.id)
            if Path(filename).suffix.lower() in (".png", ".jpg", ".jpeg"):
                images.append({"type": "input_image", "file_id": f.id, "detail": "low"})
            store.put(
                "remote_file",
                {
                    "source_id": r["id"],
                    "source_digest": r["digest"],
                    "blob_id": item.get("blob", {}).get("id"),
                    "content_sha256": hashlib.sha256(raw).hexdigest(),
                    "file_id": f.id,
                    "status": "uploaded_for_compute",
                },
                problem_id,
            )
        context = {
            "problem": problem["data"],
            "artifacts": evidence,
            "task": instruction,
            "uploaded_file_ids": uploaded,
            "input_manifest": input_manifest,
            "predecessor": predecessor,
            "revision_requirement": "Inspect the supplied predecessor code and results before making the requested revision. Preserve comparable baseline definitions and report exactly what changed."
            if predecessor
            else None,
            "execution_scope": "exploratory; independent validation is required",
        }
        with span(store, problem_id, "compute.hosted_python"):
            response = provider._call(
                "heavy",
                {
                    "instructions": prompt("simulation_engineer"),
                    "input": (
                        [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": canonical(context)}
                                ]
                                + images,
                            }
                        ]
                        if images
                        else canonical(context)
                    ),
                    "tools": [
                        {
                            "type": "code_interpreter",
                            "container": {
                                "type": "auto",
                                "memory_limit": "1g",
                                "file_ids": uploaded,
                            },
                        }
                    ],
                    "tool_choice": "required",
                    "max_tool_calls": 4,
                    "max_output_tokens": 6000,
                },
                problem_id,
                compute=True,
            )
        calls = [
            x.model_dump() for x in response.output if x.type == "code_interpreter_call"
        ]
        if not calls or not any(c.get("status") == "completed" for c in calls):
            raise Invalid("No completed Python tool execution was returned")
        run_id = store.put(
            "compute_run",
            {
                "status": "executed",
                "input_digest": input_digest,
                "response_id": response.id,
                "model": response.model,
                "calls": calls,
                "summary": response.output_text,
                "source_ids": list(dict.fromkeys(r["id"] for r in input_manifest)),
                "input_manifest": input_manifest,
                "predecessor_package_id": predecessor_package_id,
                "empirically_validated": False,
                "scope": "Agent-authored computation; diagnostics and result claims require independent review.",
            },
            problem_id,
        )
        return collect_outputs(store, provider, problem_id, run_id)
    finally:
        for file_id in uploaded:
            try:
                provider.client.files.delete(file_id)
            except Exception:
                store.put(
                    "cleanup_gap",
                    {"file_id": file_id, "resource": "uploaded_compute_input"},
                    problem_id,
                )


def save_blob(store, raw, filename, problem_id, basis, run_id=None):
    name = Path(filename).name
    extension = Path(name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise Invalid("Unsupported output format")
    sha = hashlib.sha256(raw).hexdigest()
    folder = store.root / "blobs"
    folder.mkdir(exist_ok=True)
    path = folder / sha
    if not path.exists():
        path.write_bytes(raw)
    return store.put(
        "file_blob",
        {
            "filename": name,
            "sha256": sha,
            "size": len(raw),
            "mime": ALLOWED_EXTENSIONS[extension],
            "basis": basis,
            "run_id": run_id,
        },
        problem_id,
    )


def read_blob(store, record):
    sha = record["data"]["sha256"]
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise Invalid("Invalid blob digest")
    raw = (store.root / "blobs" / sha).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise Invalid("Blob integrity failure")
    return raw


def collect_outputs(store, provider, problem_id, run_id):
    record = store.get(run_id)
    if record["kind"] != "compute_run" or record["parent"] != problem_id:
        raise Invalid("Choose a compute run for this problem")
    for package in store.list("compute_package"):
        if (
            package["data"].get("run_id") == run_id
            and not package["stale"]
            and not package["data"].get("file_failures")
        ):
            from symplex.evaluation.execution import assess_package
            assess_package(store, problem_id, package["id"])
            return package["id"]
    calls = record["data"]["calls"]
    containers = {c["container_id"] for c in calls if c.get("container_id")}
    files = []
    failures = []
    for container in containers:
        # Generated files are ephemeral. Preserve outputs now; never follow model-provided URLs.
        for file in provider.client.containers.files.list(container, limit=20):
            data = file.model_dump()
            name = Path(data.get("path", "output")).name
            if (
                data.get("source") == "user"
                or Path(name).suffix.lower() not in ALLOWED_EXTENSIONS
            ):
                continue
            if (data.get("bytes") or 0) > 5000000:
                failures.append({"filename": name, "reason": "Output exceeds 5 MB"})
                continue
            try:
                body = bytearray()
                with (
                    provider.client.containers.files.content.with_streaming_response.retrieve(
                        file.id, container_id=container
                    ) as stream
                ):
                    for chunk in stream.iter_bytes():
                        body.extend(chunk)
                        if len(body) > 5000000:
                            raise Invalid("Output exceeds 5 MB")
                bid = save_blob(
                    store, bytes(body), name, problem_id, "generated", run_id
                )
                files.append(bid)
            except Exception as exc:
                failures.append({"filename": name, "error_type": type(exc).__name__})
            if len(files) >= 12:
                break
    package = store.put(
        "compute_package",
        {
            "title": "Model, inference and visual results",
            "run_id": run_id,
            "file_ids": files,
            "file_failures": failures,
            "summary": record["data"]["summary"],
            "execution_status": "executed",
            "empirically_validated": False,
            "scope": "Executed Python with recorded code and outputs. Scientific validity and independent evaluation remain separate.",
        },
        problem_id,
    )
    from symplex.evaluation.execution import assess_package
    assess_package(store, problem_id, package)
    return package
