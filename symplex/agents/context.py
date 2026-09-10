"""Scoped working memory assembled from artifacts, excluding protected evaluation data."""

from symplex.core.contracts import canonical
from symplex.agents.input_state import input_authority

# Highest-value current state first. Source documents remain data with provenance.
KINDS = {
    "steering": 4,
    "artifact_inspection": 1,
    "problem_dna": 1,
    "experiment_protocol": 1,
    "experiment_comparison": 1,
    "execution_assessment": 1,
    "numerical_verification": 1,
    "candidate_design": 2,
    "evidence_synthesis": 1,
    "decision_brief": 1,
    "hypothesis_review": 1,
    "complex_system": 1,
    "solution": 1,
    "model_critique": 1,
    "compute_package": 1,
    "scenario_report": 1,
    "evidence_note": 2,
    "context": 3,
    "binary_context": 3,
}


def working_context(store, problem_id, max_chars=26000, *, excluded_kinds=()):
    selected = []
    remaining = max_chars
    for kind, limit in KINDS.items():
        if kind in excluded_kinds:
            continue
        records = [
            r for r in store.list(kind) if r["parent"] == problem_id and not r["stale"]
        ][-limit:]
        for r in records:
            data = dict(r["data"])
            if kind == "experiment_protocol":
                data.pop("result_contract", None)
                data["full_output_contract_artifact_id"] = r["id"]
            if kind == "context":
                data.pop("inspection", None)
                data["content"] = data["content"][:4000]
            if kind == "scenario_report":
                data["scenario_results"] = {
                    k: {a: b for a, b in v.items() if a != "mean_trajectory"}
                    for k, v in data["scenario_results"].items()
                }
            text = canonical(data)
            entry_limit = 18000 if kind == "experiment_protocol" else 12000 if kind == "artifact_inspection" else 6500
            if len(text) > entry_limit:
                data = {
                    "excerpt": text[: entry_limit - 500],
                    "truncated": True,
                    "full_artifact_id": r["id"],
                }
            entry = {
                "id": r["id"],
                "kind": kind,
                "digest": r["digest"],
                "data": data,
                "authority": input_authority(kind),
            }
            size = len(canonical(entry))
            if size > remaining:
                continue
            selected.append(entry)
            remaining -= size
    return selected



def review_context(store, problem_id, max_chars=32000):
    """Prioritize actual source and diagnostics; inventory uninspected outputs explicitly."""
    from pathlib import Path
    from symplex.connectors.compute import read_blob

    packages = [r for r in store.list("compute_package")
                if r["parent"] == problem_id and not r["stale"]][-1:]
    files, inventory, remaining = [], [], max_chars // 2
    for package in packages:
        run = store.get(package["data"]["run_id"])
        if run["kind"] != "compute_run" or run["parent"] != problem_id or run["stale"]:
            continue
        items = [store.get(i) for i in package["data"].get("file_ids", [])]
        items = [r for r in items if r["kind"] == "file_blob" and not r["stale"] and
                 r["parent"] == problem_id and r["data"].get("run_id") == run["id"] and r["data"].get("basis") == "generated"]
        def priority(item):
            name = item["data"]["filename"]
            return (0 if name.endswith(".py") else 1 if name == "symplex_experiment.json"
                    else 2 if name.endswith(".json") else 3, name)
        items.sort(key=priority)
        inventory.extend({"id": r["id"], "digest": r["digest"],
                          "filename": r["data"]["filename"], "bytes": r["data"].get("size"),
                          "run_id": run["id"], "content_status": "Available; executed_files identifies inspected excerpts"} for r in items)
        for item in items:
            if Path(item["data"]["filename"]).suffix not in (".py", ".json", ".csv"):
                continue
            raw = read_blob(store, item).decode("utf-8", errors="replace")
            limit = min(7000, remaining)
            if limit <= 0:
                break
            files.append({"id": item["id"], "digest": item["digest"],
                          "filename": item["data"]["filename"], "content": raw[:limit],
                          "truncated": len(raw) > limit,
                          "basis": "executed maker output, not independent evidence"})
            remaining -= len(raw[:limit])
    return {"artifacts": working_context(store, problem_id, max_chars // 2),
            "executed_files": files, "file_inventory": inventory,
            "scope": "Bounded excerpts; checker did not rerun these programs. Inventory distinguishes available-but-uninspected files from missing files."}
