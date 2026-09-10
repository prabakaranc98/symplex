"""Persistent search state derived from immutable proposals and measured artifacts."""

from symplex.core.contracts import Invalid

CANDIDATE_KINDS = {"complex_system", "solution", "candidate_design", "compute_package", "method_candidate"}
EVALUATION_KINDS = {"hypothesis_review", "model_critique", "experiment_comparison", "review"}
EVIDENCE_KINDS = {"context", "binary_context", "evidence_note", "evidence_synthesis"}


def investigation_state(store, problem_id, budget):
    problem = store.get(problem_id)
    if problem["kind"] != "workspace_problem" or problem["stale"]:
        raise Invalid("Select a current problem")
    records = [r for r in store.list() if r["parent"] == problem_id and not r["stale"]]
    candidates = []
    by_id = {r["id"]: r for r in records}
    for r in records:
        if r["kind"] not in CANDIDATE_KINDS:
            continue
        data = r["data"]
        candidates.append({
            "id": r["id"], "kind": r["kind"], "digest": r["digest"],
            "title": data.get("name", data.get("title", r["kind"])),
            "parent_id": data.get("parent_system_id") or data.get("parent_candidate_id") or data.get("parent_solution_id") or data.get("parent_method_id") or data.get("predecessor_package_id") or by_id.get(data.get("run_id"), {}).get("data", {}).get("predecessor_package_id"),
            "parent_method": data.get("parent_method"),
            "status": data.get("status", data.get("execution_status", "proposed")),
            "evaluation_ids": [e["id"] for e in records if e["kind"] in EVALUATION_KINDS and r["id"] in {e["data"].get("system_id"), e["data"].get("package_id")}],
        })
    return {
        "problem_id": problem_id, "problem_digest": problem["digest"],
        "candidates": candidates,
        "evidence_ids": [r["id"] for r in records if r["kind"] in EVIDENCE_KINDS],
        "evaluation_ids": [r["id"] for r in records if r["kind"] in EVALUATION_KINDS],
        "protocol_ids": [r["id"] for r in records if r["kind"] == "experiment_protocol"],
        "steering_ids": [r["id"] for r in records if r["kind"] == "steering"],
        "budget": budget.snapshot(),
        "scope": "Candidate archive, not a fitness ranking. Earlier hypotheses remain inspectable; unsupported candidates are not promoted.",
    }
