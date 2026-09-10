"""Mandatory, deterministic post-execution assessment; never a science certificate."""

from symplex.agents.scientific_cycle import current_artifact
from symplex.core.contracts import Invalid, digest

VERSION = "execution-assessment-v3"


def assess_package(store, problem_id, package_id):
    from symplex.evaluation.model_traceability import check_model_map

    package = current_artifact(store, problem_id, "compute_package", package_id)
    run = current_artifact(store, problem_id, "compute_run", package["data"]["run_id"])
    protocol_ids = [
        r["id"]
        for r in run["data"].get("input_manifest", [])
        if r.get("kind") == "experiment_protocol"
    ]
    comparison_id, verification_id, error = None, None, None
    numerical = {"status": "unavailable", "all_passed": False, "check_count": 0}
    status = "unavailable"
    if not any(c.get("status") == "completed" for c in run["data"].get("calls", [])):
        status, error = "failed", "No completed sandbox execution is recorded"
    elif protocol_ids:
        from symplex.evaluation.experiments import compare_computation

        try:
            # Revalidate before cache lookup; earlier checks do not license stale
            # source records or altered CSV bytes on subsequent package access.
            comparison_id = compare_computation(store, problem_id, package_id)
            comparison = store.get(comparison_id)["data"]
            verification_id = comparison.get("numerical_verification_id")
            numerical = comparison.get("numerical_verification", numerical)
            if numerical.get("status") == "unavailable":
                status = "unavailable"
            elif (
                numerical.get("status") == "checked"
                and numerical.get("all_passed") is True
            ):
                status = "checked"
            else:
                status = "failed"
        except (Invalid, ValueError, KeyError, OSError) as exc:
            status, error = "failed", str(exc)[:1500]
    next_requirement = (
        "Inspect the comparison, numerical verification and independent evidence gaps."
        if status == "checked"
        else "Supply and execute a new frozen experiment protocol; preserve this unverified run."
        if status == "unavailable"
        else "Repair the recorded output or model against the unchanged protocol, then rerun."
    )
    data = {
        "evaluator_version": VERSION,
        "package_id": package_id,
        "package_digest": package["digest"],
        "run_id": run["id"],
        "run_digest": run["digest"],
        "protocol_ids": protocol_ids,
        "comparison_id": comparison_id,
        "numerical_verification_id": verification_id,
        "numerical_verification": numerical,
        "model_traceability": check_model_map(store, problem_id, package_id),
        "status": status,
        "error": error,
        "independently_validated": False,
        "scope": "Host output-contract and numerical checks. Scientific accuracy requires relevant external evidence and domain validation.",
        "next_requirement": next_requirement,
    }
    expected_digest = digest(data)
    existing = [
        r
        for r in store.list("execution_assessment")
        if r["parent"] == problem_id
        and not r["stale"]
        and r["digest"] == expected_digest
    ]
    if existing:
        return existing[-1]["id"]
    return store.put("execution_assessment", data, problem_id)
