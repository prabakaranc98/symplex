"""Deterministic, MAP-Elites-inspired exploration of evaluated compute packages.

The only score is the pass fraction of frozen host numerical checks. It is a
software/numerical diagnostic proxy, never model truth, empirical utility, or
permission to promote a decision. Protocols are deliberately incomparable.
"""

from collections import defaultdict

from symplex.connectors.compute import read_blob
from symplex.core.contracts import Invalid, digest
from symplex.evaluation.experiments import EVALUATOR_VERSION
from symplex.evaluation.numerics import NumericCheck


VERSION = "protocol-scoped-diagnostic-archive-v1"
BASES = {"synthetic", "conditional_model", "observational_data"}


def _current(store, ident, kind, problem_id):
    if not isinstance(ident, str):
        raise Invalid("missing_" + kind)
    try:
        record = store.get(ident)
    except KeyError:
        raise Invalid("missing_" + kind) from None
    if record["kind"] != kind or record["parent"] != problem_id or record["stale"]:
        raise Invalid("not_current_scoped_" + kind)
    return record


def _manifest_has(run, record):
    return any(
        item.get("id") == record["id"] and item.get("digest") == record["digest"]
        for item in run["data"].get("input_manifest", [])
        if isinstance(item, dict)
    )


def _completed_run(store, ident, problem):
    run = _current(store, ident, "compute_run", problem["id"])
    if not any(c.get("status") == "completed" for c in run["data"].get("calls", [])):
        raise Invalid("run_has_no_completed_execution")
    if not _manifest_has(run, problem):
        raise Invalid("run_not_bound_to_current_problem_digest")
    return run


def _entry(store, problem, package, comparison):
    problem_id = problem["id"]
    if package["stale"]:
        raise Invalid("stale_package")
    run = _completed_run(store, package["data"].get("run_id"), problem)
    data = comparison["data"]
    if data.get("evaluator_version") != EVALUATOR_VERSION:
        raise Invalid("evaluator_version_mismatch")
    if data.get("package_id") != package["id"] or data.get("run_id") != run["id"]:
        raise Invalid("comparison_run_or_package_mismatch")
    protocol = _current(
        store, data.get("protocol_id"), "experiment_protocol", problem_id
    )
    if data.get("protocol_digest") != protocol["digest"] or not _manifest_has(
        run, protocol
    ):
        raise Invalid("protocol_not_bound_to_execution")
    if (
        protocol["data"].get("status") != "frozen_for_comparison"
        or protocol["created"] >= run["created"]
    ):
        raise Invalid("protocol_was_not_frozen_before_execution")
    system = _current(
        store, protocol["data"].get("system_id"), "complex_system", problem_id
    )
    if (
        protocol["data"].get("system_digest") != system["digest"]
        or data.get("system_id") != system["id"]
        or not _manifest_has(run, system)
    ):
        raise Invalid("system_not_bound_to_execution")
    verification = _current(
        store,
        data.get("numerical_verification_id"),
        "numerical_verification",
        problem_id,
    )
    check_data = verification["data"]
    if any(
        check_data.get(key) != expected
        for key, expected in {
            "evaluator_version": EVALUATOR_VERSION,
            "package_id": package["id"],
            "run_id": run["id"],
            "run_digest": run["digest"],
            "protocol_id": protocol["id"],
            "protocol_digest": protocol["digest"],
        }.items()
    ):
        raise Invalid("numerical_verification_binding_mismatch")
    numerical = {
        k: v
        for k, v in check_data.items()
        if k
        not in {"evaluator_version", "protocol_id", "protocol_digest", "package_id"}
    }
    if digest(numerical) != data.get("numerical_result_digest"):
        raise Invalid("numerical_result_digest_mismatch")
    checks = check_data.get("checks", [])
    frozen = [
        NumericCheck.model_validate(c).model_dump()
        for c in protocol["data"].get("numeric_checks", [])
    ]
    if (
        not frozen
        or not checks
        or check_data.get("status") not in {"checked", "failed"}
    ):
        raise Invalid("missing_host_numerical_fitness")
    if check_data.get("checks_digest") != digest(frozen):
        raise Invalid("frozen_check_suite_mismatch")
    expected = {c["id"]: c for c in frozen}
    if len(checks) != len(expected) or {c.get("id") for c in checks} != set(expected):
        raise Invalid("incomplete_host_check_coverage")
    if any(
        type(c.get("passed")) is not bool
        or c.get("check_digest") != digest(expected[c["id"]])
        for c in checks
    ):
        raise Invalid("invalid_host_check_result")
    if check_data.get("all_passed") is not all(c["passed"] for c in checks):
        raise Invalid("inconsistent_host_check_summary")
    package_files = set(package["data"].get("file_ids", []))
    output = _current(store, data.get("output_id"), "file_blob", problem_id)
    if output["id"] not in package_files or output["digest"] != data.get(
        "output_digest"
    ):
        raise Invalid("comparison_output_mismatch")
    sources = check_data.get("source_manifest", [])
    if not sources:
        raise Invalid("missing_host_checked_sources")
    for source in sources:
        blob = _current(store, source.get("id"), "file_blob", problem_id)
        if (
            blob["id"] not in package_files
            or source.get("digest") != blob["digest"]
            or source.get("sha256") != blob["data"].get("sha256")
            or blob["data"].get("run_id") != run["id"]
            or blob["data"].get("basis") != "generated"
        ):
            raise Invalid("checked_source_binding_mismatch")
        read_blob(store, blob)
    if (
        output["data"].get("run_id") != run["id"]
        or output["data"].get("basis") != "generated"
    ):
        raise Invalid("comparison_output_not_generated_by_run")
    read_blob(store, output)
    model_kinds = sorted({c["kind"] for c in system["data"]["components"]})
    if not model_kinds:
        raise Invalid("missing_declared_component_kinds")
    basis = data.get("basis")
    if basis not in BASES:
        raise Invalid("unknown_evidence_basis")
    passed = [c["id"] for c in checks if c["passed"]]
    return {
        "package_id": package["id"],
        "run_id": run["id"],
        "system_id": system["id"],
        "system_digest": system["digest"],
        "protocol_id": protocol["id"],
        "protocol_digest": protocol["digest"],
        "protocol_created": protocol["created"],
        "created": package["created"],
        "comparison_id": comparison["id"],
        "numerical_verification_id": verification["id"],
        "evaluator_version": EVALUATOR_VERSION,
        "declared_component_kinds": model_kinds,
        "evidence_basis": basis,
        "basis_authority": "maker_declared_unverified",
        "passed_check_ids": passed,
        "failed_check_ids": [c["id"] for c in checks if not c["passed"]],
        "diagnostic_pass_count": len(passed),
        "diagnostic_check_count": len(checks),
        "diagnostic_pass_fraction": len(passed) / len(checks),
        "alternative_ids": [
            protocol["data"]["baseline_id"],
            *protocol["data"]["candidate_ids"],
        ],
        "predecessor_package_id": run["data"].get("predecessor_package_id"),
    }


def build_search_archive(store, problem_id, *, protocol_digest=None):
    """Build the full archive; an optional protocol restricts parent selection only."""
    problem = store.get(problem_id)
    if problem["kind"] != "workspace_problem" or problem["stale"]:
        raise Invalid("Select a current problem")
    packages = [r for r in store.list("compute_package") if r["parent"] == problem_id]
    comparisons = [
        r
        for r in store.list("experiment_comparison")
        if r["parent"] == problem_id and not r["stale"]
    ]
    by_package = defaultdict(list)
    for record in comparisons:
        by_package[record["data"].get("package_id")].append(record)
    visits = defaultdict(set)
    ignored_visits = []
    for record in store.list("compute_run"):
        if (
            record["parent"] != problem_id
            or record["stale"]
            or not record["data"].get("predecessor_package_id")
        ):
            continue
        try:
            run = _completed_run(store, record["id"], problem)
            parent = _current(
                store,
                run["data"]["predecessor_package_id"],
                "compute_package",
                problem_id,
            )
            if not _manifest_has(run, parent):
                raise Invalid("predecessor_not_bound_in_child_input_manifest")
            visits[parent["id"]].add(run["id"])
        except (Invalid, KeyError, TypeError, ValueError) as exc:
            ignored_visits.append({"run_id": record["id"], "reason": str(exc)[:300]})
    cells, excluded = {}, []
    for package in packages:
        candidates = sorted(
            by_package[package["id"]],
            key=lambda r: (r["created"], r["id"]),
            reverse=True,
        )
        candidates = [
            r
            for r in candidates
            if r["data"].get("evaluator_version") == EVALUATOR_VERSION
        ]
        failures, entry = [], None
        for comparison in candidates:
            try:
                entry = _entry(store, problem, package, comparison)
                break
            except (Invalid, KeyError, TypeError, ValueError) as exc:
                failures.append(str(exc)[:300])
        if entry is None:
            excluded.append(
                {
                    "package_id": package["id"],
                    "run_id": package["data"].get("run_id"),
                    "reasons": failures or ["missing_current_host_evaluation"],
                    "fitness": None,
                }
            )
            continue
        entry["parent_visit_count"] = len(visits[package["id"]])
        descriptor = {
            "declared_component_kinds": entry["declared_component_kinds"],
            "evidence_basis": entry["evidence_basis"],
            "protocol_digest": entry["protocol_digest"],
        }
        cell_id = "niche_" + digest(descriptor)[:20]
        cell = cells.setdefault(
            cell_id, {"id": cell_id, "descriptor": descriptor, "entries": []}
        )
        cell["entries"].append(entry)
    for cell in cells.values():
        entries = sorted(
            cell["entries"],
            key=lambda e: (
                -e["diagnostic_pass_fraction"],
                e["parent_visit_count"],
                e["created"],
                e["package_id"],
            ),
        )
        cell["entries"] = entries
        best = entries[0]
        cell.update(
            {
                "elite_package_id": best["package_id"],
                "diagnostic_pass_fraction": best["diagnostic_pass_fraction"],
                "tie_package_ids": sorted(
                    e["package_id"]
                    for e in entries
                    if e["diagnostic_pass_fraction"] == best["diagnostic_pass_fraction"]
                ),
                "visit_count": sum(e["parent_visit_count"] for e in entries),
                "protocol_created": best["protocol_created"],
            }
        )
    grouped = defaultdict(list)
    for cell in cells.values():
        grouped[cell["descriptor"]["protocol_digest"]].append(cell)
    selected = None
    eligible_groups = {
        key: cells
        for key, cells in grouped.items()
        if protocol_digest is None or key == protocol_digest
    }
    if eligible_groups:
        # Protocol-level scheduling never compares scores from different check suites.
        selected_digest = min(
            eligible_groups,
            key=lambda key: (
                min(c["visit_count"] for c in eligible_groups[key]),
                sum(c["visit_count"] for c in eligible_groups[key]),
                min(c["protocol_created"] for c in eligible_groups[key]),
                key,
            ),
        )
        selected = min(
            eligible_groups[selected_digest],
            key=lambda c: (c["visit_count"], -c["diagnostic_pass_fraction"], c["id"]),
        )
    return {
        "version": VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "problem_id": problem_id,
        "cells": sorted(cells.values(), key=lambda c: c["id"]),
        "excluded_candidates": sorted(excluded, key=lambda e: e["package_id"]),
        "ignored_visit_records": ignored_visits,
        "suggested_parent_id": selected["elite_package_id"] if selected else None,
        "selection_protocol_digest": protocol_digest,
        "selection_scope": "specified_frozen_protocol" if protocol_digest is not None else "all_frozen_protocols",
        "selected_protocol_digest": selected["descriptor"]["protocol_digest"]
        if selected
        else None,
        "selection_rule": (
            "Within the specified frozen protocol, select the least-visited niche; "
            if protocol_digest is not None
            else "Least-visited protocol group, then least-visited niche; "
        ) + "diagnostic quality ranks only within an identical frozen protocol. Ties use parent visits, creation time and artifact ID.",
        "quality_definition": "Fraction of frozen host numerical checks passed; software/numerical proxy only. Maker metrics and scientific claims are never fitness.",
        "promotion_allowed": False,
        "scope": "MAP-Elites-inspired archive of declared model kinds and unverified evidence basis. Retains every evaluated package and decision alternative; it does not establish empirical model quality, calibrated inference or recursive self-improvement.",
    }
