"""Audit configured use-case coverage and existing artifacts without calling models.

This checks configuration, source-code interfaces and recorded artifact integrity.
It does not run the solver or validate a scientific model. Optional --link arguments
associate existing demonstrations explicitly; matching a topic is not proof that the
complete configured brief was executed.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from symplex.agents.context import review_context  # noqa: E402
from symplex.agents.tools import REGISTRY  # noqa: E402
from symplex.connectors.compute import read_blob  # noqa: E402
from symplex.connectors.registry import CATALOG, HANDLERS  # noqa: E402
from symplex.core.contracts import digest  # noqa: E402
from symplex.infrastructure.storage import Store  # noqa: E402


REQUIRED_FIELDS = {
    "id",
    "title",
    "problem",
    "decision_or_estimation_target",
    "representations",
    "required_inputs",
    "compute_requirements",
    "validation_requirements",
    "visualizations",
    "current_scope",
    "dependency_gaps",
    "resources",
}
STAGES = {
    "framing": {"tools": ["inspect_context"], "artifacts": ["problem_dna"]},
    "system_representation": {
        "tools": ["represent_system", "review_hypotheses"],
        "artifacts": ["complex_system", "hypothesis_review"],
    },
    "evidence_synthesis": {
        "tools": ["research_evidence", "synthesize_evidence"],
        "artifacts": ["evidence_note", "evidence_synthesis"],
    },
    "frozen_experiment": {
        "tools": ["plan_experiment"],
        "artifacts": ["experiment_protocol"],
    },
    "computation": {
        "tools": ["run_model_code", "run_system_scenarios"],
        "artifacts": ["compute_run", "scenario_run"],
    },
    "comparison": {
        "tools": ["compare_computation"],
        "artifacts": ["experiment_comparison", "scenario_report"],
    },
    "delivery": {
        "tools": ["review_model", "build_outcome"],
        "artifacts": ["decision_brief", "deliverable"],
    },
}


def reference(record):
    return {key: record[key] for key in ("id", "kind", "digest")}


def recorded_probe(store, records):
    probes = [
        r
        for r in records
        if r["kind"] == "file_blob"
        and not r["stale"]
        and r["data"].get("filename") == "capabilities.json"
    ]
    if not probes:
        return {"status": "not_recorded", "packages": {}}
    latest = probes[-1]
    data = json.loads(read_blob(store, latest))
    return {
        "status": "recorded_discoverability_probe",
        "artifact": reference(latest),
        "scope": data.get("scope"),
        "method": data.get("method"),
        "packages": data.get("packages", {}),
        "verified_by_this_audit": "Stored JSON content digest only; no package imports or workloads executed",
        "limitation": data.get("caveat"),
    }


def inspect_demonstration(store, records, problem_id):
    problem = store.get(problem_id)
    if problem["kind"] != "workspace_problem" or problem["stale"]:
        raise ValueError("Linked demonstration must be a current workspace problem")
    scoped = [r for r in records if r["parent"] == problem_id and not r["stale"]]
    files, executions = [], []
    for package in [r for r in scoped if r["kind"] == "compute_package"]:
        run = store.get(package["data"]["run_id"])
        if run["kind"] != "compute_run" or run["parent"] != problem_id or run["stale"]:
            raise ValueError("Invalid compute package/run binding")
        executions.append(
            {
                "package": reference(package),
                "run": reference(run),
                "completed_tool_calls_recorded": sum(
                    c.get("status") == "completed" for c in run["data"].get("calls", [])
                ),
                "predecessor_package_id": run["data"].get("predecessor_package_id"),
                "scope": "Recorded hosted execution, not independently reproduced",
            }
        )
        for ident in package["data"].get("file_ids", []):
            blob = store.get(ident)
            if (
                blob["kind"] != "file_blob"
                or blob["parent"] != problem_id
                or blob["stale"]
                or blob["data"].get("run_id") != run["id"]
            ):
                raise ValueError("Invalid generated file provenance")
            raw = read_blob(store, blob)
            files.append(
                {
                    **reference(blob),
                    "filename": blob["data"]["filename"],
                    "bytes_verified": len(raw),
                    "sha256": blob["data"]["sha256"],
                    "package_id": package["id"],
                    "basis": blob["data"].get("basis"),
                    "verified": "Stored content checksum and same-problem run binding",
                }
            )
    try:
        reviewer = review_context(store, problem_id)
        reviewer_ids = {r["id"] for r in reviewer["executed_files"]}
        inventory = reviewer.get("file_inventory", [])
        inventory_ids = {r["id"] for r in inventory}
        latest_package = executions[-1]["package"]["id"] if executions else None
        reviewer_inspection = {
            "supplied_file_ids": sorted(reviewer_ids),
            "supplied_artifact_ids": [r["id"] for r in reviewer["artifacts"]],
            "file_inventory": inventory,
            "source_code_ids_not_supplied": [
                f["id"]
                for f in files
                if f["package_id"] == latest_package
                and f["filename"].endswith(".py")
                and f["id"] not in reviewer_ids
            ],
            "latest_file_ids_not_in_inventory": [
                f["id"]
                for f in files
                if f["package_id"] == latest_package and f["id"] not in inventory_ids
            ],
            "truncated_file_ids": [
                r["id"] for r in reviewer["executed_files"] if r["truncated"]
            ],
            "scope": "Current reviewer context assembly, not evidence of what an earlier model saw",
        }
    except (KeyError, ValueError) as exc:
        reviewer_inspection = {
            "status": "invalid_context",
            "error_type": type(exc).__name__,
        }
    return {
        "problem": reference(problem),
        "question": problem["data"]["question"],
        "association": "Explicit audit link to related demonstration; not a full use-case benchmark",
        "artifact_ids_by_kind": {
            kind: [r["id"] for r in scoped if r["kind"] == kind]
            for kind in {r["kind"] for r in scoped}
            if kind not in {"model_call", "message", "model_failure"}
        },
        "recorded_executions": executions,
        "verified_file_records": files,
        "reviewer_context": reviewer_inspection,
        "empirical_validation": "Not established by this audit; no independent domain observations or model reruns evaluated",
    }


def evaluate(store, briefs, links):
    if not isinstance(briefs, list) or len({b["id"] for b in briefs}) != len(briefs):
        raise ValueError("Use-case configuration must have unique IDs")
    for brief in briefs:
        if set(brief) != REQUIRED_FIELDS:
            raise ValueError(
                "Use-case fields do not match the declared launch-brief schema"
            )
        for name in REQUIRED_FIELDS - {
            "id",
            "title",
            "problem",
            "decision_or_estimation_target",
            "current_scope",
        }:
            if not isinstance(brief[name], list) or not brief[name]:
                raise ValueError("Use-case list is missing: " + name)
    if not set(links).issubset({b["id"] for b in briefs}):
        raise ValueError("An audit link names an unknown configured use case")
    records = store.list()
    tools = REGISTRY.descriptions()
    probe = recorded_probe(store, records)
    extensions = json.loads((ROOT / "symplex/modeling/extensions.json").read_text())
    results = []
    for brief in briefs:
        demonstration = (
            inspect_demonstration(store, records, links[brief["id"]])
            if brief["id"] in links
            else None
        )
        artifacts = demonstration["artifact_ids_by_kind"] if demonstration else {}
        milestones = {}
        for stage, definition in STAGES.items():
            ids = [
                ident
                for kind in definition["artifacts"]
                for ident in artifacts.get(kind, [])
            ]
            milestones[stage] = {
                "configured_interface_tools": {
                    tool: tool in tools for tool in definition["tools"]
                },
                "recorded_related_artifact_ids": ids,
                "status": "recorded_related_demonstration"
                if ids
                else "configured_not_demonstrated",
                "scientific_validity": "not_established",
            }
        generated = demonstration["verified_file_records"] if demonstration else []
        milestones["visualization"] = {
            "specified": brief["visualizations"],
            "recorded_visual_file_ids": [
                f["id"]
                for f in generated
                if f["filename"].lower().endswith((".png", ".jpg", ".geojson"))
            ],
            "status": "recorded_related_artifacts"
            if any(
                f["filename"].lower().endswith((".png", ".jpg", ".geojson"))
                for f in generated
            )
            else "specification_only_for_this_case",
            "verified": "File integrity where referenced; visual correctness and full scientific meaning not evaluated",
        }
        mentioned = " ".join(
            brief["compute_requirements"] + brief["dependency_gaps"]
        ).lower()
        package_mapping = {
            name: {
                "recorded_probe_available": entry.get("available"),
                "recorded_version": entry.get("version"),
                "scope": "Name mentioned in brief and present in saved capability probe; not a validated adapter",
            }
            for name, entry in probe["packages"].items()
            if name.lower() in mentioned
        }
        extension_mapping = []
        for extension in extensions:
            names = [
                part.strip().split()[0]
                for part in extension["name"].replace("+", "/").split("/")
            ]
            if any(name.lower() in mentioned for name in names):
                extension_mapping.append(
                    {
                        "name": extension["name"],
                        "declared_status": extension["status"],
                        "required_validation": extension["required_validation"],
                        "scope": "Named extension specification, not evidence of an installed adapter",
                    }
                )
        results.append(
            {
                "id": brief["id"],
                "title": brief["title"],
                "brief_digest": digest(brief),
                "configuration": "required_fields_verified",
                "current_scope": brief["current_scope"],
                "milestones": milestones,
                "mentioned_runtime_packages": package_mapping,
                "mentioned_extension_specs": extension_mapping,
                "dependency_gaps": brief["dependency_gaps"],
                "related_demonstration": demonstration,
                "usefulness_assessment": "Available as an agent investigation brief. Actual domain usefulness requires the named evidence, valid model interfaces and independent outcome assessment; field coverage is not a scientific success score.",
            }
        )
    return {
        "schema": "symplex-use-case-audit-v1",
        "mode": "deterministic_read_only_coverage_audit",
        "model_calls": 0,
        "new_scientific_simulations": 0,
        "verified": "Configuration fields, registered tool names, artifact hashes and file/run bindings only",
        "not_verified": [
            "Full use-case execution",
            "Scientific validity",
            "Calibration",
            "Real-world decision improvement",
            "General recursive self-improvement",
            "Fresh hosted library imports",
        ],
        "configured_case_count": len(briefs),
        "linked_related_demonstration_count": len(links),
        "registry": [
            {
                "id": c.id,
                "declared_status": c.status,
                "handler_registered": c.id in HANDLERS,
                "package": c.package,
            }
            for c in CATALOG
        ],
        "recorded_runtime_probe": probe,
        "use_cases": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT / ".symplex")
    parser.add_argument(
        "--briefs", type=Path, default=ROOT / "symplex/domains/use_cases.json"
    )
    parser.add_argument(
        "--link", action="append", default=[], metavar="USE_CASE_ID=PROBLEM_ID"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/private/tmp/symplex-usecase-audit.json")
    )
    args = parser.parse_args()
    if not (args.workspace / "metadata.sqlite3").is_file():
        parser.error(
            "Workspace must already exist; this audit does not create investigations"
        )
    links = {}
    for item in args.link:
        key, separator, value = item.partition("=")
        if not separator or not key or not value or key in links:
            parser.error("Each --link must be a unique USE_CASE_ID=PROBLEM_ID")
        links[key] = value
    report = evaluate(Store(args.workspace), json.loads(args.briefs.read_text()), links)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "mode": report["mode"],
                "configured_cases": report["configured_case_count"],
                "related_demonstrations": len(links),
                "model_calls": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
