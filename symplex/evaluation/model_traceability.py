"""Check links from proposed mechanisms to source symbols and recorded columns.

This is structural traceability, not proof that code implements an equation.
Agent-authored Python is parsed as bounded text and is never executed here.
"""

import ast
import csv
import io
import json
from typing import Literal

from pydantic import Field, model_validator

from symplex.agents.scientific_cycle import current_artifact
from symplex.connectors.compute import read_blob
from symplex.core.contracts import Invalid
from symplex.modeling.complex_system import ClosedContract, Identifier, Text


class ComponentBinding(ClosedContract):
    component_id: Identifier
    status: Literal["implemented", "omitted"]
    filename: str | None
    symbol: str | None
    mathematical_description: Text
    limitation: Text

    @model_validator(mode="after")
    def implementation_reference(self):
        if self.status == "implemented" and (not self.filename or not self.symbol):
            raise ValueError("Implemented components require a source file and symbol")
        if self.status == "omitted" and (
            self.filename is not None or self.symbol is not None
        ):
            raise ValueError("Omitted components cannot claim a source binding")
        return self


class StateBinding(ClosedContract):
    state_id: Identifier
    filename: str
    column: str
    unit_id: Identifier
    interpretation: Text


class ModelMap(ClosedContract):
    system_id: str
    components: list[ComponentBinding] = Field(min_length=1, max_length=40)
    outputs: list[StateBinding] = Field(max_length=100)
    scope: Text


def _symbols(raw):
    if len(raw) > 200000:
        raise Invalid("Traceability source exceeds 200 KB")
    tree = ast.parse(raw.decode("utf-8"))
    symbols = {}

    def visit(body, prefix=""):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + node.name
                symbols[name] = {"line": node.lineno, "end_line": node.end_lineno}
                visit(node.body, name + ".")

    visit(tree.body)
    return symbols


def check_model_map(store, problem_id, package_id):
    """Revalidate every linked file before returning even an earlier check result."""
    package = current_artifact(store, problem_id, "compute_package", package_id)
    run = current_artifact(store, problem_id, "compute_run", package["data"]["run_id"])
    scope = "Source-symbol and CSV-column linkage only. Equations, mechanism fidelity, causal validity and scientific accuracy are not verified by these checks."
    unavailable = {
        "status": "unavailable",
        "scope": scope,
        "independently_validated": False,
    }
    try:
        files = {}
        for ident in package["data"].get("file_ids", []):
            blob = current_artifact(store, problem_id, "file_blob", ident)
            if (
                blob["data"].get("run_id") != run["id"]
                or blob["data"].get("basis") != "generated"
            ):
                raise Invalid("Model map files must belong to this generated run")
            name = blob["data"]["filename"].rsplit("/", 1)[-1]
            if name in files:
                raise Invalid("Ambiguous duplicate output filename")
            files[name] = blob
        if "symplex_model_map.json" not in files:
            return {
                **unavailable,
                "reason": "No agent-authored semantic-to-code map was returned",
            }
        if not any(
            c.get("status") == "completed" for c in run["data"].get("calls", [])
        ):
            raise Invalid("Model map requires completed sandbox execution")
        raw = read_blob(store, files["symplex_model_map.json"])
        if len(raw) > 100000:
            raise Invalid("Model map exceeds 100 KB")
        model_map = ModelMap.model_validate(json.loads(raw))
        system = current_artifact(
            store, problem_id, "complex_system", model_map.system_id
        )
        if not any(
            r.get("id") == system["id"] and r.get("digest") == system["digest"]
            for r in run["data"].get("input_manifest", [])
        ):
            raise Invalid(
                "Model map system must match the representation supplied to this run"
            )
        expected = {c["id"] for c in system["data"]["components"]}
        actual = [c.component_id for c in model_map.components]
        if len(actual) != len(expected) or set(actual) != expected:
            raise Invalid(
                "Model map must account for every component exactly once, including omissions"
            )
        states = {s["id"]: s for s in system["data"]["states"]}
        manifest, bindings = {}, []

        def source(name):
            if name not in files:
                raise Invalid("Model map references a missing output file: " + name)
            blob = files[name]
            data = read_blob(store, blob)
            manifest[blob["id"]] = {k: blob[k] for k in ("id", "digest")}
            return data

        source("symplex_model_map.json")
        for component in model_map.components:
            if component.status == "omitted":
                bindings.append(component.model_dump())
                continue
            if not component.filename.endswith(".py"):
                raise Invalid("Component binding requires a Python source file")
            symbols = _symbols(source(component.filename))
            if component.symbol not in symbols:
                raise Invalid(
                    "Declared component symbol does not exist: " + component.symbol
                )
            bindings.append({**component.model_dump(), **symbols[component.symbol]})
        seen = set()
        for output in model_map.outputs:
            if (
                output.state_id not in states
                or output.unit_id != states[output.state_id]["unit_id"]
            ):
                raise Invalid(
                    "Output state and declared unit must match the system representation"
                )
            key = (output.state_id, output.filename, output.column)
            if key in seen:
                raise Invalid("Duplicate state-output binding")
            seen.add(key)
            if not output.filename.endswith(".csv"):
                raise Invalid("State binding requires a CSV output")
            reader = csv.reader(
                io.StringIO(source(output.filename).decode("utf-8-sig")), strict=True
            )
            columns = next(reader)
            if len(columns) != len(set(columns)) or output.column not in columns:
                raise Invalid("State output column is missing or ambiguous")
        return {
            "status": "linked",
            "system_id": system["id"],
            "system_digest": system["digest"],
            "components": bindings,
            "outputs": [o.model_dump() for o in model_map.outputs],
            "implemented_count": sum(
                c.status == "implemented" for c in model_map.components
            ),
            "omitted_count": sum(c.status == "omitted" for c in model_map.components),
            "source_manifest": list(manifest.values()),
            "scope": scope,
            "independently_validated": False,
        }
    except (
        ValueError,
        KeyError,
        OSError,
        SyntaxError,
        StopIteration,
        csv.Error,
        RecursionError,
    ) as error:
        return {
            "status": "failed",
            "error": str(error)[:1500],
            "scope": scope,
            "independently_validated": False,
        }
