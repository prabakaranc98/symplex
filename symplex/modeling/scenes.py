"""Bounded 3D trajectories from executed artifacts, without invented coordinates.

A scene is a visualization contract. Its declared basis is not scientific
validation, and an arbitrary three-column projection is never a geographic model.
"""

import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from symplex.connectors.compute import read_blob, save_blob
from symplex.core.contracts import Invalid, canonical, digest
from symplex.modeling.complex_system import ClosedContract, Identifier, Text


class SceneAgent(ClosedContract):
    id: Identifier
    label: Text


class ScenePosition(ClosedContract):
    agent_id: Identifier
    x: float
    y: float
    z: float


class SceneFrame(ClosedContract):
    time: float
    positions: list[ScenePosition] = Field(min_length=1, max_length=300)


class SimulationScene(ClosedContract):
    title: Text
    basis: Literal["synthetic", "conditional_model", "observational_data"]
    coordinate_frame: Text
    position_unit: Text
    time_unit: Text
    agents: list[SceneAgent] = Field(min_length=1, max_length=300)
    frames: list[SceneFrame] = Field(min_length=1, max_length=400)
    limitations: list[Text] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def trajectory_integrity(self):
        agents = {a.id for a in self.agents}
        if len(agents) != len(self.agents):
            raise ValueError("Scene agent IDs must be unique")
        seen = set()
        count = 0
        previous = None
        for frame in self.frames:
            if previous is not None and frame.time <= previous:
                raise ValueError("Scene times must be strictly ascending")
            previous = frame.time
            ids = [p.agent_id for p in frame.positions]
            if len(ids) != len(set(ids)):
                raise ValueError("An agent can have only one position per frame")
            if not set(ids) <= agents:
                raise ValueError("Scene position references an unknown agent")
            seen.update(ids)
            count += len(ids)
        if seen != agents:
            raise ValueError(
                "Every declared scene agent must have an observed position"
            )
        if count > 60000:
            raise ValueError("Scene exceeds the 60000-position envelope")
        return self

    @classmethod
    def parse(cls, value):
        return cls.model_validate(value).model_dump()

    @classmethod
    def json_schema(cls):
        return cls.model_json_schema()


def _executed_blob(store, problem_id, blob_id, extension):
    try:
        problem = store.get(problem_id)
        blob = store.get(blob_id)
        run = store.get(blob["data"].get("run_id"))
    except (KeyError, TypeError):
        raise Invalid(
            "A source artifact from a completed compute run is required"
        ) from None
    if problem["kind"] != "workspace_problem" or problem["stale"]:
        raise Invalid("Select a current problem")
    if (
        blob["kind"] != "file_blob"
        or blob["parent"] != problem_id
        or blob["stale"]
        or blob["data"].get("basis") != "generated"
        or Path(blob["data"].get("filename", "")).suffix.lower() != extension
    ):
        raise Invalid(
            "Scene source must be a current generated file of the required format"
        )
    if (
        run["kind"] != "compute_run"
        or run["parent"] != problem_id
        or run["stale"]
        or not any(c.get("status") == "completed" for c in run["data"].get("calls", []))
    ):
        raise Invalid("Scene source must belong to a completed run for this problem")
    if not any(
        r.get("id") == problem_id and r.get("digest") == problem["digest"]
        for r in run["data"].get("input_manifest", [])
    ):
        raise Invalid("Scene run must be bound to this problem's input digest")
    return blob, run


def load_scene(store, problem_id, blob_id):
    """Read an executed, problem-bound scene; no URLs or remote assets are followed."""
    blob, _ = _executed_blob(store, problem_id, blob_id, ".json")
    raw = read_blob(store, blob)
    if len(raw) > 16000000:
        raise Invalid("Scene JSON exceeds 16 MB")
    try:
        return SimulationScene.parse(json.loads(raw))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Invalid("Scene must be valid UTF-8 JSON") from exc


def scene_from_csv(
    store,
    problem_id,
    blob_id,
    time_column,
    x_column,
    y_column,
    z_column,
    group_column=None,
    *,
    filters=None,
):
    """Project actual CSV columns into phase space and retain full source provenance.

    Sorting existing samples changes presentation order only. Missing coordinates,
    ambiguous duplicate times/groups, and nonfinite values fail instead of being
    interpolated, filled, averaged, or silently discarded.
    """
    blob, run = _executed_blob(store, problem_id, blob_id, ".csv")
    raw = read_blob(store, blob)
    if len(raw) > 5000000:
        raise Invalid("CSV scene input exceeds 5 MB")
    columns = {
        "time": time_column,
        "x": x_column,
        "y": y_column,
        "z": z_column,
        "group": group_column,
    }
    if any(
        not isinstance(c, str) or not c or len(c) > 200
        for c in (time_column, x_column, y_column, z_column)
    ):
        raise Invalid("Choose explicit nonempty time and coordinate column names")
    requested = [c for c in columns.values() if c is not None]
    filters = {} if filters is None else filters
    if (
        not isinstance(filters, dict)
        or len(filters) > 12
        or any(
            not isinstance(k, str)
            or not k
            or len(k) > 200
            or not isinstance(v, str)
            or len(v) > 1600
            for k, v in filters.items()
        )
    ):
        raise Invalid(
            "Scene filters must be at most 12 explicit column-to-string selections"
        )
    requested.extend(filters)
    if any(not isinstance(c, str) or not c or len(c) > 200 for c in requested):
        raise Invalid("Choose explicit nonempty CSV column names")
    try:
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")), strict=True)
        headers = reader.fieldnames
        if (
            not headers
            or len(headers) != len(set(headers))
            or any(not h for h in headers)
        ):
            raise Invalid("CSV must have unique nonempty column headers")
        if not set(requested) <= set(headers):
            raise Invalid("Selected scene columns are absent from the CSV")
        times, labels, count, source_rows, declared_bases = {}, {}, 0, 0, set()
        for row in reader:
            source_rows += 1
            if None in row or any(value is None for value in row.values()):
                raise Invalid("CSV contains a row that does not match its header")
            if any(row[key] != value for key, value in filters.items()):
                continue
            if "basis" in row:
                declared_bases.add(row["basis"])
            values = []
            for column in (time_column, x_column, y_column, z_column):
                try:
                    value = float(row[column])
                except (ValueError, TypeError):
                    raise Invalid(
                        "Scene coordinates and times must all be numeric"
                    ) from None
                if not math.isfinite(value):
                    raise Invalid("Scene coordinates and times must all be finite")
                values.append(value)
            label = (
                row[group_column] if group_column is not None else "Source trajectory"
            )
            if not label.strip() or len(label) > 1600:
                raise Invalid("CSV group labels must be nonempty and bounded")
            ident = "trajectory_" + hashlib.sha256(label.encode()).hexdigest()[:16]
            labels[ident] = label
            frame = times.setdefault(values[0], {})
            if ident in frame:
                raise Invalid("CSV has duplicate positions for the same time and group")
            frame[ident] = {
                "agent_id": ident,
                "x": values[1],
                "y": values[2],
                "z": values[3],
            }
            count += 1
            if len(labels) > 300 or len(times) > 400 or count > 60000:
                raise Invalid(
                    "CSV exceeds the 300-agent, 400-frame, 60000-position scene envelope"
                )
    except (csv.Error, UnicodeDecodeError) as exc:
        raise Invalid("Scene source must be a valid bounded UTF-8 CSV") from exc
    if not count:
        raise Invalid("CSV has no trajectory samples")
    basis = (
        next(iter(declared_bases)) if len(declared_bases) == 1 else "conditional_model"
    )
    if basis not in {"synthetic", "conditional_model", "observational_data"}:
        basis = "conditional_model"
    scene = SimulationScene.parse(
        {
            "title": "Phase-space trajectory from " + blob["data"]["filename"],
            "basis": basis,
            "coordinate_frame": f"Phase-space derived result: X={x_column}; Y={y_column}; Z={z_column}. These are source-column values, not inferred geographic coordinates.",
            "position_unit": "Source-column units; not inferred or converted",
            "time_unit": f"Source column {time_column}; unit not inferred",
            "agents": [
                {"id": ident, "label": labels[ident]} for ident in sorted(labels)
            ],
            "frames": [
                {
                    "time": time,
                    "positions": [positions[ident] for ident in sorted(positions)],
                }
                for time, positions in sorted(times.items())
            ],
            "limitations": [
                "Derived directly from generated CSV samples; scientific validity and source units are not independently established.",
                "A phase-space projection is not a physical or geographic scene unless a separately validated coordinate model establishes that meaning.",
                "Missing samples remain absent. Playback steps through recorded frames without generating intermediate positions.",
                "Basis is a unanimous recognized maker-declared CSV value when available, otherwise conditional_model; it is not a host verification of observations.",
            ],
        }
    )
    identity = digest(
        {
            "source_id": blob_id,
            "source_digest": blob["digest"],
            "columns": columns,
            "filters": filters,
            "version": "csv-phase-space-v1",
        }
    )
    for existing in store.list("simulation_scene"):
        if (
            existing["parent"] == problem_id
            and not existing["stale"]
            and existing["data"].get("input_digest") == identity
        ):
            load_scene(store, problem_id, existing["data"]["scene_blob_id"])
            return existing["id"]
    output_id = save_blob(
        store,
        canonical(scene).encode(),
        "symplex_scene.json",
        problem_id,
        "generated",
        run["id"],
    )
    return store.put(
        "simulation_scene",
        {
            "scene_blob_id": output_id,
            "source_blob_id": blob_id,
            "source_blob_digest": blob["digest"],
            "source_content_sha256": blob["data"]["sha256"],
            "run_id": run["id"],
            "run_digest": run["digest"],
            "column_mapping": columns,
            "filters": filters,
            "source_row_count": source_rows,
            "input_digest": identity,
            "scene_digest": digest(scene),
            "basis": scene["basis"],
            "basis_authority": "maker-declared from unanimous selected CSV rows; otherwise conditional_model fallback",
            "coordinate_frame": scene["coordinate_frame"],
            "frame_count": len(scene["frames"]),
            "agent_count": len(scene["agents"]),
            "position_count": count,
            "independently_validated": False,
            "scope": "Deterministic phase-space visualization of executed CSV samples; no inferred geometry or invented samples.",
        },
        problem_id,
    )
