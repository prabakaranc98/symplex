"""Agent-selected projection of actual computed rows into the 3D viewer."""
import csv
import io
from pathlib import Path
from typing import ClassVar
from pydantic import Field
from symplex.agents.scientific_cycle import current_artifact
from symplex.connectors.compute import read_blob
from symplex.core.contracts import Invalid
from symplex.modeling.complex_system import ClosedContract, Text
from symplex.modeling.scenes import _executed_blob, scene_from_csv


class RowFilter(ClosedContract):
    column: Text
    equals: str


class SceneProjection(ClosedContract):
    output_token_budget: ClassVar[int] = 1800
    file_id: Text
    time_column: Text
    x_column: Text
    y_column: Text
    z_column: Text
    group_column: str | None
    filters: list[RowFilter] = Field(max_length=12)
    rationale: Text
    interpretation_limits: Text

    @classmethod
    def parse(cls, value):
        return cls.model_validate(value).model_dump()

    @classmethod
    def json_schema(cls):
        return cls.model_json_schema()


def build_scene(store, provider, problem_id, package_id):
    package = current_artifact(store, problem_id, 'compute_package', package_id)
    run = current_artifact(store, problem_id, 'compute_run', package['data']['run_id'])
    inventory = []
    for ident in package['data']['file_ids']:
        blob = current_artifact(store, problem_id, 'file_blob', ident)
        if Path(blob['data']['filename']).suffix.lower() != '.csv':
            continue
        # Validate execution and package lineage before exposing any rows to the model.
        blob, source_run = _executed_blob(store, problem_id, ident, '.csv')
        if source_run['id'] != run['id']:
            raise Invalid('Visualization CSV must belong to the selected package run')
        raw = read_blob(store, blob)
        if len(raw) > 5000000:
            raise Invalid('CSV scene input exceeds 5 MB')
        try:
            rows = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')), strict=True)
            headers = rows.fieldnames
            samples = [r for _, r in zip(range(3), rows)]
            if not headers or len(headers) != len(set(headers)) or any(not h for h in headers):
                raise Invalid('CSV must have unique nonempty column headers')
            if any(None in r or any(v is None for v in r.values()) for r in samples):
                raise Invalid('CSV contains a row that does not match its header')
        except (csv.Error, UnicodeDecodeError) as exc:
            raise Invalid('Scene source must be a valid bounded UTF-8 CSV') from exc
        inventory.append({'id': ident, 'filename': blob['data']['filename'],
                          'columns': headers, 'sample_rows': samples})
    if not inventory:
        raise Invalid('The selected computation has no recorded CSV to visualize')
    proposal = SceneProjection.parse(provider.propose(SceneProjection, {
        'available_csv': inventory, 'problem': store.get(problem_id)['data'],
        'instruction': 'Select an interpretable three-variable state trajectory from these actual columns. Explicitly select one fixture/tolerance via exact filters when duplicate group/time rows exist. Preserve actual rows; no invented coordinates or geography. Justify what this phase-space view reveals and its limitations. Different units must remain explicit in column labels. Maximum 300 groups, 400 frames, 60000 positions; use an exact subset only if scientifically meaningful.',
    }, problem_id))
    if proposal['file_id'] not in {r['id'] for r in inventory}:
        raise Invalid('Projection must use an offered executed CSV')
    if len({r['column'] for r in proposal['filters']}) != len(proposal['filters']):
        raise Invalid('Projection filters must have unique columns')
    scene_id = scene_from_csv(store, problem_id, proposal['file_id'],
        proposal['time_column'], proposal['x_column'], proposal['y_column'], proposal['z_column'],
        proposal['group_column'], filters={r['column']: r['equals'] for r in proposal['filters']})
    store.put('visualization_design', {**proposal, 'scene_id': scene_id, 'package_id': package_id,
                                      'basis': 'agent_interpretation'}, problem_id)
    return scene_id
