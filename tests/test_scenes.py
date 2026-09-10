import copy
import json
import tempfile
import unittest

from pydantic import ValidationError

from symplex.connectors.compute import save_blob
from symplex.core.contracts import Invalid
from symplex.infrastructure.storage import Store
from symplex.modeling.scenes import SimulationScene, load_scene, scene_from_csv


def valid_scene():
    return {
        "title": "Synthetic recorded trajectory",
        "basis": "synthetic",
        "coordinate_frame": "Hypothetical phase space, not geographic coordinates",
        "position_unit": "Declared arbitrary source units",
        "time_unit": "Declared source time units",
        "agents": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
        "frames": [
            {
                "time": 0.0,
                "positions": [{"agent_id": "a", "x": 1.0, "y": 2.0, "z": 3.0}],
            },
            {
                "time": 1.0,
                "positions": [
                    {"agent_id": "a", "x": 4.0, "y": 5.0, "z": 6.0},
                    {"agent_id": "b", "x": 2.0, "y": 1.0, "z": 0.0},
                ],
            },
        ],
        "limitations": ["Software fixture; not a validated physical simulation."],
    }


CSV_DATA = (
    "time,group,x,y,z,fixture,basis\n"
    "1,A,1,2,3,nominal,synthetic\n"
    "0,A,4,5,6,nominal,synthetic\n"
    "0,B,7,8,9,nominal,synthetic\n"
    "1,B,10,11,12,nominal,synthetic\n"
    "0,A,100,101,102,other,synthetic\n"
)


class SceneTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.store = Store(folder.name)
        self.problem_id = self.store.put(
            "workspace_problem", {"question": "Inspect executed trajectories."}
        )
        problem = self.store.get(self.problem_id)
        self.run_id = self.store.put(
            "compute_run",
            {
                "calls": [{"status": "completed"}],
                "input_manifest": [
                    {
                        "id": problem["id"],
                        "digest": problem["digest"],
                        "kind": "workspace_problem",
                    }
                ],
            },
            self.problem_id,
        )

    def source(self, content=CSV_DATA, *, basis="generated", run_id=None, parent=None):
        return save_blob(
            self.store,
            content.encode(),
            "trajectories.csv",
            parent or self.problem_id,
            basis,
            run_id or self.run_id,
        )

    def convert(self, blob_id, **kwargs):
        return scene_from_csv(
            self.store,
            self.problem_id,
            blob_id,
            "time",
            "x",
            "y",
            "z",
            "group",
            **kwargs,
        )

    def test_csv_projection_preserves_exact_selected_rows_and_provenance(self):
        blob_id = self.source()
        ident = self.convert(blob_id, filters={"fixture": "nominal"})
        record = self.store.get(ident)["data"]
        data = load_scene(self.store, self.problem_id, record["scene_blob_id"])
        labels = {agent["id"]: agent["label"] for agent in data["agents"]}
        rows = [
            (frame["time"], labels[p["agent_id"]], p["x"], p["y"], p["z"])
            for frame in data["frames"]
            for p in frame["positions"]
        ]
        self.assertEqual(
            sorted(rows),
            [
                (0.0, "A", 4.0, 5.0, 6.0),
                (0.0, "B", 7.0, 8.0, 9.0),
                (1.0, "A", 1.0, 2.0, 3.0),
                (1.0, "B", 10.0, 11.0, 12.0),
            ],
        )
        self.assertEqual(
            record["source_blob_digest"], self.store.get(blob_id)["digest"]
        )
        self.assertEqual(record["run_digest"], self.store.get(self.run_id)["digest"])
        self.assertEqual(record["source_row_count"], 5)
        self.assertEqual(record["position_count"], 4)
        self.assertEqual(record["filters"], {"fixture": "nominal"})
        self.assertEqual(data["basis"], "synthetic")
        self.assertIn("Phase-space derived result", data["coordinate_frame"])
        self.assertIn("not inferred geographic", data["coordinate_frame"])
        self.assertFalse(record["independently_validated"])
        self.assertEqual(self.convert(blob_id, filters={"fixture": "nominal"}), ident)
        self.assertEqual(len(self.store.list("simulation_scene")), 1)

    def test_missing_samples_are_not_filled_and_units_are_not_invented(self):
        blob_id = self.source("time,group,x,y,z\n0,A,1,2,3\n1,A,4,5,6\n1,B,7,8,9\n")
        ident = self.convert(blob_id)
        data = load_scene(
            self.store, self.problem_id, self.store.get(ident)["data"]["scene_blob_id"]
        )
        self.assertEqual([len(f["positions"]) for f in data["frames"]], [1, 2])
        self.assertEqual(data["basis"], "conditional_model")
        self.assertIn("not inferred", data["position_unit"])
        self.assertIn("unit not inferred", data["time_unit"])

    def test_no_group_column_means_one_actual_trajectory(self):
        blob_id = self.source("time,x,y,z\n0,1,2,3\n1,4,5,6\n")
        ident = scene_from_csv(
            self.store, self.problem_id, blob_id, "time", "x", "y", "z"
        )
        data = load_scene(
            self.store, self.problem_id, self.store.get(ident)["data"]["scene_blob_id"]
        )
        self.assertEqual(len(data["agents"]), 1)
        self.assertEqual(len(data["frames"]), 2)

    def test_ambiguous_or_invalid_csv_is_rejected_without_silent_row_dropping(self):
        cases = [
            "time,group,x,y,z\n0,A,1,2,3\n0,A,4,5,6\n",
            "time,group,x,y,z\n0,A,nan,2,3\n",
            "time,group,x,y,z\n0,A,1,inf,3\n",
            "time,group,x,y,z\n0,A,,2,3\n",
            "time,group,x,y,z\n0,A,1,2\n",
            "time,group,x,y,z\n0,A,1,2,3,extra\n",
            "time,group,x,x,z\n0,A,1,2,3\n",
            "time,group,x,y,z\n0,,1,2,3\n",
            "time,group,x,y,z\n",
        ]
        for index, content in enumerate(cases):
            with self.subTest(csv=index), self.assertRaises(Invalid):
                self.convert(self.source(content))
        self.assertFalse(self.store.list("simulation_scene"))

    def test_filter_and_column_selection_must_be_explicit_and_valid(self):
        blob_id = self.source()
        for filters in ({"absent": "x"}, {"fixture": "absent"}, {"fixture": 1}, []):
            with self.subTest(filters=filters), self.assertRaises(Invalid):
                self.convert(blob_id, filters=filters)
        with self.assertRaises(Invalid):
            scene_from_csv(self.store, self.problem_id, blob_id, None, "x", "y", "z")
        with self.assertRaises(Invalid):
            scene_from_csv(
                self.store, self.problem_id, blob_id, "time", "missing", "y", "z"
            )

    def test_scope_completion_and_input_digest_bindings_are_required(self):
        bad_runs = [
            {
                "calls": [{"status": "failed"}],
                "input_manifest": self.store.get(self.run_id)["data"]["input_manifest"],
            },
            {"calls": [{"status": "completed"}], "input_manifest": []},
            {
                "calls": [{"status": "completed"}],
                "input_manifest": [{"id": self.problem_id, "digest": "wrong"}],
            },
        ]
        for data in bad_runs:
            run_id = self.store.put("compute_run", data, self.problem_id)
            with self.subTest(run=data), self.assertRaises(Invalid):
                self.convert(self.source(run_id=run_id), filters={"fixture": "nominal"})
        with self.assertRaises(Invalid):
            self.convert(
                self.source(basis="user_context"), filters={"fixture": "nominal"}
            )
        other = self.store.put("workspace_problem", {"question": "Foreign problem"})
        with self.assertRaises(Invalid):
            self.convert(self.source(parent=other), filters={"fixture": "nominal"})
        self.store.invalidate(self.run_id)
        with self.assertRaises(Invalid):
            self.convert(self.source(), filters={"fixture": "nominal"})

    def test_recorded_scene_ids_times_and_numbers_are_strict(self):
        self.assertEqual(SimulationScene.parse(valid_scene()), valid_scene())
        mutations = [
            lambda d: d["frames"][0]["positions"][0].update(x=float("nan")),
            lambda d: d["frames"][0]["positions"][0].update(z=True),
            lambda d: d["frames"][0]["positions"][0].update(agent_id="unknown"),
            lambda d: d["frames"][1].update(time=0.0),
            lambda d: d["frames"][1].update(time=float("inf")),
            lambda d: d["frames"][1]["positions"].append(
                copy.deepcopy(d["frames"][1]["positions"][0])
            ),
            lambda d: d["agents"].append({"id": "unused", "label": "Unused"}),
            lambda d: d.update(external_assets=["https://untrusted.invalid/model"]),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                data = valid_scene()
                mutate(data)
                with self.assertRaises(ValidationError):
                    SimulationScene.parse(data)

    def test_scene_size_envelopes_and_closed_schema(self):
        data = valid_scene()
        data["agents"] = [{"id": "a" + str(i), "label": str(i)} for i in range(301)]
        with self.assertRaises(ValidationError):
            SimulationScene.parse(data)
        data = valid_scene()
        data["frames"] = [
            {
                "time": float(i),
                "positions": [{"agent_id": "a", "x": 0.0, "y": 0.0, "z": 0.0}],
            }
            for i in range(401)
        ]
        with self.assertRaises(ValidationError):
            SimulationScene.parse(data)
        data = valid_scene()
        data["agents"] = [{"id": "a" + str(i), "label": str(i)} for i in range(300)]
        positions = [
            {"agent_id": a["id"], "x": 0.0, "y": 0.0, "z": 0.0} for a in data["agents"]
        ]
        data["frames"] = [
            {"time": float(i), "positions": positions} for i in range(201)
        ]
        with self.assertRaisesRegex(ValidationError, "60000-position"):
            SimulationScene.parse(data)

        def inspect(node):
            if isinstance(node, dict):
                if node.get("type") == "object":
                    self.assertIs(node.get("additionalProperties"), False)
                    self.assertEqual(set(node["required"]), set(node["properties"]))
                for value in node.values():
                    inspect(value)
            elif isinstance(node, list):
                for value in node:
                    inspect(value)

        inspect(SimulationScene.json_schema())

    def test_json_scene_loading_retains_generated_run_provenance(self):
        blob_id = save_blob(
            self.store,
            json.dumps(valid_scene()).encode(),
            "symplex_scene.json",
            self.problem_id,
            "generated",
            self.run_id,
        )
        self.assertEqual(
            load_scene(self.store, self.problem_id, blob_id), valid_scene()
        )
        bad = save_blob(
            self.store,
            b'{"frames":[]}',
            "symplex_scene.json",
            self.problem_id,
            "user_context",
            self.run_id,
        )
        with self.assertRaises(Invalid):
            load_scene(self.store, self.problem_id, bad)
        source = self.store.get(blob_id)
        (self.store.root / "blobs" / source["data"]["sha256"]).write_bytes(b"tampered")
        with self.assertRaisesRegex(Invalid, "integrity"):
            load_scene(self.store, self.problem_id, blob_id)


if __name__ == "__main__":
    unittest.main()
