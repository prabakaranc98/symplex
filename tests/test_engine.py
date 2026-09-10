import copy
import json
import tempfile
import threading
import unittest

from symplex.agents.evolution import Engine
from symplex.core.contracts import Invalid, MethodPatch, ProgramPatch
from symplex.evaluation.fixtures import synthetic_pack
from symplex.evaluation.metrics import score
from symplex.evidence.admission import admit, prediction_rows, relations
from symplex.evidence.spatial import validate_geojson
from symplex.infrastructure.providers import RuleProvider
from symplex.infrastructure.runner import Cancelled, execute, execute_system
from symplex.infrastructure.storage import Budget, BudgetExhausted, Store
from symplex.modeling.dynamics import simulate
from symplex.modeling.templates import DEFAULT_PROGRAM, compile_patch


def system_fixture():
    return {
        "title": "Synthetic coupled stocks",
        "boundary": "Test only",
        "time_step": "abstract step",
        "nodes": [
            {
                "id": n,
                "name": n,
                "meaning": "normalized state",
                "initial": v,
                "inertia": 0.7,
                "drift": 0.0,
                "noise_sd": 0.03,
                "evidence_ids": [],
                "assumption": "Test assumption",
            }
            for n, v in [("demand", 0.6), ("load", 0.3), ("benefit", 0.4)]
        ],
        "edges": [
            {
                "source": "demand",
                "target": "load",
                "weight": 0.7,
                "lag": 1,
                "mechanism": "positive load",
            },
            {
                "source": "load",
                "target": "benefit",
                "weight": -0.6,
                "lag": 0,
                "mechanism": "negative congestion",
            },
        ],
        "scenarios": [
            {
                "id": "baseline",
                "name": "Baseline",
                "interventions": [],
                "interpretation": "No action",
            },
            {
                "id": "demand_cap",
                "name": "Demand cap",
                "interventions": [{"node": "demand", "value": 0.2}],
                "interpretation": "Assumption",
            },
        ],
        "preferences": [
            {"node": "benefit", "weight": 1.0, "rationale": "Test preference"}
        ],
        "assumptions": ["Not empirical"],
        "unmodeled": ["Physics"],
        "disconfirmation": "Observe the actual system",
    }


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(self.temp.name)
        self.budget = Budget(self.store)
        self.data = synthetic_pack()

    def tearDown(self):
        self.temp.cleanup()

    def test_whole_groups_and_temporal_leakage(self):
        admit(self.data)
        for key, value in [
            ("available_at", "2021-01-01T00:00:00Z"),
            ("rules_available_at", "2021-01-01T00:00:00Z"),
            ("split", "confirmation"),
            ("observed_at", "2020-01-01T00:00:00"),
        ]:
            data = copy.deepcopy(self.data)
            data["rows"][0][key] = value
            with self.assertRaises(Invalid):
                admit(data)

    def test_future_training_labels_rejected(self):
        self.data["rows"][0]["resolved_at"] = "2026-01-01T00:00:00Z"
        with self.assertRaises(Invalid):
            admit(self.data)

    def test_nonfinite_and_units(self):
        for key, value in [
            ("price", float("nan")),
            ("unit", "meters"),
            ("label", True),
        ]:
            d = copy.deepcopy(self.data)
            d["rows"][0][key] = value
            with self.assertRaises(Invalid):
                admit(d)

    def test_near_match_is_not_joined(self):
        rows = prediction_rows(self.data["rows"][:3])
        rows[1]["entity"] = "OTHER STATION"
        audit = relations(rows)
        self.assertTrue(audit["rejected_near_matches"])
        self.assertFalse(
            any(rows[1]["id"] in (r["lower"], r["higher"]) for r in audit["accepted"])
        )

    def test_protected_patch_fields(self):
        p = {
            "parent_id": "a",
            "operation": "fit_calibrator",
            "value": 0,
            "evidence_ids": ["e"],
            "expectation": "x",
            "disconfirmation": "y",
        }
        ProgramPatch.parse(p)
        with self.assertRaises(Invalid):
            ProgramPatch.parse(dict(p, evaluator="cheat"))
        with self.assertRaises(Invalid):
            ProgramPatch.parse(dict(p, operation="run_shell"))
        with self.assertRaises(Invalid):
            compile_patch(dict(DEFAULT_PROGRAM, template="evil"), ProgramPatch.parse(p))

    def test_protected_method_threshold(self):
        with self.assertRaises(Invalid):
            MethodPatch.parse(
                {
                    "parent_id": "p",
                    "diagnostic_first": True,
                    "observed_failure": "x",
                    "promotion_margin": -1,
                }
            )

    def test_budget_persists_and_is_atomic(self):
        b = Budget(
            self.store, {"model_requests": 1}
        )  # Existing cap is immutable at 64.
        ident = b.reserve(model_requests=64)
        with self.assertRaises(BudgetExhausted):
            Budget(self.store).reserve(model_requests=1)
        self.assertEqual(b.snapshot()["model_requests"]["used_or_reserved"], 64)
        with self.assertRaises(BudgetExhausted):
            b.reserve(worker_seconds=1, usd=1)
        self.assertEqual(b.snapshot()["worker_seconds"]["used_or_reserved"], 0)
        b.settle(ident, model_requests=1)
        with self.assertRaises(Invalid):
            b.settle(ident, model_requests=0)

    def test_protected_inputs_do_not_enter_worker(self):
        with self.assertRaises(Invalid):
            execute(
                self.store,
                self.budget,
                DEFAULT_PROGRAM,
                self.data["rows"][:3],
                self.data["rows"][3:6],
            )

    def test_fit_and_reproduction(self):
        training = self.data["rows"][:18]
        inputs = prediction_rows(self.data["rows"][18:27])
        program = dict(DEFAULT_PROGRAM, projection=True, calibrator=True)
        first = execute(self.store, self.budget, program, training, inputs)
        second = execute(self.store, self.budget, program, training, inputs)
        self.assertEqual(first["predictions"], second["predictions"])
        self.assertEqual(first["input_digest"], second["input_digest"])
        metrics = score(self.data["rows"][18:27], first["predictions"])
        self.assertEqual(metrics["coherence_violations"], 0)

    def test_evaluator_rejects_bad_output(self):
        rows = self.data["rows"][:3]
        with self.assertRaises(Invalid):
            score(rows, {})
        with self.assertRaises(Invalid):
            score(rows, {r["id"]: float("inf") for r in rows})

    def test_complete_generation_and_idempotence(self):
        e = Engine(self.store, self.budget, RuleProvider())
        r = e.investigate(self.data)
        self.assertEqual(r["status"], "succeeded")
        self.assertEqual(len(self.store.list("program")), 3)
        decision = self.store.get(r["result_id"])["data"]
        self.assertTrue(decision["synthetic"])
        self.assertEqual(decision["status"], "inconclusive")
        before = self.budget.snapshot()
        again = e.investigate(self.data)
        self.assertEqual(r["id"], again["id"])
        self.assertEqual(before, self.budget.snapshot())
        self.assertTrue(self.store.list("action_result"))
        self.assertTrue(self.store.list("archive"))

    def test_failed_mutations_are_preserved(self):
        class Broken(RuleProvider):
            def propose(self, contract, context, *args, **kwargs):
                if contract is ProgramPatch:
                    raise Invalid("broken mutation")
                return super().propose(contract, context, *args, **kwargs)

        result = Engine(self.store, self.budget, Broken()).investigate(self.data)
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(len(self.store.list("failed_mutation")), 2)

    def test_cancel_and_timeout(self):
        event = threading.Event()
        event.set()
        with self.assertRaises(Cancelled):
            execute(
                self.store,
                self.budget,
                DEFAULT_PROGRAM,
                self.data["rows"][:3],
                prediction_rows(self.data["rows"][3:6]),
                event,
            )
        with self.assertRaises(TimeoutError):
            execute(
                self.store,
                self.budget,
                DEFAULT_PROGRAM,
                self.data["rows"][:3],
                prediction_rows(self.data["rows"][3:6]),
                timeout=0.00001,
            )

    def test_integrity_and_stale_descendants(self):
        a = self.store.put("problem", {"x": 1})
        b = self.store.put("run", {"x": 2}, a)
        c = self.store.put("decision", {"x": 3}, b)
        self.store.invalidate(a)
        self.assertTrue(self.store.get(c)["stale"])
        row = self.store.get(b)
        (self.store.root / "artifacts" / (row["digest"] + ".json")).write_text("{}")
        with self.assertRaises(Invalid):
            self.store.get(b)

    def test_audit_is_single_use(self):
        self.store.seal("audit", {"selection": "a"})
        with self.assertRaises(Invalid):
            self.store.seal("audit", {"selection": "b"})

    def test_dynamics_common_streams_and_ablation(self):
        spec = system_fixture()
        a = simulate(spec)
        b = simulate(spec)
        self.assertEqual(a, b)
        self.assertGreater(a["scenarios"]["demand_cap"]["paired_difference"], 0)
        ablated = copy.deepcopy(spec)
        for edge in ablated["edges"]:
            edge["weight"] = 0
        c = simulate(ablated)
        self.assertEqual(c["scenarios"]["demand_cap"]["paired_difference"], 0)
        output = execute_system(self.store, self.budget, spec)
        self.assertEqual(output["scenarios"], a["scenarios"])

    def test_dynamics_schema_guards(self):
        from pydantic import ValidationError

        s = system_fixture()
        s["edges"][0]["source"] = "nonexistent"
        with self.assertRaises(ValidationError):
            simulate(s)
        with self.assertRaises(Invalid):
            simulate(system_fixture(), rollouts=10000)

    def test_spatial_validation(self):
        point = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [77, 12]},
                    "properties": {"name": "Test"},
                }
            ],
        }
        validate_geojson(point)
        point["features"][0]["geometry"]["coordinates"][0] = 181
        with self.assertRaises(Invalid):
            validate_geojson(point)


class ServerTests(unittest.TestCase):
    def test_api_boundary_and_persistence(self):
        from fastapi.testclient import TestClient

        from symplex.interfaces.api import create_app

        with tempfile.TemporaryDirectory() as t:
            client = TestClient(create_app(t))
            s = client.get("/api/state").json()
            h = {"X-Symplex-Token": s["token"]}
            self.assertNotIn("OPENAI_API_KEY", json.dumps(s))
            self.assertEqual(client.get("/").status_code, 200)
            self.assertEqual(
                client.post(
                    "/api/problems", json={"question": "Example problem"}
                ).status_code,
                403,
            )
            self.assertEqual(
                client.post(
                    "/api/problems",
                    headers=dict(h, Origin="https://evil.example"),
                    json={"question": "Example problem"},
                ).status_code,
                403,
            )
            r = client.post(
                "/api/problems",
                headers=h,
                json={"question": "Airport planning with uncertain demand"},
            )
            self.assertEqual(r.status_code, 200)
            ident = r.json()["id"]
            self.assertEqual(
                client.post(
                    "/api/problems", headers=h, json={"question": "x", "extra": "a"}
                ).status_code,
                422,
            )
            context = client.post(
                "/api/context",
                headers=h,
                json={
                    "problem_id": ident,
                    "title": "Local assumptions",
                    "content": "No site selected.",
                    "format": "text",
                    "basis": "assumption",
                },
            )
            self.assertEqual(context.status_code, 200)
            second = TestClient(create_app(t))
            self.assertEqual(
                len(
                    [
                        r
                        for r in second.get("/api/state").json()["records"]
                        if r["kind"] == "context"
                    ]
                ),
                1,
            )
            revision = client.post(
                "/api/revise",
                headers=h,
                json={"id": ident, "assumption": "Demand is uncertain"},
            )
            self.assertEqual(revision.status_code, 200)
            self.assertTrue(
                client.get("/api/artifacts/" + context.json()["id"]).json()["stale"]
            )


if __name__ == "__main__":
    unittest.main()
