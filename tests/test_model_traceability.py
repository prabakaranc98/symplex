"""A model map cannot substitute plausible prose for resolvable output bindings."""

import copy
import json
import unittest

from symplex.connectors.compute import save_blob
from symplex.evaluation.model_traceability import check_model_map
from tests import test_experiments as fixtures


class ModelTraceabilityTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ExperimentTests(
            "test_protocol_is_frozen_with_system_digest_and_output_contract"
        )
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.store, self.problem = self.fixture.store, self.fixture.problem_id
        self.model_map = {
            "system_id": self.fixture.system_id,
            "components": [
                {
                    "component_id": "source",
                    "status": "omitted",
                    "filename": None,
                    "symbol": None,
                    "mathematical_description": "Exogenous fixture",
                    "limitation": "Not observed",
                },
                {
                    "component_id": "response_model",
                    "status": "implemented",
                    "filename": "model.py",
                    "symbol": "response",
                    "mathematical_description": "x + 1",
                    "limitation": "Illustrative only",
                },
            ],
            "outputs": [
                {
                    "state_id": "response",
                    "filename": "trajectory.csv",
                    "column": "quality",
                    "unit_id": "celsius",
                    "interpretation": "Fixture output",
                }
            ],
            "scope": "Traceability fixture, not a physical model",
        }

    def package(
        self, value=None, code=b"def response(x):\n    return x + 1\n", foreign=False
    ):
        package_id, run_id, _ = self.fixture.package()
        files = self.store.get(package_id)["data"]["file_ids"]
        for name, raw in (
            ("symplex_model_map.json", json.dumps(value or self.model_map).encode()),
            ("model.py", code),
        ):
            files.append(
                save_blob(
                    self.store,
                    raw,
                    name,
                    self.problem,
                    "generated",
                    "foreign_run" if foreign else run_id,
                )
            )
        return self.store.put(
            "compute_package", {"run_id": run_id, "file_ids": files}, self.problem
        )

    def check(self, package_id):
        return check_model_map(self.store, self.problem, package_id)

    def test_resolvable_bindings_expose_omissions_without_scientific_claim(self):
        result = self.check(self.package())
        self.assertEqual(result["status"], "linked")
        self.assertEqual((result["implemented_count"], result["omitted_count"]), (1, 1))
        self.assertEqual(result["components"][1]["line"], 1)
        self.assertEqual(len(result["source_manifest"]), 3)
        self.assertFalse(result["independently_validated"])

    def test_missing_map_is_unavailable(self):
        package, _, _ = self.fixture.package()
        self.assertEqual(self.check(package)["status"], "unavailable")

    def test_fabricated_symbols_columns_units_and_coverage_fail(self):
        for path, value in (
            (("components", 1, "symbol"), "invented"),
            (("outputs", 0, "column"), "absent"),
            (("outputs", 0, "unit_id"), "kelvin"),
            (("components", 0, "component_id"), "invented"),
        ):
            model_map = copy.deepcopy(self.model_map)
            model_map[path[0]][path[1]][path[2]] = value
            with self.subTest(path=path):
                self.assertEqual(
                    self.check(self.package(model_map))["status"], "failed"
                )

    def test_foreign_files_and_corrupt_sources_fail(self):
        self.assertEqual(self.check(self.package(foreign=True))["status"], "failed")
        package = self.package()
        self.assertEqual(self.check(package)["status"], "linked")
        blob = self.store.get(self.store.get(package)["data"]["file_ids"][-1])
        (self.store.root / "blobs" / blob["data"]["sha256"]).write_bytes(b"corrupt")
        self.assertEqual(self.check(package)["status"], "failed")

    def test_python_is_parsed_not_executed_and_invalid_syntax_fails(self):
        package = self.package(
            code=b"raise RuntimeError('must never execute')\ndef response(x):\n    return x\n"
        )
        self.assertEqual(self.check(package)["status"], "linked")
        self.assertEqual(
            self.check(self.package(code=b"def invalid !!!"))["status"], "failed"
        )


if __name__ == "__main__":
    unittest.main()
