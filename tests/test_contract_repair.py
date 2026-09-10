"""Small proposal repairs cannot broaden the host mutation or validation surface."""

import copy
import json
import unittest

from pydantic import ValidationError

from symplex.core.contracts import Invalid
from symplex.core.repair import ProposalRepair, apply_repair
from symplex.modeling.complex_system import ComplexSystemSpec
from tests.test_complex_system import system_spec


def replacement(path, value, reason="Correct the rejected leaf"):
    return {"path": path, "value_json": json.dumps(value), "reason": reason}


def patch(*operations):
    return {"operations": list(operations)}


class ContractRepairTests(unittest.TestCase):
    def test_repair_fixes_one_port_scale_and_final_contract_still_validates(self):
        original = system_spec()
        correct_scale = original["components"][0]["ports"][0]["scale_id"]
        original["components"][0]["ports"][0]["scale_id"] = "undeclared_clock"
        saved = copy.deepcopy(original)
        with self.assertRaises(ValidationError):
            ComplexSystemSpec.parse(original)
        proposal = ProposalRepair.parse(patch(replacement("/components/0/ports/0/scale_id", correct_scale)))
        repaired = apply_repair(original, proposal)
        ComplexSystemSpec.parse(repaired)
        self.assertEqual(original, saved)
        self.assertIsNot(original, repaired)
        self.assertEqual(repaired, system_spec())
        self.assertEqual(ProposalRepair.output_token_budget, 1800)

    def test_leaf_patch_does_not_weaken_the_final_contract(self):
        original = system_spec()
        repaired = apply_repair(original, patch(replacement("/components/0/ports/0/scale_id", "still_undeclared")))
        with self.assertRaises(ValidationError):
            ComplexSystemSpec.parse(repaired)

    def test_json_pointer_decoding_and_existing_array_leaves(self):
        original = {"a/b": {"~key": [1, None]}, "~1": "literal", "": "empty key"}
        result = apply_repair(original, patch(
            replacement("/a~1b/~0key/0", 2),
            replacement("/a~1b/~0key/1", False),
            replacement("/~01", "correctly decoded once"),
            replacement("/", "existing empty key"),
        ))
        self.assertEqual(result["a/b"]["~key"], [2, False])
        self.assertEqual(result["~1"], "correctly decoded once")
        self.assertEqual(result[""], "existing empty key")
        self.assertEqual(original["a/b"]["~key"], [1, None])

    def test_paths_cannot_replace_root_containers_add_delete_or_escape_bounds(self):
        original = {"leaf": 1, "nested": {"leaf": 2}, "arr": [3]}
        paths = ("", "leaf", "#/leaf", "/leaf~", "/leaf~2", "/missing", "/nested/missing",
                 "/arr/-", "/arr/-1", "/arr/+0", "/arr/01", "/arr/1", "/leaf/x",
                 "/nested", "/arr", "/" + "a" * 501, "/a" * 33)
        for path in paths:
            with self.subTest(path=path), self.assertRaises(Invalid):
                apply_repair(original, patch(replacement(path, 9)))

    def test_scalar_values_only_and_nonfinite_or_malformed_json_are_rejected(self):
        values = ("{}", "[]", "[1]", "NaN", "Infinity", "-Infinity", "1e10000",
                  "", "undefined", "'string'", "1 2", '{"unterminated":', '"' + "x" * 4001 + '"')
        for value_json in values:
            with self.subTest(value_json=value_json[:30]), self.assertRaises(Invalid):
                apply_repair({"leaf": 1}, patch({"path": "/leaf", "value_json": value_json, "reason": "Repair"}))

    def test_operations_are_closed_bounded_unique_and_not_identity(self):
        valid = replacement("/leaf", 2)
        invalid_patches = [
            {}, {"operations": [valid], "other": 1},
            patch(*[replacement("/" + str(i), i) for i in range(13)]),
            patch(dict(valid, op="remove")), patch({"path": "/leaf", "value_json": "2"}),
            patch(dict(valid, reason="")), patch(dict(valid, reason="x" * 501)),
            patch(dict(valid, value_json=2)), patch(dict(valid, path=1)),
            patch(valid, replacement("/leaf", 3)),
        ]
        for value in invalid_patches:
            with self.subTest(value=str(value)[:80]), self.assertRaises(Invalid):
                ProposalRepair.parse(value)
        with self.assertRaises(Invalid):
            apply_repair({"leaf": 1}, patch(replacement("/leaf", 1)))
        with self.assertRaises(Invalid):
            apply_repair({"leaf": 1}, ProposalRepair([]))

    def test_empty_operations_are_explicit_abstention_and_cannot_accept_original(self):
        proposal = ProposalRepair.parse(patch())
        self.assertEqual(proposal.operations, [])
        self.assertEqual(ProposalRepair.json_schema()["properties"]["operations"]["minItems"], 0)
        original = {"leaf": "still invalid"}
        with self.assertRaisesRegex(Invalid, "No justified leaf repair"):
            apply_repair(original, proposal)
        self.assertEqual(original, {"leaf": "still invalid"})

    def test_failure_is_atomic_and_identity_compares_json_types(self):
        original = {"leaf": 1, "second": 2, "truth": True}
        saved = copy.deepcopy(original)
        with self.assertRaises(Invalid):
            apply_repair(original, patch(replacement("/leaf", 3), replacement("/absent", 4)))
        self.assertEqual(original, saved)
        result = apply_repair(original, patch(replacement("/truth", 1)))
        self.assertIs(type(result["truth"]), int)

    def test_host_size_and_json_tree_bounds_apply_to_original_and_patch(self):
        for original in ({"leaf": float("nan")}, {"leaf": float("inf")},
                         {"leaf": (1, 2)}, {1: "non-string key"}, {"leaf": "x" * 1000000}):
            with self.subTest(original_type=str(type(original))), self.assertRaises(Invalid):
                apply_repair(original, patch(replacement("/leaf", 2)))
        nested = {"leaf": 1}
        for _ in range(65):
            nested = {"child": nested}
        with self.assertRaises(Invalid):
            apply_repair(nested, patch(replacement("/leaf", 2)))
        with self.assertRaises(Invalid):
            ProposalRepair.parse(patch(*[replacement("/" + str(i), "x" * 3000) for i in range(6)]))

    def test_python_object_names_are_plain_json_keys_and_schema_is_closed(self):
        result = apply_repair({"__proto__": "old"}, patch(replacement("/__proto__", "new")))
        self.assertEqual(result, {"__proto__": "new"})
        schema = ProposalRepair.json_schema()
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["operations"]["maxItems"], 12)
        operation = schema["properties"]["operations"]["items"]
        self.assertFalse(operation["additionalProperties"])
        self.assertEqual(set(operation["required"]), {"path", "value_json", "reason"})


if __name__ == "__main__":
    unittest.main()
