import copy
import json
import unittest

from pydantic import ValidationError

from symplex.modeling.complex_system import (
    ComplexSystemSpec,
    validate_artifact_references,
)


def system_spec():
    """Synthetic contract fixture; no experiment or domain claim is being validated."""
    return {
        "title": "Two coupled thermal representations",
        "representation_rationale": "One sensor modality and one mechanistic family are enough to test the proposed interface.",
        "complexity": {
            "nonlinearity": "Not assumed in this interface fixture.",
            "stochasticity": "Instrument error is not yet characterized.",
            "cross_scale_effects": "Only one temporal scale is proposed initially.",
            "emergence": "No emergent behavior has been established.",
        },
        "decision": {
            "question": "Which experimental condition discriminates the candidate mechanisms?",
            "beneficiary": "Experiment designer",
            "endpoints": [
                {
                    "id": "endpoint",
                    "name": "Response",
                    "state_id": "response",
                    "direction": "characterize",
                    "criterion": "Distinguish the competing response predictions.",
                }
            ],
            "alternatives": [
                {
                    "id": "baseline",
                    "name": "Reference condition",
                    "interventions": [],
                    "rationale": "Reference for comparison.",
                },
                {
                    "id": "perturbed",
                    "name": "Perturbed condition",
                    "interventions": [
                        {
                            "state_id": "temperature",
                            "change": "Controlled thermal perturbation.",
                        }
                    ],
                    "rationale": "Excite a distinguishing response.",
                },
            ],
            "constraints": [
                {
                    "id": "constraint",
                    "description": "Stay within instrument range.",
                    "state_ids": ["temperature"],
                    "test": "Check device limits before execution.",
                }
            ],
        },
        "boundary": {
            "included": ["Declared temperature response"],
            "excluded": ["No physical mechanism has been established."],
            "scales": [
                {
                    "id": "seconds",
                    "name": "One-second observations",
                    "step_seconds": 1.0,
                    "clock": "elapsed time since experimental start",
                }
            ],
        },
        "units": [
            {
                "id": "kelvin",
                "symbol": "K",
                "dimension": "temperature",
                "scale_to_canonical": 1.0,
                "offset_to_canonical": 0.0,
            },
            {
                "id": "celsius",
                "symbol": "degC",
                "dimension": "temperature",
                "scale_to_canonical": 1.0,
                "offset_to_canonical": 273.15,
            },
        ],
        "entities": [
            {
                "id": "system",
                "name": "Experimental system",
                "meaning": "Hypothetical instrumented system.",
            }
        ],
        "states": [
            {
                "id": "temperature",
                "entity_id": "system",
                "name": "Input temperature",
                "kind": "observed",
                "unit_id": "kelvin",
                "scale_id": "seconds",
                "meaning": "Sensor input before conversion.",
            },
            {
                "id": "response",
                "entity_id": "system",
                "name": "Response temperature",
                "kind": "latent",
                "unit_id": "celsius",
                "scale_id": "seconds",
                "meaning": "Candidate response state.",
            },
        ],
        "observations": [
            {
                "id": "sensor",
                "name": "Instrument observation",
                "modality": "sensor",
                "source_artifact_ids": ["context_known"],
                "state_ids": ["temperature"],
                "measurement_process": "Instrument protocol pending verification.",
                "missingness": "Unknown missing observations.",
                "uncertainty": "Calibration error unknown.",
                "status": "available",
            }
        ],
        "components": [
            {
                "id": "source",
                "name": "Boundary condition",
                "kind": "mechanistic",
                "mechanism": "Supply the proposed boundary state.",
                "ports": [
                    {
                        "id": "source_out",
                        "direction": "output",
                        "state_id": "temperature",
                        "unit_id": "kelvin",
                        "scale_id": "seconds",
                    }
                ],
                "assumptions": ["Boundary is externally controlled."],
                "evidence_ids": [],
            },
            {
                "id": "response_model",
                "name": "Response mechanism",
                "kind": "mechanistic",
                "mechanism": "Compare memoryless and delayed response hypotheses.",
                "ports": [
                    {
                        "id": "response_in",
                        "direction": "input",
                        "state_id": "response",
                        "unit_id": "celsius",
                        "scale_id": "seconds",
                    }
                ],
                "assumptions": ["Thermal abstraction is provisionally useful."],
                "evidence_ids": ["context_known"],
            },
        ],
        "couplings": [
            {
                "id": "temperature_coupling",
                "source_port": "source_out",
                "target_port": "response_in",
                "conversion": {"scale": 1.0, "offset": -273.15},
                "time_alignment": "synchronous",
                "alignment_rationale": "A shared observation clock and interval.",
                "mechanism": "Declared thermal input to the response model.",
            }
        ],
        "feedback": [],
        "hypotheses": [
            {
                "id": "memoryless",
                "claim": "Response has no memory.",
                "rivals": ["delayed"],
                "component_ids": ["response_model"],
                "evidence_ids": [],
                "discriminating_test": "Apply a held perturbation and measure lag.",
                "expected_observation": "No resolved lag.",
                "rejection_condition": "Lag exceeds instrument uncertainty.",
            },
            {
                "id": "delayed",
                "claim": "Response has memory.",
                "rivals": ["memoryless"],
                "component_ids": ["response_model"],
                "evidence_ids": [],
                "discriminating_test": "Apply a held perturbation and measure lag.",
                "expected_observation": "Resolved response lag.",
                "rejection_condition": "Responses consistently indistinguishable from immediate response.",
            },
        ],
        "experiments": [
            {
                "id": "lag_test",
                "name": "Lag comparison",
                "hypothesis_ids": ["memoryless", "delayed"],
                "baseline_alternative_id": "baseline",
                "candidate_alternative_ids": ["perturbed"],
                "endpoint_ids": ["endpoint"],
                "method": "Compare measured trajectories under controlled conditions.",
                "comparison_controls": "Same instrument, observation window, and noise protocol.",
                "uncertainty_plan": "Separate instrument error from repeated-run variability.",
                "stopping_rule": "Stop at the registered observation budget; report inconclusive if unresolved.",
            }
        ],
        "validation": {
            "identifiability_gaps": ["Noise can conceal lag."],
            "checks": [
                {
                    "id": "instrument_check",
                    "name": "Instrument check",
                    "component_ids": ["source"],
                    "method": "Use independent calibration observations.",
                    "acceptance_condition": "Error is below the registered threshold.",
                    "evidence_ids": ["context_known"],
                }
            ],
            "calibration_plan": "Obtain independent observations; no calibration is yet established.",
            "extrapolation_limits": [
                "No conclusions outside the observed thermal range."
            ],
        },
        "resilience": None,
        "assumptions": ["This is a schema fixture, not a validated model."],
        "unresolved_questions": [
            "Whether either mechanism fits independent observations."
        ],
    }


class ComplexSystemTests(unittest.TestCase):
    def test_practical_strict_schema_and_single_family_are_supported(self):
        data = system_spec()
        parsed = ComplexSystemSpec.parse(data)
        self.assertEqual(parsed, data)
        self.assertIsNone(parsed["resilience"])
        self.assertEqual({c["kind"] for c in parsed["components"]}, {"mechanistic"})
        self.assertEqual(ComplexSystemSpec.output_token_budget, 11000)
        schema = ComplexSystemSpec.json_schema()

        def inspect(node):
            if isinstance(node, dict):
                if node.get("type") == "object":
                    self.assertIs(node.get("additionalProperties"), False)
                    self.assertEqual(set(node["properties"]), set(node["required"]))
                self.assertNotIn("default", node)
                for value in node.values():
                    inspect(value)
            elif isinstance(node, list):
                for value in node:
                    inspect(value)

        inspect(schema)
        self.assertLess(len(json.dumps(data)), 14000)

    def test_reference_integrity_across_all_link_types(self):
        invalid_paths = [
            ("states", 0, "entity_id"),
            ("states", 0, "unit_id"),
            ("states", 0, "scale_id"),
            ("couplings", 0, "source_port"),
            ("couplings", 0, "target_port"),
        ]
        for collection, index, key in invalid_paths:
            with self.subTest(collection=collection, key=key):
                data = system_spec()
                data[collection][index][key] = "unknown"
                with self.assertRaisesRegex(ValidationError, "Unknown"):
                    ComplexSystemSpec.parse(data)
        for collection, key in [
            ("observations", "state_ids"),
            ("hypotheses", "component_ids"),
            ("hypotheses", "rivals"),
            ("experiments", "hypothesis_ids"),
            ("experiments", "endpoint_ids"),
        ]:
            with self.subTest(collection=collection, key=key):
                data = system_spec()
                data[collection][0][key] = ["unknown"]
                with self.assertRaisesRegex(ValidationError, "Unknown"):
                    ComplexSystemSpec.parse(data)

    def test_unit_conversion_dimensions_and_direction_are_host_checked(self):
        for mutation, message in [
            (
                lambda d: d["couplings"][0]["conversion"].update(offset=273.15),
                "Affine conversion",
            ),
            (
                lambda d: d["units"][1].update(dimension="length"),
                "Incompatible declared dimensions",
            ),
            (
                lambda d: d["components"][0]["ports"][0].update(direction="input"),
                "output port to an input",
            ),
            (
                lambda d: d["components"][0]["ports"][0].update(unit_id="celsius"),
                "Affine conversion",
            ),
        ]:
            with self.subTest(message=message):
                data = system_spec()
                mutation(data)
                with self.assertRaisesRegex(ValidationError, message):
                    ComplexSystemSpec.parse(data)

    def test_time_resolution_requires_explicit_compatible_alignment(self):
        data = system_spec()
        data["boundary"]["scales"].append(
            {
                "id": "minutes",
                "name": "One-minute sample",
                "step_seconds": 60.0,
                "clock": data["boundary"]["scales"][0]["clock"],
            }
        )
        data["states"][1]["scale_id"] = "minutes"
        data["components"][1]["ports"][0]["scale_id"] = "minutes"
        with self.assertRaisesRegex(ValidationError, "equal declared sampling"):
            ComplexSystemSpec.parse(data)
        data["couplings"][0]["time_alignment"] = "aggregate"
        ComplexSystemSpec.parse(data)
        data["couplings"][0]["time_alignment"] = "interpolate"
        with self.assertRaisesRegex(ValidationError, "coarser interval"):
            ComplexSystemSpec.parse(data)
        data["couplings"][0]["time_alignment"] = "aggregate"
        data["boundary"]["scales"][1]["clock"] = "another experiment"
        with self.assertRaisesRegex(ValidationError, "reference clocks"):
            ComplexSystemSpec.parse(data)

    def test_event_timing_and_numerically_extreme_conversions_cannot_pass_silently(
        self,
    ):
        data = system_spec()
        data["boundary"]["scales"][0]["step_seconds"] = None
        with self.assertRaisesRegex(ValidationError, "event_mapping"):
            ComplexSystemSpec.parse(data)
        data["couplings"][0]["time_alignment"] = "event_mapping"
        ComplexSystemSpec.parse(data)
        data = system_spec()
        data["units"][0]["scale_to_canonical"] = 1e-15
        data["couplings"][0]["conversion"]["scale"] = 1e-20
        with self.assertRaisesRegex(ValidationError, "Affine conversion"):
            ComplexSystemSpec.parse(data)
        data["units"][0]["scale_to_canonical"] = 1e-308
        data["units"][1]["scale_to_canonical"] = 1e308
        with self.assertRaisesRegex(ValidationError, "numerical precision"):
            ComplexSystemSpec.parse(data)

    def test_nonfinite_unknown_fields_duplicate_ids_and_self_rivals_fail(self):
        mutations = [
            lambda d: d["units"][0].update(scale_to_canonical=float("nan")),
            lambda d: d["units"][0].update(offset_to_canonical=float("inf")),
            lambda d: d["boundary"]["scales"][0].update(step_seconds=0.0),
            lambda d: d["components"][1]["ports"][0].update(id="source_out"),
            lambda d: d["hypotheses"][0].update(rivals=["memoryless"]),
            lambda d: d["experiments"][0].update(
                candidate_alternative_ids=["baseline"]
            ),
            lambda d: d.update(empirically_validated=True),
        ]
        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=index):
                data = system_spec()
                mutation(data)
                with self.assertRaises(ValidationError):
                    ComplexSystemSpec.parse(data)

    def test_multiple_producers_require_explicit_fusion(self):
        data = system_spec()
        coupling = copy.deepcopy(data["couplings"][0])
        coupling["id"] = "second_producer"
        data["couplings"].append(coupling)
        with self.assertRaisesRegex(ValidationError, "explicit fusion"):
            ComplexSystemSpec.parse(data)

    def test_feedback_requires_a_real_directed_cycle(self):
        data = system_spec()
        data["feedback"] = [
            {
                "id": "loop",
                "component_ids": ["source", "response_model"],
                "description": "Proposed feedback.",
                "polarity": "unknown",
                "nonlinearity": "Undetermined.",
                "stochasticity": "Undetermined.",
            }
        ]
        with self.assertRaisesRegex(ValidationError, "explicit directed cycle"):
            ComplexSystemSpec.parse(data)
        data["components"][0]["ports"].append(
            {
                "id": "source_in",
                "direction": "input",
                "state_id": "temperature",
                "unit_id": "kelvin",
                "scale_id": "seconds",
            }
        )
        data["components"][1]["ports"].append(
            {
                "id": "response_out",
                "direction": "output",
                "state_id": "response",
                "unit_id": "celsius",
                "scale_id": "seconds",
            }
        )
        data["couplings"].append(
            {
                "id": "return_coupling",
                "source_port": "response_out",
                "target_port": "source_in",
                "conversion": {"scale": 1.0, "offset": 273.15},
                "time_alignment": "synchronous",
                "alignment_rationale": "Same clock.",
                "mechanism": "Hypothesized return influence.",
            }
        )
        ComplexSystemSpec.parse(data)

    def test_resilience_is_optional_and_references_real_states(self):
        data = system_spec()
        data["resilience"] = {
            "disturbances": [
                {
                    "id": "shock",
                    "name": "Proposed shock",
                    "affected_state_ids": ["temperature"],
                    "perturbation": "Controlled perturbation.",
                }
            ],
            "recovery_observables": [
                {
                    "state_id": "response",
                    "criterion": "Return to the registered range.",
                    "observation_window": "Predeclared observation period.",
                }
            ],
        }
        ComplexSystemSpec.parse(data)
        data["resilience"]["recovery_observables"][0]["state_id"] = "unknown"
        with self.assertRaisesRegex(ValidationError, "recovery observable state"):
            ComplexSystemSpec.parse(data)

    def test_artifact_admission_checks_every_evidence_bearing_contract(self):
        data = system_spec()
        self.assertEqual(validate_artifact_references(data, ["context_known"]), data)
        for collection, key in [
            ("observations", "source_artifact_ids"),
            ("components", "evidence_ids"),
            ("hypotheses", "evidence_ids"),
        ]:
            with self.subTest(collection=collection):
                changed = copy.deepcopy(data)
                changed[collection][0][key] = ["unavailable_artifact"]
                with self.assertRaisesRegex(ValueError, "source artifact"):
                    validate_artifact_references(changed, ["context_known"])
        data["validation"]["checks"][0]["evidence_ids"] = ["unavailable_artifact"]
        with self.assertRaisesRegex(ValueError, "source artifact"):
            validate_artifact_references(data, ["context_known"])

    def test_available_observations_cannot_invent_sourceless_evidence(self):
        data = system_spec()
        data["observations"][0]["source_artifact_ids"] = []
        with self.assertRaisesRegex(ValidationError, "requires source artifacts"):
            ComplexSystemSpec.parse(data)
        data["observations"][0]["status"] = "missing"
        ComplexSystemSpec.parse(data)


if __name__ == "__main__":
    unittest.main()
