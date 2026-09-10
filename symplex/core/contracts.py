"""Small, closed mutation surface. Scientific validity is checked separately."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass


class Invalid(ValueError):
    pass


class Unsupported(Invalid):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def closed(data, keys):
    if not isinstance(data, dict) or set(data) != set(keys):
        raise Invalid("Fields must be exactly: " + ", ".join(keys))


def number(value, low=None, high=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise Invalid("Expected a finite number")
    if low is not None and value < low or high is not None and value > high:
        raise Invalid("Number outside permitted bounds")
    return float(value)


def nonempty(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 12000:
        raise Invalid("Expected a nonempty string of at most 12000 characters")
    return value


def string_list(value):
    if not isinstance(value, list):
        raise Invalid("Expected a string list")
    return [nonempty(x) for x in value]


def schema(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


TEXT = {"type": "string"}
TEXTS = {"type": "array", "items": TEXT}


@dataclass(frozen=True)
class ProgramPatch:
    parent_id: str
    operation: str
    value: float
    evidence_ids: list[str]
    expectation: str
    disconfirmation: str

    @classmethod
    def parse(cls, data):
        closed(data, cls.__annotations__)
        if data["operation"] not in (
            "enable_projection",
            "fit_calibrator",
            "age_shrinkage",
        ):
            raise Unsupported(
                "unsupported_operation: mutation is outside the template grammar"
            )
        number(data["value"], 0, 1)
        for key in ("parent_id", "expectation", "disconfirmation"):
            nonempty(data[key])
        string_list(data["evidence_ids"])
        return cls(**data)

    @staticmethod
    def json_schema():
        return schema(
            {
                "parent_id": TEXT,
                "operation": {
                    "type": "string",
                    "enum": ["enable_projection", "fit_calibrator", "age_shrinkage"],
                },
                "value": {"type": "number"},
                "evidence_ids": TEXTS,
                "expectation": TEXT,
                "disconfirmation": TEXT,
            }
        )


@dataclass(frozen=True)
class ActionProposal:
    action_type: str
    target_ids: list[str]
    uncertainty_addressed: str
    preconditions: list[str]
    tool_plan: list[str]
    estimated_cost: float
    expected_observable_change: str
    completion_condition: str
    stop_condition: str

    @classmethod
    def parse(cls, data):
        closed(data, cls.__annotations__)
        if data["action_type"] not in (
            "inspect_definitions",
            "ablate_projection",
            "mutate_program",
            "stop",
        ):
            raise Unsupported("unsupported_operation: research action")
        number(data["estimated_cost"], 0)
        for key in ("target_ids", "preconditions", "tool_plan"):
            string_list(data[key])
        for key in (
            "uncertainty_addressed",
            "expected_observable_change",
            "completion_condition",
            "stop_condition",
        ):
            nonempty(data[key])
        return cls(**data)

    @staticmethod
    def json_schema():
        return schema(
            {
                key: (
                    {"type": "number"}
                    if key == "estimated_cost"
                    else TEXTS
                    if key in ("target_ids", "preconditions", "tool_plan")
                    else {
                        "type": "string",
                        "enum": [
                            "inspect_definitions",
                            "ablate_projection",
                            "mutate_program",
                            "stop",
                        ],
                    }
                    if key == "action_type"
                    else TEXT
                )
                for key in ActionProposal.__annotations__
            }
        )


@dataclass(frozen=True)
class MethodPatch:
    parent_id: str
    diagnostic_first: bool
    observed_failure: str
    promotion_margin: float

    @classmethod
    def parse(cls, data):
        closed(data, cls.__annotations__)
        nonempty(data["parent_id"])
        nonempty(data["observed_failure"])
        if type(data["diagnostic_first"]) is not bool:
            raise Invalid("diagnostic_first must be boolean")
        # The threshold is frozen by the host; a proposer cannot loosen it.
        if data["promotion_margin"] != 0.0:
            raise Invalid("Promotion margin is protected at 0.0")
        return cls(**data)

    @staticmethod
    def json_schema():
        return schema(
            {
                "parent_id": TEXT,
                "diagnostic_first": {"type": "boolean"},
                "observed_failure": TEXT,
                "promotion_margin": {"type": "number", "enum": [0]},
            }
        )


@dataclass(frozen=True)
class ProblemDNA:
    beneficiary: str
    decision: str
    objective: str
    actors: list[str]
    constraints: list[str]
    assumptions: list[str]
    missing_evidence: list[str]

    @classmethod
    def parse(cls, data):
        closed(data, cls.__annotations__)
        for key in ("beneficiary", "decision", "objective"):
            nonempty(data[key])
        for key in ("actors", "constraints", "assumptions", "missing_evidence"):
            string_list(data[key])
        return cls(**data)

    @staticmethod
    def json_schema():
        return schema(
            {
                k: TEXT if k in ("beneficiary", "decision", "objective") else TEXTS
                for k in ProblemDNA.__annotations__
            }
        )


def record(obj):
    return asdict(obj)
