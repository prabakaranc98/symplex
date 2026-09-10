"""Bounded leaf repairs for rejected, unfrozen proposals; callers revalidate the contract."""

import copy
import json
import math
import re
from dataclasses import dataclass
from typing import ClassVar

from symplex.core.contracts import Invalid, canonical, closed, schema


MAX_OPERATIONS = 12
MAX_PATCH_CHARS = 16000
MAX_PROPOSAL_CHARS = 1000000
MAX_VALUE_CHARS = 4000
MAX_PATH_CHARS = 500
MAX_REASON_CHARS = 500


def _pointer(path):
    if not isinstance(path, str) or not path.startswith("/") or len(path) > MAX_PATH_CHARS:
        raise Invalid("Repair path must be a bounded JSON Pointer to an existing leaf; root replacement is forbidden")
    tokens = path[1:].split("/")
    if len(tokens) > 32 or any(re.search(r"~(?:[^01]|$)", token) for token in tokens):
        raise Invalid("Repair path has invalid JSON Pointer escaping or excessive depth")
    return tuple(token.replace("~1", "/").replace("~0", "~") for token in tokens)


def _scalar_json(text):
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_VALUE_CHARS:
        raise Invalid("Repair value_json must be a bounded JSON scalar")

    def reject_constant(value):
        raise Invalid("Repair values must be finite JSON values")

    try:
        value = json.loads(text, parse_constant=reject_constant)
    except (ValueError, RecursionError) as exc:
        raise Invalid("Repair value_json must contain one valid finite JSON scalar") from exc
    if isinstance(value, (dict, list)) or isinstance(value, float) and not math.isfinite(value):
        raise Invalid("Repair replacements must be finite scalar leaves, never containers")
    return value


def _json_tree(value, depth=0):
    if depth > 64:
        raise Invalid("Proposal exceeds the repair nesting bound")
    if type(value) is dict:
        for key, child in value.items():
            if not isinstance(key, str):
                raise Invalid("Repair requires JSON object keys")
            _json_tree(child, depth + 1)
    elif type(value) is list:
        for child in value:
            _json_tree(child, depth + 1)
    elif type(value) not in (str, int, float, bool, type(None)):
        raise Invalid("Repair requires JSON proposal data")
    elif isinstance(value, float) and not math.isfinite(value):
        raise Invalid("Repair requires finite JSON proposal data")


def _bounded_json(value, maximum):
    try:
        _json_tree(value)
        if len(canonical(value)) > maximum:
            raise Invalid("Repair data exceeds the host size bound")
    except (ValueError, TypeError, RecursionError) as exc:
        raise Invalid("Repair requires bounded finite JSON data") from exc


@dataclass(frozen=True)
class ProposalRepair:
    operations: list[dict[str, str]]
    output_token_budget: ClassVar[int] = 1800

    @classmethod
    def parse(cls, data):
        closed(data, ("operations",))
        operations = data["operations"]
        if not isinstance(operations, list) or not 0 <= len(operations) <= MAX_OPERATIONS:
            raise Invalid("A proposal repair permits 0 to 12 leaf replacements; empty operations abstain")
        seen = set()
        for operation in operations:
            closed(operation, ("path", "value_json", "reason"))
            path = _pointer(operation["path"])
            if path in seen:
                raise Invalid("Duplicate repair paths are forbidden")
            seen.add(path)
            _scalar_json(operation["value_json"])
            reason = operation["reason"]
            if not isinstance(reason, str) or not reason.strip() or len(reason) > MAX_REASON_CHARS:
                raise Invalid("Each repair requires a concise reason")
        _bounded_json(data, MAX_PATCH_CHARS)
        return cls(copy.deepcopy(operations))

    @staticmethod
    def json_schema():
        operation = schema({
            "path": {"type": "string", "maxLength": MAX_PATH_CHARS,
                     "description": "JSON Pointer to an existing scalar leaf; ~0 means tilde and ~1 means slash."},
            "value_json": {"type": "string", "maxLength": MAX_VALUE_CHARS,
                           "description": "The replacement scalar encoded as JSON; no objects or arrays."},
            "reason": {"type": "string", "maxLength": MAX_REASON_CHARS},
        })
        return schema({"operations": {"type": "array", "items": operation,
                                      "minItems": 0, "maxItems": MAX_OPERATIONS}})


def _existing(container, token):
    if isinstance(container, dict):
        if token not in container:
            raise Invalid("Repair cannot add a missing field")
        return token
    if isinstance(container, list):
        if not re.fullmatch(r"0|[1-9][0-9]*", token):
            raise Invalid("Repair array indices must identify existing elements")
        index = int(token)
        if index >= len(container):
            raise Invalid("Repair array index is out of bounds")
        return index
    raise Invalid("Repair path cannot traverse a scalar leaf")


def apply_repair(original, patch):
    """Copy and replace existing scalar leaves only; final contract.parse is mandatory.

    This helper is for rejected proposals before freezing. It cannot authorize a
    change to a stored/frozen artifact or relax the target contract's invariants.
    """
    _bounded_json(original, MAX_PROPOSAL_CHARS)
    parsed = ProposalRepair.parse({"operations": patch.operations} if isinstance(patch, ProposalRepair) else patch)
    if not parsed.operations:
        raise Invalid("No justified leaf repair was proposed; empty operations abstain")
    repaired = copy.deepcopy(original)
    for operation in parsed.operations:
        tokens = _pointer(operation["path"])
        container = repaired
        for token in tokens[:-1]:
            container = container[_existing(container, token)]
        key = _existing(container, tokens[-1])
        previous = container[key]
        if isinstance(previous, (dict, list)):
            raise Invalid("Repair cannot replace a container")
        replacement = _scalar_json(operation["value_json"])
        if canonical(previous) == canonical(replacement):
            raise Invalid("Identity repair leaves the proposed value unchanged")
        container[key] = replacement
    _bounded_json(repaired, MAX_PROPOSAL_CHARS)
    return repaired
