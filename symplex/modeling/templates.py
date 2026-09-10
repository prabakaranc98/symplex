"""Trusted, reviewable grammar: raw proxies, isotonic constraints and fitted shrinkage."""

from symplex.core.contracts import Invalid, digest, number
from symplex.evidence.admission import relation_key, timestamp

TEMPLATE_VERSION = "p1-trusted-v1"
DEFAULT_PROGRAM = {
    "template": TEMPLATE_VERSION,
    "projection": False,
    "calibrator": False,
    "age_weight": 0.0,
}


def validate(program):
    if set(program) != set(DEFAULT_PROGRAM) or program["template"] != TEMPLATE_VERSION:
        raise Invalid("Unknown component or incompatible backend")
    if (
        type(program["projection"]) is not bool
        or type(program["calibrator"]) is not bool
    ):
        raise Invalid("Component flags must be booleans")
    number(program["age_weight"], 0, 1)
    return program


def compile_patch(parent, patch):
    result = dict(validate(parent))
    if patch.operation == "enable_projection":
        result["projection"] = True
    elif patch.operation == "fit_calibrator":
        result["calibrator"] = True
    elif patch.operation == "age_shrinkage":
        result["age_weight"] = patch.value
    else:
        raise Invalid("Unknown component")
    if result == parent:
        raise Invalid("Patch does not change the parent")
    return validate(result)


def project(rows, predictions):
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(relation_key(row), []).append(i)
    out = list(predictions)
    for indexes in groups.values():
        indexes.sort(key=lambda i: rows[i]["threshold"])
        # PAVA for non-increasing probabilities, including equality at equal thresholds.
        blocks = []
        for i in indexes:
            if blocks and rows[blocks[-1][0][-1]]["threshold"] == rows[i]["threshold"]:
                blocks[-1][0].append(i)
                blocks[-1][1] += predictions[i]
            else:
                blocks.append([[i], predictions[i]])
            while len(blocks) > 1 and blocks[-2][1] / len(blocks[-2][0]) < blocks[-1][
                1
            ] / len(blocks[-1][0]):
                right = blocks.pop()
                blocks[-1][0].extend(right[0])
                blocks[-1][1] += right[1]
        for members, total in blocks:
            for i in members:
                out[i] = total / len(members)
    return out


def run(program, training, inputs):
    validate(program)
    if not training or not inputs:
        raise Invalid("Empty training or prediction inputs")
    prior = sum(r["label"] for r in training) / len(training)
    alpha = 0.0
    if program["calibrator"]:
        # Fit one scalar only on training labels, never selection or confirmation labels.
        numerator = sum(
            (r["label"] - r["price"]) * (prior - r["price"]) for r in training
        )
        denominator = sum((prior - r["price"]) ** 2 for r in training)
        alpha = max(0.0, min(1.0, numerator / denominator)) if denominator else 0.0
    values = []
    for row in inputs:
        age_days = max(
            0,
            (timestamp(row["cutoff"]) - timestamp(row["observed_at"])).total_seconds()
            / 86400,
        )
        weight = 1 - (1 - alpha) * (1 - program["age_weight"] * min(1, age_days))
        values.append((1 - weight) * row["price"] + weight * prior)
    if program["projection"]:
        values = project(inputs, values)
    for value in values:
        number(value, 0, 1)
    return {
        "predictions": {r["id"]: p for r, p in zip(inputs, values)},
        "fitted": {
            "prior": prior,
            "shrinkage": alpha,
            "training_digest": digest(training),
        },
        "template_version": TEMPLATE_VERSION,
    }
