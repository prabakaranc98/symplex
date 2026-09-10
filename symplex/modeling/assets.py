"""Component contracts make unsupported training/merging explicit."""

from symplex.core.contracts import Invalid

ASSETS = [
    {
        "id": "astra",
        "provider": "openai",
        "model": "gpt-6-astra",
        "revision": "gpt-6-astra",
        "task": "solution_synthesis",
        "input": "structured evidence",
        "output": "typed proposal",
        "units": None,
        "horizon": None,
        "adaptation": [],
        "status": "api",
        "limitations": "No accessible weights or weight merging",
        "license": "OpenAI account terms",
    },
    {
        "id": "p1_calibrator",
        "provider": "scientific_python",
        "model": "bounded_scalar_shrinkage",
        "revision": "p1-trusted-v1",
        "task": "binary_forecast",
        "input": "price proxy",
        "output": "probability",
        "units": "probability",
        "horizon": "pack cutoff",
        "adaptation": ["fit_training_only"],
        "status": "executable",
        "limitations": "One pooled parameter; no broad calibration guarantee",
        "license": "Project license",
    },
    {
        "id": "hf_extension",
        "provider": "huggingface",
        "model": None,
        "revision": None,
        "task": "specialist model selected after data inspection",
        "input": None,
        "output": None,
        "units": None,
        "horizon": None,
        "adaptation": [],
        "status": "dependency_gap",
        "limitations": "Pin model revision, model card, tokenizer, license and evaluation before loading",
        "license": "unknown",
    },
]


def compatible(left, right, operation="ensemble"):
    if operation == "weight_merge":
        raise Invalid(
            "unsupported_operation: no compatible local checkpoints or merge backend configured"
        )
    if operation != "ensemble" or any(
        left.get(k) is None or left.get(k) != right.get(k)
        for k in ("task", "output", "units", "horizon")
    ):
        raise Invalid("Incompatible target, units, horizon or operation")
    return True


def training_plan(asset):
    if "fit_training_only" not in asset["adaptation"]:
        return {
            "status": "dependency_gap",
            "reason": "No supported adaptation backend for this asset",
        }
    return {
        "status": "supported",
        "objective": "training Brier",
        "trainable_parameters": ["shrinkage"],
        "optimizer": "bounded closed-form least squares",
        "stopping_rule": "single fit",
        "checkpoint": "immutable run artifact",
        "rollback": "retain parent program",
        "protected": ["development labels during fitting", "confirmation labels"],
    }
