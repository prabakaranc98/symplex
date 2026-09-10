"""The investigation method as a typed, versioned object with a bounded edit surface.

A method here is not a prompt string. It is prompt fragments per role, action
priors over tools, diagnostic routing rules, tool ordering, stopping rules,
evidence-admission thresholds and revision budgets: the parts of an
investigation method a bounded editor may touch, named explicitly so that what
it may *not* touch is equally explicit.

The master specification's Loop C constraint is enforced structurally by
``FORBIDDEN_FIELDS``. A patch naming the evaluator, the objective, final labels,
task allocation, split assignment, the permission envelope, budget enforcement,
the promotion threshold or audit records is rejected with a recorded reason.
Rejections are returned, never silently dropped.

Nothing in this module installs, activates, promotes or executes anything. A
policy becomes runnable only by being handed to the existing frozen-overlay
machinery in ``symplex.agents.prompts`` and the operator-reviewed stage gates in
``symplex.infrastructure.method_registry``.
"""

import copy
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from symplex.core.contracts import Invalid, digest
from symplex.modeling.complex_system import ClosedContract, Identifier, Text

VERSION = "typed-method-policy-v1"
SCOPE = (
    "A declared, content-addressed investigation method and its allowlisted edit "
    "surface. Applying a patch establishes that the edit stayed inside the permitted "
    "surface; it establishes nothing about the method's usefulness, correctness or "
    "safety, and it neither installs nor activates anything."
)

MAX_PATCH_CHANGES = 8
MAX_PATH_SEGMENTS = 6
MAX_GENERATION = 64
MAX_FRAGMENT_CHARS = 4000

Fragment = Annotated[str, Field(min_length=1, max_length=MAX_FRAGMENT_CHARS)]

# Roles a method patch may re-instruct. Deliberately the same six that
# `symplex.agents.search_policy.MethodChange` already accepts, so a policy stays
# compatible with the existing method-candidate flow.
EDITABLE_ROLES = (
    "complexity_architect",
    "hypothesis_critic",
    "evidence_synthesist",
    "experiment_designer",
    "simulation_engineer",
    "metareasoner",
)
EditableRole = Literal[EDITABLE_ROLES]

# Uncertainty-type vocabulary mirrored from `symplex.agents.uncertainty.TYPES`.
# Copied rather than imported so this governance module cannot be broken, or
# quietly re-scoped, by a change to the agent-facing ledger it routes for.
UNCERTAINTY_TYPES = (
    "parametric",
    "structural",
    "observational",
    "semantic",
    "numerical",
    "evidential",
    "decision",
)
UncertaintyType = Literal[UNCERTAINTY_TYPES]

# Tool names a policy may reorder or re-weight. A copy of the registered
# investigation tools; `tests/test_meta_improvement.py` asserts it stays a subset
# of the live registry so drift is caught rather than assumed away.
PLANNABLE_TOOLS = (
    "acquire_bamtwoogle",
    "ask_user",
    "build_outcome",
    "compare_computation",
    "design_solution",
    "develop_candidate",
    "discover_datasets",
    "evolve_model",
    "fetch_artifact",
    "inspect_artifact",
    "inspect_context",
    "inspect_system_graph",
    "plan_experiment",
    "profile_table",
    "reframe_problem",
    "represent_system",
    "research_evidence",
    "review_hypotheses",
    "review_model",
    "run_model_code",
    "run_system_scenarios",
    "search_artifacts",
    "search_literature",
    "synthesize_evidence",
)

# Tools whose cost is dominated by evaluation or execution. Wasting one of these
# on an unverified join is the specification's own worked failure example.
EVALUATION_TOOLS = (
    "compare_computation",
    "evolve_model",
    "run_model_code",
    "run_system_scenarios",
)

MUTABLE_FIELDS = (
    "prompt_fragments",
    "action_priors",
    "diagnostic_routes",
    "tool_order",
    "stopping_rules",
    "evidence_admission",
    "revision_budget",
)

# Loop C, master specification: "The editor cannot change final labels, objective,
# evaluator, task allocation, execution permissions, budget enforcement or audit
# records." Split assignment and the promotion threshold are added because both
# decide a comparison's outcome without appearing in its measurements. Aliases are
# matched as whole path segments, so the allowlisted `revision_budget` is not
# caught by the `budget` alias.
FORBIDDEN_FIELDS = {
    "evaluator": ("evaluator", "evaluator_version", "grader", "scorer", "scoring_function", "criteria"),
    "objective": ("objective", "loss", "fitness", "target_metric", "reward"),
    "final_labels": ("final_labels", "labels", "label", "ground_truth", "outcome_labels"),
    "task_allocation": ("task_allocation", "task_ids", "task_assignment", "task_selection", "arm_assignment"),
    "split_assignment": ("split_assignment", "split", "splits", "fold", "holdout", "held_out"),
    "permission_envelope": ("permission_envelope", "permissions", "execution_permissions", "tool_permissions", "allowed_tools"),
    "budget_enforcement": ("budget_enforcement", "budget", "budgets", "cap", "caps", "resource_caps", "spend_limit"),
    "promotion_threshold": ("promotion_threshold", "promotion_margin", "promotion_alpha", "threshold", "promotion_rule"),
    "audit_records": ("audit_records", "audit", "audit_log", "ledger", "provenance_records", "seals"),
}


class DiagnosticRoute(ClosedContract):
    """Which uncertainty type triggers which check, and what the check precedes."""

    uncertainty_type: UncertaintyType
    check_tool: Identifier
    before_tools: list[Identifier] = Field(max_length=8)
    rationale: Text


class StoppingRules(ClosedContract):
    max_actions: int = Field(ge=1, le=64)
    max_consecutive_unresolving_actions: int = Field(ge=1, le=16)
    min_expected_uncertainty_reduction: float = Field(ge=0.0, le=1.0)
    stop_on_identifiability_failure: bool


class EvidenceAdmission(ClosedContract):
    min_independent_sources: int = Field(ge=1, le=8)
    require_provenance: bool
    require_identifiability_check: bool
    require_join_key_verification: bool
    max_unresolved_semantic_items: int = Field(ge=0, le=32)


class RevisionBudget(ClosedContract):
    max_program_revisions: int = Field(ge=0, le=16)
    max_method_patches_per_generation: int = Field(ge=1, le=4)
    max_repair_attempts: int = Field(ge=0, le=4)


def _address(prefix, payload):
    return prefix + "_" + digest(payload)[:24]


class MethodPolicy(ClosedContract):
    """A content-addressed investigation method with explicit parent lineage."""

    policy_id: str = Field(default="", max_length=64)
    parent_policy_id: str | None = Field(default=None, max_length=64)
    generation: int = Field(default=0, ge=0, le=MAX_GENERATION)
    label: Text
    prompt_fragments: dict[EditableRole, Fragment] = Field(max_length=len(EDITABLE_ROLES))
    action_priors: dict[Identifier, float] = Field(max_length=len(PLANNABLE_TOOLS))
    diagnostic_routes: list[DiagnosticRoute] = Field(max_length=16)
    tool_order: list[Identifier] = Field(max_length=len(PLANNABLE_TOOLS))
    stopping_rules: StoppingRules
    evidence_admission: EvidenceAdmission
    revision_budget: RevisionBudget

    @model_validator(mode="after")
    def _closed_vocabulary(self):
        unknown = set(self.action_priors) - set(PLANNABLE_TOOLS)
        if unknown:
            raise ValueError("action_priors names unregistered tools: " + ", ".join(sorted(unknown)))
        if any(not 0.0 <= v <= 1.0 for v in self.action_priors.values()):
            raise ValueError("action_priors are ordering weights in [0, 1], not probabilities")
        if len(set(self.tool_order)) != len(self.tool_order):
            raise ValueError("tool_order must not repeat a tool")
        unknown = set(self.tool_order) - set(PLANNABLE_TOOLS)
        if unknown:
            raise ValueError("tool_order names unregistered tools: " + ", ".join(sorted(unknown)))
        for route in self.diagnostic_routes:
            if route.check_tool not in PLANNABLE_TOOLS:
                raise ValueError("diagnostic_routes name an unregistered check tool")
            if set(route.before_tools) - set(PLANNABLE_TOOLS):
                raise ValueError("diagnostic_routes name an unregistered downstream tool")
        seen = [(r.uncertainty_type, r.check_tool) for r in self.diagnostic_routes]
        if len(set(seen)) != len(seen):
            raise ValueError("diagnostic_routes must not repeat an uncertainty/check pair")
        expected = _address("policy", {k: v for k, v in self.model_dump().items() if k != "policy_id"})
        if self.policy_id and self.policy_id != expected:
            raise ValueError("policy_id must be this policy's content address")
        self.policy_id = expected
        return self


class MethodPatch(ClosedContract):
    """The Loop C patch record: parent, observed failure, changed fields, cost, checks, criterion."""

    patch_id: str = Field(default="", max_length=64)
    parent_policy_id: str = Field(min_length=1, max_length=64)
    operation: Identifier
    failure_mode: Identifier
    observed_failure: Text
    evidence_trace_ids: list[str] = Field(min_length=1, max_length=64)
    changes: dict[str, Any] = Field(min_length=1, max_length=MAX_PATCH_CHANGES)
    discovery_cost: float = Field(ge=0.0)
    regression_checks: list[Text] = Field(min_length=1, max_length=8)
    promotion_criterion: Text
    expected_effect: Text
    disconfirming_result: Text

    @model_validator(mode="after")
    def _addressed(self):
        if any(not isinstance(k, str) or not k.strip() for k in self.changes):
            raise ValueError("Every change needs a nonempty field path")
        expected = _address("patch", {k: v for k, v in self.model_dump().items() if k != "patch_id"})
        if self.patch_id and self.patch_id != expected:
            raise ValueError("patch_id must be this patch's content address")
        self.patch_id = expected
        return self


def _as_policy(value):
    if isinstance(value, MethodPolicy):
        return value
    if not isinstance(value, dict):
        raise Invalid("Expected a MethodPolicy or its mapping")
    try:
        return MethodPolicy.model_validate(value)
    except Exception as exc:
        raise Invalid("Invalid method policy: " + str(exc)[:600]) from None


def _as_patch(value):
    if isinstance(value, MethodPatch):
        return value
    if not isinstance(value, dict):
        raise Invalid("Expected a MethodPatch or its mapping")
    try:
        return MethodPatch.model_validate(value)
    except Exception as exc:
        raise Invalid("Invalid method patch: " + str(exc)[:600]) from None


def _nested_keys(value, depth=0):
    if depth > 6:
        return []
    if isinstance(value, dict):
        keys = [k for k in value if isinstance(k, str)]
        for item in value.values():
            keys += _nested_keys(item, depth + 1)
        return keys
    if isinstance(value, (list, tuple)):
        return [k for item in value for k in _nested_keys(item, depth + 1)]
    return []


def forbidden_hits(path, value=None):
    """Whole-segment matches against the protected surface, path and payload alike."""
    if not isinstance(path, str) or not path.strip():
        raise Invalid("A change path must be a nonempty string")
    segments = [s for s in path.split(".") if s]
    if not segments or len(segments) > MAX_PATH_SEGMENTS:
        raise Invalid("A change path must name 1 to 6 segments")
    segments = segments + _nested_keys(value)
    hits = []
    for field, aliases in FORBIDDEN_FIELDS.items():
        if any(segment in aliases for segment in segments):
            hits.append(field)
    return sorted(set(hits))


_KEYED = {
    "prompt_fragments": set(EDITABLE_ROLES),
    "action_priors": set(PLANNABLE_TOOLS),
    "stopping_rules": set(StoppingRules.model_fields),
    "evidence_admission": set(EvidenceAdmission.model_fields),
    "revision_budget": set(RevisionBudget.model_fields),
}


def _surface_reason(path):
    segments = [s for s in path.split(".") if s]
    if segments[0] not in MUTABLE_FIELDS:
        return "outside_mutable_surface: " + segments[0]
    if len(segments) == 1:
        return None
    if len(segments) > 2:
        return "path_deeper_than_the_declared_surface: " + path
    if segments[0] not in _KEYED:
        return "field_replaced_as_a_whole_only: " + segments[0]
    if segments[1] not in _KEYED[segments[0]]:
        return "unknown_key_on_mutable_field: " + path
    return None


def baseline_policy(label="Baseline investigation method"):
    """The frozen parent every Loop C comparison starts from."""
    policy = MethodPolicy(
        label=label,
        prompt_fragments={},
        action_priors={
            "inspect_context": 0.5,
            "profile_table": 0.5,
            "represent_system": 0.6,
            "research_evidence": 0.5,
            "synthesize_evidence": 0.5,
            "plan_experiment": 0.5,
            "run_model_code": 0.5,
            "compare_computation": 0.5,
            "evolve_model": 0.4,
            "review_model": 0.4,
        },
        diagnostic_routes=[],
        tool_order=["inspect_context", "represent_system", "research_evidence",
                    "plan_experiment", "run_model_code", "compare_computation"],
        stopping_rules=StoppingRules(
            max_actions=12,
            max_consecutive_unresolving_actions=3,
            min_expected_uncertainty_reduction=0.0,
            stop_on_identifiability_failure=False,
        ),
        evidence_admission=EvidenceAdmission(
            min_independent_sources=1,
            require_provenance=True,
            require_identifiability_check=False,
            require_join_key_verification=False,
            max_unresolved_semantic_items=8,
        ),
        revision_budget=RevisionBudget(
            max_program_revisions=2,
            max_method_patches_per_generation=1,
            max_repair_attempts=1,
        ),
    )
    return {
        "version": VERSION,
        "policy": policy.model_dump(),
        "mutable_fields": list(MUTABLE_FIELDS),
        "forbidden_fields": sorted(FORBIDDEN_FIELDS),
        "scope": SCOPE,
    }


def policy_diff(parent, child):
    """Field-level differences across the mutable surface only."""
    left, right = _as_policy(parent).model_dump(), _as_policy(child).model_dump()
    changes = []
    for field in MUTABLE_FIELDS:
        before, after = left[field], right[field]
        if before == after:
            continue
        if isinstance(before, dict) and isinstance(after, dict):
            for key in sorted(set(before) | set(after)):
                if before.get(key) != after.get(key):
                    changes.append({"path": field + "." + key, "before": before.get(key), "after": after.get(key)})
        else:
            changes.append({"path": field, "before": before, "after": after})
    return {
        "version": VERSION,
        "parent_policy_id": left["policy_id"],
        "child_policy_id": right["policy_id"],
        "changes": changes,
        "changed_fields": sorted({c["path"].split(".")[0] for c in changes}),
        "scope": SCOPE,
    }


def apply_patch(parent, patch):
    """Apply an allowlisted patch, or record exactly why it was refused.

    A refusal returns ``child: None`` so a caller that ignores ``applied`` cannot
    accidentally run a rejected method. Refusal reasons are preserved.
    """
    parent = _as_policy(parent)
    patch = _as_patch(patch)
    rejected = []
    if patch.parent_policy_id != parent.policy_id:
        rejected.append({"path": None, "reason": "patch_parent_is_not_this_policy",
                         "detail": patch.parent_policy_id + " != " + parent.policy_id})
    for path, value in sorted(patch.changes.items()):
        hits = forbidden_hits(path, value)
        if hits:
            rejected.append({"path": path, "reason": "forbidden_field", "forbidden_fields": hits,
                             "detail": "Loop C forbids editing " + ", ".join(hits)})
            continue
        reason = _surface_reason(path)
        if reason:
            rejected.append({"path": path, "reason": reason.split(":")[0], "detail": reason})
    if rejected:
        return {
            "version": VERSION, "applied": False, "child": None, "diff": None,
            "parent_policy_id": parent.policy_id, "patch_id": patch.patch_id,
            "rejected": rejected,
            "scope": SCOPE + " This patch was refused; the parent policy is unchanged and retained.",
        }
    data = copy.deepcopy(parent.model_dump())
    data["policy_id"] = ""
    data["parent_policy_id"] = parent.policy_id
    data["generation"] = parent.generation + 1
    data["label"] = patch.operation + " from " + parent.policy_id
    for path, value in sorted(patch.changes.items()):
        segments = [s for s in path.split(".") if s]
        if len(segments) == 1:
            data[segments[0]] = copy.deepcopy(value)
        else:
            data[segments[0]] = dict(data[segments[0]])
            data[segments[0]][segments[1]] = copy.deepcopy(value)
    try:
        child = MethodPolicy.model_validate(data)
    except Exception as exc:
        return {
            "version": VERSION, "applied": False, "child": None, "diff": None,
            "parent_policy_id": parent.policy_id, "patch_id": patch.patch_id,
            "rejected": [{"path": None, "reason": "invalid_child_policy", "detail": str(exc)[:600]}],
            "scope": SCOPE + " This patch was refused; the parent policy is unchanged and retained.",
        }
    return {
        "version": VERSION, "applied": True, "child": child.model_dump(),
        "diff": policy_diff(parent, child), "parent_policy_id": parent.policy_id,
        "patch_id": patch.patch_id, "rejected": [],
        "scope": SCOPE,
    }


def to_prompt_overlay(policy, base=None):
    """Render a policy's prompt fragments as a frozen overlay.

    ``result["overlay"]`` is exactly the mapping ``symplex.agents.prompts.prompt_overlay``
    accepts: role name to complete instruction text, keys drawn from the supplied
    base snapshot. The surrounding dict carries the scope statement the house
    requires; the overlay itself carries nothing but instructions.
    """
    policy = _as_policy(policy)
    if base is None:
        from symplex.agents.prompts import snapshot

        base = snapshot()
    if not isinstance(base, dict) or not base:
        raise Invalid("A prompt overlay needs a nonempty base snapshot")
    unknown = set(policy.prompt_fragments) - set(base)
    if unknown:
        raise Invalid("Policy fragments name roles absent from the base: " + ", ".join(sorted(unknown)))
    overlay = {}
    for role, fragment in sorted(policy.prompt_fragments.items()):
        text = str(base[role]).rstrip() + "\n\nEvaluated method addition (" + policy.policy_id + "):\n" + fragment.strip()
        if not text.strip() or len(text) > 16000:
            raise Invalid("Overlaid instruction for " + role + " is empty or exceeds the frozen bound")
        overlay[role] = text
    return {
        "version": VERSION,
        "policy_id": policy.policy_id,
        "overlay": overlay,
        "roles": sorted(overlay),
        "unchanged_roles": sorted(set(base) - set(overlay)),
        "scope": (
            "A frozen prompt overlay for one evaluation context. Only prompt fragments "
            "cross into the runnable arm; action priors, routing, stopping rules, "
            "admission thresholds and revision budgets are enforced by host call sites, "
            "not by this text. Rendering an overlay installs nothing."
        ),
    }
