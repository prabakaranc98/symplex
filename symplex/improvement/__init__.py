"""Improving how the agent improves, when no single metric is authoritative.

AlphaEvolve improves executable candidates under a machine-gradeable evaluator: it
can hill-climb a scalar because the scalar is complete for its task. This package
exists because that assumption does not hold here. Symplex investigates uncertain
worlds where the evaluator is incomplete, competing explanations stay viable,
simulations can mislead, and every stronger claim has to be earned through
evidence, comparison and protected evaluation. Self-improvement without an
authoritative metric is the distinctive technical problem, and everything in these
modules follows from it.

The layers:

* ``policy``    — the investigation method as a typed, versioned, content-addressed
                  object with an explicit allowlisted mutable surface and an equally
                  explicit ``FORBIDDEN_FIELDS`` denylist enforcing the master
                  specification's Loop C constraint.
* ``proposer``  — recurring failure modes computed structurally from recorded traces,
                  and a deterministic patch selected from a typed grammar. The
                  trace-sampling strategy is a recorded, swappable parameter because
                  it is itself the Loop D target.
* ``protocol``  — protected evaluation: a board of non-redundant proxies with primary,
                  protected and diagnostic roles plus declared unmeasured dimensions,
                  paired replay, a promotion rule that answers ``inconclusive`` rather
                  than pass, an always-valid sequential test with an asymmetric budget,
                  and a power report that refuses underpowered comparisons.
* ``goodhart``  — structural detection of a candidate winning the proxy while losing
                  the thing: divergence, specification gaming, proxy-correlation drift
                  and task-family concentration.
* ``meta``      — Loop D: improvement productivity per unit compute on fresh task
                  families, a comparison of improvement procedures at equal budget,
                  and a ladder of claims that cannot be climbed out of order.

House rules this package keeps. Every module here is deterministic and host-owned:
no model call, no network, no subprocess, no filesystem writes. Governance code that
the thing being evaluated could influence would be worthless. Every public function
returns a mapping carrying a ``scope`` string that states what is *not* established.
Bad input raises ``symplex.core.contracts.Invalid``. Rejected patches, refused
claims, excluded tasks and unclassified failures are returned with their reasons and
are never silently dropped.

Nothing here auto-promotes. A ``promote`` verdict means a candidate is eligible to be
submitted to the existing stage gates. Human operator review, the scoped canary and
rollback in ``symplex.infrastructure.method_registry`` remain the only path to
activation, and this package deliberately holds no capability to shortcut it.
"""

from symplex.improvement.goodhart import (
    divergence_report,
    goodhart_report,
    overfitting_to_task_family,
    proxy_correlation_drift,
    specification_gaming_checks,
)
from symplex.improvement.meta import (
    CLAIM_LADDER,
    MetaPolicy,
    compare_meta_policies,
    effective_promotion_alpha,
    meta_generation,
    rsi_evidence_report,
)
from symplex.improvement.policy import (
    EDITABLE_ROLES,
    FORBIDDEN_FIELDS,
    MUTABLE_FIELDS,
    PLANNABLE_TOOLS,
    EvidenceAdmission,
    DiagnosticRoute,
    MethodPatch,
    MethodPolicy,
    RevisionBudget,
    StoppingRules,
    apply_patch,
    baseline_policy,
    forbidden_hits,
    policy_diff,
    to_prompt_overlay,
)
from symplex.improvement.proposer import (
    FAILURE_MODES,
    PATCH_GRAMMAR,
    SAMPLING_STRATEGIES,
    InvestigationTrace,
    TraceStep,
    failure_taxonomy,
    propose_patch,
    sample_traces,
)
from symplex.improvement.protocol import (
    ArmRecord,
    EvaluationBoard,
    ProxySpec,
    ReplayTask,
    describe_board,
    paired_replay,
    power_report,
    promotion_rule,
    sequential_verdict,
)

VERSION = "improvement-layer-v1"

__all__ = [
    "VERSION",
    # policy
    "MethodPolicy", "MethodPatch", "DiagnosticRoute", "StoppingRules", "EvidenceAdmission",
    "RevisionBudget", "FORBIDDEN_FIELDS", "MUTABLE_FIELDS", "EDITABLE_ROLES", "PLANNABLE_TOOLS",
    "baseline_policy", "apply_patch", "policy_diff", "to_prompt_overlay", "forbidden_hits",
    # proposer
    "InvestigationTrace", "TraceStep", "FAILURE_MODES", "SAMPLING_STRATEGIES", "PATCH_GRAMMAR",
    "failure_taxonomy", "propose_patch", "sample_traces",
    # protocol
    "ProxySpec", "EvaluationBoard", "ArmRecord", "ReplayTask", "describe_board",
    "paired_replay", "promotion_rule", "sequential_verdict", "power_report",
    # goodhart
    "divergence_report", "specification_gaming_checks", "proxy_correlation_drift",
    "overfitting_to_task_family", "goodhart_report",
    # meta
    "MetaPolicy", "CLAIM_LADDER", "meta_generation", "compare_meta_policies",
    "rsi_evidence_report", "effective_promotion_alpha",
]
