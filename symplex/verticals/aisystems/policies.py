"""The decision object under study: five rival routing policies and six verification rules.

Every policy is a pure function of observable state. It sees a RequestView (no difficulty, no
correctness, no safety label) and a SystemView (queue depths, utilisation, spend, recent
latency). It never sees ground truth, and `audit_policy_observability` enforces that by probing
each policy with a view that raises rather than answering for a hidden field.

A policy that could see true difficulty would be an oracle, and comparing an oracle to a real
router flatters the router's designer, not the router.
"""

import math
from dataclasses import dataclass, replace
from typing import Annotated, Literal

from pydantic import Field

from symplex.core.contracts import Invalid
from symplex.modeling.complex_system import ClosedContract, Text
from symplex.verticals.aisystems.queueing import RouteDecision, SystemView
from symplex.verticals.aisystems.workload import (
    HIDDEN_FIELDS,
    OBSERVABLE_FIELDS,
    RequestView,
    crn_uniform,
)

TIER_LADDER = ("cheap", "mid", "frontier")
ROUTING_POLICY_NAMES = (
    "cheapest_first",
    "complexity_cascade",
    "verifier_gated",
    "uncertainty_aware",
    "queue_aware_adaptive",
)
VERIFICATION_POLICY_NAMES = (
    "always",
    "never",
    "sampled",
    "uncertainty_triggered",
    "cost_bounded",
    "human_escalation_threshold",
)
SYSTEM_VIEW_FIELDS = (
    "now",
    "queue_depth",
    "busy_servers",
    "utilisation",
    "headroom",
    "recent_shed_rate",
    "recent_timeout_rate",
    "recent_p95_latency_seconds",
    "admitted",
    "completed",
    "spend_usd",
    "budget_remaining_usd",
    "slo_seconds",
)

SCOPE = (
    "Declared decision rules evaluated inside a simulator. A policy's advantage here is "
    "conditional on the declared workload, service-time and failure models; it is not a "
    "measurement of any real deployed router."
)


def _step_up(tier):
    index = TIER_LADDER.index(tier)
    return TIER_LADDER[index + 1] if index + 1 < len(TIER_LADDER) else None


def _step_down(tier):
    index = TIER_LADDER.index(tier)
    return TIER_LADDER[index - 1] if index > 0 else tier


class VerificationPolicy(ClosedContract):
    """How much checking to buy, decided before the answer exists.

    Verification is not free: it costs a verifier call, adds latency, and consumes the same
    concurrency the models need. Allocating it is a decision, not a default.
    """

    kind: Literal[
        "always",
        "never",
        "sampled",
        "uncertainty_triggered",
        "cost_bounded",
        "human_escalation_threshold",
    ]
    rate: Annotated[float, Field(ge=0.0, le=1.0)]
    uncertainty_threshold: Annotated[float, Field(ge=0.0, le=1.0)]
    cost_floor_usd: Annotated[float, Field(ge=0.0, le=1000.0)]
    human_threshold: Annotated[float, Field(ge=0.0, le=1.0)]
    salt: Annotated[int, Field(ge=0, le=2**31)]
    rationale: Text

    def allocate(self, view, system):
        """Return (verify, escalate_human) from observable state only."""
        if self.kind == "always":
            return True, view.difficulty_hint >= self.human_threshold
        if self.kind == "never":
            return False, False
        if self.kind == "sampled":
            return crn_uniform(self.salt, view.id, "verify_sample") < self.rate, False
        if self.kind == "uncertainty_triggered":
            return view.difficulty_hint >= self.uncertainty_threshold, False
        if self.kind == "cost_bounded":
            affordable = system.budget_remaining_usd > self.cost_floor_usd
            return bool(affordable and view.difficulty_hint >= self.uncertainty_threshold), False
        return (
            True,
            bool(
                view.difficulty_hint >= self.human_threshold
                and view.value_usd >= self.rate
            ),
        )


def verification_policy(kind, **overrides):
    """Build one of the six named verification-allocation rules."""
    if kind not in VERIFICATION_POLICY_NAMES:
        raise Invalid("Unknown verification policy: " + str(kind))
    defaults = {
        "always": {"rationale": "Check every answer; buy containment with cost and latency."},
        "never": {"rationale": "Never check; the cheapest and least contained option."},
        "sampled": {
            "rate": 0.30,
            "rationale": "Check a fixed deterministic sample; bounded cost, partial containment.",
        },
        "uncertainty_triggered": {
            "uncertainty_threshold": 0.45,
            "rationale": "Check when the observable difficulty hint is high.",
        },
        "cost_bounded": {
            "uncertainty_threshold": 0.40,
            "cost_floor_usd": 0.0,
            "rationale": "Check while budget remains, preferring the requests that look hard.",
        },
        "human_escalation_threshold": {
            "human_threshold": 0.65,
            "rate": 0.60,
            "rationale": "Check everything and send high-hint, high-value requests to a person.",
        },
    }[kind]
    spec = {
        "kind": kind,
        "rate": 0.0,
        "uncertainty_threshold": 1.0,
        "cost_floor_usd": 0.0,
        "human_threshold": 1.0,
        "salt": 4242,
        "rationale": "",
    }
    spec.update(defaults)
    spec.update(overrides)
    return VerificationPolicy.model_validate(spec)


@dataclass(frozen=True)
class RoutingPolicy:
    """One rival router. `decide` picks the plan; `escalate` answers a verifier rejection."""

    name: str
    verification: VerificationPolicy
    cheap_threshold: float = 0.35
    mid_threshold: float = 0.62
    max_escalations: int = 2
    congestion_threshold: float = 0.85
    queue_pressure_requests: int = 12
    confidence_threshold: float = 0.55
    decline_shed_rate: float = 0.55
    description: str = ""

    # -- helpers ----------------------------------------------------------------

    def _base_tier(self, hint):
        if hint < self.cheap_threshold:
            return "cheap"
        if hint < self.mid_threshold:
            return "mid"
        return "frontier"

    def _congested(self, system, tier):
        node = "model_" + tier
        utilisation = system.utilisation.get(node, 0.0)
        depth = system.queue_depth.get(node, 0)
        return utilisation >= self.congestion_threshold and depth >= self.queue_pressure_requests

    # -- interface --------------------------------------------------------------

    def decide(self, view, system):
        verify, human = self.verification.allocate(view, system)
        if self.name == "cheapest_first":
            tier = "cheap"
        elif self.name == "complexity_cascade":
            tier = self._base_tier(view.difficulty_hint)
        elif self.name == "verifier_gated":
            tier = "mid" if view.difficulty_hint >= self.cheap_threshold else "cheap"
            verify = True
        elif self.name == "uncertainty_aware":
            tier = self._base_tier(view.difficulty_hint)
            if abs(view.difficulty_hint - self.cheap_threshold) < 0.08:
                tier = "mid"
            verify = verify or view.difficulty_hint >= self.confidence_threshold
        elif self.name == "queue_aware_adaptive":
            tier = self._base_tier(view.difficulty_hint)
            while tier != "cheap" and self._congested(system, tier):
                tier = _step_down(tier)
            if system.recent_p95_latency_seconds > 1.5 * system.slo_seconds:
                tier = "cheap"
                verify = False
        else:
            raise Invalid("Unknown routing policy: " + str(self.name))

        admit = True
        if self.name == "queue_aware_adaptive":
            overloaded = (
                system.recent_shed_rate > self.decline_shed_rate
                and system.utilisation.get("model_cheap", 0.0) > 0.98
                and system.headroom.get("model_mid", 1.0) <= 0.0
            )
            admit = not (overloaded and view.value_usd < 0.2)
            if system.budget_remaining_usd <= 0.0:
                verify = False
        return RouteDecision(
            model_tier=tier,
            use_cache=True,
            use_retrieval=view.retrieval_requested,
            use_tool=view.tool_requested,
            verify=bool(verify),
            escalate_human=bool(human),
            max_escalations=self.max_escalations,
            admit=bool(admit),
            rationale=self.name,
        )

    def escalate(self, view, system, feedback):
        """Answer a verifier rejection. Returning None means 'live with this answer'."""
        tier = feedback["tier"]
        confidence = feedback.get("confidence", 0.5)
        if self.name == "cheapest_first":
            return "frontier" if tier != "frontier" else None
        if self.name == "complexity_cascade":
            return _step_up(tier)
        if self.name == "verifier_gated":
            return _step_up(tier)
        if self.name == "uncertainty_aware":
            if confidence >= self.confidence_threshold and feedback.get("round", 1) > 1:
                return None
            return _step_up(tier)
        if self.name == "queue_aware_adaptive":
            target = _step_up(tier)
            if target is None:
                return None
            if self._congested(system, target) or system.headroom.get(
                "model_" + target, 1.0
            ) <= 0.0:
                return None
            if system.recent_p95_latency_seconds > 1.5 * system.slo_seconds:
                return None
            return target
        raise Invalid("Unknown routing policy: " + str(self.name))

    def describe(self):
        return {
            "name": self.name,
            "description": self.description,
            "verification": self.verification.kind,
            "verification_rationale": self.verification.rationale,
            "cheap_threshold": self.cheap_threshold,
            "mid_threshold": self.mid_threshold,
            "max_escalations": self.max_escalations,
            "reads_only": {
                "request": list(OBSERVABLE_FIELDS),
                "system": list(SYSTEM_VIEW_FIELDS),
            },
            "never_reads": list(HIDDEN_FIELDS),
            "scope": SCOPE,
        }


_ROUTING_DEFAULTS = {
    "cheapest_first": {
        "verification": ("sampled", {"rate": 0.30}),
        "description": "Always start at the smallest model; on rejection jump straight to the "
        "frontier tier regardless of how loaded it is.",
    },
    "complexity_cascade": {
        "verification": ("uncertainty_triggered", {"uncertainty_threshold": 0.45}),
        "description": "Pick a starting tier from the observable difficulty hint, then climb "
        "the ladder one rung at a time.",
    },
    "verifier_gated": {
        "verification": ("always", {"human_threshold": 0.78}),
        "description": "Check every answer and escalate on rejection; buys containment with "
        "cost, latency and verifier concurrency.",
    },
    "uncertainty_aware": {
        "verification": ("uncertainty_triggered", {"uncertainty_threshold": 0.38}),
        "description": "Route on the hint, verify when uncertain, and stop escalating once "
        "the reported confidence is high.",
    },
    "queue_aware_adaptive": {
        "verification": ("cost_bounded", {"uncertainty_threshold": 0.42}),
        "description": "Route on the hint but read the queues: downgrade the tier and refuse "
        "to escalate into a saturated station.",
    },
}


def build_policy(name, verification=None, **overrides):
    """Construct one rival router, optionally with a different verification-allocation rule."""
    if name not in ROUTING_POLICY_NAMES:
        raise Invalid("Unknown routing policy: " + str(name))
    spec = _ROUTING_DEFAULTS[name]
    kind, defaults = spec["verification"]
    if verification is None:
        rule = verification_policy(kind, **defaults)
    elif isinstance(verification, VerificationPolicy):
        rule = verification
    else:
        rule = verification_policy(verification)
    policy = RoutingPolicy(
        name=name, verification=rule, description=spec["description"]
    )
    if overrides:
        policy = replace(policy, **overrides)
    return policy


def default_policies():
    """The five rivals, in the order the product owner named them."""
    return [build_policy(name) for name in ROUTING_POLICY_NAMES]


class _ObservationProbe:
    """A view that answers observable fields and raises loudly on ground truth."""

    def __init__(self, target, allowed, forbidden, log):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_allowed", frozenset(allowed))
        object.__setattr__(self, "_forbidden", frozenset(forbidden))
        object.__setattr__(self, "_log", log)

    def __getattr__(self, name):
        if name in self._forbidden:
            raise Invalid("policy_reached_ground_truth: " + name)
        if name not in self._allowed:
            raise Invalid("policy_reached_undeclared_field: " + name)
        self._log.add(name)
        return getattr(self._target, name)

    def __setattr__(self, name, value):
        raise Invalid("A policy may not mutate its observation")


def audit_policy_observability(policy, samples=32, seed=99):
    """Probe a policy with a view that refuses ground truth, and report what it actually read.

    This is a structural check, not a promise about a policy nobody has run: it exercises
    `decide` and `escalate` over a spread of synthetic observations and records every field
    touched. A policy that reaches for `true_difficulty` fails here rather than silently
    scoring well.
    """
    if not 1 <= int(samples) <= 512:
        raise Invalid("samples must lie in [1, 512]")
    leaked = []
    request_reads = set()
    system_reads = set()
    for index in range(int(samples)):
        hint = index / max(int(samples) - 1, 1)
        view = RequestView(
            id=index,
            arrival_time=float(index),
            tenant="tenant_probe",
            prompt_tokens=200 + index,
            declared_class="moderate",
            difficulty_hint=hint,
            value_usd=round(crn_uniform(seed, index, "value"), 6),
            retrieval_requested=bool(index % 2),
            tool_requested=bool(index % 3 == 0),
        )
        system = SystemView(
            now=float(index),
            queue_depth={"model_cheap": index, "model_mid": index // 2, "model_frontier": index},
            busy_servers={"model_cheap": index, "model_mid": index, "model_frontier": index},
            utilisation={
                "model_cheap": min(1.0, hint),
                "model_mid": min(1.0, hint),
                "model_frontier": min(1.0, hint),
            },
            headroom={
                "model_cheap": 1.0 - min(1.0, hint),
                "model_mid": 1.0 - min(1.0, hint),
                "model_frontier": 1.0 - min(1.0, hint),
            },
            recent_shed_rate=hint,
            recent_timeout_rate=hint / 2,
            recent_p95_latency_seconds=hint * 20.0,
            admitted=index,
            completed=index,
            spend_usd=hint,
            budget_remaining_usd=10.0 - hint,
            slo_seconds=10.0,
        )
        probe_request = _ObservationProbe(
            view, OBSERVABLE_FIELDS, HIDDEN_FIELDS, request_reads
        )
        probe_system = _ObservationProbe(system, SYSTEM_VIEW_FIELDS, (), system_reads)
        try:
            decision = policy.decide(probe_request, probe_system)
        except Invalid as failure:
            leaked.append(str(failure))
            continue
        if not isinstance(decision, RouteDecision):
            raise Invalid("A policy must return a RouteDecision")
        if decision.model_tier not in TIER_LADDER:
            raise Invalid("A policy returned an undeclared model tier")
        for tier in TIER_LADDER:
            try:
                policy.escalate(
                    probe_request,
                    probe_system,
                    {"tier": tier, "confidence": hint, "verifier_rejected": True, "round": 1},
                )
            except Invalid as failure:
                leaked.append(str(failure))
    return {
        "policy": policy.name,
        "samples": int(samples),
        "request_fields_read": sorted(request_reads),
        "system_fields_read": sorted(system_reads),
        "ground_truth_fields": list(HIDDEN_FIELDS),
        "leaked_accesses": leaked,
        "observable_only": not leaked,
        "view_type_has_hidden_fields": bool(
            set(HIDDEN_FIELDS) & set(RequestView.__dataclass_fields__)
        ),
        "checks": [
            "RequestView carries no ground-truth field",
            "every attribute a policy reads is on the declared observable list",
            "decide returns a RouteDecision with a declared tier",
            "escalate never reaches for a hidden field",
        ],
        "scope": SCOPE,
    }


def describe_policies():
    routers = [p.describe() for p in default_policies()]
    return {
        "routing_policies": routers,
        "verification_policies": [
            verification_policy(kind).model_dump() for kind in VERIFICATION_POLICY_NAMES
        ],
        "tier_ladder": list(TIER_LADDER),
        "interface": "decide(RequestView, SystemView) -> RouteDecision; "
        "escalate(RequestView, SystemView, feedback) -> tier | None",
        "scope": SCOPE,
    }


_ = math
