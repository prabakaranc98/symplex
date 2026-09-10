"""Request processes: when work arrives, what kind it is, and how long it really takes.

Arrivals are non-homogeneous Poisson with a diurnal shape, Markov-modulated Poisson, or a
Hawkes self-exciting process, because bursts — not the mean rate — are what break an agent
system. Request difficulty is drawn from a mix over classes and is *hidden*: a policy sees a
noisy hint, never the truth. Service work is heavy-tailed, so the mean is not the story.

Nothing here is fitted to observed traffic. These are declared generative assumptions; every
downstream conclusion is conditional on them.
"""

import hashlib
import math
import random
from dataclasses import dataclass, fields
from typing import Annotated, Literal

from pydantic import Field, model_validator

from symplex.core.contracts import Invalid, digest
from symplex.modeling.complex_system import ClosedContract, Identifier, Text

MAX_REQUESTS = 40_000
MAX_HORIZON_SECONDS = 86_400.0
MAX_RATE_PER_SECOND = 500.0

SCOPE = (
    "Synthetic request process from declared arrival, mix and service-work assumptions. "
    "No observed traffic is reproduced and no real deployment's load is characterised."
)

DIFFICULTY_CLASSES = {
    "simple": 0.16,
    "moderate": 0.44,
    "hard": 0.70,
    "adversarial": 0.87,
}
CLASS_IDS = tuple(DIFFICULTY_CLASSES)
WorkLaw = Literal["lognormal", "pareto"]
ArrivalKind = Literal["poisson", "nhpp", "mmpp", "hawkes"]


def quantile(values, fraction):
    """Linear-interpolated order statistic; the same convention as modeling.dynamics."""
    if not values:
        raise Invalid("Quantile of an empty sample is undefined")
    ordered = sorted(values)
    index = (len(ordered) - 1) * min(max(fraction, 0.0), 1.0)
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def crn_uniform(seed, *keys):
    """A path-independent uniform draw keyed by content, not by call order.

    This is what makes common random numbers survive divergent policy arms: two arms that
    route request 41 differently still draw the same number for the same (request, purpose),
    so their difference is a paired comparison rather than two independent samples.
    """
    material = "|".join(str(k) for k in (seed,) + keys).encode()
    raw = hashlib.blake2b(material, digest_size=8).digest()
    return int.from_bytes(raw, "big") / 2**64


def crn_exponential(seed, *keys):
    return -math.log(1.0 - crn_uniform(seed, *keys) * (1 - 1e-15))


class ArrivalProcess(ClosedContract):
    kind: ArrivalKind
    base_rate_per_second: Annotated[float, Field(gt=0.0, le=MAX_RATE_PER_SECOND)]
    diurnal_amplitude: Annotated[float, Field(ge=0.0, le=0.95)]
    period_seconds: Annotated[float, Field(gt=0.0, le=MAX_HORIZON_SECONDS)]
    burst_rate_multiplier: Annotated[float, Field(ge=1.0, le=40.0)]
    burst_onset_per_second: Annotated[float, Field(ge=0.0, le=10.0)]
    burst_exit_per_second: Annotated[float, Field(ge=0.0, le=10.0)]
    excitation: Annotated[float, Field(ge=0.0, le=0.95)]
    decay_per_second: Annotated[float, Field(gt=0.0, le=100.0)]

    @model_validator(mode="after")
    def stationary(self):
        if self.kind == "hawkes" and self.excitation >= 0.95:
            raise ValueError("Hawkes branching ratio must stay below one to be stationary")
        if self.kind == "mmpp" and (
            self.burst_onset_per_second <= 0 or self.burst_exit_per_second <= 0
        ):
            raise ValueError("A modulating chain needs positive onset and exit rates")
        if self.kind == "mmpp" and self.burst_rate_multiplier <= 1.0:
            raise ValueError("A modulating chain needs a burst state that is actually faster")
        return self

    def peak_rate(self):
        if self.kind == "poisson":
            return self.base_rate_per_second
        if self.kind == "nhpp":
            return self.base_rate_per_second * (1.0 + self.diurnal_amplitude)
        if self.kind == "mmpp":
            return self.base_rate_per_second * self.burst_rate_multiplier
        return self.base_rate_per_second / max(1e-6, 1.0 - self.excitation)


class WorkloadRegime(ClosedContract):
    id: Identifier
    name: Text
    description: Text
    arrival: ArrivalProcess
    class_weights: dict[str, Annotated[float, Field(ge=0.0, le=1.0)]]
    burst_class_weights: dict[str, Annotated[float, Field(ge=0.0, le=1.0)]]
    work_law: WorkLaw
    work_dispersion: Annotated[float, Field(gt=0.0, le=4.0)]
    hint_noise: Annotated[float, Field(ge=0.0, le=0.5)]
    injection_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    safety_critical_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    retrieval_useful_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    tool_needed_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    memory_read_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    expected_stress: Text

    @model_validator(mode="after")
    def mixes_normalised(self):
        for label, weights in (
            ("class_weights", self.class_weights),
            ("burst_class_weights", self.burst_class_weights),
        ):
            if set(weights) != set(CLASS_IDS):
                raise ValueError(label + " must name exactly the declared difficulty classes")
            total = math.fsum(weights.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(label + " must sum to one")
        if self.work_law == "pareto" and self.work_dispersion <= 1.0:
            raise ValueError("Pareto work requires a tail index above one for a finite mean")
        return self


@dataclass(frozen=True)
class Request:
    """One arriving request. Hidden fields are ground truth and never reach a policy."""

    id: int
    arrival_time: float
    tenant: str
    prompt_tokens: int
    declared_class: str
    difficulty_hint: float
    value_usd: float
    retrieval_requested: bool
    tool_requested: bool
    in_burst: bool
    true_class: str
    true_difficulty: float
    work_factor: float
    retrieval_actually_helps: bool
    tool_actually_needed: bool
    safety_critical: bool
    carries_injection: bool
    reads_memory: bool


@dataclass(frozen=True)
class RequestView:
    """Everything a routing policy is allowed to know at decision time.

    There is deliberately no difficulty, no correctness and no safety label on this object.
    A policy that wants those must estimate them, which is the whole problem.
    """

    id: int
    arrival_time: float
    tenant: str
    prompt_tokens: int
    declared_class: str
    difficulty_hint: float
    value_usd: float
    retrieval_requested: bool
    tool_requested: bool


OBSERVABLE_FIELDS = tuple(f.name for f in fields(RequestView))
HIDDEN_FIELDS = tuple(f.name for f in fields(Request) if f.name not in OBSERVABLE_FIELDS)


def view_of(request):
    return RequestView(**{name: getattr(request, name) for name in OBSERVABLE_FIELDS})


@dataclass(frozen=True)
class Trace:
    """A frozen arrival trace. Reused verbatim across policy arms; that is the CRN pairing."""

    id: str
    regime_id: str
    seed: int
    horizon_seconds: float
    requests: tuple
    trace_digest: str

    def views(self):
        return tuple(view_of(r) for r in self.requests)

    def arrival_times(self):
        return tuple(r.arrival_time for r in self.requests)

    def summary(self):
        work = [r.work_factor for r in self.requests]
        counts = {c: sum(1 for r in self.requests if r.true_class == c) for c in CLASS_IDS}
        return {
            "trace_id": self.id,
            "regime_id": self.regime_id,
            "seed": self.seed,
            "horizon_seconds": self.horizon_seconds,
            "request_count": len(self.requests),
            "arrival_rate_per_second": len(self.requests) / max(self.horizon_seconds, 1e-9),
            "class_counts": counts,
            "burst_fraction": (
                sum(1 for r in self.requests if r.in_burst) / len(self.requests)
                if self.requests
                else 0.0
            ),
            "injection_fraction": (
                sum(1 for r in self.requests if r.carries_injection) / len(self.requests)
                if self.requests
                else 0.0
            ),
            "work_factor_mean": (math.fsum(work) / len(work)) if work else 0.0,
            "work_factor_p99": quantile(work, 0.99) if work else 0.0,
            "work_factor_tail_ratio": (
                quantile(work, 0.99) / (math.fsum(work) / len(work)) if work else 0.0
            ),
            "arrival_time_digest": digest([round(t, 9) for t in self.arrival_times()]),
            "trace_digest": self.trace_digest,
            "observable_fields": list(OBSERVABLE_FIELDS),
            "hidden_fields": list(HIDDEN_FIELDS),
            "scope": SCOPE,
        }


def poisson_arrivals(rate, horizon, rng):
    """Homogeneous Poisson process by exponential interarrival times."""
    times, now = [], 0.0
    while True:
        now += rng.expovariate(rate)
        if now > horizon:
            return times
        times.append((now, False))


def nhpp_arrivals(process, horizon, rng):
    """Non-homogeneous Poisson by thinning against the diurnal peak rate."""
    peak = process.peak_rate()
    times, now = [], 0.0
    while True:
        now += rng.expovariate(peak)
        if now > horizon:
            return times
        intensity = process.base_rate_per_second * (
            1.0
            + process.diurnal_amplitude
            * math.sin(2.0 * math.pi * now / process.period_seconds)
        )
        if rng.random() <= intensity / peak:
            times.append((now, intensity >= 0.80 * peak))


def mmpp_arrivals(process, horizon, rng):
    """Markov-modulated Poisson: a two-state chain switches the arrival rate."""
    rates = (
        process.base_rate_per_second,
        process.base_rate_per_second * process.burst_rate_multiplier,
    )
    switch = (process.burst_onset_per_second, process.burst_exit_per_second)
    state, now, times = 0, 0.0, []
    while now < horizon:
        dwell = rng.expovariate(switch[state])
        segment_end = min(horizon, now + dwell)
        clock = now
        while True:
            clock += rng.expovariate(rates[state])
            if clock > segment_end:
                break
            times.append((clock, state == 1))
        now = segment_end
        state = 1 - state
    return times


def hawkes_arrivals(process, horizon, rng):
    """Hawkes self-excitation by Ogata thinning.

    The exponential kernel lets the excitation be carried as one decaying scalar, so the
    generator is exact and O(1) per proposal instead of re-summing the whole history.
    """
    background = process.base_rate_per_second
    beta = process.decay_per_second
    alpha = process.excitation * beta
    stationary = background / max(1e-9, 1.0 - process.excitation)
    excitation, times, now = 0.0, [], 0.0
    while True:
        bound = background + excitation
        step = rng.expovariate(max(bound, 1e-9))
        candidate = now + step
        if candidate > horizon:
            return times
        excitation *= math.exp(-beta * step)
        now = candidate
        intensity = background + excitation
        if rng.random() <= intensity / max(bound, 1e-9):
            excitation += alpha
            times.append((now, intensity > 1.5 * stationary))


_ARRIVAL_GENERATORS = {
    "poisson": lambda p, h, r: poisson_arrivals(p.base_rate_per_second, h, r),
    "nhpp": nhpp_arrivals,
    "mmpp": mmpp_arrivals,
    "hawkes": hawkes_arrivals,
}


def _work_factor(regime, unit):
    """Heavy-tailed service work, normalised to unit mean so tiers stay comparable."""
    if regime.work_law == "lognormal":
        sigma = regime.work_dispersion
        normal = _inverse_normal(unit)
        return math.exp(sigma * normal - 0.5 * sigma * sigma)
    alpha = regime.work_dispersion
    scale = (alpha - 1.0) / alpha
    return scale * (1.0 - unit * (1 - 1e-12)) ** (-1.0 / alpha)


def _inverse_normal(unit):
    """Acklam's rational approximation; deterministic and adequate for a simulator."""
    unit = min(max(unit, 1e-12), 1 - 1e-12)
    a = (-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.3577518672690, -30.66479806614716, 2.506628277459239)
    b = (-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.28068155288572)
    c = (-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783)
    d = (0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416)
    low, high = 0.02425, 1 - 0.02425
    if unit < low:
        q = math.sqrt(-2 * math.log(unit))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if unit > high:
        q = math.sqrt(-2 * math.log(1 - unit))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = unit - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )


def _pick_class(weights, unit):
    total = 0.0
    for name in CLASS_IDS:
        total += weights[name]
        if unit <= total:
            return name
    return CLASS_IDS[-1]


def generate_trace(regime, seed=2026, horizon_seconds=120.0, max_requests=MAX_REQUESTS):
    """Draw one frozen trace. Same regime and seed always yield the same requests."""
    if isinstance(regime, str):
        regime = regime_library()[regime] if regime in REGIME_SPECS else None
        if regime is None:
            raise Invalid("Unknown workload regime")
    regime = WorkloadRegime.model_validate(
        regime.model_dump() if isinstance(regime, WorkloadRegime) else regime
    )
    if not isinstance(seed, int) or not 0 <= seed <= 2**31:
        raise Invalid("Seed must be a non-negative integer below 2^31")
    if not 1.0 <= horizon_seconds <= MAX_HORIZON_SECONDS:
        raise Invalid("Horizon must lie in [1, 86400] seconds")
    if not 1 <= max_requests <= MAX_REQUESTS:
        raise Invalid("max_requests must lie in [1, 40000]")
    rng = random.Random(hashlib.blake2b(f"{seed}|{regime.id}|arrivals".encode()).digest())
    stamped = _ARRIVAL_GENERATORS[regime.arrival.kind](regime.arrival, horizon_seconds, rng)
    if len(stamped) > max_requests:
        raise Invalid(
            f"Arrival envelope exceeded: {len(stamped)} arrivals above the cap {max_requests}"
        )
    requests = []
    for index, (arrival, in_burst) in enumerate(stamped):
        key = (seed, regime.id, index)
        weights = regime.burst_class_weights if in_burst else regime.class_weights
        true_class = _pick_class(weights, crn_uniform(*key, "class"))
        centre = DIFFICULTY_CLASSES[true_class]
        spread = 0.08
        difficulty = min(
            1.0, max(0.0, centre + spread * _inverse_normal(crn_uniform(*key, "difficulty")))
        )
        hint = min(
            1.0,
            max(
                0.0,
                difficulty
                + regime.hint_noise * _inverse_normal(crn_uniform(*key, "hint")),
            ),
        )
        declared = min(
            CLASS_IDS,
            key=lambda name: abs(DIFFICULTY_CLASSES[name] - hint),
        )
        work = _work_factor(regime, crn_uniform(*key, "work"))
        requests.append(
            Request(
                id=index,
                arrival_time=arrival,
                tenant="tenant_" + str(index % 7),
                prompt_tokens=int(180 + 2600 * min(1.0, hint * work / 3.0)),
                declared_class=declared,
                difficulty_hint=hint,
                value_usd=round(0.05 + 0.95 * crn_uniform(*key, "value"), 6),
                retrieval_requested=crn_uniform(*key, "retrieval_request") < 0.55,
                tool_requested=crn_uniform(*key, "tool_request") < regime.tool_needed_rate,
                in_burst=in_burst,
                true_class=true_class,
                true_difficulty=difficulty,
                work_factor=work,
                retrieval_actually_helps=(
                    crn_uniform(*key, "retrieval_useful") < regime.retrieval_useful_rate
                ),
                tool_actually_needed=(
                    crn_uniform(*key, "tool_needed") < regime.tool_needed_rate
                ),
                safety_critical=(
                    crn_uniform(*key, "safety") < regime.safety_critical_rate
                ),
                carries_injection=(
                    crn_uniform(*key, "injection") < regime.injection_rate
                ),
                reads_memory=crn_uniform(*key, "memory_read") < regime.memory_read_rate,
            )
        )
    payload = [
        (r.id, round(r.arrival_time, 9), r.true_class, round(r.true_difficulty, 9),
         round(r.work_factor, 9), r.in_burst)
        for r in requests
    ]
    return Trace(
        id=f"{regime.id}_s{seed}_h{int(horizon_seconds)}",
        regime_id=regime.id,
        seed=seed,
        horizon_seconds=float(horizon_seconds),
        requests=tuple(requests),
        trace_digest=digest(payload),
    )


def _process(kind, base_rate, **overrides):
    spec = {
        "kind": kind,
        "base_rate_per_second": base_rate,
        "diurnal_amplitude": 0.0,
        "period_seconds": 300.0,
        "burst_rate_multiplier": 1.0,
        "burst_onset_per_second": 0.0,
        "burst_exit_per_second": 0.0,
        "excitation": 0.0,
        "decay_per_second": 1.0,
    }
    spec.update(overrides)
    return spec


def _mix(simple, moderate, hard, adversarial):
    return {
        "simple": simple,
        "moderate": moderate,
        "hard": hard,
        "adversarial": adversarial,
    }


REGIME_SPECS = {
    "steady": {
        "id": "steady",
        "name": "Steady cheap load",
        "description": "Homogeneous Poisson arrivals, mostly easy requests, mild service spread.",
        "arrival": _process("poisson", 6.0),
        "class_weights": _mix(0.55, 0.33, 0.10, 0.02),
        "burst_class_weights": _mix(0.55, 0.33, 0.10, 0.02),
        "work_law": "lognormal",
        "work_dispersion": 0.55,
        "hint_noise": 0.10,
        "injection_rate": 0.01,
        "safety_critical_rate": 0.10,
        "retrieval_useful_rate": 0.45,
        "tool_needed_rate": 0.15,
        "memory_read_rate": 0.25,
        "expected_stress": "None. This is the regime where the cheapest policy should win on cost.",
    },
    "diurnal": {
        "id": "diurnal",
        "name": "Diurnal cycle",
        "description": "Non-homogeneous Poisson with a sinusoidal day; the mix hardens at peak.",
        "arrival": _process("nhpp", 6.5, diurnal_amplitude=0.70, period_seconds=120.0),
        "class_weights": _mix(0.42, 0.36, 0.18, 0.04),
        "burst_class_weights": _mix(0.28, 0.36, 0.30, 0.06),
        "work_law": "lognormal",
        "work_dispersion": 0.75,
        "hint_noise": 0.12,
        "injection_rate": 0.02,
        "safety_critical_rate": 0.15,
        "retrieval_useful_rate": 0.50,
        "tool_needed_rate": 0.20,
        "memory_read_rate": 0.30,
        "expected_stress": "Peak-hour concurrency; capacity sized for the mean is wrong twice a day.",
    },
    "flash_crowd": {
        "id": "flash_crowd",
        "name": "Flash crowd",
        "description": "Hawkes self-excitation: every request makes the next one likelier, and "
        "the class mix hardens inside the cluster, so bursts are correlated across types.",
        "arrival": _process("hawkes", 3.0, excitation=0.80, decay_per_second=2.0),
        "class_weights": _mix(0.40, 0.35, 0.20, 0.05),
        "burst_class_weights": _mix(0.14, 0.30, 0.44, 0.12),
        "work_law": "lognormal",
        "work_dispersion": 0.95,
        "hint_noise": 0.15,
        "injection_rate": 0.04,
        "safety_critical_rate": 0.20,
        "retrieval_useful_rate": 0.50,
        "tool_needed_rate": 0.25,
        "memory_read_rate": 0.35,
        "expected_stress": "Correlated bursts of hard requests; escalation demand spikes with arrivals.",
    },
    "degraded_upstream": {
        "id": "degraded_upstream",
        "name": "Degraded upstream",
        "description": "Markov-modulated arrivals with long slow phases and a wide service spread; "
        "paired with retrieval degradation and rate-limit onset in the fault schedule.",
        "arrival": _process(
            "mmpp",
            5.0,
            burst_rate_multiplier=2.4,
            burst_onset_per_second=0.05,
            burst_exit_per_second=0.10,
        ),
        "class_weights": _mix(0.35, 0.38, 0.22, 0.05),
        "burst_class_weights": _mix(0.22, 0.36, 0.33, 0.09),
        "work_law": "lognormal",
        "work_dispersion": 1.15,
        "hint_noise": 0.18,
        "injection_rate": 0.05,
        "safety_critical_rate": 0.25,
        "retrieval_useful_rate": 0.60,
        "tool_needed_rate": 0.35,
        "memory_read_rate": 0.45,
        "expected_stress": "Tool and retrieval failure while load is already modulated upward.",
    },
    "adversarial_mix": {
        "id": "adversarial_mix",
        "name": "Adversarial mix",
        "description": "Poisson arrivals but a hostile mix: hard requests, prompt injections and "
        "Pareto service work, so the mean tells you almost nothing about the tail.",
        "arrival": _process("poisson", 5.0),
        "class_weights": _mix(0.12, 0.26, 0.38, 0.24),
        "burst_class_weights": _mix(0.12, 0.26, 0.38, 0.24),
        "work_law": "pareto",
        "work_dispersion": 1.5,
        "hint_noise": 0.22,
        "injection_rate": 0.30,
        "safety_critical_rate": 0.55,
        "retrieval_useful_rate": 0.55,
        "tool_needed_rate": 0.40,
        "memory_read_rate": 0.50,
        "expected_stress": "Unsafe-action containment and calibration, not throughput.",
    },
}

REGIME_IDS = tuple(REGIME_SPECS)


def regime_library():
    """The named workload regimes, validated. Each is an assumption set, not a measurement."""
    return {ident: WorkloadRegime.model_validate(spec) for ident, spec in REGIME_SPECS.items()}


def describe_regimes():
    library = regime_library()
    return {
        "regimes": [
            {
                "id": regime.id,
                "name": regime.name,
                "description": regime.description,
                "arrival_kind": regime.arrival.kind,
                "mean_rate_per_second": regime.arrival.base_rate_per_second,
                "peak_rate_per_second": regime.arrival.peak_rate(),
                "work_law": regime.work_law,
                "work_dispersion": regime.work_dispersion,
                "hardest_class_weight": regime.class_weights["adversarial"],
                "injection_rate": regime.injection_rate,
                "expected_stress": regime.expected_stress,
            }
            for regime in library.values()
        ],
        "difficulty_classes": dict(DIFFICULTY_CLASSES),
        "observable_fields": list(OBSERVABLE_FIELDS),
        "hidden_fields": list(HIDDEN_FIELDS),
        "scope": SCOPE,
    }


def tail_report(values, label="work_factor"):
    """Report the tail, never only the mean; a mean-only latency claim is a false claim."""
    if not values:
        raise Invalid("Tail report of an empty sample is undefined")
    mean = math.fsum(values) / len(values)
    return {
        "label": label,
        "count": len(values),
        "mean": mean,
        "p50": quantile(values, 0.50),
        "p95": quantile(values, 0.95),
        "p99": quantile(values, 0.99),
        "max": max(values),
        "p99_over_mean": quantile(values, 0.99) / mean if mean > 0 else float("inf"),
        "scope": SCOPE,
    }
