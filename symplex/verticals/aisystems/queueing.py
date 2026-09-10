"""Discrete-event queueing for agent systems, cross-checked against closed-form queues.

A spreadsheet multiplies a mean rate by a mean service time and calls it capacity. That is
wrong in exactly the place agent systems break: concurrency limits, bounded queues, per-node
rate limits, batching and heavy tails all live in the variance, not the mean. This module
simulates those, and — because a simulator nobody has checked is a rumour — reproduces
M/M/1, M/M/c and Erlang-B/C analytically so the tests can assert that the two agree.

Randomness is drawn by content key, not call order, so two policy arms that route the same
request differently still see the same underlying draws. Comparisons are therefore paired.
"""

import collections
import heapq
import math
from dataclasses import dataclass, field

from symplex.core.contracts import Invalid, digest
from symplex.verticals.aisystems.topology import Topology, architecture_profile
from symplex.verticals.aisystems.workload import (
    crn_uniform,
    _inverse_normal,
    quantile,
    view_of,
)

MAX_EVENTS = 400_000
MAX_SIM_SECONDS = 86_400.0
MAX_ARRIVALS = 60_000
MAX_STAGES_PER_REQUEST = 64
CAPABILITY_MARGIN = 0.42
BATCH_PENALTY = 0.12

SCOPE = (
    "Discrete-event queueing over declared service, concurrency and rate-limit parameters, "
    "cross-checked against closed-form M/M/c results. Latency, shedding and utilisation here "
    "are properties of the declared model, not measurements of a deployed service."
)


# --------------------------------------------------------------------------------------
# Closed-form queues. These are the ground truth the simulator is held against.
# --------------------------------------------------------------------------------------


def erlang_b(offered_load, servers):
    """Blocking probability of M/M/c/c by the numerically stable recursion."""
    offered_load = float(offered_load)
    servers = int(servers)
    if offered_load < 0 or servers < 1 or servers > 4096:
        raise Invalid("Erlang-B needs a non-negative load and 1..4096 servers")
    probability = 1.0
    for index in range(1, servers + 1):
        probability = offered_load * probability / (index + offered_load * probability)
    return probability


def erlang_c(offered_load, servers):
    """Probability that an arrival to M/M/c must wait, derived from Erlang-B."""
    offered_load = float(offered_load)
    servers = int(servers)
    utilisation = offered_load / servers
    if utilisation >= 1.0:
        raise Invalid("Erlang-C is undefined at or above full utilisation")
    blocking = erlang_b(offered_load, servers)
    return blocking / (1.0 - utilisation * (1.0 - blocking))


def little_law(arrival_rate, mean_time_in_system):
    """L = lambda W. Stated as its own function because it is the check, not a formula."""
    arrival_rate = float(arrival_rate)
    mean_time_in_system = float(mean_time_in_system)
    if arrival_rate < 0 or mean_time_in_system < 0:
        raise Invalid("Little's Law needs non-negative rate and time")
    return arrival_rate * mean_time_in_system


def mmc_metrics(arrival_rate, service_rate, servers=1):
    """M/M/c stationary metrics. M/M/1 is the c = 1 case and is not special-cased."""
    arrival_rate = float(arrival_rate)
    service_rate = float(service_rate)
    servers = int(servers)
    if arrival_rate <= 0 or service_rate <= 0 or servers < 1:
        raise Invalid("M/M/c needs positive rates and at least one server")
    offered = arrival_rate / service_rate
    utilisation = offered / servers
    if utilisation >= 1.0:
        raise Invalid("M/M/c is unstable at or above full utilisation")
    delay_probability = erlang_c(offered, servers)
    mean_wait = delay_probability / (servers * service_rate - arrival_rate)
    mean_sojourn = mean_wait + 1.0 / service_rate
    return {
        "arrival_rate": arrival_rate,
        "service_rate": service_rate,
        "servers": servers,
        "offered_load_erlangs": offered,
        "utilisation": utilisation,
        "delay_probability_erlang_c": delay_probability,
        "mean_wait_seconds": mean_wait,
        "mean_sojourn_seconds": mean_sojourn,
        "mean_in_queue": little_law(arrival_rate, mean_wait),
        "mean_in_system": little_law(arrival_rate, mean_sojourn),
        "model": "M/M/c FCFS, infinite buffer, exponential service",
        "scope": "Closed-form stationary result for an idealised queue; no deployment is described.",
    }


def mmck_metrics(arrival_rate, service_rate, servers, capacity):
    """M/M/c/K with a finite total system size, giving the loss probability directly."""
    arrival_rate = float(arrival_rate)
    service_rate = float(service_rate)
    servers = int(servers)
    total = servers + int(capacity)
    if arrival_rate <= 0 or service_rate <= 0 or servers < 1 or total < servers:
        raise Invalid("M/M/c/K needs positive rates, c >= 1 and K >= c")
    if total > 4096:
        raise Invalid("M/M/c/K state space exceeds the declared envelope")
    offered = arrival_rate / service_rate
    unnormalised = [1.0]
    for state in range(1, total + 1):
        rate = min(state, servers)
        unnormalised.append(unnormalised[-1] * offered / rate)
    norm = math.fsum(unnormalised)
    probabilities = [p / norm for p in unnormalised]
    loss = probabilities[-1]
    effective = arrival_rate * (1.0 - loss)
    in_system = math.fsum(state * p for state, p in enumerate(probabilities))
    sojourn = in_system / effective if effective > 0 else 0.0
    return {
        "arrival_rate": arrival_rate,
        "service_rate": service_rate,
        "servers": servers,
        "system_capacity": total,
        "offered_load_erlangs": offered,
        "loss_probability": loss,
        "effective_arrival_rate": effective,
        "mean_in_system": in_system,
        "mean_sojourn_seconds": sojourn,
        "mean_wait_seconds": max(0.0, sojourn - 1.0 / service_rate),
        "erlang_b_reference": erlang_b(offered, servers) if total == servers else None,
        "model": "M/M/c/K FCFS with loss at the boundary",
        "scope": "Closed-form stationary result for an idealised queue; no deployment is described.",
    }


# --------------------------------------------------------------------------------------
# Service-time laws.
# --------------------------------------------------------------------------------------


def service_draw(law, mean_seconds, dispersion, unit):
    """One service time from a unit draw. Mean-preserving for every law."""
    unit = min(max(float(unit), 1e-12), 1 - 1e-12)
    if mean_seconds <= 0:
        raise Invalid("Service mean must be positive")
    if law == "deterministic":
        return mean_seconds
    if law == "exponential":
        return -mean_seconds * math.log(1.0 - unit)
    if law == "lognormal":
        sigma = max(dispersion, 1e-9)
        return mean_seconds * math.exp(sigma * _inverse_normal(unit) - 0.5 * sigma * sigma)
    if law == "pareto":
        alpha = dispersion
        if alpha <= 1.0:
            raise Invalid("Pareto service needs a tail index above one")
        return mean_seconds * (alpha - 1.0) / alpha * (1.0 - unit) ** (-1.0 / alpha)
    raise Invalid("Unknown service law: " + str(law))


def service_time_samples(node, count=4000, seed=11):
    """Draw service times for one declared node; used to show the tail, never just the mean."""
    if not 1 <= int(count) <= 200_000:
        raise Invalid("Sample count must lie in [1, 200000]")
    return [
        service_draw(
            node.service_law,
            node.service_seconds,
            node.service_dispersion,
            crn_uniform(seed, node.id, index, "service"),
        )
        for index in range(int(count))
    ]


def latency_stats(values, label="latency_seconds"):
    """Tail-first summary. A mean alone is not a latency report."""
    if not values:
        return {
            "label": label,
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "p99_over_mean": 0.0,
        }
    mean = math.fsum(values) / len(values)
    return {
        "label": label,
        "count": len(values),
        "mean": mean,
        "p50": quantile(values, 0.50),
        "p95": quantile(values, 0.95),
        "p99": quantile(values, 0.99),
        "max": max(values),
        "p99_over_mean": quantile(values, 0.99) / mean if mean > 0 else 0.0,
    }


# --------------------------------------------------------------------------------------
# Single-station simulator: the object the analytic results are asserted against.
# --------------------------------------------------------------------------------------


def simulate_station(
    arrival_rate,
    service_rate,
    servers=1,
    queue_capacity=None,
    arrivals=20_000,
    seed=2026,
    service_law="exponential",
    dispersion=0.0,
    warmup_fraction=0.05,
):
    """One FCFS station with c servers and a finite buffer, driven by Poisson arrivals.

    Deliberately narrow: this exists so the tests can compare it against M/M/c and Erlang-B/C.
    The network simulator below shares its service laws and admission rule.
    """
    arrival_rate = float(arrival_rate)
    service_rate = float(service_rate)
    servers = int(servers)
    arrivals = int(arrivals)
    if arrival_rate <= 0 or service_rate <= 0:
        raise Invalid("Station simulation needs positive arrival and service rates")
    if not 1 <= servers <= 512:
        raise Invalid("Servers must lie in [1, 512]")
    if not 100 <= arrivals <= MAX_ARRIVALS:
        raise Invalid("Arrivals must lie in [100, 60000]")
    if not 0.0 <= warmup_fraction < 0.5:
        raise Invalid("Warm-up fraction must lie in [0, 0.5)")
    capacity = 10**9 if queue_capacity is None else int(queue_capacity)
    if capacity < 0:
        raise Invalid("Queue capacity must be non-negative")
    limit = servers + capacity

    times = []
    clock = 0.0
    for index in range(arrivals):
        clock += -math.log(1.0 - crn_uniform(seed, index, "interarrival")) / arrival_rate
        times.append(clock)
    horizon = times[-1]
    warmup = horizon * warmup_fraction

    heap = []
    order = 0
    queue = collections.deque()
    busy = 0
    blocked = 0
    admitted = 0
    completed = 0
    waits = []
    sojourns = []
    occupancy_integral = 0.0
    busy_integral = 0.0
    last_change = 0.0
    integral_start = None
    events = 0
    next_arrival = 0
    now = 0.0

    def touch(moment):
        nonlocal occupancy_integral, busy_integral, last_change, integral_start
        if integral_start is None:
            if moment >= warmup:
                integral_start = warmup if last_change <= warmup else last_change
                last_change = max(last_change, warmup)
            else:
                last_change = moment
                return
        span = moment - last_change
        occupancy_integral += (busy + len(queue)) * span
        busy_integral += busy * span
        last_change = moment

    def start_service(moment):
        nonlocal busy, order
        while queue and busy < servers:
            entry_time, index = queue.popleft()
            wait = moment - entry_time
            duration = service_draw(
                service_law,
                1.0 / service_rate,
                dispersion,
                crn_uniform(seed, index, "service"),
            )
            if entry_time >= warmup:
                waits.append(wait)
                sojourns.append(wait + duration)
            busy += 1
            order += 1
            heapq.heappush(heap, (moment + duration, order, 1, index))

    while True:
        if events > MAX_EVENTS:
            raise Invalid("Station simulation exceeded the declared event envelope")
        arrival_time = times[next_arrival] if next_arrival < arrivals else None
        if arrival_time is not None and (not heap or arrival_time <= heap[0][0]):
            now = arrival_time
            next_arrival += 1
            events += 1
            touch(now)
            if busy + len(queue) >= limit:
                if now >= warmup:
                    blocked += 1
                continue
            if now >= warmup:
                admitted += 1
            queue.append((now, next_arrival - 1))
            start_service(now)
            continue
        if not heap:
            break
        now, _, _, _ = heapq.heappop(heap)
        events += 1
        touch(now)
        busy -= 1
        completed += 1
        start_service(now)

    touch(now)
    observed = max(now - (integral_start if integral_start is not None else 0.0), 1e-9)
    offered = arrival_rate / service_rate
    effective_rate = (admitted) / observed
    mean_sojourn = math.fsum(sojourns) / len(sojourns) if sojourns else 0.0
    mean_wait = math.fsum(waits) / len(waits) if waits else 0.0
    return {
        "arrival_rate": arrival_rate,
        "service_rate": service_rate,
        "servers": servers,
        "queue_capacity": None if queue_capacity is None else int(queue_capacity),
        "system_capacity": None if queue_capacity is None else limit,
        "arrivals_generated": arrivals,
        "arrivals_measured": admitted + blocked,
        "admitted": admitted,
        "blocked": blocked,
        "completed": completed,
        "blocking_fraction": blocked / max(admitted + blocked, 1),
        "waited_fraction": sum(1 for w in waits if w > 1e-12) / max(len(waits), 1),
        "mean_wait_seconds": mean_wait,
        "mean_sojourn_seconds": mean_sojourn,
        "wait": latency_stats(waits, "wait_seconds"),
        "sojourn": latency_stats(sojourns, "sojourn_seconds"),
        "utilisation": busy_integral / observed / servers,
        "mean_in_system_observed": occupancy_integral / observed,
        "mean_in_system_little": little_law(effective_rate, mean_sojourn),
        "little_law_relative_gap": (
            abs(occupancy_integral / observed - little_law(effective_rate, mean_sojourn))
            / max(occupancy_integral / observed, 1e-12)
        ),
        "effective_arrival_rate": effective_rate,
        "observed_seconds": observed,
        "offered_load_erlangs": offered,
        "seed": seed,
        "service_law": service_law,
        "warmup_seconds": warmup,
        "events": events,
        "scope": SCOPE,
    }


# --------------------------------------------------------------------------------------
# Interface objects. The simulator produces SystemView and consumes RouteDecision; policies
# are written against these two types and nothing else.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SystemView:
    """Observable system state at a decision point. No ground truth appears here."""

    now: float
    queue_depth: dict
    busy_servers: dict
    utilisation: dict
    headroom: dict
    recent_shed_rate: float
    recent_timeout_rate: float
    recent_p95_latency_seconds: float
    admitted: int
    completed: int
    spend_usd: float
    budget_remaining_usd: float
    slo_seconds: float


@dataclass(frozen=True)
class RouteDecision:
    """What a policy is allowed to choose. Pure data; the simulator executes it."""

    model_tier: str
    use_cache: bool
    use_retrieval: bool
    use_tool: bool
    verify: bool
    escalate_human: bool
    max_escalations: int
    admit: bool
    rationale: str


class IdealDynamics:
    """Fault-free semantics: who answers correctly, what a verifier catches, what memory holds.

    The success rule is monotone in model capability by construction — a single per-request
    draw sets an effective difficulty, and a tier succeeds when its capability clears it. That
    monotonicity is what lets evaluate.py define an oracle whose regret is non-negative.
    """

    kind = "ideal"

    def __init__(self, topology, seed=2026):
        self.topology = topology
        self.seed = int(seed)
        self.capability = {
            tier: topology.model_tier(tier).capability
            for tier in ("cheap", "mid", "frontier")
        }
        caches = topology.by_kind("cache")
        retrievers = topology.by_kind("retriever")
        verifiers = topology.by_kind("verifier")
        humans = topology.by_kind("human_reviewer")
        self.cache_hit_rate = caches[0].capability if caches else 0.0
        self.retriever_quality = retrievers[0].capability if retrievers else 0.0
        self.verifier_recall = verifiers[0].capability if verifiers else 0.0
        self.human_recall = humans[0].capability if humans else 0.0
        self.tool_reliability = 0.985

    def node_available(self, node_id, now):
        return True

    def service_multiplier(self, node_id, now):
        return 1.0

    def rate_limit_multiplier(self, node_id, now):
        return 1.0

    def cache_lookup(self, request, now, state):
        hit = (
            crn_uniform(self.seed, request.id, "cache") < self.cache_hit_rate
            and request.true_difficulty < 0.5
        )
        return {"hit": hit, "stale": False}

    def retrieve(self, request, now, state):
        helped = request.retrieval_actually_helps and (
            crn_uniform(self.seed, request.id, "retrieval") < self.retriever_quality
        )
        return {"helped": helped, "poisoned": False, "degraded": False}

    def tool_call(self, request, now, attempt, state):
        ok = crn_uniform(self.seed, request.id, attempt, "tool") < self.tool_reliability
        return {"ok": ok, "timeout": not ok, "hard_failure": False, "injection_reached": False}

    def effective_difficulty(self, request, state):
        draw = crn_uniform(self.seed, request.id, "capability")
        base = request.true_difficulty + CAPABILITY_MARGIN * (draw - 0.5)
        bonus = 0.0
        if state.get("retrieval_helped"):
            bonus += 0.12
        if state.get("tool_ok") and request.tool_actually_needed:
            bonus += 0.10
        if state.get("contaminated"):
            bonus -= 0.28
        return min(1.0, max(0.0, base - bonus))

    def answer(self, request, tier, now, state):
        difficulty = self.effective_difficulty(request, state)
        correct = self.capability[tier] >= difficulty
        confidence = min(
            1.0, max(0.0, 0.5 + (self.capability[tier] - request.difficulty_hint))
        )
        unsafe = bool(
            (request.safety_critical and not correct)
            or state.get("contaminated")
            or state.get("injection_live")
        )
        return {
            "correct": correct,
            "confidence": confidence,
            "unsafe": unsafe,
            "effective_difficulty": difficulty,
        }

    def verify(self, request, now, state):
        caught = state.get("answer_correct") is False and (
            crn_uniform(self.seed, request.id, state.get("round", 0), "verify")
            < self.verifier_recall
        )
        unsafe_caught = state.get("unsafe") and (
            crn_uniform(self.seed, request.id, state.get("round", 0), "verify_safety")
            < self.verifier_recall
        )
        return {"reject": bool(caught or unsafe_caught), "caught_unsafe": bool(unsafe_caught)}

    def human_review(self, request, now, state):
        corrected = crn_uniform(self.seed, request.id, "human") < self.human_recall
        return {"corrected": corrected, "caught_unsafe": corrected and state.get("unsafe", False)}

    def memory_write(self, request, now, state):
        return {"contaminated_written": False, "quarantined": False}

    def memory_read(self, request, now, state):
        return {"contaminated": False}

    def report(self):
        return {
            "dynamics": self.kind,
            "faults_injected": 0,
            "scope": "Fault-free reference semantics; nothing is contained because nothing fails.",
        }


class _Station:
    __slots__ = (
        "node", "servers", "capacity", "queue", "busy", "tokens", "last_refill",
        "arrivals", "admitted", "shed", "completed", "timeouts", "deferrals", "batches",
        "busy_integral", "queue_integral", "last_change", "max_queue", "wake_pending",
        "batch_deadline", "batch_forced", "unavailable_rejections",
    )

    def __init__(self, node):
        self.node = node
        self.servers = node.servers
        self.capacity = node.queue_capacity
        self.queue = collections.deque()
        self.busy = 0
        self.tokens = float(node.rate_burst)
        self.last_refill = 0.0
        self.arrivals = 0
        self.admitted = 0
        self.shed = 0
        self.completed = 0
        self.timeouts = 0
        self.deferrals = 0
        self.batches = 0
        self.busy_integral = 0.0
        self.queue_integral = 0.0
        self.last_change = 0.0
        self.max_queue = 0
        self.wake_pending = False
        self.batch_deadline = None
        self.batch_forced = False
        self.unavailable_rejections = 0

    def touch(self, now):
        span = now - self.last_change
        if span > 0:
            self.busy_integral += self.busy * span
            self.queue_integral += len(self.queue) * span
            self.last_change = now


@dataclass
class _Job:
    request: object
    view: object
    decision: object = None
    plan: list = field(default_factory=list)
    index: int = -1
    tier: str = "cheap"
    start: float = 0.0
    finish: float = 0.0
    pending: int = 0
    escalations: int = 0
    fallbacks_used: int = 0
    tool_attempts: int = 0
    cost_usd: float = 0.0
    energy_joules: float = 0.0
    compute_units: float = 0.0
    tiers_used: list = field(default_factory=list)
    stations_visited: list = field(default_factory=list)
    state: dict = field(default_factory=dict)
    done: bool = False
    shed: bool = False
    timed_out: bool = False
    shed_at: str = ""


_KIND_FOR_STAGE = {
    "cache": "cache",
    "retrieve": "retriever",
    "tool": "tool",
    "verify": "verifier",
    "human": "human_reviewer",
    "memory": "memory",
    "router": "router",
}
_STAGE_FOR_KIND = {v: k for k, v in _KIND_FOR_STAGE.items()}
_STAGE_FOR_KIND["model"] = "model"
_OPTIONAL_STAGES = frozenset({"cache", "retrieve", "tool", "verify", "human", "memory"})
_WORK_SCALED_KINDS = frozenset({"model", "verifier", "tool", "retriever"})
MAX_FALLBACKS = 2
ENSEMBLE_BONUS = {1: 0.0, 2: 0.03, 3: 0.05}


class _Network:
    """The discrete-event engine. One request walks a policy-chosen sequence of stations."""

    def __init__(self, topology, trace, policy, seed, dynamics, slo_seconds,
                 abandon_seconds, budget_usd, max_events, trajectory_samples):
        self.topology = topology
        self.trace = trace
        self.policy = policy
        self.seed = int(seed)
        self.dynamics = dynamics
        self.slo_seconds = float(slo_seconds)
        self.abandon_seconds = float(abandon_seconds)
        self.budget_usd = budget_usd
        self.max_events = int(max_events)
        self.profile = architecture_profile(topology)
        self.stations = {n.id: _Station(n) for n in topology.nodes}
        self.fallback = {}
        for edge in topology.edges:
            if edge.fallback is not None and edge.source not in self.fallback:
                self.fallback[edge.source] = edge.fallback
        self.retry = {}
        for edge in topology.edges:
            if edge.source not in self.retry:
                self.retry[edge.source] = edge.retry
        self.kind_node = {}
        for node in topology.nodes:
            self.kind_node.setdefault(node.kind, node)
        self.entry = topology.node(topology.entry)
        self.terminal = topology.node(topology.terminals[0])
        self.heap = []
        self.order = 0
        self.now = 0.0
        self.events = 0
        self.records = []
        self.spend_usd = 0.0
        self.completed = 0
        self.admitted = 0
        self.recent_latency = collections.deque(maxlen=256)
        self.recent_outcome = collections.deque(maxlen=256)
        self.trajectory = {n.id: [] for n in topology.nodes}
        self.trajectory_samples = int(trajectory_samples)
        self.ensemble = ENSEMBLE_BONUS.get(self.profile["model_calls_per_request"], 0.05)

    # -- plumbing ------------------------------------------------------------------

    def _push(self, time, kind, payload):
        self.order += 1
        heapq.heappush(self.heap, (time, self.order, kind, payload))

    def _node_for(self, stage, job):
        if stage == "model":
            return self.topology.model_tier(job.tier)
        if stage == "intake":
            return self.entry
        if stage == "deliver":
            return self.terminal
        return self.kind_node.get(_KIND_FOR_STAGE[stage])

    def _rate(self, node, now):
        """Effective refill rate; a rate-limit fault throttles the declared limit in place."""
        multiplier = max(0.02, float(self.dynamics.rate_limit_multiplier(node.id, now)))
        return node.rate_limit_per_second * multiplier

    def _refill(self, station, now):
        node = station.node
        span = max(0.0, now - station.last_refill)
        station.tokens = min(
            float(node.rate_burst), station.tokens + span * self._rate(node, now)
        )
        station.last_refill = now

    def _pump(self, station, now):
        node = station.node
        while station.queue and station.busy < station.servers:
            if node.rate_limit_per_second is not None:
                self._refill(station, now)
                if station.tokens < 1.0:
                    if not station.wake_pending:
                        station.wake_pending = True
                        station.deferrals += 1
                        self._push(
                            now + (1.0 - station.tokens) / self._rate(node, now), 2, node.id
                        )
                    return
            if node.batch_size > 1 and not station.batch_forced:
                if len(station.queue) < node.batch_size:
                    if station.batch_deadline is None:
                        station.batch_deadline = now + node.batch_wait_seconds
                        self._push(station.batch_deadline, 3, node.id)
                    return
            take = min(node.batch_size, len(station.queue)) if node.batch_size > 1 else 1
            if node.batch_size > 1:
                station.batches += 1
                station.batch_forced = False
                station.batch_deadline = None
            if node.rate_limit_per_second is not None:
                station.tokens -= 1.0
            visits = [station.queue.popleft() for _ in range(take)]
            duration = 0.0
            for job, stage, node_id, enqueued, attempt in visits:
                draw = crn_uniform(
                    self.seed, job.request.id, node_id, attempt,
                    job.state.get("round", 0), "service",
                )
                base = service_draw(
                    node.service_law, node.service_seconds, node.service_dispersion, draw
                )
                if node.kind in _WORK_SCALED_KINDS:
                    base *= job.request.work_factor
                base *= self.dynamics.service_multiplier(node_id, now)
                duration = max(duration, base)
            duration *= 1.0 + BATCH_PENALTY * (take - 1)
            station.busy += 1
            self._push(now + duration, 1, (node.id, visits))

    def _offer(self, job, stage, node, now, attempt=0):
        station = self.stations[node.id]
        station.touch(now)
        station.arrivals += 1
        if not self.dynamics.node_available(node.id, now):
            station.unavailable_rejections += 1
            return False
        if station.busy + len(station.queue) >= station.servers + station.capacity:
            station.shed += 1
            return False
        station.admitted += 1
        station.queue.append((job, stage, node.id, now, attempt))
        station.max_queue = max(station.max_queue, len(station.queue))
        self._pump(station, now)
        return True

    # -- request lifecycle ---------------------------------------------------------

    def _enter_stage(self, job, now):
        while True:
            job.index += 1
            if job.index >= len(job.plan):
                return self._finish(job, now)
            if job.index > MAX_STAGES_PER_REQUEST:
                raise Invalid("Request exceeded the declared stage envelope")
            if now - job.start > self.abandon_seconds:
                job.timed_out = True
                return self._finish(job, now)
            stage = job.plan[job.index]
            node = self._node_for(stage, job)
            if node is None:
                continue
            if stage == "model":
                branches = self.profile["model_calls_per_request"]
                job.pending = branches
                job.tiers_used.append(job.tier)
                accepted = 0
                for branch in range(branches):
                    if self._offer(job, stage, node, now, attempt=branch):
                        accepted += 1
                    else:
                        job.pending -= 1
                if accepted == 0:
                    return self._stage_failed(job, stage, node, now)
                return
            if self._offer(job, stage, node, now):
                return
            return self._stage_failed(job, stage, node, now)

    def _stage_failed(self, job, stage, node, now):
        """A station refused the visit. Take the declared fallback, degrade, or shed.

        A model that cannot be reached is not the same as a verifier that cannot be reached.
        Optional stages degrade silently — that is what a real system does when the checker is
        saturated — but a request that never reached any answering node is shed and counted,
        never quietly reported as served.
        """
        if stage == "human" and job.state.get("human_required"):
            job.shed = True
            job.shed_at = node.id
            job.state["shed_reason"] = "no_answering_node_available"
            return self._finish(job, now)
        target = self.fallback.get(node.id)
        if target is not None and job.fallbacks_used < MAX_FALLBACKS:
            fallback_node = self.topology.node(target)
            new_stage = _STAGE_FOR_KIND.get(fallback_node.kind)
            usable = fallback_node.kind == "model" or stage != "model" or new_stage == "human"
            if new_stage is not None and usable:
                job.fallbacks_used += 1
                job.state.setdefault("fallback_path", []).append(f"{node.id}->{target}")
                if fallback_node.kind == "model":
                    job.tier = fallback_node.tier
                elif stage == "model":
                    job.state["human_required"] = True
                job.index -= 1
                job.plan.insert(job.index + 1, new_stage)
                return self._enter_stage(job, now)
        if stage in _OPTIONAL_STAGES and not (
            stage == "human" and job.state.get("human_required")
        ):
            job.state["skipped_" + stage] = True
            return self._enter_stage(job, now)
        job.shed = True
        job.shed_at = node.id
        job.state.setdefault("shed_reason", "station_refused_" + stage)
        return self._finish(job, now)

    def _complete_visit(self, job, stage, node, now, attempt, timed_out):
        request = job.request
        state = job.state
        if timed_out:
            state["timeout_at_" + node.id] = True
        if stage == "model":
            job.pending -= 1
            if job.pending > 0:
                return
            state["ensemble"] = self.profile["model_calls_per_request"]
            outcome = self.dynamics.answer(request, job.tier, now, state)
            state["answer_correct"] = bool(outcome["correct"])
            state["confidence"] = float(outcome["confidence"])
            state["effective_difficulty"] = outcome["effective_difficulty"]
            state["unsafe"] = bool(outcome["unsafe"]) or bool(state.get("unsafe"))
            return self._enter_stage(job, now)
        if stage == "intake":
            return self._enter_stage(job, now)
        if stage == "router":
            return self._route(job, now)
        if stage == "cache":
            result = self.dynamics.cache_lookup(request, now, state)
            state["cache_hit"] = bool(result["hit"])
            state["cache_stale"] = bool(result.get("stale"))
            if result["hit"]:
                state["answer_correct"] = not result.get("stale", False)
                state["confidence"] = 0.85
                state["unsafe"] = bool(result.get("stale") and request.safety_critical)
                job.plan = job.plan[: job.index + 1] + ["deliver"]
            return self._enter_stage(job, now)
        if stage == "retrieve":
            result = self.dynamics.retrieve(request, now, state)
            state["retrieval_helped"] = bool(result["helped"])
            state["retrieval_poisoned"] = bool(result.get("poisoned"))
            state["retrieval_degraded"] = bool(result.get("degraded"))
            if result.get("poisoned"):
                state["contaminated"] = True
                state["contamination_source"] = "retrieval_poisoning"
            return self._enter_stage(job, now)
        if stage == "tool":
            job.tool_attempts += 1
            result = self.dynamics.tool_call(request, now, attempt, state)
            state["tool_ok"] = bool(result["ok"])
            if result.get("injection_reached"):
                state["injection_live"] = True
                state["unsafe"] = True
            if not result["ok"]:
                policy = self.retry.get(node.id)
                limit = policy.max_attempts if policy is not None else 1
                if attempt + 1 < limit and not timed_out:
                    jitter = policy.jitter * crn_uniform(
                        self.seed, request.id, attempt, "retry_jitter"
                    )
                    delay = policy.backoff_seconds * (2**attempt) * (1.0 + jitter)
                    self._push(now + delay, 4, (job, stage, node.id, attempt + 1))
                    return
                state["tool_failed"] = True
            return self._enter_stage(job, now)
        if stage == "verify":
            state["verified"] = True
            state["round"] = state.get("round", 0) + 1
            result = self.dynamics.verify(request, now, state)
            if result.get("caught_unsafe"):
                state["contained_by"] = "verifier"
                state["unsafe_contained"] = True
            if result["reject"]:
                state["verifier_rejections"] = state.get("verifier_rejections", 0) + 1
                if (
                    job.escalations < job.decision.max_escalations
                    and state["round"] <= self.profile["max_verify_rounds"]
                    and len(job.plan) < MAX_STAGES_PER_REQUEST - 4
                ):
                    upgrade = self.policy.escalate(job.view, self._system_view(now), {
                        "tier": job.tier,
                        "confidence": state.get("confidence", 0.5),
                        "verifier_rejected": True,
                        "round": state["round"],
                    })
                    if upgrade is not None and upgrade != job.tier:
                        job.escalations += 1
                        job.tier = upgrade
                        job.plan[job.index + 1 : job.index + 1] = ["model", "verify"]
                    elif job.decision.escalate_human and "human" not in job.plan:
                        job.plan[job.index + 1 : job.index + 1] = ["human"]
                elif job.decision.escalate_human and "human" not in job.plan:
                    job.plan[job.index + 1 : job.index + 1] = ["human"]
            return self._enter_stage(job, now)
        if stage == "human":
            state["human_reviewed"] = True
            result = self.dynamics.human_review(request, now, state)
            if result["corrected"]:
                state["answer_correct"] = True
            if result.get("caught_unsafe"):
                state["contained_by"] = "human_reviewer"
                state["unsafe_contained"] = True
            return self._enter_stage(job, now)
        if stage == "memory":
            result = self.dynamics.memory_write(request, now, state)
            state["memory_written"] = True
            state["memory_contaminated_written"] = bool(result.get("contaminated_written"))
            if result.get("quarantined"):
                state["contained_by"] = state.get("contained_by") or "memory_quarantine"
                state["unsafe_contained"] = True
            return self._enter_stage(job, now)
        return self._enter_stage(job, now)

    def _route(self, job, now):
        decision = self.policy.decide(job.view, self._system_view(now))
        job.decision = decision
        job.tier = decision.model_tier
        if not decision.admit:
            job.shed = True
            job.shed_at = "router"
            job.state["shed_reason"] = "policy_declined"
            return self._finish(job, now)
        if job.request.reads_memory:
            job.state.update(self.dynamics.memory_read(job.request, now, job.state))
            if job.state.get("contaminated"):
                job.state.setdefault("contamination_source", "shared_memory")
        plan = []
        if decision.use_cache and self.kind_node.get("cache") is not None:
            plan.append("cache")
        if decision.use_retrieval and self.kind_node.get("retriever") is not None:
            plan.append("retrieve")
        if decision.use_tool and self.kind_node.get("tool") is not None:
            plan.append("tool")
        plan.append("model")
        if decision.verify and self.kind_node.get("verifier") is not None:
            plan.append("verify")
        if decision.escalate_human and self.kind_node.get("human_reviewer") is not None:
            plan.append("human")
        if self.kind_node.get("memory") is not None:
            plan.append("memory")
        plan.append("deliver")
        job.plan = job.plan[: job.index + 1] + plan
        return self._enter_stage(job, now)

    def _system_view(self, now):
        depth = {}
        busy = {}
        utilisation = {}
        headroom = {}
        for ident, station in self.stations.items():
            depth[ident] = len(station.queue)
            busy[ident] = station.busy
            utilisation[ident] = station.busy / station.servers
            headroom[ident] = max(0, station.servers - station.busy) / station.servers
        outcomes = list(self.recent_outcome)
        latencies = list(self.recent_latency)
        return SystemView(
            now=now,
            queue_depth=depth,
            busy_servers=busy,
            utilisation=utilisation,
            headroom=headroom,
            recent_shed_rate=(
                sum(1 for o in outcomes if o == "shed") / len(outcomes) if outcomes else 0.0
            ),
            recent_timeout_rate=(
                sum(1 for o in outcomes if o == "timeout") / len(outcomes)
                if outcomes
                else 0.0
            ),
            recent_p95_latency_seconds=quantile(latencies, 0.95) if latencies else 0.0,
            admitted=self.admitted,
            completed=self.completed,
            spend_usd=self.spend_usd,
            budget_remaining_usd=(
                float("inf") if self.budget_usd is None else self.budget_usd - self.spend_usd
            ),
            slo_seconds=self.slo_seconds,
        )

    def _finish(self, job, now):
        if job.done:
            return
        job.done = True
        job.finish = now
        latency = now - job.start
        if latency > self.abandon_seconds:
            job.timed_out = True
        state = job.state
        self.spend_usd += job.cost_usd
        if job.shed:
            self.recent_outcome.append("shed")
        elif job.timed_out:
            self.recent_outcome.append("timeout")
        else:
            self.recent_outcome.append("served")
            self.recent_latency.append(latency)
            self.completed += 1
        served = not job.shed and not job.timed_out
        self.records.append(
            {
                "request_id": job.request.id,
                "arrival_time": job.start,
                "completion_time": now,
                "latency_seconds": latency,
                "served": served,
                "shed": job.shed,
                "shed_at": job.shed_at,
                "timed_out": job.timed_out,
                "slo_met": served and latency <= self.slo_seconds,
                "cost_usd": job.cost_usd,
                "energy_joules": job.energy_joules,
                "compute_units": job.compute_units,
                "tier": job.tier,
                "tiers_used": list(job.tiers_used),
                "escalations": job.escalations,
                "fallbacks_used": job.fallbacks_used,
                "tool_attempts": job.tool_attempts,
                "stations_visited": list(job.stations_visited),
                "correct": bool(state.get("answer_correct")) and served,
                "confidence": float(state.get("confidence", 0.5)),
                "verified": bool(state.get("verified")),
                "human_reviewed": bool(state.get("human_reviewed")),
                "cache_hit": bool(state.get("cache_hit")),
                "retrieval_helped": bool(state.get("retrieval_helped")),
                "retrieval_poisoned": bool(state.get("retrieval_poisoned")),
                "tool_ok": bool(state.get("tool_ok")),
                "tool_failed": bool(state.get("tool_failed")),
                "contaminated": bool(state.get("contaminated")),
                "contamination_source": state.get("contamination_source"),
                "unsafe_attempted": bool(state.get("unsafe")),
                "unsafe_contained": bool(state.get("unsafe_contained")),
                "contained_by": state.get("contained_by"),
                "memory_contaminated_written": bool(state.get("memory_contaminated_written")),
                "verifier_rejections": int(state.get("verifier_rejections", 0)),
                "decision": (
                    {
                        "model_tier": job.decision.model_tier,
                        "use_retrieval": job.decision.use_retrieval,
                        "use_tool": job.decision.use_tool,
                        "verify": job.decision.verify,
                        "escalate_human": job.decision.escalate_human,
                        "admit": job.decision.admit,
                    }
                    if job.decision is not None
                    else None
                ),
            }
        )

    def _depart(self, node_id, visits, now):
        station = self.stations[node_id]
        station.touch(now)
        station.busy -= 1
        node = station.node
        for job, stage, ident, enqueued, attempt in visits:
            station.completed += 1
            job.cost_usd += node.cost_per_call_usd
            job.energy_joules += node.energy_per_call_joules
            job.compute_units += node.compute_units
            job.stations_visited.append(ident)
            timed_out = (now - enqueued) > node.timeout_seconds
            if timed_out:
                station.timeouts += 1
            if job.done:
                continue
            self._complete_visit(job, stage, node, now, attempt, timed_out)
        self._pump(station, now)

    def run(self):
        for request in self.trace.requests:
            self._push(request.arrival_time, 0, request)
        horizon = max(self.trace.horizon_seconds, 1e-6)
        if self.trajectory_samples > 0:
            step = horizon / self.trajectory_samples
            moment = step
            while moment <= horizon + 1e-9:
                self._push(moment, 5, None)
                moment += step
        while self.heap:
            if self.events >= self.max_events:
                raise Invalid("System simulation exceeded the declared event envelope")
            moment, _, kind, payload = heapq.heappop(self.heap)
            self.now = moment
            self.events += 1
            if kind == 0:
                job = _Job(
                    request=payload,
                    view=view_of(payload),
                    start=payload.arrival_time,
                    plan=["intake", "router"],
                    index=-1,
                )
                self.admitted += 1
                self._enter_stage(job, moment)
            elif kind == 1:
                self._depart(payload[0], payload[1], moment)
            elif kind == 2:
                station = self.stations[payload]
                station.wake_pending = False
                station.touch(moment)
                self._pump(station, moment)
            elif kind == 3:
                station = self.stations[payload]
                station.batch_forced = True
                station.touch(moment)
                self._pump(station, moment)
                station.batch_forced = False
            elif kind == 4:
                job, stage, node_id, attempt = payload
                if job.done:
                    continue
                node = self.topology.node(node_id)
                if not self._offer(job, stage, node, moment, attempt):
                    self._stage_failed(job, stage, node, moment)
            elif kind == 5:
                for ident, station in self.stations.items():
                    station.touch(moment)
                    self.trajectory[ident].append((round(moment, 6), len(station.queue)))
        for station in self.stations.values():
            station.touch(self.now)
        return self._result()

    def _result(self):
        observed = max(self.now, 1e-9)
        stations = {}
        for ident, station in self.stations.items():
            stations[ident] = {
                "node_id": ident,
                "kind": station.node.kind,
                "tier": station.node.tier,
                "servers": station.servers,
                "queue_capacity": station.capacity,
                "arrivals": station.arrivals,
                "admitted": station.admitted,
                "shed": station.shed,
                "unavailable_rejections": station.unavailable_rejections,
                "completed": station.completed,
                "timeouts": station.timeouts,
                "rate_limit_deferrals": station.deferrals,
                "batches": station.batches,
                "utilisation": station.busy_integral / observed / station.servers,
                "mean_queue_depth": station.queue_integral / observed,
                "max_queue_depth": station.max_queue,
                "shed_fraction": station.shed / max(station.arrivals, 1),
                "mean_in_station": (station.busy_integral + station.queue_integral) / observed,
            }
        served = [r for r in self.records if r["served"]]
        latencies = [r["latency_seconds"] for r in served]
        occupancy = math.fsum(s["mean_in_station"] for s in stations.values())
        mean_latency = math.fsum(latencies) / len(latencies) if latencies else 0.0
        effective_rate = len(served) / observed
        return {
            "records": self.records,
            "stations": stations,
            "latency": latency_stats(latencies),
            "queue_trajectory": {
                k: v for k, v in self.trajectory.items() if any(d for _, d in v)
            },
            "arrivals": len(self.trace.requests),
            "served": len(served),
            "shed_count": sum(1 for r in self.records if r["shed"]),
            "timeout_count": sum(1 for r in self.records if r["timed_out"]),
            "shed_fraction": (
                sum(1 for r in self.records if r["shed"]) / max(len(self.records), 1)
            ),
            "slo_seconds": self.slo_seconds,
            "slo_met_fraction": (
                sum(1 for r in self.records if r["slo_met"]) / max(len(self.records), 1)
            ),
            "throughput_per_second": effective_rate,
            "little_law_network": {
                "mean_in_system_observed": occupancy,
                "mean_in_system_little": little_law(effective_rate, mean_latency),
                "note": "Fork-join branches place one request at several stations at once, so "
                "the network identity is reported for inspection and is exact only per station.",
            },
            "little_law_by_station": {
                ident: {
                    "mean_in_station_observed": stats["mean_in_station"],
                    "arrival_rate": station.admitted / observed,
                }
                for (ident, stats), station in zip(stations.items(), self.stations.values())
            },
            "observed_seconds": observed,
            "events": self.events,
            "architecture_profile": self.profile,
            "topology_digest": digest(self.topology.model_dump()),
            "trace_digest": self.trace.trace_digest,
            "arrival_time_digest": digest(
                [round(t, 9) for t in self.trace.arrival_times()]
            ),
            "arrival_time_head": [round(t, 6) for t in self.trace.arrival_times()[:8]],
            "policy": getattr(self.policy, "name", "unnamed"),
            "dynamics": self.dynamics.report(),
            "seed": self.seed,
            "scope": SCOPE,
        }


def simulate_system(
    topology,
    trace,
    policy,
    seed=2026,
    dynamics=None,
    slo_seconds=10.0,
    abandon_seconds=120.0,
    budget_usd=None,
    max_events=MAX_EVENTS,
    trajectory_samples=120,
):
    """Run one policy over one frozen trace and return timing, resource and outcome records.

    The same `trace` object handed to several policies is what makes the arms paired: arrival
    times, difficulties and service draws are identical, so a difference between arms is the
    policy and not the weather.
    """
    if not isinstance(topology, Topology):
        topology = Topology.model_validate(topology)
    if not hasattr(policy, "decide") or not hasattr(policy, "escalate"):
        raise Invalid("A policy must expose decide() and escalate()")
    if not 0.001 <= float(slo_seconds) <= MAX_SIM_SECONDS:
        raise Invalid("slo_seconds must lie in (0, 86400]")
    if not float(slo_seconds) <= float(abandon_seconds) <= MAX_SIM_SECONDS:
        raise Invalid("abandon_seconds must be at least slo_seconds and at most 86400")
    if not 1000 <= int(max_events) <= MAX_EVENTS:
        raise Invalid("max_events must lie in [1000, 400000]")
    if not 0 <= int(trajectory_samples) <= 1000:
        raise Invalid("trajectory_samples must lie in [0, 1000]")
    if len(trace.requests) > MAX_ARRIVALS:
        raise Invalid("Trace exceeds the declared arrival envelope")
    if trace.horizon_seconds > MAX_SIM_SECONDS:
        raise Invalid("Trace horizon exceeds the declared simulation envelope")
    engine = _Network(
        topology,
        trace,
        policy,
        seed,
        dynamics if dynamics is not None else IdealDynamics(topology, seed),
        slo_seconds,
        abandon_seconds,
        budget_usd,
        max_events,
        trajectory_samples,
    )
    return engine.run()
