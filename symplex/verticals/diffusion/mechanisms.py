"""Rival diffusion mechanisms behind one node-hazard interface.

Every mechanism is expressed as a per-step activation probability for each
inactive node, so all seven consume the same random stream in the same order.
That makes common random numbers across mechanisms real rather than nominal: the
same node is "lucky" at the same step under contagion, under a shared exogenous
driver and under homophily, and the difference between their outputs is the
mechanism rather than the draw.

Two of the seven mechanisms transmit nothing at all. `exogenous_shock` is a
common external driver and `homophily` is a trait ordering; both can reproduce an
S-shaped adoption curve without any node influencing any other. They are here
because a diffusion study that omits them will confidently misattribute
correlated adoption to contagion.

Simulating a mechanism demonstrates its dynamics. It is not evidence that any
real adoption, infection, compromise or failure process obeys it.
"""

from __future__ import annotations

import dataclasses
from typing import Annotated, ClassVar, Literal

import numpy as np
from pydantic import ConfigDict, Field

from symplex.core.contracts import Invalid, digest
from symplex.modeling.complex_system import ClosedContract, Text
from symplex.verticals.diffusion.networks import MAX_NODES, Network

MAX_HORIZON = 400
MAX_STREAM_CELLS = 4_000_000
MAX_REPLICATES = 400
MAX_SEEDS = 500

SIMULATION_SCOPE = (
    "simulated dynamics of a stated mechanism on a stated graph under an explicit "
    "seed; not an observation, not a calibrated forecast, and not evidence that a "
    "real process follows this mechanism"
)


class ParameterContract(ClosedContract):
    """Frozen, closed per-mechanism parameter set.

    Numeric coercion is allowed so integer literals bind to float parameters;
    every other field stays strict and unknown fields are rejected.
    """

    model_config = ConfigDict(
        extra="forbid",
        strict=False,
        allow_inf_nan=False,
        frozen=True,
        str_strip_whitespace=True,
    )


Probability = Annotated[float, Field(ge=0.0, le=1.0)]
Rate = Annotated[float, Field(ge=0.0, le=20.0)]


class IndependentCascadeParams(ParameterContract):
    p: Probability = 0.15


class LinearThresholdParams(ParameterContract):
    threshold: Annotated[float, Field(gt=0.0, le=1.0)] = 0.3
    heterogeneity: Probability = 0.4


class ComplexContagionParams(ParameterContract):
    k: Annotated[int, Field(ge=1, le=20)] = 2
    p: Annotated[float, Field(gt=0.0, le=1.0)] = 0.5


class HawkesParams(ParameterContract):
    background: Annotated[float, Field(ge=0.0, le=1.0)] = 0.002
    excitation: Rate = 0.25
    decay: Annotated[float, Field(gt=0.0, le=10.0)] = 0.6


class TrustMemoryParams(ParameterContract):
    gain: Rate = 0.6
    memory_decay: Probability = 0.5
    trust_exponent: Annotated[float, Field(gt=0.0, le=8.0)] = 2.0


class ExogenousShockParams(ParameterContract):
    saturation: Annotated[float, Field(gt=0.0, le=1.0)] = 0.6
    midpoint: Annotated[float, Field(ge=0.0, le=float(MAX_HORIZON))] = 8.0
    steepness: Annotated[float, Field(gt=0.0, le=5.0)] = 0.6
    shape: Annotated[float, Field(gt=0.0, le=8.0)] = 1.0
    baseline: Probability = 0.0


class HomophilyParams(ParameterContract):
    base_rate: Annotated[float, Field(gt=0.0, le=1.0)] = 0.03
    trait_sensitivity: Annotated[float, Field(ge=0.0, le=12.0)] = 6.0
    ramp: Annotated[float, Field(ge=0.0, le=2.0)] = 0.08


@dataclasses.dataclass(frozen=True)
class RandomStreams:
    """Pre-drawn randomness shared by every mechanism on one network and horizon."""

    node_uniform: np.ndarray
    node_threshold: np.ndarray
    node_trait: np.ndarray
    edge_trust: np.ndarray
    seed: int
    horizon: int
    nodes: int


def streams(network, seed, horizon):
    """Draw the common random numbers in a fixed order so mechanisms stay paired."""
    if not isinstance(network, Network):
        raise Invalid("Expected a Network")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= int(seed) < 2**32:
        raise Invalid("Seed must be an integer in [0, 2**32)")
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)):
        raise Invalid("horizon must be an integer")
    horizon = int(horizon)
    if not 1 <= horizon <= MAX_HORIZON:
        raise Invalid(f"horizon must lie in [1, {MAX_HORIZON}]")
    if (horizon + 1) * network.n > MAX_STREAM_CELLS:
        raise Invalid("Random-stream envelope exceeded: reduce horizon or node count")
    rng = np.random.default_rng(int(seed))
    return RandomStreams(
        node_uniform=rng.random((horizon + 1, network.n)),
        node_threshold=rng.random(network.n),
        node_trait=rng.random(network.n),
        edge_trust=rng.random(max(network.edge_count, 1)),
        seed=int(seed),
        horizon=horizon,
        nodes=network.n,
    )


@dataclasses.dataclass(frozen=True)
class CascadeResult:
    """Per-node activation times plus the trajectory and exposure record."""

    mechanism: str
    parameters: dict
    activation_time: np.ndarray
    exposures_at_adoption: np.ndarray
    neighbour_preceded: np.ndarray
    size_trajectory: np.ndarray
    increments: np.ndarray
    seeds: np.ndarray
    blocked: np.ndarray
    horizon: int
    nodes: int
    stream_seed: int
    exposure_scale: float

    @property
    def adopters(self):
        return np.isfinite(self.activation_time)

    @property
    def final_size(self):
        return int(self.adopters.sum())

    @property
    def final_fraction(self):
        return float(self.final_size / self.nodes)

    def to_dict(self):
        return {
            "mechanism": self.mechanism,
            "parameters": dict(self.parameters),
            "final_size": self.final_size,
            "final_fraction": self.final_fraction,
            "size_trajectory": self.size_trajectory.tolist(),
            "increments": self.increments.tolist(),
            "peak_step": int(np.argmax(self.increments)) + 1 if self.increments.size else 0,
            "seed_count": int(self.seeds.size),
            "blocked_count": int(self.blocked.sum()),
            "stream_seed": self.stream_seed,
            "exposure_scale": self.exposure_scale,
            "horizon": self.horizon,
            "scope": SIMULATION_SCOPE,
        }


@dataclasses.dataclass
class _Context:
    """State handed to a mechanism at one step."""

    step: int
    network: Network
    streams: RandomStreams
    active: np.ndarray
    transmitting: np.ndarray
    newly_transmitting: np.ndarray
    exposure_count: np.ndarray
    new_exposure_count: np.ndarray
    exposure_scale: float
    state: dict


class Mechanism:
    """One rival explanation. Subclasses supply `prepare` and `hazard`."""

    name: ClassVar[str] = "mechanism"
    params_model: ClassVar[type] = ParameterContract
    transmits: ClassVar[bool] = True
    submodular: ClassVar[bool] = False
    scaled_parameter: ClassVar[str | None] = None
    summary: ClassVar[str] = ""
    identifiability_note: ClassVar[str] = ""

    def parse(self, params):
        if params is None:
            return self.params_model()
        if isinstance(params, self.params_model):
            return params
        if isinstance(params, ParameterContract):
            raise Invalid(f"Parameters for {self.name} must be {self.params_model.__name__}")
        if not isinstance(params, dict):
            raise Invalid("Mechanism parameters must be a dict or a parameter contract")
        try:
            return self.params_model.model_validate(params)
        except Exception as error:  # pydantic ValidationError, kept as Invalid at the boundary.
            raise Invalid(f"Invalid {self.name} parameters: {error}") from None

    def prepare(self, network, params, stream_set, exposure_scale):
        return {}

    def hazard(self, params, ctx):
        raise NotImplementedError

    def contract(self):
        fields = {}
        for name, field in self.params_model.model_fields.items():
            bounds = {}
            for item in field.metadata:
                for key in ("ge", "gt", "le", "lt"):
                    value = getattr(item, key, None)
                    if value is not None:
                        bounds[key] = value
            fields[name] = {"default": field.default, "bounds": bounds}
        return {
            "mechanism": self.name,
            "summary": self.summary,
            "transmits": self.transmits,
            "submodular_spread": self.submodular,
            "scaled_parameter": self.scaled_parameter,
            "parameters": fields,
            "identifiability_note": self.identifiability_note,
            "scope": "declared parameter domain of a proposed mechanism; the bounds "
            "are modeling choices, not measured quantities",
        }


class IndependentCascade(Mechanism):
    name = "independent_cascade"
    params_model = IndependentCascadeParams
    submodular = True
    scaled_parameter = "p"
    summary = "Each newly activated node gets one attempt per edge at probability p."
    identifiability_note = (
        "Simple contagion. Aggregate adoption curves from this mechanism are "
        "reproducible by a shared exogenous driver; only node-level or edge-level "
        "observation separates them."
    )

    def hazard(self, params, ctx):
        probability = min(1.0, params.p * ctx.exposure_scale)
        return 1.0 - np.power(1.0 - probability, ctx.new_exposure_count)


class LinearThreshold(Mechanism):
    name = "linear_threshold"
    params_model = LinearThresholdParams
    submodular = True
    scaled_parameter = None
    summary = "A node activates once the active fraction of its neighbours reaches its threshold."
    identifiability_note = (
        "Deterministic given thresholds, so replicate variation comes only from "
        "the seed set. Heterogeneity is not identified from a single cascade."
    )

    def prepare(self, network, params, stream_set, exposure_scale):
        thresholds = params.threshold + params.heterogeneity * (stream_set.node_threshold - 0.5)
        return {"thresholds": np.clip(thresholds, 0.01, 1.0)}

    def hazard(self, params, ctx):
        degree = np.maximum(ctx.network.degree.astype(np.float64), 1.0)
        fraction = np.minimum(ctx.exposure_count * ctx.exposure_scale, degree) / degree
        return (fraction >= ctx.state["thresholds"]).astype(np.float64)


class ComplexContagion(Mechanism):
    name = "complex_contagion"
    params_model = ComplexContagionParams
    submodular = False
    scaled_parameter = "p"
    summary = "Reinforcement: k distinct active neighbours are required before adoption is possible."
    identifiability_note = (
        "Not submodular for k > 1, so greedy influence maximization carries no "
        "approximation guarantee here. Needs a locally dense seed cluster to start, "
        "which is the qualitative signature that separates it from simple contagion."
    )

    def hazard(self, params, ctx):
        eligible = ctx.exposure_count >= params.k
        return eligible.astype(np.float64) * min(1.0, params.p * ctx.exposure_scale)


class Hawkes(Mechanism):
    name = "hawkes"
    params_model = HawkesParams
    submodular = False
    scaled_parameter = "excitation"
    summary = "Self-exciting intensity with an exponentially decaying kernel over neighbours."
    identifiability_note = (
        "Background rate and excitation trade off against each other: a high "
        "background with weak excitation and a low background with strong "
        "excitation produce similar aggregate curves."
    )

    def prepare(self, network, params, stream_set, exposure_scale):
        return {"excitation_state": np.zeros(network.n)}

    def hazard(self, params, ctx):
        decayed = ctx.state["excitation_state"] * np.exp(-params.decay)
        gain = params.excitation * ctx.exposure_scale
        ctx.state["excitation_state"] = decayed + gain * ctx.new_exposure_count
        intensity = params.background + ctx.state["excitation_state"]
        return 1.0 - np.exp(-intensity)


class TrustWeightedMemory(Mechanism):
    name = "trust_memory"
    params_model = TrustMemoryParams
    submodular = False
    scaled_parameter = "gain"
    summary = "Heterogeneous edge trust accumulates into a decaying memory of exposure."
    identifiability_note = (
        "Edge trust is latent. Without an independent measurement of tie strength "
        "the trust weights are absorbed into the gain and are not identified."
    )

    def prepare(self, network, params, stream_set, exposure_scale):
        weights = np.zeros((network.n, network.n))
        if network.edge_count:
            trust = np.power(stream_set.edge_trust[: network.edge_count], params.trust_exponent)
            rows = network.edges[:, 0]
            cols = network.edges[:, 1]
            weights[rows, cols] = trust
            weights[cols, rows] = trust
        return {"weights": weights, "memory": np.zeros(network.n)}

    def hazard(self, params, ctx):
        arrivals = ctx.state["weights"] @ ctx.newly_transmitting.astype(np.float64)
        memory = ctx.state["memory"] * params.memory_decay
        memory = memory + params.gain * ctx.exposure_scale * arrivals
        ctx.state["memory"] = memory
        return 1.0 - np.exp(-memory)


class ExogenousShock(Mechanism):
    name = "exogenous_shock"
    params_model = ExogenousShockParams
    transmits = False
    submodular = False
    scaled_parameter = "saturation"
    summary = (
        "A common external driver or algorithmic amplification: every node faces the "
        "same time-varying hazard and no node transmits to any other."
    )
    identifiability_note = (
        "The null that naive cascade analysis mistakes for contagion. A logistic "
        "driver reproduces the S-curve of simple contagion exactly at the aggregate "
        "level; separation requires per-node adoption times on a known graph, "
        "edge-level exposure logging, or randomized seeding."
    )

    def prepare(self, network, params, stream_set, exposure_scale):
        steps = np.arange(0, stream_set.horizon + 1, dtype=np.float64)
        saturation = min(1.0, params.saturation * exposure_scale)
        # Generalized-logistic (Richards) driver: shape 1 is the plain logistic and
        # other values skew the rise, because the profile of an external driver is
        # not known a priori and a rigid logistic understates the confound.
        cumulative = saturation * np.power(
            1.0 + np.exp(-params.steepness * (steps - params.midpoint)), -1.0 / params.shape
        )
        previous = np.concatenate([[0.0], cumulative[:-1]])
        hazard = (cumulative - previous) / np.maximum(1.0 - previous, 1e-9)
        hazard = np.clip(hazard + params.baseline * exposure_scale, 0.0, 1.0)
        return {"schedule": hazard}

    def hazard(self, params, ctx):
        return np.full(ctx.network.n, float(ctx.state["schedule"][ctx.step]))


class Homophily(Mechanism):
    name = "homophily"
    params_model = HomophilyParams
    transmits = False
    submodular = False
    scaled_parameter = "base_rate"
    summary = (
        "Similar nodes adopt at similar times because of a shared trait, with no "
        "influence between them."
    )
    identifiability_note = (
        "The classic confound. On an assortative graph, trait-ordered adoption "
        "produces neighbour-preceded adoptions and edge-wise time concordance that "
        "look exactly like transmission. Node attributes must be observed to "
        "separate it, and unobserved traits cannot be ruled out."
    )

    def prepare(self, network, params, stream_set, exposure_scale):
        trait = network.attributes if network.attributes is not None else stream_set.node_trait
        trait = np.asarray(trait, dtype=np.float64)
        return {"trait": np.clip(trait, 0.0, 1.0)}

    def hazard(self, params, ctx):
        propensity = np.exp(params.trait_sensitivity * (ctx.state["trait"] - 0.5))
        rate = params.base_rate * ctx.exposure_scale * propensity * (1.0 + params.ramp * ctx.step)
        return np.clip(rate, 0.0, 1.0)


_MECHANISMS = {
    m.name: m()
    for m in (
        IndependentCascade,
        LinearThreshold,
        ComplexContagion,
        Hawkes,
        TrustWeightedMemory,
        ExogenousShock,
        Homophily,
    )
}
MECHANISM_NAMES = tuple(_MECHANISMS)
TRANSMITTING_MECHANISMS = tuple(n for n, m in _MECHANISMS.items() if m.transmits)
NON_TRANSMITTING_MECHANISMS = tuple(n for n, m in _MECHANISMS.items() if not m.transmits)


def get_mechanism(name):
    """Look up a mechanism by name; unknown names fail closed."""
    if isinstance(name, Mechanism):
        return name
    if name not in _MECHANISMS:
        raise Invalid(
            "unsupported_operation: unknown mechanism " + str(name) + "; available: "
            + ", ".join(MECHANISM_NAMES)
        )
    return _MECHANISMS[name]


def default_params(name):
    """Default parameter dict for one mechanism."""
    mechanism = get_mechanism(name)
    return dict(mechanism.params_model().model_dump(), **{})


def parameter_contracts(names=None):
    """Declared parameter domains and identifiability notes for the candidate set."""
    names = tuple(names or MECHANISM_NAMES)
    return {
        "mechanisms": {name: get_mechanism(name).contract() for name in names},
        "common_random_numbers": "All mechanisms consume one shared node-uniform "
        "stream in the same order, so paired comparisons across mechanisms differ "
        "by mechanism rather than by draw.",
        "scope": "declared contracts only; no parameter here has been estimated "
        "from a real observed diffusion process",
    }


PARAMETER_GRIDS = {
    "independent_cascade": {"p": (0.03, 0.07, 0.12, 0.2, 0.32, 0.5)},
    "linear_threshold": {"threshold": (0.1, 0.2, 0.3, 0.45), "heterogeneity": (0.2, 0.6)},
    "complex_contagion": {"k": (2, 3), "p": (0.15, 0.35, 0.7)},
    "hawkes": {"excitation": (0.05, 0.15, 0.35, 0.7), "decay": (0.3, 1.0)},
    "trust_memory": {"gain": (0.2, 0.5, 1.0, 2.0), "memory_decay": (0.2, 0.7)},
    "exogenous_shock": {
        "saturation": (0.2, 0.45, 0.7, 0.95),
        "midpoint": (0.0, 3.0, 8.0, 14.0),
        "steepness": (0.35, 0.8),
        "shape": (0.4, 1.0, 2.5),
    },
    "homophily": {
        "base_rate": (0.01, 0.03, 0.08, 0.18),
        "trait_sensitivity": (2.0, 6.0),
        "ramp": (0.0, 0.15),
    },
}


def parameter_grid(name, resolution=None):
    """Bounded search grid for one mechanism, thinned to `resolution` values per axis."""
    mechanism = get_mechanism(name)
    axes = PARAMETER_GRIDS[mechanism.name]
    if resolution is not None:
        if isinstance(resolution, bool) or not isinstance(resolution, (int, np.integer)):
            raise Invalid("resolution must be an integer")
        resolution = int(resolution)
        if not 1 <= resolution <= 12:
            raise Invalid("resolution must lie in [1, 12]")
        thinned = {}
        for key, values in axes.items():
            if len(values) <= resolution:
                thinned[key] = values
            else:
                index = np.linspace(0, len(values) - 1, resolution).round().astype(int)
                thinned[key] = tuple(values[i] for i in sorted(set(index.tolist())))
        axes = thinned
    keys = sorted(axes)
    points = [{}]
    for key in keys:
        points = [dict(point, **{key: value}) for point in points for value in axes[key]]
    base = default_params(mechanism.name)
    return [dict(base, **point) for point in points]


def _seed_array(seeds, n):
    array = np.asarray(list(seeds), dtype=np.int64)
    if array.ndim != 1 or array.size == 0:
        raise Invalid("At least one seed node is required")
    if array.size > MAX_SEEDS:
        raise Invalid(f"At most {MAX_SEEDS} seed nodes are permitted")
    if array.min() < 0 or array.max() >= n:
        raise Invalid("Seed node indices out of range")
    if np.unique(array).size != array.size:
        raise Invalid("Seed nodes must be distinct")
    return array


def _mask(values, n, label):
    if values is None:
        return np.zeros(n, dtype=bool)
    array = np.asarray(values)
    if array.dtype == bool:
        if array.shape != (n,):
            raise Invalid(f"{label} mask must cover every node")
        return array.copy()
    array = array.astype(np.int64)
    if array.ndim != 1 or (array.size and (array.min() < 0 or array.max() >= n)):
        raise Invalid(f"{label} indices out of range")
    mask = np.zeros(n, dtype=bool)
    mask[array] = True
    return mask


def simulate(
    network,
    mechanism,
    params=None,
    seeds=(0,),
    seed=0,
    horizon=40,
    blocked=None,
    monitored=None,
    quarantine_delay=None,
    exposure_scale=1.0,
    stream_set=None,
):
    """Run one mechanism to `horizon` steps and return the full activation record.

    `blocked` nodes can never adopt (immunization or corrective pre-treatment).
    `monitored` nodes stop transmitting `quarantine_delay` steps after they adopt,
    which is how a monitoring budget shows up in the dynamics. `exposure_scale`
    multiplies whichever parameter that mechanism declares as its amplification
    dial. Supplying `stream_set` reuses drawn randomness and ignores `seed`.
    """
    if not isinstance(network, Network):
        raise Invalid("Expected a Network")
    if network.n > MAX_NODES:
        raise Invalid("Network exceeds the node envelope")
    mechanism = get_mechanism(mechanism)
    parameters = mechanism.parse(params)
    if isinstance(exposure_scale, bool) or not isinstance(exposure_scale, (int, float, np.floating)):
        raise Invalid("exposure_scale must be a number")
    exposure_scale = float(exposure_scale)
    if not 0.0 <= exposure_scale <= 20.0:
        raise Invalid("exposure_scale must lie in [0, 20]")
    if stream_set is None:
        stream_set = streams(network, seed, horizon)
    if not isinstance(stream_set, RandomStreams):
        raise Invalid("stream_set must come from streams()")
    if stream_set.nodes != network.n:
        raise Invalid("Random streams were drawn for a different network")
    horizon = int(min(int(horizon), stream_set.horizon))
    if horizon < 1:
        raise Invalid("horizon must be at least 1")

    n = network.n
    seed_nodes = _seed_array(seeds, n)
    blocked_mask = _mask(blocked, n, "blocked")
    monitored_mask = _mask(monitored, n, "monitored")
    if quarantine_delay is None:
        quarantine_delay = 0 if monitored_mask.any() else None
    if quarantine_delay is not None:
        if isinstance(quarantine_delay, bool) or not isinstance(quarantine_delay, (int, np.integer)):
            raise Invalid("quarantine_delay must be an integer")
        quarantine_delay = int(quarantine_delay)
        if not 0 <= quarantine_delay <= horizon:
            raise Invalid("quarantine_delay must lie in [0, horizon]")
    blocked_mask[seed_nodes] = False

    active = np.zeros(n, dtype=bool)
    active[seed_nodes] = True
    activation_time = np.full(n, np.inf)
    activation_time[seed_nodes] = 0.0
    exposures_at_adoption = np.full(n, np.nan)
    exposures_at_adoption[seed_nodes] = 0.0
    neighbour_preceded = np.zeros(n, dtype=bool)
    quarantined = np.zeros(n, dtype=bool)

    state = mechanism.prepare(network, parameters, stream_set, exposure_scale)
    trajectory = np.zeros(horizon + 1, dtype=np.int64)
    trajectory[0] = int(active.sum())
    adjacency = network.adjacency
    previous_transmitting = np.zeros(n, dtype=bool)
    newly = active.copy()

    for step in range(1, horizon + 1):
        if quarantine_delay is not None:
            release = monitored_mask & active & (activation_time + quarantine_delay <= step - 1)
            quarantined |= release
        transmitting = active & ~quarantined
        newly_transmitting = newly & ~quarantined
        exposure_count = adjacency @ transmitting.astype(np.float64)
        new_exposure_count = adjacency @ newly_transmitting.astype(np.float64)
        ctx = _Context(
            step=step,
            network=network,
            streams=stream_set,
            active=active,
            transmitting=transmitting,
            newly_transmitting=newly_transmitting,
            exposure_count=exposure_count,
            new_exposure_count=new_exposure_count,
            exposure_scale=exposure_scale,
            state=state,
        )
        hazard = np.clip(np.asarray(mechanism.hazard(parameters, ctx), dtype=np.float64), 0.0, 1.0)
        eligible = (~active) & (~blocked_mask)
        newly = eligible & (stream_set.node_uniform[step] < hazard)
        if newly.any():
            activation_time[newly] = float(step)
            exposures_at_adoption[newly] = exposure_count[newly]
            neighbour_preceded[newly] = exposure_count[newly] > 0
            active |= newly
        previous_transmitting = transmitting
        trajectory[step] = int(active.sum())

    del previous_transmitting
    return CascadeResult(
        mechanism=mechanism.name,
        parameters=parameters.model_dump(),
        activation_time=activation_time,
        exposures_at_adoption=exposures_at_adoption,
        neighbour_preceded=neighbour_preceded,
        size_trajectory=trajectory,
        increments=np.diff(trajectory),
        seeds=seed_nodes,
        blocked=blocked_mask,
        horizon=horizon,
        nodes=n,
        stream_seed=stream_set.seed,
        exposure_scale=exposure_scale,
    )


def replicate(
    network,
    mechanism,
    params=None,
    seeds=(0,),
    seed=0,
    horizon=40,
    replicates=16,
    **kwargs,
):
    """Run `replicates` independent draws with stream seeds `seed`, `seed+1`, ....

    Passing the same `seed` to two mechanisms pairs them replicate by replicate.
    """
    if isinstance(replicates, bool) or not isinstance(replicates, (int, np.integer)):
        raise Invalid("replicates must be an integer")
    replicates = int(replicates)
    if not 1 <= replicates <= MAX_REPLICATES:
        raise Invalid(f"replicates must lie in [1, {MAX_REPLICATES}]")
    return [
        simulate(
            network,
            mechanism,
            params=params,
            seeds=seeds,
            seed=int(seed) + index,
            horizon=horizon,
            **kwargs,
        )
        for index in range(replicates)
    ]


def compare_under_common_randomness(network, candidates, seeds=(0,), seed=0, horizon=40):
    """Run every candidate against one shared random stream and report the spread.

    Identical draws with different outcomes isolate the mechanism. This is a
    demonstration of paired comparison, not evidence about any real process.
    """
    if not isinstance(candidates, (list, tuple)) or not 1 <= len(candidates) <= len(MECHANISM_NAMES):
        raise Invalid("Supply between 1 and 7 candidate mechanisms")
    stream_set = streams(network, seed, horizon)
    results = {}
    for candidate in candidates:
        name = candidate if isinstance(candidate, str) else candidate.get("mechanism")
        params = None if isinstance(candidate, str) else candidate.get("parameters")
        result = simulate(
            network, name, params=params, seeds=seeds, horizon=horizon, stream_set=stream_set
        )
        results[result.mechanism] = result
    sizes = {name: result.final_fraction for name, result in results.items()}
    return {
        "results": results,
        "final_fraction": sizes,
        "spread": float(max(sizes.values()) - min(sizes.values())) if sizes else 0.0,
        "stream_seed": int(seed),
        "stream_digest": digest({"seed": int(seed), "horizon": int(horizon), "nodes": network.n}),
        "scope": SIMULATION_SCOPE
        + "; a spread across mechanisms under one shared stream measures model "
        "disagreement, not which mechanism is correct",
    }
