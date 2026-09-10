"""Host-owned quality-diversity archive over injected, measured behaviour descriptors.

Coverage, QD score, entropy and novelty describe where a search went. They are
search bookkeeping and never evidence that a retained candidate is correct,
calibrated, cheap enough, or safe to act on. An occupied cell records that one
candidate reached one measured behaviour under one evaluation protocol; it is
not a posterior probability, a replication, or an independent confirmation.

The archive never measures anything itself. Descriptors, objectives and
feasibility arrive from injected host functions so measurement stays in
evaluation code and can be replaced without touching the search machinery.
Every stochastic choice takes an explicit seed; nothing here calls a model, the
network, or a subprocess.
"""

import math
import random

from symplex.core.contracts import Invalid, canonical, number

VERSION = "measured-behaviour-quality-diversity-archive-v1"
SCALARIZATION = "bounded_normalized_weighted_sum_v1"

ACCEPTING_OUTCOMES = ("new_cell", "dominates_existing", "nondominated_added")
REJECTING_OUTCOMES = ("dominated_rejected", "infeasible")
OUTCOMES = ACCEPTING_OUTCOMES + REJECTING_OUTCOMES
RETENTION_REASONS = (
    "dominated_rejected",
    "infeasible",
    "crowding_pruned",
    "displaced_by_domination",
    "front_capacity_pruned",
)
SELECTION_STRATEGIES = ("curiosity", "uniform", "least_visited", "novelty")
DIRECTIONS = ("minimize", "maximize")
OBJECTIVE_FIELDS = {"name", "direction", "lower", "upper", "weight", "units", "meaning"}

# 12 dimensions accommodates the measured-behaviour descriptor at its finest
# declared resolution; MAX_REACHABLE_CELLS is the binding limit on grid size.
MAX_DESCRIPTOR_DIMENSIONS = 12
MAX_BIN_EDGES = 129
MAX_REACHABLE_CELLS = 200_000
MAX_FRONT_CAPACITY = 32
MAX_OBJECTIVES = 8
MAX_RETAINED_REJECTED = 20_000
MAX_HISTORY = 10_000
MAX_IDENTIFIER = 200
MAX_GENOME_CHARS = 200_000
EPSILON = 1e-9

SCOPE = (
    "Search bookkeeping over one declared behaviour space and one declared "
    "objective vector. Coverage, QD score, archive entropy and novelty measure "
    "how the search moved, never whether any retained candidate is scientifically "
    "correct, calibrated, or permitted to inform a decision. An occupied cell is "
    "one measured behaviour under one evaluation protocol, not a posterior "
    "probability or an independent confirmation. Descriptors and objectives are "
    "supplied by injected host functions; this module does not verify that they "
    "measure what their names claim."
)


def _identifier(value, field):
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_IDENTIFIER:
        raise Invalid(f"invalid_{field}: expected a nonempty string identifier")
    return value


def _clip(value, low, high):
    return low if value < low else high if value > high else value


def _finite_or_none(value):
    return None if value is None or math.isinf(value) else float(value)


def linear_bins(low, high, count, *, name=None):
    """Equal-width bin edges for one descriptor dimension.

    Returns the edge list together with its scope: bin edges are a modelling
    choice about which behavioural differences the search is allowed to see.
    Two behaviours inside one bin are indistinguishable to this archive.
    """
    low, high = number(low), number(high)
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= MAX_BIN_EDGES - 1:
        raise Invalid(f"bin_count_out_of_bounds: expected 1..{MAX_BIN_EDGES - 1}")
    if high <= low:
        raise Invalid("bin_range_not_increasing: high must exceed low")
    width = (high - low) / count
    edges = [low + width * i for i in range(count)] + [high]
    return {
        "name": name,
        "low": low,
        "high": high,
        "count": count,
        "edges": edges,
        "scope": (
            "Equal-width discretisation of one declared descriptor. Resolution is a "
            "choice; behaviours inside one bin are indistinguishable to the archive, "
            "and values outside [low, high] are clamped to the edge bins and counted."
        ),
    }


def trivial_descriptor(candidate):
    """Fallback descriptor: echoes what the caller already declared.

    It measures nothing. It exists so the archive can be exercised before a real
    measured-behaviour descriptor is wired in, and so tests do not have to depend
    on a dynamical-analysis module. Never use it in a run whose diversity claim
    matters.
    """
    if not isinstance(candidate, dict) or "descriptor" not in candidate:
        raise Invalid("missing_declared_descriptor: candidate has no 'descriptor'")
    values = candidate["descriptor"]
    if not isinstance(values, (list, tuple)):
        raise Invalid("invalid_declared_descriptor: expected a sequence of numbers")
    return tuple(number(v) for v in values)


trivial_descriptor.scope = (
    "Declared, not measured. This descriptor repeats the candidate's own claim "
    "about where it belongs, so coverage computed with it reflects bookkeeping "
    "only and establishes no behavioural diversity whatsoever."
)


def pareto_compare(left, right, objectives):
    """Pareto relation between two objective vectors under declared directions."""
    specs = [_objective_spec(o, i) for i, o in enumerate(objectives)]
    a = _objective_values(left, specs)
    b = _objective_values(right, specs)
    better = worse = False
    for spec in specs:
        sign = -1.0 if spec["direction"] == "minimize" else 1.0
        difference = sign * (a[spec["name"]] - b[spec["name"]])
        if difference > 0:
            better = True
        elif difference < 0:
            worse = True
    relation = (
        "a_dominates_b"
        if better and not worse
        else "b_dominates_a"
        if worse and not better
        else "mutually_nondominated"
        if better and worse
        else "identical_objective_vector"
    )
    return {
        "relation": relation,
        "objective_names": [s["name"] for s in specs],
        "rule": (
            "a dominates b when a is no worse on every declared objective and "
            "strictly better on at least one, each objective read in its declared "
            "direction. No objective is traded against another here."
        ),
        "scope": (
            "A dominance relation between two recorded objective vectors. It says "
            "nothing about whether either vector was measured correctly or whether "
            "the objective set is the right one for the question."
        ),
    }


def crowding_distances(front, objectives):
    """NSGA-II crowding distance per front member; boundary members are unbounded."""
    specs = [_objective_spec(o, i) for i, o in enumerate(objectives)]
    entries = list(front)
    if not entries:
        raise Invalid("empty_front_has_no_crowding_distance")
    distance = {e["candidate_id"]: 0.0 for e in entries}
    if len(entries) != len(distance):
        raise Invalid("duplicate_candidate_ids_in_front")
    if len(entries) <= 2:
        return {
            "distances": {k: math.inf for k in distance},
            "finite_distances": {k: None for k in distance},
            "rule": "Fronts of at most two members are all boundary members.",
            "scope": (
                "Crowding distance is a within-front spacing statistic used to keep "
                "the extremes of a trade-off visible. It is not a quality score."
            ),
        }
    for spec in specs:
        name = spec["name"]
        ordered = sorted(entries, key=lambda e: (e["objectives"][name], e["candidate_id"]))
        low = ordered[0]["objectives"][name]
        high = ordered[-1]["objectives"][name]
        distance[ordered[0]["candidate_id"]] = math.inf
        distance[ordered[-1]["candidate_id"]] = math.inf
        if high <= low:
            continue
        for i in range(1, len(ordered) - 1):
            ident = ordered[i]["candidate_id"]
            if math.isinf(distance[ident]):
                continue
            span = ordered[i + 1]["objectives"][name] - ordered[i - 1]["objectives"][name]
            distance[ident] += span / (high - low)
    return {
        "distances": distance,
        "finite_distances": {k: _finite_or_none(v) for k, v in distance.items()},
        "rule": (
            "Per objective, sort the front and give each interior member the "
            "normalised gap between its neighbours; the two extremes are unbounded "
            "so a trade-off's endpoints are never pruned before its middle."
        ),
        "scope": (
            "Crowding distance is a within-front spacing statistic used to keep the "
            "extremes of a trade-off visible. It is not a quality score and carries "
            "no scientific meaning."
        ),
    }


def _objective_spec(spec, index):
    if not isinstance(spec, dict):
        raise Invalid(f"invalid_objective_{index}: expected a mapping")
    unknown = sorted(set(spec) - OBJECTIVE_FIELDS)
    if unknown:
        raise Invalid("unknown_objective_fields: " + ", ".join(unknown))
    name = _identifier(spec.get("name"), f"objective_name_{index}")
    direction = spec.get("direction")
    if direction not in DIRECTIONS:
        raise Invalid("invalid_objective_direction: expected minimize or maximize")
    low, high = number(spec.get("lower")), number(spec.get("upper"))
    if high <= low:
        raise Invalid(f"objective_bounds_not_increasing: {name}")
    weight = number(spec.get("weight", 1.0), 0)
    if weight <= 0:
        raise Invalid(f"objective_weight_not_positive: {name}")
    return {
        "name": name,
        "direction": direction,
        "lower": low,
        "upper": high,
        "weight": weight,
        "units": spec.get("units") or "undeclared",
        "meaning": spec.get("meaning") or "undeclared",
    }


def _objective_values(values, specs):
    if not isinstance(values, dict):
        raise Invalid("invalid_objective_vector: expected a mapping")
    names = {s["name"] for s in specs}
    if set(values) != names:
        raise Invalid(
            "objective_vector_mismatch: expected exactly " + ", ".join(sorted(names))
        )
    return {name: number(values[name]) for name in names}


class Archive:
    """A MAP-Elites archive whose cells keep a bounded Pareto front, not one scalar.

    A single scalar fitness cannot show a trade-off, and saturates the moment
    every candidate passes every check. Each cell therefore keeps a Pareto front
    over the declared objective vector, capped at a front capacity and pruned by
    crowding distance so the endpoints of a trade-off outlive its middle.
    """

    SCOPE = SCOPE

    def __init__(
        self,
        descriptor_names,
        bins,
        objectives,
        *,
        descriptor_fn=None,
        objective_fn=None,
        feasibility_fn=None,
        selection_strategy="curiosity",
        front_capacity=8,
        novelty_neighbours=5,
        reachable_cells=None,
        reachable_cells_basis=None,
        curiosity_initial=1.0,
        curiosity_reward=1.0,
        curiosity_penalty=0.5,
        curiosity_floor=0.0,
        max_retained_rejected=MAX_RETAINED_REJECTED,
        retain_genomes=True,
    ):
        if not isinstance(descriptor_names, (list, tuple)) or not (
            1 <= len(descriptor_names) <= MAX_DESCRIPTOR_DIMENSIONS
        ):
            raise Invalid(
                f"descriptor_dimensions_out_of_bounds: expected 1..{MAX_DESCRIPTOR_DIMENSIONS}"
            )
        names = [_identifier(n, f"descriptor_name_{i}") for i, n in enumerate(descriptor_names)]
        if len(set(names)) != len(names):
            raise Invalid("duplicate_descriptor_names")
        if not isinstance(bins, (list, tuple)) or len(bins) != len(names):
            raise Invalid("bins_must_match_descriptor_dimensions")
        self.descriptor_names = names
        self.bins = [self._edges(spec, i) for i, spec in enumerate(bins)]
        self.shape = [len(edges) - 1 for edges in self.bins]
        total = 1
        for size in self.shape:
            total *= size
        if total > MAX_REACHABLE_CELLS:
            raise Invalid(f"behaviour_space_too_large: {total} > {MAX_REACHABLE_CELLS}")
        self.total_cells = total
        if reachable_cells is None:
            self.reachable_cells = total
            self.reachable_cells_basis = (
                "Every cell of the declared grid is treated as reachable; no domain "
                "grammar was supplied to exclude cells."
            )
        else:
            if (
                not isinstance(reachable_cells, int)
                or isinstance(reachable_cells, bool)
                or not 1 <= reachable_cells <= total
            ):
                raise Invalid(f"reachable_cells_out_of_bounds: expected 1..{total}")
            if not isinstance(reachable_cells_basis, str) or not reachable_cells_basis.strip():
                raise Invalid(
                    "reachable_cells_basis_required: state why fewer cells are reachable"
                )
            self.reachable_cells = reachable_cells
            self.reachable_cells_basis = reachable_cells_basis
        if not isinstance(objectives, (list, tuple)) or not 1 <= len(objectives) <= MAX_OBJECTIVES:
            raise Invalid(f"objective_count_out_of_bounds: expected 1..{MAX_OBJECTIVES}")
        self.objectives = [_objective_spec(o, i) for i, o in enumerate(objectives)]
        if len({o["name"] for o in self.objectives}) != len(self.objectives):
            raise Invalid("duplicate_objective_names")
        if selection_strategy not in SELECTION_STRATEGIES:
            raise Invalid(
                "unknown_selection_strategy: expected " + ", ".join(SELECTION_STRATEGIES)
            )
        if (
            not isinstance(front_capacity, int)
            or isinstance(front_capacity, bool)
            or not 2 <= front_capacity <= MAX_FRONT_CAPACITY
        ):
            raise Invalid(f"front_capacity_out_of_bounds: expected 2..{MAX_FRONT_CAPACITY}")
        if (
            not isinstance(novelty_neighbours, int)
            or isinstance(novelty_neighbours, bool)
            or not 1 <= novelty_neighbours <= 64
        ):
            raise Invalid("novelty_neighbours_out_of_bounds: expected 1..64")
        if (
            not isinstance(max_retained_rejected, int)
            or isinstance(max_retained_rejected, bool)
            or not 1 <= max_retained_rejected <= MAX_RETAINED_REJECTED
        ):
            raise Invalid(
                f"max_retained_rejected_out_of_bounds: expected 1..{MAX_RETAINED_REJECTED}"
            )
        self.selection_strategy = selection_strategy
        self.front_capacity = front_capacity
        self.novelty_neighbours = novelty_neighbours
        self.curiosity_initial = number(curiosity_initial)
        self.curiosity_reward = number(curiosity_reward, 0)
        self.curiosity_penalty = number(curiosity_penalty, 0)
        self.curiosity_floor = number(curiosity_floor)
        self.max_retained_rejected = max_retained_rejected
        self.retain_genomes = bool(retain_genomes)
        self.descriptor_fn = descriptor_fn or trivial_descriptor
        self.objective_fn = objective_fn
        self.feasibility_fn = feasibility_fn
        self.cells = {}
        self.cell_of_candidate = {}
        self.rejected = []
        self.rejected_truncated = 0
        self.history = []
        self.selection_log = []
        self.generation = 0
        self.add_sequence = 0
        self.outcome_counts = {name: 0 for name in OUTCOMES}
        self.clamp_events = {
            name: {"below_low_edge": 0, "above_high_edge": 0}
            for name in self.descriptor_names
        }
        self.clamped_candidate_ids = []
        self.objective_clip_events = {o["name"]: 0 for o in self.objectives}

    @staticmethod
    def _edges(spec, index):
        if isinstance(spec, dict):
            spec = spec.get("edges")
        if not isinstance(spec, (list, tuple)) or not 2 <= len(spec) <= MAX_BIN_EDGES:
            raise Invalid(
                f"bin_edges_out_of_bounds: dimension {index} needs 2..{MAX_BIN_EDGES} edges"
            )
        edges = [number(v) for v in spec]
        if any(b <= a for a, b in zip(edges, edges[1:])):
            raise Invalid(f"bin_edges_not_strictly_increasing: dimension {index}")
        return edges

    # ---------------------------------------------------------------- geometry

    def locate(self, descriptor, *, candidate_id=None, count_clamps=True):
        """Map a descriptor to a cell index, clamping out-of-range values loudly.

        Out-of-range policy: a value below the first edge lands in bin 0 and a
        value at or above the last edge lands in the final bin. Clamping is
        counted per dimension and per direction, because a silent clamp hides a
        mis-specified descriptor range behind apparently healthy coverage.
        """
        if not isinstance(descriptor, (list, tuple)) or len(descriptor) != len(self.bins):
            raise Invalid(
                f"descriptor_dimension_mismatch: expected {len(self.bins)} values"
            )
        values = [number(v) for v in descriptor]
        index, clamped = [], []
        for position, (value, edges) in enumerate(zip(values, self.bins)):
            name = self.descriptor_names[position]
            if value < edges[0]:
                index.append(0)
                clamped.append(
                    {"descriptor": name, "direction": "below_low_edge", "value": value}
                )
                if count_clamps:
                    self.clamp_events[name]["below_low_edge"] += 1
            elif value >= edges[-1]:
                index.append(len(edges) - 2)
                if value > edges[-1]:
                    clamped.append(
                        {
                            "descriptor": name,
                            "direction": "above_high_edge",
                            "value": value,
                        }
                    )
                    if count_clamps:
                        self.clamp_events[name]["above_high_edge"] += 1
            else:
                slot = 0
                for i in range(len(edges) - 1):
                    if edges[i] <= value < edges[i + 1]:
                        slot = i
                        break
                index.append(slot)
        if clamped and count_clamps and candidate_id is not None:
            if len(self.clamped_candidate_ids) < self.max_retained_rejected:
                self.clamped_candidate_ids.append(candidate_id)
        return {
            "index": index,
            "cell_id": _cell_id(index),
            "descriptor": values,
            "clamped": clamped,
            "policy": (
                "Values below the first edge fall in bin 0; values above the last "
                "edge fall in the final bin. Every clamp is counted per dimension "
                "and per direction so a mis-specified descriptor range is visible."
            ),
            "scope": (
                "A grid lookup. It does not check that the descriptor was measured "
                "from real behaviour or that its range was chosen sensibly."
            ),
        }

    def _cell_centre(self, index):
        return [
            (edges[i] + edges[i + 1]) / 2.0 for i, edges in zip(index, self.bins)
        ]

    def _normalised_centre(self, index):
        centre = self._cell_centre(index)
        return [
            (value - edges[0]) / (edges[-1] - edges[0])
            for value, edges in zip(centre, self.bins)
        ]

    def scalarize(self, objectives):
        """The named scalarization used only for QD score and tie-breaking.

        It is a reporting convenience, never the selection rule: selection inside
        a cell is Pareto, so no trade-off is hidden inside this weighted sum.
        Values outside a declared objective range are clipped and counted.
        """
        values = _objective_values(objectives, self.objectives)
        total = weight_sum = 0.0
        clipped = []
        for spec in self.objectives:
            raw = values[spec["name"]]
            bounded = _clip(raw, spec["lower"], spec["upper"])
            if bounded != raw:
                clipped.append(spec["name"])
            unit = (bounded - spec["lower"]) / (spec["upper"] - spec["lower"])
            if spec["direction"] == "minimize":
                unit = 1.0 - unit
            total += spec["weight"] * unit
            weight_sum += spec["weight"]
        return {
            "value": total / weight_sum,
            "name": SCALARIZATION,
            "clipped_objectives": clipped,
            "definition": (
                "Each objective is mapped to [0, 1] against its declared bounds, "
                "oriented so higher is better, then combined by declared weights. "
                "Used for QD score reporting and deterministic tie-breaking only."
            ),
            "scope": (
                "A reporting scalar. It compresses a declared trade-off into one "
                "number and must not be read as candidate quality; the per-cell "
                "Pareto front is the retained result."
            ),
        }

    # ------------------------------------------------------------------- adding

    def add(self, candidate, *, parent_ids=None, operator=None):
        """Evaluate one candidate against its cell and record an explicit outcome.

        Outcomes: new_cell, dominates_existing, nondominated_added,
        dominated_rejected, infeasible. Rejected, infeasible, displaced and
        crowding-pruned candidates are all retained with the reason they lost, so
        a losing branch stays inspectable instead of vanishing from the record.
        """
        if not isinstance(candidate, dict):
            raise Invalid("invalid_candidate: expected a mapping")
        candidate_id = _identifier(candidate.get("id"), "candidate_id")
        if candidate_id in self.cell_of_candidate or any(
            r["candidate_id"] == candidate_id for r in self.rejected
        ):
            raise Invalid("duplicate_candidate_id: " + candidate_id)
        parents = list(parent_ids if parent_ids is not None else candidate.get("parent_ids") or [])
        parents = [_identifier(p, "parent_id") for p in parents[:8]]
        operator = None if operator is None else _identifier(operator, "operator")
        self.add_sequence += 1
        genome = candidate.get("genome")
        if genome is not None and self.retain_genomes:
            try:
                encoded = canonical(genome)
            except (TypeError, ValueError) as exc:
                raise Invalid("genome_is_not_plain_json: " + str(exc)[:200]) from None
            if len(encoded) > MAX_GENOME_CHARS:
                raise Invalid(f"genome_exceeds_{MAX_GENOME_CHARS}_characters")
        elif not self.retain_genomes:
            genome = None
        violations = self._violations(candidate)
        if violations:
            return self._reject(
                candidate_id,
                outcome="infeasible",
                reason="infeasible: " + "; ".join(violations[:8]),
                cell_id=None,
                descriptor=None,
                objectives=None,
                parents=parents,
                operator=operator,
                genome=genome,
                extra={"violations": violations[:8]},
            )
        descriptor = self.descriptor_fn(candidate)
        placement = self.locate(descriptor, candidate_id=candidate_id)
        values = _objective_values(
            self.objective_fn(candidate) if self.objective_fn else candidate.get("objectives"),
            self.objectives,
        )
        scalar = self.scalarize(values)
        for name in scalar["clipped_objectives"]:
            self.objective_clip_events[name] += 1
        entry = {
            "candidate_id": candidate_id,
            "descriptor": placement["descriptor"],
            "cell_index": placement["index"],
            "objectives": values,
            "scalarization": scalar["value"],
            "generation": self.generation,
            "parent_ids": parents,
            "operator": operator,
            "added_at": self.add_sequence,
            "genome": genome,
        }
        cell = self.cells.get(placement["cell_id"])
        if cell is None:
            cell = {
                "id": placement["cell_id"],
                "index": placement["index"],
                "centre": self._cell_centre(placement["index"]),
                "front": [entry],
                "curiosity": self.curiosity_initial,
                "visit_count": 0,
                "accepted_count": 1,
                "rejected_count": 0,
                "offspring_accepted": 0,
                "offspring_rejected": 0,
                "first_generation": self.generation,
                "last_improved_generation": self.generation,
            }
            self.cells[placement["cell_id"]] = cell
            return self._accept(
                entry, cell, "new_cell", placement, displaced=[], pruned=[], parents=parents
            )
        dominated_by = []
        displaced = []
        for member in cell["front"]:
            relation = self._relation(member["objectives"], entry["objectives"])
            if relation == "a_dominates_b":
                dominated_by.append(member["candidate_id"])
            elif relation == "b_dominates_a":
                displaced.append(member)
        if dominated_by:
            cell["rejected_count"] += 1
            return self._reject(
                candidate_id,
                outcome="dominated_rejected",
                reason=(
                    "dominated on every declared objective by retained elite(s): "
                    + ", ".join(sorted(dominated_by)[:8])
                ),
                cell_id=cell["id"],
                descriptor=placement["descriptor"],
                objectives=values,
                parents=parents,
                operator=operator,
                genome=genome,
                extra={"dominated_by": sorted(dominated_by)},
            )
        for member in displaced:
            cell["front"].remove(member)
            self._retain(
                member,
                outcome="displaced_by_domination",
                reason="dominated by later candidate " + candidate_id,
                cell_id=cell["id"],
            )
        cell["front"].append(entry)
        cell["accepted_count"] += 1
        cell["last_improved_generation"] = self.generation
        pruned = self._prune(cell)
        outcome = "dominates_existing" if displaced else "nondominated_added"
        return self._accept(
            entry,
            cell,
            outcome,
            placement,
            displaced=[m["candidate_id"] for m in displaced],
            pruned=pruned,
            parents=parents,
        )

    def _violations(self, candidate):
        if self.feasibility_fn is not None:
            violations = self.feasibility_fn(candidate)
        else:
            violations = candidate.get("infeasible_reasons") or []
        if not isinstance(violations, (list, tuple)):
            raise Invalid("invalid_feasibility_result: expected a list of reasons")
        reasons = []
        for violation in violations[:16]:
            if not isinstance(violation, str) or not violation.strip():
                raise Invalid("invalid_violation: expected a nonempty reason string")
            reasons.append(violation[:400])
        return reasons

    def _relation(self, left, right):
        better = worse = False
        for spec in self.objectives:
            sign = -1.0 if spec["direction"] == "minimize" else 1.0
            difference = sign * (left[spec["name"]] - right[spec["name"]])
            if difference > 0:
                better = True
            elif difference < 0:
                worse = True
        if better and not worse:
            return "a_dominates_b"
        if worse and not better:
            return "b_dominates_a"
        return "mutually_nondominated"

    def _prune(self, cell):
        pruned = []
        while len(cell["front"]) > self.front_capacity:
            distances = crowding_distances(cell["front"], self.objectives)["distances"]
            victim = min(
                cell["front"],
                key=lambda e: (
                    distances[e["candidate_id"]],
                    e["scalarization"],
                    e["candidate_id"],
                ),
            )
            cell["front"].remove(victim)
            self._retain(
                victim,
                outcome="crowding_pruned",
                reason=(
                    "front exceeded capacity "
                    f"{self.front_capacity}; lowest crowding distance in its cell"
                ),
                cell_id=cell["id"],
            )
            pruned.append(victim["candidate_id"])
        cell["front"].sort(key=lambda e: (-e["scalarization"], e["candidate_id"]))
        return pruned

    def _accept(self, entry, cell, outcome, placement, *, displaced, pruned, parents):
        self.outcome_counts[outcome] += 1
        self.cell_of_candidate[entry["candidate_id"]] = cell["id"]
        self._credit(parents, accepted=True)
        return {
            "outcome": outcome,
            "accepted": True,
            "retained": True,
            "candidate_id": entry["candidate_id"],
            "cell_id": cell["id"],
            "cell_index": cell["index"],
            "descriptor": entry["descriptor"],
            "descriptor_names": list(self.descriptor_names),
            "clamped": placement["clamped"],
            "objectives": dict(entry["objectives"]),
            "scalarization": entry["scalarization"],
            "front_size": len(cell["front"]),
            "displaced_candidate_ids": displaced,
            "crowding_pruned_candidate_ids": pruned,
            "generation": self.generation,
            "reason": None,
            "curiosity_after": cell["curiosity"],
            "scope": SCOPE,
        }

    def _reject(
        self,
        candidate_id,
        *,
        outcome,
        reason,
        cell_id,
        descriptor,
        objectives,
        parents,
        operator,
        genome,
        extra,
    ):
        self.outcome_counts[outcome] += 1
        self._retain(
            {
                "candidate_id": candidate_id,
                "descriptor": descriptor,
                "objectives": objectives,
                "generation": self.generation,
                "parent_ids": parents,
                "operator": operator,
                "genome": genome,
                "added_at": self.add_sequence,
            },
            outcome=outcome,
            reason=reason,
            cell_id=cell_id,
            extra=extra,
        )
        self._credit(parents, accepted=False)
        return {
            "outcome": outcome,
            "accepted": False,
            "retained": True,
            "candidate_id": candidate_id,
            "cell_id": cell_id,
            "cell_index": None if cell_id is None else self.cells[cell_id]["index"],
            "descriptor": descriptor,
            "descriptor_names": list(self.descriptor_names),
            "clamped": [],
            "objectives": objectives,
            "scalarization": None,
            "front_size": 0 if cell_id is None else len(self.cells[cell_id]["front"]),
            "displaced_candidate_ids": [],
            "crowding_pruned_candidate_ids": [],
            "generation": self.generation,
            "reason": reason,
            "curiosity_after": None if cell_id is None else self.cells[cell_id]["curiosity"],
            "scope": SCOPE,
        }

    def _retain(self, entry, *, outcome, reason, cell_id, extra=None):
        if outcome not in RETENTION_REASONS:
            raise Invalid("unknown_retention_reason: " + str(outcome))
        record = {
            "candidate_id": entry["candidate_id"],
            "outcome": outcome,
            "reason": reason,
            "cell_id": cell_id,
            "descriptor": entry.get("descriptor"),
            "objectives": entry.get("objectives"),
            "generation": entry.get("generation", self.generation),
            "parent_ids": entry.get("parent_ids", []),
            "operator": entry.get("operator"),
            "genome": entry.get("genome"),
            "added_at": entry.get("added_at", self.add_sequence),
        }
        if extra:
            record.update(extra)
        self.rejected.append(record)
        while len(self.rejected) > self.max_retained_rejected:
            self.rejected.pop(0)
            self.rejected_truncated += 1

    def _credit(self, parents, *, accepted):
        """Cully & Demiris curiosity: reward a cell whose offspring survive.

        A recombined candidate has two parents, so both contributing cells are
        credited or debited. Crediting only one would make half the work that
        produced the child invisible to the next scheduling decision.
        """
        credited = set()
        for parent_id in parents:
            cell_id = self.cell_of_candidate.get(parent_id)
            if cell_id is None or cell_id in credited:
                continue
            credited.add(cell_id)
            cell = self.cells[cell_id]
            delta = self.curiosity_reward if accepted else -self.curiosity_penalty
            cell["curiosity"] = max(self.curiosity_floor, cell["curiosity"] + delta)
            cell["offspring_accepted" if accepted else "offspring_rejected"] += 1

    # ---------------------------------------------------------------- selection

    def select_parent(self, seed, *, strategy=None):
        """Pick a parent cell and elite under an explicit, recorded strategy."""
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise Invalid("selection_seed_required: pass an explicit integer seed")
        strategy = strategy or self.selection_strategy
        if strategy not in SELECTION_STRATEGIES:
            raise Invalid("unknown_selection_strategy: " + str(strategy))
        if not self.cells:
            raise Invalid("empty_archive_has_no_parent")
        cells = [self.cells[key] for key in sorted(self.cells)]
        novelty = self.novelty()["per_cell"] if strategy == "novelty" else {}
        if strategy == "uniform":
            scores = [1.0 for _ in cells]
            arithmetic = "score(cell) = 1; every occupied cell is equally likely."
        elif strategy == "curiosity":
            scores = [max(0.0, c["curiosity"]) + EPSILON for c in cells]
            arithmetic = (
                "score(cell) = max(0, curiosity) + 1e-9, sampled proportionally. "
                f"curiosity starts at {self.curiosity_initial}, gains "
                f"{self.curiosity_reward} when an offspring of this cell is accepted "
                f"and loses {self.curiosity_penalty} when one is rejected, floored at "
                f"{self.curiosity_floor}."
            )
        elif strategy == "least_visited":
            fewest = min(c["visit_count"] for c in cells)
            scores = [1.0 if c["visit_count"] == fewest else 0.0 for c in cells]
            arithmetic = (
                "score(cell) = 1 for cells tied at the fewest parent selections, else "
                "0. This reproduces the earlier least-visited scheduling rule."
            )
        else:
            scores = [novelty.get(c["id"], 0.0) + EPSILON for c in cells]
            arithmetic = (
                f"score(cell) = mean Euclidean distance to its {self.novelty_neighbours} "
                "nearest occupied cells in unit-normalised descriptor space, + 1e-9, "
                "sampled proportionally."
            )
        total = sum(scores)
        rng = random.Random(seed)
        target = rng.random() * total
        running = 0.0
        chosen = cells[-1]
        for cell, score in zip(cells, scores):
            running += score
            if target < running:
                chosen = cell
                break
        front_index = rng.randrange(len(chosen["front"]))
        elite = chosen["front"][front_index]
        chosen["visit_count"] += 1
        result = {
            "strategy": strategy,
            "seed": seed,
            "selected_cell_id": chosen["id"],
            "selected_cell_index": list(chosen["index"]),
            "selected_candidate_id": elite["candidate_id"],
            "selected_front_index": front_index,
            "selected_genome": elite.get("genome"),
            "selected_objectives": dict(elite["objectives"]),
            "visit_count_after": chosen["visit_count"],
            "curiosity": chosen["curiosity"],
            "arithmetic": arithmetic,
            "weights": [
                {
                    "cell_id": cell["id"],
                    "score": score,
                    "probability": score / total if total > 0 else 0.0,
                    "curiosity": cell["curiosity"],
                    "visit_count": cell["visit_count"],
                }
                for cell, score in zip(cells, scores)
            ],
            "alternatives": list(SELECTION_STRATEGIES),
            "scope": (
                "A search scheduling choice recorded with its arithmetic. Selecting a "
                "parent expresses where the search will spend budget next; it asserts "
                "nothing about the parent's scientific standing. " + SCOPE
            ),
        }
        if len(self.selection_log) < MAX_HISTORY:
            self.selection_log.append(
                {
                    "generation": self.generation,
                    "strategy": strategy,
                    "seed": seed,
                    "cell_id": chosen["id"],
                    "candidate_id": elite["candidate_id"],
                }
            )
        return result

    # ------------------------------------------------------------------ metrics

    def coverage(self):
        occupied = len(self.cells)
        return {
            "occupied_cells": occupied,
            "reachable_cells": self.reachable_cells,
            "total_grid_cells": self.total_cells,
            "coverage": occupied / self.reachable_cells,
            "shape": list(self.shape),
            "reachable_cells_basis": self.reachable_cells_basis,
            "scope": (
                "Fraction of the declared behaviour grid that the search has reached. "
                "Coverage measures search spread, not scientific breadth: filling "
                "cells with badly measured or scientifically worthless candidates "
                "raises coverage exactly as much as filling them with good ones."
            ),
        }

    def qd_score(self):
        per_cell = {}
        for key in sorted(self.cells):
            front = self.cells[key]["front"]
            per_cell[key] = max(e["scalarization"] for e in front) if front else 0.0
        return {
            "qd_score": sum(per_cell.values()),
            "per_cell": per_cell,
            "occupied_cells": len(per_cell),
            "scalarization": SCALARIZATION,
            "definition": (
                "Sum over occupied cells of the best "
                f"{SCALARIZATION} value on that cell's Pareto front. Objectives are "
                "normalised against their declared bounds, so the score is bounded by "
                "the number of occupied cells."
            ),
            "objectives": [
                {k: o[k] for k in ("name", "direction", "lower", "upper", "weight", "units")}
                for o in self.objectives
            ],
            "scope": (
                "A search-progress statistic under one named scalarization. A high QD "
                "score means the search found many cells with good declared-objective "
                "values; it is never a validated result, a calibrated estimate, or "
                "permission to promote a candidate."
            ),
        }

    def archive_entropy(self):
        counts = {key: len(cell["front"]) for key, cell in self.cells.items()}
        total = sum(counts.values())
        if total == 0:
            entropy = 0.0
        else:
            entropy = -sum(
                (n / total) * math.log2(n / total) for n in counts.values() if n > 0
            )
        maximum = math.log2(self.reachable_cells) if self.reachable_cells > 1 else 0.0
        return {
            "entropy_bits": entropy,
            "maximum_entropy_bits": maximum,
            "normalized_entropy": entropy / maximum if maximum > 0 else 0.0,
            "occupancy": dict(sorted(counts.items())),
            "definition": (
                "Shannon entropy in bits of retained elites over occupied cells, "
                "against the maximum achievable if every reachable cell held an equal "
                "share."
            ),
            "scope": (
                "How evenly the search spread its retained candidates. High entropy "
                "means an even spread, not a diverse set of scientifically distinct "
                "explanations."
            ),
        }

    def novelty(self, *, neighbours=None):
        k = neighbours if neighbours is not None else self.novelty_neighbours
        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 64:
            raise Invalid("novelty_neighbours_out_of_bounds: expected 1..64")
        keys = sorted(self.cells)
        points = {key: self._normalised_centre(self.cells[key]["index"]) for key in keys}
        per_cell = {}
        for key in keys:
            others = [
                math.dist(points[key], points[other]) for other in keys if other != key
            ]
            others.sort()
            nearest = others[:k]
            per_cell[key] = sum(nearest) / len(nearest) if nearest else 0.0
        values = sorted(per_cell.values())
        return {
            "per_cell": per_cell,
            "neighbours": k,
            "minimum": values[0] if values else 0.0,
            "median": values[len(values) // 2] if values else 0.0,
            "mean": sum(values) / len(values) if values else 0.0,
            "maximum": values[-1] if values else 0.0,
            "definition": (
                "Mean Euclidean distance from each occupied cell centre to its k "
                "nearest occupied cell centres, in descriptor space normalised to the "
                "unit cube by the declared bin ranges."
            ),
            "scope": (
                "Distance in a declared descriptor space. Novel behaviour in this "
                "sense is not new science; it is a candidate that measured differently "
                "on the descriptors someone chose."
            ),
        }

    def advance_generation(self):
        """Close the current generation, snapshot its metrics, and open the next."""
        coverage = self.coverage()
        qd = self.qd_score()
        entropy = self.archive_entropy()
        previous = self.history[-1] if self.history else None
        snapshot = {
            "generation": self.generation,
            "occupied_cells": coverage["occupied_cells"],
            "coverage": coverage["coverage"],
            "qd_score": qd["qd_score"],
            "best_scalarization": max(qd["per_cell"].values()) if qd["per_cell"] else 0.0,
            "entropy_bits": entropy["entropy_bits"],
            "outcome_counts": dict(self.outcome_counts),
            "retained_rejected": len(self.rejected),
            "coverage_delta": coverage["coverage"] - (previous["coverage"] if previous else 0.0),
            "qd_score_delta": qd["qd_score"] - (previous["qd_score"] if previous else 0.0),
        }
        snapshot["improved"] = (
            snapshot["coverage_delta"] > 0 or snapshot["qd_score_delta"] > EPSILON
        )
        snapshot["scope"] = SCOPE
        self.history.append(snapshot)
        while len(self.history) > MAX_HISTORY:
            self.history.pop(0)
        self.generation += 1
        return snapshot

    def improvement_over_generations(self):
        """The per-generation series, so stagnation is visible instead of implied."""
        series = [dict(record) for record in self.history]
        improved = [r["generation"] for r in series if r["improved"]]
        last = improved[-1] if improved else None
        return {
            "generations": series,
            "generations_recorded": len(series),
            "last_improved_generation": last,
            "generations_since_improvement": (
                None if not series else (series[-1]["generation"] - last) if last is not None
                else series[-1]["generation"] + 1
            ),
            "stagnant": bool(series) and not series[-1]["improved"],
            "definition": (
                "A generation counts as an improvement when coverage rose or QD score "
                "rose by more than 1e-9 against the previous recorded generation."
            ),
            "scope": (
                "Stagnation here means the search stopped changing this archive under "
                "these descriptors and objectives. It is not evidence that the "
                "underlying scientific question is exhausted or answered."
            ),
        }

    def elites(self):
        return {
            "cells": [
                {
                    "cell_id": key,
                    "index": list(self.cells[key]["index"]),
                    "centre": list(self.cells[key]["centre"]),
                    "curiosity": self.cells[key]["curiosity"],
                    "visit_count": self.cells[key]["visit_count"],
                    "offspring_accepted": self.cells[key]["offspring_accepted"],
                    "offspring_rejected": self.cells[key]["offspring_rejected"],
                    "front": [dict(e) for e in self.cells[key]["front"]],
                }
                for key in sorted(self.cells)
            ],
            "descriptor_names": list(self.descriptor_names),
            "front_capacity": self.front_capacity,
            "scope": SCOPE,
        }

    def retained_rejections(self, *, outcome=None):
        """Every losing and infeasible branch, with the reason it lost."""
        records = [
            dict(r) for r in self.rejected if outcome is None or r["outcome"] == outcome
        ]
        return {
            "records": records,
            "count": len(records),
            "truncated_count": self.rejected_truncated,
            "retention_bound": self.max_retained_rejected,
            "outcomes": list(RETENTION_REASONS),
            "scope": (
                "Losing branches are retained with their reason so a search can be "
                "audited rather than admired. When the retention bound is reached the "
                "oldest records are dropped and counted in truncated_count; nothing is "
                "dropped silently."
            ),
        }

    def clamp_report(self):
        total = sum(
            sum(counts.values()) for counts in self.clamp_events.values()
        )
        return {
            "total_clamps": total,
            "per_dimension": {k: dict(v) for k, v in self.clamp_events.items()},
            "clamped_candidate_ids": list(self.clamped_candidate_ids),
            "objective_clip_events": dict(self.objective_clip_events),
            "policy": (
                "Descriptor values outside the declared bin range are clamped to the "
                "edge bins and counted; objective values outside a declared range are "
                "clipped for scalarization and counted."
            ),
            "scope": (
                "A nonzero clamp count means the declared descriptor range does not "
                "cover the behaviour actually produced. Coverage computed over a range "
                "that is being clamped is not trustworthy as a diversity claim."
            ),
        }

    def metrics(self):
        coverage = self.coverage()
        qd = self.qd_score()
        return {
            "version": VERSION,
            "generation": self.generation,
            "selection_strategy": self.selection_strategy,
            "coverage": coverage,
            "qd_score": qd,
            "archive_entropy": self.archive_entropy(),
            "novelty": self.novelty(),
            "improvement_over_generations": self.improvement_over_generations(),
            "clamps": self.clamp_report(),
            "outcome_counts": dict(self.outcome_counts),
            "candidates_added": self.add_sequence,
            "retained_rejected": len(self.rejected),
            "rejected_truncated": self.rejected_truncated,
            "objectives_with_undeclared_units": [
                o["name"] for o in self.objectives if o["units"] == "undeclared"
            ],
            "descriptor_provenance": getattr(
                self.descriptor_fn, "scope", "Injected descriptor function; provenance undeclared."
            ),
            "promotion_allowed": False,
            "scope": SCOPE,
        }

    # ------------------------------------------------------------ serialization

    def to_dict(self):
        """Plain-JSON state for the content-addressed artifact store."""
        return {
            "version": VERSION,
            "descriptor_names": list(self.descriptor_names),
            "bins": [list(edges) for edges in self.bins],
            "objectives": [dict(o) for o in self.objectives],
            "selection_strategy": self.selection_strategy,
            "front_capacity": self.front_capacity,
            "novelty_neighbours": self.novelty_neighbours,
            "reachable_cells": self.reachable_cells,
            "reachable_cells_basis": self.reachable_cells_basis,
            "curiosity": {
                "initial": self.curiosity_initial,
                "reward": self.curiosity_reward,
                "penalty": self.curiosity_penalty,
                "floor": self.curiosity_floor,
            },
            "max_retained_rejected": self.max_retained_rejected,
            "retain_genomes": self.retain_genomes,
            "cells": {
                key: {
                    "id": cell["id"],
                    "index": list(cell["index"]),
                    "centre": list(cell["centre"]),
                    "front": [dict(e) for e in cell["front"]],
                    "curiosity": cell["curiosity"],
                    "visit_count": cell["visit_count"],
                    "accepted_count": cell["accepted_count"],
                    "rejected_count": cell["rejected_count"],
                    "offspring_accepted": cell["offspring_accepted"],
                    "offspring_rejected": cell["offspring_rejected"],
                    "first_generation": cell["first_generation"],
                    "last_improved_generation": cell["last_improved_generation"],
                }
                for key, cell in sorted(self.cells.items())
            },
            "cell_of_candidate": dict(sorted(self.cell_of_candidate.items())),
            "rejected": [dict(r) for r in self.rejected],
            "rejected_truncated": self.rejected_truncated,
            "history": [dict(h) for h in self.history],
            "selection_log": [dict(s) for s in self.selection_log],
            "generation": self.generation,
            "add_sequence": self.add_sequence,
            "outcome_counts": dict(self.outcome_counts),
            "clamp_events": {k: dict(v) for k, v in self.clamp_events.items()},
            "clamped_candidate_ids": list(self.clamped_candidate_ids),
            "objective_clip_events": dict(self.objective_clip_events),
            "scope": SCOPE,
        }

    @classmethod
    def from_dict(cls, data, *, descriptor_fn=None, objective_fn=None, feasibility_fn=None):
        """Rebuild an archive from stored state; injected functions are re-supplied."""
        if not isinstance(data, dict):
            raise Invalid("invalid_archive_state: expected a mapping")
        if data.get("version") != VERSION:
            raise Invalid("archive_version_mismatch: expected " + VERSION)
        curiosity = data.get("curiosity") or {}
        grid = 1
        for edges in data.get("bins") or []:
            grid *= max(0, len(edges) - 1)
        stored = data.get("reachable_cells")
        override = stored if isinstance(stored, int) and stored != grid else None
        archive = cls(
            data.get("descriptor_names"),
            data.get("bins"),
            data.get("objectives"),
            descriptor_fn=descriptor_fn,
            objective_fn=objective_fn,
            feasibility_fn=feasibility_fn,
            selection_strategy=data.get("selection_strategy", "curiosity"),
            front_capacity=data.get("front_capacity", 8),
            novelty_neighbours=data.get("novelty_neighbours", 5),
            reachable_cells=override,
            reachable_cells_basis=data.get("reachable_cells_basis"),
            curiosity_initial=curiosity.get("initial", 1.0),
            curiosity_reward=curiosity.get("reward", 1.0),
            curiosity_penalty=curiosity.get("penalty", 0.5),
            curiosity_floor=curiosity.get("floor", 0.0),
            max_retained_rejected=data.get("max_retained_rejected", MAX_RETAINED_REJECTED),
            retain_genomes=data.get("retain_genomes", True),
        )
        cells = data.get("cells") or {}
        if not isinstance(cells, dict):
            raise Invalid("invalid_archive_cells: expected a mapping")
        for key, cell in cells.items():
            archive.cells[key] = {
                "id": cell["id"],
                "index": list(cell["index"]),
                "centre": list(cell["centre"]),
                "front": [dict(e) for e in cell["front"]],
                "curiosity": number(cell["curiosity"]),
                "visit_count": int(cell["visit_count"]),
                "accepted_count": int(cell["accepted_count"]),
                "rejected_count": int(cell["rejected_count"]),
                "offspring_accepted": int(cell.get("offspring_accepted", 0)),
                "offspring_rejected": int(cell.get("offspring_rejected", 0)),
                "first_generation": int(cell["first_generation"]),
                "last_improved_generation": int(cell["last_improved_generation"]),
            }
        archive.cell_of_candidate = dict(data.get("cell_of_candidate") or {})
        archive.rejected = [dict(r) for r in data.get("rejected") or []]
        archive.rejected_truncated = int(data.get("rejected_truncated", 0))
        archive.history = [dict(h) for h in data.get("history") or []]
        archive.selection_log = [dict(s) for s in data.get("selection_log") or []]
        archive.generation = int(data.get("generation", 0))
        archive.add_sequence = int(data.get("add_sequence", 0))
        archive.outcome_counts.update(data.get("outcome_counts") or {})
        for name, counts in (data.get("clamp_events") or {}).items():
            if name in archive.clamp_events:
                archive.clamp_events[name].update(counts)
        archive.clamped_candidate_ids = list(data.get("clamped_candidate_ids") or [])
        archive.objective_clip_events.update(data.get("objective_clip_events") or {})
        return archive


def _cell_id(index):
    return "cell_" + "_".join(str(i) for i in index)
