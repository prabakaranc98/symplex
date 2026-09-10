"""Evolutionary provenance: which operator produced which candidate, and what happened.

A search result is only inspectable if the path to it survives. This is a
directed acyclic graph over candidates in which every edge carries the operator
that was applied and the outcome the archive returned, so a losing branch is as
retrievable as a winning one. Recombination contributes two parent edges.

`innovation_path` is the story of how a candidate was reached; `stagnation_report`
is the story of where a search stopped being productive. Both describe the search
process, never the standing of a scientific claim. Nothing here calls a model,
the network, or a subprocess.
"""

from symplex.agents.qd_archive import ACCEPTING_OUTCOMES, OUTCOMES, RETENTION_REASONS
from symplex.core.contracts import Invalid

VERSION = "evolutionary-lineage-dag-v1"

ORIGINS = ("seed", "mutation", "recombination", "external")
EDGE_OUTCOMES = tuple(sorted(set(OUTCOMES) | set(RETENTION_REASONS) | {"unevaluated"}))

MAX_NODES = 50_000
MAX_EDGES = 200_000
MAX_PARENTS = 8
MAX_TEXT = 1600

SCOPE = (
    "A provenance record of one search. Edges state which operator was applied and "
    "what the archive did with the result; depth, productivity and stagnation "
    "describe the search process only. A long innovation path is not a chain of "
    "evidence, an exhausted operator is not a refuted mechanism, and a productive "
    "branch is not a validated one."
)


def _ident(value, field):
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise Invalid(f"invalid_{field}: expected a nonempty identifier")
    return value


def _optional_text(value, field):
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_TEXT:
        raise Invalid(f"invalid_{field}: expected nonempty text under {MAX_TEXT} chars")
    return value


def _generation(value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise Invalid("invalid_generation: expected a non-negative integer")
    return value


class LineageGraph:
    """A bounded DAG over candidates, with operator-labelled, outcome-labelled edges."""

    SCOPE = SCOPE

    def __init__(self, *, max_nodes=MAX_NODES, max_edges=MAX_EDGES):
        for name, value, limit in (
            ("max_nodes", max_nodes, MAX_NODES),
            ("max_edges", max_edges, MAX_EDGES),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= limit:
                raise Invalid(f"{name}_out_of_bounds: expected 1..{limit}")
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.nodes = {}
        self.edges = []
        self.parents_of = {}
        self.children_of = {}
        self.sequence = 0

    # ------------------------------------------------------------------ building

    def add_seed(self, candidate_id, *, generation=0, note=None, cell_id=None):
        """Record a root candidate: a competent baseline or a distinct template."""
        return self._add_node(
            candidate_id,
            generation=generation,
            origin="seed",
            outcome=None,
            cell_id=cell_id,
            note=note,
        )

    def add_child(
        self,
        candidate_id,
        parent_ids,
        operator,
        outcome,
        *,
        generation=0,
        cell_id=None,
        note=None,
    ):
        """Record a candidate produced by one operator from one or two parents."""
        parents = [_ident(p, "parent_id") for p in (parent_ids or [])]
        if not parents or len(set(parents)) != len(parents):
            raise Invalid("child_requires_distinct_named_parents")
        if len(parents) > MAX_PARENTS:
            raise Invalid(f"too_many_parents: at most {MAX_PARENTS}")
        origin = "recombination" if len(parents) > 1 else "mutation"
        node = self._add_node(
            candidate_id,
            generation=generation,
            origin=origin,
            outcome=outcome,
            cell_id=cell_id,
            note=note,
        )
        edges = [
            self.add_edge(parent, candidate_id, operator, outcome)["edge"]
            for parent in parents
        ]
        return {
            "node": node["node"],
            "edges": edges,
            "parent_count": len(edges),
            "origin": origin,
            "scope": SCOPE,
        }

    def _add_node(self, candidate_id, *, generation, origin, outcome, cell_id, note):
        candidate_id = _ident(candidate_id, "candidate_id")
        if candidate_id in self.nodes:
            raise Invalid("duplicate_candidate_id: " + candidate_id)
        if len(self.nodes) >= self.max_nodes:
            raise Invalid(f"node_count_out_of_bounds: at most {self.max_nodes}")
        if origin not in ORIGINS:
            raise Invalid("unknown_origin: expected " + ", ".join(ORIGINS))
        if outcome is not None and outcome not in EDGE_OUTCOMES:
            raise Invalid("unknown_outcome: expected " + ", ".join(EDGE_OUTCOMES))
        self.sequence += 1
        node = {
            "id": candidate_id,
            "generation": _generation(generation),
            "origin": origin,
            "outcome": outcome,
            "accepted": outcome in ACCEPTING_OUTCOMES,
            "cell_id": None if cell_id is None else _ident(cell_id, "cell_id"),
            "note": _optional_text(note, "note"),
            "sequence": self.sequence,
        }
        self.nodes[candidate_id] = node
        self.parents_of[candidate_id] = []
        self.children_of[candidate_id] = []
        return {"node": dict(node), "scope": SCOPE}

    def add_edge(self, parent_id, child_id, operator, outcome):
        """Label one parent-to-child step with the operator applied and its outcome."""
        parent_id = _ident(parent_id, "parent_id")
        child_id = _ident(child_id, "child_id")
        operator = _ident(operator, "operator")
        if parent_id not in self.nodes or child_id not in self.nodes:
            raise Invalid("edge_endpoint_is_not_a_recorded_candidate")
        if parent_id == child_id:
            raise Invalid("self_edge_is_not_a_lineage")
        if outcome is not None and outcome not in EDGE_OUTCOMES:
            raise Invalid("unknown_outcome: expected " + ", ".join(EDGE_OUTCOMES))
        if len(self.edges) >= self.max_edges:
            raise Invalid(f"edge_count_out_of_bounds: at most {self.max_edges}")
        if any(e["parent_id"] == parent_id and e["child_id"] == child_id for e in self.edges):
            raise Invalid("duplicate_lineage_edge: " + parent_id + " -> " + child_id)
        if parent_id in self._reachable(child_id, self.children_of):
            raise Invalid(
                "edge_would_create_a_cycle: " + child_id + " already precedes " + parent_id
            )
        self.sequence += 1
        edge = {
            "id": f"edge_{self.sequence}",
            "parent_id": parent_id,
            "child_id": child_id,
            "operator": operator,
            "outcome": outcome,
            "accepted": outcome in ACCEPTING_OUTCOMES,
            "sequence": self.sequence,
        }
        self.edges.append(edge)
        self.parents_of[child_id].append(edge)
        self.children_of[parent_id].append(edge)
        return {"edge": dict(edge), "scope": SCOPE}

    # ------------------------------------------------------------------ traversal

    def _require(self, candidate_id):
        candidate_id = _ident(candidate_id, "candidate_id")
        if candidate_id not in self.nodes:
            raise Invalid("unknown_candidate: " + candidate_id)
        return candidate_id

    def _reachable(self, start, adjacency):
        if start not in self.nodes:
            return set()
        seen, stack = set(), [start]
        key = "child_id" if adjacency is self.children_of else "parent_id"
        while stack:
            current = stack.pop()
            for edge in adjacency[current]:
                nxt = edge[key]
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    def ancestry(self, candidate_id):
        """Every candidate this one descends from, nearest generation first."""
        candidate_id = self._require(candidate_id)
        ancestors = self._reachable(candidate_id, self.parents_of)
        ordered = sorted(
            ancestors,
            key=lambda i: (-self.nodes[i]["generation"], self.nodes[i]["sequence"]),
        )
        seeds = [i for i in ordered if self.nodes[i]["origin"] == "seed"]
        return {
            "candidate_id": candidate_id,
            "ancestor_ids": ordered,
            "ancestor_count": len(ordered),
            "seed_ids": seeds,
            "immediate_parent_ids": sorted(
                e["parent_id"] for e in self.parents_of[candidate_id]
            ),
            "scope": SCOPE,
        }

    def descendants(self, candidate_id):
        """Every candidate derived from this one, however indirectly."""
        candidate_id = self._require(candidate_id)
        found = self._reachable(candidate_id, self.children_of)
        ordered = sorted(
            found, key=lambda i: (self.nodes[i]["generation"], self.nodes[i]["sequence"])
        )
        return {
            "candidate_id": candidate_id,
            "descendant_ids": ordered,
            "descendant_count": len(ordered),
            "accepted_descendant_ids": [i for i in ordered if self.nodes[i]["accepted"]],
            "immediate_child_ids": sorted(
                e["child_id"] for e in self.children_of[candidate_id]
            ),
            "scope": SCOPE,
        }

    def lineage_depth(self, candidate_id):
        """Longest number of operator applications between a seed and this candidate."""
        candidate_id = self._require(candidate_id)
        path = self._deepest_path(candidate_id)
        return {
            "candidate_id": candidate_id,
            "depth": len(path["edges"]),
            "seed_id": path["nodes"][0],
            "definition": (
                "Number of operator applications on the longest path from a seed to "
                "this candidate. Ties are broken by the earliest recorded parent edge."
            ),
            "scope": SCOPE,
        }

    def _deepest_path(self, candidate_id, seen=None):
        seen = set() if seen is None else seen
        if candidate_id in seen:
            raise Invalid("cycle_detected_in_lineage: " + candidate_id)
        incoming = self.parents_of[candidate_id]
        if not incoming:
            return {"nodes": [candidate_id], "edges": [], "branching": 0}
        best = None
        for edge in sorted(incoming, key=lambda e: e["sequence"]):
            upstream = self._deepest_path(edge["parent_id"], seen | {candidate_id})
            key = (len(upstream["edges"]), -edge["sequence"])
            if best is None or key > best[0]:
                best = (key, upstream, edge)
        _, upstream, edge = best
        return {
            "nodes": upstream["nodes"] + [candidate_id],
            "edges": upstream["edges"] + [edge],
            "branching": upstream["branching"] + (1 if len(incoming) > 1 else 0),
        }

    def innovation_path(self, candidate_id):
        """The operator sequence from a seed to this candidate: how it was found."""
        candidate_id = self._require(candidate_id)
        path = self._deepest_path(candidate_id)
        return {
            "candidate_id": candidate_id,
            "seed_id": path["nodes"][0],
            "node_ids": path["nodes"],
            "operators": [e["operator"] for e in path["edges"]],
            "outcomes": [e["outcome"] for e in path["edges"]],
            "edge_ids": [e["id"] for e in path["edges"]],
            "generations": [self.nodes[i]["generation"] for i in path["nodes"]],
            "depth": len(path["edges"]),
            "path_is_unique": path["branching"] == 0,
            "merge_points": path["branching"],
            "rule": (
                "The longest path back to a seed, so the fullest account of how this "
                "candidate arose is reported; where a recombination gives a candidate "
                "two parents, path_is_unique is false and the other parent's own path "
                "is retrievable through ancestry()."
            ),
            "scope": (
                "How the search reached this candidate. The operator sequence is a "
                "record of edits attempted, not a derivation, a proof, or a chain of "
                "evidence for the candidate's content. " + SCOPE
            ),
        }

    # ------------------------------------------------------------------ diagnosis

    def stagnation_report(self, *, current_generation=None, window=5, min_attempts=3):
        """Where the search stopped paying: generations, branches and operators."""
        if current_generation is None:
            current_generation = max(
                (n["generation"] for n in self.nodes.values()), default=0
            )
        current_generation = _generation(current_generation)
        if not isinstance(window, int) or isinstance(window, bool) or not 1 <= window <= 1000:
            raise Invalid("window_out_of_bounds: expected 1..1000")
        if (
            not isinstance(min_attempts, int)
            or isinstance(min_attempts, bool)
            or not 1 <= min_attempts <= 1000
        ):
            raise Invalid("min_attempts_out_of_bounds: expected 1..1000")
        accepted = [n for n in self.nodes.values() if n["accepted"]]
        last = max((n["generation"] for n in accepted), default=None)
        operators = {}
        for edge in self.edges:
            row = operators.setdefault(
                edge["operator"],
                {
                    "operator": edge["operator"],
                    "attempts": 0,
                    "accepted": 0,
                    "recent_attempts": 0,
                    "recent_accepted": 0,
                    "last_accepted_generation": None,
                },
            )
            generation = self.nodes[edge["child_id"]]["generation"]
            recent = generation > current_generation - window
            row["attempts"] += 1
            row["recent_attempts"] += 1 if recent else 0
            if edge["accepted"]:
                row["accepted"] += 1
                row["recent_accepted"] += 1 if recent else 0
                row["last_accepted_generation"] = max(
                    generation, row["last_accepted_generation"] or 0
                )
        for row in operators.values():
            row["acceptance_rate"] = (
                row["accepted"] / row["attempts"] if row["attempts"] else None
            )
        branches = {}
        for node in self.nodes.values():
            seeds = (
                [node["id"]]
                if node["origin"] == "seed"
                else self.ancestry(node["id"])["seed_ids"]
            )
            for seed in seeds or ["unrooted"]:
                row = branches.setdefault(
                    seed,
                    {
                        "seed_id": seed,
                        "candidates": 0,
                        "accepted": 0,
                        "last_accepted_generation": None,
                    },
                )
                row["candidates"] += 1
                if node["accepted"]:
                    row["accepted"] += 1
                    row["last_accepted_generation"] = max(
                        node["generation"], row["last_accepted_generation"] or 0
                    )
        for row in branches.values():
            row["acceptance_rate"] = (
                row["accepted"] / row["candidates"] if row["candidates"] else None
            )
            row["generations_since_accepted"] = (
                None
                if row["last_accepted_generation"] is None
                else current_generation - row["last_accepted_generation"]
            )
        exhausted = sorted(
            row["operator"]
            for row in operators.values()
            if row["attempts"] >= min_attempts and row["recent_accepted"] == 0
        )
        return {
            "current_generation": current_generation,
            "last_improved_generation": last,
            "generations_since_last_improvement": (
                None if last is None else current_generation - last
            ),
            "candidates": len(self.nodes),
            "accepted_candidates": len(accepted),
            "branch_productivity": [branches[k] for k in sorted(branches)],
            "operator_productivity": [operators[k] for k in sorted(operators)],
            "exhausted_operators": exhausted,
            "window": window,
            "min_attempts": min_attempts,
            "definitions": {
                "improvement": "a candidate the archive accepted (" + ", ".join(ACCEPTING_OUTCOMES) + ")",
                "exhausted_operator": (
                    f"at least {min_attempts} recorded attempts and no accepted child "
                    f"in the last {window} generations"
                ),
                "branch": "all candidates descended from one seed; a recombined candidate counts under both seeds",
            },
            "scope": (
                "Stagnation is a statement about this search under these descriptors, "
                "objectives and budget. An exhausted operator has stopped producing "
                "archive entries here; that is not evidence that the mechanism it "
                "edits is absent from the system under study. " + SCOPE
            ),
        }

    # ------------------------------------------------------------------- export

    def to_graph_json(self):
        """A {nodes, edges} payload a frontend graph renderer can draw directly."""
        depths = {}
        for ident in self.nodes:
            depths[ident] = len(self._deepest_path(ident)["edges"])
        return {
            "version": VERSION,
            "nodes": [
                {
                    "id": node["id"],
                    "label": node["id"],
                    "generation": node["generation"],
                    "depth": depths[node["id"]],
                    "origin": node["origin"],
                    "outcome": node["outcome"],
                    "accepted": node["accepted"],
                    "cell_id": node["cell_id"],
                    "note": node["note"],
                    "parent_count": len(self.parents_of[node["id"]]),
                    "child_count": len(self.children_of[node["id"]]),
                }
                for node in sorted(self.nodes.values(), key=lambda n: n["sequence"])
            ],
            "edges": [
                {
                    "id": edge["id"],
                    "source": edge["parent_id"],
                    "target": edge["child_id"],
                    "label": edge["operator"],
                    "operator": edge["operator"],
                    "outcome": edge["outcome"],
                    "accepted": edge["accepted"],
                }
                for edge in self.edges
            ],
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "legend": {
                "origin": list(ORIGINS),
                "outcome": list(EDGE_OUTCOMES),
                "accepted": "outcome is one of " + ", ".join(ACCEPTING_OUTCOMES),
            },
            "scope": SCOPE,
        }

    def to_dict(self):
        """Plain-JSON state for the content-addressed artifact store."""
        return {
            "version": VERSION,
            "max_nodes": self.max_nodes,
            "max_edges": self.max_edges,
            "nodes": [dict(self.nodes[k]) for k in sorted(self.nodes)],
            "edges": [dict(e) for e in self.edges],
            "sequence": self.sequence,
            "scope": SCOPE,
        }

    @classmethod
    def from_dict(cls, data):
        """Rebuild a lineage graph from stored state."""
        if not isinstance(data, dict) or data.get("version") != VERSION:
            raise Invalid("lineage_version_mismatch: expected " + VERSION)
        graph = cls(
            max_nodes=data.get("max_nodes", MAX_NODES),
            max_edges=data.get("max_edges", MAX_EDGES),
        )
        for node in data.get("nodes") or []:
            ident = _ident(node.get("id"), "candidate_id")
            graph.nodes[ident] = dict(node)
            graph.parents_of[ident] = []
            graph.children_of[ident] = []
        for edge in data.get("edges") or []:
            record = dict(edge)
            if record["parent_id"] not in graph.nodes or record["child_id"] not in graph.nodes:
                raise Invalid("edge_endpoint_is_not_a_recorded_candidate")
            graph.edges.append(record)
            graph.parents_of[record["child_id"]].append(record)
            graph.children_of[record["parent_id"]].append(record)
        graph.sequence = int(data.get("sequence", len(graph.nodes) + len(graph.edges)))
        return graph
