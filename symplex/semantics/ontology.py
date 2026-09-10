"""A typed knowledge graph over an investigation, where an ill-typed triple is refused.

`symplex/connectors/semantics.py` serialises a component graph to RDF and checks nothing:
its own scope line says so. This module supplies the missing half. Every node has a type,
every relation has a declared domain and range, and `add_relation` rejects a triple whose
endpoints do not match. Storing a triple therefore means slightly more than it did.

The observation pattern follows SOSA/SSN (W3C Recommendation, *Semantic Sensor Network
Ontology*, 2017): a `Measurement` here plays the role of `sosa:Observation`, a `Quantity`
of `sosa:ObservableProperty`, and an `Entity` of `sosa:FeatureOfInterest`. The
quantity-kind / unit split follows QUDT (`qudt:hasQuantityKind` alongside `qudt:unit`),
so a quantity carries a dimension independent of the unit any particular channel reports
it in. Derivation and activity terms come from PROV-O (W3C Recommendation, 2013).

Two deliberate non-uses of standard terms: `same_as` is NOT emitted as `owl:sameAs`, and
`supports` is not emitted as a PROV term. `owl:sameAs` licenses a reasoner to merge two
individuals, and a proposed entity alignment is exactly the thing this repo's audits say
must not be auto-merged. `supports` is epistemic, not provenance.

Convention: module-level public functions return a dict carrying a `scope` string.
`OntologyGraph` is a container; its mutators return the parsed value and raise `Invalid`.

Nothing here establishes that a relation is true. A well-typed graph is a well-formed
proposal.
"""

import hashlib
from typing import Annotated, ClassVar, Literal

from pydantic import Field, ValidationError, model_validator
from rdflib import RDF, BNode, Graph, Literal as RdfLiteral, Namespace

from symplex.core.contracts import Invalid, digest
from symplex.modeling.complex_system import ClosedContract, Text
from symplex.semantics.dimensions import Dimension, check_unit_definitions
from symplex.semantics.dimensions import unit as registry_unit

MAX_NODES = 4000
MAX_RELATIONS = 16000
MAX_LIST_ITEMS = 64
MAX_FINDINGS = 512

S = Namespace("urn:symplex:")
SOSA = Namespace("http://www.w3.org/ns/sosa/")
SSN = Namespace("http://www.w3.org/ns/ssn/")
QUDT = Namespace("http://qudt.org/schema/qudt/")
PROV = Namespace("http://www.w3.org/ns/prov#")

SCOPE_GRAPH = (
    "A typed proposal graph. Domain/range typing rejects an ill-formed triple; it does "
    "not establish that any asserted relation holds, that a measurement was taken, or "
    "that a claim is supported by the source it cites."
)
SCOPE_REPORT = (
    "Structural findings about the declared graph. A clean report means the declarations "
    "are internally consistent, not that the representation is scientifically adequate. "
    "Absence of a finding is not evidence; findings are limited to the checks listed in "
    "`checks_performed`."
)
SCOPE_RDF = (
    "A lossless RDF projection plus an interoperability layer of SOSA/SSN/QUDT/PROV-O "
    "terms. Reusing a standard IRI records an intended reading; it does not import that "
    "standard's validation, and no reasoner is run. `from_rdf` reads only the "
    "urn:symplex: terms it wrote."
)
SCOPE_PROJECTION = (
    "A typed projection of a declared ComplexSystemSpec. It inherits that spec's status: "
    "a proposed representation, not a validated system. The unit report is the first "
    "point at which a declared dimension in that spec is parsed rather than compared as "
    "text."
)

NodeId = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,159}$")]
Modality = Literal[
    "text",
    "table",
    "image",
    "audio",
    "video",
    "spatial",
    "time_series",
    "graph",
    "molecular",
    "sensor",
    "simulation",
    "human_judgment",
    "other",
]


class Entity(ClosedContract):
    """A thing the investigation is about; `sosa:FeatureOfInterest`."""

    node_kind: ClassVar[str] = "entity"
    id: NodeId
    name: Text
    meaning: Text


class Quantity(ClosedContract):
    """A property of an entity carrying a dimension; `sosa:ObservableProperty`.

    `dimension` is parsed by `symplex.semantics.dimensions`, so it is checkable text.
    `unit_id` is the unit a channel reports it in and is separate from the dimension,
    following QUDT's quantity-kind / unit split.
    """

    node_kind: ClassVar[str] = "quantity"
    id: NodeId
    entity_id: NodeId
    name: Text
    dimension: Text
    unit_id: str = ""
    observability: Literal["observed", "latent", "decision", "exogenous", "derived"]
    meaning: Text

    @model_validator(mode="after")
    def dimension_parses(self):
        Dimension.parse(self.dimension)
        return self


class Measurement(ClosedContract):
    """A quantity observed through an instrument or process; `sosa:Observation`.

    Support in time and space, missingness and uncertainty are required fields because a
    measurement without them cannot be joined to another measurement safely; see
    `symplex.semantics.alignment`.
    """

    node_kind: ClassVar[str] = "measurement"
    id: NodeId
    quantity_id: NodeId
    name: Text
    instrument: Text
    procedure: Text
    modality: Modality
    time_support: Text
    spatial_support: Text
    missingness: Text
    uncertainty: Text
    unit_id: str = ""
    status: Literal["available", "proposed", "missing"]


class Mechanism(ClosedContract):
    """A process relating quantities; `prov:Activity` with used and generated quantities.

    `dimensional_relation` is what makes the endpoints checkable. `unspecified` is
    honest and permitted, and is reported as an unchecked declaration rather than passed
    over in silence.
    """

    node_kind: ClassVar[str] = "mechanism"
    id: NodeId
    name: Text
    description: Text
    input_quantity_ids: list[NodeId] = Field(max_length=MAX_LIST_ITEMS)
    output_quantity_ids: list[NodeId] = Field(max_length=MAX_LIST_ITEMS)
    dimensional_relation: Literal[
        "identity", "rate_per_time", "per_time_squared", "ratio", "unspecified"
    ]


Process = Mechanism
"""`Process` and `Mechanism` name one node type; the spec vocabulary uses both words."""


class Assumption(ClosedContract):
    node_kind: ClassVar[str] = "assumption"
    id: NodeId
    statement: Text
    scope_note: Text
    status: Literal["asserted", "challenged", "retired"]


class Claim(ClosedContract):
    node_kind: ClassVar[str] = "claim"
    id: NodeId
    statement: Text
    disconfirmation: Text
    status: Literal["proposed", "supported", "contradicted", "retired"]


class Source(ClosedContract):
    node_kind: ClassVar[str] = "source"
    id: NodeId
    citation: Text
    source_kind: Literal[
        "artifact",
        "dataset",
        "document",
        "instrument_log",
        "expert_judgment",
        "simulation",
        "unknown",
    ]
    locator: str = ""


NODE_TYPES = {
    cls.node_kind: cls
    for cls in (Entity, Quantity, Measurement, Mechanism, Assumption, Claim, Source)
}

Predicate = Literal[
    "has_quantity",
    "measured_by",
    "derived_from",
    "influences",
    "part_of",
    "same_as",
    "contradicts",
    "supports",
]

RELATIONS = {
    "has_quantity": {
        "pairs": frozenset({("entity", "quantity")}),
        "symmetric": False,
        "rdf": SSN.hasProperty,
        "note": "An entity carries a quantity as a property.",
    },
    "measured_by": {
        "pairs": frozenset({("quantity", "measurement")}),
        "symmetric": False,
        "rdf": None,
        "note": "A quantity has an observation channel; the inverse SOSA term sosa:observedProperty is emitted from the measurement.",
    },
    "derived_from": {
        "pairs": frozenset(
            {
                ("measurement", "source"),
                ("measurement", "measurement"),
                ("quantity", "quantity"),
                ("quantity", "measurement"),
                ("claim", "source"),
                ("claim", "measurement"),
                ("mechanism", "source"),
            }
        ),
        "symmetric": False,
        "rdf": PROV.wasDerivedFrom,
        "note": "Provenance: the subject exists because of the object.",
    },
    "influences": {
        "pairs": frozenset({("quantity", "quantity")}),
        "symmetric": False,
        "rdf": None,
        "note": "A proposed directed influence between quantities. Not a verified causal arrow; see symplex.semantics.causal.",
    },
    "part_of": {
        "pairs": frozenset(
            {("entity", "entity"), ("quantity", "quantity"), ("mechanism", "mechanism")}
        ),
        "symmetric": False,
        "rdf": None,
        "note": "Mereology within one node type only.",
    },
    "same_as": {
        "pairs": frozenset(
            {
                ("entity", "entity"),
                ("quantity", "quantity"),
                ("measurement", "measurement"),
                ("source", "source"),
                ("claim", "claim"),
            }
        ),
        "symmetric": True,
        "rdf": None,
        "note": "A PROPOSED identity between two records of one type. Deliberately not owl:sameAs: nothing here licenses a merge.",
    },
    "contradicts": {
        "pairs": frozenset(
            {
                ("claim", "claim"),
                ("assumption", "assumption"),
                ("claim", "assumption"),
                ("assumption", "claim"),
            }
        ),
        "symmetric": True,
        "rdf": None,
        "note": "The two statements cannot both hold.",
    },
    "supports": {
        "pairs": frozenset(
            {
                ("measurement", "claim"),
                ("source", "claim"),
                ("claim", "claim"),
                ("assumption", "claim"),
                ("measurement", "assumption"),
                ("source", "assumption"),
            }
        ),
        "symmetric": False,
        "rdf": None,
        "note": "Evidential support asserted by the author. Not entailment and not a verified relation.",
    },
}

NODE_CLASS_IRIS = {
    "entity": (SOSA.FeatureOfInterest, PROV.Entity),
    "quantity": (SOSA.ObservableProperty,),
    "measurement": (SOSA.Observation,),
    "mechanism": (PROV.Activity,),
    "assumption": (PROV.Entity,),
    "claim": (PROV.Entity,),
    "source": (PROV.Entity,),
}

_FIELD_NAMES = {name for cls in NODE_TYPES.values() for name in cls.model_fields}
_RESERVED = {"nodeKind", "graphId", "label", "claimStatus", "relationCount"}
if _FIELD_NAMES & set(RELATIONS) or _FIELD_NAMES & _RESERVED:
    raise Invalid(
        "Node field names must not collide with relation or reserved predicate names"
    )


class Relation(ClosedContract):
    subject: NodeId
    predicate: Predicate
    object: NodeId


def source_id_for(text) -> str:
    """Deterministic node id for a free-text artifact reference. Value helper, not a report."""
    if not isinstance(text, str) or not text.strip():
        raise Invalid("Expected a nonempty source reference")
    return "src_" + hashlib.sha256(text.strip().encode()).hexdigest()[:16]


class OntologyGraph:
    """Typed nodes plus typed relations. Mutators raise `Invalid` and return the value."""

    def __init__(self, graph_id="ontology"):
        if (
            not isinstance(graph_id, str)
            or not graph_id
            or len(graph_id) > 160
            or not all(c.isalnum() or c in "_-." for c in graph_id)
        ):
            raise Invalid("graph_id must be a short alphanumeric identifier")
        self.graph_id = graph_id
        self.nodes = {}
        self.relations = []
        self._seen = set()

    def add_node(self, node):
        if len(self.nodes) >= MAX_NODES:
            raise Invalid(f"Ontology graph exceeds the {MAX_NODES}-node envelope")
        if isinstance(node, dict):
            kind = node.get("node_kind") or node.get("kind")
            if kind not in NODE_TYPES:
                raise Invalid(
                    "Node mapping needs a node_kind in: "
                    + ", ".join(sorted(NODE_TYPES))
                )
            payload = {k: v for k, v in node.items() if k not in ("node_kind", "kind")}
            try:
                node = NODE_TYPES[kind].model_validate(payload)
            except ValidationError as error:
                raise Invalid(
                    f"Invalid {kind} node: {error.errors()[0]['msg']}"
                ) from None
        if type(node) not in NODE_TYPES.values():
            raise Invalid("Expected an ontology node instance or mapping")
        if node.id in self.nodes:
            raise Invalid("Duplicate ontology node ID: " + node.id)
        self.nodes[node.id] = node
        return node

    def add_nodes(self, nodes):
        return [self.add_node(node) for node in nodes]

    def kind_of(self, node_id):
        node = self.nodes.get(node_id)
        if node is None:
            raise Invalid("Unknown ontology node: " + str(node_id))
        return type(node).node_kind

    def add_relation(self, subject, predicate, object_):
        """Assert a typed triple, refusing one whose endpoint types the relation forbids."""
        if len(self.relations) >= MAX_RELATIONS:
            raise Invalid(
                f"Ontology graph exceeds the {MAX_RELATIONS}-relation envelope"
            )
        if predicate not in RELATIONS:
            raise Invalid(
                "Unknown relation "
                + repr(predicate)
                + "; the vocabulary is closed: "
                + ", ".join(sorted(RELATIONS))
            )
        try:
            relation = Relation.model_validate(
                {"subject": subject, "predicate": predicate, "object": object_}
            )
        except ValidationError as error:
            raise Invalid(f"Invalid relation: {error.errors()[0]['msg']}") from None
        subject_kind = self.kind_of(relation.subject)
        object_kind = self.kind_of(relation.object)
        if relation.subject == relation.object:
            raise Invalid(
                f"Relation {predicate} cannot relate {relation.subject} to itself"
            )
        pairs = RELATIONS[predicate]["pairs"]
        if (subject_kind, object_kind) not in pairs:
            raise Invalid(
                f"Relation {predicate} does not accept {subject_kind} -> {object_kind}; "
                "permitted: " + ", ".join(f"{a} -> {b}" for a, b in sorted(pairs))
            )
        key = (relation.subject, predicate, relation.object)
        if RELATIONS[predicate]["symmetric"]:
            key = (predicate, *sorted((relation.subject, relation.object)))
        if key in self._seen:
            return relation
        self._seen.add(key)
        self.relations.append(relation)
        return relation

    def add_relations(self, triples):
        return [self.add_relation(*triple) for triple in triples]

    def by_kind(self, kind):
        return [n for n in self.nodes.values() if type(n).node_kind == kind]

    def out(self, subject, predicate=None):
        return [
            r.object
            for r in self.relations
            if r.subject == subject and (predicate is None or r.predicate == predicate)
        ]

    def incoming(self, object_, predicate=None):
        return [
            r.subject
            for r in self.relations
            if r.object == object_ and (predicate is None or r.predicate == predicate)
        ]

    def neighbours(self, node_id, predicate):
        """Both directions, which is what a symmetric relation such as same_as needs."""
        return set(self.out(node_id, predicate)) | set(
            self.incoming(node_id, predicate)
        )

    def snapshot(self):
        return {
            "graph_id": self.graph_id,
            "nodes": [
                {"node_kind": type(n).node_kind, **n.model_dump()}
                for n in sorted(self.nodes.values(), key=lambda n: n.id)
            ],
            "relations": sorted(
                (r.model_dump() for r in self.relations),
                key=lambda r: (r["subject"], r["predicate"], r["object"]),
            ),
        }

    def digest(self):
        return digest(self.snapshot())


def relation_vocabulary() -> dict:
    """The closed relation vocabulary with its domain/range table."""
    return {
        "relations": [
            {
                "predicate": name,
                "permitted": sorted(f"{a} -> {b}" for a, b in spec["pairs"]),
                "symmetric": spec["symmetric"],
                "rdf_term": str(spec["rdf"]) if spec["rdf"] else None,
                "note": spec["note"],
            }
            for name, spec in sorted(RELATIONS.items())
        ],
        "node_kinds": sorted(NODE_TYPES),
        "closed": True,
        "scope": SCOPE_GRAPH,
    }


def graph_summary(graph) -> dict:
    """A JSON-safe view of the graph."""
    graph = _require_graph(graph)
    snapshot = graph.snapshot()
    counts = {}
    for node in snapshot["nodes"]:
        counts[node["node_kind"]] = counts.get(node["node_kind"], 0) + 1
    predicate_counts = {}
    for relation in snapshot["relations"]:
        predicate_counts[relation["predicate"]] = (
            predicate_counts.get(relation["predicate"], 0) + 1
        )
    return {
        **snapshot,
        "node_count": len(snapshot["nodes"]),
        "relation_count": len(snapshot["relations"]),
        "nodes_by_kind": dict(sorted(counts.items())),
        "relations_by_predicate": dict(sorted(predicate_counts.items())),
        "graph_digest": graph.digest(),
        "scope": SCOPE_GRAPH,
    }


def _require_graph(graph) -> OntologyGraph:
    if not isinstance(graph, OntologyGraph):
        raise Invalid("Expected an OntologyGraph")
    return graph


def _quantity_dimension(node):
    return Dimension.parse(node.dimension)


def _components(graph, predicate):
    """Undirected connected components over one relation."""
    parent = {node_id: node_id for node_id in graph.nodes}

    def find(node_id):
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    for relation in graph.relations:
        if relation.predicate != predicate:
            continue
        a, b = find(relation.subject), find(relation.object)
        if a != b:
            parent[a] = b
    groups = {}
    for node_id in graph.nodes:
        groups.setdefault(find(node_id), []).append(node_id)
    return [sorted(members) for members in groups.values() if len(members) > 1]


CONSISTENCY_CHECKS = (
    "quantity_without_measurement_channel",
    "measurement_without_source",
    "orphan_entity",
    "unconnected_node",
    "same_as_merges_different_dimensions",
    "contradicting_claims_share_a_source",
    "mechanism_endpoint_dimension_mismatch",
    "mechanism_dimensional_relation_unspecified",
    "quantity_unit_dimension_mismatch",
    "dangling_quantity_entity",
)


def consistency_report(graph) -> dict:
    """Report structural defects in a typed graph. Findings are defects, not verdicts."""
    graph = _require_graph(graph)
    findings = []

    def add(code, severity, subject, detail, **extra):
        if len(findings) < MAX_FINDINGS:
            findings.append(
                {
                    "code": code,
                    "severity": severity,
                    "subject": subject,
                    "detail": detail,
                    **extra,
                }
            )

    for quantity in graph.by_kind("quantity"):
        if not graph.out(quantity.id, "measured_by"):
            add(
                "quantity_without_measurement_channel",
                "warning",
                quantity.id,
                "No measurement channel is declared for this quantity, so nothing in the "
                "evidence base can constrain it.",
            )
        if quantity.entity_id not in graph.nodes:
            add(
                "dangling_quantity_entity",
                "error",
                quantity.id,
                "The quantity names entity "
                + quantity.entity_id
                + ", which is absent.",
            )
        if quantity.unit_id:
            try:
                declared_unit = registry_unit(quantity.unit_id)
            except Invalid:
                declared_unit = None
            if (
                declared_unit is not None
                and declared_unit.dimension != _quantity_dimension(quantity)
            ):
                add(
                    "quantity_unit_dimension_mismatch",
                    "error",
                    quantity.id,
                    "Unit "
                    + declared_unit.symbol
                    + " has dimension "
                    + declared_unit.dimension.canonical
                    + " but the quantity declares "
                    + _quantity_dimension(quantity).canonical,
                )

    for measurement in graph.by_kind("measurement"):
        sources = [
            other
            for other in graph.out(measurement.id, "derived_from")
            if graph.kind_of(other) == "source"
        ]
        if not sources:
            add(
                "measurement_without_source",
                "warning",
                measurement.id,
                "No source is attached, so the observation has no provenance and cannot "
                "be audited.",
            )

    for entity in graph.by_kind("entity"):
        if not graph.out(entity.id, "has_quantity") and not graph.neighbours(
            entity.id, "part_of"
        ):
            add(
                "orphan_entity",
                "warning",
                entity.id,
                "The entity carries no quantity and belongs to no larger entity, so it "
                "plays no role in the representation.",
            )

    connected = set()
    for relation in graph.relations:
        connected.add(relation.subject)
        connected.add(relation.object)
    for node_id in sorted(set(graph.nodes) - connected):
        node = graph.nodes[node_id]
        kind = type(node).node_kind
        if kind == "entity":
            continue
        if kind == "mechanism" and (
            node.input_quantity_ids or node.output_quantity_ids
        ):
            continue
        add(
            "unconnected_node",
            "info",
            node_id,
            "The node participates in no relation.",
        )

    for component in _components(graph, "same_as"):
        dimensions = {}
        for node_id in component:
            if graph.kind_of(node_id) == "quantity":
                dimensions.setdefault(
                    _quantity_dimension(graph.nodes[node_id]).canonical, []
                ).append(node_id)
        if len(dimensions) > 1:
            add(
                "same_as_merges_different_dimensions",
                "error",
                component[0],
                "A same_as component would identify quantities of different dimensions: "
                + "; ".join(
                    f"{canonical}: {', '.join(ids)}"
                    for canonical, ids in sorted(dimensions.items())
                ),
                component=component,
            )

    for relation in graph.relations:
        if relation.predicate != "contradicts":
            continue
        shared = set()
        for node_id in (relation.subject, relation.object):
            shared = (
                set(graph.out(node_id, "derived_from"))
                if not shared
                else shared & set(graph.out(node_id, "derived_from"))
            )
        shared = {node_id for node_id in shared if graph.kind_of(node_id) == "source"}
        if shared:
            add(
                "contradicting_claims_share_a_source",
                "warning",
                relation.subject,
                "Contradicting statements are derived from the same source(s): "
                + ", ".join(sorted(shared))
                + ". Either the source is being read two ways or the contradiction is not real.",
                other=relation.object,
                shared_sources=sorted(shared),
            )

    for mechanism in graph.by_kind("mechanism"):
        endpoints = {"input": [], "output": []}
        missing = []
        for role, ids in (
            ("input", mechanism.input_quantity_ids),
            ("output", mechanism.output_quantity_ids),
        ):
            for node_id in ids:
                node = graph.nodes.get(node_id)
                if node is None or type(node).node_kind != "quantity":
                    missing.append(node_id)
                    continue
                endpoints[role].append((node_id, _quantity_dimension(node)))
        if missing:
            add(
                "mechanism_endpoint_dimension_mismatch",
                "error",
                mechanism.id,
                "Mechanism endpoints are absent or are not quantities: "
                + ", ".join(sorted(missing)),
            )
            continue
        if mechanism.dimensional_relation == "unspecified":
            add(
                "mechanism_dimensional_relation_unspecified",
                "info",
                mechanism.id,
                "The mechanism declares no dimensional relation between its endpoints, "
                "so no dimensional check was performed on it.",
            )
            continue
        for role in ("input", "output"):
            distinct = {d.canonical for _, d in endpoints[role]}
            if len(distinct) > 1:
                add(
                    "mechanism_endpoint_dimension_mismatch",
                    "error",
                    mechanism.id,
                    f"The {role} quantities do not share one dimension: "
                    + ", ".join(sorted(distinct)),
                )
        if not endpoints["input"] or not endpoints["output"]:
            continue
        source = endpoints["input"][0][1]
        target = endpoints["output"][0][1]
        expected = {
            "identity": source,
            "rate_per_time": source / Dimension.base("time"),
            "per_time_squared": source / Dimension.base("time") ** 2,
            "ratio": Dimension.dimensionless(),
        }[mechanism.dimensional_relation]
        if target != expected:
            add(
                "mechanism_endpoint_dimension_mismatch",
                "error",
                mechanism.id,
                "Declared relation "
                + mechanism.dimensional_relation
                + " over input "
                + source.canonical
                + " requires output "
                + expected.canonical
                + ", but the output quantity declares "
                + target.canonical,
            )

    severities = {"error": 0, "warning": 0, "info": 0}
    for finding in findings:
        severities[finding["severity"]] += 1
    return {
        "status": "failed" if severities["error"] else "checked",
        "consistent": not severities["error"],
        "findings": findings,
        "finding_count": len(findings),
        "severity_counts": severities,
        "truncated": len(findings) >= MAX_FINDINGS,
        "checks_performed": list(CONSISTENCY_CHECKS),
        "node_count": len(graph.nodes),
        "relation_count": len(graph.relations),
        "graph_digest": graph.digest(),
        "scope": SCOPE_REPORT,
    }


def _node_iri(graph_id, node_id):
    return S[f"{graph_id}:{node_id}"]


def _kind_slug(canonical):
    """A URI-safe rendering of a canonical dimension, reversible by inspection."""
    return canonical.replace("^", "~").replace("*", ".")


def _write_list(rdf, subject, predicate, values):
    if not values:
        rdf.add((subject, predicate, RDF.nil))
        return
    head = BNode()
    rdf.add((subject, predicate, head))
    for index, value in enumerate(values):
        rdf.add((head, RDF.first, RdfLiteral(value)))
        tail = RDF.nil if index == len(values) - 1 else BNode()
        rdf.add((head, RDF.rest, tail))
        head = tail


def _read_list(rdf, subject, predicate):
    node = rdf.value(subject, predicate)
    if node is None or node == RDF.nil:
        return []
    values = []
    while node is not None and node != RDF.nil:
        if len(values) > MAX_LIST_ITEMS:
            raise Invalid("RDF list exceeds the ontology list envelope")
        item = rdf.value(node, RDF.first)
        if item is None:
            raise Invalid("Malformed RDF list in ontology payload")
        values.append(item.toPython())
        node = rdf.value(node, RDF.rest)
    return values


def to_rdf(graph) -> dict:
    """Serialise the typed graph to Turtle over stable urn:symplex: IRIs.

    Node and relation terms live in the same `urn:symplex:` namespace that
    `symplex/connectors/semantics.py` already uses, including `symplex:influences` and
    `symplex:claimStatus`, so the two projections can share a store. SOSA/SSN/QUDT/PROV-O
    triples are added alongside for interoperability and are not read back.
    """
    graph = _require_graph(graph)
    rdf = Graph()
    rdf.bind("symplex", S)
    rdf.bind("sosa", SOSA)
    rdf.bind("ssn", SSN)
    rdf.bind("qudt", QUDT)
    rdf.bind("prov", PROV)
    root = S[graph.graph_id]
    rdf.add((root, RDF.type, S.SemanticGraph))
    rdf.add((root, S.graphId, RdfLiteral(graph.graph_id)))
    rdf.add((root, S.claimStatus, RdfLiteral("proposed_semantic_graph")))
    for node in graph.nodes.values():
        kind = type(node).node_kind
        subject = _node_iri(graph.graph_id, node.id)
        rdf.add((subject, RDF.type, S[kind.capitalize()]))
        rdf.add((subject, S.nodeKind, RdfLiteral(kind)))
        for iri in NODE_CLASS_IRIS[kind]:
            rdf.add((subject, RDF.type, iri))
        for name, value in node.model_dump().items():
            if isinstance(value, list):
                _write_list(rdf, subject, S[name], value)
            elif value is not None:
                rdf.add((subject, S[name], RdfLiteral(value)))
        label = getattr(node, "name", None) or getattr(node, "statement", None)
        if label:
            rdf.add((subject, S.label, RdfLiteral(label)))
        if kind == "quantity":
            canonical = Dimension.parse(node.dimension).canonical
            kind_iri = S["quantitykind:" + _kind_slug(canonical)]
            rdf.add((subject, QUDT.hasQuantityKind, kind_iri))
            rdf.add((kind_iri, RDF.type, QUDT.QuantityKind))
            rdf.add((kind_iri, S.label, RdfLiteral(canonical)))
            if node.unit_id:
                rdf.add((subject, QUDT.unit, S["unit:" + node.unit_id]))
            entity = _node_iri(graph.graph_id, node.entity_id)
            rdf.add((subject, SSN.isPropertyOf, entity))
        if kind == "measurement":
            rdf.add(
                (
                    subject,
                    SOSA.observedProperty,
                    _node_iri(graph.graph_id, node.quantity_id),
                )
            )
            sensor = S[f"{graph.graph_id}:sensor:{node.id}"]
            procedure = S[f"{graph.graph_id}:procedure:{node.id}"]
            rdf.add((sensor, RDF.type, SOSA.Sensor))
            rdf.add((sensor, S.label, RdfLiteral(node.instrument)))
            rdf.add((procedure, RDF.type, SOSA.Procedure))
            rdf.add((procedure, S.label, RdfLiteral(node.procedure)))
            rdf.add((subject, SOSA.madeBySensor, sensor))
            rdf.add((subject, SOSA.usedProcedure, procedure))
            quantity = graph.nodes.get(node.quantity_id)
            if quantity is not None and type(quantity).node_kind == "quantity":
                rdf.add(
                    (
                        subject,
                        SOSA.hasFeatureOfInterest,
                        _node_iri(graph.graph_id, quantity.entity_id),
                    )
                )
        if kind == "mechanism":
            for used in node.input_quantity_ids:
                rdf.add((subject, PROV.used, _node_iri(graph.graph_id, used)))
            for generated in node.output_quantity_ids:
                rdf.add(
                    (
                        _node_iri(graph.graph_id, generated),
                        PROV.wasGeneratedBy,
                        subject,
                    )
                )
    for relation in graph.relations:
        subject = _node_iri(graph.graph_id, relation.subject)
        object_ = _node_iri(graph.graph_id, relation.object)
        rdf.add((subject, S[relation.predicate], object_))
        interop = RELATIONS[relation.predicate]["rdf"]
        if interop is not None:
            rdf.add((subject, interop, object_))
    return {
        "graph_id": graph.graph_id,
        "turtle": rdf.serialize(format="turtle"),
        "triple_count": len(rdf),
        "node_count": len(graph.nodes),
        "relation_count": len(graph.relations),
        "namespaces": {
            "symplex": str(S),
            "sosa": str(SOSA),
            "ssn": str(SSN),
            "qudt": str(QUDT),
            "prov": str(PROV),
        },
        "reasoner_applied": False,
        "claim_status": "proposed_semantic_graph",
        "graph_digest": graph.digest(),
        "scope": SCOPE_RDF,
    }


def from_rdf(turtle, rdf_format="turtle") -> dict:
    """Rebuild a typed graph from Turtle written by `to_rdf`, re-checking every triple.

    Reading is not trusting: each node is revalidated against its contract and each
    relation goes back through `add_relation`, so a hand-edited payload with an ill-typed
    triple is rejected here exactly as it would be on first assertion.
    """
    if not isinstance(turtle, str) or not turtle.strip():
        raise Invalid("Expected a nonempty RDF payload")
    if len(turtle) > 20_000_000:
        raise Invalid("RDF payload exceeds the 20 MB envelope")
    rdf = Graph()
    try:
        rdf.parse(data=turtle, format=rdf_format)
    except Exception:
        raise Invalid("Expected a parseable RDF payload") from None
    root = next(rdf.subjects(RDF.type, S.SemanticGraph), None)
    graph_id = str(rdf.value(root, S.graphId)) if root is not None else "ontology"
    graph = OntologyGraph(graph_id)
    prefix = f"{S}{graph_id}:"
    entries = []
    for subject, kind_literal in rdf.subject_objects(S.nodeKind):
        kind = str(kind_literal)
        if kind not in NODE_TYPES:
            raise Invalid("Unknown node kind in RDF payload: " + kind)
        if not str(subject).startswith(prefix):
            raise Invalid(
                "Node IRI outside the declared graph namespace: " + str(subject)
            )
        payload = {}
        for name, field in NODE_TYPES[kind].model_fields.items():
            if str(field.annotation).startswith("list"):
                payload[name] = _read_list(rdf, subject, S[name])
                continue
            value = rdf.value(subject, S[name])
            if value is not None:
                payload[name] = value.toPython()
        entries.append((str(subject)[len(prefix) :], kind, payload))
    for node_id, kind, payload in sorted(entries):
        if payload.get("id") != node_id:
            raise Invalid("RDF node IRI disagrees with its declared id: " + node_id)
        graph.add_node({"node_kind": kind, **payload})
    for predicate in sorted(RELATIONS):
        for subject, object_ in sorted(rdf.subject_objects(S[predicate])):
            if not (
                str(subject).startswith(prefix) and str(object_).startswith(prefix)
            ):
                continue
            graph.add_relation(
                str(subject)[len(prefix) :], predicate, str(object_)[len(prefix) :]
            )
    return {
        "graph": graph,
        "graph_id": graph_id,
        "node_count": len(graph.nodes),
        "relation_count": len(graph.relations),
        "triple_count": len(rdf),
        "revalidated": True,
        "graph_digest": graph.digest(),
        "scope": SCOPE_RDF,
    }


def _text(value, fallback):
    value = (value or "").strip() if isinstance(value, str) else ""
    return (value or fallback)[:1600]


def project_from_complex_system(spec_dict, graph_id="complex_system") -> dict:
    """Build a typed ontology graph from a saved `ComplexSystemSpec` dict.

    This is the join between the existing host contract and this layer, and it modifies
    nothing in `symplex/modeling/complex_system.py`. States become quantities carrying the
    parsed dimension of their declared unit, observation channels become measurements with
    their source artifacts, components become mechanisms over their port quantities, rival
    hypotheses become mutually contradicting claims, and declared assumptions become
    assumption nodes.

    Relations the spec implies but that this vocabulary refuses are collected in
    `rejected_relations` rather than dropped, so a projection failure is visible.
    """
    from symplex.modeling.complex_system import ComplexSystemSpec

    if isinstance(spec_dict, ComplexSystemSpec):
        spec = spec_dict.model_dump()
    else:
        try:
            spec = ComplexSystemSpec.model_validate(spec_dict).model_dump()
        except ValidationError as error:
            raise Invalid(
                "project_from_complex_system needs a valid ComplexSystemSpec: "
                + error.errors()[0]["msg"]
            ) from None
    unit_check = check_unit_definitions(spec["units"])
    units = {u["id"]: u for u in spec["units"]}
    graph = OntologyGraph(graph_id)
    rejected = []

    def relate(subject, predicate, object_):
        try:
            graph.add_relation(subject, predicate, object_)
        except Invalid as error:
            rejected.append(
                {
                    "subject": subject,
                    "predicate": predicate,
                    "object": object_,
                    "reason": str(error),
                }
            )

    for entity in spec["entities"]:
        graph.add_node(
            Entity(
                id="entity_" + entity["id"],
                name=entity["name"],
                meaning=entity["meaning"],
            )
        )
    for state in spec["states"]:
        unit = units[state["unit_id"]]
        graph.add_node(
            Quantity(
                id="quantity_" + state["id"],
                entity_id="entity_" + state["entity_id"],
                name=state["name"],
                dimension=Dimension.parse(unit["dimension"]).canonical,
                unit_id=state["unit_id"],
                observability=state["kind"],
                meaning=state["meaning"],
            )
        )
        relate(
            "entity_" + state["entity_id"], "has_quantity", "quantity_" + state["id"]
        )

    sources = {}
    for channel in spec["observations"]:
        for reference in channel["source_artifact_ids"]:
            node_id = source_id_for(reference)
            if node_id not in sources:
                sources[node_id] = graph.add_node(
                    Source(
                        id=node_id,
                        citation=_text(reference, "unnamed artifact"),
                        source_kind="artifact",
                        locator=reference[:512],
                    )
                )
    for holder in list(spec["hypotheses"]) + list(spec["components"]):
        for reference in holder["evidence_ids"]:
            node_id = source_id_for(reference)
            if node_id not in sources:
                sources[node_id] = graph.add_node(
                    Source(
                        id=node_id,
                        citation=_text(reference, "unnamed artifact"),
                        source_kind="artifact",
                        locator=reference[:512],
                    )
                )

    for channel in spec["observations"]:
        for state_id in channel["state_ids"]:
            measurement_id = f"measurement_{channel['id']}__{state_id}"[:160]
            graph.add_node(
                Measurement(
                    id=measurement_id,
                    quantity_id="quantity_" + state_id,
                    name=channel["name"],
                    instrument=_text(
                        channel["measurement_process"], "unspecified instrument"
                    ),
                    procedure=_text(
                        channel["measurement_process"], "unspecified procedure"
                    ),
                    modality=channel["modality"],
                    time_support=_text(
                        None,
                        "Not declared in the source spec; ObservationChannel carries no time support field.",
                    ),
                    spatial_support=_text(
                        None,
                        "Not declared in the source spec; ObservationChannel carries no spatial support field.",
                    ),
                    missingness=channel["missingness"],
                    uncertainty=channel["uncertainty"],
                    unit_id="",
                    status=channel["status"],
                )
            )
            relate("quantity_" + state_id, "measured_by", measurement_id)
            for reference in channel["source_artifact_ids"]:
                relate(measurement_id, "derived_from", source_id_for(reference))

    for component in spec["components"]:
        graph.add_node(
            Mechanism(
                id="mechanism_" + component["id"],
                name=component["name"],
                description=component["mechanism"],
                input_quantity_ids=sorted(
                    {
                        "quantity_" + p["state_id"]
                        for p in component["ports"]
                        if p["direction"] == "input"
                    }
                ),
                output_quantity_ids=sorted(
                    {
                        "quantity_" + p["state_id"]
                        for p in component["ports"]
                        if p["direction"] == "output"
                    }
                ),
                dimensional_relation="unspecified",
            )
        )
        for reference in component["evidence_ids"]:
            relate(
                "mechanism_" + component["id"], "derived_from", source_id_for(reference)
            )
    ports = {p["id"]: p for c in spec["components"] for p in c["ports"]}
    for coupling in spec["couplings"]:
        source_state = ports[coupling["source_port"]]["state_id"]
        target_state = ports[coupling["target_port"]]["state_id"]
        if source_state != target_state:
            relate("quantity_" + source_state, "influences", "quantity_" + target_state)

    for hypothesis in spec["hypotheses"]:
        graph.add_node(
            Claim(
                id="claim_" + hypothesis["id"],
                statement=hypothesis["claim"],
                disconfirmation=hypothesis["rejection_condition"],
                status="proposed",
            )
        )
    for hypothesis in spec["hypotheses"]:
        for rival in hypothesis["rivals"]:
            relate("claim_" + hypothesis["id"], "contradicts", "claim_" + rival)
        for reference in hypothesis["evidence_ids"]:
            relate(
                "claim_" + hypothesis["id"], "derived_from", source_id_for(reference)
            )
            relate(source_id_for(reference), "supports", "claim_" + hypothesis["id"])

    statements = list(spec["assumptions"])
    for component in spec["components"]:
        statements.extend(component["assumptions"])
    for index, statement in enumerate(statements[:MAX_LIST_ITEMS]):
        graph.add_node(
            Assumption(
                id=f"assumption_{index}",
                statement=_text(statement, "unstated assumption"),
                scope_note="Declared in the ComplexSystemSpec; not independently checked.",
                status="asserted",
            )
        )

    expression_environment = {
        state["id"]: Dimension.parse(units[state["unit_id"]]["dimension"]).canonical
        for state in spec["states"]
    }
    consistency = consistency_report(graph)
    return {
        "graph": graph,
        "graph_id": graph_id,
        "title": spec["title"],
        "node_count": len(graph.nodes),
        "relation_count": len(graph.relations),
        "unit_check": unit_check,
        "consistency": consistency,
        "rejected_relations": rejected,
        "rejected_relation_count": len(rejected),
        "expression_environment": expression_environment,
        "spec_digest": digest(spec),
        "empirically_validated": False,
        "scope": SCOPE_PROJECTION,
    }
