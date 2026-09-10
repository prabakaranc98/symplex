"""Ontological and semantic modeling: what the declarations mean, and whether they cohere.

The repo declares a chain intent -> semantics -> structure -> mathematics -> computation
-> evaluation. Until now the semantic link was free text validated for reference
integrity. This package makes four of its promises checkable:

- `dimensions` parses `UnitDefinition.dimension` and gives it algebra, so a declared unit
  stops being prose and `mass + time` stops being possible.
- `ontology` types the investigation graph and refuses an ill-typed triple, projecting to
  and from RDF over SOSA/SSN/QUDT/PROV-O terms.
- `causal` states what an assumed DAG implies, including the conditional independencies
  that make a causal claim falsifiable, and is explicit that identifiability is a
  property of the assumed graph rather than of the world.
- `alignment` looks for the entity, clock and join failures the audits name as recurring,
  before compute is spent on them.

Everything here is deterministic: no network, no model calls, no subprocess, no code
execution. Every public function returns a dict carrying a `scope` string stating what it
has not established.
"""

from symplex.semantics.alignment import (
    align_entities,
    align_time,
    join_risk_report,
)
from symplex.semantics.causal import (
    CausalEdge,
    CausalGraph,
    CausalNode,
    acyclic_check,
    ancestors,
    backdoor_sets,
    d_separation,
    descendants,
    frontdoor_check,
    identifiable,
    instrument_check,
    testable_implications,
)
from symplex.semantics.dimensions import (
    BASE_DIMENSIONS,
    Dimension,
    Unit,
    check_expression_dimensions,
    check_unit_definition,
    check_unit_definitions,
    convert,
    parse_dimension,
    registry,
    unit,
)
from symplex.semantics.ontology import (
    Assumption,
    Claim,
    Entity,
    Measurement,
    Mechanism,
    OntologyGraph,
    Process,
    Quantity,
    Relation,
    Source,
    consistency_report,
    from_rdf,
    graph_summary,
    project_from_complex_system,
    relation_vocabulary,
    to_rdf,
)

__all__ = [
    "BASE_DIMENSIONS",
    "Assumption",
    "CausalEdge",
    "CausalGraph",
    "CausalNode",
    "Claim",
    "Dimension",
    "Entity",
    "Measurement",
    "Mechanism",
    "OntologyGraph",
    "Process",
    "Quantity",
    "Relation",
    "Source",
    "Unit",
    "acyclic_check",
    "align_entities",
    "align_time",
    "ancestors",
    "backdoor_sets",
    "check_expression_dimensions",
    "check_unit_definition",
    "check_unit_definitions",
    "consistency_report",
    "convert",
    "d_separation",
    "descendants",
    "from_rdf",
    "frontdoor_check",
    "graph_summary",
    "identifiable",
    "instrument_check",
    "join_risk_report",
    "parse_dimension",
    "project_from_complex_system",
    "registry",
    "relation_vocabulary",
    "testable_implications",
    "to_rdf",
    "unit",
]
