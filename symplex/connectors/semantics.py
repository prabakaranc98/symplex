"""RDF projection of proposed system relationships. No remote JSON-LD or SPARQL execution."""

from rdflib import RDF, Graph, Literal, Namespace

from symplex.core.contracts import Invalid

S = Namespace("urn:symplex:")


def project(store, problem_id):
    designs = [
        r
        for r in store.list("solution")
        if r["parent"] == problem_id and not r["stale"]
    ]
    if not designs:
        raise Invalid("Create a current system design before projecting its semantics")
    r = designs[-1]
    g = Graph()
    g.bind("symplex", S)
    nodes = []
    edges = []
    prefix = r["id"] + ":"
    for c in r["data"]["components"]:
        ident = S[prefix + c["id"]]
        g.add((ident, RDF.type, S.Component))
        g.add((ident, S.label, Literal(c["name"])))
        g.add((ident, S.kind, Literal(c["kind"])))
        g.add((ident, S.sourceArtifact, Literal(r["id"])))
        nodes.append({"id": c["id"], "label": c["name"], "kind": c["kind"]})
    for i, c in enumerate(r["data"]["couplings"]):
        relation = S[prefix + "relation:" + str(i)]
        for key, value in c.items():
            g.add((relation, S[key], Literal(value)))
        g.add((relation, S.claimStatus, Literal("proposed_mechanism")))
        g.add((S[prefix + c["source"]], S.influences, S[prefix + c["target"]]))
        edges.append(c)
    return {
        "source_id": r["id"],
        "source_digest": r["digest"],
        "nodes": nodes,
        "edges": edges,
        "turtle": g.serialize(format="turtle"),
        "triple_count": len(g),
        "claim_status": "proposed_mechanisms",
        "scope": "Typed component graph, not a verified causal ontology. Units and timing remain declared text.",
    }
