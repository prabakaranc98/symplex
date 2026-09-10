"""Scoped local retrieval; lexical similarity is not evidence of entailment."""

import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from symplex.core.contracts import Invalid


def search(store, problem_id, query, limit=5):
    if (
        not isinstance(query, str)
        or not query.strip()
        or len(query) > 2000
        or not 1 <= limit <= 10
    ):
        raise Invalid("Query must be 1–2000 characters, limit 1–10")
    chunks = []
    for r in store.list():
        if (
            r["parent"] != problem_id
            or r["stale"]
            or r["kind"]
            not in ("context", "evidence_note", "solution", "model_critique")
        ):
            continue
        d = r["data"]
        body = d.get("content") or d.get("text") or json.dumps(d)
        for start in range(0, min(len(body), 40000), 900):
            chunks.append(
                {
                    "artifact_id": r["id"],
                    "digest": r["digest"],
                    "offset": start,
                    "text": body[start : start + 1100],
                    "kind": r["kind"],
                }
            )
    if not chunks:
        return {
            "hits": [],
            "method": "local_tfidf",
            "scope": "No searchable current artifacts",
        }
    if len(chunks) > 2000:
        raise Invalid("Local retrieval envelope exceeded; prepare a scoped corpus")
    try:
        matrix = TfidfVectorizer(sublinear_tf=True, max_features=20000).fit_transform(
            [c["text"] for c in chunks] + [query]
        )
        scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    except ValueError:
        return {"hits": [], "method": "local_tfidf", "scope": "No indexable terms"}
    top = sorted(range(len(chunks)), key=lambda i: float(scores[i]), reverse=True)[
        :limit
    ]
    return {
        "hits": [
            dict(chunks[i], similarity=float(scores[i])) for i in top if scores[i] > 0
        ],
        "method": "local_tfidf",
        "scope": "Lexical retrieval over this problem only. Scores are relevance, not truth or semantic equivalence.",
    }
