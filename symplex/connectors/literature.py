"""Public literature metadata access through a fixed Crossref endpoint."""

import json
from datetime import UTC, datetime

import httpx

from symplex.core.contracts import Invalid, digest


def search(store, budget, problem_id, query):
    if not 1 <= len(query) <= 2000:
        raise Invalid("Literature query exceeds envelope")
    budget.reserve(evidence_requests=1)
    with httpx.stream(
        "GET",
        "https://api.crossref.org/works",
        params={"query.bibliographic": query, "rows": 5},
        headers={"User-Agent": "Symplex/0.1 (local research workbench)"},
        timeout=30,
        follow_redirects=False,
    ) as response:
        response.raise_for_status()
        body = bytearray()
        for chunk in response.iter_bytes():
            body.extend(chunk)
            if len(body) > 500000:
                raise Invalid("Literature metadata response too large")
    works = json.loads(body)["message"]["items"]
    items = [
        {
            "title": (w.get("title") or ["Untitled"])[0],
            "doi": w.get("DOI"),
            "url": w.get("URL"),
            "publisher": w.get("publisher"),
            "published": w.get("published"),
            "license": w.get("license", []),
        }
        for w in works
    ]
    result = {
        "text": "Literature metadata candidates only; full text and claim support have not been verified.\n"
        + json.dumps(items),
        "citations": [
            {"title": w["title"], "url": w["url"]} for w in items if w["url"]
        ],
        "items": items,
        "query": query,
        "source": "https://api.crossref.org/works",
        "retrieved_at": datetime.now(UTC).isoformat(),
        "response_digest": digest(json.loads(body)),
        "mode": "literature_metadata",
        "scope": "Discovery is not scientific validation or full-text access.",
    }
    ident = store.put("evidence_note", result, problem_id)
    return {"evidence_id": ident, **result}
