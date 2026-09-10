"""Bounded source acquisition with Croissant metadata validation."""

import csv
import hashlib
import io
import json
from datetime import UTC, datetime

import httpx

from symplex.core.contracts import Invalid

BAM_URL = "https://raw.githubusercontent.com/google-research-datasets/BamTwoogle/main/bamtwoogle.csv"
BAM_SOURCE = "https://github.com/google-research-datasets/BamTwoogle"


def download_bam(store, budget):
    existing = [
        r for r in store.list("research_dataset") if r["data"]["name"] == "BamTwoogle"
    ]
    if existing:
        return existing[-1]
    budget.reserve(evidence_requests=1)
    with httpx.stream("GET", BAM_URL, timeout=25, follow_redirects=False) as response:
        response.raise_for_status()
        body = bytearray()
        for chunk in response.iter_bytes():
            body.extend(chunk)
            if len(body) > 100000:
                raise Invalid("Dataset exceeds its approved download envelope")
    rows = list(csv.DictReader(io.StringIO(body.decode())))
    if len(rows) != 100 or any(
        set(r) != {"question", "answer"} or not all(r.values()) for r in rows
    ):
        raise Invalid("BamTwoogle schema or row-count changed; review before use")
    sha = hashlib.sha256(body).hexdigest()
    folder = store.root / "datasets" / sha
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / "bamtwoogle.csv"
    file.write_bytes(body)
    import mlcroissant as mlc

    # Let the maintained SDK supply the full JSON-LD context instead of hand-maintaining it.
    context = mlc.Metadata(
        name="BamTwoogle", description="Google Research benchmark", url=BAM_SOURCE
    ).to_json()["@context"]
    metadata = {
        "@context": context,
        "@type": "sc:Dataset",
        "name": "BamTwoogle",
        "description": "100 multi-step information-seeking questions by Google Research. Local Croissant mapping authored by Symplex; original data unmodified.",
        "url": BAM_SOURCE,
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "distribution": [
            {
                "@type": "cr:FileObject",
                "@id": "bamtwoogle.csv",
                "name": "bamtwoogle.csv",
                "contentUrl": BAM_URL,
                "encodingFormat": "text/csv",
                "sha256": sha,
            }
        ],
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "@id": "questions",
                "name": "questions",
                "field": [
                    {
                        "@type": "cr:Field",
                        "@id": "questions/" + name,
                        "name": name,
                        "dataType": "sc:Text",
                        "source": {
                            "fileObject": {"@id": "bamtwoogle.csv"},
                            "extract": {"column": name},
                        },
                    }
                    for name in ("question", "answer")
                ],
            }
        ],
    }
    croissant = folder / "croissant.json"
    croissant.write_text(json.dumps(metadata, indent=2))
    parsed = mlc.Dataset(jsonld=str(croissant))
    # Metadata validation only. Do not ask SDK to follow arbitrary resources or download data.
    info = {
        "name": "BamTwoogle",
        "source": BAM_SOURCE,
        "download_url": BAM_URL,
        "sha256": sha,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "license": "CC-BY-4.0",
        "rows": len(rows),
        "local_file": str(file),
        "croissant_file": str(croissant),
        "croissant_status": "validated_by_mlcroissant",
        "metadata": parsed.metadata.to_json(),
        "status": "data_verified",
        "synthetic": False,
    }
    ident = store.put("research_dataset", info)
    return store.get(ident)
