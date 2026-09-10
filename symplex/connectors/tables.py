"""DuckDB runs fixed queries over an uploaded CSV; no model-generated SQL or filenames."""

import json
import tempfile
from pathlib import Path

import duckdb

from symplex.core.contracts import Invalid


def profile(store, problem_id, artifact_id):
    r = store.get(artifact_id)
    if (
        r["parent"] != problem_id
        or r["kind"] != "context"
        or r["stale"]
        or r["data"].get("format") != "csv"
    ):
        raise Invalid("Choose a current CSV context belonging to this problem")
    content = r["data"]["content"]
    if len(content.encode()) > 1000000:
        raise Invalid("CSV exceeds 1 MB envelope")
    with tempfile.TemporaryDirectory(prefix="symplex-table-") as folder:
        path = Path(folder) / "input.csv"
        path.write_text(content)
        with duckdb.connect(
            config={
                "memory_limit": "128MB",
                "threads": "1",
                "allow_unsigned_extensions": "false",
                "autoinstall_known_extensions": "false",
                "autoload_known_extensions": "false",
            }
        ) as db:
            table = db.read_csv(str(path), header=True, strict_mode=True)
            table.create("uploaded")
            db.execute("SET enable_external_access=false")
            count = db.sql("SELECT count(*) FROM uploaded").fetchone()[0]
            schema = [
                {"column": c[0], "type": c[1]}
                for c in db.sql("DESCRIBE uploaded").fetchall()
            ]
            sample = [
                dict(zip([x["column"] for x in schema], row))
                for row in db.sql("SELECT * FROM uploaded LIMIT 5").fetchall()
            ]
    return {
        "source_id": r["id"],
        "source_digest": r["digest"],
        "rows": count,
        "schema": schema,
        "sample": json.loads(json.dumps(sample, default=str)),
        "method": "duckdb_csv_profile",
        "scope": "Structural inspection only; measurements and time-of-availability remain unverified.",
    }
