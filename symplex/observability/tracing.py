"""Artifact-linked spans. No prompts, raw documents or credentials leave the process."""

import contextvars
import time
import uuid
from contextlib import contextmanager

from opentelemetry.sdk.trace import TracerProvider

_PROVIDER = (
    TracerProvider()
)  # No exporter: external telemetry is opt-in, not ambient env driven.
_TRACER = _PROVIDER.get_tracer("symplex", "0.1.0")
_PARENT = contextvars.ContextVar("symplex_span", default=None)


@contextmanager
def span(store, problem_id, name):
    ident = uuid.uuid4().hex
    parent = _PARENT.get()
    token = _PARENT.set(ident)
    start = time.time()
    status, error = "succeeded", None
    try:
        with _TRACER.start_as_current_span(
            name, record_exception=False, set_status_on_exception=False
        ) as current:
            current.set_attribute("symplex.problem_id", problem_id or "workspace")
            current.set_attribute("symplex.span_id", ident)
            yield ident
    except Exception as exc:
        status, error = "failed", type(exc).__name__
        raise
    finally:
        _PARENT.reset(token)
        store.put(
            "trace_span",
            {
                "span_id": ident,
                "parent_span_id": parent,
                "name": name,
                "started_at": start,
                "duration_ms": round((time.time() - start) * 1000, 2),
                "status": status,
                "error_type": error,
                "payload_policy": "metadata_only",
            },
            problem_id,
        )


def export_trace(store, problem_id):
    allowed = {
        "trace_span",
        "model_call",
        "model_failure",
        "agent_action",
        "agent_result",
    }
    ids = {problem_id}
    records = store.list()
    for _ in range(12):
        ids.update(r["id"] for r in records if r["parent"] in ids)
    events = []
    for record in records:
        if record["id"] not in ids or record["kind"] not in allowed:
            continue
        d = record["data"]
        fields = (
            "span_id",
            "parent_span_id",
            "name",
            "started_at",
            "duration_ms",
            "status",
            "error_type",
            "role",
            "model",
            "response_id",
            "usage",
            "usd_upper_estimate",
            "runtime_seconds",
            "request_digest",
            "tool",
            "step",
            "action_id",
        )
        events.append(
            {
                "id": record["id"],
                "parent": record["parent"],
                "kind": record["kind"],
                "created": record["created"],
                **{k: d[k] for k in fields if k in d},
            }
        )
    return events
