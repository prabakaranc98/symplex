"""Bound planner memory by serialized bytes; full artifacts remain addressable."""

from copy import deepcopy
from symplex.core.contracts import canonical, Invalid


def compact_history(events):
    items = []
    for index, event in enumerate(events[-8:]):
        result = event.get("result", {})
        if not isinstance(result, dict):
            result = {"summary": str(result)}
        retained = {
            k: v for k, v in result.items() if k.endswith("_id") and isinstance(v, str)
        }
        retained.update(
            {
                k: str(result[k])[:800]
                for k in ("status", "error", "summary", "required_response")
                if k in result
            }
        )
        # The next decision must actually observe the newest tool result. IDs
        # alone erased retrieval hits and caused repeated blind searches.
        if index == len(events[-8:]) - 1:
            encoded = canonical(result)
            retained["observed_result"] = (
                result
                if len(encoded.encode("utf-8")) <= 6500
                else {
                    "excerpt": encoded.encode("utf-8")[:6000].decode(
                        "utf-8", errors="ignore"
                    ),
                    "truncated": True,
                    "full_result_action_id": event.get("artifact_id"),
                }
            )
        items.append(
            {
                "tool": event["tool"],
                "artifact_id": event.get("artifact_id"),
                "result": retained,
            }
        )
    return items


def bounded_planner_context(context, max_bytes=48000):
    """Preserve instructions, current user intent and resource guards before history.

    Never truncate a scientific contract into an apparently valid object: a reduced
    artifact is explicitly an excerpt with its original ID and digest.
    """
    value = deepcopy(context)
    history = value.get("completed_actions", [])
    value["completed_actions"] = compact_history(history)
    value["context_selection"] = {
        "history_total": len(history),
        "history_retained": len(value["completed_actions"]),
        "omitted_artifact_ids": [],
        "excerpted_artifact_ids": [],
        "scope": "Planner summaries only. Retrieve the immutable source artifact before making a claim that requires omitted content.",
    }

    def size():
        return len(canonical(value).encode("utf-8"))

    for entry in reversed(value.get("working_context", [])):
        if size() <= max_bytes:
            break
        data = canonical(entry["data"])
        if len(data) <= 1000:
            continue
        entry["data"] = {
            "excerpt": data[:800],
            "truncated": True,
            "full_artifact_id": entry["id"],
        }
        value["context_selection"]["excerpted_artifact_ids"].append(entry["id"])
    while size() > max_bytes and len(value.get("completed_actions", [])) > 1:
        value["completed_actions"].pop(0)
    while size() > max_bytes and value.get("working_context"):
        removed = value["working_context"].pop()
        value["context_selection"]["omitted_artifact_ids"].append(removed["id"])
    value["context_selection"]["history_retained"] = len(value["completed_actions"])
    if size() > max_bytes:
        raise Invalid(
            "Current intent and host controls exceed the planner context envelope; reduce or split attached directions rather than silently dropping them"
        )
    return value
