"""Fit complete primary contracts plus auditable ancillary inputs to a proposal request."""

import copy
import os

from symplex.agents.prompts import prompt
from symplex.core.contracts import Invalid, canonical
from symplex.infrastructure.limits import DEFAULT_ROLE_REQUEST_BYTES, MAX_REQUEST_BYTES


def proposal_request_bytes(contract, context, provider=None, role="heavy"):
    """Mirror the provider's canonical wire envelope, including escaped context and schema."""
    from symplex.infrastructure.providers import ROLES, api_schema
    models = getattr(provider, "models", ROLES)
    output = getattr(contract, "output_token_budget", 6000 if contract.__name__ in
                     ("SolutionBlueprint", "SystemSimulation", "CodeProposal") else 2400)
    request = {
        "instructions": prompt("proposer"), "input": [{"role": "user", "content": canonical(context)}],
        "tools": [{"type": "function", "name": "submit_proposal",
                   "description": "Submit one bounded proposal for host validation. It does not execute itself.",
                   "strict": True, "parameters": api_schema(contract.json_schema())}],
        "max_output_tokens": output, "tool_choice": {"type": "function", "name": "submit_proposal"},
        "parallel_tool_calls": False, "model": models.get(role, ROLES[role]),
        "store": False, "service_tier": "default",
        "reasoning": {"effort": os.getenv("OPENAI_" + role.upper() + "_REASONING_EFFORT",
                         os.getenv("OPENAI_REASONING_EFFORT", getattr(provider, "reasoning_effort", "medium")
                                   if role == "heavy" else "low"))},
    }
    return len(canonical(request).encode("utf-8")) + 2048


def _set_path(value, path, entry):
    keys = (path,) if isinstance(path, str) else path
    parent = value
    for key in keys[:-1]:
        parent = parent.setdefault(key, {})
    parent[keys[-1]] = entry


def _reference(entry, field, reason=None):
    reference = {k: entry[k] for k in ("id", "kind", "digest", "filename") if k in entry}
    reference["field"] = ".".join(field) if isinstance(field, tuple) else field
    if reason:
        reference["reason"] = reason
    return reference


def bounded_proposal_context(store, provider, contract, primary, collections, problem_id, *,
                             role="heavy", primary_ids=(), optional_records=None, max_request_bytes=None,
                             derive_metadata=None):
    """Never truncate primary contracts or ancillary records; omit whole inputs with lineage.

    Collections map field names or nested path tuples to ordered artifact lists.
    Optional records retain their original shape if admitted, otherwise remain None.
    """
    explicit_limit = max_request_bytes is not None
    max_request_bytes = max_request_bytes or DEFAULT_ROLE_REQUEST_BYTES
    if not 2048 < max_request_bytes <= MAX_REQUEST_BYTES:
        raise Invalid("Role request limit must fit the provider envelope")
    context = copy.deepcopy(primary)
    included, omitted = [], []
    optional_records = optional_records or {}
    for field in collections:
        _set_path(context, field, [])
    for field in optional_records:
        _set_path(context, field, None)
    # Fixed-size placeholder keeps the final persisted manifest ID inside the same admission bound.
    context["context_selection"] = {"manifest_id": "role_context_manifest_" + "0" * 16,
        "included_count": 0, "omitted_count": 0,
        "scope": "Full primary contracts retained; ancillary inputs may be omitted. Only actually supplied content supports citations; the manifest inventories omissions."}
    if derive_metadata:
        derive_metadata(context)
    primary_bytes = proposal_request_bytes(contract, context, provider, role)
    if not explicit_limit and primary_bytes > max_request_bytes:
        # Grow only enough for the complete primary contract and its manifest.
        # Do not fill a larger model context with ancillary historical records.
        max_request_bytes = min(MAX_REQUEST_BYTES, primary_bytes + 512)
    if primary_bytes > max_request_bytes:
        raise Invalid("Complete primary contract and output schema exceed the request envelope; select a smaller scoped task rather than truncate the contract")
    seen = set(primary_ids)
    admitted = {field: [] for field in collections}

    def attempt(field, entry, singleton=False):
        ident = entry.get("id")
        if ident and ident in seen:
            omitted.append(_reference(entry, field, "already supplied as primary or ancillary content"))
            return
        trial = copy.deepcopy(context)
        _set_path(trial, field, entry if singleton else [*admitted[field], entry])
        trial["context_selection"].update(included_count=len(included) + 1, omitted_count=len(omitted))
        if derive_metadata:
            derive_metadata(trial)
        if proposal_request_bytes(contract, trial, provider, role) > max_request_bytes - 64:
            omitted.append(_reference(entry, field, "request byte budget"))
            return
        if singleton:
            _set_path(context, field, entry)
        else:
            admitted[field].append(entry)
            _set_path(context, field, admitted[field])
        included.append(_reference(entry, field))
        if ident:
            seen.add(ident)

    for field, entry in optional_records.items():
        if entry is not None:
            attempt(field, entry, singleton=True)
    for field, entries in collections.items():
        for entry in entries:
            attempt(field, entry)
    context["context_selection"].update(included_count=len(included), omitted_count=len(omitted))
    if derive_metadata:
        derive_metadata(context)
    size = proposal_request_bytes(contract, context, provider, role)
    manifest_id = store.put("role_context_manifest", {
        "contract": contract.__name__, "role": role, "primary_artifact_ids": list(primary_ids),
        "included": included, "omitted": omitted, "request_bytes": size,
        "max_request_bytes": max_request_bytes,
        "primary_request_bytes": primary_bytes,
        "expanded_for_primary": max_request_bytes > DEFAULT_ROLE_REQUEST_BYTES,
        "scope": "Exact complete primary inputs with bounded ancillary admission; omitted artifacts were not presented to this role.",
    }, problem_id)
    context["context_selection"]["manifest_id"] = manifest_id
    if proposal_request_bytes(contract, context, provider, role) > max_request_bytes:
        raise Invalid("Final proposal context exceeds its bounded request envelope")
    return context
