"""Preserve problem allocations and exclusive execution across process entrypoints."""

import fcntl
import threading
from contextlib import contextmanager
from functools import wraps

from symplex.core.contracts import Invalid, digest
from symplex.infrastructure.scoped_budget import ScopedBudget

_HELD = threading.local()


class ExecutionBusy(RuntimeError):
    pass


@contextmanager
def execution_lock(store, name, *, shared=False, blocking=False, reentrant=False):
    path = str((store.root / name).resolve())
    held = getattr(_HELD, "paths", None)
    if held is None:
        held = _HELD.paths = {}
    if reentrant and path in held:
        if held[path] and not shared:
            raise ExecutionBusy("Cannot upgrade a shared execution lock")
        yield
        return
    with open(path, "a+") as handle:
        operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        if not blocking:
            operation |= fcntl.LOCK_NB
        try:
            fcntl.flock(handle, operation)
        except BlockingIOError:
            raise ExecutionBusy("Another runner owns this execution: " + name) from None
        held[path] = shared
        try:
            yield
        finally:
            held.pop(path, None)
            fcntl.flock(handle, fcntl.LOCK_UN)


def workspace_problem_id(store, ident):
    seen = set()
    while ident:
        if ident in seen:
            raise Invalid("Artifact ancestry contains a cycle")
        seen.add(ident)
        artifact = store.get(ident)
        if artifact["kind"] == "workspace_problem":
            if artifact["stale"]:
                raise Invalid("Select a current problem")
            return ident
        ident = artifact["parent"]
    return None


def problem_budget(store, budget, problem_id):
    """Reopen an existing allocation; never synthesize new campaign capacity."""
    if not problem_id:
        return budget
    scopes = {r["data"].get("scope_id") for r in store.list("problem_budget_scope")
              if r["parent"] == problem_id}
    scopes.update(r["data"].get("budget_scope_id") for r in store.list("run_manifest")
                  if r["parent"] == problem_id)
    scopes.update(r["data"].get("scope_id") for r in store.list("campaign_attempt")
                  if r["data"].get("problem_id") == problem_id)
    if isinstance(budget, ScopedBudget):
        scopes.add(budget.scope_id)
    scopes.discard(None)
    if len(scopes) > 1:
        raise Invalid("Problem has conflicting budget allocations; execution is blocked")
    if not scopes:
        if store.get(problem_id)["data"].get("campaign_id"):
            raise Invalid("Campaign problem has no persisted budget scope; execution is blocked")
        return budget
    scope_id = next(iter(scopes))
    with store.db() as db:
        if not db.execute("SELECT 1 FROM budget_scopes WHERE scope_id=?", (scope_id,)).fetchone():
            raise Invalid("The problem's persisted budget scope is unavailable")
    base = budget.parent if isinstance(budget, ScopedBudget) else budget
    return budget if getattr(budget, "scope_id", None) == scope_id else ScopedBudget(base, scope_id)


@contextmanager
def problem_execution(store, budget, problem_id):
    if not problem_id:
        yield budget
        return
    with execution_lock(store, "problem-" + digest(problem_id) + ".lock", reentrant=True):
        allocation = problem_budget(store, budget, problem_id)
        scope_id = getattr(allocation, "scope_id", None)
        # This is also the campaign runner's case lock, keyed by its persisted scope.
        name = "campaign-case-" + digest(scope_id) + ".lock" if scope_id else "problem-" + digest(problem_id) + ".lock"
        with execution_lock(store, name, reentrant=True):
            if scope_id and not any(r["parent"] == problem_id for r in store.list("problem_budget_scope")):
                store.put("problem_budget_scope", {"scope_id": scope_id}, problem_id)
            yield allocation


def scoped_solver(function):
    @wraps(function)
    def invoke(store, budget, provider, problem_id, *args, **kwargs):
        with problem_execution(store, budget, problem_id) as allocation:
            prior_budget = getattr(provider, "budget", None)
            if prior_budget is not None:
                provider.budget = allocation
            try:
                return function(store, allocation, provider, problem_id, *args, **kwargs)
            finally:
                if prior_budget is not None:
                    provider.budget = prior_budget
    return invoke
