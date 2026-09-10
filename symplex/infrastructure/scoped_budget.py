"""Persistent per-investigation allocations sharing one atomic global ledger."""

import time

from symplex.core.contracts import Invalid, canonical, number
from symplex.infrastructure.storage import (
    DEFAULT_CAPS,
    Budget,
    _reserve_budget,
    _settle_budget,
)


class ScopedBudget:
    """Provider-compatible budget for independently concurrent investigations.

    ``initial_usage`` attributes already-recorded global usage to a new scope;
    it never charges the global ledger again. It is applied only on first scope
    creation. Reopening a scope preserves all caps, usage, and reservations,
    regardless of the new constructor arguments. Scope IDs must be stable across
    retries and unique across investigations that have separate allocations.
    """

    def __init__(self, global_budget, scope_id, caps=None, initial_usage=None):
        if not isinstance(global_budget, Budget):
            raise Invalid("A scope requires the underlying global Budget")
        if not isinstance(scope_id, str) or not scope_id.strip():
            raise Invalid("A stable nonempty budget scope ID is required")
        limits = dict(DEFAULT_CAPS, **(caps or {}))
        seed = dict(initial_usage or {})
        if set(limits) != set(DEFAULT_CAPS) or not set(seed).issubset(DEFAULT_CAPS):
            raise Invalid("Unknown resource cap or initial usage")
        for value in (*limits.values(), *seed.values()):
            number(value, 0)
        self.parent = global_budget
        self.store = global_budget.store
        self.scope_id = scope_id
        with self.store.db() as db:
            db.execute("BEGIN IMMEDIATE")
            exists = db.execute("SELECT scope_id FROM budget_scopes WHERE scope_id=?", (scope_id,)).fetchone()
            if exists:
                return
            db.execute(
                "INSERT INTO budget_scopes VALUES (?,?,?)",
                (scope_id, canonical(seed), time.time()),
            )
            for name, cap in limits.items():
                db.execute(
                    "INSERT INTO scope_budget VALUES (?,?,?,?)",
                    (scope_id, name, cap, seed.get(name, 0)),
                )

    def reserve(self, **amounts):
        with self.store.db() as db:
            db.execute("BEGIN IMMEDIATE")
            ident = _reserve_budget(db, amounts, self.scope_id)
        return ident

    def settle(self, ident, **actual):
        with self.store.db() as db:
            db.execute("BEGIN IMMEDIATE")
            settlement = _settle_budget(db, ident, actual, self.scope_id)
        return settlement

    def snapshot(self):
        """Show own usage with capacity limited by both immutable ledgers.

        The read is a coherent single SQL statement. Admission remains atomic in
        reserve; another scope may consume globally shared capacity after a read.
        """
        with self.store.db() as db:
            return {
                row["name"]: {
                    "cap": min(row["scope_cap"], row["scope_used"] + row["global_cap"] - row["global_used"]),
                    "used_or_reserved": row["scope_used"],
                }
                for row in db.execute(
                    """SELECT s.name, s.cap AS scope_cap, s.used AS scope_used,
                              g.cap AS global_cap, g.used AS global_used
                       FROM scope_budget s JOIN budget g ON s.name=g.name
                       WHERE s.scope_id=?""", (self.scope_id,)
                )
            }

    def scope_snapshot(self):
        """Return the raw persisted allocation and usage for audit reports."""
        with self.store.db() as db:
            return {
                row["name"]: {"cap": row["cap"], "used_or_reserved": row["used"]}
                for row in db.execute("SELECT name,cap,used FROM scope_budget WHERE scope_id=?", (self.scope_id,))
            }
