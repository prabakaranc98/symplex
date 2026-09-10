"""Atomic per-investigation usage shares a global cap across concurrent workers."""

import json
import sqlite3
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from symplex.core.contracts import Invalid
from symplex.infrastructure.scoped_budget import ScopedBudget
from symplex.infrastructure.storage import Budget, BudgetExhausted, Store


class ScopedBudgetTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = Store(directory.name)
        self.budget = Budget(self.store, {"usd": 20})

    def scope(self, name="case", usd=10, **kwargs):
        return ScopedBudget(self.budget, name, {"usd": usd}, **kwargs)

    def concurrent(self, callbacks):
        barrier = threading.Barrier(len(callbacks))

        def run(callback):
            barrier.wait(timeout=10)
            try:
                return callback()
            except BudgetExhausted:
                return None

        with ThreadPoolExecutor(max_workers=len(callbacks)) as executor:
            return list(executor.map(run, callbacks))

    def test_different_scopes_race_for_global_capacity_without_cross_charging(self):
        # Six independently valid scope reservations compete for four global slots.
        scopes = [self.scope("case_" + str(index)) for index in range(6)]
        results = self.concurrent([lambda scope=scope: scope.reserve(usd=5, model_requests=1) for scope in scopes])
        self.assertEqual(sum(result is not None for result in results), 4)
        self.assertEqual(self.budget.snapshot()["usd"]["used_or_reserved"], 20)
        for scope, reservation in zip(scopes, results):
            expected = 5 if reservation else 0
            self.assertEqual(scope.scope_snapshot()["usd"]["used_or_reserved"], expected)
        with self.store.db() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM reservations").fetchone()[0], 4)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM reservation_scopes").fetchone()[0], 4)

    def test_same_scope_race_cannot_exceed_own_cap(self):
        scope = self.scope(usd=3)
        results = self.concurrent([lambda: scope.reserve(usd=1) for _ in range(8)])
        self.assertEqual(sum(result is not None for result in results), 3)
        self.assertEqual(scope.scope_snapshot()["usd"]["used_or_reserved"], 3)
        self.assertEqual(self.budget.snapshot()["usd"]["used_or_reserved"], 3)

    def test_interrupted_reservations_and_legacy_seed_survive_reopen_without_reset(self):
        self.budget.reserve(usd=4, model_requests=2)
        scope = self.scope(initial_usage={"usd": 4, "model_requests": 2})
        reservation = scope.reserve(usd=3, model_requests=1)
        reopened_global = Budget(Store(self.store.root), {"usd": 1000})
        reopened = ScopedBudget(reopened_global, "case", {"usd": 1000}, initial_usage={"usd": 0})
        self.assertEqual(reopened.scope_snapshot()["usd"], {"cap": 10, "used_or_reserved": 7})
        self.assertEqual(reopened_global.snapshot()["usd"], {"cap": 20, "used_or_reserved": 7})
        with self.assertRaises(BudgetExhausted):
            reopened.reserve(usd=3.01)
        reopened.settle(reservation, usd=2)
        self.assertEqual(reopened.scope_snapshot()["usd"]["used_or_reserved"], 6)
        self.assertEqual(reopened.scope_snapshot()["model_requests"]["used_or_reserved"], 3)
        self.assertEqual(reopened_global.snapshot()["usd"]["used_or_reserved"], 6)

    def test_concurrent_creation_seeds_once_and_never_charges_global_twice(self):
        self.budget.reserve(usd=4)
        self.concurrent([lambda: self.scope(initial_usage={"usd": 4}) for _ in range(8)])
        self.assertEqual(self.scope().scope_snapshot()["usd"]["used_or_reserved"], 4)
        self.assertEqual(self.budget.snapshot()["usd"]["used_or_reserved"], 4)
        with self.store.db() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM budget_scopes").fetchone()[0], 1)

    def test_over_cap_actual_is_retained_and_blocks_only_affected_scope(self):
        scope = self.scope(usd=2)
        sibling = self.scope("sibling")
        reservation = scope.reserve(usd=1, model_requests=1)
        settlement = scope.settle(reservation, usd=3)
        self.assertEqual(settlement["status"], "settled_over_cap")
        self.assertEqual(settlement["exceeded_caps"], {})
        self.assertEqual(settlement["scope_exceeded_caps"]["usd"], {"cap": 2, "used": 3})
        self.assertEqual(scope.scope_snapshot()["usd"], {"cap": 2, "used_or_reserved": 3})
        self.assertEqual(self.budget.snapshot()["usd"]["used_or_reserved"], 3)
        with self.assertRaises(BudgetExhausted):
            scope.reserve(worker_seconds=1)
        sibling.reserve(usd=1)
        self.assertEqual(sibling.scope_snapshot()["usd"]["used_or_reserved"], 1)

    def test_global_overrun_blocks_other_scopes_and_unscoped_work(self):
        first = self.scope()
        other = self.scope("other")
        reservation = first.reserve(usd=1)
        settlement = first.settle(reservation, usd=21)
        self.assertEqual(settlement["exceeded_caps"]["usd"]["cap"], 20)
        for budget in (first, other, self.budget):
            with self.assertRaises(BudgetExhausted):
                budget.reserve(model_requests=1)

    def test_global_settlement_always_uses_persisted_scope_mapping(self):
        scope = self.scope()
        reservation = scope.reserve(usd=2, input_tokens=100)
        settlement = self.budget.settle(reservation, usd=1)
        self.assertEqual(settlement["scope_id"], "case")
        self.assertEqual(scope.scope_snapshot()["usd"]["used_or_reserved"], 1)
        self.assertEqual(scope.scope_snapshot()["input_tokens"]["used_or_reserved"], 100)
        self.assertEqual(self.budget.snapshot()["input_tokens"]["used_or_reserved"], 100)
        with self.store.db() as db:
            row = db.execute("SELECT data FROM budget_settlements WHERE reservation_id=?", (reservation,)).fetchone()
        self.assertEqual(json.loads(row["data"]), settlement)
        with self.assertRaises(Invalid):
            scope.settle(reservation, usd=0)

    def test_wrong_scope_or_unscoped_reservation_cannot_be_settled_through_scope(self):
        first = self.scope()
        other = self.scope("other")
        reservation = first.reserve(usd=2)
        unscoped = self.budget.reserve(usd=1)
        before = self.budget.snapshot()
        for ident in (reservation, unscoped):
            with self.assertRaises(Invalid):
                other.settle(ident, usd=0)
        self.assertEqual(self.budget.snapshot(), before)
        self.assertEqual(first.scope_snapshot()["usd"]["used_or_reserved"], 2)
        first.settle(reservation, usd=1)

    def test_provider_snapshot_limits_capacity_to_remaining_global_budget(self):
        first = self.scope()
        other = self.scope("other", usd=20)
        first.reserve(usd=3)
        other.reserve(usd=15)
        self.assertEqual(first.snapshot()["usd"], {"cap": 5, "used_or_reserved": 3})
        self.assertEqual(first.scope_snapshot()["usd"]["cap"], 10)
        first.reserve(usd=2)
        self.assertEqual(other.snapshot()["usd"], {"cap": 15, "used_or_reserved": 15})

    def test_multiresource_failure_and_invalid_actuals_leave_both_ledgers_unchanged(self):
        scope = self.scope(usd=1)
        before = self.budget.snapshot(), scope.scope_snapshot()
        with self.assertRaises(BudgetExhausted):
            scope.reserve(model_requests=1, usd=2)
        self.assertEqual((self.budget.snapshot(), scope.scope_snapshot()), before)
        reservation = scope.reserve(usd=1)
        before = self.budget.snapshot(), scope.scope_snapshot()
        for actual in ({"input_tokens": 1}, {"usd": float("nan")}, {"usd": -1}):
            with self.assertRaises(Invalid):
                scope.settle(reservation, **actual)
            self.assertEqual((self.budget.snapshot(), scope.scope_snapshot()), before)
        scope.settle(reservation, usd=0.5)

    def test_failed_mapping_insert_rolls_back_both_usage_updates_and_reservation(self):
        scope = self.scope()
        with self.store.db() as db:
            db.execute("CREATE TRIGGER reject_mapping BEFORE INSERT ON reservation_scopes BEGIN SELECT RAISE(ABORT, 'injected failure'); END")
        before = self.budget.snapshot(), scope.scope_snapshot()
        with self.assertRaises(sqlite3.IntegrityError):
            scope.reserve(usd=1)
        self.assertEqual((self.budget.snapshot(), scope.scope_snapshot()), before)
        with self.store.db() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM reservations").fetchone()[0], 0)

    def test_failed_audit_insert_rolls_back_both_settlements_and_remains_retryable(self):
        scope = self.scope()
        reservation = scope.reserve(usd=2)
        with self.store.db() as db:
            db.execute("CREATE TRIGGER reject_settlement BEFORE INSERT ON budget_settlements BEGIN SELECT RAISE(ABORT, 'injected failure'); END")
        before = self.budget.snapshot(), scope.scope_snapshot()
        with self.assertRaises(sqlite3.IntegrityError):
            scope.settle(reservation, usd=1)
        self.assertEqual((self.budget.snapshot(), scope.scope_snapshot()), before)
        with self.store.db() as db:
            self.assertEqual(db.execute("SELECT settled FROM reservations WHERE id=?", (reservation,)).fetchone()[0], 0)
            db.execute("DROP TRIGGER reject_settlement")
        scope.settle(reservation, usd=1)


if __name__ == "__main__":
    unittest.main()
