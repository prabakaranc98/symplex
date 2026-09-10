"""Content-addressed artifacts plus transactional metadata and global reservations."""

import json
import sqlite3
import time
import uuid
from pathlib import Path

from symplex.core.contracts import Invalid, canonical, digest, number


class BudgetExhausted(RuntimeError):
    pass


DEFAULT_CAPS = {
    "model_requests": 64,
    "input_tokens": 250000,
    "output_tokens": 50000,
    "worker_seconds": 900,
    "evidence_requests": 20,
    "usd": 0,
}


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "artifacts").mkdir(exist_ok=True)
        with self.db() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS records (
              id TEXT PRIMARY KEY, kind TEXT NOT NULL, parent TEXT, digest TEXT NOT NULL,
              created REAL NOT NULL, stale INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY, idem TEXT UNIQUE NOT NULL, status TEXT NOT NULL,
              result_id TEXT, error TEXT, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS budget (name TEXT PRIMARY KEY, cap REAL NOT NULL, used REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS reservations (id TEXT PRIMARY KEY, data TEXT NOT NULL, settled INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS budget_settlements (
              reservation_id TEXT PRIMARY KEY, data TEXT NOT NULL, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS budget_scopes (
              scope_id TEXT PRIMARY KEY, initial_usage TEXT NOT NULL, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS scope_budget (
              scope_id TEXT NOT NULL, name TEXT NOT NULL, cap REAL NOT NULL, used REAL NOT NULL,
              PRIMARY KEY(scope_id,name));
            CREATE TABLE IF NOT EXISTS reservation_scopes (
              reservation_id TEXT PRIMARY KEY, scope_id TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS seals (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)

    def db(self):
        db = sqlite3.connect(str(self.root / "metadata.sqlite3"), timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def put(self, kind, data, parent=None):
        sha = digest(data)
        path = self.root / "artifacts" / (sha + ".json")
        try:
            with path.open("x") as f:
                f.write(canonical(data))
        except FileExistsError:
            if path.read_text() != canonical(data):
                raise Invalid("Artifact integrity failure")
        ident = kind + "_" + uuid.uuid4().hex[:16]
        with self.db() as db:
            db.execute(
                "INSERT INTO records(id,kind,parent,digest,created) VALUES (?,?,?,?,?)",
                (ident, kind, parent, sha, time.time()),
            )
        return ident

    def get(self, ident):
        with self.db() as db:
            row = db.execute("SELECT * FROM records WHERE id=?", (ident,)).fetchone()
        if row is None:
            raise KeyError(ident)
        value = json.loads(
            (self.root / "artifacts" / (row["digest"] + ".json")).read_text()
        )
        if digest(value) != row["digest"]:
            raise Invalid("Artifact integrity failure")
        return dict(row, data=value)

    def list(self, kind=None):
        with self.db() as db:
            rows = db.execute(
                "SELECT id FROM records "
                + ("WHERE kind=? " if kind else "")
                + "ORDER BY created",
                (kind,) if kind else (),
            ).fetchall()
        return [self.get(r["id"]) for r in rows]

    def invalidate(self, ident):
        with self.db() as db:
            db.execute(
                """WITH RECURSIVE descendants(id) AS (
              SELECT id FROM records WHERE id=? UNION SELECT r.id FROM records r JOIN descendants d ON r.parent=d.id
              ) UPDATE records SET stale=1 WHERE id IN (SELECT id FROM descendants)""",
                (ident,),
            )

    def seal(self, key, value):
        with self.db() as db:
            try:
                db.execute("INSERT INTO seals VALUES (?,?)", (key, canonical(value)))
            except sqlite3.IntegrityError:
                raise Invalid(
                    "Protected evaluation already consumed or selection already frozen: "
                    + key
                )

    def start_job(self, idem):
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT * FROM jobs WHERE idem=?", (idem,)).fetchone()
            if old:
                return dict(old), False
            ident = "job_" + uuid.uuid4().hex[:16]
            db.execute(
                "INSERT INTO jobs VALUES (?,?,?,NULL,NULL,?)",
                (ident, idem, "proposed", time.time()),
            )
        return {"id": ident, "status": "proposed"}, True

    def job(self, ident, status, result_id=None, error=None):
        if status not in (
            "proposed",
            "validated",
            "queued",
            "waiting_user",
            "running",
            "succeeded",
            "failed",
            "cancelled",
            "budget-exhausted",
        ):
            raise Invalid("Unknown job state")
        with self.db() as db:
            db.execute(
                "UPDATE jobs SET status=?,result_id=?,error=? WHERE id=?",
                (status, result_id, error, ident),
            )

    def jobs(self):
        with self.db() as db:
            return [
                dict(x) for x in db.execute("SELECT * FROM jobs ORDER BY created DESC")
            ]


class Budget:
    def __init__(self, store, caps=None):
        self.store = store
        limits = dict(DEFAULT_CAPS, **(caps or {}))
        if set(limits) != set(DEFAULT_CAPS):
            raise Invalid("Unknown resource cap")
        with store.db() as db:
            for name, cap in limits.items():
                number(cap, 0)
                db.execute("INSERT OR IGNORE INTO budget VALUES (?,?,0)", (name, cap))
        # Reopening a workspace never resets or increases its ceiling.

    def reserve(self, **amounts):
        with self.store.db() as db:
            db.execute("BEGIN IMMEDIATE")
            ident = _reserve_budget(db, amounts)
        return ident

    def settle(self, ident, **actual):
        """Record observed usage even when an estimate was too small.

        A reservation is admission control, not authority to discard billed usage.
        Caps never increase: an over-cap settlement blocks every future reservation.
        Resources without reported actuals retain their original reservation.
        """
        with self.store.db() as db:
            db.execute("BEGIN IMMEDIATE")
            settlement = _settle_budget(db, ident, actual)
        return settlement

    def snapshot(self):
        with self.store.db() as db:
            return {
                r["name"]: {"cap": r["cap"], "used_or_reserved": r["used"]}
                for r in db.execute("SELECT * FROM budget")
            }


def _reserve_budget(db, amounts, scope_id=None):
    """Reserve both ledgers inside the caller's BEGIN IMMEDIATE transaction."""
    exceeded = db.execute(
        "SELECT name FROM budget WHERE used > cap + 1e-9 ORDER BY name"
    ).fetchall()
    if exceeded:
        raise BudgetExhausted(
            "Recorded usage exceeded immutable global caps: "
            + ", ".join(row["name"] for row in exceeded)
        )
    scope = {}
    if scope_id is not None:
        scope = {
            row["name"]: row for row in db.execute(
                "SELECT name,cap,used FROM scope_budget WHERE scope_id=?", (scope_id,)
            )
        }
        if set(scope) != set(DEFAULT_CAPS):
            raise Invalid("Unknown or incomplete budget scope")
        exceeded_scope = [name for name, row in scope.items() if row["used"] > row["cap"] + 1e-9]
        if exceeded_scope:
            raise BudgetExhausted(
                "Recorded usage exceeded immutable scope caps: " + ", ".join(sorted(exceeded_scope))
            )
    for name, amount in amounts.items():
        number(amount, 0)
        row = db.execute("SELECT cap,used FROM budget WHERE name=?", (name,)).fetchone()
        if row is None or row["used"] + amount > row["cap"] + 1e-9:
            raise BudgetExhausted("Global budget exhausted: " + name)
        if scope_id is not None and scope[name]["used"] + amount > scope[name]["cap"] + 1e-9:
            raise BudgetExhausted("Scope allocation exhausted: " + name)
    ident = uuid.uuid4().hex
    for name, amount in amounts.items():
        db.execute("UPDATE budget SET used=used+? WHERE name=?", (amount, name))
        if scope_id is not None:
            db.execute(
                "UPDATE scope_budget SET used=used+? WHERE scope_id=? AND name=?",
                (amount, scope_id, name),
            )
    db.execute("INSERT INTO reservations(id,data) VALUES (?,?)", (ident, canonical(amounts)))
    if scope_id is not None:
        db.execute("INSERT INTO reservation_scopes VALUES (?,?)", (ident, scope_id))
    return ident


def _settle_budget(db, ident, actual, expected_scope=None):
    """Settle actuals once in both ledgers, even when called via global Budget."""
    row = db.execute("SELECT * FROM reservations WHERE id=?", (ident,)).fetchone()
    if not row or row["settled"]:
        raise Invalid("Unknown or settled reservation")
    mapping = db.execute(
        "SELECT scope_id FROM reservation_scopes WHERE reservation_id=?", (ident,)
    ).fetchone()
    scope_id = mapping["scope_id"] if mapping else None
    if expected_scope is not None and scope_id != expected_scope:
        raise Invalid("Reservation does not belong to this budget scope")
    reserved = json.loads(row["data"])
    if not set(actual).issubset(reserved):
        raise Invalid("Unreserved resource")
    for amount in actual.values():
        number(amount, 0)
    if scope_id is not None:
        names = {r["name"] for r in db.execute("SELECT name FROM scope_budget WHERE scope_id=?", (scope_id,))}
        if set(DEFAULT_CAPS) != names:
            raise Invalid("Unknown or incomplete budget scope")
    for name, amount in actual.items():
        db.execute("UPDATE budget SET used=used-?+? WHERE name=?", (reserved[name], amount, name))
        if scope_id is not None:
            db.execute(
                "UPDATE scope_budget SET used=used-?+? WHERE scope_id=? AND name=?",
                (reserved[name], amount, scope_id, name),
            )
    db.execute("UPDATE reservations SET settled=1 WHERE id=?", (ident,))
    exceeded = {
        row["name"]: {"cap": row["cap"], "used": row["used"]}
        for row in db.execute("SELECT name,cap,used FROM budget WHERE used > cap + 1e-9")
    }
    scope_exceeded = {}
    if scope_id is not None:
        scope_exceeded = {
            row["name"]: {"cap": row["cap"], "used": row["used"]}
            for row in db.execute(
                "SELECT name,cap,used FROM scope_budget WHERE scope_id=? AND used > cap + 1e-9", (scope_id,)
            )
        }
    settlement = {
        "reservation_id": ident,
        "reserved": reserved,
        "reported_actual": actual,
        "retained_reservations": {k: v for k, v in reserved.items() if k not in actual},
        "over_reservation": {k: v - reserved[k] for k, v in actual.items() if v > reserved[k] + 1e-9},
        "exceeded_caps": exceeded,
        "status": "settled_over_cap" if exceeded or scope_exceeded else "settled",
    }
    if scope_id is not None:
        settlement["scope_id"] = scope_id
        settlement["scope_exceeded_caps"] = scope_exceeded
    db.execute(
        "INSERT INTO budget_settlements VALUES (?,?,?)",
        (ident, canonical(settlement), time.time()),
    )
    return settlement
