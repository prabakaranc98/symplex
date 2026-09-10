"""Operator-only explicit budget amendment; never registered as an agent tool.

Dry-run by default. Preserves all used/reserved amounts and existing settlement
records. Authorization text and before/after caps are committed with the update.
"""
import argparse
import json
import sqlite3
import sys
import time
import uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from symplex.core.contracts import canonical, number
from symplex.infrastructure.storage import Store, Budget


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--caps-json', required=True)
    parser.add_argument('--reason-file', required=True)
    parser.add_argument('--combined-usd-ceiling', type=float, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    store = Store(args.workspace)
    before = Budget(store).snapshot()
    caps = json.loads(args.caps_json)
    if not isinstance(caps, dict) or not set(caps) <= set(before):
        raise ValueError('Only existing resource caps can be amended')
    for value in caps.values():
        number(value, 0)
    reason = Path(args.reason_file).read_text().strip()
    if not reason:
        raise ValueError('Explicit operator authorization must be recorded')
    others = 0
    for path in Path.cwd().glob('.symplex*/metadata.sqlite3'):
        if path.parent.resolve() == store.root:
            continue
        with sqlite3.connect(path) as db:
            others += db.execute("SELECT cap FROM budget WHERE name='usd'").fetchone()[0]
    target = caps.get('usd', before['usd']['cap'])
    if others + target > args.combined_usd_ceiling or args.combined_usd_ceiling >= 50:
        raise ValueError('Combined local Symplex ceiling must remain below $50')
    amendment = {'before':before, 'new_caps':caps, 'other_workspace_usd_caps':others,
                 'combined_usd_ceiling':args.combined_usd_ceiling, 'authorization':reason,
                 'scope':'Local Symplex API ledgers, not an account invoice', 'created':time.time()}
    if args.execute:
        with store.db() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('CREATE TABLE IF NOT EXISTS budget_authorizations (id TEXT PRIMARY KEY, data TEXT NOT NULL)')
            for name, cap in caps.items():
                current = db.execute('SELECT cap,used FROM budget WHERE name=?',(name,)).fetchone()
                if current['cap'] != before[name]['cap'] or cap < current['used']:
                    raise ValueError('Concurrent grant or target below recorded usage')
                db.execute('UPDATE budget SET cap=? WHERE name=?',(cap,name))
            db.execute('INSERT INTO budget_authorizations VALUES (?,?)',(uuid.uuid4().hex,canonical(amendment)))
        store.put('budget_authorization', amendment)
    print(json.dumps({'executed':args.execute, **amendment}, indent=2))

if __name__ == '__main__':
    main()
