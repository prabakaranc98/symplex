"""Reviewable opt-in live campaign. Dry-run is default; never bypass an existing ledger."""
import argparse
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from symplex.core.contracts import digest
from symplex.agents.solver import solve
from symplex.infrastructure.storage import Budget, BudgetExhausted, Store
from symplex.infrastructure.providers import OpenAIProvider
from symplex.infrastructure.scoped_budget import ScopedBudget
from symplex.infrastructure.execution_scope import execution_lock



class CaseBudget:
    """A non-resetting allocation for this sequential campaign's single case turn."""
    def __init__(self, global_budget, usd, consumed=None, campaign_usd=None):
        self.parent = global_budget
        self.store = global_budget.store
        self.start = global_budget.snapshot()
        self.consumed = consumed or {}
        self.campaign_usd = campaign_usd
        scale = usd / 5
        self.limits = {"usd": usd, "model_requests": 40 * scale, "input_tokens": 300000 * scale,
                       "output_tokens": 45000 * scale, "evidence_requests": 16 * scale, "worker_seconds": 225 * scale}

    def snapshot(self):
        result = self.parent.snapshot()
        if self.campaign_usd is not None:
            result["usd"]["cap"] = min(result["usd"]["cap"], self.campaign_usd)
        for name, limit in self.limits.items():
            result[name]["cap"] = min(result[name]["cap"], self.start[name]["used_or_reserved"] + limit - self.consumed.get(name, 0))
        return result

    def reserve(self, **amounts):
        state = self.snapshot()
        for name, row in state.items():
            if row["used_or_reserved"] > row["cap"] + 1e-9:
                raise BudgetExhausted("Recorded case usage exceeded allocation: " + name)
        for name, amount in amounts.items():
            if state[name]["used_or_reserved"] + amount > state[name]["cap"] + 1e-9:
                raise BudgetExhausted("Case allocation exhausted: " + name)
        return self.parent.reserve(**amounts)

    def settle(self, ident, **actual):
        return self.parent.settle(ident, **actual)


def usage_delta(before, after):
    """Reservations count as spend until the global ledger actually settles them."""
    return {
        name: max(0, row["used_or_reserved"] - before.get(name, {}).get("used_or_reserved", 0))
        for name, row in after.items()
    }


def recover_attempts(store, budget, campaign_id):
    """Recover a legacy global-span attempt before scoped execution begins."""
    ended = {r["parent"] for r in store.list("campaign_attempt_end")}
    pending = [r for r in store.list("campaign_attempt")
               if r["parent"] == campaign_id and r["id"] not in ended and not r["data"].get("scope_id")]
    scoped = [r for r in store.list("campaign_attempt")
              if r["parent"] == campaign_id and r["data"].get("scope_id")]
    if pending and scoped:
        raise RuntimeError("Legacy global-span recovery is unsafe after scoped execution begins")
    if len(pending) > 1:
        raise RuntimeError("Multiple unfinished campaign attempts require accounting recovery before execution")
    for attempt in pending:
        after = budget.snapshot()
        store.put("campaign_attempt_end", {
            "case_id": attempt["data"]["case_id"],
            "status": "interrupted_recovered",
            "budget_end": after,
            "usage_delta": usage_delta(attempt["data"]["budget_start"], after),
        }, attempt["id"])


def recover_scoped_attempts(store, allocation, campaign_id, case_id):
    """Recover only this case while holding its exclusive execution lock."""
    ended = {r["parent"] for r in store.list("campaign_attempt_end")}
    pending = [r for r in store.list("campaign_attempt")
               if r["parent"] == campaign_id and r["data"]["case_id"] == case_id
               and r["data"].get("scope_id") == allocation.scope_id and r["id"] not in ended]
    if len(pending) > 1:
        raise RuntimeError("Multiple unfinished attempts for one scope require accounting recovery")
    for attempt in pending:
        after = allocation.scope_snapshot()
        store.put("campaign_attempt_end", {
            "case_id": case_id, "scope_id": allocation.scope_id,
            "status": "interrupted_recovered", "scope_end": after,
            "budget_end": allocation.parent.snapshot(),
            "usage_delta": usage_delta(attempt["data"]["scope_start"], after),
        }, attempt["id"])


def case_scope_id(campaign_id, case_id):
    return campaign_id + ":" + case_id


def case_usage(store, campaign_id, case_id):
    """Use the authoritative scope ledger, or reconstruct pre-scope usage once."""
    with store.db() as db:
        rows = db.execute("SELECT name,used FROM scope_budget WHERE scope_id=?", (case_scope_id(campaign_id, case_id),)).fetchall()
    if rows:
        return {row["name"]: row["used"] for row in rows}
    consumed = {}

    def add(delta):
        for name, amount in delta.items():
            consumed[name] = consumed.get(name, 0) + max(0, amount)

    previous = {}
    for report in store.list("campaign_result"):
        if report["parent"] != campaign_id:
            continue
        data = report["data"]
        if not data.get("attempt_id") and data["case_id"] == case_id:
            add(data.get("usage_delta") if "usage_delta" in data else usage_delta(previous, data["budget"]))
        previous = data["budget"]
    endings = {r["parent"]: r for r in store.list("campaign_attempt_end")}
    for attempt in store.list("campaign_attempt"):
        if attempt["parent"] != campaign_id or attempt["data"]["case_id"] != case_id:
            continue
        if attempt["data"].get("scope_id"):
            raise RuntimeError("Scoped attempt is missing its persistent budget scope")
        ending = endings.get(attempt["id"])
        if ending is None:
            raise RuntimeError("Recover unfinished campaign attempts before allocating a case")
        add(usage_delta(attempt["data"]["budget_start"], ending["data"]["budget_end"]))
    return consumed


def missing_execution_milestones(report, plan):
    required = set(plan["milestones"]) & {"compute_package", "experiment_comparison", "model_critique"}
    return sorted(kind for kind in required if not report.get("milestones", {}).get(kind))


def should_run(completed, plan, *, resume=False, retry_incomplete=False):
    if not completed:
        return True
    latest = completed[-1]["data"]
    if latest["job"]["status"] != "succeeded":
        return resume or retry_incomplete
    return retry_incomplete and bool(missing_execution_milestones(latest, plan))


@contextmanager
def campaign_lock(store, name, *, shared=False, blocking=False):
    with execution_lock(store, name, shared=shared, blocking=blocking):
        yield


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Makes paid calls; run only within explicitly authorized campaign scope")
    parser.add_argument("--workspace", default=".symplex-campaign")
    parser.add_argument("--plan", default="evaluations/real_world_flow.json")
    parser.add_argument("--resume", action="store_true", help="Retry unfinished cases within their original cumulative allocation")
    parser.add_argument("--case", dest="case_id", help="Run only this case ID from the campaign plan")
    parser.add_argument("--retry-incomplete", action="store_true", help="Also retry succeeded cases missing compute or evaluation milestones, within their original allocation")
    parser.add_argument("--case-budget-usd", type=float, help="Explicitly authorized cumulative per-case allocation; preserves prior case spend")
    parser.add_argument("--campaign-budget-usd", type=float, help="Must match the separately granted persistent global ledger ceiling")
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    for amount in (args.case_budget_usd, args.campaign_budget_usd):
        if amount is not None and (not math.isfinite(amount) or amount <= 0):
            parser.error("Budget overrides must be finite positive USD amounts")
    if args.case_id and args.case_id not in {case["id"] for case in plan["cases"]}:
        parser.error("--case must name a case ID in the plan")
    if not args.execute:
        print(json.dumps({"dry_run": True, "plan_digest": digest(plan), "plan": plan,
                          "selected_case": args.case_id, "retry_incomplete": args.retry_incomplete,
                          "case_budget_usd": args.case_budget_usd,
                          "campaign_budget_usd": args.campaign_budget_usd}, indent=2))
        return
    from dotenv import load_dotenv
    load_dotenv('.env')
    store = Store(args.workspace)
    try:
        run_campaign(store, plan, args)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))


def run_campaign(store, plan, args):
    # New scoped runners share this lock. Old sequential runners hold it
    # exclusively, so migration cannot overlap their global-span accounting.
    with campaign_lock(store, "campaign.lock", shared=True):
        return _run_campaign(store, plan, args)


def _run_campaign(store, plan, args):
    budget = Budget(store, {"usd": plan["proposed_additional_budget_usd"],
                           "model_requests": 160, "input_tokens": 1200000,
                           "output_tokens": 180000, "evidence_requests": 64})
    ledger_cap = budget.snapshot()["usd"]["cap"]
    campaign_usd = args.campaign_budget_usd if args.campaign_budget_usd is not None else ledger_cap
    if abs(campaign_usd - ledger_cap) > 1e-9:
        raise ValueError("Parallel campaigns require the persistent ledger ceiling; record the authorized cap separately first")
    case_usd = args.case_budget_usd if args.case_budget_usd is not None else plan["case_budget_usd"]
    # Reopening this workspace preserves its original cap and all usage.
    key = digest(plan)
    # Serialize campaign identity, legacy recovery and first-time scope seeding;
    # release before any paid work so different case locks can proceed together.
    with campaign_lock(store, "campaign.init.lock", blocking=True):
        prior = [r for r in store.list("campaign_plan") if r["data"].get("plan_digest") == key]
        if prior:
            campaign = prior[-1]
        else:
            ident = store.put("campaign_plan", {"plan": plan, "plan_digest": key})
            campaign = store.get(ident)
        recover_attempts(store, budget, campaign["id"])
        for case in plan["cases"]:
            allocation = ScopedBudget(budget, case_scope_id(campaign["id"], case["id"]),
                                      CaseBudget(budget, case_usd).limits,
                                      initial_usage=case_usage(store, campaign["id"], case["id"]))
            if abs(allocation.scope_snapshot()["usd"]["cap"] - case_usd) > 1e-9:
                raise ValueError("Case override must match its persistent scope cap; record any authorized cap change separately")
    for case in plan["cases"]:
        if args.case_id and case["id"] != args.case_id:
            continue
        scope_id = case_scope_id(campaign["id"], case["id"])
        with campaign_lock(store, "campaign-case-" + digest(scope_id) + ".lock"):
            run_case(store, budget, campaign, case, plan, args, case_usd, campaign_usd)
        if budget.snapshot()["usd"]["used_or_reserved"] >= campaign_usd:
            break


def run_case(store, budget, campaign, case, plan, args, case_usd, campaign_usd):
    scope_id = case_scope_id(campaign["id"], case["id"])
    allocation = ScopedBudget(budget, scope_id, CaseBudget(budget, case_usd).limits)
    recover_scoped_attempts(store, allocation, campaign["id"], case["id"])
    completed = [r for r in store.list("campaign_result") if r["parent"] == campaign["id"] and r["data"]["case_id"] == case["id"]]
    if not should_run(completed, plan, resume=args.resume, retry_incomplete=args.retry_incomplete):
        print(json.dumps({"case_id":case["id"], "status":"already_recorded", "result_id":completed[-1]["id"]}), flush=True)
        return
    existing = [r for r in store.list("workspace_problem") if r["data"].get("campaign_id") == campaign["id"] and r["data"].get("case_id") == case["id"]]
    if existing:
        problem = existing[-1]["id"]
    else:
        problem = store.put("workspace_problem", {"question":case["problem"],"dataset":"general","status":"draft", "campaign_id":campaign["id"],"case_id":case["id"]})
        store.put("context", {"title":"Investigation evaluation charter", "format":"text", "basis":"assumption", "content":plan["shared_context"], "inspection":{}}, problem)
    if completed and completed[-1]["data"]["job"]["status"] == "succeeded":
        missing = missing_execution_milestones(completed[-1]["data"], plan)
        store.put("context", {
            "title": "Bounded campaign continuation",
            "format": "text", "basis": "assumption", "inspection": {},
            "prior_campaign_result": completed[-1]["id"],
            "content": "Continue this investigation within its remaining original case allocation. "
                       "The prior delivery omitted these execution/evaluation milestones: "
                       + ", ".join(missing) + ". Produce the missing reusable outputs when feasible; "
                       "retain real-world validation gaps and report precise blockers honestly.",
        }, problem)
    print(json.dumps({"case_id":case["id"],"problem_id":problem,"status":"starting"}), flush=True)
    consumed = case_usage(store, campaign["id"], case["id"])
    before = budget.snapshot()
    scope_before = allocation.scope_snapshot()
    attempt = store.put("campaign_attempt", {
        "case_id": case["id"], "problem_id": problem,
        "scope_id": scope_id, "scope_start": scope_before,
        "budget_start": before, "prior_case_usage": consumed,
        "case_budget_usd": case_usd, "campaign_budget_usd": campaign_usd,
    }, campaign["id"])
    try:
        result = solve(store,allocation,OpenAIProvider(store,allocation),problem,depth=plan["depth"])
    except BaseException as error:
        after = budget.snapshot()
        scope_after = allocation.scope_snapshot()
        store.put("campaign_attempt_end", {
            "case_id": case["id"], "status": "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
            "error_type": type(error).__name__, "budget_end": after,
            "scope_id": scope_id, "scope_end": scope_after,
            "usage_delta": usage_delta(scope_before, scope_after),
        }, attempt)
        raise
    after = budget.snapshot()
    scope_after = allocation.scope_snapshot()
    store.put("campaign_attempt_end", {
        "case_id": case["id"], "status": result["status"],
        "job": result, "budget_end": after, "usage_delta": usage_delta(scope_before, scope_after),
        "scope_id": scope_id, "scope_end": scope_after,
    }, attempt)
    records = [r for r in store.list() if r["parent"] == problem and not r["stale"]]
    report = {"case_id":case["id"],"problem_id":problem,"job":result,"attempt_id":attempt,
              "milestones":{kind:[r["id"] for r in records if r["kind"]==kind] for kind in plan["milestones"]},
              "scientific_validation":"Not established by this flow test", "budget":after,
              "scope_id":scope_id, "scope_budget":scope_after,
              "usage_delta":usage_delta(scope_before, scope_after),
              "prior_case_usage":consumed}
    store.put("campaign_result", report, campaign["id"])
    print(json.dumps(report), flush=True)

if __name__ == "__main__":
    main()
