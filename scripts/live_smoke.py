"""Explicit live integration check. Reads local .env; all usage enters the workspace ledger."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv

from symplex.agents.solver import solve
from symplex.infrastructure.providers import OpenAIProvider
from symplex.infrastructure.storage import Budget, Store

if __name__ == "__main__":
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    store = Store(".symplex")
    budget = Budget(store, {"usd": 5})
    problem = store.put(
        "workspace_problem",
        {
            "question": "Integration demonstration: explore an airport-planning system with uncertain demand, road access, congestion and environmental pressure. Build a system hypothesis and compare conditional scenarios. No location recommendation or measured impact is possible without local evidence.",
            "dataset": "general",
            "status": "draft",
            "integration_test": True,
        },
    )
    store.put(
        "context",
        {
            "problem_id": problem,
            "title": "Integration-test scope",
            "content": "Use the provided context first. Do not search the web for this integration test. All parameters are assumptions. Design a small model, execute stochastic scenarios, review it, and deliver the result with empirical gaps. Do not run unrelated benchmarks or draft arbitrary code.",
            "format": "text",
            "basis": "assumption",
            "status": "supplied_unverified",
            "inspection": {},
        },
        problem,
    )
    print(json.dumps({"problem_id": problem, "status": "started"}), flush=True)
    result = solve(
        store, budget, OpenAIProvider(store, budget), problem, depth="focused"
    )
    print(json.dumps(result), flush=True)
    print(json.dumps({"budget": budget.snapshot()}), flush=True)
