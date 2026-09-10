import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from symplex.infrastructure.storage import Budget, Store


def main():
    load_dotenv(Path.cwd() / ".env")
    parser = argparse.ArgumentParser(description="Symplex")
    parser.add_argument("--workspace", default=".symplex")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=8765)
    run = sub.add_parser("run")
    run.add_argument("--data")
    run.add_argument("--synthetic", action="store_true")
    run.add_argument("--live", action="store_true")
    sub.add_parser("doctor")
    sub.add_parser("research-pilot")
    sub.add_parser("acquire-bamtwoogle")
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn

        from symplex.interfaces.api import create_app

        uvicorn.run(create_app(args.workspace), host="127.0.0.1", port=args.port)
        return
    store = Store(args.workspace)
    budget = Budget(store, {"usd": float(os.getenv("SYMPLEX_MAX_USD", "5"))})
    from symplex.infrastructure.providers import OpenAIProvider, RuleProvider

    if args.command == "doctor":
        output = {
            "key_configured": bool(os.getenv("OPENAI_API_KEY")),
            "budget": budget.snapshot(),
        }
        if output["key_configured"]:
            output["models"] = OpenAIProvider(store, budget).capability()
    elif args.command == "acquire-bamtwoogle":
        from symplex.evidence.acquisition import download_bam

        result = download_bam(store, budget)
        output = {
            "id": result["id"],
            "name": result["data"]["name"],
            "rows": result["data"]["rows"],
            "status": result["data"]["status"],
        }
    elif args.command == "research-pilot":
        from symplex.evaluation.research import benchmark

        output = benchmark(store, budget, OpenAIProvider(store, budget))
    else:
        from symplex.agents.evolution import Engine
        from symplex.evaluation.fixtures import synthetic_pack

        if bool(args.data) == args.synthetic:
            parser.error("Choose exactly one of --data PATH or --synthetic")
        data = (
            synthetic_pack()
            if args.synthetic
            else json.loads(Path(args.data).read_text())
        )
        output = Engine(
            store,
            budget,
            OpenAIProvider(store, budget) if args.live else RuleProvider(),
        ).investigate(data)
    print(json.dumps(output, indent=2))
    if output.get("status") in ("failed", "budget-exhausted", "cancelled"):
        raise SystemExit(1)
