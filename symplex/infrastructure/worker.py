"""Private subprocess entrypoint. Executes bundled templates only, never candidate code."""

import json
import sys

if __name__ == "__main__":
    # -I removes caller-controlled PYTHONPATH; add only the bundled trusted package root.
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import resource

    from symplex.modeling.templates import run

    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_FSIZE, (4 * 1024 * 1024, 4 * 1024 * 1024))
    # macOS does not enforce RLIMIT_AS consistently. Linux workers additionally cap address space.
    if sys.platform.startswith("linux"):
        resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
    payload = json.load(sys.stdin)
    if payload.get("task") == "system_simulation":
        from symplex.modeling.dynamics import simulate

        result = simulate(payload["spec"])
    else:
        result = run(payload["program"], payload["training"], payload["inputs"])
    json.dump(result, sys.stdout, allow_nan=False)
