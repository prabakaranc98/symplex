import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from symplex.core.contracts import Invalid, canonical, digest
from symplex.modeling.templates import validate


class Cancelled(RuntimeError):
    pass


def execute_system(store, budget, spec, cancel=None):
    from symplex.modeling.dynamics import SystemSimulation

    SystemSimulation.model_validate(spec)
    return _process(budget, {"task": "system_simulation", "spec": spec}, cancel, 6)


def execute(store, budget, program, training, inputs, cancel=None, timeout=6):
    validate(program)
    if any("label" in row or "resolved_at" in row or "split" in row for row in inputs):
        raise Invalid("Protected outcomes cannot enter prediction inputs")
    if cancel and cancel.is_set():
        raise Cancelled("Cancelled before execution")
    payload = {"program": program, "training": training, "inputs": inputs}
    return _process(budget, payload, cancel, timeout)


def _process(budget, payload, cancel, timeout):
    if cancel and cancel.is_set():
        raise Cancelled("Cancelled before execution")
    reservation = budget.reserve(worker_seconds=timeout)
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="symplex-worker-") as directory:
            process = subprocess.Popen(
                [sys.executable, "-I", str(Path(__file__).with_name("worker.py"))],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=directory,
                env={"PATH": os.defpath, "LANG": "C.UTF-8"},
                start_new_session=True,
            )
            body = canonical(payload).encode()
            try:
                while True:
                    try:
                        remaining = timeout - (time.monotonic() - started)
                        if remaining <= 0:
                            raise TimeoutError("Numerical worker timed out")
                        stdout, stderr = process.communicate(
                            body, timeout=min(0.1, remaining)
                        )
                        break
                    except subprocess.TimeoutExpired:
                        body = None
                        if cancel and cancel.is_set():
                            raise Cancelled("Cancelled during worker execution")
                        if time.monotonic() - started >= timeout:
                            raise TimeoutError("Numerical worker timed out")
                if process.returncode:
                    raise Invalid(
                        "Worker failed: " + stderr.decode(errors="replace")[-2000:]
                    )
                result = json.loads(stdout)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
        result["runtime_seconds"] = time.monotonic() - started
        result["execution"] = "trusted_template_subprocess"
        result["memory_limit"] = "256 MiB on Linux; not enforced on macOS"
        result["input_digest"] = digest(payload)
        return result
    finally:
        elapsed = min(timeout, time.monotonic() - started)
        budget.settle(reservation, worker_seconds=elapsed)
