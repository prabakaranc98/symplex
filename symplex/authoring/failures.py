"""Read the failure, do not just retry. Classification, typed guidance, and when to stop.

A loop that reruns the same program because it failed is not iterating; it is paying
twice for the same answer. This module turns raw sandbox output into a typed failure,
turns the typed failure into a bounded instruction that says what to do differently,
and -- the part that actually saves money -- says when the next attempt should not be a
code attempt at all.

Two judgements matter more than the taxonomy:

* `is_retryable` separates a defect in the code from a defect in the plan. A missing
  import is fixable in code. A program that ran cleanly and produced none of the files
  it promised is a disagreement between the plan and the implementation, and another
  code attempt will reproduce it.
* `escalate` names where a non-retryable failure goes back to: the plan, or the
  metareasoner that owns the envelope. Knowing when to stop coding and re-plan is the
  scarce skill; encoding it is the point of this file.

Nothing here executes anything, imports the agent's code, or calls a model. It reads
bounded text and returns dicts.
"""

import re
from typing import Literal

from pydantic import ConfigDict, Field

from symplex.core.contracts import Invalid
from symplex.modeling.complex_system import ClosedContract, Identifier, Text

MAX_STREAM_CHARS = 20000
MAX_EVIDENCE_CHARS = 1200
MAX_GUIDANCE_CHARS = 1200
MAX_ACTION_CHARS = 300
MAX_ACTIONS = 6
# The third occurrence of one failure kind leaves the code loop. Two identical failures
# can be a partial fix that did not go far enough; three cannot.
REPEAT_ESCALATION = 3

SCOPE = (
    "Classification of an observed sandbox failure from bounded text. It identifies "
    "what went wrong mechanically; it does not diagnose whether the model was a good "
    "idea, and a clean run establishes nothing about scientific validity."
)

FailureKind = Literal[
    "syntax_error",
    "import_unavailable",
    "numerical_instability",
    "convergence_failure",
    "assertion_failure",
    "contract_violation",
    "resource_exhausted",
    "timeout",
    "dimension_mismatch",
    "empty_output",
]

FAILURE_KINDS = (
    "syntax_error",
    "import_unavailable",
    "numerical_instability",
    "convergence_failure",
    "assertion_failure",
    "contract_violation",
    "resource_exhausted",
    "timeout",
    "dimension_mismatch",
    "empty_output",
)

# The simulation-engineer output contract in symplex/agents/prompts/simulation_engineer.md.
REQUIRED_OUTPUTS = ("symplex_model.py", "symplex_results.json", "symplex_results.csv")
REQUIRED_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")

# Failures a further code attempt can plausibly fix. Everything else goes back up.
RETRYABLE_KINDS = frozenset(
    {
        "syntax_error",
        "import_unavailable",
        "assertion_failure",
        "numerical_instability",
        "convergence_failure",
        "dimension_mismatch",
        "empty_output",
    }
)

ESCALATION_TARGET = {
    "contract_violation": "plan",
    "resource_exhausted": "metareasoner",
    "timeout": "metareasoner",
}

_FRAME = re.compile(r'File "([^"]{1,400})", line (\d{1,9})(?:, in (\S{1,200}))?')
_EXCEPTION = re.compile(
    r"^(?:[A-Za-z_][\w.]*\.)?([A-Z][A-Za-z0-9_]*(?:Error|Exception|Warning|Interrupt|Exit))\b:?(.*)$"
)
_TEST_TAG = re.compile(r"symplex_test\s*[:=]\s*([a-z][a-z0-9_]{0,63})")
_MODULE_NAME = re.compile(r"No module named ['\"]([^'\"]{1,120})['\"]")

_SHAPE_WORDS = (
    "could not be broadcast",
    "operands could not be broadcast",
    "not aligned",
    "shapes",
    "dimension mismatch",
    "dimensions must be equal",
    "size mismatch",
    "inhomogeneous shape",
    "matmul",
    "axis",
)
_UNSTABLE_WORDS = (
    "nan",
    "inf",
    "overflow",
    "underflow",
    "divide by zero",
    "division by zero",
    "singular matrix",
    "not positive definite",
    "ill-conditioned",
    "stiff",
    "step size",
)
_CONVERGE_WORDS = (
    "did not converge",
    "failed to converge",
    "not converge",
    "convergence",
    "maximum number of iterations",
    "maxiter",
    "no convergence",
)
_RESOURCE_WORDS = (
    "out of memory",
    "memoryerror",
    "killed",
    "cannot allocate",
    "disk quota",
    "no space left",
    "resource limit",
)
_TIMEOUT_WORDS = ("timed out", "timeout", "time limit exceeded", "execution exceeded")

_EXCEPTION_KIND = {
    "SyntaxError": "syntax_error",
    "IndentationError": "syntax_error",
    "TabError": "syntax_error",
    "ModuleNotFoundError": "import_unavailable",
    "ImportError": "import_unavailable",
    "MemoryError": "resource_exhausted",
    "TimeoutError": "timeout",
    "AssertionError": "assertion_failure",
    "FloatingPointError": "numerical_instability",
    "OverflowError": "numerical_instability",
    "ZeroDivisionError": "numerical_instability",
    "FileNotFoundError": "contract_violation",
    "KeyboardInterrupt": "resource_exhausted",
}


class Frame(ClosedContract):
    model_config = ConfigDict(
        extra="forbid", strict=True, allow_inf_nan=False, frozen=True
    )
    filename: str = Field(max_length=400)
    line: int = Field(ge=0)
    function: str | None = Field(default=None, max_length=200)


class ExecutionFailure(ClosedContract):
    """One classified failure. Bounded, typed, and safe to persist verbatim."""

    model_config = ConfigDict(
        extra="forbid", strict=True, allow_inf_nan=False, frozen=True
    )

    kind: FailureKind
    summary: Text
    exception_type: str | None = Field(default=None, max_length=200)
    offending_frame: Frame | None = None
    failing_test_id: Identifier | None = None
    unavailable_module: str | None = Field(default=None, max_length=200)
    missing_outputs: list[str] = Field(default_factory=list, max_length=16)
    signals: list[str] = Field(default_factory=list, max_length=12)
    evidence: str = Field(default="", max_length=MAX_EVIDENCE_CHARS)


def _tail(value, limit=MAX_STREAM_CHARS):
    if value is None:
        return ""
    if not isinstance(value, str):
        raise Invalid("Sandbox streams must be text")
    return value[-limit:]


def _filenames(files):
    if files is None:
        return []
    if not isinstance(files, list):
        raise Invalid("Supply generated files as a list")
    if len(files) > 256:
        raise Invalid("Too many generated files to classify")
    names = []
    for item in files:
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            name = item.get("filename") or item.get("path") or item.get("name") or ""
        else:
            raise Invalid("Generated file entries must be names or mappings")
        if not isinstance(name, str):
            raise Invalid("Generated file names must be text")
        names.append(name.rsplit("/", 1)[-1].strip())
    return [n for n in names if n]


def _last_exception(text):
    """Take the exception from the end of the traceback, where Python actually puts it."""
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if not stripped or line.startswith(("  ", "\t")) and "Error" not in stripped:
            continue
        match = _EXCEPTION.match(stripped)
        if match:
            return match.group(1), match.group(2).strip()
    return None, ""


def _frame(text):
    frames = _FRAME.findall(text)
    if not frames:
        return None
    filename, line, function = frames[-1]
    return Frame(
        filename=filename[:400],
        line=int(line[:9]),
        function=(function or None) and function[:200],
    )


def _classify(exception_type, message, blob, names):
    """Exception type first; message patterns only where the type is ambiguous."""
    signals = []
    lowered = (message + "\n" + blob).lower()

    def has(words):
        return [w for w in words if w in lowered]

    direct = _EXCEPTION_KIND.get(exception_type or "")
    if direct == "assertion_failure":
        return "assertion_failure", ["AssertionError"]
    if direct in ("syntax_error", "import_unavailable", "resource_exhausted", "timeout"):
        return direct, [exception_type]

    resource = has(_RESOURCE_WORDS)
    if resource and exception_type in (None, "RuntimeError", "OSError", "MemoryError"):
        return "resource_exhausted", resource[:4]
    timeouts = has(_TIMEOUT_WORDS)
    if timeouts and exception_type in (None, "RuntimeError", "TimeoutError", "OSError"):
        return "timeout", timeouts[:4]

    shape = has(_SHAPE_WORDS)
    unstable = has(_UNSTABLE_WORDS)
    converge = has(_CONVERGE_WORDS)

    if direct == "numerical_instability":
        return "numerical_instability", [exception_type, *unstable[:3]]
    if direct == "contract_violation":
        return "contract_violation", [exception_type]

    if exception_type in ("LinAlgError", "LinAlgWarning"):
        if any(w in lowered for w in ("singular", "positive definite", "ill-conditioned")):
            return "numerical_instability", [exception_type, *unstable[:3]]
        if shape:
            return "dimension_mismatch", [exception_type, *shape[:3]]
        return "numerical_instability", [exception_type]

    if shape and not converge:
        return "dimension_mismatch", shape[:4]
    if converge:
        return "convergence_failure", converge[:4]
    if unstable:
        return "numerical_instability", unstable[:4]

    if exception_type:
        signals.append(exception_type)
    if not names:
        return "empty_output", signals + ["no generated files"]
    return "contract_violation", signals + ["unclassified exception with generated files"]


def _output_contract(names):
    missing = [n for n in REQUIRED_OUTPUTS if n not in names]
    has_image = any(n.lower().endswith(REQUIRED_IMAGE_SUFFIXES) for n in names)
    if not has_image:
        missing.append("an explanatory PNG plot")
    return missing, has_image


def parse_execution_failure(stdout, stderr, files=None):
    """Classify what actually went wrong in one sandbox attempt.

    `stdout` and `stderr` are bounded text from the attempt; `files` are the generated
    outputs that were actually collected (names, or mappings carrying `filename`).
    Classification uses the exception type at the end of the traceback first, then
    message patterns, then the output contract, then the absence of any output at all.

    Returns a dict with `kind` set to None when nothing failed. A None kind means the
    program ran and produced its declared files. It does not mean the model is right.
    """
    out, err = _tail(stdout), _tail(stderr)
    names = _filenames(files)
    blob = (err + "\n" + out).strip()
    exception_type, message = _last_exception(err) or (None, "")
    if not exception_type:
        exception_type, message = _last_exception(out)
    missing, has_image = _output_contract(names)

    if not blob and not names:
        failure = ExecutionFailure(
            kind="empty_output",
            summary="The attempt produced no output stream and no generated files.",
            signals=["no stdout", "no stderr", "no files"],
            missing_outputs=list(missing)[:16],
        )
    elif exception_type or any(
        w in blob.lower()
        for w in ("traceback (most recent call last)", *_TIMEOUT_WORDS, *_RESOURCE_WORDS)
    ):
        kind, signals = _classify(exception_type, message, blob, names)
        module = _MODULE_NAME.search(blob)
        test_tag = _TEST_TAG.search(blob)
        frame = _frame(blob)
        failing = test_tag.group(1) if test_tag else None
        if (
            failing is None
            and kind == "assertion_failure"
            and frame
            and frame.function
            and frame.function.startswith("test_")
            and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", frame.function)
        ):
            failing = frame.function
        failure = ExecutionFailure(
            kind=kind,
            summary=((exception_type + ": ") if exception_type else "")
            + (message[:400] or "Failure detected in the attempt output"),
            exception_type=exception_type,
            offending_frame=frame,
            failing_test_id=failing,
            unavailable_module=module.group(1)[:200]
            if module and kind == "import_unavailable"
            else None,
            missing_outputs=list(missing)[:16],
            signals=[str(s)[:120] for s in signals if s][:12],
            evidence=blob[-MAX_EVIDENCE_CHARS:],
        )
    elif missing:
        failure = ExecutionFailure(
            kind="contract_violation",
            summary="The attempt ran without raising but did not produce its declared outputs: "
            + ", ".join(missing[:6]),
            missing_outputs=list(missing)[:16],
            signals=["output contract"],
            evidence=("generated: " + ", ".join(sorted(names)[:20]))[-MAX_EVIDENCE_CHARS:],
        )
    else:
        failure = None

    return {
        "status": "failed" if failure else "clean",
        "kind": failure.kind if failure else None,
        "failure": failure.model_dump() if failure else None,
        "generated_filenames": sorted(names)[:64],
        "output_contract": {
            "required": list(REQUIRED_OUTPUTS),
            "missing": missing[:16],
            "has_image": has_image,
        },
        "clean_run_note": (
            "A clean run means the program executed and produced its declared files. It "
            "is not a scientific result and is not independent verification."
        ),
        "independently_validated": False,
        "scope": SCOPE,
    }


def _as_failure(value):
    """Accept a parse result, a failure mapping, or an ExecutionFailure."""
    if isinstance(value, ExecutionFailure):
        return value
    if not isinstance(value, dict):
        raise Invalid("Supply a classified execution failure")
    inner = value.get("failure") if "failure" in value else value
    if inner is None:
        raise Invalid("No failure was classified; there is nothing to act on")
    if isinstance(inner, ExecutionFailure):
        return inner
    if not isinstance(inner, dict) or inner.get("kind") not in FAILURE_KINDS:
        raise Invalid("Unknown execution failure kind")
    try:
        return ExecutionFailure.model_validate(inner)
    except ValueError as exc:
        raise Invalid("Invalid execution failure contract: " + str(exc)[:800]) from exc


def _guidance_body(failure):
    """Per-kind instruction. Every branch says what to do differently, never 'run it again'."""
    kind = failure.kind
    where = ""
    if failure.offending_frame:
        frame = failure.offending_frame
        where = f" The last frame was {frame.filename} line {frame.line}" + (
            f" in {frame.function}." if frame.function else "."
        )
    if kind == "syntax_error":
        return (
            "The program did not parse, so nothing was executed and no result exists."
            + where
            + " Fix the statement at that line, then re-read the surrounding block for the "
            "same mistake before submitting.",
            [
                "Correct the parse error at the reported line.",
                "Re-read the enclosing function; a parse error usually hides a second one.",
                "Keep every other part of the program unchanged so the next failure is attributable.",
            ],
            ["Do not restructure the model while fixing a parse error."],
            "code",
        )
    if kind == "import_unavailable":
        module = failure.unavailable_module or "the requested package"
        return (
            f"{module} is not installed in this sandbox. The same import cannot succeed on "
            "a second attempt, so repeating it is not a fix. Choose one: implement the "
            "routine you need directly in numpy/scipy or plain Python, import an equivalent "
            "from the supplied symplex_sci.py bundle if it provides one, or declare the "
            "dependency as an unmet gap and reduce the claim to what the available packages "
            "actually support.",
            [
                f"Remove the import of {module} and every call that depends on it.",
                "Implement the needed routine directly, or use an available equivalent, or drop the claim.",
                "If the capability is genuinely required, record it as an unmet dependency instead of simulating it.",
            ],
            [
                "Do not repeat the same import.",
                "Do not claim a result the unavailable package was supposed to produce.",
            ],
            "code",
        )
    if kind == "numerical_instability":
        return (
            "The computation produced non-finite or unstable values, so the numbers it "
            "printed describe the solver, not the mechanism."
            + where
            + " Treat this as a modelling defect: check the sign and magnitude of every rate, "
            "guard divisions and logs against zero, rescale badly conditioned quantities, and "
            "use a stiff-capable integrator with an explicit tolerance if the system is stiff.",
            [
                "Locate the first step at which a value stops being finite and print its inputs.",
                "Guard divisions, logarithms and powers against zero and negative arguments.",
                "Rescale states to comparable magnitudes, or switch to a stiff solver with a declared tolerance.",
                "Add an invariant assertion that fails at the first non-finite value rather than at the end.",
            ],
            ["Do not clip or mask non-finite values to make the run complete."],
            "code",
        )
    if kind == "convergence_failure":
        return (
            "A solver or estimator stopped without converging, so its output is an "
            "unconverged iterate and must not be reported as a result."
            + where
            + " Change the problem the solver is given, not only its iteration count: rescale "
            "or reparameterize, supply an analytic Jacobian or better starting values, or "
            "loosen a tolerance you can justify and record.",
            [
                "Report the residual actually reached and the tolerance requested.",
                "Rescale or reparameterize so the search is better conditioned.",
                "Supply informed starting values, bounds, or a Jacobian.",
                "If it still fails to converge, say so and stop; an unconverged number is not a result.",
            ],
            ["Do not report the last iterate as if it had converged."],
            "code",
        )
    if kind == "assertion_failure":
        target = failure.failing_test_id or "an unnamed test"
        return (
            f"Your own test {target} failed."
            + where
            + " Before changing the code, decide which side is wrong: the implementation, or "
            "the assertion. If the assertion was wrong, say so explicitly and record why -- "
            "silently weakening a test until it passes destroys the only diagnostic you have.",
            [
                f"State whether the defect is in the implementation or in {target}.",
                "Fix the implementation, or amend the test and record the reason in the results JSON.",
                "Keep the test in the suite; do not delete it.",
            ],
            [
                "Do not loosen a tolerance without recording the change and its justification.",
                "Do not delete the failing test.",
            ],
            "code",
        )
    if kind == "dimension_mismatch":
        return (
            "Arrays or declared quantities did not line up."
            + where
            + " This is usually the plan's units or shapes disagreeing with the code, not a "
            "typo. Print the shape and unit of each operand at the failing line and check them "
            "against the declared states before editing.",
            [
                "Print the shape and declared unit of every operand at the failing line.",
                "Reconcile them with the plan's declared states, units and time scale.",
                "Fix the disagreement at its source rather than reshaping to make the call succeed.",
            ],
            ["Do not add a reshape or broadcast that hides a unit or time-base error."],
            "plan",
        )
    if kind == "contract_violation":
        missing = ", ".join(failure.missing_outputs[:6]) or "one or more declared outputs"
        return (
            f"The program did not produce {missing}. Nothing downstream can read this run: "
            "the host recomputes its checks from the generated CSV, so a missing or "
            "mis-shaped output file means there is no result to verify, whatever the "
            "program printed. Confirm the plan's declared outputs and the writing code "
            "agree before spending another attempt.",
            [
                "Write every declared output file to /mnt/data before the program exits.",
                "Check the CSV column names and units against the plan's declared states.",
                "If an output cannot be produced, revise the plan's declared outputs rather than omitting the file silently.",
            ],
            [
                "Do not report metrics for a file that was never written.",
                "Do not rename a declared output to whatever happened to be produced.",
            ],
            "plan",
        )
    if kind == "resource_exhausted":
        return (
            "The attempt exceeded the sandbox resource envelope, so it was stopped rather "
            "than completed. A smaller version of the same program is the only thing worth "
            "running next: reduce the grid, horizon, ensemble size or replicate count, and "
            "state the reduction in the results rather than presenting it as the intended run.",
            [
                "Reduce grid resolution, horizon, ensemble size or replicate count to fit the envelope.",
                "Stream or downsample outputs instead of accumulating full trajectories in memory.",
                "Record the reduction and what it costs the conclusion.",
            ],
            ["Do not resubmit the same size and hope for a larger container."],
            "metareasoner",
        )
    if kind == "timeout":
        return (
            "The attempt did not finish inside the time envelope, so there is no result and "
            "no evidence about the model. Reduce the work or change the method; the envelope "
            "belongs to the host and is not negotiable from inside the sandbox.",
            [
                "Cut the dominant cost: shorter horizon, coarser grid, fewer replicates, or a cheaper solver.",
                "Profile which stage consumed the time and report it.",
                "If the intended computation cannot fit, say so and propose the reduced version explicitly.",
            ],
            ["Do not resubmit the same computation expecting more time."],
            "metareasoner",
        )
    return (
        "The attempt produced no usable output at all. Establish that the program runs and "
        "writes one small file before adding any modelling: a run that produces nothing "
        "cannot be diagnosed further from here.",
        [
            "Write a minimal file to /mnt/data first and confirm it is collected.",
            "Print the package versions actually available before using them.",
            "Add the model back one component at a time.",
        ],
        ["Do not add modelling detail while the program produces nothing."],
        "code",
    )


def revision_guidance(failure):
    """A bounded, typed, actionable instruction for the NEXT attempt.

    The instruction always says what to do differently. There is deliberately no branch
    that amounts to running the same thing again: a blind retry is the failure mode this
    function exists to design out.
    """
    parsed = _as_failure(failure)
    instruction, actions, forbidden, return_to = _guidance_body(parsed)
    return {
        "status": "guidance",
        "kind": parsed.kind,
        "instruction": instruction[:MAX_GUIDANCE_CHARS],
        "required_actions": [a[:MAX_ACTION_CHARS] for a in actions][:MAX_ACTIONS],
        "forbidden": [f[:MAX_ACTION_CHARS] for f in forbidden][:MAX_ACTIONS],
        "return_to": return_to,
        "failing_test_id": parsed.failing_test_id,
        "offending_frame": parsed.offending_frame.model_dump()
        if parsed.offending_frame
        else None,
        "retryable": parsed.kind in RETRYABLE_KINDS,
        "independently_validated": False,
        "scope": (
            "Instruction for the next authoring attempt. Following it does not make the "
            "next attempt correct, and no guidance here is a scientific judgement."
        ),
    }


def is_retryable(failure):
    """True when another CODE attempt can plausibly fix this; False when it cannot.

    Returns a bare boolean rather than a scoped dict: it is a predicate used in control
    flow, and `escalate` carries the scoped explanation for the negative case.
    """
    return _as_failure(failure).kind in RETRYABLE_KINDS


def escalate(failure, *, repeats=1):
    """Decide whether the next step leaves the code loop, and where it goes.

    Two ways out. A structurally non-retryable failure leaves immediately: a contract
    violation goes back to the plan, an exhausted envelope goes to the metareasoner that
    owns the envelope. And a failure that keeps recurring leaves even when it is
    nominally fixable -- the second identical failure is evidence that the fix is not in
    the code, and a third attempt at it is a purchase, not an experiment.
    """
    parsed = _as_failure(failure)
    if not isinstance(repeats, int) or isinstance(repeats, bool) or not 1 <= repeats <= 16:
        raise Invalid("repeats must be an integer between 1 and 16")
    retryable = parsed.kind in RETRYABLE_KINDS
    if not retryable:
        target = ESCALATION_TARGET.get(parsed.kind, "plan")
        reason = (
            "This failure is structural: another code attempt reproduces it. "
            + (
                "The program and the plan disagree about what it must produce."
                if parsed.kind == "contract_violation"
                else "The resource envelope is owned by the host and cannot be changed from inside the attempt."
            )
        )
        return {
            "status": "escalate",
            "escalate": True,
            "target": target,
            "kind": parsed.kind,
            "repeats": repeats,
            "reason": reason,
            "independently_validated": False,
            "scope": "Routing decision for a failed authoring attempt; it is not a judgement about the model.",
        }
    if repeats >= REPEAT_ESCALATION:
        return {
            "status": "escalate",
            "escalate": True,
            "target": "plan",
            "kind": parsed.kind,
            "repeats": repeats,
            "reason": (
                f"The same {parsed.kind} has now occurred {repeats} times. Two identical failures "
                "can be a partial fix that did not go far enough; a third is evidence that the fix "
                "is not in the code. Re-plan instead of buying another attempt."
            ),
            "independently_validated": False,
            "scope": "Routing decision for a failed authoring attempt; it is not a judgement about the model.",
        }
    return {
        "status": "continue",
        "escalate": False,
        "target": "code",
        "kind": parsed.kind,
        "repeats": repeats,
        "reason": "A first occurrence of a fixable failure; one revised code attempt is warranted.",
        "independently_validated": False,
        "scope": "Routing decision for a failed authoring attempt; it is not a judgement about the model.",
    }
