"""The agent's test contract for its OWN model. A diagnostic, never independent evidence.

READ THIS BEFORE USING ANYTHING IN THIS MODULE.

A test written by the same agent that wrote the model is a *maker-reported diagnostic*.
It establishes that the program does what its author intended. It does not establish
that the model is correct, well-posed, identifiable, or empirically valid, and it can
never be counted as evidence for a scientific claim. `docs/release-audit.md` finding 2
records the concrete failure mode this module exists to prevent: a program that reports
a passing boolean while violating the bound the boolean claims to check.

Two structural defences encode that line so it cannot be argued away in prose:

1. `TestSpec.authority` is `Literal["maker_reported"]` on a frozen model. There is no
   value an agent can supply, and no assignment it can make afterwards, that promotes
   its own test to independent authority.

2. `protected_subset(tests)` returns only the tests the HOST can recompute for itself
   from generated CSV bytes -- that is, the tests the agent expressed as a
   `NumericCheck` in `symplex/evaluation/numerics.py`. That subset, and only that
   subset, can become evidence, because the host recomputes it from the bytes without
   consulting anything the agent said about the outcome. Everything else is commentary.

A test asserting "the model is biologically plausible" is a legitimate thing for an
agent to write down and a legitimate thing to fail. It is not host-recomputable, so it
stays outside the protected subset forever.
"""

from typing import Literal

from pydantic import ConfigDict, Field, ValidationError, model_validator

from symplex.core.contracts import Invalid, digest
from symplex.evaluation.numerics import MAX_CHECKS, NumericCheck
from symplex.modeling.complex_system import ClosedContract, Identifier, Text

MAX_TESTS = 64
AUTHORITY = "maker_reported"
SCOPE = (
    "Agent-written tests of the agent's own model. Passing establishes that the code "
    "runs as its author intended; it does not establish that the model is correct, "
    "well-posed or empirically valid."
)
PROTECTED_SCOPE = (
    "Host-recomputable subset only. The host recomputes these from generated CSV bytes "
    "and ignores the author's reported outcome; it still does not validate the model."
)

TestKind = Literal[
    "unit",
    "invariant",
    "convergence",
    "recovery",
    "sensitivity",
    "regression",
    "failure_case",
]

KIND_MEANING = {
    "unit": "Declared units and dimensions of a computed quantity.",
    "invariant": "A quantity the mechanism must conserve or bound at every step.",
    "convergence": "Numerical answer stops moving as the discretization is refined.",
    "recovery": "A known planted truth is recovered from generated observations.",
    "sensitivity": "Response to a declared parameter perturbation stays in a stated range.",
    "regression": "A previously recorded value is reproduced under a fixed seed.",
    "failure_case": "The model is asserted to fail, or to refuse, in a declared regime.",
}


class TestSpec(ClosedContract):
    """One agent-written test. `authority` is fixed and unassignable by construction."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        str_strip_whitespace=True,
        frozen=True,
    )

    id: Identifier
    name: Text
    kind: TestKind
    assertion: Text = Field(
        description="What is asserted, in words, precisely enough that a reader can tell whether it held."
    )
    check: Text = Field(
        description="How it is checked in code: the function, expression or comparison that decides it."
    )
    numeric_check: NumericCheck | None = Field(
        default=None,
        description=(
            "Supply a NumericCheck only when this exact assertion can be recomputed by "
            "the host from generated CSV bytes. Its id must equal this test's id, so a "
            "host failure names the agent test it contradicts. Leave null otherwise; a "
            "null is honest, an invented check is not."
        ),
    )
    authority: Literal["maker_reported"] = Field(
        default=AUTHORITY,
        description="Fixed. An agent's test about its own model is maker-reported and is not independent evidence.",
    )

    @model_validator(mode="after")
    def coherent(self):
        if self.numeric_check is not None and self.numeric_check.id != self.id:
            raise ValueError(
                "A protected numeric_check must carry this test's id so a host failure names the test"
            )
        return self


class TestOutcome(ClosedContract):
    """What the agent reports about running one of its own tests. Still maker-reported."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        str_strip_whitespace=True,
        frozen=True,
    )

    test_id: Identifier
    passed: bool
    detail: Text
    authority: Literal["maker_reported"] = AUTHORITY


def _parse_tests(tests):
    if not isinstance(tests, list):
        raise Invalid("Supply a list of TestSpec contracts")
    if len(tests) > MAX_TESTS:
        raise Invalid(f"At most {MAX_TESTS} agent tests may be declared at once")
    parsed = []
    for item in tests:
        if isinstance(item, TestSpec):
            parsed.append(item)
            continue
        try:
            parsed.append(TestSpec.model_validate(item))
        except ValidationError as exc:
            raise Invalid("Invalid agent test contract: " + str(exc)[:1200]) from exc
    if len({t.id for t in parsed}) != len(parsed):
        raise Invalid("Agent test IDs must be unique")
    return parsed


def protected_subset(tests):
    """Return only the tests the host can independently recompute from generated CSV.

    A test enters the protected subset exactly when the agent expressed it as a
    `NumericCheck`, because that is the only form the host can evaluate for itself from
    the bytes. The returned `numeric_checks` are ready to hand to
    `symplex.evaluation.numerics.execute_numeric_checks` unchanged.

    Being protected does not make a test true or the model valid. It only means the
    outcome is computed by the host rather than reported by the maker.
    """
    parsed = _parse_tests(tests)
    protected = [t for t in parsed if t.numeric_check is not None]
    if len(protected) > MAX_CHECKS:
        raise Invalid(
            f"Host numerical checks are capped at {MAX_CHECKS}; reduce the protected subset"
        )
    excluded = [
        {
            "id": t.id,
            "kind": t.kind,
            "assertion": t.assertion[:400],
            "reason": "Not expressible as a host NumericCheck over generated CSV bytes; remains maker-reported commentary",
        }
        for t in parsed
        if t.numeric_check is None
    ]
    checks = [t.numeric_check.model_dump() for t in protected]
    return {
        "status": "resolved",
        "protected": [t.model_dump() for t in protected],
        "protected_ids": [t.id for t in protected],
        "numeric_checks": checks,
        "checks_digest": digest(checks),
        "excluded": excluded,
        "protected_count": len(protected),
        "excluded_count": len(excluded),
        "total_count": len(parsed),
        "csv_filenames": sorted({c["csv_filename"] for c in checks}),
        "authority_note": (
            "Protected tests are recomputed by the host from generated bytes. Excluded "
            "tests stay maker_reported and can never become evidence, however carefully worded."
        ),
        "independently_validated": False,
        "scope": PROTECTED_SCOPE,
    }


def summarize_outcomes(tests, outcomes):
    """Summarize the agent's report on its own tests. Every field here is maker-reported.

    A missing or unexpected outcome is a defect in the report, not a pass. This function
    computes no verdict about the model; it only makes the maker's claim inspectable and
    separable from anything the host recomputes.
    """
    parsed = _parse_tests(tests)
    if not isinstance(outcomes, list) or len(outcomes) > MAX_TESTS:
        raise Invalid(f"Supply at most {MAX_TESTS} reported test outcomes")
    reported = []
    for item in outcomes:
        if isinstance(item, TestOutcome):
            reported.append(item)
            continue
        try:
            reported.append(TestOutcome.model_validate(item))
        except ValidationError as exc:
            raise Invalid("Invalid reported test outcome: " + str(exc)[:1200]) from exc
    if len({o.test_id for o in reported}) != len(reported):
        raise Invalid("Reported test outcomes must be unique per test")
    declared = {t.id for t in parsed}
    seen = {o.test_id for o in reported}
    failed = [o.test_id for o in reported if not o.passed]
    protected = {t.id for t in parsed if t.numeric_check is not None}
    return {
        "status": "reported",
        "authority": AUTHORITY,
        "declared_count": len(parsed),
        "reported_count": len(reported),
        "passed_ids": [o.test_id for o in reported if o.passed],
        "failed_ids": failed,
        "unreported_ids": sorted(declared - seen),
        "undeclared_ids": sorted(seen - declared),
        "protected_failed_ids": [i for i in failed if i in protected],
        "all_reported_passed": bool(reported)
        and not failed
        and not (declared - seen)
        and not (seen - declared),
        "outcomes": [o.model_dump() for o in reported],
        "independently_validated": False,
        "scope": SCOPE,
    }
