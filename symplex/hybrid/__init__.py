"""Hybrid and neurosymbolic scientific modelling: explicit interfaces between families.

The audit's requirement is that combining an observation format, a computational
component, a symbolic rule and a proposed mechanism needs explicit interfaces and evidence
about those interfaces. This package supplies the interfaces:

* `sindy` - sparse identification of nonlinear dynamics. Trajectory data in, a readable
  equation out, with the term-inclusion decision exposed through stability selection and
  an explicit accuracy-versus-sparsity front instead of a hidden threshold.
* `surrogate` - Gaussian process and numpy-MLP components that report the training support
  they were fitted on and their approximation error inside and outside it, which is the
  interface a learned component must expose where it couples to a mechanistic model.
* `ude` - Universal Differential Equations: a known right-hand side composed with a learned
  closure, fitted by multiple shooting, with the closure's magnitude reported against the
  mechanism and `symbolic_recovery` converting the closure back into a candidate equation.
* `constraints` - symbolic rules as executable constraints: projection onto the feasible
  set, per-constraint violation reporting, and soft residuals for a fitting objective.
* `discrepancy` - Kennedy-O'Hagan model discrepancy, an error decomposition that names its
  unattributed remainder, and a deterministic structured-versus-unstructured residual test.

Every public function returns a dict carrying a `scope` string stating what the result does
not establish. Everything is deterministic under an explicit seed. No module here imports
the modelling runtime, the inference package or any network, subprocess or model provider:
the surface is arrays and caller-supplied callables in, dicts out.

Nothing in this package validates a scientific claim. A recovered equation is a sparse fit
inside a chosen library, a surrogate is accurate only where it was trained, a satisfied
constraint set is necessary and not sufficient, and a fitted discrepancy is not an
identification of which part of the model is wrong.
"""

from symplex.hybrid.constraints import (
    penalty_terms,
    project,
    project_trajectory,
    validate_constraints,
    violation_report,
)
from symplex.hybrid.discrepancy import (
    decompose_error,
    discrepancy_diagnosis,
    fit_discrepancy,
)
from symplex.hybrid.sindy import (
    estimate_derivatives,
    fit_sindy,
    pareto_sweep,
    polynomial_library,
    render_equation,
    stability_selection,
    stlsq,
    weak_form_sindy,
)
from symplex.hybrid.surrogate import (
    GPEmulator,
    NeuralSurrogate,
    TrainingSupport,
    approximation_error,
    kernel_matrix,
)
from symplex.hybrid.ude import (
    closure_contribution,
    fit_ude,
    simulate_ude,
    symbolic_recovery,
)

__all__ = [
    "GPEmulator",
    "NeuralSurrogate",
    "TrainingSupport",
    "approximation_error",
    "closure_contribution",
    "decompose_error",
    "discrepancy_diagnosis",
    "estimate_derivatives",
    "fit_discrepancy",
    "fit_sindy",
    "fit_ude",
    "kernel_matrix",
    "pareto_sweep",
    "penalty_terms",
    "polynomial_library",
    "project",
    "project_trajectory",
    "render_equation",
    "simulate_ude",
    "stability_selection",
    "stlsq",
    "symbolic_recovery",
    "validate_constraints",
    "violation_report",
    "weak_form_sindy",
]
