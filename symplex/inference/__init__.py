"""Inference: fitting a model to data, quantifying what the fit does not determine, and
scoring whether its predictions are calibrated.

The rest of the repository simulates systems and compares alternatives under declared
assumptions. This package is the missing step between those two: it estimates parameters from
observations, reports the uncertainty and identifiability of the estimate, attributes output
variance to inputs, and scores predictive distributions against realized outcomes.

The entry point is deliberately a pure numeric API. Every routine that needs a model receives
a caller-supplied ``simulate(theta: ndarray) -> ndarray`` returning predictions aligned
element-by-element with ``observations``, plus optional ``sigma`` and per-parameter
``(lower, upper)`` bounds. Nothing here imports a model class, reads storage, executes
generated code, calls a network or a language model, or consumes randomness without an
explicit ``seed``. Every routine that calls the simulator enforces a hard ``max_evaluations``
budget and raises ``symplex.core.contracts.Invalid`` when it is exhausted, rather than
returning a partially explored result that would read like a finished one.

Every public function returns a dict carrying a ``scope`` string stating what the result does
not establish. Read it before quoting the number next to it. In particular: a fitted parameter
with a standard error is the easiest artifact in this repository to over-trust. Intervals here
are asymptotic and conditional on the supplied model being correct; they are not empirical
validation, and a tight interval on a non-identifiable parameter is a regularization artifact.
"""

from symplex.inference.calibration import (
    crps_ensemble,
    interval_score,
    pit_coverage,
    posterior_predictive_check,
    reliability,
)
from symplex.inference.estimation import (
    bootstrap_intervals,
    fisher_information,
    least_squares_fit,
    parameter_uncertainty,
    practical_identifiability,
    profile_likelihood,
)
from symplex.inference.sensitivity import (
    local_sensitivity,
    morris_screening,
    sobol_indices,
    uncertainty_propagation,
)

__all__ = [
    "bootstrap_intervals",
    "crps_ensemble",
    "fisher_information",
    "interval_score",
    "least_squares_fit",
    "local_sensitivity",
    "morris_screening",
    "parameter_uncertainty",
    "pit_coverage",
    "posterior_predictive_check",
    "practical_identifiability",
    "profile_likelihood",
    "reliability",
    "sobol_indices",
    "uncertainty_propagation",
]
