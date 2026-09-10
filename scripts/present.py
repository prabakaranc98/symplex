"""Offline presentation demo: zero model calls, zero network, zero cost.

Runs two verified capabilities end to end and prints them for a live audience.
Nothing here establishes empirical validity in any domain; both scenarios use
synthetic data with planted truth, which tests inference mechanics only.
"""

import argparse
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

from symplex.hybrid.sindy import fit_sindy, stability_selection
from symplex.inference.estimation import (
    fisher_information,
    least_squares_fit,
    parameter_uncertainty,
    practical_identifiability,
)
from symplex.modeling.analysis import behavior_descriptor, find_equilibria

RULE = "=" * 72


def head(number, title, claim):
    print("\n" + RULE)
    print(f"  BEAT {number}  ·  {title}")
    print(f"  {claim}")
    print(RULE)


def discover(seed):
    """Recover governing equations from trajectory data alone."""
    head(1, "DISCOVER", "It writes down the equation it was never told.")
    a, b, c, d = 1.5, 1.0, 3.0, 1.0

    def lotka_volterra(t, x):
        return np.array([a * x[0] - b * x[0] * x[1], -c * x[1] + d * x[0] * x[1]])

    times = np.linspace(0, 12, 2400)
    states = solve_ivp(
        lotka_volterra, (0, 12), [4.0, 2.0], t_eval=times, rtol=1e-10, atol=1e-12
    ).y.T

    print("\n  Given: 2400 noisy-free samples of two interacting populations.")
    print("  Withheld: the equations that generated them.\n")

    fit = fit_sindy(states, t=times, degree=2, threshold=0.05)
    truth = ["dx/dt = 1.500 x - 1.000 x y", "dy/dt = -3.000 y + 1.000 x y"]
    for recovered, actual, r2 in zip(fit["equations"], truth, fit["r_squared"]):
        print(f"    recovered  {recovered}")
        print(f"    truth      {actual}")
        print(f"    R^2 = {r2:.6f}\n")

    selection = stability_selection(
        states, t=times, degree=2, threshold=0.05, n_bootstrap=24, seed=seed
    )
    kept = [
        (name, prob)
        for name, prob in zip(
            selection.get("term_names", []),
            np.max(np.atleast_2d(selection["inclusion_probability"]), axis=0),
        )
        if prob >= 0.5
    ]
    if kept:
        print("  Term inclusion probability under resampling (robust terms only):")
        for name, prob in kept:
            print(f"    {name:<10} {prob:.2f}")

    equilibria = find_equilibria(lotka_volterra, [[3.0, 1.5], [0.01, 0.01]])
    print("\n  Then it characterises what it recovered:")
    for point in equilibria["equilibria"][:2]:
        state = np.round(point["state"], 4)
        print(f"    x* = {state}   {point['stability']['classification']}")

    descriptor = behavior_descriptor(times, states)
    period = descriptor["dominant_period"]
    analytic = 2 * np.pi / np.sqrt(a * c)
    print(f"    oscillatory = {descriptor['oscillatory']}   "
          f"period = {period:.3f} (analytic {analytic:.3f})")
    print(f"\n  SCOPE  {fit['scope'][:180]}")


def identify(seed):
    """Catch a fit that looks confident but is not identifiable."""
    head(2, "IDENTIFY", "It refuses to trust its own confident-looking fit.")
    rng = np.random.default_rng(seed)
    x = np.linspace(1, 5, 60)

    def simulate(theta):
        return (theta[0] * theta[1]) * x  # only the PRODUCT is identifiable

    observations = 6.0 * x + rng.normal(0, 0.1, x.size)

    print("\n  Model: y = a * b * x.  Only the product a*b is recoverable.")
    print("  A naive fit still reports a and b separately.\n")

    fit = least_squares_fit(simulate, observations, [2.0, 3.0],
                            [(0.1, 20.0), (0.1, 20.0)], sigma=0.1)
    theta = fit["theta_hat"]
    information = fisher_information(simulate, theta, sigma=0.1)["fim"]
    uncertainty = parameter_uncertainty(information, parameter_names=["a", "b"])
    verdict = practical_identifiability(information, parameter_names=["a", "b"])

    print(f"    fitted a, b   = [{theta[0]:.3f}, {theta[1]:.3f}]")
    print(f"    product a*b   = {theta[0] * theta[1]:.4f}   (true 6.0)  <- recovered")
    errors = uncertainty["standard_errors"]
    print(f"    std errors    = [{errors[0]:.1f}, {errors[1]:.1f}]"
          f"   <- on parameters of size 2 and 3")
    print(f"    correlation   = {uncertainty['correlation'][0][1]:.9f}")
    print(f"\n    VERDICT       = {verdict['overall'].upper()}")
    for parameter in verdict["parameters"]:
        print(f"      {parameter['name']}: {parameter['verdict']}"
              f"  (max |corr| {parameter['max_abs_correlation']:.6f}"
              f" with {parameter['most_correlated_with']})")
    print(f"    numerical rank = {verdict['numerical_rank']} of 2"
          f"   -> one direction is unconstrained by the data")
    print(f"\n  SCOPE  {verdict['scope'][:180]}")


SCENARIOS = {"discover": discover, "identify": identify}


def main():
    parser = argparse.ArgumentParser(
        description="Offline Symplex demo. No model calls, no network, no cost."
    )
    parser.add_argument("--scenario", default="all",
                        choices=[*SCENARIOS, "all"])
    parser.add_argument("--seed", type=int, default=7)
    arguments = parser.parse_args()

    chosen = list(SCENARIOS) if arguments.scenario == "all" else [arguments.scenario]
    started = time.monotonic()
    for name in chosen:
        SCENARIOS[name](arguments.seed)

    print("\n" + RULE)
    print(f"  Completed in {time.monotonic() - started:.1f}s. "
          f"Zero model calls, zero network, zero cost.")
    print("  Both scenarios use synthetic data with planted truth. Recovering it")
    print("  demonstrates inference mechanics, not empirical validity in a domain.")
    print(RULE + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
