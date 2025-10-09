"""Pathwise Monte Carlo sensitivity estimators."""
import numpy as np
from monte_carlo import MonteCarloPricing

def _simulate_with_draws(mc: MonteCarloPricing, risk_neutral: bool, Z: np.ndarray | None):
    """Simulate paths and ensure the driving noise is returned."""
    if Z is None:
        Z = np.random.standard_normal((mc.steps, mc.num_paths))
    paths = mc._simulate_paths(risk_neutral=risk_neutral, Z=Z)
    return paths, Z

def _payoff_derivative(S_T: np.ndarray, strike: float, call: bool) -> np.ndarray:
    """Derivative of vanilla payoff w.r.t terminal stock (pathwise)."""
    if call:
        return (S_T > strike).astype(float)
    return -(S_T < strike).astype(float)

def pathwise_delta(mc: MonteCarloPricing, *, call: bool = True, risk_neutral: bool = True, Z: np.ndarray | None = None, return_std: bool = False):
    """Estimate delta via the pathwise derivative estimator.

    Parameters accept the MonteCarloPricing instance along with an optional
    pre-generated noise matrix Z so greeks can share the same randomness.
    """

    paths, Z = _simulate_with_draws(mc, risk_neutral, Z)
    S_T = paths[-1]
    payoff_prime = _payoff_derivative(S_T, mc.X, call)

    discount = np.exp(-mc.r * mc.T)
    pathwise = discount * payoff_prime * (S_T / mc.S_0) # dS_T/dS_0 = S_T / S_0

    estimate = np.mean(pathwise)
    if not return_std:
        return estimate
    stderr = np.std(pathwise, ddof=1) / np.sqrt(mc.num_paths)
    return estimate, stderr

def pathwise_vega(mc: MonteCarloPricing, *, call: bool = True, risk_neutral: bool = True, Z: np.ndarray | None = None, return_std: bool = False):
    """Estimate vega via the pathwise derivative estimator.

    Relies on the same Brownian increments used for pricing so pass Z when
    coordinating multiple sensitivities with common random numbers.
    """

    paths, Z = _simulate_with_draws(mc, risk_neutral, Z)
    S_T = paths[-1]
    payoff_prime = _payoff_derivative(S_T, mc.X, call)

    dt = mc.T / mc.steps
    W_T = np.sqrt(dt) * np.sum(Z, axis=0)  # Total Brownian increment per path

    discount = np.exp(-mc.r * mc.T)
    pathwise = discount * payoff_prime * S_T * (W_T - mc.sigma * mc.T)

    estimate = np.mean(pathwise)
    if not return_std:
        return estimate
    stderr = np.std(pathwise, ddof=1) / np.sqrt(mc.num_paths)
    return estimate, stderr


__all__ = ["pathwise_delta", "pathwise_vega"]

