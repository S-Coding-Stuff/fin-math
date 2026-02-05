"""OLS baseline wrapper for the existing LSM American option pricer.

This module keeps things simple and relies on the MonteCarloPricing.american()
implementation, using a monomial basis as a basic OLS regression baseline.
"""
from __future__ import annotations
from dataclasses import dataclass
import pathlib, sys
repo_root = pathlib.Path.cwd().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
from engines.monte_carlo import MonteCarloPricing, build_mask, immediate_payoff


@dataclass
class OLSResult:
    price: float
    stderr: float


def price_american_ols(*, S0: float, K: float, r: float, sigma: float, T: float, num_paths: int,
                       steps: int, call: bool = True, seed: int | None = None, 
                       include_all_paths: bool = False, antithetic: bool = False) -> OLSResult:
    """Price an American option using LSM with a simple OLS (monomial) basis."""

    pricer = MonteCarloPricing(S_0=S0, X=K, sigma=sigma, T=T, r=r, num_paths=num_paths,
                               steps=steps, seed=seed)

    price, stderr = pricer.american(call=call, basis_fn="monomial", antithetic=antithetic,
                                    include_all_paths=include_all_paths, mask_tolerance=0.0)

    return OLSResult(price=float(price), stderr=float(stderr))


def price_american_svr(*, S0: float, K: float, r: float, sigma: float, T: float, num_paths: int,
                       steps: int, call: bool = True, seed: int | None = None, 
                       include_all_paths: bool = False, mask_tolerance: float = 0.0, 
                       antithetic: bool = False, svr_kwargs: dict | None = None, 
                       scale_inputs: bool = True, min_samples: int = 2) -> OLSResult:
    """Price an American option using LSM with support vector regression.

    Requires scikit-learn to be installed. The SVR model is fit separately
    at each exercise time using the in-sample continuation cashflows.
    """
    try:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("price_american_svr requires scikit-learn. "
                          "Install with `pip install scikit-learn`.") from exc

    if steps < 1:
        raise ValueError("steps must be at least 1 for American valuation.")
    if T <= 0.0:
        raise ValueError("T must be positive for American valuation.")
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1.")

    pricer = MonteCarloPricing(S_0=S0, X=K, sigma=sigma, T=T, r=r, num_paths=num_paths,
                               steps=steps, seed=seed)

    paths = pricer._simulate_paths(antithetic=antithetic)
    n_steps, n_paths = paths.shape
    if n_steps < 2:
        raise ValueError("Need at least one time step for American valuation.")

    dt = T / (n_steps - 1)
    discount = np.exp(-r * dt)

    payoff = immediate_payoff(paths, strike=K, call=call)
    cashflow = payoff[-1].copy()

    mask = build_mask(paths, strike=K, call=call, include_all=include_all_paths,
                      tolerance=mask_tolerance)

    model_kwargs = dict(kernel="rbf", C=1.0, epsilon=0.01, gamma="scale")
    if svr_kwargs:
        model_kwargs.update(svr_kwargs)

    for t in range(n_steps - 2, -1, -1):
        include = mask[t]
        if np.any(include):
            states = paths[t, include]
            continuation_targets = cashflow[include] * discount
            features = states.reshape(-1, 1)

            if features.shape[0] < min_samples:
                continuation = np.full_like(continuation_targets, np.mean(continuation_targets))
            else:
                if scale_inputs:
                    model = make_pipeline(StandardScaler(), SVR(**model_kwargs))
                else:
                    model = SVR(**model_kwargs)
                model.fit(features, continuation_targets)
                continuation = model.predict(features)

            exercise = payoff[t, include]
            exercise_now = (exercise > 0.0) & (exercise > continuation)
            idx = np.where(include)[0]
            cashflow[idx] = np.where(exercise_now, exercise, cashflow[idx] * discount)
        cashflow[~include] *= discount

    price = float(np.mean(cashflow))
    ddof = 1 if n_paths > 1 else 0
    stderr = float(np.std(cashflow, ddof=ddof) / np.sqrt(n_paths))
    return OLSResult(price=price, stderr=stderr)


def price_european(*, S0: float, K: float, r: float, sigma: float, T: float, num_paths: int, 
                   steps: int, call: bool = True, seed: int | None = None, 
                   antithetic: bool = False) -> OLSResult:
    """Simple Monte Carlo European price for comparison checks."""
    pricer = MonteCarloPricing(S_0=S0, X=K, sigma=sigma, T=T, r=r, num_paths=num_paths,
                               steps=steps, seed=seed)
    price, stderr = pricer.european(call=call, antithetic=antithetic)
    return OLSResult(price=float(price), stderr=float(stderr))


def demo() -> None:
    """Run a quick sanity check comparing American call vs European call."""
    params = dict(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, num_paths=50_000, steps=50, seed=123)
    american_call = price_american_ols(call=True, **params)
    euro_call = price_european(call=True, **params)

    print("American Call (OLS baseline):", american_call)
    print("European Call (MC):", euro_call)

    try:
        american_call_svr = price_american_svr(
            call=True,
            **params,
            svr_kwargs={"C": 10.0, "epsilon": 0.05, "gamma": "scale"},
        )
    except ImportError as exc:
        print("American Call (SVR): skipped (scikit-learn not installed)")
        print("  ", exc)
    else:
        print("American Call (SVR):", american_call_svr)


if __name__ == "__main__":
    demo()

__all__ = [
    "OLSResult",
    "price_american_ols",
    "price_american_svr",
    "price_european",
    "demo",
]