from dataclasses import dataclass
from typing import Callable
import pathlib, sys
repo_root = pathlib.Path.cwd().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
from engines.monte_carlo import MonteCarloPricing, build_mask, immediate_payoff

PayoffFn = Callable[[np.ndarray], np.ndarray]
StateFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class OLSResult:
    price: float
    stderr: float


def _feature_matrix(states: np.ndarray, *, state_fn: StateFn | None = None) -> np.ndarray:
    raw = np.asarray(states, dtype=float)
    n_samples = raw.shape[0]
    feats = np.asarray(state_fn(states) if state_fn is not None else raw, dtype=float)
    if feats.ndim == 1:
        if feats.shape[0] != n_samples:
            raise ValueError("state_fn must return one value per path.")
        return feats.reshape(-1, 1)
    if feats.ndim == 2:
        if feats.shape[0] != n_samples:
            raise ValueError("state_fn must return an array with first dimension equal to path count.")
        return feats
    raise ValueError("state_fn output must be 1D or 2D.")


def _payoff_grid(paths: np.ndarray, *, strike: float, call: bool, payoff_fn: PayoffFn | None) -> np.ndarray:
    if payoff_fn is None:
        return immediate_payoff(paths, strike=strike, call=call)
    payoff = np.asarray(payoff_fn(paths), dtype=float)
    expected = (paths.shape[0], paths.shape[1])
    if payoff.shape != expected:
        raise ValueError(f"payoff_fn must return shape {expected}, received {payoff.shape}.")
    return payoff


def _continuation_mask(paths: np.ndarray, payoff: np.ndarray, *, strike: float, include_all_paths: bool,
                       mask_tolerance: float, state_fn: StateFn | None) -> np.ndarray:
    n_steps = payoff.shape[0]
    n_paths = payoff.shape[1]
    if include_all_paths:
        return np.ones((n_steps - 1, n_paths), dtype=bool)

    mask = payoff[:-1] > 0.0
    if mask_tolerance > 0.0:
        for t in range(n_steps - 1):
            ref_state = _feature_matrix(paths[t], state_fn=state_fn)[:, 0]
            mask[t] |= np.abs(ref_state - strike) <= mask_tolerance
    return mask


def price_american_ols(*, S0: float | np.ndarray, K: float, r: float, sigma: float | np.ndarray, T: float, num_paths: int,
                       steps: int, call: bool = True, seed: int | None = None, 
                       include_all_paths: bool = False, antithetic: bool = False,
                       corr: np.ndarray | None = None, div: float | np.ndarray = 0.0,
                       payoff_fn: PayoffFn | None = None, state_fn: StateFn | None = None) -> OLSResult:
    """Price an American option using LSM with a simple OLS (monomial) basis."""

    pricer = MonteCarloPricing(S_0=S0, X=K, sigma=sigma, T=T, r=r, num_paths=num_paths,
                               steps=steps, seed=seed, corr=corr, div=div)

    price, stderr = pricer.american(call=call, basis_fn="monomial", antithetic=antithetic,
                                    include_all_paths=include_all_paths, mask_tolerance=0.0,
                                    payoff_fn=payoff_fn, state_fn=state_fn)

    return OLSResult(price=float(price), stderr=float(stderr))


def price_american_svr(*, S0: float | np.ndarray, K: float, r: float, sigma: float | np.ndarray, T: float, num_paths: int,
                       steps: int, call: bool = True, seed: int | None = None, 
                       include_all_paths: bool = False, mask_tolerance: float = 0.0, 
                       antithetic: bool = False, svr_kwargs: dict | None = None, 
                       scale_inputs: bool = True, min_samples: int = 2,
                       corr: np.ndarray | None = None, div: float | np.ndarray = 0.0,
                       payoff_fn: PayoffFn | None = None, state_fn: StateFn | None = None) -> OLSResult:
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
                               steps=steps, seed=seed, corr=corr, div=div)

    paths = pricer._simulate_paths(antithetic=antithetic)
    n_steps = paths.shape[0]
    n_paths = paths.shape[1]
    if n_steps < 2:
        raise ValueError("Need at least one time step for American valuation.")

    dt = T / (n_steps - 1)
    discount = np.exp(-r * dt)

    payoff = _payoff_grid(paths, strike=K, call=call, payoff_fn=payoff_fn)
    cashflow = payoff[-1].copy()

    if payoff_fn is None and state_fn is None:
        mask = build_mask(paths, strike=K, call=call, include_all=include_all_paths,
                          tolerance=mask_tolerance)
    else:
        mask = _continuation_mask(
            paths,
            payoff,
            strike=K,
            include_all_paths=include_all_paths,
            mask_tolerance=mask_tolerance,
            state_fn=state_fn,
        )

    model_kwargs = dict(kernel="rbf", C=1.0, epsilon=0.01, gamma="scale")
    if svr_kwargs:
        model_kwargs.update(svr_kwargs)

    for t in range(n_steps - 2, -1, -1):
        include = mask[t]
        if np.any(include):
            states = _feature_matrix(paths[t, include], state_fn=state_fn)
            continuation_targets = cashflow[include] * discount
            features = states

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


def price_european(*, S0: float | np.ndarray, K: float, r: float, sigma: float | np.ndarray, T: float, num_paths: int, 
                   steps: int, call: bool = True, seed: int | None = None, 
                   antithetic: bool = False, corr: np.ndarray | None = None,
                   div: float | np.ndarray = 0.0, payoff_fn: PayoffFn | None = None) -> OLSResult:
    """Simple Monte Carlo European price for comparison checks."""
    pricer = MonteCarloPricing(S_0=S0, X=K, sigma=sigma, T=T, r=r, num_paths=num_paths,
                               steps=steps, seed=seed, corr=corr, div=div)
    price, stderr = pricer.european(call=call, antithetic=antithetic, payoff_fn=payoff_fn)
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
