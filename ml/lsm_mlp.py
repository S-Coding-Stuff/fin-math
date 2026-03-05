"""MLP baseline wrapper for American option pricing via LSM.

This module mirrors the OLS/SVR wrappers but replaces the regression step
with a small PyTorch MLP trained at each exercise time.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

from engines.monte_carlo import MonteCarloPricing, build_mask, immediate_payoff

PayoffFn = Callable[[np.ndarray], np.ndarray]
StateFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class MLPResult:
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


def american_mlp_pricing(*, S0: float | np.ndarray, K: float, r: float, sigma: float | np.ndarray, T: float,
                         num_paths: int, steps: int, call: bool = True, 
                         seed: int | None = None, include_all_paths: bool = False,
                         mask_tolerance: float = 0.0, antithetic: bool = False,
                         hidden_sizes: tuple[int, ...] = (32, 32), lr: float = 1e-3,
                         epochs: int = 50, batch_size: int = 256, 
                         scale_inputs: bool = True, min_samples: int = 8,
                         corr: np.ndarray | None = None, div: float | np.ndarray = 0.0,
                         payoff_fn: PayoffFn | None = None,
                         state_fn: StateFn | None = None) -> MLPResult:
    """Price an American option using LSM with a simple PyTorch MLP.
    The MLP is fit separately at each exercise time using in-sample continuation cashflows.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "american_mlp_pricing requires PyTorch. Install with `pip install torch`."
        ) from exc

    if steps < 1:
        raise ValueError("steps must be at least 1 for American valuation.")
    if T <= 0.0:
        raise ValueError("T must be positive for American valuation.")
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if epochs < 1:
        raise ValueError("epochs must be >= 1.")

    if seed is not None:
        torch.manual_seed(seed)

    pricer = MonteCarloPricing(S_0=S0, X=K, sigma=sigma, T=T, r=r, 
                               num_paths=num_paths, steps=steps, seed=seed, corr=corr, div=div)

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

    def _build_mlp(input_dim: int) -> nn.Module:
        layers: list[nn.Module] = []
        in_dim = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(in_dim, size))
            layers.append(nn.ReLU())
            in_dim = size
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)

    for t in range(n_steps - 2, -1, -1):
        include = mask[t]
        if np.any(include):
            states = _feature_matrix(paths[t, include], state_fn=state_fn)
            continuation_targets = cashflow[include] * discount

            features = states.astype(np.float32)
            targets = continuation_targets.reshape(-1, 1).astype(np.float32)

            if features.shape[0] < min_samples:
                continuation = np.full_like(continuation_targets, np.mean(continuation_targets))
            else:
                x = features
                if scale_inputs:
                    mean = np.mean(x, axis=0, keepdims=True)
                    std = np.std(x, axis=0, keepdims=True)
                    std = np.where(std == 0.0, 1.0, std)
                    x = (x - mean) / std

                x_t = torch.from_numpy(x)
                y_t = torch.from_numpy(targets)

                model = _build_mlp(x.shape[1])
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                loss_fn = nn.MSELoss()

                n_samples = x_t.shape[0]
                for _ in range(epochs):
                    perm = torch.randperm(n_samples)
                    for start in range(0, n_samples, batch_size):
                        idx = perm[start : start + batch_size]
                        xb = x_t[idx]
                        yb = y_t[idx]
                        pred = model(xb)
                        loss = loss_fn(pred, yb)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                with torch.no_grad():
                    preds = model(x_t).squeeze(-1).cpu().numpy()
                continuation = preds

            exercise = payoff[t, include]
            exercise_now = (exercise > 0.0) & (exercise > continuation)
            idx = np.where(include)[0]
            cashflow[idx] = np.where(exercise_now, exercise, cashflow[idx] * discount)
        cashflow[~include] *= discount

    price = float(np.mean(cashflow))
    ddof = 1 if n_paths > 1 else 0
    stderr = float(np.std(cashflow, ddof=ddof) / np.sqrt(n_paths))
    return MLPResult(price=price, stderr=stderr)


def demo() -> None:
    """Run a quick sanity check comparing American call vs European call (MLP)."""
    params = dict(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, num_paths=50_000, steps=50, seed=123)
    try:
        american_call_mlp = american_mlp_pricing(call=True, **params)
    except ImportError as exc:
        print("American Call (MLP): skipped (PyTorch not installed)")
        print("  ", exc)
    else:
        print("American Call (MLP):", american_call_mlp)


if __name__ == "__main__":
    demo()


__all__ = ["MLPResult", "american_mlp_pricing", "demo"]
