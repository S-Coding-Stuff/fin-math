"""MLP baseline wrapper for American option pricing via LSM.

This module mirrors the OLS/SVR wrappers but replaces the regression step
with a small PyTorch MLP trained at each exercise time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engines.monte_carlo import MonteCarloPricing, build_mask, immediate_payoff


@dataclass
class MLPResult:
    price: float
    stderr: float


def american_mlp_pricing(
    *,
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    num_paths: int,
    steps: int,
    call: bool = True,
    seed: int | None = None,
    include_all_paths: bool = False,
    mask_tolerance: float = 0.0,
    antithetic: bool = False,
    hidden_sizes: tuple[int, ...] = (32, 32),
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 256,
    scale_inputs: bool = True,
    min_samples: int = 8,
) -> MLPResult:
    """Price an American option using LSM with a simple PyTorch MLP.

    Requires torch to be installed. The MLP is fit separately at each
    exercise time using in-sample continuation cashflows.
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

    pricer = MonteCarloPricing(
        S_0=S0,
        X=K,
        sigma=sigma,
        T=T,
        r=r,
        num_paths=num_paths,
        steps=steps,
        seed=seed,
    )

    paths = pricer._simulate_paths(antithetic=antithetic)
    n_steps, n_paths = paths.shape
    if n_steps < 2:
        raise ValueError("Need at least one time step for American valuation.")

    dt = T / (n_steps - 1)
    discount = np.exp(-r * dt)

    payoff = immediate_payoff(paths, strike=K, call=call)
    cashflow = payoff[-1].copy()

    mask = build_mask(
        paths,
        strike=K,
        call=call,
        include_all=include_all_paths,
        tolerance=mask_tolerance,
    )

    def _build_mlp() -> nn.Module:
        layers: list[nn.Module] = []
        in_dim = 1
        for size in hidden_sizes:
            layers.append(nn.Linear(in_dim, size))
            layers.append(nn.ReLU())
            in_dim = size
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)

    for t in range(n_steps - 2, -1, -1):
        include = mask[t]
        if np.any(include):
            states = paths[t, include]
            continuation_targets = cashflow[include] * discount

            features = states.reshape(-1, 1).astype(np.float32)
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

                model = _build_mlp()
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
