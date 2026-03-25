import numpy as np
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engines.monte_carlo import MonteCarloPricing, build_mask, immediate_payoff

PayoffFn = Callable[[np.ndarray], np.ndarray]
StateFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class MLPResult:
    price: float
    stderr: float
    diagnostics: dict[str, Any]


class FeedForward(nn.Module if nn is not None else object):
    """Simple continuation-value network used inside the LSM backward pass."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: tuple[int, ...],
        *,
        negative_slope: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim

        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LeakyReLU(negative_slope=negative_slope))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)

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


def _scaled_features(features: np.ndarray, *, scale_inputs: bool) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    if not scale_inputs:
        return x

    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    std = np.where(std == 0.0, 1.0, std)
    return (x - mean) / std


def _copy_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _fit_continuation_network(*, features: np.ndarray, targets: np.ndarray, hidden_sizes: tuple[int, ...],
                              negative_slope: float, lr: float, epochs: int, batch_size: int, scale_inputs: bool, 
                              min_samples: int, seed: int | None,
                              initial_state_dict: dict[str, torch.Tensor] | None) -> tuple[np.ndarray, dict[str, torch.Tensor] | None, float, bool]:

    x = np.asarray(features, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    elif x.ndim != 2:
        raise ValueError("features must be a 1D or 2D array.")

    y = np.asarray(targets, dtype=np.float32).reshape(-1, 1)
    if x.shape[0] != y.shape[0]:
        raise ValueError("features and targets must have the same number of rows.")

    if x.shape[0] < min_samples:
        fallback = np.full(y.shape[0], float(np.mean(y)))
        mse = float(np.mean((fallback - y.reshape(-1)) ** 2))
        return fallback, None, mse, True

    if seed is not None:
        torch.manual_seed(seed)

    x_scaled = _scaled_features(x, scale_inputs=scale_inputs)
    x_tensor = torch.from_numpy(x_scaled)
    y_tensor = torch.from_numpy(y)

    model = FeedForward(
        input_dim=x_tensor.shape[1],
        hidden_sizes=hidden_sizes,
        negative_slope=negative_slope,
    )
    if initial_state_dict is not None:
        try:
            model.load_state_dict(initial_state_dict)
        except RuntimeError:
            pass

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n_samples = x_tensor.shape[0]
    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(n_samples)
        for start in range(0, n_samples, batch_size):
            batch_idx = permutation[start : start + batch_size]
            x_batch = x_tensor[batch_idx]
            y_batch = y_tensor[batch_idx]

            optimizer.zero_grad()
            prediction = model(x_batch)
            loss = loss_fn(prediction, y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        prediction = model(x_tensor).squeeze(-1).cpu().numpy()

    fit_mse = float(np.mean((prediction - y.reshape(-1)) ** 2))
    return prediction, _copy_state_dict(model), fit_mse, False


def american_mlp_pricing_from_paths(*, paths: np.ndarray, K: float, r: float, T: float, call: bool = False,
                                    include_all_paths: bool = False,
                                    mask_tolerance: float = 0.0,
                                    hidden_sizes: tuple[int, ...] = (32,),
                                    negative_slope: float = 0.3,
                                    lr: float = 1e-3,
                                    epochs: int = 1,
                                    first_step_epochs: int | None = 10,
                                    batch_size: int = 256,
                                    scale_inputs: bool = True,
                                    min_samples: int = 8,
                                    warm_start: bool = True,
                                    seed: int | None = None,
                                    payoff_fn: PayoffFn | None = None,
                                    state_fn: StateFn | None = None) -> MLPResult:
    """Price from pre-simulated paths using LSM + an MLP continuation regressor."""

    if T <= 0.0:
        raise ValueError("T must be positive.")
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if epochs < 1:
        raise ValueError("epochs must be >= 1.")
    if first_step_epochs is not None and first_step_epochs < 1:
        raise ValueError("first_step_epochs must be >= 1 when provided.")

    paths_arr = np.asarray(paths, dtype=float)
    if paths_arr.ndim not in (2, 3):
        raise ValueError("paths must be a 2D (time, paths) or 3D (time, paths, assets) array.")

    n_steps = paths_arr.shape[0]
    n_paths = paths_arr.shape[1]
    if n_steps < 2:
        raise ValueError("Need at least one exercise opportunity.")

    dt = T / (n_steps - 1)
    discount = np.exp(-r * dt)

    payoff = _payoff_grid(paths_arr, strike=K, call=call, payoff_fn=payoff_fn)
    cashflow = payoff[-1].copy()

    if payoff_fn is None and state_fn is None and paths_arr.ndim == 2:
        mask = build_mask(
            paths_arr,
            strike=K,
            call=call,
            include_all=include_all_paths,
            tolerance=mask_tolerance,
        )
    else:
        mask = _continuation_mask(
            paths_arr,
            payoff,
            strike=K,
            include_all_paths=include_all_paths,
            mask_tolerance=mask_tolerance,
            state_fn=state_fn,
        )

    selected_count_by_step = np.zeros(n_steps - 1, dtype=int)
    fit_mse_by_step = np.full(n_steps - 1, np.nan, dtype=float)
    epochs_by_step = np.zeros(n_steps - 1, dtype=int)
    fallback_by_step = np.zeros(n_steps - 1, dtype=bool)

    previous_state_dict: dict[str, torch.Tensor] | None = None

    for t in range(n_steps - 2, -1, -1):
        include = mask[t]
        if np.any(include):
            states = _feature_matrix(paths_arr[t, include], state_fn=state_fn)
            targets = cashflow[include] * discount
            exercise_value = payoff[t, include]

            step_epochs = first_step_epochs if t == (n_steps - 2) and first_step_epochs is not None else epochs
            continuation_value, fitted_state_dict, fit_mse, used_fallback = _fit_continuation_network(
                features=states,
                targets=targets,
                hidden_sizes=hidden_sizes,
                negative_slope=negative_slope,
                lr=lr,
                epochs=step_epochs,
                batch_size=batch_size,
                scale_inputs=scale_inputs,
                min_samples=min_samples,
                seed=seed,
                initial_state_dict=previous_state_dict if warm_start else None,
            )

            if warm_start and fitted_state_dict is not None:
                previous_state_dict = fitted_state_dict

            selected_idx = np.where(include)[0]
            exercise_now = (exercise_value > 0.0) & (exercise_value > continuation_value)
            cashflow[selected_idx] = np.where(
                exercise_now,
                exercise_value,
                cashflow[selected_idx] * discount,
            )

            selected_count_by_step[t] = int(states.shape[0])
            fit_mse_by_step[t] = fit_mse
            epochs_by_step[t] = 0 if used_fallback else int(step_epochs)
            fallback_by_step[t] = used_fallback

        cashflow[~include] *= discount

    ddof = 1 if n_paths > 1 else 0
    price = float(np.mean(cashflow))
    stderr = float(np.std(cashflow, ddof=ddof) / np.sqrt(n_paths))
    diagnostics = {
        "selected_count_by_step": selected_count_by_step,
        "fit_mse_by_step": fit_mse_by_step,
        "epochs_by_step": epochs_by_step,
        "fallback_by_step": fallback_by_step,
        "mask_shape": np.array(mask.shape, dtype=int),
        "n_paths": np.array([n_paths], dtype=int),
        "n_steps": np.array([n_steps], dtype=int),
    }
    return MLPResult(price=price, stderr=stderr, diagnostics=diagnostics)


def american_mlp_pricing(*, S0: float | np.ndarray, K: float, r: float, sigma: float | np.ndarray, 
                         T: float, num_paths: int, steps: int, call: bool = False, seed: int | None = None, 
                         include_all_paths: bool = False, mask_tolerance: float = 0.0, antithetic: bool = False,
                         hidden_sizes: tuple[int, ...] = (32,), negative_slope: float = 0.3, lr: float = 1e-3, 
                         epochs: int = 1, first_step_epochs: int | None = 10, batch_size: int = 256, 
                         scale_inputs: bool = True, min_samples: int = 8, warm_start: bool = True, corr: np.ndarray | None = None, 
                         div: float | np.ndarray = 0.0, payoff_fn: PayoffFn | None = None, state_fn: StateFn | None = None) -> MLPResult:
    """Price an American-style option using LSM with a PyTorch MLP."""
    if steps < 1:
        raise ValueError("steps must be at least 1.")

    pricer = MonteCarloPricing(
        S_0=S0,
        X=K,
        sigma=sigma,
        T=T,
        r=r,
        num_paths=num_paths,
        steps=steps,
        seed=seed,
        corr=corr,
        div=div,
    )
    paths = pricer._simulate_paths(antithetic=antithetic)
    return american_mlp_pricing_from_paths(
        paths=paths,
        K=K,
        r=r,
        T=T,
        call=call,
        include_all_paths=include_all_paths,
        mask_tolerance=mask_tolerance,
        hidden_sizes=hidden_sizes,
        negative_slope=negative_slope,
        lr=lr,
        epochs=epochs,
        first_step_epochs=first_step_epochs,
        batch_size=batch_size,
        scale_inputs=scale_inputs,
        min_samples=min_samples,
        warm_start=warm_start,
        seed=seed,
        payoff_fn=payoff_fn,
        state_fn=state_fn,
    )


def demo() -> None:
    """Run a small Bermudan put example."""
    params = dict(S0=100.0, K=110.0, r=0.1, sigma=0.25, T=1.0, num_paths=50_000, steps=10, seed=123)
    try:
        result = american_mlp_pricing(call=False, **params)
    except ImportError as exc:
        print("American Put (MLP): skipped (PyTorch not installed)")
        print("  ", exc)
    else:
        print("American Put (MLP):", result)


if __name__ == "__main__":
    demo()


__all__ = [
    "MLPResult",
    "PayoffFn",
    "StateFn",
    "FeedForward",
    "american_mlp_pricing",
    "american_mlp_pricing_from_paths",
    "demo",
]
