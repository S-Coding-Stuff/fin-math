import numpy as np
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable

import torch
import torch.nn as nn
from tqdm.auto import tqdm

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


class NetworkNLSM(nn.Module if nn is not None else object):
    """OptStopRandNN-style continuation network with configurable depth."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_sizes: tuple[int, ...],
    ) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer width.")

        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.LeakyReLU(0.5))
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def _resolve_hidden_sizes(
    *,
    hidden_sizes: tuple[int, ...] | None = None,
    hidden_size: int | None = None,
    optstop_compatible: bool = False,
) -> tuple[int, ...]:
    if hidden_size is not None:
        return (int(hidden_size),)
    if hidden_sizes is None:
        return (10,)
    resolved = tuple(int(width) for width in hidden_sizes)
    if not resolved:
        raise ValueError("hidden_sizes must contain at least one layer width.")
    if optstop_compatible:
        return (resolved[0],)
    return resolved


def _feature_matrix(
    states: np.ndarray,
    *,
    state_fn: StateFn | None = None,
    payoff_values: np.ndarray | None = None,
    use_payoff_as_input: bool = False,
) -> np.ndarray:
    raw = np.asarray(states, dtype=float)
    n_samples = raw.shape[0]
    feats = np.asarray(state_fn(states) if state_fn is not None else raw, dtype=float)
    if feats.ndim == 1:
        if feats.shape[0] != n_samples:
            raise ValueError("state_fn must return one value per path.")
        feats = feats.reshape(-1, 1)
        if not use_payoff_as_input:
            return feats
        if payoff_values is None:
            raise ValueError("payoff_values must be provided when use_payoff_as_input=True.")
        payoff_arr = np.asarray(payoff_values, dtype=float).reshape(-1, 1)
        if payoff_arr.shape[0] != n_samples:
            raise ValueError("payoff_values must provide one value per path.")
        return np.concatenate((feats, payoff_arr), axis=1)
    if feats.ndim == 2:
        if feats.shape[0] != n_samples:
            raise ValueError("state_fn must return an array with first dimension equal to path count.")
        if not use_payoff_as_input:
            return feats
        if payoff_values is None:
            raise ValueError("payoff_values must be provided when use_payoff_as_input=True.")
        payoff_arr = np.asarray(payoff_values, dtype=float).reshape(-1, 1)
        if payoff_arr.shape[0] != n_samples:
            raise ValueError("payoff_values must provide one value per path.")
        return np.concatenate((feats, payoff_arr), axis=1)
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


def _copy_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _init_nlsm_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)


def _fit_continuation_network(
    *,
    features: np.ndarray,
    targets: np.ndarray,
    hidden_sizes: tuple[int, ...],
    lr: float,
    epochs: int,
    batch_size: int,
    min_samples: int,
    seed: int | None,
    initial_state_dict: dict[str, torch.Tensor] | None,
    show_progress: bool,
    progress_desc: str,
) -> tuple[np.ndarray, dict[str, torch.Tensor] | None, float, bool]:

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

    x_tensor = torch.from_numpy(x).double()
    y_tensor = torch.from_numpy(y).double()

    model = NetworkNLSM(
        input_dim=x_tensor.shape[1],
        hidden_sizes=hidden_sizes,
    ).double()
    model.apply(_init_nlsm_weights)
    if initial_state_dict is not None:
        try:
            model.load_state_dict(initial_state_dict)
        except RuntimeError:
            pass

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n_samples = x_tensor.shape[0]
    model.train(True)
    epoch_iter = range(epochs)
    if show_progress:
        epoch_iter = tqdm(
            epoch_iter,
            total=epochs,
            desc=progress_desc,
            leave=False,
            dynamic_ncols=True,
        )
    for _ in epoch_iter:
        permutation = torch.randperm(n_samples)
        last_loss = np.nan
        for start in range(0, n_samples, batch_size):
            batch_idx = permutation[start : start + batch_size]
            x_batch = x_tensor[batch_idx]
            y_batch = y_tensor[batch_idx]

            optimizer.zero_grad()
            prediction = model(x_batch)
            loss = loss_fn(prediction, y_batch)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())

        if show_progress and hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(loss=f"{last_loss:.3e}")

    model.train(False)
    with torch.no_grad():
        prediction = model(x_tensor).squeeze(-1).cpu().numpy()

    fit_mse = float(np.mean((prediction - y.reshape(-1)) ** 2))
    return prediction, _copy_state_dict(model), fit_mse, False


def american_nlsm_pricing_from_paths(
    *,
    paths: np.ndarray,
    K: float,
    r: float,
    T: float,
    call: bool = False,
    include_all_paths: bool = False,
    mask_tolerance: float = 0.0,
    hidden_sizes: tuple[int, ...] = (10,),
    hidden_size: int | None = None,
    lr: float = 1e-3,
    epochs: int = 20,
    batch_size: int = 2000,
    min_samples: int = 8,
    warm_start: bool = False,
    train_itm_only: bool = True,
    use_payoff_as_input: bool = False,
    optstop_compatible: bool = False,
    seed: int | None = None,
    payoff_fn: PayoffFn | None = None,
    state_fn: StateFn | None = None,
    show_progress: bool = False,
    progress_label: str | None = None,
) -> MLPResult:
    """Price from pre-simulated paths using an OptStopRandNN-style NLSM regressor."""

    if T <= 0.0:
        raise ValueError("T must be positive.")
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if epochs < 1:
        raise ValueError("epochs must be >= 1.")
    if optstop_compatible:
        warm_start = False
        include_all_paths = not bool(train_itm_only)

    paths_arr = np.asarray(paths, dtype=float)
    if paths_arr.ndim not in (2, 3):
        raise ValueError("paths must be a 2D (time, paths) or 3D (time, paths, assets) array.")

    n_steps = paths_arr.shape[0]
    n_paths = paths_arr.shape[1]
    if n_steps < 2:
        raise ValueError("Need at least one exercise opportunity.")

    dt = T / (n_steps - 1)
    discount = np.exp(-r * dt)
    resolved_hidden_sizes = _resolve_hidden_sizes(
        hidden_sizes=hidden_sizes,
        hidden_size=hidden_size,
        optstop_compatible=optstop_compatible,
    )

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

    progress_prefix = "NLSM" if progress_label is None else progress_label
    step_iter = range(n_steps - 2, -1, -1)
    if show_progress:
        step_iter = tqdm(
            step_iter,
            total=n_steps - 1,
            desc=f"{progress_prefix} backward",
            leave=False,
            dynamic_ncols=True,
        )

    for t in step_iter:
        include = mask[t]
        if np.any(include):
            states = _feature_matrix(
                paths_arr[t, include],
                state_fn=state_fn,
                payoff_values=payoff[t, include],
                use_payoff_as_input=use_payoff_as_input,
            )
            targets = cashflow[include] * discount
            exercise_value = payoff[t, include]

            continuation_value, fitted_state_dict, fit_mse, used_fallback = _fit_continuation_network(
                features=states,
                targets=targets,
                hidden_sizes=resolved_hidden_sizes,
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
                min_samples=min_samples,
                seed=seed,
                initial_state_dict=previous_state_dict if warm_start else None,
                show_progress=show_progress,
                progress_desc=f"{progress_prefix} t={t} epochs",
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
            epochs_by_step[t] = 0 if used_fallback else int(epochs)
            fallback_by_step[t] = used_fallback
            if show_progress and hasattr(step_iter, "set_postfix"):
                step_iter.set_postfix(itm=int(states.shape[0]), mse=f"{fit_mse:.3e}")

        cashflow[~include] *= discount

    ddof = 1 if n_paths > 1 else 0
    price = float(np.mean(cashflow))
    stderr = float(np.std(cashflow, ddof=ddof) / np.sqrt(n_paths))
    diagnostics = {
        "model": "optstoprandnn_nlsm" if optstop_compatible else "nlsm",
        "hidden_sizes": np.array(resolved_hidden_sizes, dtype=int),
        "selected_count_by_step": selected_count_by_step,
        "fit_mse_by_step": fit_mse_by_step,
        "epochs_by_step": epochs_by_step,
        "fallback_by_step": fallback_by_step,
        "mask_shape": np.array(mask.shape, dtype=int),
        "n_paths": np.array([n_paths], dtype=int),
        "n_steps": np.array([n_steps], dtype=int),
        "train_itm_only": np.array([bool(train_itm_only)], dtype=bool),
        "use_payoff_as_input": np.array([bool(use_payoff_as_input)], dtype=bool),
        "optstop_compatible": np.array([bool(optstop_compatible)], dtype=bool),
    }
    return MLPResult(price=price, stderr=stderr, diagnostics=diagnostics)


def american_nlsm_pricing(
    *,
    S0: float | np.ndarray,
    K: float,
    r: float,
    sigma: float | np.ndarray,
    T: float,
    num_paths: int,
    steps: int,
    call: bool = False,
    seed: int | None = None,
    include_all_paths: bool = False,
    mask_tolerance: float = 0.0,
    antithetic: bool = False,
    hidden_sizes: tuple[int, ...] = (10,),
    hidden_size: int | None = None,
    lr: float = 1e-3,
    epochs: int = 20,
    batch_size: int = 2000,
    min_samples: int = 8,
    warm_start: bool = False,
    train_itm_only: bool = True,
    use_payoff_as_input: bool = False,
    optstop_compatible: bool = False,
    corr: np.ndarray | None = None,
    div: float | np.ndarray = 0.0,
    payoff_fn: PayoffFn | None = None,
    state_fn: StateFn | None = None,
    show_progress: bool = False,
    progress_label: str | None = None,
) -> MLPResult:
    """Price an American-style option using LSM with OptStopRandNN-style NLSM."""
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
    return american_nlsm_pricing_from_paths(
        paths=paths,
        K=K,
        r=r,
        T=T,
        call=call,
        include_all_paths=include_all_paths,
        mask_tolerance=mask_tolerance,
        hidden_sizes=hidden_sizes,
        hidden_size=hidden_size,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        min_samples=min_samples,
        warm_start=warm_start,
        train_itm_only=train_itm_only,
        use_payoff_as_input=use_payoff_as_input,
        optstop_compatible=optstop_compatible,
        seed=seed,
        payoff_fn=payoff_fn,
        state_fn=state_fn,
        show_progress=show_progress,
        progress_label=progress_label,
    )


def american_mlp_pricing_from_paths(
    *,
    paths: np.ndarray,
    K: float,
    r: float,
    T: float,
    call: bool = False,
    include_all_paths: bool = False,
    mask_tolerance: float = 0.0,
    hidden_sizes: tuple[int, ...] = (10,),
    hidden_size: int | None = None,
    negative_slope: float = 0.5,
    lr: float = 1e-3,
    epochs: int = 20,
    first_step_epochs: int | None = None,
    batch_size: int = 2000,
    scale_inputs: bool = False,
    min_samples: int = 8,
    warm_start: bool = False,
    train_itm_only: bool = True,
    use_payoff_as_input: bool = False,
    optstop_compatible: bool = False,
    seed: int | None = None,
    payoff_fn: PayoffFn | None = None,
    state_fn: StateFn | None = None,
    show_progress: bool = False,
    progress_label: str | None = None,
) -> MLPResult:
    del negative_slope, first_step_epochs, scale_inputs
    return american_nlsm_pricing_from_paths(
        paths=paths,
        K=K,
        r=r,
        T=T,
        call=call,
        include_all_paths=include_all_paths,
        mask_tolerance=mask_tolerance,
        hidden_sizes=hidden_sizes,
        hidden_size=hidden_size,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        min_samples=min_samples,
        warm_start=warm_start,
        train_itm_only=train_itm_only,
        use_payoff_as_input=use_payoff_as_input,
        optstop_compatible=optstop_compatible,
        seed=seed,
        payoff_fn=payoff_fn,
        state_fn=state_fn,
        show_progress=show_progress,
        progress_label=progress_label,
    )


def american_mlp_pricing(
    *,
    S0: float | np.ndarray,
    K: float,
    r: float,
    sigma: float | np.ndarray,
    T: float,
    num_paths: int,
    steps: int,
    call: bool = False,
    seed: int | None = None,
    include_all_paths: bool = False,
    mask_tolerance: float = 0.0,
    antithetic: bool = False,
    hidden_sizes: tuple[int, ...] = (10,),
    hidden_size: int | None = None,
    negative_slope: float = 0.5,
    lr: float = 1e-3,
    epochs: int = 20,
    first_step_epochs: int | None = None,
    batch_size: int = 2000,
    scale_inputs: bool = False,
    min_samples: int = 8,
    warm_start: bool = False,
    train_itm_only: bool = True,
    use_payoff_as_input: bool = False,
    optstop_compatible: bool = False,
    corr: np.ndarray | None = None,
    div: float | np.ndarray = 0.0,
    payoff_fn: PayoffFn | None = None,
    state_fn: StateFn | None = None,
    show_progress: bool = False,
    progress_label: str | None = None,
) -> MLPResult:
    del negative_slope, first_step_epochs, scale_inputs
    return american_nlsm_pricing(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        num_paths=num_paths,
        steps=steps,
        call=call,
        seed=seed,
        include_all_paths=include_all_paths,
        mask_tolerance=mask_tolerance,
        antithetic=antithetic,
        hidden_sizes=hidden_sizes,
        hidden_size=hidden_size,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        min_samples=min_samples,
        warm_start=warm_start,
        train_itm_only=train_itm_only,
        use_payoff_as_input=use_payoff_as_input,
        optstop_compatible=optstop_compatible,
        corr=corr,
        div=div,
        payoff_fn=payoff_fn,
        state_fn=state_fn,
        show_progress=show_progress,
        progress_label=progress_label,
    )


def demo() -> None:
    """Run a small Bermudan put example."""
    params = dict(S0=100.0, K=110.0, r=0.1, sigma=0.25, T=1.0, num_paths=50_000, steps=10, seed=123)
    try:
        result = american_nlsm_pricing(call=False, **params)
    except ImportError as exc:
        print("American Put (NLSM): skipped (PyTorch not installed)")
        print("  ", exc)
    else:
        print("American Put (NLSM):", result)


if __name__ == "__main__":
    demo()


__all__ = [
    "MLPResult",
    "NetworkNLSM",
    "PayoffFn",
    "StateFn",
    "american_nlsm_pricing",
    "american_nlsm_pricing_from_paths",
    "american_mlp_pricing",
    "american_mlp_pricing_from_paths",
    "demo",
]
