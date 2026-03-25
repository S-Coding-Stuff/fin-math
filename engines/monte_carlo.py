from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

import numpy as np
from engines.rlsm import (
    RLSMPolicy,
    evaluate_rlsm_policy,
    fit_rlsm_policy_from_paths,
)

BasisName = Literal["laguerre", "monomial", "hermite", "paper_poly2", "nn", "rlsm"]
PolicyBasisName = Literal["laguerre", "monomial", "hermite", "paper_poly2", "paper_poly4", "paper_poly6"]
PayoffFn = Callable[[np.ndarray], np.ndarray]
StateFn = Callable[[np.ndarray], np.ndarray]

class LSMConfig:
    def __init__(self, *, basis: BasisName = "laguerre", include_all_paths: bool = True,
                 mask_tolerance: float = 0.0, nn_kwargs: Optional[dict] = None,
                 seed: int | None = None, rlsm_kwargs: Optional[dict] = None,
                 train_eval_split: float | None = None) -> None:
        self.basis = basis
        self.include_all_paths = bool(include_all_paths)
        self.mask_tolerance = float(mask_tolerance)
        self.nn_kwargs = {} if nn_kwargs is None else dict(nn_kwargs)
        self.rlsm_kwargs = {} if rlsm_kwargs is None else dict(rlsm_kwargs)
        self.seed = seed
        self.train_eval_split = train_eval_split


def _coerce_asset_vector(value: float | np.ndarray, *, n_assets: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n_assets, float(arr), dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be scalar or a 1D array.")
    if arr.size == 1:
        return np.full(n_assets, float(arr[0]), dtype=float)
    if arr.size != n_assets:
        raise ValueError(f"{name} must have length 1 or {n_assets}, received length {arr.size}.")
    return arr.astype(float, copy=False)


def _as_feature_matrix(states: np.ndarray, *, state_fn: StateFn | None = None) -> np.ndarray:
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


def _default_payoff_grid(paths: np.ndarray, *, strike: float, call: bool) -> np.ndarray:
    arr = np.asarray(paths, dtype=float)
    if arr.ndim == 2:
        return immediate_payoff(arr, strike=strike, call=call)
    if arr.ndim == 3:
        if arr.shape[2] != 1:
            raise ValueError(
                "Multi-asset paths require a custom payoff_fn. "
                "Default intrinsic payoff only supports single-asset paths."
            )
        return immediate_payoff(arr[:, :, 0], strike=strike, call=call)
    raise ValueError("paths must be 2D (time, paths) or 3D (time, paths, assets).")


def immediate_payoff(paths: np.ndarray, *, strike: float, call: bool) -> np.ndarray:
    prices = np.asarray(paths, dtype=float)
    if prices.ndim == 3:
        if prices.shape[2] != 1:
            raise ValueError(
                "immediate_payoff only supports scalar states. "
                "For multi-asset states pass a custom payoff_fn to american()/european()."
            )
        prices = prices[:, :, 0]
    if call:
        return np.maximum(prices - strike, 0.0)
    return np.maximum(strike - prices, 0.0)


def build_mask(paths: np.ndarray, *, strike: float, call: bool, include_all: bool, tolerance: float) -> np.ndarray:
    if include_all:
        return np.ones((paths.shape[0] - 1, paths.shape[1]), dtype=bool)

    payoff = _default_payoff_grid(paths, strike=strike, call=call)
    mask = payoff[:-1] > 0.0
    if tolerance > 0.0:
        states = np.asarray(paths[:-1], dtype=float)
        if states.ndim == 3:
            if states.shape[2] != 1:
                raise ValueError(
                    "mask_tolerance with multi-asset paths requires custom masking. "
                    "Use include_all_paths=False with payoff-driven masks, or supply continuation_mask."
                )
            states = states[:, :, 0]
        mask |= np.abs(states - strike) <= tolerance
    return mask


def lsm_basis(states: np.ndarray, *, strike: float, basis: BasisName) -> np.ndarray:
    raw = np.asarray(states, dtype=float)
    if raw.ndim == 2 and raw.shape[1] > 1:
        if basis in {"nn", "rlsm"}:
            raise ValueError("Neural-network bases must be handled by dedicated regressors.")
        if basis == "paper_poly2":
            cols: list[np.ndarray] = [np.ones(raw.shape[0], dtype=float)]
            cols.extend([raw[:, j] for j in range(raw.shape[1])])
            cols.extend([raw[:, j] ** 2 for j in range(raw.shape[1])])
            for i in range(raw.shape[1]):
                for j in range(i + 1, raw.shape[1]):
                    cols.append(raw[:, i] * raw[:, j])
            return np.column_stack(cols)
        if basis == "monomial":
            cols: list[np.ndarray] = [np.ones(raw.shape[0], dtype=float)]
            cols.extend([raw[:, j] for j in range(raw.shape[1])])
            cols.extend([raw[:, j] ** 2 for j in range(raw.shape[1])])
            return np.column_stack(cols)
        if basis == "hermite":
            cols = [np.ones(raw.shape[0], dtype=float)]
            for j in range(raw.shape[1]):
                cols.append(np.polynomial.hermite.hermvander(raw[:, j], 3)[:, 1:])
            return np.column_stack(cols)
        if basis == "laguerre":
            cols = [np.ones(raw.shape[0], dtype=float)]
            normalized = (raw / strike).astype(float, copy=False)
            for j in range(normalized.shape[1]):
                lag = np.exp(-normalized[:, j] / 2.0)[:, None] * np.polynomial.laguerre.lagvander(
                    normalized[:, j], 3
                )
                cols.append(lag[:, 1:])
            return np.column_stack(cols)
        raise ValueError(f"Unsupported basis '{basis}'.")

    flattened = np.ravel(raw)
    if flattened.size == 0:
        return np.zeros((0, 3))
    if basis in {"nn", "rlsm"}:
        raise ValueError("Neural-network bases must be handled by dedicated regressors.")
    if basis == "paper_poly2":
        return np.column_stack((np.ones_like(flattened), flattened, flattened**2))
    if basis == "monomial":
        return np.vander(flattened, N=3, increasing=True)
    if basis == "hermite":
        return np.polynomial.hermite.hermvander(flattened, 3)
    if basis != "laguerre":
        raise ValueError(f"Unsupported basis '{basis}'.")
    normalized = (flattened / strike).astype(float, copy=False)
    return np.exp(-normalized[:, None] / 2.0) * np.polynomial.laguerre.lagvander(normalized, 3)


def _policy_basis(states: np.ndarray, *, strike: float, basis_name: PolicyBasisName) -> np.ndarray:
    basis = basis_name.lower()
    raw = np.asarray(states, dtype=float)
    if raw.ndim == 0:
        raw = raw.reshape(1)
    if raw.ndim == 1:
        x = raw
        base_states = x
    elif raw.ndim == 2:
        if raw.shape[0] == 0:
            return np.zeros((0, 1), dtype=float)
        x = np.mean(raw, axis=1)
        base_states = raw
    else:
        raise ValueError("states must be 1D or 2D.")

    if x.size == 0:
        return np.zeros((0, 1), dtype=float)
    if basis in {"laguerre", "monomial", "hermite", "paper_poly2"}:
        return lsm_basis(base_states, strike=strike, basis=basis)  # type: ignore[arg-type]
    centered = x - strike
    if basis == "paper_poly4":
        return np.column_stack([np.ones_like(centered), centered, centered**2, centered**3, centered**4])
    if basis == "paper_poly6":
        return np.column_stack(
            [np.ones_like(centered), centered, centered**2, centered**3, centered**4, centered**5, centered**6]
        )
    raise ValueError(
        "Unsupported basis_name. Use one of: "
        "'laguerre', 'monomial', 'hermite', 'paper_poly2', 'paper_poly4', 'paper_poly6'."
    )


def _eval_slice(cashflows: np.ndarray, split: float | None) -> np.ndarray:
    arr = np.asarray(cashflows, dtype=float).reshape(-1)
    if split is None:
        return arr
    if arr.size < 2:
        raise ValueError("train_eval_split requires at least 2 paths.")
    split_f = float(split)
    if not (0.0 < split_f < 1.0):
        raise ValueError("train_eval_split must be strictly between 0 and 1.")
    train_count = max(1, min(int(arr.size * split_f), arr.size - 1))
    sample = arr[train_count:]
    if sample.size == 0:
        raise ValueError("train_eval_split leaves no evaluation paths.")
    return sample


@dataclass
class LSMContinuationPolicy:
    """Reusable LSM continuation policy fitted from paths."""

    call: bool
    strike: float
    rate: float
    maturity: float
    steps: int
    basis_name: PolicyBasisName
    coefficients: list[np.ndarray | None]
    fallback_values: list[float]

    @property
    def dt(self) -> float:
        return self.maturity / self.steps

    @property
    def discount_grid(self) -> np.ndarray:
        return np.exp(-self.rate * self.dt * np.arange(self.steps + 1))

    def continuation_value(self, t: int, states: np.ndarray) -> np.ndarray:
        arr = np.asarray(states, dtype=float)
        n_samples = arr.shape[0] if arr.ndim >= 1 else 1
        if t >= self.steps:
            return np.zeros(n_samples, dtype=float)
        coeffs = self.coefficients[t]
        if coeffs is None:
            return np.full(n_samples, self.fallback_values[t], dtype=float)
        basis = _policy_basis(arr, strike=self.strike, basis_name=self.basis_name)
        return basis @ coeffs

    def stopping_times(self, paths: np.ndarray) -> np.ndarray:
        if paths.ndim == 3 and paths.shape[2] > 1:
            raise ValueError(
                "LSMContinuationPolicy currently supports scalar intrinsic payoffs only."
            )
        payoff = immediate_payoff(paths, strike=self.strike, call=self.call)
        tau = np.full(paths.shape[1], self.steps, dtype=int)
        alive = np.ones(paths.shape[1], dtype=bool)

        for t in range(self.steps):
            if not np.any(alive):
                break
            idx = np.where(alive)[0]
            states = paths[t, idx]
            exercise = payoff[t, idx]
            continuation = self.continuation_value(t, states)
            exercise_now = (exercise > 0.0) & (exercise > continuation)
            chosen = idx[exercise_now]
            tau[chosen] = t
            alive[chosen] = False
        return tau

    def discounted_cashflows(self, paths: np.ndarray) -> np.ndarray:
        payoff = immediate_payoff(paths, strike=self.strike, call=self.call)
        tau = self.stopping_times(paths)
        return self.discount_grid[tau] * payoff[tau, np.arange(paths.shape[1])]


@dataclass
class LSMPolicyFit:
    policy: LSMContinuationPolicy
    stage1_estimate: float
    stage1_stderr: float
    selected_count_by_step: np.ndarray
    fit_mse_by_step: np.ndarray


def fit_lsm_policy_from_paths(paths: np.ndarray, *, strike: float, rate: float, maturity: float,
                              call: bool = False, basis_name: PolicyBasisName = "laguerre",
                              include_all_paths: bool = False, mask_tolerance: float = 0.0) -> LSMPolicyFit:
    """Fit LSM continuation regressions and return a reusable policy object."""
    if paths.ndim == 3 and paths.shape[2] > 1:
        raise ValueError(
            "fit_lsm_policy_from_paths currently supports scalar intrinsic payoffs only. "
            "For multi-asset payoffs, use MonteCarloPricing.american(..., payoff_fn=..., state_fn=...)."
        )
    n_times = paths.shape[0]
    n_paths = paths.shape[1]
    steps = n_times - 1
    if steps < 1:
        raise ValueError("paths must contain at least one exercise step.")
    if maturity <= 0.0:
        raise ValueError("maturity must be positive.")

    dt = maturity / steps
    discount = float(np.exp(-rate * dt))
    payoff = immediate_payoff(paths, strike=strike, call=call)
    mask = build_mask(
        paths,
        strike=strike,
        call=call,
        include_all=include_all_paths,
        tolerance=mask_tolerance,
    )

    cashflow = payoff[-1].copy()
    coeffs_by_step: list[np.ndarray | None] = [None for _ in range(steps)]
    fallback_by_step: list[float] = [0.0 for _ in range(steps)]
    selected = np.zeros(steps, dtype=int)
    fit_mse = np.full(steps, np.nan, dtype=float)

    for t in range(steps - 1, -1, -1):
        include = mask[t]
        if np.any(include):
            states = paths[t, include]
            targets = cashflow[include] * discount
            basis = _policy_basis(states, strike=strike, basis_name=basis_name)
            coeffs, *_ = np.linalg.lstsq(basis, targets, rcond=None)
            continuation = basis @ coeffs
            intrinsic = payoff[t, include]
            exercise_now = (intrinsic > 0.0) & (intrinsic > continuation)

            idx = np.where(include)[0]
            cashflow[idx] = np.where(exercise_now, intrinsic, cashflow[idx] * discount)
            cashflow[~include] *= discount

            coeffs_by_step[t] = coeffs
            fallback_by_step[t] = float(np.mean(targets))
            selected[t] = int(np.sum(include))
            fit_mse[t] = float(np.mean((continuation - targets) ** 2))
        else:
            cashflow *= discount
            coeffs_by_step[t] = None
            fallback_by_step[t] = float(np.mean(cashflow))

    ddof = 1 if n_paths > 1 else 0
    policy = LSMContinuationPolicy(
        call=call,
        strike=strike,
        rate=rate,
        maturity=maturity,
        steps=steps,
        basis_name=basis_name,
        coefficients=coeffs_by_step,
        fallback_values=fallback_by_step,
    )
    return LSMPolicyFit(
        policy=policy,
        stage1_estimate=float(np.mean(cashflow)),
        stage1_stderr=float(np.std(cashflow, ddof=ddof) / np.sqrt(n_paths)),
        selected_count_by_step=selected,
        fit_mse_by_step=fit_mse,
    )


def evaluate_lsm_policy(policy: LSMContinuationPolicy, paths: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Evaluate a fitted LSM policy on provided paths."""
    cashflows = policy.discounted_cashflows(paths)
    n_paths = cashflows.size
    ddof = 1 if n_paths > 1 else 0
    mean = float(np.mean(cashflows))
    stderr = float(np.std(cashflows, ddof=ddof) / np.sqrt(n_paths))
    return mean, stderr, cashflows


def _evaluate_payoff_grid(paths: np.ndarray, *, strike: float, call: bool,
                          payoff_fn: PayoffFn | None) -> np.ndarray:
    if payoff_fn is None:
        return _default_payoff_grid(paths, strike=strike, call=call)

    payoff = np.asarray(payoff_fn(paths), dtype=float)
    if payoff.shape != (paths.shape[0], paths.shape[1]):
        raise ValueError(
            "payoff_fn must return a 2D array with shape (time, paths) when used for American pricing."
        )
    return payoff


def _build_continuation_mask(paths: np.ndarray, payoff: np.ndarray, *, strike: float,
                             include_all: bool, tolerance: float,
                             state_fn: StateFn | None) -> np.ndarray:
    n_steps, n_paths = payoff.shape
    if include_all:
        return np.ones((n_steps - 1, n_paths), dtype=bool)

    mask = payoff[:-1] > 0.0
    if tolerance > 0.0:
        for t in range(n_steps - 1):
            state_values = _as_feature_matrix(paths[t], state_fn=state_fn)
            ref_state = state_values[:, 0]
            mask[t] |= np.abs(ref_state - strike) <= tolerance
    return mask


def _lsm_cashflows(paths: np.ndarray, *, strike: float, call: bool, rate: float, maturity: float,
                   config: LSMConfig, mask: Optional[np.ndarray] = None, capture_mask: bool = False,
                   capture_diagnostics: bool = False, payoff_fn: PayoffFn | None = None,
                   state_fn: StateFn | None = None) -> tuple[np.ndarray, Optional[np.ndarray], Optional[dict[str, np.ndarray]]]:
    if paths.ndim not in (2, 3):
        raise ValueError("paths must be a 2D (time, paths) or 3D (time, paths, assets) array.")

    n_steps = paths.shape[0]
    n_paths = paths.shape[1]
    if n_steps < 2:
        raise ValueError("Need at least one time step for American valuation.")
    if maturity <= 0.0:
        raise ValueError("Maturity must be positive for American valuation.")

    if config.basis == "rlsm":
        if mask is not None:
            raise ValueError("RLSM does not use continuation masks; pass freeze policy in the Greek estimator.")
        rlsm_fit = fit_rlsm_policy_from_paths(
            paths,
            strike=strike,
            rate=rate,
            maturity=maturity,
            call=call,
            payoff_fn=payoff_fn,
            state_fn=state_fn,
            **config.rlsm_kwargs,
        )
        diag: Optional[dict[str, np.ndarray]] = None
        if capture_diagnostics:
            diag = rlsm_fit.policy.diagnostics(paths)
            diag.update(
                {
                    "train_count_by_step": rlsm_fit.train_count_by_step.astype(float),
                    "fit_mse_by_step": rlsm_fit.fit_mse_by_step.astype(float),
                    "stage1_estimate": np.array([rlsm_fit.stage1_estimate], dtype=float),
                    "stage1_stderr": np.array([rlsm_fit.stage1_stderr], dtype=float),
                }
            )
        mask_return = np.ones((n_steps - 1, n_paths), dtype=bool) if capture_mask else None
        return rlsm_fit.cashflows, mask_return, diag

    dt = maturity / (n_steps - 1)
    discount = np.exp(-rate * dt)

    payoff = _evaluate_payoff_grid(paths, strike=strike, call=call, payoff_fn=payoff_fn)
    cashflow = payoff[-1].copy()
    expected = (n_steps - 1, n_paths)
    if mask is not None:
        mask_to_use = np.asarray(mask, dtype=bool)
        if mask_to_use.shape != expected:
            raise ValueError(f"Continuation mask must have shape {expected}, received {mask_to_use.shape}.")
    else:
        mask_to_use = _build_continuation_mask(
            paths,
            payoff,
            strike=strike,
            include_all=config.include_all_paths,
            tolerance=config.mask_tolerance,
            state_fn=state_fn,
        )

    continuation_est = None
    exercise_time = None
    immediate_payoff_slice = None
    if capture_diagnostics:
        continuation_est = np.full((n_steps - 1, n_paths), np.nan)
        exercise_time = np.full(n_paths, -1, dtype=int)
        immediate_payoff_slice = payoff[:-1].copy()

    for t in range(n_steps - 2, -1, -1):
        include = mask_to_use[t]
        if np.any(include):
            states = _as_feature_matrix(paths[t], state_fn=state_fn)[include]
            continuation_targets = cashflow[include] * discount
            if config.basis == "nn":
                from engines.lsm_regressors import estimate_continuation_nn

                continuation = estimate_continuation_nn(
                    states,
                    continuation_targets,
                    seed=config.seed,
                    **config.nn_kwargs,
                )
            else:
                basis = lsm_basis(states, strike=strike, basis=config.basis)
                coeffs, *_ = np.linalg.lstsq(basis, continuation_targets, rcond=None)
                continuation = basis @ coeffs
            exercise = payoff[t, include]
            # Only allow exercise when the payoff is strictly positive
            # to avoid spurious "exercise" decisions for out-of-the-money states.
            exercise_now = (exercise > 0.0) & (exercise > continuation)
            idx = np.where(include)[0]
            cashflow[idx] = np.where(exercise_now, exercise, cashflow[idx] * discount)

            if capture_diagnostics:
                continuation_est[t, include] = continuation
                exercise_time[idx[exercise_now]] = t
        cashflow[~include] *= discount

    diagnostics = None
    if capture_diagnostics:
        assert continuation_est is not None and exercise_time is not None and immediate_payoff_slice is not None
        exercise_mask = np.zeros((n_steps - 1, n_paths), dtype=bool)
        valid = exercise_time >= 0
        exercise_mask[exercise_time[valid], np.where(valid)[0]] = True
        diagnostics = {
            "paths": paths,
            "time_grid": np.linspace(0.0, maturity, n_steps),
            "exercise_mask": exercise_mask,
            "exercise_time": exercise_time,
            "immediate_payoff": immediate_payoff_slice,
            "continuation_estimate": continuation_est,
        }

    mask_return = mask_to_use.copy() if capture_mask else None
    return cashflow, mask_return, diagnostics

class MonteCarloPricing:
    def __init__(self, S_0: float | np.ndarray, X: float, sigma: float | np.ndarray, T: float,
                 r: float | np.ndarray = None, mu: float | np.ndarray = None, num_paths: int = 1000,
                 steps: int = 252, *, div: float | np.ndarray = 0.0, corr: np.ndarray | None = None,
                 rng: np.random.Generator | None = None, seed: int | None = None):
        s0_arr = np.asarray(S_0, dtype=float)
        if s0_arr.ndim == 0:
            self._s0_vec = np.array([float(s0_arr)], dtype=float)
            self.S_0 = float(s0_arr)
        elif s0_arr.ndim == 1:
            if s0_arr.size == 0:
                raise ValueError("S_0 cannot be empty.")
            self._s0_vec = s0_arr.astype(float, copy=False)
            self.S_0 = self._s0_vec.copy()
        else:
            raise ValueError("S_0 must be a scalar or 1D array.")

        self.n_assets = int(self._s0_vec.size)
        self.X = X
        self._sigma_vec = _coerce_asset_vector(sigma, n_assets=self.n_assets, name="sigma")
        self.sigma = float(self._sigma_vec[0]) if self.n_assets == 1 else self._sigma_vec.copy()
        self.r = r  # Risk Free rate
        self.mu = mu  # Real-World Drift
        self.T = T # Time to maturity in years
        self.num_paths = num_paths
        self.steps = steps
        self._div_vec = _coerce_asset_vector(div, n_assets=self.n_assets, name="div")
        self.div = float(self._div_vec[0]) if self.n_assets == 1 else self._div_vec.copy()

        if corr is None:
            corr_arr = np.eye(self.n_assets, dtype=float)
        else:
            corr_arr = np.asarray(corr, dtype=float)
            if corr_arr.shape != (self.n_assets, self.n_assets):
                raise ValueError(
                    f"corr must have shape ({self.n_assets}, {self.n_assets}), "
                    f"received {corr_arr.shape}."
                )
        self._corr = corr_arr
        self._chol_corr = np.linalg.cholesky(self._corr)

        # If a seed is supplied (and no custom rng), remember it so each pricing call can
        # reset to the same random stream for deterministic results.
        self._seed = None if rng is not None else seed
        self.rng = rng if rng is not None else np.random.default_rng(seed)

    def _reset_rng(self) -> None:
        """Recreate the RNG from the stored seed when one was provided."""
        if self._seed is not None:
            self.rng = np.random.default_rng(self._seed)

    def _risk_free_scalar(self) -> float:
        if self.r is None:
            raise ValueError("Risk-free rate r must be set before pricing under the risk-neutral measure.")
        r_arr = np.asarray(self.r, dtype=float)
        if r_arr.ndim == 0:
            return float(r_arr)
        if r_arr.ndim == 1 and r_arr.size == 1:
            return float(r_arr[0])
        raise ValueError("Pricing routines require scalar risk-free rate r.")

    def _simulate_paths(self, risk_neutral: bool = True, Z: np.ndarray | None = None, 
                        *, antithetic: bool = False) -> np.ndarray:
        """Simulate stock prices over time using Geometric Brownian Motion."""
        num_paths = self.num_paths
        num_steps = self.steps
        dt = self.T / self.steps # Step size

        if Z is None:
            self._reset_rng()
            if antithetic:
                half_paths = (num_paths + 1) // 2
                Z_half = self.rng.standard_normal(size=(num_steps, half_paths, self.n_assets))
                Z = np.concatenate((Z_half, -Z_half), axis=1)[:, :num_paths, :]
            else:
                Z = self.rng.standard_normal(size=(num_steps, num_paths, self.n_assets))
        else:
            Z = np.asarray(Z, dtype=float)
            if Z.ndim == 2:
                if self.n_assets != 1:
                    raise ValueError(
                        "For multi-asset simulation, Z must have shape (steps, paths, assets)."
                    )
                if Z.shape != (num_steps, num_paths):
                    raise ValueError(
                        f"Z must have shape {(num_steps, num_paths)} for single-asset simulation."
                    )
                Z = Z[:, :, None]
            elif Z.ndim == 3:
                if Z.shape != (num_steps, num_paths, self.n_assets):
                    raise ValueError(
                        f"Z must have shape {(num_steps, num_paths, self.n_assets)}."
                    )
            else:
                raise ValueError("Z must be a 2D or 3D array.")

        drift_param = self.r if risk_neutral else self.mu
        if drift_param is None:
            raise ValueError("Set r for risk-neutral or mu for real-world simulations before calling _simulate_paths")
        drift_vec = _coerce_asset_vector(drift_param, n_assets=self.n_assets, name="drift")

        correlated_z = np.einsum("tpa,ab->tpb", Z, self._chol_corr.T)
        log_returns = (
            (drift_vec - self._div_vec - 0.5 * self._sigma_vec ** 2)[None, None, :] * dt
            + self._sigma_vec[None, None, :] * np.sqrt(dt) * correlated_z
        )

        S = np.empty((num_steps + 1, num_paths, self.n_assets), dtype=float)
        S[0, :, :] = self._s0_vec[None, :]
        S[1:] = self._s0_vec[None, None, :] * np.exp(np.cumsum(log_returns, axis=0))

        if self.n_assets == 1:
            return S[:, :, 0]
        return S

    def simulate_paths(self, risk_neutral: bool = True, *, antithetic: bool = True) -> np.ndarray:
        """Generate GBM paths; antithetic variates are on by default for variance reduction."""
        return self._simulate_paths(risk_neutral=risk_neutral, antithetic=antithetic)

    def plot_paths(self, num_plots: int = 1, call: bool = True, *, antithetic: bool = True):
        if self.n_assets != 1:
            raise ValueError("plot_paths currently supports single-asset simulations only.")
        import matplotlib.pyplot as plt
        paths = self._simulate_paths(antithetic=antithetic) # Antithetic Variates Method used while plotting
        plt.figure(figsize=(12, 8))

        if num_plots > 1:
            for i in range(min(num_plots, self.num_paths)):
                plt.plot(paths[:, i], lw=1, alpha=0.7)

        else:
            plt.plot(paths[:, 0], lw=2)
            S_T = paths[-1, 0]
            plt.scatter(len(paths) - 1, S_T, color='red', s=10, zorder=5, label=f'{paths[-1, 0]:.2f}')
            plt.hlines(self.X, 0, self.steps, label='Strike', color='orange', linestyle='-')

            # payoff = max(S_T - self.X, 0) if call else max(self.X - S_T, 0)

            pl = S_T - self.X if call else self.X - S_T

            if pl < 0:
                plt.vlines(self.steps, S_T, self.X, color='red', label=f'{pl:.2f}')
            else:
                plt.vlines(self.steps, self.X, S_T, color='green', label=f'+{pl:.2f}')

        plt.title(f'Monte Carlo Simulation for {num_plots} Paths')
        plt.xlabel('Step')
        plt.ylabel('Stock Price')
        plt.grid(True)
        plt.legend()
        plt.show()

    def european(self, call: bool = True, *, antithetic: bool = True,
                 payoff_fn: PayoffFn | None = None) -> tuple[float, float]:
        """Price a European option via Monte Carlo (returns mean price and standard error)."""
        r_scalar = self._risk_free_scalar()

        paths = self._simulate_paths(antithetic=antithetic)
        terminal_states = paths[-1]
        if payoff_fn is None:
            if self.n_assets != 1:
                raise ValueError(
                    "Multi-asset European pricing requires payoff_fn. "
                    "For example: payoff_fn=lambda s: np.maximum(s @ w - K, 0.0)."
                )
            if call:
                payoffs = np.maximum(terminal_states - self.X, 0)
            else:
                payoffs = np.maximum(self.X - terminal_states, 0)
        else:
            payoffs = np.asarray(payoff_fn(terminal_states), dtype=float).reshape(-1)
            if payoffs.shape[0] != self.num_paths:
                raise ValueError(
                    f"payoff_fn must return one payoff per path ({self.num_paths})."
                )

        discounted = np.exp(-r_scalar * self.T) * payoffs
        return np.mean(discounted), np.std(discounted) / np.sqrt(self.num_paths)

    def american(self, call: bool = True, basis_fn: BasisName = "laguerre", *, antithetic: bool = True,
                 include_all_paths: bool = True, mask_tolerance: float = 0.0,
                 continuation_mask: Optional[np.ndarray] = None, paths: Optional[np.ndarray] = None,
                 return_diagnostics: bool = False, return_mask: bool = False,
                 nn_kwargs: Optional[dict] = None, payoff_fn: PayoffFn | None = None,
                 state_fn: StateFn | None = None, rlsm_kwargs: Optional[dict] = None,
                 train_eval_split: float | None = None) -> tuple:
        """Price an American option using the Least Squares Monte Carlo method.

        Parameters beyond the classic signature provide hooks for advanced workflows:
        - include_all_paths/mask_tolerance control the regression mask construction.
        - continuation_mask allows supplying a precomputed ITM mask (e.g., from an unbumped run).
        - paths may be pre-simulated using shared random draws.
        - return_mask surfaces the regression mask alongside the diagnostics if requested.
        - payoff_fn defines intrinsic values from simulated states; required for multi-asset payoffs.
        - state_fn maps simulated states to regression features used for continuation fitting.
        """

        r_scalar = self._risk_free_scalar()

        if paths is None:
            paths = self._simulate_paths(antithetic=antithetic)
        n_steps = paths.shape[0]
        n_paths = paths.shape[1]

        basis = basis_fn.lower() if isinstance(basis_fn, str) else basis_fn
        config = LSMConfig(
            basis=basis,
            include_all_paths=include_all_paths,
            mask_tolerance=mask_tolerance,
            nn_kwargs=nn_kwargs,
            rlsm_kwargs=rlsm_kwargs,
            seed=self._seed,
            train_eval_split=train_eval_split,
        )
        capture_diag = return_diagnostics
        capture_mask = return_mask
        cashflow, mask_out, diagnostics = _lsm_cashflows(paths, strike=self.X, call=call, rate=r_scalar, maturity=self.T,
                                                         config=config, mask=continuation_mask, capture_mask=capture_mask,
                                                         capture_diagnostics=capture_diag, payoff_fn=payoff_fn,
                                                         state_fn=state_fn)
        if train_eval_split is not None:
            sample = _eval_slice(cashflow, train_eval_split)
        elif basis == "rlsm":
            sample = _eval_slice(cashflow, float((rlsm_kwargs or {}).get("train_eval_split", 0.5)))
        else:
            sample = np.asarray(cashflow, dtype=float)
        ddof = 1 if sample.size > 1 else 0
        price = float(np.mean(sample))
        stderr = float(np.std(sample, ddof=ddof) / np.sqrt(sample.size))
        result: list = [price, stderr]
        if return_diagnostics:
            result.append(diagnostics if diagnostics is not None else {})
        if return_mask:
            if mask_out is None:
                payoff = _evaluate_payoff_grid(paths, strike=self.X, call=call, payoff_fn=payoff_fn)
                mask_out = _build_continuation_mask(
                    paths,
                    payoff,
                    strike=self.X,
                    include_all=include_all_paths,
                    tolerance=mask_tolerance,
                    state_fn=state_fn,
                )
            result.append(mask_out)
        return tuple(result)

    def american_cashflows(self, paths: np.ndarray, *, call: bool = True, basis_fn: BasisName = "laguerre",
                            include_all_paths: bool = True, mask_tolerance: float = 0.0,
                            mask: Optional[np.ndarray] = None, return_mask: bool = False,
                            return_diagnostics: bool = False, nn_kwargs: Optional[dict] = None,
                            payoff_fn: PayoffFn | None = None, state_fn: StateFn | None = None,
                            rlsm_kwargs: Optional[dict] = None,
                            train_eval_split: float | None = None):
        """Direct access to discounted American cashflows for custom workflows."""

        r_scalar = self._risk_free_scalar()

        basis = basis_fn.lower() if isinstance(basis_fn, str) else basis_fn
        config = LSMConfig(
            basis=basis,
            include_all_paths=include_all_paths,
            mask_tolerance=mask_tolerance,
            nn_kwargs=nn_kwargs,
            rlsm_kwargs=rlsm_kwargs,
            seed=self._seed,
            train_eval_split=train_eval_split,
        )
        capture_mask = return_mask
        capture_diag = return_diagnostics
        cashflow, mask_out, diagnostics = _lsm_cashflows(paths, strike=self.X, call=call, rate=r_scalar, maturity=self.T,
                                                         config=config, mask=mask, capture_mask=capture_mask,
                                                         capture_diagnostics=capture_diag, payoff_fn=payoff_fn,
                                                         state_fn=state_fn)
        outputs: list = [cashflow]
        if return_diagnostics:
            outputs.append(diagnostics if diagnostics is not None else {})
        if return_mask:
            if mask_out is None:
                payoff = _evaluate_payoff_grid(paths, strike=self.X, call=call, payoff_fn=payoff_fn)
                mask_out = _build_continuation_mask(
                    paths,
                    payoff,
                    strike=self.X,
                    include_all=include_all_paths,
                    tolerance=mask_tolerance,
                    state_fn=state_fn,
                )
            outputs.append(mask_out)
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def fit_american_policy(
        self,
        *,
        call: bool = True,
        basis_fn: BasisName = "laguerre",
        include_all_paths: bool = True,
        mask_tolerance: float = 0.0,
        paths: Optional[np.ndarray] = None,
        antithetic: bool = False,
        rlsm_kwargs: Optional[dict] = None,
        train_eval_split: float | None = None,
    ) -> Any:
        """Fit and return a reusable continuation policy."""
        r_scalar = self._risk_free_scalar()
        if paths is None:
            paths = self._simulate_paths(antithetic=antithetic)

        basis = basis_fn.lower() if isinstance(basis_fn, str) else basis_fn
        if basis in {"laguerre", "monomial", "hermite", "paper_poly2"}:
            fit_paths = paths
            if train_eval_split is not None:
                split_f = float(train_eval_split)
                if not (0.0 < split_f < 1.0):
                    raise ValueError("train_eval_split must be strictly between 0 and 1.")
                n_paths = paths.shape[1]
                if n_paths < 2:
                    raise ValueError("train_eval_split requires at least 2 paths.")
                train_count = max(1, min(int(n_paths * split_f), n_paths - 1))
                fit_paths = paths[:, :train_count, ...]
            fit = fit_lsm_policy_from_paths(
                fit_paths,
                strike=self.X,
                rate=r_scalar,
                maturity=self.T,
                call=call,
                basis_name=basis,  # type: ignore[arg-type]
                include_all_paths=include_all_paths,
                mask_tolerance=mask_tolerance,
            )
            return fit.policy
        if basis == "rlsm":
            kwargs = {} if rlsm_kwargs is None else dict(rlsm_kwargs)
            if train_eval_split is not None:
                kwargs["train_eval_split"] = float(train_eval_split)
            fit = fit_rlsm_policy_from_paths(
                paths,
                strike=self.X,
                rate=r_scalar,
                maturity=self.T,
                call=call,
                **kwargs,
            )
            return fit.policy
        raise ValueError(
            "fit_american_policy supports basis_fn in "
            "{'laguerre','monomial','hermite','paper_poly2','rlsm'}."
        )

    def evaluate_american_policy(self, policy: Any, paths: np.ndarray) -> np.ndarray:
        """Evaluate discounted cashflows for a pre-fitted policy."""
        if isinstance(policy, LSMContinuationPolicy):
            _, _, cashflows = evaluate_lsm_policy(policy, paths)
            return cashflows
        if isinstance(policy, RLSMPolicy):
            _, _, cashflows = evaluate_rlsm_policy(policy, paths)
            return cashflows
        raise TypeError("Unsupported policy type for evaluate_american_policy().")

    def plot_american_exercise(self, call: bool = True, basis_fn: BasisName = "laguerre", *, antithetic: bool = True,
                               max_paths: int | None = 500, show_boundary: bool = True,
                               figsize: tuple[float, float] = (10, 6),
                               nn_kwargs: Optional[dict] = None):
        """Plot simulated paths with optimal exercise decisions highlighted."""
        if self.n_assets != 1:
            raise ValueError("plot_american_exercise currently supports single-asset simulations only.")
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("plot_american_exercise requires matplotlib to be installed.") from exc

        price, stderr, diagnostics = self.american(call=call, basis_fn=basis_fn, antithetic=antithetic,
                                                   return_diagnostics=True, nn_kwargs=nn_kwargs)

        paths = diagnostics["paths"]
        exercise_mask = diagnostics["exercise_mask"]
        n_steps, n_paths = paths.shape
        step_grid = np.arange(n_steps)

        if max_paths is None or max_paths >= n_paths:
            path_indices = np.arange(n_paths)
        else:
            path_indices = np.arange(max_paths)

        plt.figure(figsize=figsize)
        for idx in path_indices:
            plt.plot(step_grid, paths[:, idx], color="gray", alpha=0.15, linewidth=0.8)

        mask_subset = exercise_mask[:, path_indices]
        if mask_subset.any():
            t_idx, p_idx = np.nonzero(mask_subset)
            plt.scatter(t_idx, paths[t_idx, path_indices[p_idx]],
                        c="red", s=20, alpha=0.6, label="Exercise decision")

        if show_boundary:
            boundary = np.full(exercise_mask.shape[0], np.nan)
            for t in range(exercise_mask.shape[0]):
                exercised_states = paths[t, exercise_mask[t]]
                if exercised_states.size:
                    boundary[t] = np.mean(exercised_states)
            if not np.all(np.isnan(boundary)):
                plt.plot(step_grid[:-1], boundary, color="red", linewidth=2.0, label="Average exercise level")

        plt.axhline(self.X, color="orange", linestyle="--", linewidth=1.2, label="Strike")
        plt.title("American option exercise profile")
        plt.xlabel("Step")
        plt.ylabel("Underlying price")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        return price, stderr

__all__ = [
    "BasisName",
    "PolicyBasisName",
    "PayoffFn",
    "StateFn",
    "LSMConfig",
    "LSMContinuationPolicy",
    "LSMPolicyFit",
    "immediate_payoff",
    "build_mask",
    "lsm_basis",
    "fit_lsm_policy_from_paths",
    "evaluate_lsm_policy",
    "MonteCarloPricing",
]
