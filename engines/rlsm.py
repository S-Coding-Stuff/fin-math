"""Randomized Least Squares Monte Carlo (RLSM) for American options.

Implements the paper-style random-feature continuation approximation where
hidden-layer parameters are sampled once (or per step for RLSMreinit) and only
the last layer is fitted with linear regression.
"""
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

ActivationName = Literal[
    "tanh",
    "relu",
    "leaky_relu",
    "sigmoid",
    "softplus",
    "identity",
    "linear",
    "gelu",
    "silu",
    "elu",
]
WeightDistName = Literal["normal", "gaussian"]
PayoffFn = Callable[[np.ndarray], np.ndarray]
StateFn = Callable[[np.ndarray], np.ndarray]


def _as_path_cube(paths: np.ndarray) -> np.ndarray:
    arr = np.asarray(paths, dtype=float)
    if arr.ndim == 2:
        return arr[:, :, None]
    if arr.ndim == 3:
        return arr
    raise ValueError("paths must be a 2D (time, paths) or 3D (time, paths, assets) array.")


def _default_payoff(paths_cube: np.ndarray, *, strike: float, call: bool) -> np.ndarray:
    if paths_cube.shape[2] != 1:
        raise ValueError(
            "Default RLSM payoff supports single-asset paths only. "
            "For multi-asset paths, pass payoff_fn to fit_rlsm_policy_from_paths()."
        )
    s = paths_cube[:, :, 0]
    if call:
        return np.maximum(s - strike, 0.0)
    return np.maximum(strike - s, 0.0)


def _evaluate_payoff_grid(
    paths_cube: np.ndarray,
    *,
    strike: float,
    call: bool,
    payoff_fn: PayoffFn | None,
) -> np.ndarray:
    if payoff_fn is None:
        return _default_payoff(paths_cube, strike=strike, call=call)
    payoff = np.asarray(payoff_fn(paths_cube), dtype=float)
    expected = (paths_cube.shape[0], paths_cube.shape[1])
    if payoff.shape != expected:
        raise ValueError(
            "payoff_fn must return a 2D array with shape (time, paths); "
            f"received {payoff.shape}, expected {expected}."
        )
    return payoff


def _state_features(
    states: np.ndarray,
    *,
    state_fn: StateFn | None,
    payoff_values: np.ndarray | None = None,
    use_payoff_as_input: bool = False,
    expected_dim: int | None = None,
) -> np.ndarray:
    raw = np.asarray(states, dtype=float)
    if raw.ndim == 1:
        if expected_dim is not None and raw.size == expected_dim and expected_dim > 1:
            raw2 = raw.reshape(1, -1)
        else:
            raw2 = raw.reshape(-1, 1)
    elif raw.ndim == 2:
        raw2 = raw
    else:
        raise ValueError("states must be 1D or 2D.")

    feats = np.asarray(state_fn(raw2) if state_fn is not None else raw2, dtype=float)
    if feats.ndim == 1:
        if feats.shape[0] == raw2.shape[0]:
            feats = feats.reshape(-1, 1)
        elif raw2.shape[0] == 1:
            feats = feats.reshape(1, -1)
        else:
            raise ValueError("state_fn output must align with path count.")
    elif feats.ndim == 2:
        if feats.shape[0] != raw2.shape[0]:
            raise ValueError("state_fn output first dimension must equal path count.")
    else:
        raise ValueError("state_fn output must be 1D or 2D.")

    if use_payoff_as_input:
        if payoff_values is None:
            raise ValueError("payoff_values must be provided when use_payoff_as_input=True.")
        payoff_arr = np.asarray(payoff_values, dtype=float).reshape(-1, 1)
        if payoff_arr.shape[0] != feats.shape[0]:
            raise ValueError("payoff_values must align with path count.")
        feats = np.concatenate((feats, payoff_arr), axis=1)

    if expected_dim is not None and feats.shape[1] != expected_dim:
        raise ValueError(
            f"State feature dimension mismatch: expected {expected_dim}, received {feats.shape[1]}."
        )
    return feats


def _activation(x: np.ndarray, name: ActivationName, *, parameter: float = 1.0) -> np.ndarray:
    if name == "tanh":
        return np.tanh(x)
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "leaky_relu":
        return np.where(x >= 0.0, x, float(parameter) * x)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    if name == "softplus":
        beta = max(float(parameter), 1e-12)
        return np.log1p(np.exp(-np.abs(beta * x))) / beta + np.maximum(x, 0.0)
    if name == "gelu":
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x ** 3))))
    if name == "silu":
        return x / (1.0 + np.exp(-x))
    if name == "elu":
        alpha = float(parameter)
        return np.where(x > 0.0, x, alpha * (np.exp(x) - 1.0))
    if name in {"identity", "linear"}:
        return x
    raise ValueError(f"Unsupported activation '{name}'.")


def _sample_hidden_weights(
    *,
    rng: np.random.Generator,
    hidden_size: int,
    input_dim: int,
    weight_dist: WeightDistName,
    weight_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    if weight_dist not in {"normal", "gaussian"}:
        raise ValueError("weight_dist must be one of {'normal', 'gaussian'}.")
    A = rng.normal(loc=0.0, scale=weight_scale, size=(hidden_size, input_dim))
    b = rng.normal(loc=0.0, scale=weight_scale, size=(hidden_size,))
    return A.astype(float, copy=False), b.astype(float, copy=False)


def _feature_matrix(
    states: np.ndarray,
    *,
    A: np.ndarray,
    b: np.ndarray,
    activation: ActivationName,
    activation_parameter: float = 1.0,
    input_scale: float = 1.0,
) -> np.ndarray:
    x = np.asarray(states, dtype=float).reshape(-1, A.shape[1]) * float(input_scale)
    hidden = _activation(x @ A.T + b[None, :], activation, parameter=activation_parameter)
    ones = np.ones((hidden.shape[0], 1), dtype=float)
    return np.concatenate((hidden, ones), axis=1)


def _solve_last_layer(phi: np.ndarray, targets: np.ndarray, *, ridge_lambda: float) -> np.ndarray:
    if phi.ndim != 2:
        raise ValueError("phi must be a 2D array.")
    y = np.asarray(targets, dtype=float).reshape(-1)
    if phi.shape[0] != y.shape[0]:
        raise ValueError("phi and targets must have the same number of rows.")

    gram = phi.T @ phi
    if ridge_lambda > 0.0:
        gram = gram + ridge_lambda * np.eye(gram.shape[0], dtype=float)
    rhs = phi.T @ y
    try:
        theta = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        theta = np.linalg.pinv(gram) @ rhs
    return theta.astype(float, copy=False)


@dataclass
class RLSMPolicy:
    call: bool
    strike: float
    rate: float
    maturity: float
    steps: int
    hidden_size: int
    activation: ActivationName
    weight_dist: WeightDistName
    weight_scale: float
    ridge_lambda: float
    reinit_per_step: bool
    input_dim: int
    train_indices: np.ndarray
    eval_indices: np.ndarray
    weights_by_step: list[tuple[np.ndarray, np.ndarray]]
    theta_by_step: list[np.ndarray]
    fallback_by_step: list[float]
    activation_parameter: float = 1.0
    input_scale: float = 1.0
    use_payoff_as_input: bool = False
    payoff_fn: PayoffFn | None = None
    state_fn: StateFn | None = None

    @property
    def dt(self) -> float:
        return self.maturity / self.steps

    @property
    def discount_grid(self) -> np.ndarray:
        return np.exp(-self.rate * self.dt * np.arange(self.steps + 1))

    def _continuation_from_features(self, t: int, features: np.ndarray) -> np.ndarray:
        if t >= self.steps:
            return np.zeros(features.shape[0], dtype=float)
        if t < 0:
            raise ValueError("t must be >= 0.")
        A, b = self.weights_by_step[t]
        theta = self.theta_by_step[t]
        if theta.size == 0:
            return np.full(features.shape[0], self.fallback_by_step[t], dtype=float)
        phi = _feature_matrix(
            features,
            A=A,
            b=b,
            activation=self.activation,
            activation_parameter=self.activation_parameter,
            input_scale=self.input_scale,
        )
        return phi @ theta

    def continuation_value(self, t: int, states: np.ndarray, payoff_values: np.ndarray | None = None) -> np.ndarray:
        features = _state_features(
            states,
            state_fn=self.state_fn,
            payoff_values=payoff_values,
            use_payoff_as_input=self.use_payoff_as_input,
            expected_dim=self.input_dim,
        )
        return self._continuation_from_features(t, features)

    def stopping_times(self, paths: np.ndarray) -> np.ndarray:
        cube = _as_path_cube(paths)
        payoff = _evaluate_payoff_grid(cube, strike=self.strike, call=self.call, payoff_fn=self.payoff_fn)
        n_times, n_paths, _ = cube.shape
        tau = np.full(n_paths, n_times - 1, dtype=int)
        alive = np.ones(n_paths, dtype=bool)
        for t in range(n_times - 1):
            if not np.any(alive):
                break
            idx = np.where(alive)[0]
            exercise = payoff[t, idx]
            features = _state_features(
                cube[t, idx, :],
                state_fn=self.state_fn,
                payoff_values=exercise,
                use_payoff_as_input=self.use_payoff_as_input,
                expected_dim=self.input_dim,
            )
            continuation = self._continuation_from_features(t, features)
            exercise_now = (exercise > 0.0) & (exercise >= continuation)
            chosen = idx[exercise_now]
            tau[chosen] = t
            alive[chosen] = False
        return tau

    def discounted_cashflows(self, paths: np.ndarray) -> np.ndarray:
        cube = _as_path_cube(paths)
        payoff = _evaluate_payoff_grid(cube, strike=self.strike, call=self.call, payoff_fn=self.payoff_fn)
        tau = self.stopping_times(cube)
        return self.discount_grid[tau] * payoff[tau, np.arange(cube.shape[1])]

    def diagnostics(self, paths: np.ndarray) -> dict[str, np.ndarray]:
        cube = _as_path_cube(paths)
        payoff = _evaluate_payoff_grid(cube, strike=self.strike, call=self.call, payoff_fn=self.payoff_fn)
        n_times, n_paths, n_assets = cube.shape
        tau = self.stopping_times(cube)
        exercise_mask = np.zeros((n_times - 1, n_paths), dtype=bool)
        valid = tau < (n_times - 1)
        exercise_mask[tau[valid], np.where(valid)[0]] = True
        continuation = np.full((n_times - 1, n_paths), np.nan, dtype=float)
        for t in range(n_times - 1):
            continuation[t] = self.continuation_value(t, cube[t], payoff_values=payoff[t])
        paths_out = cube[:, :, 0] if n_assets == 1 else cube
        return {
            "paths": paths_out,
            "time_grid": np.linspace(0.0, self.maturity, n_times),
            "exercise_time": tau,
            "exercise_mask": exercise_mask,
            "immediate_payoff": payoff[:-1].copy(),
            "continuation_estimate": continuation,
            "train_indices": self.train_indices.copy(),
            "eval_indices": self.eval_indices.copy(),
        }


@dataclass
class RLSMFit:
    policy: RLSMPolicy
    stage1_estimate: float
    stage1_stderr: float
    train_count_by_step: np.ndarray
    fit_mse_by_step: np.ndarray
    cashflows: np.ndarray


def fit_rlsm_policy_from_paths(
    paths: np.ndarray,
    *,
    strike: float,
    rate: float,
    maturity: float,
    call: bool = False,
    hidden_size: int = 20,
    activation: ActivationName = "leaky_relu",
    weight_dist: WeightDistName = "normal",
    weight_scale: float = 1.0,
    ridge_lambda: float = 0.0,
    reinit_per_step: bool = False,
    seed: int | None = None,
    train_eval_split: float = 0.5,
    payoff_fn: PayoffFn | None = None,
    state_fn: StateFn | None = None,
    train_itm_only: bool = False,
    use_payoff_as_input: bool = False,
    factors: tuple[float, ...] = (1.0,),
    optstop_compatible: bool = False,
) -> RLSMFit:
    cube = _as_path_cube(paths)
    n_times, n_paths, _ = cube.shape
    if n_times < 2:
        raise ValueError("paths must include at least one exercise step.")
    if maturity <= 0.0:
        raise ValueError("maturity must be positive.")
    n_assets = cube.shape[2]
    if optstop_compatible and hidden_size < 0:
        hidden_size = 50 + abs(int(hidden_size)) * n_assets
    if hidden_size < 1:
        raise ValueError("hidden_size must be >= 1.")
    if not (0.0 < train_eval_split < 1.0):
        raise ValueError("train_eval_split must be strictly between 0 and 1.")
    if weight_scale <= 0.0:
        raise ValueError("weight_scale must be positive.")
    if ridge_lambda < 0.0:
        raise ValueError("ridge_lambda must be >= 0.")
    if len(factors) < 1:
        raise ValueError("factors must contain at least one value.")

    input_scale = float(factors[0])
    activation_parameter = 1.0
    if optstop_compatible:
        if activation == "leaky_relu":
            activation_parameter = input_scale / 2.0
        elif activation in {"softplus", "elu"} and len(factors) > 1:
            activation_parameter = float(factors[1])

    n_steps = n_times - 1
    dt = maturity / n_steps
    discount = float(np.exp(-rate * dt))
    payoff = _evaluate_payoff_grid(cube, strike=strike, call=call, payoff_fn=payoff_fn)

    train_count = int(n_paths * train_eval_split)
    train_count = max(1, min(train_count, n_paths - 1))
    train_indices = np.arange(train_count, dtype=int)
    eval_indices = np.arange(train_count, n_paths, dtype=int)
    if eval_indices.size == 0:
        raise ValueError("train_eval_split leaves no evaluation paths.")

    first_features = _state_features(
        cube[0, train_indices, :],
        state_fn=state_fn,
        payoff_values=payoff[0, train_indices],
        use_payoff_as_input=use_payoff_as_input,
    )
    input_dim = int(first_features.shape[1])

    rng = np.random.default_rng(seed)
    if reinit_per_step:
        weights_template: list[tuple[np.ndarray, np.ndarray]] = [
            _sample_hidden_weights(
                rng=rng,
                hidden_size=hidden_size,
                input_dim=input_dim,
                weight_dist=weight_dist,
                weight_scale=weight_scale,
            )
            for _ in range(n_steps)
        ]
    else:
        shared = _sample_hidden_weights(
            rng=rng,
            hidden_size=hidden_size,
            input_dim=input_dim,
            weight_dist=weight_dist,
            weight_scale=weight_scale,
        )
        weights_template = [shared for _ in range(n_steps)]

    cashflow = payoff[-1].copy()
    theta_by_step: list[np.ndarray] = [np.array([], dtype=float) for _ in range(n_steps)]
    fallback_by_step: list[float] = [0.0 for _ in range(n_steps)]
    train_count_by_step = np.full(n_steps, train_indices.size, dtype=int)
    fit_mse_by_step = np.full(n_steps, np.nan, dtype=float)

    for t in range(n_steps - 1, -1, -1):
        A, b = weights_template[t]
        selected_train_indices = train_indices
        if train_itm_only:
            selected_train_indices = train_indices[payoff[t, train_indices] > 0.0]
        train_count_by_step[t] = int(selected_train_indices.size)
        if selected_train_indices.size == 0:
            fallback_by_step[t] = 0.0
            theta_by_step[t] = np.array([], dtype=float)
            intrinsic = payoff[t]
            exercise_now = intrinsic > 0.0
            cashflow = np.where(exercise_now, intrinsic, cashflow * discount)
            continue

        train_features = _state_features(
            cube[t, selected_train_indices, :],
            state_fn=state_fn,
            payoff_values=payoff[t, selected_train_indices],
            use_payoff_as_input=use_payoff_as_input,
            expected_dim=input_dim,
        )
        phi_train = _feature_matrix(
            train_features,
            A=A,
            b=b,
            activation=activation,
            activation_parameter=activation_parameter,
            input_scale=input_scale,
        )
        target_train = cashflow[selected_train_indices] * discount
        theta = _solve_last_layer(phi_train, target_train, ridge_lambda=ridge_lambda)
        pred_train = phi_train @ theta
        fit_mse_by_step[t] = float(np.mean((pred_train - target_train) ** 2))
        fallback_by_step[t] = float(np.mean(target_train))
        theta_by_step[t] = theta

        all_features = _state_features(
            cube[t],
            state_fn=state_fn,
            payoff_values=payoff[t],
            use_payoff_as_input=use_payoff_as_input,
            expected_dim=input_dim,
        )
        phi_all = _feature_matrix(
            all_features,
            A=A,
            b=b,
            activation=activation,
            activation_parameter=activation_parameter,
            input_scale=input_scale,
        )
        continuation_all = phi_all @ theta
        intrinsic = payoff[t]
        exercise_now = (intrinsic > 0.0) & (intrinsic >= continuation_all)
        cashflow = np.where(exercise_now, intrinsic, cashflow * discount)

    eval_cash = cashflow[eval_indices]
    ddof = 1 if eval_cash.size > 1 else 0
    stage1_est = float(np.mean(eval_cash))
    stage1_stderr = float(np.std(eval_cash, ddof=ddof) / np.sqrt(eval_cash.size))

    policy = RLSMPolicy(
        call=call,
        strike=float(strike),
        rate=float(rate),
        maturity=float(maturity),
        steps=n_steps,
        hidden_size=hidden_size,
        activation=activation,
        weight_dist=weight_dist,
        weight_scale=float(weight_scale),
        ridge_lambda=float(ridge_lambda),
        reinit_per_step=bool(reinit_per_step),
        input_dim=input_dim,
        activation_parameter=float(activation_parameter),
        input_scale=float(input_scale),
        use_payoff_as_input=bool(use_payoff_as_input),
        train_indices=train_indices,
        eval_indices=eval_indices,
        weights_by_step=weights_template,
        theta_by_step=theta_by_step,
        fallback_by_step=fallback_by_step,
        payoff_fn=payoff_fn,
        state_fn=state_fn,
    )
    return RLSMFit(
        policy=policy,
        stage1_estimate=stage1_est,
        stage1_stderr=stage1_stderr,
        train_count_by_step=train_count_by_step,
        fit_mse_by_step=fit_mse_by_step,
        cashflows=cashflow,
    )


def evaluate_rlsm_policy(
    policy: RLSMPolicy,
    paths: np.ndarray,
    *,
    eval_only: bool = False,
) -> tuple[float, float, np.ndarray]:
    cashflows = policy.discounted_cashflows(paths)
    if eval_only:
        if policy.eval_indices.size == 0:
            raise ValueError("Policy does not define evaluation indices.")
        if np.max(policy.eval_indices) >= cashflows.size:
            raise ValueError("Evaluation indices are not compatible with provided paths.")
        sample = cashflows[policy.eval_indices]
    else:
        sample = cashflows
    ddof = 1 if sample.size > 1 else 0
    mean = float(np.mean(sample))
    stderr = float(np.std(sample, ddof=ddof) / np.sqrt(sample.size))
    return mean, stderr, cashflows


__all__ = [
    "ActivationName",
    "WeightDistName",
    "RLSMPolicy",
    "RLSMFit",
    "fit_rlsm_policy_from_paths",
    "evaluate_rlsm_policy",
]
