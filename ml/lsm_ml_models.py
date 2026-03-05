"""Machine-learning models for American option pricing with LSM."""

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from engines.monte_carlo import MonteCarloPricing, build_mask, immediate_payoff

PayoffFn = Callable[[np.ndarray], np.ndarray]
StateFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class LSMModelResult:
    price: float
    stderr: float
    diagnostics: dict[str, Any]


def _simulate_paths(*, S0: float | np.ndarray, K: float, r: float, sigma: float | np.ndarray, T: float,
                    num_paths: int, steps: int, seed: int | None,
                    antithetic: bool, corr: np.ndarray | None = None,
                    div: float | np.ndarray = 0.0) -> tuple[MonteCarloPricing, np.ndarray]:
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
    return pricer, paths


def _to_result(cashflow: np.ndarray, diagnostics: dict[str, Any]) -> LSMModelResult:
    n_paths = cashflow.shape[0]
    ddof = 1 if n_paths > 1 else 0
    price = float(np.mean(cashflow))
    stderr = float(np.std(cashflow, ddof=ddof) / np.sqrt(n_paths))
    return LSMModelResult(price=price, stderr=stderr, diagnostics=diagnostics)


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


def price_american_random_forest(*, S0: float | np.ndarray, K: float, r: float, sigma: float | np.ndarray, T: float,
                                 num_paths: int, steps: int, call: bool = True, seed: int | None = None,
                                 include_all_paths: bool = False, mask_tolerance: float = 0.0,
                                 antithetic: bool = False, min_samples: int = 8,
                                 rf_kwargs: dict[str, Any] | None = None, corr: np.ndarray | None = None,
                                 div: float | np.ndarray = 0.0, payoff_fn: PayoffFn | None = None,
                                 state_fn: StateFn | None = None) -> LSMModelResult:
    """American option pricing via LSM + RandomForest continuation regression."""
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1.")

    _, paths = _simulate_paths(S0=S0, K=K, r=r, sigma=sigma, T=T, num_paths=num_paths,
                               steps=steps, seed=seed, antithetic=antithetic, corr=corr, div=div)
    n_steps = paths.shape[0]
    if n_steps < 2:
        raise ValueError("Need at least one time step for American valuation.")

    dt = T / (n_steps - 1)
    discount = np.exp(-r * dt)
    payoff = _payoff_grid(paths, strike=K, call=call, payoff_fn=payoff_fn)
    cashflow = payoff[-1].copy()
    if payoff_fn is None and state_fn is None:
        mask = build_mask(paths, strike=K, call=call, include_all=include_all_paths, tolerance=mask_tolerance)
    else:
        mask = _continuation_mask(
            paths,
            payoff,
            strike=K,
            include_all_paths=include_all_paths,
            mask_tolerance=mask_tolerance,
            state_fn=state_fn,
        )

    model_kwargs = {"n_estimators": 200, "random_state": seed, "n_jobs": -1}
    if rf_kwargs:
        model_kwargs.update(rf_kwargs)

    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []

    for t in range(n_steps - 2, -1, -1):
        include = mask[t]
        if np.any(include):
            states = _feature_matrix(paths[t, include], state_fn=state_fn)
            continuation_targets = cashflow[include] * discount
            exercise = payoff[t, include]

            if states.shape[0] < min_samples:
                continuation = np.full_like(continuation_targets, np.mean(continuation_targets))
            else:
                model = RandomForestRegressor(**model_kwargs)
                model.fit(states, continuation_targets)
                continuation = model.predict(states)

            y_true_all.append(continuation_targets)
            y_pred_all.append(continuation)
            exercise_now = (exercise > 0.0) & (exercise > continuation)
            idx = np.where(include)[0]
            cashflow[idx] = np.where(exercise_now, exercise, cashflow[idx] * discount)
        cashflow[~include] *= discount

    diagnostics = {
        "model": "random_forest",
        "continuation_true": np.concatenate(y_true_all) if y_true_all else np.array([], dtype=float),
        "continuation_pred": np.concatenate(y_pred_all) if y_pred_all else np.array([], dtype=float),
    }
    return _to_result(cashflow, diagnostics)


def price_american_svm(*, S0: float | np.ndarray, K: float, r: float, sigma: float | np.ndarray, T: float,
                       num_paths: int, steps: int, call: bool = True, seed: int | None = None,
                       include_all_paths: bool = False, mask_tolerance: float = 0.0,
                       antithetic: bool = False, scale_inputs: bool = True,
                       min_samples: int = 8, svm_kwargs: dict[str, Any] | None = None,
                       corr: np.ndarray | None = None, div: float | np.ndarray = 0.0,
                       payoff_fn: PayoffFn | None = None, state_fn: StateFn | None = None) -> LSMModelResult:
    """American option pricing via LSM + Support Vector Regression."""
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1.")

    _, paths = _simulate_paths(S0=S0, K=K, r=r, sigma=sigma, T=T, num_paths=num_paths,
                               steps=steps, seed=seed, antithetic=antithetic, corr=corr, div=div)
    n_steps = paths.shape[0]
    if n_steps < 2:
        raise ValueError("Need at least one time step for American valuation.")

    dt = T / (n_steps - 1)
    discount = np.exp(-r * dt)
    payoff = _payoff_grid(paths, strike=K, call=call, payoff_fn=payoff_fn)
    cashflow = payoff[-1].copy()
    if payoff_fn is None and state_fn is None:
        mask = build_mask(paths, strike=K, call=call, include_all=include_all_paths, tolerance=mask_tolerance)
    else:
        mask = _continuation_mask(
            paths,
            payoff,
            strike=K,
            include_all_paths=include_all_paths,
            mask_tolerance=mask_tolerance,
            state_fn=state_fn,
        )

    model_kwargs = {"kernel": "rbf", "C": 10.0, "epsilon": 0.05, "gamma": "scale"}
    if svm_kwargs:
        model_kwargs.update(svm_kwargs)

    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []

    for t in range(n_steps - 2, -1, -1):
        include = mask[t]
        if np.any(include):
            states = _feature_matrix(paths[t, include], state_fn=state_fn)
            continuation_targets = cashflow[include] * discount
            exercise = payoff[t, include]

            if states.shape[0] < min_samples:
                continuation = np.full_like(continuation_targets, np.mean(continuation_targets))
            else:
                if scale_inputs:
                    model = make_pipeline(StandardScaler(), SVR(**model_kwargs))
                else:
                    model = SVR(**model_kwargs)
                model.fit(states, continuation_targets)
                continuation = model.predict(states)

            y_true_all.append(continuation_targets)
            y_pred_all.append(continuation)
            exercise_now = (exercise > 0.0) & (exercise > continuation)
            idx = np.where(include)[0]
            cashflow[idx] = np.where(exercise_now, exercise, cashflow[idx] * discount)
        cashflow[~include] *= discount

    diagnostics = {
        "model": "svm",
        "continuation_true": np.concatenate(y_true_all) if y_true_all else np.array([], dtype=float),
        "continuation_pred": np.concatenate(y_pred_all) if y_pred_all else np.array([], dtype=float),
    }
    return _to_result(cashflow, diagnostics)


def price_american_logistic_regression(*, S0: float | np.ndarray, K: float, r: float,
                                       sigma: float | np.ndarray, T: float,
                                       num_paths: int, steps: int, call: bool = True, seed: int | None = None,
                                       include_all_paths: bool = False, mask_tolerance: float = 0.0,
                                       antithetic: bool = False, scale_inputs: bool = True,
                                       min_samples: int = 8, prob_threshold: float = 0.5,
                                       logistic_kwargs: dict[str, Any] | None = None,
                                       corr: np.ndarray | None = None, div: float | np.ndarray = 0.0,
                                       payoff_fn: PayoffFn | None = None,
                                       state_fn: StateFn | None = None) -> LSMModelResult:
    """American option pricing via LSM + logistic-regression exercise policy."""
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1.")
    if not (0.0 <= prob_threshold <= 1.0):
        raise ValueError("prob_threshold must be in [0, 1].")

    _, paths = _simulate_paths(S0=S0, K=K, r=r, sigma=sigma, T=T, num_paths=num_paths,
                               steps=steps, seed=seed, antithetic=antithetic, corr=corr, div=div)
    n_steps = paths.shape[0]
    if n_steps < 2:
        raise ValueError("Need at least one time step for American valuation.")

    dt = T / (n_steps - 1)
    discount = np.exp(-r * dt)
    payoff = _payoff_grid(paths, strike=K, call=call, payoff_fn=payoff_fn)
    cashflow = payoff[-1].copy()
    if payoff_fn is None and state_fn is None:
        mask = build_mask(paths, strike=K, call=call, include_all=include_all_paths, tolerance=mask_tolerance)
    else:
        mask = _continuation_mask(
            paths,
            payoff,
            strike=K,
            include_all_paths=include_all_paths,
            mask_tolerance=mask_tolerance,
            state_fn=state_fn,
        )

    model_kwargs = {"solver": "lbfgs", "max_iter": 1000, "random_state": seed}
    if logistic_kwargs:
        model_kwargs.update(logistic_kwargs)

    y_true_all: list[np.ndarray] = []
    y_prob_all: list[np.ndarray] = []

    for t in range(n_steps - 2, -1, -1):
        include = mask[t]
        if np.any(include):
            states = _feature_matrix(paths[t, include], state_fn=state_fn)
            continuation_targets = cashflow[include] * discount
            exercise = payoff[t, include]

            labels = ((exercise > 0.0) & (exercise > continuation_targets)).astype(int)
            features = np.column_stack((states, exercise.reshape(-1, 1)))

            if features.shape[0] < min_samples:
                probs = labels.astype(float)
            elif np.unique(labels).size == 1:
                probs = np.full(labels.shape[0], float(labels[0]))
            else:
                if scale_inputs:
                    model = make_pipeline(StandardScaler(), LogisticRegression(**model_kwargs))
                else:
                    model = LogisticRegression(**model_kwargs)
                model.fit(features, labels)
                probs = model.predict_proba(features)[:, 1]

            y_true_all.append(labels)
            y_prob_all.append(probs)
            exercise_now = (exercise > 0.0) & (probs >= prob_threshold)
            idx = np.where(include)[0]
            cashflow[idx] = np.where(exercise_now, exercise, cashflow[idx] * discount)
        cashflow[~include] *= discount

    diagnostics = {
        "model": "logistic_regression",
        "exercise_label_true": np.concatenate(y_true_all) if y_true_all else np.array([], dtype=int),
        "exercise_prob_pred": np.concatenate(y_prob_all) if y_prob_all else np.array([], dtype=float),
    }
    return _to_result(cashflow, diagnostics)


__all__ = [
    "LSMModelResult",
    "price_american_logistic_regression",
    "price_american_random_forest",
    "price_american_svm",
]
