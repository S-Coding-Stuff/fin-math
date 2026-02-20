"""Machine-learning models for American option pricing with LSM."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from engines.monte_carlo import MonteCarloPricing, build_mask, immediate_payoff


@dataclass
class LSMModelResult:
    price: float
    stderr: float
    diagnostics: dict[str, Any]


def _simulate_paths(*, S0: float, K: float, r: float, sigma: float, T: float,
                    num_paths: int, steps: int, seed: int | None,
                    antithetic: bool) -> tuple[MonteCarloPricing, np.ndarray]:
    pricer = MonteCarloPricing(S_0=S0, X=K, sigma=sigma, T=T, r=r, num_paths=num_paths, steps=steps, seed=seed)
    paths = pricer._simulate_paths(antithetic=antithetic)
    return pricer, paths


def _to_result(cashflow: np.ndarray, diagnostics: dict[str, Any]) -> LSMModelResult:
    n_paths = cashflow.shape[0]
    ddof = 1 if n_paths > 1 else 0
    price = float(np.mean(cashflow))
    stderr = float(np.std(cashflow, ddof=ddof) / np.sqrt(n_paths))
    return LSMModelResult(price=price, stderr=stderr, diagnostics=diagnostics)


def price_american_random_forest(*, S0: float, K: float, r: float, sigma: float, T: float,
                                 num_paths: int, steps: int, call: bool = True, seed: int | None = None,
                                 include_all_paths: bool = False, mask_tolerance: float = 0.0,
                                 antithetic: bool = False, min_samples: int = 8,
                                 rf_kwargs: dict[str, Any] | None = None) -> LSMModelResult:
    """American option pricing via LSM + RandomForest continuation regression."""
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1.")

    _, paths = _simulate_paths(S0=S0, K=K, r=r, sigma=sigma, T=T, num_paths=num_paths,
                               steps=steps, seed=seed, antithetic=antithetic)
    n_steps, _ = paths.shape
    if n_steps < 2:
        raise ValueError("Need at least one time step for American valuation.")

    dt = T / (n_steps - 1)
    discount = np.exp(-r * dt)
    payoff = immediate_payoff(paths, strike=K, call=call)
    cashflow = payoff[-1].copy()
    mask = build_mask(paths, strike=K, call=call, include_all=include_all_paths, tolerance=mask_tolerance)

    model_kwargs = {"n_estimators": 200, "random_state": seed, "n_jobs": -1}
    if rf_kwargs:
        model_kwargs.update(rf_kwargs)

    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []

    for t in range(n_steps - 2, -1, -1):
        include = mask[t]
        if np.any(include):
            states = paths[t, include].reshape(-1, 1)
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


def price_american_svm(*, S0: float, K: float, r: float, sigma: float, T: float,
                       num_paths: int, steps: int, call: bool = True, seed: int | None = None,
                       include_all_paths: bool = False, mask_tolerance: float = 0.0,
                       antithetic: bool = False, scale_inputs: bool = True,
                       min_samples: int = 8, svm_kwargs: dict[str, Any] | None = None) -> LSMModelResult:
    """American option pricing via LSM + Support Vector Regression."""
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1.")

    _, paths = _simulate_paths(S0=S0, K=K, r=r, sigma=sigma, T=T, num_paths=num_paths,
                               steps=steps, seed=seed, antithetic=antithetic)
    n_steps, _ = paths.shape
    if n_steps < 2:
        raise ValueError("Need at least one time step for American valuation.")

    dt = T / (n_steps - 1)
    discount = np.exp(-r * dt)
    payoff = immediate_payoff(paths, strike=K, call=call)
    cashflow = payoff[-1].copy()
    mask = build_mask(paths, strike=K, call=call, include_all=include_all_paths, tolerance=mask_tolerance)

    model_kwargs = {"kernel": "rbf", "C": 10.0, "epsilon": 0.05, "gamma": "scale"}
    if svm_kwargs:
        model_kwargs.update(svm_kwargs)

    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []

    for t in range(n_steps - 2, -1, -1):
        include = mask[t]
        if np.any(include):
            states = paths[t, include].reshape(-1, 1)
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


def price_american_logistic_regression(*, S0: float, K: float, r: float, sigma: float, T: float,
                                       num_paths: int, steps: int, call: bool = True, seed: int | None = None,
                                       include_all_paths: bool = False, mask_tolerance: float = 0.0,
                                       antithetic: bool = False, scale_inputs: bool = True,
                                       min_samples: int = 8, prob_threshold: float = 0.5,
                                       logistic_kwargs: dict[str, Any] | None = None) -> LSMModelResult:
    """American option pricing via LSM + logistic-regression exercise policy."""
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1.")
    if not (0.0 <= prob_threshold <= 1.0):
        raise ValueError("prob_threshold must be in [0, 1].")

    _, paths = _simulate_paths(S0=S0, K=K, r=r, sigma=sigma, T=T, num_paths=num_paths,
                               steps=steps, seed=seed, antithetic=antithetic)
    n_steps, _ = paths.shape
    if n_steps < 2:
        raise ValueError("Need at least one time step for American valuation.")

    dt = T / (n_steps - 1)
    discount = np.exp(-r * dt)
    payoff = immediate_payoff(paths, strike=K, call=call)
    cashflow = payoff[-1].copy()
    mask = build_mask(paths, strike=K, call=call, include_all=include_all_paths, tolerance=mask_tolerance)

    model_kwargs = {"solver": "lbfgs", "max_iter": 1000, "random_state": seed}
    if logistic_kwargs:
        model_kwargs.update(logistic_kwargs)

    y_true_all: list[np.ndarray] = []
    y_prob_all: list[np.ndarray] = []

    for t in range(n_steps - 2, -1, -1):
        include = mask[t]
        if np.any(include):
            states = paths[t, include].reshape(-1, 1)
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
