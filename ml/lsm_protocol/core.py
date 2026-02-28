"""Numerical core for LSM-ML training and policy evaluation."""

import math
from typing import Any

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from engines.monte_carlo import MonteCarloPricing, build_mask, immediate_payoff
from .models import LSMPolicy, OptionScenario, StepModel


def emit_progress_log(message: str, *, enabled: bool) -> None:
    if enabled:
        print(message, flush=True)


def simulate_gbm_price_paths(
    *,
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    num_paths: int,
    steps: int,
    seed: int,
    antithetic: bool,
) -> np.ndarray:
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
    return pricer._simulate_paths(antithetic=antithetic)


def price_american_option_crr(*, scenario: OptionScenario, r: float, sigma: float, binomial_steps: int) -> float:
    """Reference CRR price used as benchmark."""
    steps = int(binomial_steps)
    dt = scenario.T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    p = (np.exp(r * dt) - d) / (u - d)

    asset = np.zeros((steps + 1, steps + 1), dtype=float)
    for i in range(steps + 1):
        for j in range(i + 1):
            asset[j, i] = scenario.S0 * (u ** (i - j)) * (d ** j)

    option = np.zeros((steps + 1, steps + 1), dtype=float)
    if scenario.call:
        option[:, steps] = np.maximum(asset[:, steps] - scenario.K, 0.0)
    else:
        option[:, steps] = np.maximum(scenario.K - asset[:, steps], 0.0)

    discount = np.exp(-r * dt)
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            continuation = discount * (p * option[j, i + 1] + (1.0 - p) * option[j + 1, i + 1])
            intrinsic = max(asset[j, i] - scenario.K, 0.0) if scenario.call else max(scenario.K - asset[j, i], 0.0)
            option[j, i] = max(intrinsic, continuation)
    return float(option[0, 0])


def classify_moneyness_bucket(*, scenario: OptionScenario) -> str | None:
    m = scenario.moneyness
    if not (0.7 <= m <= 1.3):
        return None
    if 0.95 <= m <= 1.05:
        return "ATM"
    if scenario.call:
        return "ITM" if m > 1.05 else "OTM"
    return "ITM" if m < 0.95 else "OTM"


def fit_continuation_model(
    *,
    kind: str,
    states: np.ndarray,
    targets: np.ndarray,
    seed: int,
    ols_degree: int,
    svm_kwargs: dict[str, Any],
    cart_kwargs: dict[str, Any],
) -> tuple[StepModel, np.ndarray]:
    x = np.asarray(states, dtype=float).ravel()
    y = np.asarray(targets, dtype=float).ravel()

    if kind == "ols":
        features = np.vander(x, N=ols_degree + 1, increasing=True)
        coeffs, *_ = np.linalg.lstsq(features, y, rcond=None)
        pred = features @ coeffs
        return StepModel(kind="ols", model=coeffs, degree=ols_degree), pred

    if kind == "svr":
        model_params = {"kernel": "rbf", "C": 10.0, "epsilon": 0.05, "gamma": "scale"}
        model_params.update(svm_kwargs)
        model = make_pipeline(StandardScaler(), SVR(**model_params))
        model.fit(x.reshape(-1, 1), y)
        pred = np.asarray(model.predict(x.reshape(-1, 1)), dtype=float)
        return StepModel(kind="svr", model=model), pred

    if kind == "cart":
        model_params = {"random_state": seed, "min_samples_leaf": 5, "max_depth": 6}
        model_params.update(cart_kwargs)
        model = DecisionTreeRegressor(**model_params)
        model.fit(x.reshape(-1, 1), y)
        pred = np.asarray(model.predict(x.reshape(-1, 1)), dtype=float)
        return StepModel(kind="cart", model=model), pred

    raise ValueError(f"Unsupported model kind: {kind}")


def train_lsm_exercise_policy(
    *,
    kind: str,
    scenario: OptionScenario,
    r: float,
    paths: np.ndarray,
    include_all_paths: bool,
    mask_tolerance: float,
    min_samples: int,
    ols_degree: int,
    svm_kwargs: dict[str, Any],
    cart_kwargs: dict[str, Any],
    seed: int,
    verbose: bool,
    log_every: int,
    log_context: str,
    trace_rows: list[dict[str, Any]] | None,
    trace_meta: dict[str, Any] | None,
) -> LSMPolicy:
    n_steps, _ = paths.shape
    if n_steps < 2:
        raise ValueError("Need at least one time step.")

    steps = n_steps - 1
    dt = scenario.T / steps
    discount = np.exp(-r * dt)

    payoff = immediate_payoff(paths, strike=scenario.K, call=scenario.call)
    cashflow = payoff[-1].copy()
    mask = build_mask(
        paths,
        strike=scenario.K,
        call=scenario.call,
        include_all=include_all_paths,
        tolerance=mask_tolerance,
    )

    models: list[StepModel | None] = [None for _ in range(steps)]
    fallback_values: list[float] = [0.0 for _ in range(steps)]

    for t in range(steps - 1, -1, -1):
        include = mask[t]
        selected = int(np.sum(include))
        fit_mse = math.nan
        used_model = False
        reason = "no_mask"
        if np.any(include):
            states = paths[t, include]
            targets = cashflow[include] * discount
            exercise = payoff[t, include]

            fallback = float(np.mean(targets))
            fallback_values[t] = fallback

            if states.shape[0] >= min_samples:
                step_model, pred = fit_continuation_model(
                    kind=kind,
                    states=states,
                    targets=targets,
                    seed=seed,
                    ols_degree=ols_degree,
                    svm_kwargs=svm_kwargs,
                    cart_kwargs=cart_kwargs,
                )
                models[t] = step_model
                used_model = True
                reason = "fit"
            else:
                pred = np.full_like(targets, fallback)
                models[t] = None
                reason = "fallback"

            fit_mse = float(np.mean((targets - pred) ** 2))

            exercise_now = (exercise > 0.0) & (exercise > pred)
            idx = np.where(include)[0]
            cashflow[idx] = np.where(exercise_now, exercise, cashflow[idx] * discount)

        cashflow[~include] *= discount

        if trace_rows is not None:
            row = {
                "t": t,
                "steps": steps,
                "selected": selected,
                "fit_mse": fit_mse,
                "trained_model": used_model,
                "status": reason,
            }
            if trace_meta:
                row.update(trace_meta)
            trace_rows.append(row)

        backward_idx = (steps - 1) - t
        should_log = (backward_idx % max(log_every, 1) == 0) or (t == 0)
        if should_log:
            mse_text = f"{fit_mse:.6e}" if not math.isnan(fit_mse) else "nan"
            emit_progress_log(
                (
                    f"[train] {log_context} t={t:>3}/{steps - 1} "
                    f"selected={selected:>5} fit_mse={mse_text} status={reason}"
                ),
                enabled=verbose,
            )

    return LSMPolicy(
        call=scenario.call,
        strike=scenario.K,
        rate=r,
        maturity=scenario.T,
        steps=steps,
        models=models,
        fallback_values=fallback_values,
    )


def evaluate_lsm_exercise_policy(*, policy: LSMPolicy, paths: np.ndarray) -> tuple[float, float]:
    rewards = policy.discount_grid[:, None] * immediate_payoff(paths, strike=policy.strike, call=policy.call)
    n_times, n_paths = rewards.shape
    tau = np.full(n_paths, n_times - 1, dtype=int)
    alive = np.ones(n_paths, dtype=bool)

    for t in range(n_times - 1):
        if not np.any(alive):
            break
        idx = np.where(alive)[0]
        states = paths[t, idx]
        continuation = policy.continuation_value(t, states)
        exercise = rewards[t, idx]
        exercise_now = (exercise > 0.0) & (exercise >= continuation)
        chosen = idx[exercise_now]
        tau[chosen] = t
        alive[chosen] = False

    cashflows = rewards[tau, np.arange(n_paths)]
    n = cashflows.shape[0]
    ddof = 1 if n > 1 else 0
    return float(np.mean(cashflows)), float(np.std(cashflows, ddof=ddof) / np.sqrt(n))


# Backward-compatible aliases for existing imports.
log_message = emit_progress_log
simulate_paths = simulate_gbm_price_paths
benchmark_american_price = price_american_option_crr
moneyness_bucket = classify_moneyness_bucket
fit_step_model = fit_continuation_model
train_lsm_policy = train_lsm_exercise_policy
evaluate_policy = evaluate_lsm_exercise_policy
