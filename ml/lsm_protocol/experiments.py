"""Experiment orchestration and RMSE aggregation for LSM-ML evaluation."""

from typing import Any

import numpy as np
import pandas as pd

from .core import (
    classify_moneyness_bucket,
    emit_progress_log,
    evaluate_lsm_exercise_policy,
    price_american_option_crr,
    simulate_gbm_price_paths,
    train_lsm_exercise_policy,
)
from .models import OptionScenario


def build_option_scenario_grid(
    *, S0: float, strikes: list[float], maturities: list[float], call: bool
) -> list[OptionScenario]:
    scenarios: list[OptionScenario] = []
    for K in strikes:
        for T in maturities:
            scenarios.append(OptionScenario(S0=S0, K=float(K), T=float(T), call=call))
    return scenarios


def compute_per_seed_rmse(*, records: pd.DataFrame) -> pd.DataFrame:
    bucketed = records[records["bucket"].notna()].copy()
    bucketed = bucketed.assign(sq_err=bucketed["error"] ** 2)

    rows: list[pd.DataFrame] = []
    for bucket in ["ITM", "ATM", "OTM"]:
        part = bucketed[bucketed["bucket"] == bucket]
        if not part.empty:
            grp = part.groupby(["seed", "model"], as_index=False)["sq_err"].mean()
            grp["rmse"] = np.sqrt(grp["sq_err"])
            grp["bucket"] = bucket
            rows.append(grp[["seed", "model", "bucket", "rmse"]])

    total = bucketed.groupby(["seed", "model"], as_index=False)["sq_err"].mean()
    total["rmse"] = np.sqrt(total["sq_err"])
    total["bucket"] = "Total"
    rows.append(total[["seed", "model", "bucket", "rmse"]])

    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(columns=["seed", "model", "bucket", "rmse"])


def summarise_rmse_confidence_intervals(
    *,
    per_seed_rmse: pd.DataFrame,
    experiment: str,
    variable_name: str,
    variable_value: int,
) -> pd.DataFrame:
    if per_seed_rmse.empty:
        return pd.DataFrame()

    grouped = per_seed_rmse.groupby(["model", "bucket"], as_index=False).agg(
        n_seeds=("rmse", "size"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else 0.0),
    )
    grouped["rmse_ci95_low"] = grouped["rmse_mean"] - 1.96 * grouped["rmse_std"] / np.sqrt(grouped["n_seeds"])
    grouped["rmse_ci95_high"] = grouped["rmse_mean"] + 1.96 * grouped["rmse_std"] / np.sqrt(grouped["n_seeds"])
    grouped["experiment"] = experiment
    grouped[variable_name] = variable_value
    return grouped[
        [
            "experiment",
            variable_name,
            "model",
            "bucket",
            "n_seeds",
            "rmse_mean",
            "rmse_std",
            "rmse_ci95_low",
            "rmse_ci95_high",
        ]
    ]


def run_single_seed_evaluation(
    *,
    scenarios: list[OptionScenario],
    r: float,
    sigma: float,
    steps: int,
    num_paths_train: int,
    num_paths_test: int,
    antithetic: bool,
    seed: int,
    binomial_steps: int,
    include_all_paths: bool,
    mask_tolerance: float,
    min_samples: int,
    ols_degree: int,
    svm_kwargs: dict[str, Any],
    cart_kwargs: dict[str, Any],
    verbose: bool,
    train_log_every: int,
    collect_training_trace: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_kinds = ["ols", "svr", "cart"]
    rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    scenario_count = len(scenarios)

    for i, scenario in enumerate(scenarios, start=1):
        bucket = classify_moneyness_bucket(scenario=scenario)
        if bucket is None:
            continue

        emit_progress_log(
            f"[scenario] seed={seed} {i}/{scenario_count} K={scenario.K:.2f} T={scenario.T:.4f} bucket={bucket}",
            enabled=verbose,
        )

        train_paths = simulate_gbm_price_paths(
            S0=scenario.S0,
            K=scenario.K,
            r=r,
            sigma=sigma,
            T=scenario.T,
            num_paths=num_paths_train,
            steps=steps,
            seed=seed,
            antithetic=antithetic,
        )
        test_paths = simulate_gbm_price_paths(
            S0=scenario.S0,
            K=scenario.K,
            r=r,
            sigma=sigma,
            T=scenario.T,
            num_paths=num_paths_test,
            steps=steps,
            seed=seed + 1_000_000,
            antithetic=antithetic,
        )
        benchmark = price_american_option_crr(
            scenario=scenario,
            r=r,
            sigma=sigma,
            binomial_steps=binomial_steps,
        )

        for model_kind in model_kinds:
            context = f"seed={seed} model={model_kind} K={scenario.K:.2f} T={scenario.T:.4f}"
            trace_meta = {
                "seed": seed,
                "model": model_kind,
                "S0": scenario.S0,
                "K": scenario.K,
                "T": scenario.T,
                "call": scenario.call,
                "moneyness": scenario.moneyness,
                "bucket": bucket,
            }
            policy = train_lsm_exercise_policy(
                kind=model_kind,
                scenario=scenario,
                r=r,
                paths=train_paths,
                include_all_paths=include_all_paths,
                mask_tolerance=mask_tolerance,
                min_samples=min_samples,
                ols_degree=ols_degree,
                svm_kwargs=svm_kwargs,
                cart_kwargs=cart_kwargs,
                seed=seed,
                verbose=verbose,
                log_every=train_log_every,
                log_context=context,
                trace_rows=trace_rows if collect_training_trace else None,
                trace_meta=trace_meta,
            )
            price, stderr = evaluate_lsm_exercise_policy(policy=policy, paths=test_paths)
            rows.append(
                {
                    "seed": seed,
                    "model": model_kind,
                    "S0": scenario.S0,
                    "K": scenario.K,
                    "T": scenario.T,
                    "call": scenario.call,
                    "moneyness": scenario.moneyness,
                    "bucket": bucket,
                    "price": price,
                    "stderr": stderr,
                    "benchmark": benchmark,
                    "error": price - benchmark,
                }
            )
            emit_progress_log(
                (
                    f"[eval] seed={seed} model={model_kind} "
                    f"K={scenario.K:.2f} T={scenario.T:.4f} "
                    f"price={price:.6f} benchmark={benchmark:.6f}"
                ),
                enabled=verbose,
            )

    return pd.DataFrame(rows), pd.DataFrame(trace_rows)


def run_protocol_experiment(
    *,
    experiment: str,
    variable_name: str,
    variable_values: list[int],
    scenarios: list[OptionScenario],
    seeds: list[int],
    r: float,
    sigma: float,
    steps: int,
    num_paths_train: int,
    num_paths_test: int,
    antithetic: bool,
    binomial_steps: int,
    include_all_paths: bool,
    mask_tolerance: float,
    min_samples: int,
    ols_degree: int,
    svm_kwargs: dict[str, Any],
    cart_kwargs: dict[str, Any],
    verbose: bool,
    train_log_every: int,
    collect_training_trace: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_records: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []
    all_traces: list[pd.DataFrame] = []

    total_runs = len(variable_values) * len(seeds)
    run_id = 0
    for value in variable_values:
        per_value_records: list[pd.DataFrame] = []
        per_value_traces: list[pd.DataFrame] = []
        for seed in seeds:
            run_id += 1
            run_steps = value if variable_name == "steps" else steps
            run_paths = value if variable_name == "num_paths" else num_paths_train
            emit_progress_log(
                (
                    f"[run] {run_id}/{total_runs} experiment={experiment} "
                    f"{variable_name}={value} seed={seed} steps={run_steps} paths={run_paths}"
                ),
                enabled=verbose,
            )
            df, trace_df = run_single_seed_evaluation(
                scenarios=scenarios,
                r=r,
                sigma=sigma,
                steps=run_steps,
                num_paths_train=run_paths,
                num_paths_test=num_paths_test,
                antithetic=antithetic,
                seed=seed,
                binomial_steps=binomial_steps,
                include_all_paths=include_all_paths,
                mask_tolerance=mask_tolerance,
                min_samples=min_samples,
                ols_degree=ols_degree,
                svm_kwargs=svm_kwargs,
                cart_kwargs=cart_kwargs,
                verbose=verbose,
                train_log_every=train_log_every,
                collect_training_trace=collect_training_trace,
            )
            if df.empty:
                continue
            df[variable_name] = value
            df["experiment"] = experiment
            per_value_records.append(df)
            if not trace_df.empty:
                trace_df[variable_name] = value
                trace_df["experiment"] = experiment
                per_value_traces.append(trace_df)

        if not per_value_records:
            continue

        value_records = pd.concat(per_value_records, ignore_index=True)
        per_seed_rmse = compute_per_seed_rmse(records=value_records)
        summary = summarise_rmse_confidence_intervals(
            per_seed_rmse=per_seed_rmse,
            experiment=experiment,
            variable_name=variable_name,
            variable_value=value,
        )
        all_records.append(value_records)
        all_summaries.append(summary)
        if per_value_traces:
            all_traces.append(pd.concat(per_value_traces, ignore_index=True))

    records_df = pd.concat(all_records, ignore_index=True) if all_records else pd.DataFrame()
    summary_df = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    trace_df = pd.concat(all_traces, ignore_index=True) if all_traces else pd.DataFrame()
    return records_df, summary_df, trace_df


# Backward-compatible aliases for existing imports.
build_scenarios = build_option_scenario_grid
per_seed_rmse_table = compute_per_seed_rmse
aggregate_rmse_ci = summarise_rmse_confidence_intervals
evaluate_single_setting = run_single_seed_evaluation
run_experiment = run_protocol_experiment
