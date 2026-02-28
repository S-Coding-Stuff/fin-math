"""Public API for running the GBM LSM-ML evaluation protocol."""

from pathlib import Path
from typing import Any

import pandas as pd

from .experiments import build_option_scenario_grid, run_protocol_experiment
from .plots import generate_protocol_plots


def run_evaluation_protocol(
    *,
    output_dir: Path | str = Path("results/summary"),
    S0: float = 100.0,
    r: float = 0.05,
    sigma: float = 0.2,
    call: bool = False,
    strikes: list[float] | None = None,
    maturities: list[float] | None = None,
    seeds: list[int] | None = None,
    baseline_paths: int = 1000,
    baseline_steps: int = 100,
    test_paths: int = 5000,
    path_grid: list[int] | None = None,
    step_grid: list[int] | None = None,
    binomial_steps: int = 2000,
    antithetic: bool = False,
    include_all_paths: bool = False,
    mask_tolerance: float = 0.0,
    min_samples: int = 8,
    ols_degree: int = 3,
    svr_c: float = 10.0,
    svr_epsilon: float = 0.05,
    svr_gamma: str = "scale",
    cart_max_depth: int = 6,
    cart_min_samples_leaf: int = 5,
    show_progress: bool = True,
    train_log_every: int = 10,
    generate_plots: bool = True,
    write_outputs: bool = True,
) -> dict[str, Any]:
    """Run baseline + sensitivity evaluations and return all result tables."""
    strikes = strikes if strikes is not None else [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0, 130.0]
    maturities = maturities if maturities is not None else [1.0 / 12.0, 0.25, 0.5, 1.0, 2.0]
    seeds = seeds if seeds is not None else [101, 202, 303]
    path_grid = path_grid if path_grid is not None else [500, 1000, 1500, 2000]
    step_grid = step_grid if step_grid is not None else [50, 100, 150, 200]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = build_option_scenario_grid(
        S0=S0,
        strikes=[float(x) for x in strikes],
        maturities=[float(x) for x in maturities],
        call=call,
    )

    svm_kwargs: dict[str, Any] = {
        "C": svr_c,
        "epsilon": svr_epsilon,
        "gamma": svr_gamma,
    }
    cart_kwargs: dict[str, Any] = {
        "max_depth": cart_max_depth,
        "min_samples_leaf": cart_min_samples_leaf,
    }

    baseline_records, baseline_summary, baseline_trace = run_protocol_experiment(
        experiment="baseline_rmse",
        variable_name="baseline_steps",
        variable_values=[baseline_steps],
        scenarios=scenarios,
        seeds=[int(x) for x in seeds],
        r=r,
        sigma=sigma,
        steps=baseline_steps,
        num_paths_train=baseline_paths,
        num_paths_test=test_paths,
        antithetic=antithetic,
        binomial_steps=binomial_steps,
        include_all_paths=include_all_paths,
        mask_tolerance=mask_tolerance,
        min_samples=min_samples,
        ols_degree=ols_degree,
        svm_kwargs=svm_kwargs,
        cart_kwargs=cart_kwargs,
        verbose=show_progress,
        train_log_every=train_log_every,
        collect_training_trace=True,
    )

    path_records, path_summary, _ = run_protocol_experiment(
        experiment="paths_sensitivity",
        variable_name="num_paths",
        variable_values=[int(x) for x in path_grid],
        scenarios=scenarios,
        seeds=[int(x) for x in seeds],
        r=r,
        sigma=sigma,
        steps=baseline_steps,
        num_paths_train=baseline_paths,
        num_paths_test=test_paths,
        antithetic=antithetic,
        binomial_steps=binomial_steps,
        include_all_paths=include_all_paths,
        mask_tolerance=mask_tolerance,
        min_samples=min_samples,
        ols_degree=ols_degree,
        svm_kwargs=svm_kwargs,
        cart_kwargs=cart_kwargs,
        verbose=show_progress,
        train_log_every=train_log_every,
        collect_training_trace=False,
    )

    step_records, step_summary, _ = run_protocol_experiment(
        experiment="steps_sensitivity",
        variable_name="steps",
        variable_values=[int(x) for x in step_grid],
        scenarios=scenarios,
        seeds=[int(x) for x in seeds],
        r=r,
        sigma=sigma,
        steps=baseline_steps,
        num_paths_train=baseline_paths,
        num_paths_test=test_paths,
        antithetic=antithetic,
        binomial_steps=binomial_steps,
        include_all_paths=include_all_paths,
        mask_tolerance=mask_tolerance,
        min_samples=min_samples,
        ols_degree=ols_degree,
        svm_kwargs=svm_kwargs,
        cart_kwargs=cart_kwargs,
        verbose=show_progress,
        train_log_every=train_log_every,
        collect_training_trace=False,
    )

    all_records = pd.concat([baseline_records, path_records, step_records], ignore_index=True)
    all_summary = pd.concat([baseline_summary, path_summary, step_summary], ignore_index=True)
    training_trace = baseline_trace.copy()

    records_path = out_dir / "american_lsm_ml_protocol_raw.csv"
    summary_path = out_dir / "american_lsm_ml_protocol_summary.csv"
    baseline_path = out_dir / "american_lsm_ml_protocol_baseline_rmse.csv"
    paths_path = out_dir / "american_lsm_ml_protocol_paths_sensitivity.csv"
    steps_path = out_dir / "american_lsm_ml_protocol_steps_sensitivity.csv"
    trace_path = out_dir / "american_lsm_ml_training_trace.csv"

    if write_outputs:
        all_records.to_csv(records_path, index=False)
        all_summary.to_csv(summary_path, index=False)
        baseline_summary.to_csv(baseline_path, index=False)
        path_summary.to_csv(paths_path, index=False)
        step_summary.to_csv(steps_path, index=False)
        training_trace.to_csv(trace_path, index=False)

    generated_plots = generate_protocol_plots(
        all_records=all_records,
        all_summary=all_summary,
        baseline_summary=baseline_summary,
        training_trace=training_trace,
        output_dir=out_dir,
        enabled=generate_plots,
    )

    if write_outputs:
        print(f"Wrote: {records_path}")
        print(f"Wrote: {summary_path}")
        print(f"Wrote: {baseline_path}")
        print(f"Wrote: {paths_path}")
        print(f"Wrote: {steps_path}")
        print(f"Wrote: {trace_path}")
        for plot_path in generated_plots:
            print(f"Wrote: {plot_path}")

    return {
        "all_records": all_records,
        "all_summary": all_summary,
        "baseline_summary": baseline_summary,
        "path_summary": path_summary,
        "step_summary": step_summary,
        "training_trace": training_trace,
        "records_path": records_path,
        "summary_path": summary_path,
        "baseline_path": baseline_path,
        "paths_path": paths_path,
        "steps_path": steps_path,
        "trace_path": trace_path,
        "plot_paths": generated_plots,
    }
