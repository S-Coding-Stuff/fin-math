import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from greeks.binomial_fd import BinomialBumps, BinomialFiniteDifference
from greeks.mc_fd import MonteCarloFiniteDifference
from greeks.random_draws import draw_common_normals
from greeks_experiment import (
    DEFAULT_BINOMIAL_STEP_GRID,
    DEFAULT_BUMPS,
    DEFAULT_INCLUDE_ALL_PATHS,
    DEFAULT_NUM_PATHS,
    DEFAULT_SCENARIO_ID,
    DEFAULT_SEEDS,
    DEFAULT_SHOW_PROGRESS,
    DEFAULT_SAMPLING_SPECS,
    DEFAULT_STEPS,
    DEFAULT_MASK_TOLERANCE,
    SCENARIOS,
    SamplingSpec,
    Scenario,
    _make_pricer,
    _progress,
    _run_binomial_benchmark,
    _write_csv,
    export_paper_table,
)
from ml.lsm_mlp import american_nlsm_pricing_from_paths


METRICS = ("price", "delta", "gamma", "vega", "theta", "rho")
METRIC_LABELS = {
    "price": "Price",
    "delta": "Delta",
    "gamma": "Gamma",
    "vega": "Vega",
    "theta": "Theta",
    "rho": "Rho",
}


@dataclass(frozen=True)
class ModelSpec:
    code: str
    display_name: str
    basis_fn: str | None = None
    freeze_policy: bool | None = None


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(code="lsm", display_name="LSM", basis_fn="laguerre", freeze_policy=False),
    ModelSpec(code="rlsm", display_name="RLSM", basis_fn="rlsm", freeze_policy=True),
    ModelSpec(code="nlsm", display_name="NLSM"),
)

DEFAULT_RLSM_KWARGS = {
    "hidden_size": 10,
    "activation": "leaky_relu",
    "factors": (1.0,),
    "weight_scale": 1.0,
    "ridge_lambda": 0.0,
    "train_eval_split": 0.5,
    "use_payoff_as_input": True,
    "optstop_compatible": True,
}

DEFAULT_NLSM_KWARGS = {
    "hidden_size": 10,
    "lr": 1e-3,
    "epochs": 10,
    "batch_size": 2000,
    "min_samples": 8,
    "train_itm_only": True,
    "use_payoff_as_input": True,
    "optstop_compatible": True,
    "show_progress": False,
}


def _format_estimate_cell(mean: float, std: float) -> str:
    if np.isnan(mean):
        return ""
    return f"{mean:.6f} +/- {std:.6f}"


def _format_float_cell(value: float, digits: int = 6) -> str:
    if np.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def _resolve_bump_sizes(bumps: DEFAULT_BUMPS.__class__) -> dict[str, float]:
    return {
        "S_0": float(bumps.S_0),
        "sigma": float(bumps.sigma),
        "r": float(bumps.r),
        "T": float(bumps.T),
    }


def _nlsm_price_from_normals(
    *,
    scenario: Scenario,
    spec: SamplingSpec,
    seed: int,
    num_paths: int,
    steps: int,
    normals: np.ndarray,
    include_all_paths: bool,
    mask_tolerance: float,
    nlsm_kwargs: dict[str, Any],
    S0: float | None = None,
    sigma: float | None = None,
    r: float | None = None,
    T: float | None = None,
) -> float:
    s0_eff = float(scenario.S0 if S0 is None else S0)
    sigma_eff = float(scenario.sigma if sigma is None else sigma)
    r_eff = float(scenario.r if r is None else r)
    t_eff = float(scenario.T if T is None else T)

    pricer = _make_pricer(
        Scenario(
            scenario_id=scenario.scenario_id,
            S0=s0_eff,
            K=scenario.K,
            r=r_eff,
            sigma=sigma_eff,
            T=t_eff,
            call=scenario.call,
        ),
        spec=spec,
        seed=seed,
        num_paths=num_paths,
        steps=steps,
        crn=True,
    )
    paths = pricer._simulate_paths(risk_neutral=True, Z=normals, antithetic=False)
    result = american_nlsm_pricing_from_paths(
        paths=paths,
        K=scenario.K,
        r=r_eff,
        T=t_eff,
        call=scenario.call,
        include_all_paths=include_all_paths,
        mask_tolerance=mask_tolerance,
        seed=seed,
        **nlsm_kwargs,
    )
    return float(result.price)


def _run_single_nlsm_estimate(
    *,
    scenario: Scenario,
    spec: SamplingSpec,
    metric: str,
    seed: int,
    num_paths: int,
    steps: int,
    include_all_paths: bool,
    mask_tolerance: float,
    bumps: DEFAULT_BUMPS.__class__,
    nlsm_kwargs: dict[str, Any],
) -> tuple[float, float]:
    pricer = _make_pricer(
        scenario,
        spec=spec,
        seed=seed,
        num_paths=num_paths,
        steps=steps,
        crn=True,
    )
    started = time.perf_counter()
    normals = draw_common_normals(pricer, antithetic=spec.antithetic)
    bump_sizes = _resolve_bump_sizes(bumps)

    def price_at(**overrides: float) -> float:
        return _nlsm_price_from_normals(
            scenario=scenario,
            spec=spec,
            seed=seed,
            num_paths=num_paths,
            steps=steps,
            normals=normals,
            include_all_paths=include_all_paths,
            mask_tolerance=mask_tolerance,
            nlsm_kwargs=nlsm_kwargs,
            **overrides,
        )

    metric_l = str(metric).lower()
    if metric_l == "price":
        value = price_at()
    elif metric_l == "delta":
        h = float(bump_sizes["S_0"])
        value = (price_at(S0=scenario.S0 + h) - price_at(S0=scenario.S0 - h)) / (2.0 * h)
    elif metric_l == "gamma":
        h = float(bump_sizes["S_0"])
        value = (
            price_at(S0=scenario.S0 + h)
            - 2.0 * price_at(S0=scenario.S0)
            + price_at(S0=scenario.S0 - h)
        ) / (h ** 2)
    elif metric_l == "vega":
        h = float(bump_sizes["sigma"])
        value = (price_at(sigma=scenario.sigma + h) - price_at(sigma=scenario.sigma - h)) / (2.0 * h)
    elif metric_l == "theta":
        h = float(bump_sizes["T"])
        value = -(price_at(T=scenario.T + h) - price_at(T=scenario.T - h)) / (2.0 * h)
    elif metric_l == "rho":
        h = float(bump_sizes["r"])
        value = (price_at(r=scenario.r + h) - price_at(r=scenario.r - h)) / (2.0 * h)
    else:
        raise ValueError(f"Unsupported metric '{metric}'.")
    runtime = float(time.perf_counter() - started)
    return float(value), runtime


def _run_single_model_estimate(
    *,
    model: ModelSpec,
    scenario: Scenario,
    spec: SamplingSpec,
    metric: str,
    seed: int,
    num_paths: int,
    steps: int,
    include_all_paths: bool,
    mask_tolerance: float,
    bumps: DEFAULT_BUMPS.__class__,
    gamma_method: str,
) -> tuple[float, float]:
    if model.code == "nlsm":
        return _run_single_nlsm_estimate(
            scenario=scenario,
            spec=spec,
            metric=metric,
            seed=seed,
            num_paths=num_paths,
            steps=steps,
            include_all_paths=include_all_paths,
            mask_tolerance=mask_tolerance,
            bumps=bumps,
            nlsm_kwargs=DEFAULT_NLSM_KWARGS,
        )

    pricer = _make_pricer(
        scenario=scenario,
        spec=spec,
        seed=seed,
        num_paths=num_paths,
        steps=steps,
        crn=True,
    )
    mc_fd = MonteCarloFiniteDifference(
        pricer,
        call=scenario.call,
        antithetic=spec.antithetic,
        risk_neutral=True,
        bumps=bumps,
        style="american",
        basis_fn=model.basis_fn or "laguerre",
        include_all_paths=include_all_paths,
        mask_tolerance=mask_tolerance,
        freeze_policy=model.freeze_policy,
        gamma_method=gamma_method,
        rlsm_kwargs=DEFAULT_RLSM_KWARGS if model.code == "rlsm" else None,
    )
    started = time.perf_counter()
    if metric == "price":
        value = float(mc_fd.price()[0])
    else:
        value = float(mc_fd.greek(metric, scheme="central"))
    return value, float(time.perf_counter() - started)


def _run_binomial_price_benchmark(
    *,
    scenario: Scenario,
    tree_steps: int,
    bumps: DEFAULT_BUMPS.__class__,
) -> tuple[float, float]:
    tree = BinomialFiniteDifference(
        S_0=scenario.S0,
        K=scenario.K,
        r=scenario.r,
        sigma=scenario.sigma,
        T=scenario.T,
        steps=tree_steps,
        call=scenario.call,
        bumps=BinomialBumps(S_0=bumps.S_0, sigma=bumps.sigma, r=bumps.r, T=bumps.T),
    )
    started = time.perf_counter()
    return float(tree.price()), float(time.perf_counter() - started)


def _ci_from_values(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return math.nan, math.nan, math.nan
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    if arr.size <= 1:
        return mean, mean, mean
    half_width = 1.96 * float(np.std(arr, ddof=1)) / math.sqrt(arr.size)
    return mean, mean - half_width, mean + half_width


def _group_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        grouped.setdefault(key, []).append(row)
    return grouped


def summarize_records(
    records: list[dict[str, Any]],
    *,
    benchmark_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped = _group_rows(
        records,
        (
            "scenario_id",
            "model_code",
            "metric",
            "method_code",
            "num_paths",
            "steps",
        ),
    )
    summary_rows: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        scenario_id, model_code, metric, method_code, num_paths, steps = key
        first = rows[0]
        estimates = [float(row["estimate"]) for row in rows]
        runtimes = [float(row["runtime_sec"]) for row in rows]
        abs_errors = [float(row["abs_error"]) for row in rows]
        mean, ci_low, ci_high = _ci_from_values(estimates)
        err_mean, err_ci_low, err_ci_high = _ci_from_values(abs_errors)
        estimate_std = float(np.std(np.asarray(estimates), ddof=1)) if len(estimates) > 1 else 0.0
        summary_rows.append(
            {
                "scenario_id": scenario_id,
                "model_code": model_code,
                "model_display": first["model_display"],
                "metric": metric,
                "metric_display": METRIC_LABELS[metric],
                "method_code": method_code,
                "method_display": first["method_display"],
                "num_replications": len(rows),
                "num_paths": num_paths,
                "steps": steps,
                "estimate_mean": mean,
                "estimate_std": estimate_std,
                "estimate_ci95_low": ci_low,
                "estimate_ci95_high": ci_high,
                "benchmark_default_steps": benchmark_map[scenario_id]["default_steps"],
                "benchmark_value": benchmark_map[scenario_id][metric]["default"],
                "benchmark_value_step_1000": benchmark_map[scenario_id][metric].get(1000, math.nan),
                "benchmark_value_step_2000": benchmark_map[scenario_id][metric].get(2000, math.nan),
                "benchmark_value_step_4000": benchmark_map[scenario_id][metric].get(4000, math.nan),
                "abs_error_mean": err_mean,
                "abs_error_ci95_low": err_ci_low,
                "abs_error_ci95_high": err_ci_high,
                "runtime_mean_sec": float(np.mean(np.asarray(runtimes))),
            }
        )
    return summary_rows


def build_paper_table(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_order = {metric: idx for idx, metric in enumerate(METRICS)}
    grouped = _group_rows(summary_rows, ("model_display", "metric"))
    table_rows: list[dict[str, Any]] = []
    for (model_display, metric), group in sorted(grouped.items(), key=lambda item: (item[0][0], metric_order[item[0][1]])):
        row_map = {str(row["method_code"]): row for row in group}
        first = group[0]
        row: dict[str, Any] = {
            "Model": model_display,
            "Metric": METRIC_LABELS[metric],
            "Benchmark": f"{float(first['benchmark_value']):.6f}",
        }
        for spec in DEFAULT_SAMPLING_SPECS:
            summary = row_map.get(spec.code)
            row[spec.display_name] = "" if summary is None else _format_estimate_cell(
                float(summary["estimate_mean"]),
                float(summary["estimate_std"]),
            )
        table_rows.append(row)
    return table_rows


def build_metric_table(
    summary_rows: list[dict[str, Any]],
    *,
    metric_key: str,
    digits: int = 6,
    include_benchmark: bool = False,
) -> list[dict[str, Any]]:
    metric_order = {metric: idx for idx, metric in enumerate(METRICS)}
    grouped = _group_rows(summary_rows, ("model_display", "metric"))
    table_rows: list[dict[str, Any]] = []
    for (model_display, metric), group in sorted(grouped.items(), key=lambda item: (item[0][0], metric_order[item[0][1]])):
        row_map = {str(row["method_code"]): row for row in group}
        first = group[0]
        row: dict[str, Any] = {
            "Model": model_display,
            "Metric": METRIC_LABELS[metric],
        }
        if include_benchmark:
            row["Benchmark"] = _format_float_cell(float(first["benchmark_value"]), digits=digits)
        for spec in DEFAULT_SAMPLING_SPECS:
            summary = row_map.get(spec.code)
            row[spec.display_name] = "" if summary is None else _format_float_cell(
                float(summary[metric_key]),
                digits=digits,
            )
        table_rows.append(row)
    return table_rows


def run_model_comparison_suite(
    *,
    scenario: Scenario,
    seeds: tuple[int, ...],
    num_paths: int,
    steps: int,
    include_all_paths: bool,
    mask_tolerance: float,
    gamma_method: str,
    bumps: DEFAULT_BUMPS.__class__,
    binomial_steps: tuple[int, ...],
    show_progress: bool,
) -> dict[str, list[dict[str, Any]]]:
    default_benchmark_steps = 2000 if 2000 in binomial_steps else binomial_steps[len(binomial_steps) // 2]
    benchmark_map: dict[str, dict[str, Any]] = {scenario.scenario_id: {"default_steps": default_benchmark_steps}}
    benchmark_runtime: dict[str, dict[str, float]] = {scenario.scenario_id: {}}

    benchmark_iter = _progress(METRICS, enabled=show_progress, desc="Benchmarks")
    for metric in benchmark_iter:
        benchmark_iter.set_postfix(metric=metric)
        step_values: dict[int, float] = {}
        for tree_steps in binomial_steps:
            if metric == "price":
                value, runtime = _run_binomial_price_benchmark(
                    scenario=scenario,
                    tree_steps=tree_steps,
                    bumps=bumps,
                )
            else:
                value, runtime = _run_binomial_benchmark(
                    scenario=scenario,
                    greek=metric,
                    tree_steps=tree_steps,
                    bumps=bumps,
                )
            step_values[tree_steps] = value
            if tree_steps == default_benchmark_steps:
                benchmark_runtime[scenario.scenario_id][metric] = runtime
        benchmark_map[scenario.scenario_id][metric] = {
            "default": step_values[default_benchmark_steps],
            **step_values,
        }

    tasks = [
        (model, metric, spec, seed)
        for model in MODEL_SPECS
        for metric in METRICS
        for spec in DEFAULT_SAMPLING_SPECS
        for seed in seeds
    ]
    task_iter = _progress(tasks, enabled=show_progress, desc="Model comparison")
    records: list[dict[str, Any]] = []
    for model, metric, spec, seed in task_iter:
        task_iter.set_postfix(model=model.code, metric=metric, method=spec.code, seed=seed)
        benchmark_value = benchmark_map[scenario.scenario_id][metric]["default"]
        estimate, runtime_sec = _run_single_model_estimate(
            model=model,
            scenario=scenario,
            spec=spec,
            metric=metric,
            seed=seed,
            num_paths=num_paths,
            steps=steps,
            include_all_paths=include_all_paths,
            mask_tolerance=mask_tolerance,
            bumps=bumps,
            gamma_method=gamma_method,
        )
        records.append(
            {
                "scenario_id": scenario.scenario_id,
                "model_code": model.code,
                "model_display": model.display_name,
                "metric": metric,
                "metric_display": METRIC_LABELS[metric],
                "seed": seed,
                "method_code": spec.code,
                "method_display": spec.display_name,
                "num_paths": num_paths,
                "steps": steps,
                "estimate": estimate,
                "benchmark_value": benchmark_value,
                "abs_error": abs(estimate - benchmark_value),
                "runtime_sec": runtime_sec,
            }
        )

    summary_rows = summarize_records(records, benchmark_map=benchmark_map)
    benchmark_rows: list[dict[str, Any]] = []
    for metric in METRICS:
        values = benchmark_map[scenario.scenario_id][metric]
        benchmark_rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "model_code": "binomial_fd",
                "model_display": "Binomial FD",
                "metric": metric,
                "metric_display": METRIC_LABELS[metric],
                "method_code": "binomial_fd",
                "method_display": f"Binomial FD ({default_benchmark_steps})",
                "num_replications": 1,
                "num_paths": "",
                "steps": "",
                "estimate_mean": values["default"],
                "estimate_std": 0.0,
                "estimate_ci95_low": values["default"],
                "estimate_ci95_high": values["default"],
                "benchmark_default_steps": default_benchmark_steps,
                "benchmark_value": values["default"],
                "benchmark_value_step_1000": values.get(1000, math.nan),
                "benchmark_value_step_2000": values.get(2000, math.nan),
                "benchmark_value_step_4000": values.get(4000, math.nan),
                "abs_error_mean": 0.0,
                "abs_error_ci95_low": 0.0,
                "abs_error_ci95_high": 0.0,
                "runtime_mean_sec": benchmark_runtime[scenario.scenario_id][metric],
            }
        )

    return {
        "records": records,
        "summary": summary_rows + benchmark_rows,
        "summary_non_benchmark": summary_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare 1D American put price/Greeks across LSM, RLSM, and NLSM.")
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO_ID, choices=sorted(SCENARIOS))
    parser.add_argument("--num-paths", type=int, default=DEFAULT_NUM_PATHS)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--output-dir", default="results/greeks_model_comparison")
    parser.add_argument("--show-progress", action=argparse.BooleanOptionalAction, default=DEFAULT_SHOW_PROGRESS)
    args = parser.parse_args()

    scenario = SCENARIOS[args.scenario]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = run_model_comparison_suite(
        scenario=scenario,
        seeds=DEFAULT_SEEDS,
        num_paths=int(args.num_paths),
        steps=int(args.steps),
        include_all_paths=DEFAULT_INCLUDE_ALL_PATHS,
        mask_tolerance=DEFAULT_MASK_TOLERANCE,
        gamma_method="fd",
        bumps=DEFAULT_BUMPS,
        binomial_steps=DEFAULT_BINOMIAL_STEP_GRID,
        show_progress=bool(args.show_progress),
    )

    stem = f"{scenario.scenario_id}_model_compare"
    raw_path = output_dir / f"{stem}_raw.csv"
    summary_path = output_dir / f"{stem}_summary_long.csv"
    paper_csv = output_dir / f"{stem}_paper_table.csv"
    paper_tex = output_dir / f"{stem}_paper_table.tex"
    runtime_csv = output_dir / f"{stem}_runtime_table.csv"
    runtime_tex = output_dir / f"{stem}_runtime_table.tex"
    abs_error_csv = output_dir / f"{stem}_abs_error_table.csv"
    abs_error_tex = output_dir / f"{stem}_abs_error_table.tex"

    _write_csv(raw_path, result["records"])
    _write_csv(summary_path, result["summary"])

    paper_rows = build_paper_table(result["summary_non_benchmark"])
    export_paper_table(
        paper_rows,
        csv_path=paper_csv,
        tex_path=paper_tex,
        caption=(
            "One-dimensional American put price and Greek estimates across LSM, RLSM, and NLSM "
            "under MC, MC plus antithetic variates, QMC, and randomized QMC with common random numbers."
        ),
        label="tab:american_put_model_compare",
    )

    runtime_rows = build_metric_table(result["summary_non_benchmark"], metric_key="runtime_mean_sec", digits=6)
    export_paper_table(
        runtime_rows,
        csv_path=runtime_csv,
        tex_path=runtime_tex,
        caption="Mean runtime in seconds for the 1D American put model-comparison suite.",
        label="tab:american_put_model_compare_runtime",
    )

    abs_error_rows = build_metric_table(
        result["summary_non_benchmark"],
        metric_key="abs_error_mean",
        digits=6,
        include_benchmark=True,
    )
    export_paper_table(
        abs_error_rows,
        csv_path=abs_error_csv,
        tex_path=abs_error_tex,
        caption="Absolute error against the binomial benchmark for the 1D American put model-comparison suite.",
        label="tab:american_put_model_compare_abs_error",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
