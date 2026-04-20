import numpy as np
from time import perf_counter
from typing import Any, Callable
import csv
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engines.monte_carlo import MonteCarloPricing
from engines.payoffs import (
    arithmetic_basket_payoff,
    geometric_basket_payoff,
    make_payoff_fn,
    max_basket_payoff,
)
from ml.lsm_mlp import american_nlsm_pricing_from_paths

PayoffFn = Callable[[np.ndarray], np.ndarray]

@dataclass(frozen=True)
class ExperimentCase:
    name: str
    S0: float | tuple[float, ...]
    K: float
    r: float
    sigma: float | tuple[float, ...]
    T: float
    steps: int
    num_paths: int
    call: bool
    payoff_style: str = "vanilla"
    corr: tuple[tuple[float, ...], ...] | None = None
    div: float | tuple[float, ...] = 0.0
    weights: tuple[float, ...] | None = None
    antithetic: bool = False
    polynomial_basis: str = "laguerre"
    reference_style: str | None = None
    reference_steps: int | None = None

    @property
    def dimension(self) -> int:
        if isinstance(self.S0, tuple):
            return len(self.S0)
        return 1


@dataclass(frozen=True)
class MLPModelSpec:
    name: str
    hidden_sizes: tuple[int, ...]
    epochs: int
    first_step_epochs: int | None = 10


def _as_array(value: float | tuple[float, ...]) -> float | np.ndarray:
    if isinstance(value, tuple):
        return np.asarray(value, dtype=float)
    return float(value)


def _as_matrix(value: tuple[tuple[float, ...], ...] | None) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=float)


def equicorrelation_matrix(dimension: int, rho: float) -> tuple[tuple[float, ...], ...]:
    corr = np.full((dimension, dimension), rho, dtype=float)
    np.fill_diagonal(corr, 1.0)
    return tuple(tuple(float(x) for x in row) for row in corr)

def _payoff_fn(case: ExperimentCase) -> PayoffFn | None:
    weights = None if case.weights is None else np.asarray(case.weights, dtype=float)
    return make_payoff_fn(
        payoff_style=case.payoff_style,
        strike=case.K,
        call=case.call,
        weights=weights,
    )


def _crr_american_price(
    *,
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    steps: int,
    call: bool,
    div: float = 0.0,
) -> float:
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    growth = np.exp((r - div) * dt)
    p = (growth - d) / (u - d)
    discount = np.exp(-r * dt)

    # Work with a single time slice at a time. The original 2D lattice allocation
    # becomes infeasible at large step counts and is unnecessary for backward induction.
    node_index = np.arange(steps + 1, dtype=float)
    asset = S0 * (u ** (steps - node_index)) * (d ** node_index)
    option = np.maximum(asset - K, 0.0) if call else np.maximum(K - asset, 0.0)

    for _ in range(steps - 1, -1, -1):
        continuation = discount * (p * option[:-1] + (1.0 - p) * option[1:])
        asset = asset[:-1] / u
        intrinsic = np.maximum(asset - K, 0.0) if call else np.maximum(K - asset, 0.0)
        option = np.maximum(intrinsic, continuation)
    return float(option[0])


def _effective_geometric_basket_parameters(case: ExperimentCase) -> tuple[float, float, float]:
    s0 = np.asarray(case.S0, dtype=float)
    sigma = np.asarray(case.sigma, dtype=float)
    if s0.ndim != 1 or sigma.ndim != 1:
        raise ValueError(f"{case.name}: geometric basket reference requires vector S0 and sigma.")

    div = np.asarray(case.div if isinstance(case.div, tuple) else [float(case.div)] * s0.size, dtype=float)
    corr = _as_matrix(case.corr)
    if corr is None:
        corr = np.eye(s0.size, dtype=float)

    s0_hat = float(np.exp(np.mean(np.log(s0))))
    sigma_hat = float(np.sqrt(sigma @ corr @ sigma) / s0.size)
    div_hat = float(np.mean(div + 0.5 * sigma**2) - 0.5 * sigma_hat**2)
    return s0_hat, sigma_hat, div_hat


def reference_price(case: ExperimentCase) -> float | None:
    if case.reference_style is None:
        return None
    if case.reference_steps is None:
        raise ValueError(f"{case.name}: reference_steps is required when reference_style is set.")

    if case.reference_style == "crr_vanilla":
        return _crr_american_price(
            S0=float(case.S0),
            K=case.K,
            r=case.r,
            sigma=float(case.sigma),
            T=case.T,
            steps=case.reference_steps,
            call=case.call,
            div=float(case.div),
        )

    if case.reference_style == "crr_geometric_equivalent":
        s0_hat, sigma_hat, div_hat = _effective_geometric_basket_parameters(case)
        return _crr_american_price(
            S0=s0_hat,
            K=case.K,
            r=case.r,
            sigma=sigma_hat,
            T=case.T,
            steps=case.reference_steps,
            call=case.call,
            div=div_hat,
        )

    raise ValueError(f"Unknown reference_style: {case.reference_style}")


def default_black_scholes_cases() -> list[ExperimentCase]:
    return [
        ExperimentCase(
            name="bermudan_put_1d",
            S0=100.0,
            K=110.0,
            r=0.1,
            sigma=0.25,
            T=1.0,
            steps=10,
            num_paths=100_000,
            call=False,
            payoff_style="vanilla",
            polynomial_basis="laguerre",
            reference_style="crr_vanilla",
            reference_steps=2_000,
        ),
        ExperimentCase(
            name="atm_put_1d",
            S0=100.0,
            K=100.0,
            r=0.05,
            sigma=0.20,
            T=1.0,
            steps=10,
            num_paths=100_000,
            call=False,
            payoff_style="vanilla",
            polynomial_basis="laguerre",
            reference_style="crr_vanilla",
            reference_steps=2_000,
        ),
        ExperimentCase(
            name="high_vol_put_1d",
            S0=100.0,
            K=100.0,
            r=0.05,
            sigma=0.40,
            T=1.0,
            steps=10,
            num_paths=100_000,
            call=False,
            payoff_style="vanilla",
            polynomial_basis="laguerre",
            reference_style="crr_vanilla",
            reference_steps=2_000,
        ),
        ExperimentCase(
            name="short_mat_put_1d",
            S0=100.0,
            K=100.0,
            r=0.05,
            sigma=0.20,
            T=0.5,
            steps=8,
            num_paths=100_000,
            call=False,
            payoff_style="vanilla",
            polynomial_basis="laguerre",
            reference_style="crr_vanilla",
            reference_steps=2_000,
        ),
        ExperimentCase(
            name="geometric_put_d2",
            S0=(100.0, 100.0),
            K=100.0,
            r=0.05,
            sigma=(0.2, 0.2),
            T=1.0,
            steps=10,
            num_paths=100_000,
            call=False,
            payoff_style="geometric_basket",
            corr=equicorrelation_matrix(2, 0.0),
            div=(0.2, 0.2),
            reference_style="crr_geometric_equivalent",
            reference_steps=4_000,
        ),
        ExperimentCase(
            name="geometric_put_d5",
            S0=tuple([100.0] * 5),
            K=100.0,
            r=0.05,
            sigma=tuple([0.2] * 5),
            T=1.0,
            steps=10,
            num_paths=100_000,
            call=False,
            payoff_style="geometric_basket",
            corr=equicorrelation_matrix(5, 0.2),
            reference_style="crr_geometric_equivalent",
            reference_steps=4_000,
        ),
        ExperimentCase(
            name="geometric_put_d10",
            S0=tuple([100.0] * 10),
            K=100.0,
            r=0.05,
            sigma=tuple([0.2] * 10),
            T=1.0,
            steps=10,
            num_paths=100_000,
            call=False,
            payoff_style="geometric_basket",
            corr=equicorrelation_matrix(10, 0.2),
            reference_style="crr_geometric_equivalent",
            reference_steps=4_000,
        ),
        ExperimentCase(
            name="geometric_put_d20",
            S0=tuple([100.0] * 20),
            K=100.0,
            r=0.05,
            sigma=tuple([0.2] * 20),
            T=1.0,
            steps=10,
            num_paths=100_000,
            call=False,
            payoff_style="geometric_basket",
            corr=equicorrelation_matrix(20, 0.2),
            reference_style="crr_geometric_equivalent",
            reference_steps=4_000,
        ),
        ExperimentCase(
            name="basket_put_d5",
            S0=tuple([100.0] * 5),
            K=100.0,
            r=0.05,
            sigma=tuple([0.2] * 5),
            T=1.0,
            steps=10,
            num_paths=100_000,
            call=False,
            payoff_style="arithmetic_basket",
            corr=equicorrelation_matrix(5, 0.2),
            weights=tuple([1.0 / 5.0] * 5),
        ),
        ExperimentCase(
            name="basket_put_d10",
            S0=tuple([100.0] * 10),
            K=100.0,
            r=0.05,
            sigma=tuple([0.2] * 10),
            T=1.0,
            steps=10,
            num_paths=100_000,
            call=False,
            payoff_style="arithmetic_basket",
            corr=equicorrelation_matrix(10, 0.2),
            weights=tuple([1.0 / 10.0] * 10),
        ),
        ExperimentCase(
            name="basket_call_d10",
            S0=tuple([100.0] * 10),
            K=100.0,
            r=0.05,
            sigma=tuple([0.2] * 10),
            T=1.0,
            steps=10,
            num_paths=100_000,
            call=True,
            payoff_style="arithmetic_basket",
            corr=equicorrelation_matrix(10, 0.2),
            weights=tuple([1.0 / 10.0] * 10),
        ),
        ExperimentCase(
            name="basket_call_d20",
            S0=tuple([100.0] * 20),
            K=100.0,
            r=0.05,
            sigma=tuple([0.2] * 20),
            T=1.0,
            steps=10,
            num_paths=100_000,
            call=True,
            payoff_style="arithmetic_basket",
            corr=equicorrelation_matrix(20, 0.2),
            weights=tuple([1.0 / 20.0] * 20),
        ),
        ExperimentCase(
            name="max_call_d5",
            S0=tuple([100.0] * 5),
            K=100.0,
            r=0.05,
            sigma=tuple([0.2] * 5),
            T=3.0,
            steps=9,
            num_paths=100_000,
            call=True,
            payoff_style="max_basket",
            corr=equicorrelation_matrix(5, 0.0),
            div=tuple([0.1] * 5),
        ),
        ExperimentCase(
            name="max_call_d10",
            S0=tuple([100.0] * 10),
            K=100.0,
            r=0.05,
            sigma=tuple([0.2] * 10),
            T=3.0,
            steps=9,
            num_paths=100_000,
            call=True,
            payoff_style="max_basket",
            corr=equicorrelation_matrix(10, 0.0),
            div=tuple([0.1] * 10),
        ),
        ExperimentCase(
            name="max_call_d20",
            S0=tuple([100.0] * 20),
            K=100.0,
            r=0.05,
            sigma=tuple([0.2] * 20),
            T=3.0,
            steps=9,
            num_paths=100_000,
            call=True,
            payoff_style="max_basket",
            corr=equicorrelation_matrix(20, 0.0),
            div=tuple([0.1] * 20),
        ),
    ]


def available_case_names() -> list[str]:
    return [case.name for case in default_black_scholes_cases()]


def _select_cases(case_names: list[str] | None = None) -> list[ExperimentCase]:
    cases = default_black_scholes_cases()
    if case_names is None:
        return cases

    selected = set(case_names)
    filtered = [case for case in cases if case.name in selected]
    missing = selected.difference({case.name for case in filtered})
    if missing:
        raise ValueError(f"Unknown case names: {sorted(missing)}")
    return filtered


def run_single_case(
    case: ExperimentCase,
    *,
    seed: int,
    mlp_kwargs: dict[str, Any] | None = None,
    include_polynomial_baseline: bool = True,
) -> dict[str, Any]:
    payoff_fn = _payoff_fn(case)
    s0 = _as_array(case.S0)
    sigma = _as_array(case.sigma)
    corr = _as_matrix(case.corr)
    div = _as_array(case.div)

    pricer = MonteCarloPricing(
        S_0=s0,
        X=case.K,
        sigma=sigma,
        T=case.T,
        r=case.r,
        num_paths=case.num_paths,
        steps=case.steps,
        seed=seed,
        corr=corr,
        div=div,
    )
    paths = pricer._simulate_paths(antithetic=case.antithetic)

    nlsm_params = {
        "call": case.call,
        "include_all_paths": False,
        "mask_tolerance": 0.0,
        "hidden_sizes": (32, 32),
        "lr": 1e-3,
        "epochs": 1,
        "batch_size": 2_000,
        "min_samples": 8,
        "warm_start": False,
        "seed": seed,
        "payoff_fn": payoff_fn,
        "state_fn": None,
    }
    if mlp_kwargs:
        nlsm_params.update(mlp_kwargs)

    start = perf_counter()
    mlp_result = american_nlsm_pricing_from_paths(
        paths=paths,
        K=case.K,
        r=case.r,
        T=case.T,
        **nlsm_params,
    )
    mlp_runtime = perf_counter() - start

    poly_price = np.nan
    poly_stderr = np.nan
    poly_runtime = np.nan
    if include_polynomial_baseline:
        start = perf_counter()
        poly_price, poly_stderr = pricer.american(
            call=case.call,
            basis_fn=case.polynomial_basis,
            antithetic=case.antithetic,
            include_all_paths=False,
            mask_tolerance=0.0,
            paths=paths,
            payoff_fn=payoff_fn,
            state_fn=None,
        )
        poly_runtime = perf_counter() - start

    ref_price = reference_price(case)
    record = {
        "case": case.name,
        "seed": seed,
        "dimension": case.dimension,
        "payoff_style": case.payoff_style,
        "num_paths": case.num_paths,
        "steps": case.steps,
        "mlp_price": mlp_result.price,
        "mlp_stderr": mlp_result.stderr,
        "mlp_runtime_sec": mlp_runtime,
        "poly_basis": case.polynomial_basis if include_polynomial_baseline else "",
        "poly_price": poly_price,
        "poly_stderr": poly_stderr,
        "poly_runtime_sec": poly_runtime,
        "reference_price": ref_price if ref_price is not None else np.nan,
    }
    if ref_price is not None:
        record["mlp_abs_error"] = abs(mlp_result.price - ref_price)
        record["poly_abs_error"] = abs(poly_price - ref_price) if include_polynomial_baseline else np.nan
    else:
        record["mlp_abs_error"] = np.nan
        record["poly_abs_error"] = np.nan
    return record


def run_single_case_models(
    case: ExperimentCase,
    *,
    seed: int,
    mlp_model_specs: list[MLPModelSpec],
    polynomial_bases: list[str],
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    payoff_fn = _payoff_fn(case)
    s0 = _as_array(case.S0)
    sigma = _as_array(case.sigma)
    corr = _as_matrix(case.corr)
    div = _as_array(case.div)

    pricer = MonteCarloPricing(
        S_0=s0,
        X=case.K,
        sigma=sigma,
        T=case.T,
        r=case.r,
        num_paths=case.num_paths,
        steps=case.steps,
        seed=seed,
        corr=corr,
        div=div,
    )
    paths = pricer._simulate_paths(antithetic=case.antithetic)
    ref_price = reference_price(case)

    records: list[dict[str, Any]] = []

    spec_iter = mlp_model_specs
    if show_progress:
        spec_iter = tqdm(
            mlp_model_specs,
            total=len(mlp_model_specs),
            desc=f"{case.name} seed={seed} NLSM",
            leave=False,
            dynamic_ncols=True,
        )

    for spec in spec_iter:
        start = perf_counter()
        mlp_result = american_nlsm_pricing_from_paths(
            paths=paths,
            K=case.K,
            r=case.r,
            T=case.T,
            call=case.call,
            include_all_paths=False,
            mask_tolerance=0.0,
            hidden_sizes=spec.hidden_sizes,
            lr=1e-3,
            epochs=spec.epochs,
            batch_size=2_000,
            min_samples=8,
            warm_start=False,
            seed=seed,
            payoff_fn=payoff_fn,
            state_fn=None,
            show_progress=show_progress,
            progress_label=f"{case.name}|seed={seed}|{spec.name}",
        )
        runtime_sec = perf_counter() - start
        records.append(
            {
                "case": case.name,
                "seed": seed,
                "dimension": case.dimension,
                "payoff_style": case.payoff_style,
                "num_paths": case.num_paths,
                "steps": case.steps,
                "model_family": "nlsm",
                "model_name": spec.name,
                "model_detail": (
                    f"layers={len(spec.hidden_sizes)}, width={spec.hidden_sizes[0]}, epochs={spec.epochs}"
                ),
                "price": mlp_result.price,
                "stderr": mlp_result.stderr,
                "runtime_sec": runtime_sec,
                "reference_price": ref_price if ref_price is not None else np.nan,
                "abs_error": abs(mlp_result.price - ref_price) if ref_price is not None else np.nan,
            }
        )

    poly_iter = polynomial_bases
    if show_progress:
        poly_iter = tqdm(
            polynomial_bases,
            total=len(polynomial_bases),
            desc=f"{case.name} seed={seed} baseline",
            leave=False,
            dynamic_ncols=True,
        )

    for basis in poly_iter:
        start = perf_counter()
        poly_price, poly_stderr = pricer.american(
            call=case.call,
            basis_fn=basis,
            antithetic=case.antithetic,
            include_all_paths=False,
            mask_tolerance=0.0,
            paths=paths,
            payoff_fn=payoff_fn,
            state_fn=None,
        )
        runtime_sec = perf_counter() - start
        records.append(
            {
                "case": case.name,
                "seed": seed,
                "dimension": case.dimension,
                "payoff_style": case.payoff_style,
                "num_paths": case.num_paths,
                "steps": case.steps,
                "model_family": "poly",
                "model_name": f"poly_{basis}",
                "model_detail": basis,
                "price": poly_price,
                "stderr": poly_stderr,
                "runtime_sec": runtime_sec,
                "reference_price": ref_price if ref_price is not None else np.nan,
                "abs_error": abs(poly_price - ref_price) if ref_price is not None else np.nan,
            }
        )

    return records


def _mean_ci95(values: np.ndarray) -> tuple[float, float, float]:
    if values.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(values))
    if values.size == 1:
        return mean, mean, mean
    std = float(np.std(values, ddof=1))
    half = 1.96 * std / np.sqrt(values.size)
    return mean, mean - half, mean + half


def summarize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (
            str(record["case"]),
            int(record["num_paths"]),
            str(record["model_family"]),
            str(record["model_name"]),
        )
        grouped.setdefault(key, []).append(record)

    summary: list[dict[str, Any]] = []
    for (case_name, num_paths, model_family, model_name), rows in grouped.items():
        first = rows[0]
        prices = np.asarray([float(row["price"]) for row in rows], dtype=float)
        errors = np.asarray([float(row["abs_error"]) for row in rows], dtype=float)
        runtimes = np.asarray([float(row["runtime_sec"]) for row in rows], dtype=float)
        price_mean, price_ci95_low, price_ci95_high = _mean_ci95(prices)
        abs_error_mean, abs_error_ci95_low, abs_error_ci95_high = _mean_ci95(errors)
        summary.append(
            {
                "case": case_name,
                "model_family": model_family,
                "model_name": model_name,
                "model_detail": first["model_detail"],
                "dimension": first["dimension"],
                "payoff_style": first["payoff_style"],
                "num_paths": num_paths,
                "steps": first["steps"],
                "seed_count": len(rows),
                "price_mean": price_mean,
                "price_std": float(np.std(prices, ddof=1)) if len(rows) > 1 else 0.0,
                "price_ci95_low": price_ci95_low,
                "price_ci95_high": price_ci95_high,
                "stderr_mean": float(np.mean([float(row["stderr"]) for row in rows])),
                "runtime_mean_sec": float(np.mean(runtimes)),
                "reference_price": first["reference_price"],
                "abs_error_mean": abs_error_mean,
                "abs_error_ci95_low": abs_error_ci95_low,
                "abs_error_ci95_high": abs_error_ci95_high,
            }
        )
    return sorted(summary, key=lambda row: (row["case"], row["num_paths"], row["model_family"], row["model_name"]))


def _write_csv(path: Path | str, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_black_scholes_suite(
    *,
    case_names: list[str] | None = None,
    seeds: list[int] | None = None,
    mlp_model_specs: list[MLPModelSpec] | None = None,
    polynomial_bases: list[str] | None = None,
    raw_output_csv: Path | str | None = None,
    summary_output_csv: Path | str | None = None,
    show_progress: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    cases = _select_cases(case_names)
    seeds = [101, 202, 303] if seeds is None else list(seeds)
    mlp_model_specs = DEFAULT_MLP_MODEL_SPECS if mlp_model_specs is None else list(mlp_model_specs)
    polynomial_bases = DEFAULT_POLYNOMIAL_BASES if polynomial_bases is None else list(polynomial_bases)

    records: list[dict[str, Any]] = []
    tasks = [(case, seed) for case in cases for seed in seeds]
    task_iter = tasks
    if show_progress:
        task_iter = tqdm(tasks, total=len(tasks), desc="Main runs", dynamic_ncols=True)

    for case, seed in task_iter:
        if show_progress and hasattr(task_iter, "set_postfix_str"):
            task_iter.set_postfix_str(f"{case.name} seed={seed}")
        records.extend(
            run_single_case_models(
                case,
                seed=seed,
                mlp_model_specs=mlp_model_specs,
                polynomial_bases=polynomial_bases,
                show_progress=show_progress,
            )
        )

    summary = summarize_records(records)
    if raw_output_csv is not None:
        _write_csv(raw_output_csv, records)
    if summary_output_csv is not None:
        _write_csv(summary_output_csv, summary)
    return {"records": records, "summary": summary}


def run_path_count_study(
    *,
    case_names: list[str],
    path_counts: list[int],
    seeds: list[int] | None = None,
    mlp_model_specs: list[MLPModelSpec] | None = None,
    polynomial_bases: list[str] | None = None,
    raw_output_csv: Path | str | None = None,
    summary_output_csv: Path | str | None = None,
    show_progress: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    base_cases = _select_cases(case_names)
    seeds = [101, 202, 303] if seeds is None else list(seeds)
    mlp_model_specs = DEFAULT_MLP_MODEL_SPECS if mlp_model_specs is None else list(mlp_model_specs)
    polynomial_bases = DEFAULT_POLYNOMIAL_BASES if polynomial_bases is None else list(polynomial_bases)

    if not path_counts:
        raise ValueError("path_counts must contain at least one value.")

    records: list[dict[str, Any]] = []
    tasks = [
        (replace(base_case, num_paths=int(num_paths)), seed)
        for base_case in base_cases
        for num_paths in path_counts
        for seed in seeds
    ]
    task_iter = tasks
    if show_progress:
        task_iter = tqdm(tasks, total=len(tasks), desc="Path-study runs", dynamic_ncols=True)

    for case, seed in task_iter:
        if show_progress and hasattr(task_iter, "set_postfix_str"):
            task_iter.set_postfix_str(f"{case.name} n={case.num_paths} seed={seed}")
        records.extend(
            run_single_case_models(
                case,
                seed=seed,
                mlp_model_specs=mlp_model_specs,
                polynomial_bases=polynomial_bases,
                show_progress=show_progress,
            )
        )

    summary = summarize_records(records)
    if raw_output_csv is not None:
        _write_csv(raw_output_csv, records)
    if summary_output_csv is not None:
        _write_csv(summary_output_csv, summary)
    return {"records": records, "summary": summary}


def _latex_escape(text: str) -> str:
    escaped = text.replace("\\", "\\textbackslash{}")
    for old, new in [
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
    ]:
        escaped = escaped.replace(old, new)
    return escaped


def _ordered_model_names(
    *,
    mlp_model_specs: list[MLPModelSpec],
    polynomial_bases: list[str],
    summary_rows: list[dict[str, Any]],
) -> list[str]:
    desired = [spec.name for spec in mlp_model_specs] + [f"poly_{basis}" for basis in polynomial_bases]
    available = {str(row["model_name"]) for row in summary_rows}
    return [name for name in desired if name in available]


def _format_ci_cell(mean: float, ci_low: float, ci_high: float) -> str:
    if np.isnan(mean):
        return ""
    if np.isnan(ci_low) or np.isnan(ci_high):
        return f"{mean:.4f}"
    half_width = max(ci_high - mean, mean - ci_low, 0.0)
    return f"{mean:.4f} +/- {half_width:.4f}"


def build_paper_table_rows(
    summary_rows: list[dict[str, Any]],
    *,
    mlp_model_specs: list[MLPModelSpec],
    polynomial_bases: list[str],
) -> list[dict[str, Any]]:
    ordered_models = _ordered_model_names(
        mlp_model_specs=mlp_model_specs,
        polynomial_bases=polynomial_bases,
        summary_rows=summary_rows,
    )
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in summary_rows:
        key = (str(row["case"]), int(row["num_paths"]))
        grouped.setdefault(key, []).append(row)

    paper_rows: list[dict[str, Any]] = []
    for (case_name, num_paths), rows in sorted(grouped.items()):
        first = rows[0]
        paper_row: dict[str, Any] = {
            "Case": case_name,
            "Dim": int(first["dimension"]),
            "Payoff": str(first["payoff_style"]),
            "Paths": num_paths,
            "Reference": "" if np.isnan(first["reference_price"]) else f"{float(first['reference_price']):.4f}",
        }
        row_map = {str(row["model_name"]): row for row in rows}
        for model_name in ordered_models:
            row = row_map.get(model_name)
            if row is None:
                paper_row[model_name] = ""
                continue
            paper_row[model_name] = _format_ci_cell(
                float(row["price_mean"]),
                float(row["price_ci95_low"]),
                float(row["price_ci95_high"]),
            )
        paper_rows.append(paper_row)
    return paper_rows


def export_paper_table(
    summary_rows: list[dict[str, Any]],
    *,
    mlp_model_specs: list[MLPModelSpec],
    polynomial_bases: list[str],
    csv_path: Path | str,
    tex_path: Path | str,
    caption: str,
    label: str,
) -> list[dict[str, Any]]:
    rows = build_paper_table_rows(
        summary_rows,
        mlp_model_specs=mlp_model_specs,
        polynomial_bases=polynomial_bases,
    )
    _write_csv(csv_path, rows)

    if not rows:
        return rows

    destination = Path(tex_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys())
    align = "l" * min(4, len(columns)) + "c" * max(len(columns) - 4, 0)
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\label{{{_latex_escape(label)}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\hline",
        " & ".join(_latex_escape(column) for column in columns) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(" & ".join(_latex_escape(str(row[column])) for column in columns) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def _print_available_cases() -> None:
    print("Available cases:")
    for name in available_case_names():
        print(f"  - {name}")


def _print_summary(summary: list[dict[str, Any]]) -> None:
    if not summary:
        print("No results.")
        return

    for row in summary:
        ref = row["reference_price"]
        ref_text = "n/a" if np.isnan(ref) else f"{ref:.6f}"
        err = row["abs_error_mean"]
        err_text = "n/a" if np.isnan(err) else f"{err:.6f}"
        ci_low = row["price_ci95_low"]
        ci_high = row["price_ci95_high"]
        ci_text = "n/a" if np.isnan(ci_low) or np.isnan(ci_high) else f"[{ci_low:.6f}, {ci_high:.6f}]"
        print(
            (
                f"{row['case']}: "
                f"{row['model_name']}={row['price_mean']:.6f} "
                f"(std={row['price_std']:.6f}, ci95={ci_text}, err={err_text}), "
                f"ref={ref_text}"
            )
        )


# Initial architecture study: 5D put basket, standard Black-Scholes inputs, 100k paths.
RUN_ALL_CASES = False
CASE_NAMES = ["basket_put_d5"]
SEEDS = [101, 202, 303]
SHOW_PROGRESS = True
OUTPUT_DIR: str | None = "results/lsm_nlsm"
RUN_PATH_COUNT_STUDY = False
PATH_COUNT_STUDY_CASE_NAMES = ["basket_put_d5"]
PATH_COUNT_STUDY_PATHS = [100_000]
PATH_COUNT_STUDY_SEEDS = [101, 202, 303]

DEFAULT_MLP_MODEL_SPECS = [
    MLPModelSpec(
        name=f"nlsm_l{depth}_w{width}_e{epochs}",
        hidden_sizes=tuple([width] * depth),
        epochs=epochs,
        first_step_epochs=None,
    )
    for depth in (2, 4, 8)
    for width in (32, 128, 512)
    for epochs in (1, 5, 10)
]
DEFAULT_POLYNOMIAL_BASES = ["laguerre"]


def main() -> int:
    case_names = None if RUN_ALL_CASES else CASE_NAMES
    seeds = SEEDS
    output_dir = Path(OUTPUT_DIR) if OUTPUT_DIR is not None else None
    raw_output_csv = output_dir / "lsm_nlsm_raw.csv" if output_dir is not None else None
    summary_output_csv = output_dir / "lsm_nlsm_summary.csv" if output_dir is not None else None
    paper_table_csv = output_dir / "lsm_nlsm_paper_table.csv" if output_dir is not None else None
    paper_table_tex = output_dir / "lsm_nlsm_paper_table.tex" if output_dir is not None else None
    path_study_raw_csv = output_dir / "lsm_nlsm_path_study_raw.csv" if output_dir is not None else None
    path_study_summary_csv = output_dir / "lsm_nlsm_path_study_summary.csv" if output_dir is not None else None
    path_study_table_csv = output_dir / "lsm_nlsm_path_study_paper_table.csv" if output_dir is not None else None
    path_study_table_tex = output_dir / "lsm_nlsm_path_study_paper_table.tex" if output_dir is not None else None

    if not RUN_ALL_CASES and not CASE_NAMES:
        _print_available_cases()
        print()
        print("Edit CASE_NAMES or set RUN_ALL_CASES = True near the bottom of this file.")
        return 0

    selected_case_count = len(default_black_scholes_cases()) if RUN_ALL_CASES else len(case_names)
    print(
        (
            f"Running {selected_case_count} case(s) across {len(seeds)} seed(s). "
            f"NLSM specs={len(DEFAULT_MLP_MODEL_SPECS)}, polynomial bases={len(DEFAULT_POLYNOMIAL_BASES)}."
        ),
        flush=True,
    )
    if output_dir is not None:
        print(f"Output directory: {output_dir}", flush=True)

    main_result = run_black_scholes_suite(
        case_names=case_names,
        seeds=seeds,
        mlp_model_specs=DEFAULT_MLP_MODEL_SPECS,
        polynomial_bases=DEFAULT_POLYNOMIAL_BASES,
        raw_output_csv=raw_output_csv,
        summary_output_csv=summary_output_csv,
        show_progress=SHOW_PROGRESS,
    )
    _print_summary(main_result["summary"])

    if paper_table_csv is not None and paper_table_tex is not None:
        export_paper_table(
            main_result["summary"],
            mlp_model_specs=DEFAULT_MLP_MODEL_SPECS,
            polynomial_bases=DEFAULT_POLYNOMIAL_BASES,
            csv_path=paper_table_csv,
            tex_path=paper_table_tex,
            caption="LSM NLSM and polynomial regression results across the main case grid.",
            label="tab:lsm_nlsm_main_results",
        )

    path_study_result: dict[str, list[dict[str, Any]]] | None = None
    if RUN_PATH_COUNT_STUDY:
        print()
        print(
            (
                f"Running path-count study on {len(PATH_COUNT_STUDY_CASE_NAMES)} case(s) "
                f"across {len(PATH_COUNT_STUDY_PATHS)} path counts."
            ),
            flush=True,
        )
        path_study_result = run_path_count_study(
            case_names=PATH_COUNT_STUDY_CASE_NAMES,
            path_counts=PATH_COUNT_STUDY_PATHS,
            seeds=PATH_COUNT_STUDY_SEEDS,
            mlp_model_specs=DEFAULT_MLP_MODEL_SPECS,
            polynomial_bases=DEFAULT_POLYNOMIAL_BASES,
            raw_output_csv=path_study_raw_csv,
            summary_output_csv=path_study_summary_csv,
            show_progress=SHOW_PROGRESS,
        )
        if path_study_table_csv is not None and path_study_table_tex is not None:
            export_paper_table(
                path_study_result["summary"],
                mlp_model_specs=DEFAULT_MLP_MODEL_SPECS,
                polynomial_bases=DEFAULT_POLYNOMIAL_BASES,
                csv_path=path_study_table_csv,
                tex_path=path_study_table_tex,
                caption="Path-count sensitivity for LSM NLSM and polynomial regression models.",
                label="tab:lsm_nlsm_path_study",
            )

    if output_dir is not None:
        print()
        print(f"Wrote {raw_output_csv}")
        print(f"Wrote {summary_output_csv}")
        print(f"Wrote {paper_table_csv}")
        print(f"Wrote {paper_table_tex}")
        if RUN_PATH_COUNT_STUDY:
            print(f"Wrote {path_study_raw_csv}")
            print(f"Wrote {path_study_summary_csv}")
            print(f"Wrote {path_study_table_csv}")
            print(f"Wrote {path_study_table_tex}")
    return 0


__all__ = [
    "ExperimentCase",
    "MLPModelSpec",
    "arithmetic_basket_payoff",
    "available_case_names",
    "default_black_scholes_cases",
    "equicorrelation_matrix",
    "geometric_basket_payoff",
    "max_basket_payoff",
    "reference_price",
    "run_path_count_study",
    "run_black_scholes_suite",
    "run_single_case",
    "run_single_case_models",
    "summarize_records",
    "build_paper_table_rows",
    "export_paper_table",
]


if __name__ == "__main__":
    raise SystemExit(main())
