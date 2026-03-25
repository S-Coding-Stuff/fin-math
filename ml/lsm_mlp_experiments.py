import numpy as np
from time import perf_counter
from typing import Any, Callable
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engines.monte_carlo import MonteCarloPricing
from ml.lsm_mlp import american_mlp_pricing_from_paths

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
    polynomial_basis: str = "paper_poly2"
    reference_style: str | None = None
    reference_steps: int | None = None

    @property
    def dimension(self) -> int:
        if isinstance(self.S0, tuple):
            return len(self.S0)
        return 1


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


def _weighted_basket_value(states: np.ndarray, weights: np.ndarray) -> np.ndarray:
    arr = np.asarray(states, dtype=float)
    if arr.ndim == 3:
        return np.tensordot(arr, weights, axes=([2], [0]))
    if arr.ndim == 2:
        return arr @ weights
    raise ValueError("states must be a 2D or 3D array.")


def geometric_basket_payoff(states: np.ndarray, *, strike: float, call: bool) -> np.ndarray:
    arr = np.asarray(states, dtype=float)
    if arr.ndim not in (2, 3):
        raise ValueError("states must be a 2D or 3D array.")
    basket = np.exp(np.mean(np.log(np.maximum(arr, 1e-12)), axis=-1))
    if call:
        return np.maximum(basket - strike, 0.0)
    return np.maximum(strike - basket, 0.0)


def arithmetic_basket_payoff(
    states: np.ndarray,
    *,
    strike: float,
    call: bool,
    weights: np.ndarray,
) -> np.ndarray:
    basket = _weighted_basket_value(states, weights)
    if call:
        return np.maximum(basket - strike, 0.0)
    return np.maximum(strike - basket, 0.0)


def max_basket_payoff(states: np.ndarray, *, strike: float, call: bool) -> np.ndarray:
    arr = np.asarray(states, dtype=float)
    if arr.ndim not in (2, 3):
        raise ValueError("states must be a 2D or 3D array.")
    basket = np.max(arr, axis=-1)
    if call:
        return np.maximum(basket - strike, 0.0)
    return np.maximum(strike - basket, 0.0)


def _payoff_fn(case: ExperimentCase) -> PayoffFn | None:
    if case.payoff_style == "vanilla":
        return None
    if case.payoff_style == "geometric_basket":
        return lambda states: geometric_basket_payoff(states, strike=case.K, call=case.call)
    if case.payoff_style == "arithmetic_basket":
        if case.weights is None:
            raise ValueError(f"{case.name}: arithmetic basket cases require weights.")
        weights = np.asarray(case.weights, dtype=float)
        return lambda states: arithmetic_basket_payoff(states, strike=case.K, call=case.call, weights=weights)
    if case.payoff_style == "max_basket":
        return lambda states: max_basket_payoff(states, strike=case.K, call=case.call)
    raise ValueError(f"Unknown payoff_style: {case.payoff_style}")


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

    asset = np.zeros((steps + 1, steps + 1), dtype=float)
    for i in range(steps + 1):
        for j in range(i + 1):
            asset[j, i] = S0 * (u ** (i - j)) * (d ** j)

    option = np.zeros((steps + 1, steps + 1), dtype=float)
    if call:
        option[:, steps] = np.maximum(asset[:, steps] - K, 0.0)
    else:
        option[:, steps] = np.maximum(K - asset[:, steps], 0.0)

    discount = np.exp(-r * dt)
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            continuation = discount * (p * option[j, i + 1] + (1.0 - p) * option[j + 1, i + 1])
            intrinsic = max(asset[j, i] - K, 0.0) if call else max(K - asset[j, i], 0.0)
            option[j, i] = max(intrinsic, continuation)
    return float(option[0, 0])


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
            polynomial_basis="paper_poly2",
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
            reference_steps=100_000,
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
            reference_steps=100_000,
        ),
        ExperimentCase(
            name="geometric_put_d40_100k",
            S0=tuple([100.0] * 40),
            K=100.0,
            r=0.05,
            sigma=tuple([0.2] * 40),
            T=1.0,
            steps=10,
            num_paths=100_000,
            call=False,
            payoff_style="geometric_basket",
            corr=equicorrelation_matrix(40, 0.2),
            reference_style="crr_geometric_equivalent",
            reference_steps=100_000,
        ),
        ExperimentCase(
            name="geometric_put_d40_1m",
            S0=tuple([100.0] * 40),
            K=100.0,
            r=0.05,
            sigma=tuple([0.2] * 40),
            T=1.0,
            steps=10,
            num_paths=1_000_000,
            call=False,
            payoff_style="geometric_basket",
            corr=equicorrelation_matrix(40, 0.2),
            reference_style="crr_geometric_equivalent",
            reference_steps=100_000,
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
            name="basket_call_d40_100k",
            S0=tuple([100.0] * 40),
            K=100.0,
            r=0.05,
            sigma=tuple([0.2] * 40),
            T=1.0,
            steps=10,
            num_paths=100_000,
            call=True,
            payoff_style="arithmetic_basket",
            corr=equicorrelation_matrix(40, 0.2),
            weights=tuple([1.0 / 40.0] * 40),
        ),
        ExperimentCase(
            name="basket_call_d40_1m",
            S0=tuple([100.0] * 40),
            K=100.0,
            r=0.05,
            sigma=tuple([0.2] * 40),
            T=1.0,
            steps=10,
            num_paths=1_000_000,
            call=True,
            payoff_style="arithmetic_basket",
            corr=equicorrelation_matrix(40, 0.2),
            weights=tuple([1.0 / 40.0] * 40),
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
            name="max_call_d50_100k",
            S0=tuple([100.0] * 50),
            K=100.0,
            r=0.05,
            sigma=tuple([0.2] * 50),
            T=3.0,
            steps=9,
            num_paths=100_000,
            call=True,
            payoff_style="max_basket",
            corr=equicorrelation_matrix(50, 0.0),
            div=tuple([0.1] * 50),
        ),
        ExperimentCase(
            name="max_call_d50_1m",
            S0=tuple([100.0] * 50),
            K=100.0,
            r=0.05,
            sigma=tuple([0.2] * 50),
            T=3.0,
            steps=9,
            num_paths=1_000_000,
            call=True,
            payoff_style="max_basket",
            corr=equicorrelation_matrix(50, 0.0),
            div=tuple([0.1] * 50),
        ),
    ]


def available_case_names() -> list[str]:
    return [case.name for case in default_black_scholes_cases()]


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

    mlp_params = {
        "call": case.call,
        "include_all_paths": False,
        "mask_tolerance": 0.0,
        "hidden_sizes": (32,),
        "negative_slope": 0.3,
        "lr": 1e-3,
        "epochs": 1,
        "first_step_epochs": 10,
        "batch_size": 256,
        "scale_inputs": True,
        "min_samples": 8,
        "warm_start": True,
        "seed": seed,
        "payoff_fn": payoff_fn,
        "state_fn": None,
    }
    if mlp_kwargs:
        mlp_params.update(mlp_kwargs)

    start = perf_counter()
    mlp_result = american_mlp_pricing_from_paths(
        paths=paths,
        K=case.K,
        r=case.r,
        T=case.T,
        **mlp_params,
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


def summarize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["case"]), []).append(record)

    summary: list[dict[str, Any]] = []
    for case_name, rows in grouped.items():
        first = rows[0]
        mlp_prices = np.asarray([float(row["mlp_price"]) for row in rows], dtype=float)
        poly_prices = np.asarray([float(row["poly_price"]) for row in rows], dtype=float)
        summary.append(
            {
                "case": case_name,
                "dimension": first["dimension"],
                "payoff_style": first["payoff_style"],
                "num_paths": first["num_paths"],
                "steps": first["steps"],
                "seed_count": len(rows),
                "mlp_price_mean": float(np.mean(mlp_prices)),
                "mlp_price_std": float(np.std(mlp_prices, ddof=1)) if len(rows) > 1 else 0.0,
                "mlp_runtime_mean_sec": float(np.mean([float(row["mlp_runtime_sec"]) for row in rows])),
                "poly_basis": first["poly_basis"],
                "poly_price_mean": float(np.nanmean(poly_prices)),
                "poly_price_std": float(np.nanstd(poly_prices, ddof=1)) if len(rows) > 1 else 0.0,
                "poly_runtime_mean_sec": float(np.nanmean([float(row["poly_runtime_sec"]) for row in rows])),
                "reference_price": first["reference_price"],
                "mlp_abs_error_mean": float(np.nanmean([float(row["mlp_abs_error"]) for row in rows])),
                "poly_abs_error_mean": float(np.nanmean([float(row["poly_abs_error"]) for row in rows])),
            }
        )
    return summary


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
    mlp_kwargs: dict[str, Any] | None = None,
    include_polynomial_baseline: bool = True,
    raw_output_csv: Path | str | None = None,
    summary_output_csv: Path | str | None = None,
    show_progress: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    cases = default_black_scholes_cases()
    if case_names is not None:
        selected = set(case_names)
        cases = [case for case in cases if case.name in selected]
        missing = selected.difference({case.name for case in cases})
        if missing:
            raise ValueError(f"Unknown case names: {sorted(missing)}")

    seeds = [101, 202, 303] if seeds is None else list(seeds)

    records: list[dict[str, Any]] = []
    total_runs = len(cases) * len(seeds)
    run_idx = 0
    for case in cases:
        for seed in seeds:
            run_idx += 1
            if show_progress:
                print(f"[{run_idx}/{total_runs}] case={case.name} seed={seed}", flush=True)
            records.append(
                run_single_case(
                    case,
                    seed=seed,
                    mlp_kwargs=mlp_kwargs,
                    include_polynomial_baseline=include_polynomial_baseline,
                )
            )

    summary = summarize_records(records)
    if raw_output_csv is not None:
        _write_csv(raw_output_csv, records)
    if summary_output_csv is not None:
        _write_csv(summary_output_csv, summary)
    return {"records": records, "summary": summary}

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
        mlp_err = row["mlp_abs_error_mean"]
        mlp_err_text = "n/a" if np.isnan(mlp_err) else f"{mlp_err:.6f}"
        poly_err = row["poly_abs_error_mean"]
        poly_err_text = "n/a" if np.isnan(poly_err) else f"{poly_err:.6f}"
        print(
            (
                f"{row['case']}: "
                f"mlp={row['mlp_price_mean']:.6f} "
                f"(std={row['mlp_price_std']:.6f}, err={mlp_err_text}), "
                f"poly={row['poly_price_mean']:.6f} "
                f"(std={row['poly_price_std']:.6f}, err={poly_err_text}), "
                f"ref={ref_text}"
            )
        )


# IDE-run settings. Edit these values directly before pressing Run.
# The defaults below run the full dissertation suite.
RUN_ALL_CASES = True
CASE_NAMES: list[str] = []
SEEDS = [101, 202, 303]
INCLUDE_POLYNOMIAL_BASELINE = True
SHOW_PROGRESS = True
OUTPUT_DIR: str | None = "results/lsm_mlp"

MLP_KWARGS: dict[str, Any] = {
    "hidden_sizes": (32,),
    "negative_slope": 0.3,
    "lr": 1e-3,
    "epochs": 1,
    "first_step_epochs": 10,
    "batch_size": 256,
    "min_samples": 8,
    "warm_start": True,
}


def main() -> int:
    case_names = None if RUN_ALL_CASES else CASE_NAMES
    output_dir = Path(OUTPUT_DIR) if OUTPUT_DIR is not None else None
    raw_output_csv = output_dir / "lsm_mlp_raw.csv" if output_dir is not None else None
    summary_output_csv = output_dir / "lsm_mlp_summary.csv" if output_dir is not None else None

    if not RUN_ALL_CASES and not CASE_NAMES:
        _print_available_cases()
        print()
        print("Edit CASE_NAMES or set RUN_ALL_CASES = True near the bottom of this file.")
        return 0

    selected_case_count = len(default_black_scholes_cases()) if RUN_ALL_CASES else len(CASE_NAMES)
    print(
        (
            f"Running {selected_case_count} case(s) across {len(SEEDS)} seed(s). "
            f"Polynomial baseline={'on' if INCLUDE_POLYNOMIAL_BASELINE else 'off'}."
        ),
        flush=True,
    )
    if output_dir is not None:
        print(f"Output directory: {output_dir}", flush=True)

    result = run_black_scholes_suite(
        case_names=case_names,
        seeds=SEEDS,
        mlp_kwargs=MLP_KWARGS or None,
        include_polynomial_baseline=INCLUDE_POLYNOMIAL_BASELINE,
        raw_output_csv=raw_output_csv,
        summary_output_csv=summary_output_csv,
        show_progress=SHOW_PROGRESS,
    )
    _print_summary(result["summary"])
    if output_dir is not None:
        print()
        print(f"Wrote {raw_output_csv}")
        print(f"Wrote {summary_output_csv}")
    return 0


__all__ = [
    "ExperimentCase",
    "arithmetic_basket_payoff",
    "available_case_names",
    "default_black_scholes_cases",
    "equicorrelation_matrix",
    "geometric_basket_payoff",
    "max_basket_payoff",
    "reference_price",
    "run_black_scholes_suite",
    "run_single_case",
    "summarize_records",
]


if __name__ == "__main__":
    raise SystemExit(main())
