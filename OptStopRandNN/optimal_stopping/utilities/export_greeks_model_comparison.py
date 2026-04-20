import argparse
from pathlib import Path

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PACKAGE_ROOT / "output"
METRICS_DIR = OUTPUT_ROOT / "metrics_draft"
LATEX_ROOT = OUTPUT_ROOT / "latex_tables"

ALGO_ORDER = ["LSM", "NLSM", "RLSM"]
SAMPLER_ORDER = [
    "mc",
    "mc_antithetic",
    "sobol_seq",
    "sobol_bb",
    "sobol_scrambled_seq",
    "sobol_scrambled_bb",
]
SAMPLER_LABELS = {
    "mc": "MC",
    "mc_antithetic": "MC + AV",
    "sobol_seq": "QMC (Seq)",
    "sobol_bb": "QMC (BB)",
    "sobol_scrambled_seq": "RQMC (Seq)",
    "sobol_scrambled_bb": "RQMC (BB)",
}
METRICS = ["price", "delta", "gamma", "vega", "theta", "rho"]
METRIC_LABELS = {
    "price": "Price",
    "delta": "Delta",
    "gamma": "Gamma",
    "vega": "Vega",
    "theta": "Theta",
    "rho": "Rho",
}


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "nb_dates",
        "nb_paths",
        "spot",
        "strike",
        "volatility",
        "drift",
        "maturity",
        "duration",
        "delta",
        "gamma",
        "theta",
        "rho",
        "vega",
        "price",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _format_stat(mean_value: float, std_value: float, *, digits: int = 6) -> str:
    if pd.isna(mean_value):
        return ""
    if pd.isna(std_value):
        return f"{mean_value:.{digits}f}"
    return f"{mean_value:.{digits}f} +/- {std_value:.{digits}f}"


def get_run_output_dir(csv_path: Path) -> Path:
    return LATEX_ROOT / csv_path.stem


def load_metrics(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df = _coerce_numeric(df)
    df["path_sampler"] = df["path_sampler"].fillna("mc")
    df.loc[df["algo"] == "B", "path_sampler"] = "tree"
    return df


def build_benchmark_map(df: pd.DataFrame) -> tuple[dict[str, float], int]:
    tree_rows = df.loc[df["algo"] == "B"].copy()
    if tree_rows.empty:
        raise ValueError("No binomial benchmark rows found in the metrics CSV.")
    available_steps = sorted(int(value) for value in tree_rows["nb_dates"].dropna().unique())
    default_steps = 2000 if 2000 in available_steps else available_steps[0]
    benchmark_row = tree_rows.loc[tree_rows["nb_dates"] == default_steps].iloc[0]
    benchmark_map = {metric: float(benchmark_row[metric]) for metric in METRICS}
    return benchmark_map, default_steps


def build_summary(df: pd.DataFrame, *, benchmark_map: dict[str, float], benchmark_steps: int) -> pd.DataFrame:
    compare_df = df.loc[df["algo"].isin(ALGO_ORDER) & df["path_sampler"].isin(SAMPLER_ORDER)].copy()
    summary = (
        compare_df.groupby(["algo", "path_sampler"], dropna=False)
        .agg(
            runs=("price", "size"),
            price_mean=("price", "mean"),
            price_std=("price", "std"),
            delta_mean=("delta", "mean"),
            delta_std=("delta", "std"),
            gamma_mean=("gamma", "mean"),
            gamma_std=("gamma", "std"),
            vega_mean=("vega", "mean"),
            vega_std=("vega", "std"),
            theta_mean=("theta", "mean"),
            theta_std=("theta", "std"),
            rho_mean=("rho", "mean"),
            rho_std=("rho", "std"),
            duration_mean=("duration", "mean"),
            duration_std=("duration", "std"),
        )
        .reset_index()
    )

    summary["algo"] = pd.Categorical(summary["algo"], categories=ALGO_ORDER, ordered=True)
    summary["path_sampler"] = pd.Categorical(summary["path_sampler"], categories=SAMPLER_ORDER, ordered=True)
    summary = summary.sort_values(["algo", "path_sampler"]).reset_index(drop=True)
    summary["benchmark_steps"] = benchmark_steps
    for metric in METRICS:
        summary[f"{metric}_benchmark"] = benchmark_map[metric]
        summary[f"{metric}_abs_error_mean"] = (summary[f"{metric}_mean"] - benchmark_map[metric]).abs()
    return summary


def build_paper_rows(summary: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for algo in ALGO_ORDER:
        algo_rows = summary.loc[summary["algo"] == algo]
        if algo_rows.empty:
            continue
        sampler_map = {str(row["path_sampler"]): row for _, row in algo_rows.iterrows()}
        for metric in METRICS:
            row = {
                "Model": algo,
                "Metric": METRIC_LABELS[metric],
                "Benchmark": f"{float(algo_rows.iloc[0][f'{metric}_benchmark']):.6f}",
            }
            for sampler in SAMPLER_ORDER:
                sampler_row = sampler_map.get(sampler)
                row[SAMPLER_LABELS[sampler]] = "" if sampler_row is None else _format_stat(
                    float(sampler_row[f"{metric}_mean"]),
                    float(sampler_row[f"{metric}_std"]),
                )
            rows.append(row)
    return rows


def build_runtime_rows(summary: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for algo in ALGO_ORDER:
        algo_rows = summary.loc[summary["algo"] == algo]
        if algo_rows.empty:
            continue
        sampler_map = {str(row["path_sampler"]): row for _, row in algo_rows.iterrows()}
        row = {"Model": algo}
        for sampler in SAMPLER_ORDER:
            sampler_row = sampler_map.get(sampler)
            row[SAMPLER_LABELS[sampler]] = "" if sampler_row is None else _format_stat(
                float(sampler_row["duration_mean"]),
                float(sampler_row["duration_std"]),
            )
        rows.append(row)
    return rows


def build_abs_error_rows(summary: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for algo in ALGO_ORDER:
        algo_rows = summary.loc[summary["algo"] == algo]
        if algo_rows.empty:
            continue
        sampler_map = {str(row["path_sampler"]): row for _, row in algo_rows.iterrows()}
        for metric in METRICS:
            row = {"Model": algo, "Metric": METRIC_LABELS[metric]}
            for sampler in SAMPLER_ORDER:
                sampler_row = sampler_map.get(sampler)
                row[SAMPLER_LABELS[sampler]] = "" if sampler_row is None else f"{float(sampler_row[f'{metric}_abs_error_mean']):.6f}"
            rows.append(row)
    return rows


def write_table(rows: list[dict[str, str]], *, csv_path: Path, tex_path: Path, caption: str, label: str) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    latex = pd.DataFrame(rows).to_latex(index=False, escape=False, caption=caption, label=label)
    tex_path.write_text(latex, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export OptStopRandNN 1D Greek model-comparison tables.")
    parser.add_argument("--csv-filename", help="CSV file name inside output/metrics_draft")
    parser.add_argument("--csv-path", help="Absolute or relative path to a metrics CSV")
    args = parser.parse_args()

    if args.csv_path:
        csv_path = Path(args.csv_path).expanduser().resolve()
    elif args.csv_filename:
        csv_path = METRICS_DIR / args.csv_filename
    else:
        raise SystemExit("Pass --csv-filename or --csv-path.")

    df = load_metrics(csv_path)
    benchmark_map, benchmark_steps = build_benchmark_map(df)
    summary = build_summary(df, benchmark_map=benchmark_map, benchmark_steps=benchmark_steps)

    run_dir = get_run_output_dir(csv_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem

    summary_path = run_dir / f"{stem}_model_compare_summary.csv"
    summary.to_csv(summary_path, index=False)

    paper_rows = build_paper_rows(summary)
    write_table(
        paper_rows,
        csv_path=run_dir / f"{stem}_model_compare_paper_table.csv",
        tex_path=run_dir / f"{stem}_model_compare_paper_table.tex",
        caption=(
            "OptStopRandNN one-dimensional American put price and Greek estimates across "
            "LSM, NLSM, and RLSM under MC, MC plus antithetic variates, QMC, and randomized QMC."
        ),
        label=f"tab:{stem}_model_compare",
    )

    runtime_rows = build_runtime_rows(summary)
    write_table(
        runtime_rows,
        csv_path=run_dir / f"{stem}_model_compare_runtime_table.csv",
        tex_path=run_dir / f"{stem}_model_compare_runtime_table.tex",
        caption="Mean runtime in seconds for the OptStopRandNN 1D model-comparison run.",
        label=f"tab:{stem}_model_compare_runtime",
    )

    abs_error_rows = build_abs_error_rows(summary)
    write_table(
        abs_error_rows,
        csv_path=run_dir / f"{stem}_model_compare_abs_error_table.csv",
        tex_path=run_dir / f"{stem}_model_compare_abs_error_table.tex",
        caption="Absolute error against the binomial benchmark for the OptStopRandNN 1D model-comparison run.",
        label=f"tab:{stem}_model_compare_abs_error",
    )

    print(f"Wrote summary CSV to {summary_path}")
    print(f"Wrote outputs to {run_dir}")


if __name__ == "__main__":
    main()
