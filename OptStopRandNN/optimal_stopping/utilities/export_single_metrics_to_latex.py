from pathlib import Path

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PACKAGE_ROOT / "output"

CSV_FILENAME = "1774613154428.csv"
CSV_PATH = OUTPUT_ROOT / "metrics_draft" / CSV_FILENAME
OUTPUT_DIR = OUTPUT_ROOT / "latex_tables"
INCLUDE_DOS = True

ALGOS_ORDER = [
    "B",
    "LSM",
    "NLSM",
    "RLSM",
    "RLSMTanh",
    "RLSMElu",
    "RLSMSilu",
    "RLSMGelu",
    "RLSMSoftplus",
    "RLSMSoftplusReinit",
    "DOS",
]

SAMPLER_ORDER = [
    "tree",
    "mc",
    "mc_antithetic",
    "sobol",
    "sobol_seq",
    "sobol_bb",
    "sobol_scrambled",
    "sobol_scrambled_seq",
    "sobol_scrambled_bb",
]

ALGO_LABELS = {
    "B": "Binomial",
    "LSM": "LSM",
    "NLSM": "NLSM",
    "RLSM": "RLSM",
    "RLSMTanh": "RLSM Tanh",
    "RLSMElu": "RLSM ELU",
    "RLSMSilu": "RLSM SiLU",
    "RLSMGelu": "RLSM GELU",
    "RLSMSoftplus": "RLSM Softplus",
    "RLSMSoftplusReinit": "RLSM Softplus Reinit",
    "DOS": "DOS",
}

SAMPLER_LABELS = {
    "tree": "Tree",
    "mc": "MC",
    "mc_antithetic": "MC + AV",
    "sobol": "QMC Seq",
    "sobol_seq": "QMC Seq",
    "sobol_bb": "QMC BB",
    "sobol_scrambled": "RQMC Seq",
    "sobol_scrambled_seq": "RQMC Seq",
    "sobol_scrambled_bb": "RQMC BB",
}

METRICS = ["price", "delta", "gamma", "theta", "rho", "vega", "comp_time"]
VALUE_LABELS = {
    "price": "Price",
    "delta": "Delta",
    "gamma": "Gamma",
    "theta": "Theta",
    "rho": "Rho",
    "vega": "Vega",
    "comp_time": "Comp Time (s)",
}


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in [
        "drift",
        "volatility",
        "mean",
        "speed",
        "correlation",
        "hurst",
        "nb_stocks",
        "nb_paths",
        "nb_dates",
        "spot",
        "strike",
        "dividend",
        "maturity",
        "nb_epochs",
        "hidden_size",
        *METRICS,
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _format_stat(mean_value: float, std_value: float, *, is_time: bool = False) -> str:
    if pd.isna(mean_value):
        return "--"
    if pd.isna(std_value):
        return f"{mean_value:.4f}" if is_time else f"{mean_value:.6f}"
    if is_time:
        return f"{mean_value:.4f} ({std_value:.4f})"
    return f"{mean_value:.6f} ({std_value:.6f})"


def load_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {path}")
    df = pd.read_csv(path)
    df = _coerce_numeric(df)
    if not INCLUDE_DOS:
        df = df.loc[df["algo"] != "DOS"].copy()
    df["path_sampler"] = df["path_sampler"].fillna("mc")
    df.loc[df["algo"] == "B", "path_sampler"] = "tree"
    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "algo",
        "path_sampler",
        "strike",
        "nb_dates",
        "greeks_method",
    ]
    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            runs=("price", "size"),
            price_mean=("price", "mean"),
            price_std=("price", "std"),
            delta_mean=("delta", "mean"),
            delta_std=("delta", "std"),
            gamma_mean=("gamma", "mean"),
            gamma_std=("gamma", "std"),
            theta_mean=("theta", "mean"),
            theta_std=("theta", "std"),
            rho_mean=("rho", "mean"),
            rho_std=("rho", "std"),
            vega_mean=("vega", "mean"),
            vega_std=("vega", "std"),
            comp_time_mean=("comp_time", "mean"),
            comp_time_std=("comp_time", "std"),
        )
        .reset_index()
    )

    summary["algo"] = pd.Categorical(summary["algo"], categories=ALGOS_ORDER, ordered=True)
    summary["path_sampler"] = pd.Categorical(
        summary["path_sampler"], categories=SAMPLER_ORDER, ordered=True
    )
    summary = summary.sort_values(
        ["strike", "algo", "path_sampler", "nb_dates", "greeks_method"]
    ).reset_index(drop=True)
    return summary


def build_latex_table(summary: pd.DataFrame) -> pd.DataFrame:
    table = summary.copy()
    table["Algorithm"] = table["algo"].map(ALGO_LABELS).fillna(table["algo"].astype(str))
    table["Sampler"] = table["path_sampler"].map(SAMPLER_LABELS).fillna(
        table["path_sampler"].astype(str)
    )
    table["Strike"] = table["strike"].map(lambda value: f"{value:.0f}" if pd.notna(value) else "--")
    table["Steps"] = table["nb_dates"].map(lambda value: f"{int(value)}" if pd.notna(value) else "--")
    table["Runs"] = table["runs"].astype(int)
    table["Price"] = table.apply(
        lambda row: _format_stat(row["price_mean"], row["price_std"]), axis=1
    )
    table["Delta"] = table.apply(
        lambda row: _format_stat(row["delta_mean"], row["delta_std"]), axis=1
    )
    table["Gamma"] = table.apply(
        lambda row: _format_stat(row["gamma_mean"], row["gamma_std"]), axis=1
    )
    table["Theta"] = table.apply(
        lambda row: _format_stat(row["theta_mean"], row["theta_std"]), axis=1
    )
    table["Rho"] = table.apply(
        lambda row: _format_stat(row["rho_mean"], row["rho_std"]), axis=1
    )
    table["Vega"] = table.apply(
        lambda row: _format_stat(row["vega_mean"], row["vega_std"]), axis=1
    )
    table["Comp Time (s)"] = table.apply(
        lambda row: _format_stat(row["comp_time_mean"], row["comp_time_std"], is_time=True),
        axis=1,
    )
    table["Greek Method"] = table["greeks_method"].fillna("--")

    return table[
        [
            "Algorithm",
            "Sampler",
            "Strike",
            "Steps",
            "Runs",
            "Greek Method",
            "Price",
            "Delta",
            "Gamma",
            "Theta",
            "Rho",
            "Vega",
            "Comp Time (s)",
        ]
    ]


def get_run_output_dir(csv_path: Path) -> Path:
    return OUTPUT_DIR / csv_path.stem


def write_outputs(summary: pd.DataFrame, latex_table: pd.DataFrame, csv_path: Path) -> tuple[Path, Path]:
    run_output_dir = get_run_output_dir(csv_path)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem
    summary_path = run_output_dir / f"{stem}_summary.csv"
    latex_path = run_output_dir / f"{stem}_summary.tex"

    summary.to_csv(summary_path, index=False)

    latex_body = latex_table.to_latex(
        index=False,
        escape=False,
        longtable=True,
        caption=(
            "Summary of American put pricing and Greek estimates from a single "
            "OptStopRandNN metrics run. Entries are mean (standard deviation) "
            "across repeated runs; tree rows are single benchmark evaluations."
        ),
        label=f"tab:{stem}_summary",
    )
    latex_path.write_text(latex_body)
    return summary_path, latex_path


def main() -> None:
    df = load_metrics(CSV_PATH)
    summary = build_summary(df)
    latex_table = build_latex_table(summary)
    summary_path, latex_path = write_outputs(summary, latex_table, CSV_PATH)
    print(f"Wrote summary CSV to {summary_path}")
    print(f"Wrote LaTeX table to {latex_path}")


if __name__ == "__main__":
    main()
