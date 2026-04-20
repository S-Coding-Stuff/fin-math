from pathlib import Path

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PACKAGE_ROOT / "output"

CSV_FILENAME = "1774908747294.csv"
CSV_PATH = OUTPUT_ROOT / "metrics_draft" / CSV_FILENAME
OUTPUT_DIR = OUTPUT_ROOT / "latex_tables"

SAMPLER_ORDER = [
    "mc",
    "mc_antithetic",
    "sobol_seq",
    "sobol_bb",
    "sobol_scrambled_seq",
    "sobol_scrambled_bb",
    "sobol",
    "sobol_scrambled",
]

SAMPLER_LABELS = {
    "mc": "MC",
    "mc_antithetic": "MC + AV",
    "sobol": "QMC Seq",
    "sobol_seq": "QMC Seq",
    "sobol_bb": "QMC BB",
    "sobol_scrambled": "RQMC Seq",
    "sobol_scrambled_seq": "RQMC Seq",
    "sobol_scrambled_bb": "RQMC BB",
}


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in [
        "nb_stocks",
        "spot",
        "duration",
        "time_path_gen",
        "comp_time",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _format_time(mean_value: float, std_value: float) -> str:
    if pd.isna(mean_value):
        return "--"
    if pd.isna(std_value):
        return f"{mean_value:.4f}"
    return f"{mean_value:.4f} ({std_value:.4f})"


def load_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {path}")
    df = pd.read_csv(path)
    df = _coerce_numeric(df)
    df["path_sampler"] = df["path_sampler"].fillna("mc")
    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(
            ["algo", "path_sampler", "nb_stocks", "spot", "use_payoff_as_input"],
            dropna=False,
        )
        .agg(
            runs=("price", "size"),
            duration_mean=("duration", "mean"),
            duration_std=("duration", "std"),
            time_path_gen_mean=("time_path_gen", "mean"),
            time_path_gen_std=("time_path_gen", "std"),
            comp_time_mean=("comp_time", "mean"),
            comp_time_std=("comp_time", "std"),
        )
        .reset_index()
    )
    summary["path_sampler"] = pd.Categorical(
        summary["path_sampler"], categories=SAMPLER_ORDER, ordered=True
    )
    summary = summary.sort_values(
        ["algo", "path_sampler", "nb_stocks", "spot", "use_payoff_as_input"]
    ).reset_index(drop=True)
    return summary


def build_table(summary: pd.DataFrame) -> pd.DataFrame:
    table = summary.copy()
    table["Sampler"] = table["path_sampler"].map(SAMPLER_LABELS).fillna(table["path_sampler"].astype(str))
    table["Assets"] = table["nb_stocks"].astype(int)
    table["Spot"] = table["spot"].map(lambda value: f"{value:.0f}")
    table["Use Payoff"] = table["use_payoff_as_input"].map({True: "True", False: "False"})
    table["Runs"] = table["runs"].astype(int)
    table["Duration (s)"] = table.apply(
        lambda row: _format_time(row["duration_mean"], row["duration_std"]), axis=1
    )
    table["Path Gen (s)"] = table.apply(
        lambda row: _format_time(row["time_path_gen_mean"], row["time_path_gen_std"]), axis=1
    )
    table["Comp Time (s)"] = table.apply(
        lambda row: _format_time(row["comp_time_mean"], row["comp_time_std"]), axis=1
    )
    return table[
        [
            "algo",
            "Sampler",
            "Assets",
            "Spot",
            "Use Payoff",
            "Runs",
            "Duration (s)",
            "Path Gen (s)",
            "Comp Time (s)",
        ]
    ].rename(columns={"algo": "Algorithm"})


def get_run_output_dir(csv_path: Path) -> Path:
    return OUTPUT_DIR / csv_path.stem


def write_outputs(summary: pd.DataFrame, table: pd.DataFrame, csv_path: Path) -> tuple[Path, Path]:
    run_output_dir = get_run_output_dir(csv_path)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem
    summary_path = run_output_dir / f"{stem}_runtime_summary.csv"
    latex_path = run_output_dir / f"{stem}_runtime_summary.tex"
    summary.to_csv(summary_path, index=False)
    latex_text = table.to_latex(
        index=False,
        escape=False,
        longtable=True,
        caption=(
            "Runtime summary for the OptStopRandNN dimension sweep. "
            "Entries are mean (standard deviation) across repeated runs."
        ),
        label=f"tab:{stem}_runtime_summary",
    )
    latex_path.write_text(latex_text)
    return summary_path, latex_path


def main() -> None:
    df = load_metrics(CSV_PATH)
    summary = build_summary(df)
    table = build_table(summary)
    summary_path, latex_path = write_outputs(summary, table, CSV_PATH)
    print(f"Wrote runtime summary CSV to {summary_path}")
    print(f"Wrote runtime LaTeX table to {latex_path}")


if __name__ == "__main__":
    main()
