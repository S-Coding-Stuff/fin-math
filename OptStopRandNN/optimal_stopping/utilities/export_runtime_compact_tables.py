import argparse
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

METRIC_SPECS = [
    ("duration", "Duration (s)", "runtime_duration"),
    ("time_path_gen", "Path Gen (s)", "runtime_path_gen"),
    ("comp_time", "Comp Time (s)", "runtime_comp_time"),
]


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["nb_stocks", "duration", "time_path_gen", "comp_time"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {path}")
    df = pd.read_csv(path)
    df = _coerce_numeric(df)
    df["path_sampler"] = df["path_sampler"].fillna("mc")
    return df


def build_compact_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["algo", "nb_stocks", "path_sampler"], dropna=False)
        .agg(
            runs=("price", "size"),
            price_mean=("price", "mean"),
            price_std=("price", "std"),
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
    return summary.sort_values(["algo", "nb_stocks", "path_sampler"]).reset_index(drop=True)


def build_spot_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["algo", "nb_stocks", "spot", "path_sampler"], dropna=False)
        .agg(
            runs=("price", "size"),
            duration_mean=("duration", "mean"),
            duration_std=("duration", "std"),
        )
        .reset_index()
    )
    summary["path_sampler"] = pd.Categorical(
        summary["path_sampler"], categories=SAMPLER_ORDER, ordered=True
    )
    return summary.sort_values(["algo", "nb_stocks", "spot", "path_sampler"]).reset_index(drop=True)


def build_metric_table(summary: pd.DataFrame, *, metric_mean: str) -> pd.DataFrame:
    table = summary.copy()
    table["Sampler"] = table["path_sampler"].map(SAMPLER_LABELS).fillna(table["path_sampler"].astype(str))
    pivot = table.pivot_table(
        index=["algo", "nb_stocks"],
        columns="Sampler",
        values=metric_mean,
        aggfunc="first",
    )
    desired_cols = []
    seen = set()
    for name in SAMPLER_ORDER:
        label = SAMPLER_LABELS.get(name, name)
        if label in pivot.columns and label not in seen:
            desired_cols.append(label)
            seen.add(label)
    pivot = pivot.reindex(columns=desired_cols)
    pivot = pivot.reset_index()
    pivot.rename(columns={"algo": "Algorithm", "nb_stocks": "Assets"}, inplace=True)
    return pivot


def build_price_duration_table(summary: pd.DataFrame) -> pd.DataFrame:
    table = summary.copy()
    table["Sampler"] = table["path_sampler"].map(SAMPLER_LABELS).fillna(table["path_sampler"].astype(str))
    table["price_stat"] = table.apply(
        lambda row: (
            f"{row['price_mean']:.4f} ({row['price_std']:.4f})"
            if pd.notna(row["price_mean"]) and pd.notna(row["price_std"])
            else (f"{row['price_mean']:.4f}" if pd.notna(row["price_mean"]) else "--")
        ),
        axis=1,
    )
    table["duration_stat"] = table.apply(
        lambda row: (
            f"{row['duration_mean']:.4f} ({row['duration_std']:.4f})"
            if pd.notna(row["duration_mean"]) and pd.notna(row["duration_std"])
            else (f"{row['duration_mean']:.4f}" if pd.notna(row["duration_mean"]) else "--")
        ),
        axis=1,
    )

    price_pivot = table.pivot_table(
        index=["nb_stocks", "algo"],
        columns="Sampler",
        values="price_stat",
        aggfunc="first",
    )
    duration_pivot = table.pivot_table(
        index=["nb_stocks", "algo"],
        columns="Sampler",
        values="duration_stat",
        aggfunc="first",
    )

    desired_cols = []
    seen = set()
    for name in SAMPLER_ORDER:
        label = SAMPLER_LABELS.get(name, name)
        if label not in seen:
            desired_cols.append(label)
            seen.add(label)

    price_pivot = price_pivot.reindex(columns=[c for c in desired_cols if c in price_pivot.columns])
    duration_pivot = duration_pivot.reindex(columns=[c for c in desired_cols if c in duration_pivot.columns])

    combined = price_pivot.join(duration_pivot, lsuffix="__price", rsuffix="__duration")
    combined = combined.reset_index()
    combined.rename(columns={"nb_stocks": "Assets", "algo": "Algorithm"}, inplace=True)

    ordered_columns = ["Algorithm", "Assets"]
    ordered_columns.extend([f"{label} Price" for label in desired_cols if f"{label}__price" in combined.columns])
    ordered_columns.extend(
        [f"{label} Duration" for label in desired_cols if f"{label}__duration" in combined.columns]
    )

    rename_map = {}
    for label in desired_cols:
        price_col = f"{label}__price"
        duration_col = f"{label}__duration"
        if price_col in combined.columns:
            rename_map[price_col] = f"{label} Price"
        if duration_col in combined.columns:
            rename_map[duration_col] = f"{label} Duration"
    combined.rename(columns=rename_map, inplace=True)

    return combined[ordered_columns]


def write_price_duration_outputs(summary: pd.DataFrame, csv_path: Path) -> tuple[Path, Path]:
    run_output_dir = get_run_output_dir(csv_path)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem
    table = build_price_duration_table(summary)
    csv_out = run_output_dir / f"{stem}_price_duration.csv"
    tex_out = run_output_dir / f"{stem}_price_duration.tex"
    table.to_csv(csv_out, index=False)

    sampler_labels = []
    seen = set()
    for name in SAMPLER_ORDER:
        label = SAMPLER_LABELS.get(name, name)
        if label not in seen:
            sampler_labels.append(label)
            seen.add(label)

    lines = []
    lines.append("\\begin{table}")
    lines.append("\\caption{Compact summary of prices and durations across samplers for the OptStopRandNN dimension sweep. Entries are means aggregated across spots, payoff-input settings, and repeated runs.}")
    lines.append(f"\\label{{tab:{stem}_price_duration}}")
    col_spec = "rr|" + ("r" * len(sampler_labels)) + "|" + ("r" * len(sampler_labels))
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append(
        f"\\multicolumn{{2}}{{c|}}{{}} & \\multicolumn{{{len(sampler_labels)}}}{{c|}}{{Price}} & "
        f"\\multicolumn{{{len(sampler_labels)}}}{{c}}{{Duration (s)}} \\\\"
    )
    lines.append("\\midrule")
    header = ["Algorithm", "$d$"] + sampler_labels + sampler_labels
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")

    for _, row in table.iterrows():
        row_values = [
            str(row["Algorithm"]),
            f"{int(row['Assets'])}" if pd.notna(row["Assets"]) else "--",
        ]
        row_values.extend(
            str(row.get(f"{label} Price", "--"))
            for label in sampler_labels
        )
        row_values.extend(
            str(row.get(f"{label} Duration", "--"))
            for label in sampler_labels
        )
        lines.append(" & ".join(row_values) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    tex_out.write_text("\n".join(lines) + "\n")
    return csv_out, tex_out


def write_price_duration_by_spot_outputs(
    df: pd.DataFrame,
    csv_path: Path,
    *,
    use_payoff_as_input: bool = True,
) -> list[tuple[Path, Path]]:
    run_output_dir = get_run_output_dir(csv_path)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem
    outputs = []

    spot_values = sorted(value for value in df["spot"].dropna().unique())
    for spot in spot_values:
        filtered = df.loc[
            (df["spot"] == spot) & (df["use_payoff_as_input"] == use_payoff_as_input)
        ].copy()
        if filtered.empty:
            continue

        summary = build_compact_summary(filtered)
        table = build_price_duration_table(summary)
        spot_label = str(int(spot)) if float(spot).is_integer() else str(spot).replace(".", "p")
        csv_out = run_output_dir / f"{stem}_price_duration_spot_{spot_label}.csv"
        tex_out = run_output_dir / f"{stem}_price_duration_spot_{spot_label}.tex"
        table.to_csv(csv_out, index=False)

        sampler_labels = []
        seen = set()
        for name in SAMPLER_ORDER:
            label = SAMPLER_LABELS.get(name, name)
            if label not in seen:
                sampler_labels.append(label)
                seen.add(label)

        lines = []
        lines.append("\\begin{table}")
        lines.append(
            "\\caption{Compact summary of prices and durations across samplers for the "
            f"OptStopRandNN dimension sweep at spot {spot:.0f}, using only "
            f"\\texttt{{use\\_payoff\\_as\\_input={use_payoff_as_input}}}. Entries are "
            "mean (standard deviation) across repeated runs.}"
        )
        lines.append(f"\\label{{tab:{stem}_price_duration_spot_{spot_label}}}")
        col_spec = "rr|" + ("r" * len(sampler_labels)) + "|" + ("r" * len(sampler_labels))
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")
        lines.append(
            f"\\multicolumn{{2}}{{c|}}{{}} & \\multicolumn{{{len(sampler_labels)}}}{{c|}}{{Price}} & "
            f"\\multicolumn{{{len(sampler_labels)}}}{{c}}{{Duration (s)}} \\\\"
        )
        lines.append("\\midrule")
        header = ["Algorithm", "$d$"] + sampler_labels + sampler_labels
        lines.append(" & ".join(header) + " \\\\")
        lines.append("\\midrule")

        for _, row in table.iterrows():
            row_values = [
                str(row["Algorithm"]),
                f"{int(row['Assets'])}" if pd.notna(row["Assets"]) else "--",
            ]
            row_values.extend(str(row.get(f"{label} Price", "--")) for label in sampler_labels)
            row_values.extend(str(row.get(f"{label} Duration", "--")) for label in sampler_labels)
            lines.append(" & ".join(row_values) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        tex_out.write_text("\n".join(lines) + "\n")
        outputs.append((csv_out, tex_out))

    return outputs


def build_spot_duration_table(summary: pd.DataFrame) -> pd.DataFrame:
    table = summary.copy()
    table["Sampler"] = table["path_sampler"].map(SAMPLER_LABELS).fillna(table["path_sampler"].astype(str))
    pivot = table.pivot_table(
        index=["algo", "nb_stocks", "spot"],
        columns="Sampler",
        values="duration_mean",
        aggfunc="first",
    )
    desired_cols = []
    seen = set()
    for name in SAMPLER_ORDER:
        label = SAMPLER_LABELS.get(name, name)
        if label in pivot.columns and label not in seen:
            desired_cols.append(label)
            seen.add(label)
    pivot = pivot.reindex(columns=desired_cols)
    pivot = pivot.reset_index()
    pivot.rename(columns={"algo": "Algorithm", "nb_stocks": "Assets", "spot": "Spot"}, inplace=True)
    return pivot


def get_run_output_dir(csv_path: Path) -> Path:
    return OUTPUT_DIR / csv_path.stem


def resolve_csv_path(*, csv_filename: str | None, csv_path: str | None) -> Path:
    if csv_path:
        path = Path(csv_path)
        return path if path.is_absolute() else path.resolve()
    if csv_filename:
        return OUTPUT_ROOT / "metrics_draft" / csv_filename
    return CSV_PATH


def write_outputs(summary: pd.DataFrame, csv_path: Path) -> list[tuple[Path, Path]]:
    run_output_dir = get_run_output_dir(csv_path)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem
    outputs: list[tuple[Path, Path]] = []

    summary_path = run_output_dir / f"{stem}_runtime_compact_summary.csv"
    summary.to_csv(summary_path, index=False)

    for metric_key, caption_label, suffix in METRIC_SPECS:
        table = build_metric_table(summary, metric_mean=f"{metric_key}_mean")
        csv_out = run_output_dir / f"{stem}_{suffix}.csv"
        tex_out = run_output_dir / f"{stem}_{suffix}.tex"
        table.to_csv(csv_out, index=False)
        latex_text = table.to_latex(
            index=False,
            escape=False,
            float_format="%.4f",
            caption=(
                f"{caption_label} summarised across spots, payoff-input settings, "
                f"and repeated runs for the OptStopRandNN dimension sweep."
            ),
            label=f"tab:{stem}_{suffix}",
        )
        tex_out.write_text(latex_text)
        outputs.append((csv_out, tex_out))

    return outputs


def write_spot_duration_output(df: pd.DataFrame, csv_path: Path) -> tuple[Path, Path]:
    run_output_dir = get_run_output_dir(csv_path)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem
    summary = build_spot_summary(df)
    table = build_spot_duration_table(summary)
    csv_out = run_output_dir / f"{stem}_runtime_duration_by_spot.csv"
    tex_out = run_output_dir / f"{stem}_runtime_duration_by_spot.tex"
    table.to_csv(csv_out, index=False)
    latex_text = table.to_latex(
        index=False,
        escape=False,
        float_format="%.4f",
        caption=(
            "Duration (s) by spot for the OptStopRandNN dimension sweep, "
            "summarised across payoff-input settings and repeated runs."
        ),
        label=f"tab:{stem}_runtime_duration_by_spot",
    )
    tex_out.write_text(latex_text)
    return csv_out, tex_out


def select_spot_price_use_payoff_flag(df: pd.DataFrame) -> bool:
    values = {bool(v) for v in df["use_payoff_as_input"].dropna().unique().tolist()}
    if True in values:
        return True
    if False in values:
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export compact runtime and price LaTeX tables from one metrics CSV."
    )
    parser.add_argument("--csv-filename", help="CSV file name inside output/metrics_draft")
    parser.add_argument("--csv-path", help="Absolute or relative path to a metrics CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = resolve_csv_path(csv_filename=args.csv_filename, csv_path=args.csv_path)
    df = load_metrics(csv_path)
    summary = build_compact_summary(df)
    outputs = write_outputs(summary, csv_path)
    price_csv_out, price_tex_out = write_price_duration_outputs(summary, csv_path)
    spot_price_outputs = write_price_duration_by_spot_outputs(
        df,
        csv_path,
        use_payoff_as_input=select_spot_price_use_payoff_flag(df),
    )
    spot_csv_out, spot_tex_out = write_spot_duration_output(df, csv_path)
    print(f"Wrote outputs to {get_run_output_dir(csv_path)}")
    for csv_out, tex_out in outputs:
        print(f"Wrote {csv_out}")
        print(f"Wrote {tex_out}")
    print(f"Wrote {price_csv_out}")
    print(f"Wrote {price_tex_out}")
    for csv_out, tex_out in spot_price_outputs:
        print(f"Wrote {csv_out}")
        print(f"Wrote {tex_out}")
    print(f"Wrote {spot_csv_out}")
    print(f"Wrote {spot_tex_out}")


if __name__ == "__main__":
    main()
