"""Plot price and Greeks from a single OptStopRandNN metrics CSV."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PLOT_COLUMNS = ["price", "delta", "gamma", "theta", "rho", "vega"]
NUMERIC_COLUMNS = [
    "spot",
    "strike",
    "volatility",
    "maturity",
    "nb_dates",
    "nb_paths",
    *PLOT_COLUMNS,
]


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _normalize_sampler(series: pd.Series) -> pd.Series:
    normalized = series.fillna("mc").astype(str).str.strip()
    normalized = normalized.mask(normalized.eq(""), "mc")
    return normalized


def _format_group_label(group_cols: list[str], values: tuple[object, ...] | object) -> str:
    if not isinstance(values, tuple):
        values = (values,)
    parts = []
    for col, value in zip(group_cols, values):
        if col == "volatility":
            parts.append(f"sigma={float(value):.1f}")
        elif col == "maturity":
            parts.append(f"T={float(value):.1f}")
        elif col == "path_sampler":
            parts.append(str(value))
        elif col == "strike":
            parts.append(f"K={float(value):.0f}")
        elif col == "spot":
            parts.append(f"S0={float(value):.0f}")
        else:
            parts.append(f"{col}={value}")
    return ", ".join(parts)


def build_plot_frame(
    csv_path: Path,
    *,
    algo: str | None,
    greeks_method: str | None,
    path_samplers: tuple[str, ...] | None,
    x_col: str,
    group_cols: list[str],
    agg: str,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = _coerce_numeric(df)
    if "path_sampler" in df.columns:
        df["path_sampler"] = _normalize_sampler(df["path_sampler"])
    else:
        df["path_sampler"] = "mc"

    if algo is not None:
        df = df.loc[df["algo"] == algo].copy()
    if greeks_method is not None and "greeks_method" in df.columns:
        df = df.loc[df["greeks_method"] == greeks_method].copy()
    if path_samplers is not None:
        df = df.loc[df["path_sampler"].isin(path_samplers)].copy()

    required = [x_col, *PLOT_COLUMNS]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found in {csv_path}.")

    df = df.dropna(subset=[x_col, *PLOT_COLUMNS]).copy()
    if len(df) == 0:
        raise ValueError("No rows remain after filtering.")
    if df[x_col].nunique() < 2:
        raise ValueError(
            f"Cannot plot against {x_col!r}: only {df[x_col].nunique()} unique value is available."
        )

    grouped = (
        df.groupby([*group_cols, x_col], dropna=False)[PLOT_COLUMNS]
        .agg(agg)
        .reset_index()
        .sort_values([*group_cols, x_col])
    )
    return grouped


def plot_grid(
    plot_df: pd.DataFrame,
    *,
    x_col: str,
    group_cols: list[str],
    output_path: Path,
    title: str,
) -> Path:
    fig, axs = plt.subplots(2, 3, figsize=(15, 7))
    axs = axs.ravel()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    linestyles = ["-", "--", "-.", ":"]

    if group_cols:
        grouped_iter = list(plot_df.groupby(group_cols, dropna=False))
    else:
        grouped_iter = [(("all",), plot_df)]

    for ax, plot_label in zip(axs, PLOT_COLUMNS):
        for idx, (group_values, subset) in enumerate(grouped_iter):
            label = _format_group_label(group_cols, group_values) if group_cols else "all"
            ax.plot(
                subset[x_col],
                subset[plot_label],
                label=label,
                color=colors[idx % len(colors)],
                linestyle=linestyles[idx % len(linestyles)],
                marker="o",
                markersize=3,
                linewidth=1.8,
            )
        ax.set_title(plot_label)
        ax.set_xlabel(x_col)
        ax.grid(True, alpha=0.3)

    handles, labels = axs[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center right", frameon=False)
        fig.tight_layout(rect=[0.0, 0.0, 0.84, 0.96])
    else:
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    fig.suptitle(title)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot price and Greeks from one metrics CSV.")
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--algo", default=None)
    parser.add_argument("--greeks-method", default=None)
    parser.add_argument("--path-samplers", default=None)
    parser.add_argument("--x-col", default="spot", choices=["spot", "strike", "maturity"])
    parser.add_argument(
        "--group-cols",
        default="volatility,maturity",
        help="Comma-separated columns defining one curve each.",
    )
    parser.add_argument("--agg", default="median", choices=["median", "mean"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    csv_path = Path(args.csv_path).resolve()
    group_cols = [value.strip() for value in args.group_cols.split(",") if value.strip()]
    plot_df = build_plot_frame(
        csv_path,
        algo=args.algo,
        greeks_method=args.greeks_method,
        path_samplers=(
            tuple(value.strip() for value in args.path_samplers.split(",") if value.strip())
            if args.path_samplers
            else None
        ),
        x_col=args.x_col,
        group_cols=group_cols,
        agg=args.agg,
    )

    if args.output_dir is None:
        output_dir = csv_path.parent
    else:
        output_dir = Path(args.output_dir).resolve()

    stem_parts = [csv_path.stem]
    if args.algo:
        stem_parts.append(args.algo)
    if args.greeks_method:
        stem_parts.append(args.greeks_method)
    stem_parts.append(f"by_{args.x_col}")
    output_path = output_dir / ("greeks_plot_" + "_".join(stem_parts) + ".pdf")

    title_parts = ["Price and Greeks"]
    if args.algo:
        title_parts.append(args.algo)
    if args.greeks_method:
        title_parts.append(args.greeks_method)
    title_parts.append(f"vs {args.x_col}")

    saved_path = plot_grid(
        plot_df,
        x_col=args.x_col,
        group_cols=group_cols,
        output_path=output_path,
        title=" | ".join(title_parts),
    )
    print(f"Saved plot to {saved_path}")


if __name__ == "__main__":
    main()
