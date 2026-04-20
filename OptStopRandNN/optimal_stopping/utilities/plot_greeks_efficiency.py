"""Plot runtime vs Greeks error against a binomial benchmark."""
import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.append(str(PACKAGE_ROOT))

from optimal_stopping.utilities import read_data


GREEKS = ["delta", "gamma", "theta", "rho", "vega"]
METRICS = [
    ("delta_mae", "Delta MAE"),
    ("gamma_mae", "Gamma MAE"),
    ("theta_mae", "Theta MAE"),
    ("rho_mae", "Rho MAE"),
    ("vega_mae", "Vega MAE"),
    ("mean_abs_err", "Mean Greek MAE"),
]
MATCH_COLS = [
    "model",
    "payoff_key",
    "drift",
    "volatility",
    "spot",
    "strike",
    "dividend",
    "maturity",
    "nb_stocks",
]
SAMPLER_LABELS = {
    "mc": "MC",
    "mc_antithetic": "MC+Antithetic",
    "sobol": "Sobol Seq",
    "sobol_seq": "Sobol Seq",
    "sobol_bb": "Sobol BB",
    "sobol_scrambled": "Scrambled Sobol Seq",
    "sobol_scrambled_seq": "Scrambled Sobol Seq",
    "sobol_scrambled_bb": "Scrambled Sobol BB",
}
PAYOFF_EQUIV = {
    "Put1Dim": "put_1d",
    "MinPut": "put_1d",
    "Call1Dim": "call_1d",
    "MaxCall": "call_1d",
}


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in [
        "drift",
        "volatility",
        "spot",
        "strike",
        "dividend",
        "maturity",
        "nb_stocks",
        "nb_dates",
        "comp_time",
        *GREEKS,
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _normalize_payoff(payoff: object) -> object:
    return PAYOFF_EQUIV.get(payoff, payoff)


def _normalize_sampler(series: pd.Series) -> pd.Series:
    normalized = series.fillna("mc").astype(str).str.strip()
    normalized = normalized.mask(normalized.eq(""), "mc")
    return normalized


def _frontier_points(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    ranked = summary.sort_values(["comp_time_median", metric]).reset_index(drop=True)
    frontier_rows = []
    best_error = float("inf")
    for _, row in ranked.iterrows():
        error = float(row[metric])
        if error <= best_error:
            frontier_rows.append(row)
            best_error = error
    return pd.DataFrame(frontier_rows)


def build_efficiency_table(
    df: pd.DataFrame,
    *,
    algo: str = "RLSMSoftplus",
    greeks_method: str = "regression",
    methods: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    df = _coerce_numeric(df.copy())
    df = df.dropna(subset=["comp_time", *GREEKS])
    df["payoff_key"] = df["payoff"].map(_normalize_payoff)
    if "path_sampler" in df.columns:
        df["path_sampler"] = _normalize_sampler(df["path_sampler"])
    else:
        df["path_sampler"] = "mc"

    ref = df.loc[df["algo"] == "B"].copy()
    if len(ref) == 0:
        raise ValueError("No binomial reference rows found in output/metrics_draft.")

    ref = ref.sort_values("nb_dates").groupby(MATCH_COLS, as_index=False).last()
    ref = ref[MATCH_COLS + GREEKS].copy()
    ref.columns = MATCH_COLS + [f"ref_{g}" for g in GREEKS]

    target = df.loc[df["algo"] == algo].copy()
    target = target.loc[target["greeks_method"] == greeks_method].copy()
    if methods is not None:
        target = target.loc[target["path_sampler"].isin(methods)].copy()

    merged = target.merge(ref, on=MATCH_COLS, how="inner")
    if len(merged) == 0:
        raise ValueError(
            "No rows matched the available binomial reference scenarios after filtering. "
            f"algo={algo}, greeks_method={greeks_method}, methods={methods}."
        )

    for greek in GREEKS:
        merged[f"{greek}_abs_err"] = (merged[greek] - merged[f"ref_{greek}"]).abs()
    merged["mean_abs_err"] = merged[[f"{g}_abs_err" for g in GREEKS]].mean(axis=1)

    summary = (
        merged.groupby(["path_sampler"], dropna=False)
        .agg(
            matched_cases=("path_sampler", "size"),
            comp_time_median=("comp_time", "median"),
            comp_time_mean=("comp_time", "mean"),
            delta_mae=("delta_abs_err", "mean"),
            gamma_mae=("gamma_abs_err", "mean"),
            theta_mae=("theta_abs_err", "mean"),
            rho_mae=("rho_abs_err", "mean"),
            vega_mae=("vega_abs_err", "mean"),
            mean_abs_err=("mean_abs_err", "mean"),
        )
        .reset_index()
        .sort_values(["mean_abs_err", "comp_time_median"])
    )
    summary["label"] = summary["path_sampler"].map(
        lambda value: SAMPLER_LABELS.get(value, value)
    )
    return summary


def plot_efficiency(
    summary: pd.DataFrame,
    *,
    algo: str,
    greeks_method: str,
    output_path: Path,
) -> None:
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    axs = axs.ravel()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {
        row["label"]: colors[i % len(colors)]
        for i, (_, row) in enumerate(summary.reset_index().iterrows())
    }

    for ax, (metric, title) in zip(axs, METRICS):
        for _, row in summary.iterrows():
            ax.scatter(
                row["comp_time_median"],
                row[metric],
                s=70,
                color=color_map[row["label"]],
            )
            ax.annotate(
                row["label"],
                (row["comp_time_median"], row[metric]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        frontier = _frontier_points(summary, metric)
        if len(frontier) >= 2:
            ax.plot(
                frontier["comp_time_median"],
                frontier[metric],
                color="black",
                linewidth=1.2,
                linestyle="--",
                alpha=0.8,
            )

        ax.set_title(title)
        ax.set_xlabel("Median runtime (s)")
        ax.set_ylabel("Absolute error vs binomial")
        ax.set_xscale("log")
        if (summary[metric] > 0).all():
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Greeks Efficiency Frontier: {algo} ({greeks_method})",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved efficiency plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot runtime vs Greek error using binomial rows as reference."
    )
    parser.add_argument("--algo", default="RLSMSoftplus")
    parser.add_argument("--greeks-method", default="regression")
    parser.add_argument(
        "--methods",
        default=None,
        help="Comma-separated path sampler methods to plot. Defaults to all available methods.",
    )
    parser.add_argument("--read-which", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the output PDF and CSV. Defaults to <repo>/plots.",
    )
    args = parser.parse_args()

    df = read_data.read_csvs_conv(which=args.read_which)
    summary = build_efficiency_table(
        df,
        algo=args.algo,
        greeks_method=args.greeks_method,
        methods=(
            tuple(value.strip() for value in args.methods.split(",") if value.strip())
            if args.methods
            else None
        ),
    )

    if args.output_dir is None:
        output_dir = Path(__file__).resolve().parents[2] / "plots"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    method_stem = "all_methods" if args.methods is None else args.methods.replace(",", "-")
    stem = f"greeks_efficiency_{args.algo}_{args.greeks_method}_{method_stem}"
    csv_path = output_dir / f"{stem}.csv"
    pdf_path = output_dir / f"{stem}.pdf"
    summary.to_csv(csv_path, index=False)
    print(f"Saved efficiency summary to {csv_path}")
    plot_efficiency(summary, algo=args.algo, greeks_method=args.greeks_method, output_path=pdf_path)


if __name__ == "__main__":
    main()
