"""Create one spot-curve Greeks plot per path sampler from a single metrics CSV."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PLOT_COLUMNS = ["price", "delta", "gamma", "theta", "rho", "vega"]
NUMERIC_COLUMNS = ["spot", "volatility", "maturity", *PLOT_COLUMNS]
LINESTYLES = ["-", "-.", "--", ":"]
SAMPLER_LABELS = {
    "mc": "MC",
    "mc_antithetic": "MC + AV",
    "sobol": "QMC (Seq)",
    "sobol_seq": "QMC (Seq)",
    "sobol_bb": "QMC (BB)",
    "sobol_scrambled": "RQMC (Seq)",
    "sobol_scrambled_seq": "RQMC (Seq)",
    "sobol_scrambled_bb": "RQMC (BB)",
}


def _normalize_sampler(series: pd.Series) -> pd.Series:
    normalized = series.fillna("mc").astype(str).str.strip()
    return normalized.mask(normalized.eq(""), "mc")


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _method_stem(method: str) -> str:
    return method.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")


def load_filtered_data(
    csv_path: Path,
    *,
    algo: str,
    greeks_method: str,
    methods: tuple[str, ...] | None,
    volatilities: tuple[float, ...] | None,
    maturities: tuple[float, ...] | None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = _coerce_numeric(df)
    if "path_sampler" in df.columns:
        df["path_sampler"] = _normalize_sampler(df["path_sampler"])
    else:
        df["path_sampler"] = "mc"

    df = df.loc[df["algo"] == algo].copy()
    if "greeks_method" in df.columns:
        df = df.loc[df["greeks_method"] == greeks_method].copy()
    if methods is not None:
        df = df.loc[df["path_sampler"].isin(methods)].copy()
    if volatilities is not None:
        df = df.loc[df["volatility"].isin(volatilities)].copy()
    if maturities is not None:
        df = df.loc[df["maturity"].isin(maturities)].copy()

    df = df.dropna(subset=["spot", "volatility", "maturity", *PLOT_COLUMNS]).copy()
    if len(df) == 0:
        raise ValueError(
            f"No rows matched algo={algo}, greeks_method={greeks_method}, methods={methods}."
        )
    if df["spot"].nunique() < 2:
        raise ValueError(
            f"{csv_path.name} does not contain a spot sweep, so it cannot produce spot-curve plots."
        )
    return df


def plot_one_method(
    df: pd.DataFrame,
    *,
    method: str,
    algo: str,
    greeks_method: str,
    output_dir: Path,
    aggregate: str,
) -> Path:
    subset = df.loc[df["path_sampler"] == method].copy()
    if len(subset) == 0:
        raise ValueError(f"No rows found for method={method!r}.")

    grouped = (
        subset.groupby(["volatility", "maturity", "spot"], dropna=False)[PLOT_COLUMNS]
        .agg(aggregate)
        .reset_index()
        .sort_values(["volatility", "maturity", "spot"])
    )

    fig, axs = plt.subplots(2, 3, figsize=(15, 7))
    axs = axs.ravel()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, plot_label in zip(axs, PLOT_COLUMNS):
        for idx, ((vol, maturity), frame) in enumerate(grouped.groupby(["volatility", "maturity"], dropna=False)):
            ax.plot(
                frame["spot"],
                frame[plot_label],
                label=f"$\\sigma$={vol:.1f} $T$={maturity:.1f}",
                color=colors[idx % len(colors)],
                linestyle=LINESTYLES[idx % len(LINESTYLES)],
                linewidth=1.8,
            )
        ax.set_title(plot_label)
        ax.set_xlabel("spot")
        ax.grid(True, alpha=0.3)

    handles, labels = axs[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, bbox_to_anchor=(1.02, 0.5), loc="center left", frameon=False)
        fig.tight_layout(rect=[0.0, 0.0, 0.84, 1.0])
    else:
        fig.tight_layout()

    method_label = SAMPLER_LABELS.get(method, method)
    fig.suptitle(f"{algo} | {greeks_method} | {method_label}", y=1.02)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"greeks_plot_{algo}_{greeks_method}_{_method_stem(method)}.pdf"
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate one Greeks spot-curve PDF per sampler method from a single metrics CSV."
    )
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--algo", default="RLSMSoftplus")
    parser.add_argument("--greeks-method", default="regression")
    parser.add_argument(
        "--methods",
        default=None,
        help="Optional comma-separated sampler subset.",
    )
    parser.add_argument(
        "--volatilities",
        default=None,
        help="Optional comma-separated volatility subset, e.g. 0.1,0.2,0.3",
    )
    parser.add_argument(
        "--maturities",
        default=None,
        help="Optional comma-separated maturity subset, e.g. 0.5,1.0,2.0",
    )
    parser.add_argument("--aggregate", default="median", choices=["median", "mean"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    csv_path = Path(args.csv_path).resolve()
    methods = (
        tuple(value.strip() for value in args.methods.split(",") if value.strip())
        if args.methods
        else None
    )
    df = load_filtered_data(
        csv_path,
        algo=args.algo,
        greeks_method=args.greeks_method,
        methods=methods,
        volatilities=(
            tuple(float(value.strip()) for value in args.volatilities.split(",") if value.strip())
            if args.volatilities
            else None
        ),
        maturities=(
            tuple(float(value.strip()) for value in args.maturities.split(",") if value.strip())
            if args.maturities
            else None
        ),
    )
    output_dir = Path(args.output_dir).resolve() if args.output_dir else csv_path.parent

    available_methods = list(df["path_sampler"].dropna().unique())
    for method in available_methods:
        output_path = plot_one_method(
            df,
            method=method,
            algo=args.algo,
            greeks_method=args.greeks_method,
            output_dir=output_dir,
            aggregate=args.aggregate,
        )
        print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
