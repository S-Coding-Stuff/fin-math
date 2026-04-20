"""Plot Greeks vs maturity across moneyness slices from OptStopRandNN outputs."""
import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.append(str(PACKAGE_ROOT))

from optimal_stopping.run import configs


PLOT_COLUMNS = ["price", "delta", "gamma", "theta", "rho", "vega"]
DEFAULT_MONEYNESS_SLICES = (0.90, 1.00, 1.10)
NUMERIC_COLUMNS = [
    "volatility",
    "spot",
    "strike",
    "maturity",
    "nb_dates",
    "nb_paths",
    "hidden_size",
    *PLOT_COLUMNS,
]


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _get_csv_paths(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=True)
    return sorted(path for path in base_dir.iterdir() if path.suffix == ".csv")


def _read_csvs_conv(which: int = 0) -> pd.DataFrame:
    metrics_dir = PACKAGE_ROOT / "output" / "metrics"
    draft_dir = PACKAGE_ROOT / "output" / "metrics_draft"
    if which == 0:
        paths = _get_csv_paths(draft_dir)
    elif which == 1:
        paths = _get_csv_paths(metrics_dir)
    else:
        paths = _get_csv_paths(draft_dir) + _get_csv_paths(metrics_dir)
    if not paths:
        raise ValueError("No metrics CSV files found in output/metrics or output/metrics_draft.")
    return pd.concat((pd.read_csv(path, index_col=None) for path in paths), ignore_index=True)


def _select_moneyness_slices(values: pd.Series, targets: tuple[float, ...]) -> list[float]:
    available = sorted(float(v) for v in values.dropna().unique())
    if not available:
        return []
    selected: list[float] = []
    for target in targets:
        nearest = min(available, key=lambda value: (abs(value - target), value))
        if nearest not in selected:
            selected.append(nearest)
    return selected


def _apply_config_filters(df: pd.DataFrame, config: configs._DefaultConfig) -> pd.DataFrame:
    filtered = df.copy()
    filter_map = {
        "algos": "algo",
        "stock_models": "model",
        "payoffs": "payoff",
        "drift": "drift",
        "nb_stocks": "nb_stocks",
        "spots": "spot",
        "volatilities": "volatility",
        "nb_paths": "nb_paths",
        "nb_dates": "nb_dates",
        "strikes": "strike",
        "dividends": "dividend",
        "maturities": "maturity",
        "hidden_size": "hidden_size",
        "nb_epochs": "nb_epochs",
        "hurst": "hurst",
        "factors": "factors",
        "ridge_coeff": "ridge_coeff",
        "use_path": "use_path",
        "train_ITM_only": "train_ITM_only",
        "use_payoff_as_input": "use_payoff_as_input",
    }
    for attr, column in filter_map.items():
        if column not in filtered.columns:
            continue
        values = list(getattr(config, attr, []))
        if not values:
            continue
        if attr == "factors":
            values = [str(v) for v in values]
            filtered[column] = filtered[column].astype(str)
        filtered = filtered.loc[filtered[column].isin(values)]
    return filtered


def build_maturity_slice_table(
    config: configs._DefaultConfig,
    *,
    algo: str = "RLSMSoftplus",
    greeks_method: str = "regression",
    volatility: float | None = None,
    moneyness_slices: tuple[float, ...] = DEFAULT_MONEYNESS_SLICES,
    methods: tuple[str, ...] | None = None,
    use_payoff_as_input: bool | None = None,
    train_itm_only: bool | None = None,
    read_which: int = 0,
) -> pd.DataFrame:
    df = _read_csvs_conv(which=read_which)
    df = _coerce_numeric(df)
    df = _apply_config_filters(df, config)

    df = df.loc[df["algo"] == algo].copy()
    df = df.loc[df["greeks_method"] == greeks_method].copy()
    if methods is not None and "path_sampler" in df.columns:
        df = df.loc[df["path_sampler"].isin(methods)].copy()

    if volatility is not None:
        df = df.loc[df["volatility"] == float(volatility)].copy()
    if use_payoff_as_input is not None and "use_payoff_as_input" in df.columns:
        df = df.loc[df["use_payoff_as_input"] == bool(use_payoff_as_input)].copy()
    if train_itm_only is not None and "train_ITM_only" in df.columns:
        df = df.loc[df["train_ITM_only"] == bool(train_itm_only)].copy()

    df = df.dropna(subset=["strike", "spot", "maturity", *PLOT_COLUMNS])
    df = df.loc[df["strike"] != 0].copy()
    df["moneyness"] = df["spot"] / df["strike"]
    selected_moneyness = _select_moneyness_slices(df["moneyness"], moneyness_slices)
    if not selected_moneyness:
        raise ValueError("No moneyness values available after filtering.")
    df = df.loc[df["moneyness"].isin(selected_moneyness)].copy()
    df["moneyness_label"] = df["moneyness"].map(lambda x: f"S/K={x:.2f}")
    if "path_sampler" not in df.columns:
        df["path_sampler"] = "mc"

    if len(df) == 0:
        raise ValueError(
            "No data matched the requested maturity-slice filters. "
            f"algo={algo}, greeks_method={greeks_method}, volatility={volatility}, "
            f"use_payoff_as_input={use_payoff_as_input}, train_itm_only={train_itm_only}."
        )

    grouped = (
        df.groupby(
            ["path_sampler", "maturity", "moneyness", "moneyness_label"], dropna=False
        )[PLOT_COLUMNS]
        .median()
        .reset_index()
        .sort_values(["path_sampler", "moneyness", "maturity"])
    )
    return grouped


def plot_greeks_vs_maturity_moneyness_slices(
    summary: pd.DataFrame,
    *,
    algo: str,
    greeks_method: str,
    volatility: float | None,
    output_path: Path,
) -> Path:
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    axs = axs.ravel()

    methods = list(summary["path_sampler"].dropna().unique())
    moneyness_values = sorted(summary["moneyness"].unique())
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    linestyles = ["-", "--", "-.", ":"]
    color_map = {method: colors[i % len(colors)] for i, method in enumerate(methods)}
    linestyle_map = {
        moneyness: linestyles[i % len(linestyles)]
        for i, moneyness in enumerate(moneyness_values)
    }

    for ax, col in zip(axs, PLOT_COLUMNS):
        for method in methods:
            for moneyness in moneyness_values:
                sub = summary.loc[
                    (summary["path_sampler"] == method) & (summary["moneyness"] == moneyness)
                ].sort_values("maturity")
                if len(sub) == 0:
                    continue
                ax.plot(
                    sub["maturity"],
                    sub[col],
                    marker="o",
                    markersize=4,
                    linewidth=1.8,
                    color=color_map[method],
                    linestyle=linestyle_map[moneyness],
                    label=f"{method}, {sub['moneyness_label'].iloc[0]}",
                )
        ax.set_title(col)
        ax.set_xlabel("maturity")
        ax.grid(True, alpha=0.3)

    handles, labels = axs[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center right", frameon=False, title="Method, moneyness")
    sigma_label = "all sigma" if volatility is None else f"sigma={volatility:g}"
    fig.suptitle(
        f"Greeks vs maturity across methods and moneyness: {algo} "
        f"({greeks_method}, {sigma_label})"
    )
    fig.tight_layout(rect=[0.0, 0.0, 0.86, 0.96])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved maturity-slice Greeks plot to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot price and Greeks vs maturity across moneyness slices."
    )
    parser.add_argument("--config", default="table_greeks_plots", help="Config name from optimal_stopping.run.configs.")
    parser.add_argument("--algo", default="RLSMSoftplus")
    parser.add_argument("--greeks-method", default="regression")
    parser.add_argument("--volatility", type=float, default=0.2)
    parser.add_argument(
        "--methods",
        default=None,
        help=(
            "Comma-separated path sampler methods to plot. "
            "Defaults to all methods available after filtering."
        ),
    )
    parser.add_argument(
        "--moneyness-slices",
        default="0.90,1.00,1.10",
        help="Comma-separated target S/K slices to plot.",
    )
    parser.add_argument(
        "--use-payoff-as-input",
        default=None,
        choices=["true", "false", "none"],
        help="Filter on use_payoff_as_input.",
    )
    parser.add_argument(
        "--train-itm-only",
        default=None,
        choices=["true", "false", "none"],
        help="Filter on train_ITM_only.",
    )
    parser.add_argument("--read-which", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = getattr(configs, args.config)
    if not isinstance(config, configs._DefaultConfig):
        raise ValueError(f"{args.config!r} is not a valid plotting config.")

    def _parse_optional_bool(value: str | None) -> bool | None:
        if value in (None, "none"):
            return None
        return value == "true"

    summary = build_maturity_slice_table(
        config,
        algo=args.algo,
        greeks_method=args.greeks_method,
        volatility=args.volatility,
        moneyness_slices=tuple(float(v) for v in args.moneyness_slices.split(",") if v.strip()),
        methods=(
            tuple(v.strip() for v in args.methods.split(",") if v.strip())
            if args.methods
            else None
        ),
        use_payoff_as_input=_parse_optional_bool(args.use_payoff_as_input),
        train_itm_only=_parse_optional_bool(args.train_itm_only),
        read_which=args.read_which,
    )

    if args.output_dir is None:
        output_dir = Path(__file__).resolve().parents[2] / "plots"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    method_stem = "all_methods" if args.methods is None else args.methods.replace(",", "-")
    stem = f"greeks_vs_maturity_moneyness_{args.algo}_{args.greeks_method}_{method_stem}"
    plot_greeks_vs_maturity_moneyness_slices(
        summary,
        algo=args.algo,
        greeks_method=args.greeks_method,
        volatility=args.volatility,
        output_path=output_dir / f"{stem}.pdf",
    )
    summary.to_csv(output_dir / f"{stem}.csv", index=False)


if __name__ == "__main__":
    main()
