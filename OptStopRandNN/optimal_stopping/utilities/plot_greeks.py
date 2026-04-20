"""
author: Florian Krach
"""
import argparse
from pathlib import Path
import socket
import sys

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.append(str(PACKAGE_ROOT))

from optimal_stopping.run import configs
from optimal_stopping.utilities import read_data

if "ada-" not in socket.gethostname():
    SERVER = False
else:
    SERVER = True

if SERVER:
    SEND = True
    matplotlib.use("Agg")


class SendBotMessage:
    def __init__(self):
        pass

    @staticmethod
    def send_notification(text, *args, **kwargs):
        print(text)


try:
    from telegram_notifications import send_bot_message as SBM
except Exception:
    SBM = SendBotMessage()

chat_id = "-399803347"


def _normalize_sampler(series: pd.Series) -> pd.Series:
    normalized = series.fillna("mc").astype(str).str.strip()
    return normalized.mask(normalized.eq(""), "mc")


def _sampler_stem(path_samplers: tuple[str, ...] | None) -> str:
    if path_samplers is None:
        return "all_methods"
    if len(path_samplers) == 1:
        return path_samplers[0]
    return "-".join(path_samplers)


def plot_greeks(
    config: configs._DefaultConfig,
    greeks_method="regression",
    algo="RLSMSoftplus",
    volatilities=(0.1, 0.2, 0.3, 0.4),
    maturities=(0.5, 1, 2, 4, 8),
    path_samplers: tuple[str, ...] | None = None,
    read_which: int = 0,
    save_path=None,
    save_extras={"bbox_inches": "tight", "pad_inches": 0.01},
):
    del config
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    linestyles = ["-", "-.", "--", ":"]

    df = read_data.read_csvs_conv(which=read_which)
    df = df.drop(columns="duration", errors="ignore")
    if "path_sampler" in df.columns:
        df["path_sampler"] = _normalize_sampler(df["path_sampler"])
    else:
        df["path_sampler"] = "mc"
    available = df.copy()
    df = df.loc[df["algo"] == algo]
    df = df.loc[df["greeks_method"] == greeks_method]
    df = df.loc[df["volatility"].isin(volatilities)]
    df = df.loc[df["maturity"].isin(maturities)]
    if path_samplers is not None:
        df = df.loc[df["path_sampler"].isin(path_samplers)]
    if len(df) == 0:
        raise ValueError(
            "No data matched the requested Greek plot filters. "
            f"Requested algo={algo}, greeks_method={greeks_method}, "
            f"volatilities={tuple(volatilities)}, maturities={tuple(maturities)}, "
            f"path_samplers={tuple(path_samplers) if path_samplers is not None else None}. "
            f"Available algos={sorted(set(available['algo'].dropna()))}, "
            f"available greeks_methods={sorted(set(available['greeks_method'].dropna()))}, "
            f"available volatilities={sorted(set(available['volatility'].dropna()))}, "
            f"available maturities={sorted(set(available['maturity'].dropna()))}, "
            f"available spots={sorted(set(available['spot'].dropna()))}, "
            f"available path_samplers={sorted(set(available['path_sampler'].dropna()))}."
        )

    fig, axs = plt.subplots(2, 3, figsize=(15, 7))
    spots = sorted(list(set(df["spot"].values)))

    for n, plot_label in enumerate(["price", "delta", "gamma", "theta", "rho", "vega"]):
        j = n % 3
        i = n // 3
        ax = axs[i, j]

        for v in volatilities:
            for m in maturities:
                subset = df.loc[(df["volatility"] == v) & (df["maturity"] == m)]
                if len(subset) == 0:
                    continue
                vals = []
                for s in spots:
                    values = subset.loc[subset["spot"] == s, plot_label]
                    printdf = values.loc[values.abs() > 500]
                    if len(printdf) > 0:
                        print(printdf)
                    vals.append(values.median())
                ax.plot(
                    spots,
                    vals,
                    label="$\\sigma=${:.1f} $T=${:.1f}".format(v, m),
                    color=colors[(len(ax.lines)) % len(colors)],
                    linestyle=linestyles[(len(ax.lines)) % len(linestyles)],
                )
        ax.set_title(plot_label)

    if save_path is None:
        save_path = Path(__file__).resolve().parents[2] / "plots"
    else:
        save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    if handles:
        plt.legend(bbox_to_anchor=(1.04, 1.1), loc="center left")
    plt.subplots_adjust(right=0.75)
    stem = f"greeks_plot_{algo}_{greeks_method}"
    if path_samplers is not None:
        stem = f"{stem}_{_sampler_stem(path_samplers)}"
    output_path = save_path / f"{stem}.pdf"
    plt.savefig(output_path, **save_extras)
    print(f"Saved Greek plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot OptStopRandNN Greek curves by sampler.")
    parser.add_argument("--config", default="table_greeks_plots")
    parser.add_argument("--algo", default="RLSMSoftplus")
    parser.add_argument("--greeks-method", default="regression")
    parser.add_argument("--volatilities", default="0.1,0.2,0.3")
    parser.add_argument("--maturities", default="0.5,1,2")
    parser.add_argument(
        "--path-samplers",
        default=None,
        help="Optional comma-separated sampler filter, e.g. mc,sobol.",
    )
    parser.add_argument("--read-which", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--save-path", default=None)
    args = parser.parse_args()

    config = getattr(configs, args.config)
    plot_greeks(
        config=config,
        greeks_method=args.greeks_method,
        algo=args.algo,
        volatilities=tuple(float(v) for v in args.volatilities.split(",") if v.strip()),
        maturities=tuple(float(v) for v in args.maturities.split(",") if v.strip()),
        path_samplers=(
            tuple(v.strip() for v in args.path_samplers.split(",") if v.strip())
            if args.path_samplers
            else None
        ),
        read_which=args.read_which,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
