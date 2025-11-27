"""Compare American option Greeks: Binomial finite-difference vs MC finite-difference (LSM).

Run as a script to print a small table of delta/gamma/vega/theta/rho and an
optional bar chart if matplotlib is installed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# Make repository root importable when run from notebooks/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engines.monte_carlo import MonteCarloPricing
from greeks.mc_fd import MonteCarloFiniteDifference
from greeks.binomial_fd import BinomialFiniteDifference


def binomial_greeks(*, S_0: float, K: float, r: float, sigma: float, T: float, steps: int, call: bool) -> Dict[str, float]:
    """Finite-difference Greeks via binomial American pricing."""
    tree = BinomialFiniteDifference(S_0=S_0, K=K, r=r, sigma=sigma, T=T, steps=steps, call=call)
    return {
        "delta": tree.delta(),
        "gamma": tree.gamma(),
        "vega": tree.vega(),
        "theta": tree.theta(),
        "rho": tree.rho(),
    }


def mc_greeks(*, S_0: float, K: float, r: float, sigma: float, T: float, call: bool,
              num_paths: int, steps: int, seed: int) -> Dict[str, float]:
    """Finite-difference Greeks via Monte Carlo LSM (American)."""
    rng = np.random.default_rng(seed)
    pricer = MonteCarloPricing(S_0=S_0, X=K, sigma=sigma, T=T, r=r, num_paths=num_paths,
                               steps=steps, rng=rng)
    mc_fd = MonteCarloFiniteDifference(pricer, call=call, antithetic=True,
                                       risk_neutral=True, style="american",
                                       basis_fn="laguerre", include_all_paths=True)
    return {
        "delta": mc_fd.greek("delta"),
        "gamma": mc_fd.greek("gamma"),
        "vega": mc_fd.greek("vega"),
        "theta": mc_fd.greek("theta"),
        "rho": mc_fd.greek("rho"),
    }


def build_table(*, call: bool = False) -> pd.DataFrame:
    """Return a comparison table for American option Greeks."""
    params = dict(S_0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, call=call)
    binom = binomial_greeks(**params, steps=800)
    mc = mc_greeks(**params, num_paths=50_000, steps=50, seed=42)

    rows = []
    for kind in ["delta", "gamma", "vega", "theta", "rho"]:
        rows.append({"greek": kind, "method": "Binomial FD", "value": binom[kind]})
        rows.append({"greek": kind, "method": "MC FD (LSM, antithetic)", "value": mc[kind]})
        rows.append({"greek": kind, "method": "Difference (MC - Binomial)", "value": mc[kind] - binom[kind]})
    return pd.DataFrame(rows)


def plot_table(df: pd.DataFrame) -> None:
    """Bar chart for Binomial vs MC FD Greeks with value labels."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return

    pivot = df[df["method"].isin(["Binomial FD", "MC FD (LSM, antithetic)"])].pivot(
        index="greek", columns="method", values="value"
    )
    pivot = pivot.loc[["delta", "gamma", "vega", "theta", "rho"]]

    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_title("American option Greeks: Binomial FD vs MC FD")
    ax.set_ylabel("Value")
    ax.set_xlabel("Greek")
    ax.grid(True, axis="y", alpha=0.25)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(True)

    # Annotate bars with values
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=9)

    plt.tight_layout()
    plt.show()


def main() -> None:
    pd.set_option("display.precision", 6)
    table = build_table(call=False)  # put by default
    print(table.to_string(index=False))
    plot_table(table)


if __name__ == "__main__":
    main()
