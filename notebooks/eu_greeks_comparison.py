"""Compare option Greeks from Black-Scholes finite differences vs Monte Carlo finite differences.

Run as a standalone script to print a small table and optionally render a bar
chart comparing the two approaches for a European option under shared inputs.
"""
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
from greeks.bs_fd import BlackScholesFiniteDifference


def bs_fd_greeks(*, S_0: float, X: float, r: float, sigma: float, T: float, call: bool) -> Dict[str, float]:
    """Finite-difference Greeks using Black-Scholes closed form for pricing."""
    bs_fd = BlackScholesFiniteDifference(S_0=S_0, X=X, r=r, sigma=sigma, T=T, call=call)
    return {
        "delta": bs_fd.delta(),
        "gamma": bs_fd.gamma(),
        "vega": bs_fd.vega(),
        "theta": bs_fd.theta(),
        "rho": bs_fd.rho(),
    }


def mc_fd_greeks(
    *,
    S_0: float,
    X: float,
    r: float,
    sigma: float,
    T: float,
    call: bool,
    num_paths: int = 50_000,
    steps: int = 50,
    seed: int = 123,
) -> Dict[str, float]:
    """Finite-difference Greeks from Monte Carlo pricing with common random numbers."""
    rng = np.random.default_rng(seed)
    pricer = MonteCarloPricing(
        S_0=S_0,
        X=X,
        sigma=sigma,
        T=T,
        r=r,
        num_paths=num_paths,
        steps=steps,
        rng=rng,
    )
    mc_fd = MonteCarloFiniteDifference(
        pricer,
        call=call,
        antithetic=True,
        risk_neutral=True,
        style="european",
        basis_fn="laguerre",
    )
    return {
        "delta": mc_fd.greek("delta"),
        "gamma": mc_fd.greek("gamma"),
        "vega": mc_fd.greek("vega"),
        "theta": mc_fd.greek("theta"),
        "rho": mc_fd.greek("rho"),
    }


def build_table(*, call: bool = True) -> pd.DataFrame:
    """Return a DataFrame comparing BS FD and MC FD Greeks."""
    params = dict(S_0=100.0, X=100.0, r=0.05, sigma=0.2, T=1.0, call=call)
    bs = bs_fd_greeks(**params)
    mc = mc_fd_greeks(**params)

    rows = []
    for kind in ["delta", "gamma", "vega", "theta", "rho"]:
        rows.append({"greek": kind, "method": "Black-Scholes FD", "value": bs[kind]})
        rows.append({"greek": kind, "method": "MC FD (antithetic)", "value": mc[kind]})
        rows.append({"greek": kind, "method": "Difference (MC - BS)", "value": mc[kind] - bs[kind]})
    return pd.DataFrame(rows)


def plot_comparison(df: pd.DataFrame) -> None:
    """Bar chart comparing BS FD vs MC FD Greeks; ignores difference rows."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot.")
        return

    pivot = df[df["method"].isin(["Black-Scholes FD", "MC FD (antithetic)"])].pivot(
        index="greek", columns="method", values="value"
    )
    pivot = pivot.loc[["delta", "gamma", "vega", "theta", "rho"]]

    ax = pivot.plot(kind="bar", figsize=(8, 5))
    ax.set_title("Greeks: Black-Scholes FD vs Monte Carlo FD")
    ax.set_ylabel("Value")
    ax.set_xlabel("Greek")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def main() -> None:
    pd.set_option("display.precision", 6)
    table = build_table(call=True)
    print(table.to_string(index=False))
    plot_comparison(table)


if __name__ == "__main__":
    main()
