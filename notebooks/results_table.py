"""Quick comparison table for American option pricing experiments.

Run as a standalone script to produce a small table of prices and standard
errors using the repository's Monte Carlo engine (PCG64-driven) with and
without antithetic variates. Adjust scenario parameters or path counts as
needed for your slides.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

# Ensure repo root (one level up from notebooks/) is on sys.path so imports work when run in place.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engines.monte_carlo import MonteCarloPricing


def _run_pricer(pricer: MonteCarloPricing, *, call: bool, basis: str, antithetic: bool) -> tuple[float, float]:
    """Run the American LSM pricer and return (price, stderr)."""
    price, stderr = pricer.american(call=call, basis_fn=basis, antithetic=antithetic)
    return float(price), float(stderr)


def build_results_table(
    *,
    S_0: float = 100.0,
    strike: float = 100.0,
    sigma: float = 0.2,
    r: float = 0.05,
    T: float = 1.0,
    call: bool = False,
) -> pd.DataFrame:
    """Assemble a comparison table across random drivers and bases."""
    experiments: list[dict[str, object]] = [
        {
            "label": "American LSM - MC (PCG64) no antithetic",
            "pricer": MonteCarloPricing(
                S_0=S_0,
                X=strike,
                sigma=sigma,
                T=T,
                r=r,
                num_paths=50_000,
                steps=50,
                seed=42,
            ),
            "basis": "laguerre",
            "antithetic": False,
        },
        {
            "label": "American LSM - MC (PCG64) + antithetic",
            "pricer": MonteCarloPricing(
                S_0=S_0,
                X=strike,
                sigma=sigma,
                T=T,
                r=r,
                num_paths=50_000,
                steps=50,
                seed=123,
            ),
            "basis": "laguerre",
            "antithetic": True,
        },
        {
            "label": "American LSM - MC (PCG64) + antithetic, Hermite basis",
            "pricer": MonteCarloPricing(
                S_0=S_0,
                X=strike,
                sigma=sigma,
                T=T,
                r=r,
                num_paths=50_000,
                steps=50,
                seed=456,
            ),
            "basis": "hermite",
            "antithetic": True,
        },
    ]

    rows: list[dict[str, object]] = []
    for exp in experiments:
        price, stderr = _run_pricer(
            exp["pricer"],
            call=call,
            basis=str(exp["basis"]),
            antithetic=bool(exp["antithetic"]),
        )
        rows.append(
            {
                "method": exp["label"],
                "basis": exp["basis"],
                "antithetic": bool(exp["antithetic"]),
                "price": price,
                "std_error": stderr,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    df = build_results_table()
    pd.set_option("display.precision", 6)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
