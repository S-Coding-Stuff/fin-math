"""Helper functions to exercise MC sensitivity routines from notebooks."""
import numpy as np
from monte_carlo import MonteCarloPricing
from mc_sensitivities import pathwise_delta, pathwise_vega


def run_pricer_and_sensitivities(
    S_0: float = 100.0,
    strike: float = 100.0,
    sigma: float = 0.2,
    r: float = 0.05,
    T: float = 1.0,
    num_paths: int = 50_000,
    steps: int = 252,
    call: bool = True,
    use_common_random: bool = True,
):
    """Return option price, delta, and vega using shared Monte Carlo draws."""

    mc = MonteCarloPricing(S_0=S_0, X=strike, sigma=sigma, T=T, r=r, num_paths=num_paths, steps=steps)

    Z = None
    if use_common_random:
        Z = np.random.standard_normal((steps, num_paths))

    price, price_se = mc.european(call=call)
    delta, delta_se = pathwise_delta(mc, call=call, Z=Z, return_std=True)
    vega, vega_se = pathwise_vega(mc, call=call, Z=Z, return_std=True)

    return {
        "price": f"{price:.4f}",
        "price_stderr": price_se,
        "delta": f"{delta:.4f}",
        "delta_stderr": delta_se,
        "vega": f"{vega:.4f}",
        "vega_stderr": vega_se,
    }

if __name__ == "__main__":
    test_results = run_pricer_and_sensitivities()
    print(test_results)