import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from dataclasses import dataclass

from monte_carlo import MonteCarloPricing

@dataclass
class MonteCarloFiniteDifference:
    """Finite-difference Greeks that reuse a Monte Carlo pricer and common random numbers."""
    pricer: MonteCarloPricing
    call: bool = True
    antithetic: bool = True
    risk_neutral: bool = True

    # Bump sizes for finite difference
    S0_bump: float = 1e-2  # For delta and gamma
    sigma_bump: float = 1e-2   # For vega
    r_bump: float = 1e-3  # For rho

    def _common_random_normals(self) -> np.ndarray:
        """Generate the same normal draws for all bumps."""
        steps = self.pricer.steps
        paths = self.pricer.num_paths
        rng = self.pricer.rng
        state = rng.bit_generator.state

        if self.antithetic:
            half_paths = (paths + 1) // 2
            z_half = rng.standard_normal(size=(steps, half_paths))
            z = np.concatenate((z_half, -z_half), axis=1)[:, :paths]
        else:
            z = rng.standard_normal(size=(steps, paths))

        rng.bit_generator.state = state
        return z

    def _price(self, *, S0: float | None = None, sigma: float | None = None,
               r: float | None = None, mu: float | None = None,
               Z: np.ndarray | None = None, risk_neutral: bool | None = None) -> tuple[float, float]:
        pricer = self.pricer
        S0 = pricer.S_0 if S0 is None else S0
        sigma = pricer.sigma if sigma is None else sigma
        risk_neutral = self.risk_neutral if risk_neutral is None else risk_neutral

        if risk_neutral:
            r = pricer.r if r is None else r
            drift = r
        else:
            mu = pricer.mu if mu is None else mu
            if mu is None:
                raise ValueError("mu must be set on the pricer (or passed explicitly) when risk_neutral=False.")
            drift = mu

        if drift is None:
            raise ValueError("Risk-free rate r is required for risk-neutral pricing.")

        if Z is None:
            Z = self._common_random_normals()

        dt = pricer.T / pricer.steps
        log_returns = (drift - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        S_T = S0 * np.exp(np.cumsum(log_returns, axis=0)[-1])

        if self.call:
            payoffs = np.maximum(S_T - pricer.X, 0.0)
        else:
            payoffs = np.maximum(pricer.X - S_T, 0.0)

        discount = np.exp(-(pricer.r if r is None else r) * pricer.T) if risk_neutral else 1.0
        discounted = discount * payoffs

        mean_price = float(discounted.mean())
        stderr = float(discounted.std(ddof=1) / np.sqrt(discounted.size))
        return mean_price, stderr

    def price(self) -> tuple[float, float]:
        """Monte Carlo option price (mean, standard error) with current settings."""
        return self._price()

    def delta(self, h: float | None = None) -> float:
        h = self.S0_bump if h is None else h
        if self.pricer.S_0 - h <= 0:
            raise ValueError("Spot bump would make S0 non-positive; choose a smaller h.")

        Z = self._common_random_normals()
        price_up, _ = self._price(S0=self.pricer.S_0 + h, Z=Z)
        price_down, _ = self._price(S0=self.pricer.S_0 - h, Z=Z)
        return (price_up - price_down) / (2 * h)

    def gamma(self, h: float | None = None) -> float:
        h = self.S0_bump if h is None else h
        if self.pricer.S_0 - h <= 0:
            raise ValueError("Spot bump would make S0 non-positive; choose a smaller h.")

        Z = self._common_random_normals()
        price_up, _ = self._price(S0=self.pricer.S_0 + h, Z=Z)
        price_mid, _ = self._price(S0=self.pricer.S_0, Z=Z)
        price_down, _ = self._price(S0=self.pricer.S_0 - h, Z=Z)
        return (price_up - 2 * price_mid + price_down) / (h ** 2)

    def vega(self, h: float | None = None) -> float:
        h = self.sigma_bump if h is None else h
        if self.pricer.sigma + h <= 0 or self.pricer.sigma - h <= 0:
            raise ValueError("Volatility bumps must keep sigma positive.")

        Z = self._common_random_normals()
        price_up, _ = self._price(sigma=self.pricer.sigma + h, Z=Z)
        price_down, _ = self._price(sigma=self.pricer.sigma - h, Z=Z)
        return (price_up - price_down) / (2 * h)

    def rho(self, h: float | None = None) -> float:
        h = self.r_bump if h is None else h
        if self.pricer.r is None:
            raise ValueError("The pricer's risk-free rate r must be set to compute rho.")

        Z = self._common_random_normals()
        price_up, _ = self._price(r=self.pricer.r + h, Z=Z)
        price_down, _ = self._price(r=self.pricer.r - h, Z=Z)
        return (price_up - price_down) / (2 * h)

class FiniteDifference:
    """Class for computing numerical derivatives using Central Difference."""
    def __init__(self, func: Callable[[float], float], h: float = 1e-5):
        self.func = func
        self.h = h

    def derivative(self, x: float) -> float:
        return (self.func(x + self.h) - self.func(x - self.h)) / (2 * self.h)

    def second_derivative(self, x: float) -> float:
        return (self.func(x + self.h) - 2 * self.func(x) + self.func(x - self.h)) / (self.h ** 2)

    def plot_derivative(self, x_range: np.ndarray):
        derivatives = [self.derivative(x) for x in x_range]
        plt.plot(x_range, derivatives)
        plt.title("First Derivative")
        plt.xlabel("x")
        plt.ylabel("f'(x)")
        plt.grid()
        plt.show()

    def plot_second_derivative(self, x_range: np.ndarray):
        second_derivatives = [self.second_derivative(x) for x in x_range]
        plt.plot(x_range, second_derivatives)
        plt.title("Second Derivative")
        plt.xlabel("x")
        plt.ylabel("f''(x)")
        plt.grid()
        plt.show()