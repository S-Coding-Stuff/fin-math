import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from dataclasses import dataclass

from monte_carlo import MonteCarloPricing
from black_scholes import OptionPrice

# Basic Finite Difference Class for mathematical understanding
class FiniteDifference:
    """Class for computing numerical derivatives using central differences for 1D functions."""

    def __init__(self, func: Callable[[float], float], h: float = 1e-5):
        self.func = func
        self.h = float(h)

    def derivative(self, x: float) -> float:
        return (self.func(x + self.h) - self.func(x - self.h)) / (2.0 * self.h)

    def second_derivative(self, x: float) -> float:
        return (self.func(x + self.h) - 2.0 * self.func(x) + self.func(x - self.h)) / (self.h ** 2)

    def plot_derivative(self, x_range: np.ndarray):
        derivatives = [self.derivative(float(x)) for x in x_range]
        plt.plot(x_range, derivatives)
        plt.title("First Derivative")
        plt.xlabel("x")
        plt.ylabel("f'(x)")
        plt.grid()
        plt.show()

    def plot_second_derivative(self, x_range: np.ndarray):
        second_derivatives = [self.second_derivative(float(x)) for x in x_range]
        plt.plot(x_range, second_derivatives)
        plt.title("Second Derivative")
        plt.xlabel("x")
        plt.ylabel("f''(x)")
        plt.grid()
        plt.show()


class BlackScholesFiniteDifference:
    """Finite-difference Greeks for Black-Scholes pricing via simple parameter bumps.

    Uses the closed form OptionPrice by default, but you can pass a custom pricingfunction 
    `pricing_func(S_0, X, r, sigma, T, call: bool)` if desired. """

    def __init__(self, S_0: float, X: float, r: float, sigma: float, T: float, *, call: bool = True, 
                 h_S: float = 1e-2, h_sigma: float = 1e-4, h_r: float = 1e-4, h_T: float = 1e-4,
                 pricing_func: Callable[[float, float, float, float, float, bool], float] | None = None) -> None: 
        
        self.S_0 = float(S_0)
        self.X = float(X)
        self.r = float(r)
        self.sigma = float(sigma)
        self.T = float(T)
        self.call = bool(call)

        self.h_S = float(h_S)
        self.h_sigma = float(h_sigma)
        self.h_r = float(h_r)
        self.h_T = float(h_T)

        self._pricing_func = pricing_func

    def _price(self, *, S_0: float | None = None, X: float | None = None,
               r: float | None = None, sigma: float | None = None, T: float | None = None,
               call: bool | None = None) -> float:
        S0 = self.S_0 if S_0 is None else float(S_0); X = self.X if X is None else float(X)
        r = self.r if r is None else float(r); sigma = self.sigma if sigma is None else float(sigma)
        T = self.T if T is None else float(T); is_call = self.call if call is None else bool(call)

        if self._pricing_func is not None:
            return float(self._pricing_func(S0, X, r, sigma, T, is_call))

        opt = OptionPrice(S0, X, r, sigma, T)
        return float(opt.call() if is_call else opt.put())

    def price(self) -> float:
        return self._price()

    def delta(self) -> float:
        h = self.h_S; S0 = self.S_0
        if S0 - h <= 0.0:
            f_up = self._price(S_0=S0 + h)
            f_0 = self._price(S_0=S0)
            return (f_up - f_0) / h
        f_up = self._price(S_0=S0 + h)
        f_dn = self._price(S_0=S0 - h)
        return (f_up - f_dn) / (2.0 * h)

    def gamma(self) -> float:
        h = self.h_S; S0 = self.S_0
        if S0 - h <= 0.0:
            # forward second difference
            f0 = self._price(S_0=S0)
            f1 = self._price(S_0=S0 + h)
            f2 = self._price(S_0=S0 + 2.0 * h)
            return (f2 - 2.0 * f1 + f0) / (h * h)
        f_up = self._price(S_0=S0 + h)
        f_0 = self._price(S_0=S0)
        f_dn = self._price(S_0=S0 - h)
        return (f_up - 2.0 * f_0 + f_dn) / (h * h)

    def vega(self) -> float:
        h = self.h_sigma; sig = self.sigma
        if sig - h <= 0.0:
            f_up = self._price(sigma=sig + h)
            f_0 = self._price(sigma=sig)
            return (f_up - f_0) / h
        f_up = self._price(sigma=sig + h)
        f_dn = self._price(sigma=sig - h)
        return (f_up - f_dn) / (2.0 * h)

    def theta(self) -> float:
        h = self.h_T; T = self.T
        if T - h <= 0.0:
            f_up = self._price(T=T + h)
            f_0 = self._price(T=T)
            return (f_up - f_0) / h
        f_up = self._price(T=T + h)
        f_dn = self._price(T=T - h)
        return (f_up - f_dn) / (2.0 * h)

    def rho(self) -> float:
        h = self.h_r; r = self.r
        f_up = self._price(r=r + h)
        f_dn = self._price(r=r - h)
        return (f_up - f_dn) / (2.0 * h)

__all__ = ['FiniteDifference', 'BlackScholesFiniteDifference']
