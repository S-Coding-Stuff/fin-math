import numpy as np
from scipy.stats import norm


class OptionPrice:
    """Pricing an option using the Black Scholes formula"""

    def __init__(self, S_0, X, r, sigma, T):
        self.S_0 = S_0  # Spot price at time 0
        self.X = X  # Strike price
        self.r = r  # Risk-free rate
        self.sigma = sigma  # Volatility
        self.T = T  # Time to maturity

        # d1 and d2 from Black-Scholes
        self.d1 = (np.log(S_0 / X) + (r + (1 / 2) * sigma ** 2) * T) / (sigma * np.sqrt(T))
        self.d2 = self.d1 - sigma * np.sqrt(T)

    def call(self):
        return self.S_0 * norm.cdf(self.d1) - self.X * np.exp(-self.r * self.T) * norm.cdf(self.d2)

    def put(self):
        return self.X * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - self.S_0 * np.exp(-self.r * self.T) * norm.cdf(
            -self.d1)


class Greeks(OptionPrice):
    """Utilising the Greeks from the Black-Scholes formula"""

    def __init__(self):
        super(OptionPrice, self).__init__()

    # Delta, where call is a boolean variable to denote whether it's for a call or put option
    # Delta measures the rate of change of the theoretical option value w.r.t changes in the underlying asset's price
    def delta(self, call=True):
        if call:
            return norm.cdf(self.d1)
        else:
            return -norm.cdf(-self.d1)

    def dual_delta(self):
        return -np.exp(-self.r * self.T) * norm.cdf(self.d2)

    # Vega measures the sensitivity to volatility
    def vega(self):
        return self.S_0 * np.sqrt(self.T) * norm.pdf(self.d1)

    # Gamma measures the rate of change in the delta w.r.t changes in the underlying asset price
    # 2nd Derivative of Delta
    def gamma(self):
        return norm.pdf(self.d1) * (1 / (self.sigma * self.S_0 * np.sqrt(self.T)))

    # Theta measures the sensitivity of the option value to time
    def theta(self, call=True):
        term1 = (-self.S_0 * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T))
        if call:
            term2 = self.r * self.X * np.exp(-self.r * self.T) * norm.pdf(self.d2)
            return term1 - term2
        else:
            term2 = self.r * self.X * np.exp(-self.r * self.T) * norm.pdf(-self.d2)
            return term1 + term2

    # Rho measures the rate at which the derivative changes relative to a change in the risk-free interest rate
    def rho(self, call=True):
        if call:
            return self.X * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        else:
            return -self.X * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
