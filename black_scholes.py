import numpy as np
import scipy.optimize
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


class Volatility:

    def __init__(self, S_0, r, trading_days):
        self.S_0 = S_0
        self.r = r
        self.trading_days = trading_days

    @staticmethod
    def implied_volatility(self, market_price, S_0, X, r, T, call=True):
        """
        Solve for implied volatility using the Black–Scholes model.

        args:
            market_price: observed option price
            S0: spot price
            X: strike price
            r: risk-free rate
            T: time to maturity (in years)
            call: True for call, False for put
        """

        def d1(sigma):
            return (np.log(S_0 / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

        def d2(sigma):
            return d1(sigma) - sigma * np.sqrt(T)

        def blackscholes_price(sigma):
            if call:
                return S_0 * norm.cdf(d1(sigma)) - X * np.exp(-r * T) * norm.cdf(d2(sigma))
            else:
                return X * np.exp(-r * T) * norm.cdf(-d2(sigma)) - S_0 * np.exp(-r * T) * norm.cdf(-d1(sigma))

        def objective(sigma):
            return blackscholes_price(sigma) - market_price

        try:
            return scipy.optimize.brentq(objective, 1e-6, 5.0)  # Between 0.000001% and 500%
        except ValueError:
            return np.nan

    @staticmethod
    def historic_volatility(self, prices, trading_days=252):
        return None
