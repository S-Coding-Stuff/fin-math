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
        return self.X * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - self.S_0 * norm.cdf(-self.d1)


class Greeks(OptionPrice):
    """Utilising the Greeks from the Black-Scholes formula"""

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
    """Class for calculating the different types of volatility"""

    @staticmethod
    def implied_volatility(market_price, S_0, X, r, T, call=True):
        """
        Implied Volatility calculation using the Black-Scholes model
        :param market_price: observed option price
        :param S_0: spot price
        :param X: strike price
        :param r: risk-free interest rate
        :param T: time to maturity (years)
        :param call: True for call, False for put
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
    def historic_volatility(prices, trading_days=252):
        """
        Calculate the historical volatility from a series of prices.
        :param prices: daily closing prices
        :param trading_days: number of trading days in a year
        """
        prices = np.array(prices)
        log_returns = np.diff(np.log(prices))
        return np.std(log_returns, ddof=1) * np.sqrt(trading_days)
