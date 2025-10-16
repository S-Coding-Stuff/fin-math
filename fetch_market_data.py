import yfinance as yf
import datetime
from matplotlib import pyplot as plt
import numpy as np
import plotly.graph_objects as go

from black_scholes import Volatility


class MarketData:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)

    def getSpot(self) -> float:
        return self.stock.history(period="1d")['Close'].iloc[-1]

    def _getStockPrice(self, period='10y', interval='1d'):
        """
        Fetch historical close prices as a pandas Series so callers can
        access both dates (index) and values. Use `.to_numpy()` if you only
        need the raw array.
        """
        history = self.stock.history(period=period, interval=interval)
        return history['Close']

    def plotStockPrice(self, period='10y', interval='1d'):
        prices = self._getStockPrice(period=period, interval=interval)
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=prices.index, y=prices.values,
            mode='lines',
            name=f'{self.ticker} Close Price'
        ))

        # Show only the current price as a red dot
        fig.add_trace(go.Scatter(
            x=[prices.index[-1]], y=[prices.iloc[-1]],
            mode='markers',
            marker=dict(color='red', size=10),
            name=f'${prices.iloc[-1]:.2f}'
        ))

        fig.update_layout(
            title=f'{self.ticker} Stock Price',
            xaxis_title='Date',
            yaxis_title='Close Price'
        )
        fig.show()

    def getOptionValues(self, expiry=None, call=True, strike_index=None):
        S_0 = self.getSpot()
        expiries = self.getExpiries()
        if expiry is None:
            expiry = expiries[0]
        T = self.getTimeToMaturity(expiry)

        chain = self.stock.option_chain(expiry)
        options = chain.calls if call else chain.puts
        if options.empty:
            raise ValueError(f"No option data available for {self.ticker} at expiry {expiry}.")

        if strike_index is None:
            strike_idx = int(np.abs(options['strike'].to_numpy() - S_0).argmin())
        else:
            if strike_index < 0 or strike_index >= len(options):
                print(
                    f"Requested strike_index {strike_index} is out of range; using strike closest to spot instead."
                )
                strike_idx = int(np.abs(options['strike'].to_numpy() - S_0).argmin())
            else:
                strike_idx = strike_index

        strike = options['strike'].iloc[strike_idx]
        sigma = options['impliedVolatility'].iloc[strike_idx]
        if not np.isfinite(sigma) or sigma < 0.001 or sigma > 2:
            sigma = Volatility.historic_volatility(self._getStockPrice().to_numpy())
            print(f'Falling back to Historical Volatility: {sigma:.2f}')

        r = self.getRiskFreeRate(T)

        print({"S0": float(S_0), "X": float(strike), "r": float(r), "sigma": float(sigma), "T": float(T)})
        return S_0, strike, r, sigma, T

    def getExpiries(self):
        return self.stock.options

    def getTimeToMaturity(self, expiry):
        expiry_date = datetime.datetime.strptime(expiry, "%Y-%m-%d").date()
        today = datetime.date.today()
        return (expiry_date - today).days / 252

    def getRiskFreeRate(self, time=1):
        """
        Find the risk-free rate of return over a specified period
        :param time: Time in years
        :return:
        """
        yields = {0.25: '^IRX', 1: '^UST1Y', 2: '^UST2Y', 5: '^FVX', 10: '^TVX', 30: '^TYX'}

        try:
            closest_maturity = min(yields.keys(), key=lambda x: abs(x - time))
            ticker = yields[closest_maturity]

            treasury = yf.Ticker(ticker)
            r = treasury.history(period="1d")['Close'].iloc[-1] / 100
            return r

        except Exception as e:
            print(f'Could not fetch Risk Free Rate: {e}. Using default 5%')
            return 0.05
