import yfinance as yf
import datetime
from matplotlib import pyplot as plt


class MarketData:
    def __init__(self, ticker):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)

    def getSpot(self):
        return self.stock.history(period="1d")['Close'].iloc[-1]

    def getStockPrice(self, period='10y', interval='1d'):
        history = self.stock.history(period=period, interval=interval)
        return history['Close']

    def plotStockPrice(self, period='10y', interval='1d'):
        prices = self.getStockPrice(period=period, interval=interval)
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        ax.set_facecolor('black')
        plt.plot(prices.index, prices.values, label=f'{self.ticker} Close Price', color='white')
        plt.title(f'{self.ticker} Stock Price')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.grid(True)
        plt.show()

    def getExpiries(self):
        return self.stock.options

    def getTimeToMaturity(self, expiry):
        expiry_date = datetime.datetime.strptime(expiry, "%Y-%m-%d")
        today = datetime.date.today()
        return (expiry_date - today).days / 252

    def getRiskFreeRate(self, time=1):
        """
        Find the risk-free rate of return over a specified period
        :param time: Time in years
        :return:
        """
        yields = {0.25 : '^IRX', 1 : '^UST1Y', 2 : '^UST2Y', 5 : '^FVX', 10 : '^TVX', 30 : '^TYX'}

        try:
            closest_maturity = min(yields.keys(), key=lambda x: abs(x - time))
            ticker = yields[closest_maturity]

            treasury = yf.Ticker(ticker)
            r = treasury.history(period="1d")['Close'].iloc[-1] / 100
            return r

        except Exception as e:
            print(f'Could not fetch Risk Free Rate: {e}. Using default 5%')
            return 0.05