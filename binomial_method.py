import numpy as np
import matplotlib.pyplot as plt

class BinomialPricing:
    def __init__(self, S_0, K, r, sigma, T, steps):
        self.S_0 = S_0  
        self.K = K      
        self.r = r      
        self.sigma = sigma
        self.T = T      
        self.steps = steps  

    def _build_tree(self):
        dt = self.T / self.steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u  
        p = (np.exp(self.r * dt) - d) / (u - d) 

        # Initialize asset prices at maturity
        asset_prices = np.zeros((self.steps + 1, self.steps + 1))
        for i in range(self.steps + 1):
            for j in range(i + 1):
                asset_prices[j, i] = self.S_0 * (u ** (i - j)) * (d ** j)

        return asset_prices, p, u, d, dt

    def european(self, call=True):
        asset_prices, p, u, d, dt = self._build_tree()
        option_values = np.zeros((self.steps + 1, self.steps + 1))

        # Calculate option values at maturity
        if call:
            option_values[:, self.steps] = np.maximum(0, asset_prices[:, self.steps] - self.K)
        else:
            option_values[:, self.steps] = np.maximum(0, self.K - asset_prices[:, self.steps])

        # Backward induction to get option price at t=0
        for i in range(self.steps - 1, -1, -1):
            for j in range(i + 1):
                option_values[j, i] = np.exp(-self.r * dt) * (p * option_values[j, i + 1] + (1 - p) * option_values[j + 1, i + 1])

        return option_values[0, 0]

    def american(self, call=True):
        asset_prices, p, u, d, dt = self._build_tree()
        option_values = np.zeros((self.steps + 1, self.steps + 1))