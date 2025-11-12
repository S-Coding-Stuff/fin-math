import numpy as np
import matplotlib.pyplot as plt

class BinomialPricing:
    def __init__(self, S_0: float, K: float, r: float, sigma: float, T: float, steps: int):
        self.S_0 = S_0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.steps = steps

    def _build_tree(self):
        dt = self.T / self.steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = np.exp(-self.sigma * np.sqrt(dt)) 
        p = (1/2) + ((self.r - 0.5 * self.sigma**2) * np.sqrt(dt) / (2 * self.sigma))

        # Initialising asset prices at maturity
        asset_prices = np.zeros((self.steps + 1, self.steps + 1))
        for i in range(self.steps + 1):
            for j in range(i + 1):
                asset_prices[j, i] = self.S_0 * (u ** (i - j)) * (d ** j)

        return asset_prices, p, u, d, dt

    def european(self, call: bool = True):
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

    def american(self, call: bool = True):
        asset_prices, p, u, d, dt = self._build_tree()
        option_values = np.zeros((self.steps + 1, self.steps + 1))

        # Since C(American) = C(European) for calls on non-dividend paying stocks
        if call:
            option_values[:, self.steps] = np.maximum(0, asset_prices[:, self.steps] - self.K)
        else:
            option_values[:, self.steps] = np.maximum(0, self.K - asset_prices[:, self.steps])

        for i in range(self.steps - 1, -1, -1):
            for j in range(i + 1):
                continuation = np.exp(-self.r * dt) * (p * option_values[j, i + 1] + (1 - p) * option_values[j + 1, i + 1])
                if call:
                    intrinsic = max(0, asset_prices[j, i] - self.K)
                else:
                    intrinsic = max(0, self.K - asset_prices[j, i])
                option_values[j, i] = max(intrinsic, continuation)

        return option_values[0, 0]

__all__ = ['BinomialPricing']