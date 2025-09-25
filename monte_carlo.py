import numpy as np
import matplotlib.pyplot as plt


class MonteCarloPricing:
    def __init__(self, S_0, X, sigma, T, r=None, mu=None, num_iter=1000, steps=252):
        self.S_0 = S_0
        self.X = X
        self.sigma = sigma
        self.r = r  # Risk Free rate
        self.mu = mu  # Real-World Drift
        self.T = T
        self.num_iter = num_iter
        self.steps = steps

    def simulate_paths(self, risk_neutral=True):
        """Simulating stock prices over time using Geometric Brownian Motion"""
        dt = self.T / self.steps
        Z = np.random.normal(size=(self.steps, self.num_iter))
        paths = np.zeros((self.steps + 1, self.num_iter))
        paths[0] = self.S_0

        drift = self.r if risk_neutral else self.mu

        for t in range(1, self.steps + 1):
            paths[t] = paths[t - 1] * np.exp(
                (drift - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * Z[t - 1])

        return paths

    def plot_paths(self, num_plots=1, call=True):
        paths = self.simulate_paths()
        plt.figure(figsize=(12, 8))

        if num_plots > 1:
            for i in range(min(num_plots, self.num_iter)):
                plt.plot(paths[:, i], lw=1, alpha=0.7)

        else:
            plt.plot(paths[:, 0], lw=2)
            S_T = paths[-1, 0]
            plt.scatter(len(paths) - 1, S_T, color='red', s=10, zorder=5, label=f'{paths[-1, 0]:.2f}')
            plt.hlines(self.X, 0, self.steps, label='Strike', color='orange', linestyle='-')

            # payoff = max(S_T - self.X, 0) if call else max(self.X - S_T, 0)

            pl = S_T - self.X if call else self.X - S_T

            if pl < 0:
                plt.vlines(self.steps, S_T, self.X, color='red', label=f'{pl:.2f}')
            else:
                plt.vlines(self.steps, self.X, S_T, color='green', label=f'+{pl:.2f}')

        plt.title(f'Monte Carlo Simulation for {num_plots} Paths')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.grid(True)
        plt.legend()
        plt.show()

    def european(self, call=True):
        """Price a European option using Monte Carlo simulation"""
        paths = self.simulate_paths()
        S_T = paths[-1]
        if call:
            payoffs = np.maximum(S_T - self.X, 0)  # Basic call option payoff equation
        else:
            payoffs = np.maximum(self.X - S_T, 0)  # Basic put option payoff equation

        discounted = np.exp(-self.r * self.T) * payoffs
        return np.mean(discounted), np.std(discounted) / np.sqrt(self.num_iter)
    
    def american(self, call=True):
        """Price an American option using the Least Squares Monte Carlo (LSM) method"""
        paths = self.simulate_paths()
        n_steps, n_paths = paths.shape
        dt = self.T / (n_steps - 1)
        discount = np.exp(-self.r * dt)

        if call:
            payoff = np.maximum(paths - self.X, 0)
        else:
            payoff = np.maximum(self.X - paths, 0)

        cashflow = payoff[-1].copy()

        for t in range(n_steps - 2, -1, -1):
            in_the_money = payoff[t] > 0  # In-the-money paths
            if np.any(in_the_money):
                # Regression on in-the-money paths
                X = paths[t, in_the_money]
                Y = cashflow[in_the_money] * discount
                # Use simple basis: [1, S, S^2]
                A = np.vstack([np.ones_like(X), X, X**2]).T
                coeffs, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
                continuation = coeffs[0] + coeffs[1]*X + coeffs[2]*X**2
                exercise = payoff[t, in_the_money]

                # Exercise if immediate payoff > continuation value
                exercise_now = exercise > continuation
                cashflow[in_the_money] = np.where(exercise_now, exercise, cashflow[in_the_money] * discount)
            cashflow[~in_the_money] *= discount # Discount out-of-the-money paths

        price = np.mean(cashflow)
        stderr = np.std(cashflow * np.exp(-self.r * dt)) / np.sqrt(n_paths)
        return price, stderr