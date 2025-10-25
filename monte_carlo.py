import numpy as np

class MonteCarloPricing:
    def __init__(self, S_0: float, X: float, sigma: float, T: float, r: float = None, mu: float = None, 
                 num_paths: int = 1000, steps: int = 252, *, rng: np.random.Generator | None = None, 
                 seed: int | None = None):
        self.S_0 = S_0
        self.X = X
        self.sigma = sigma
        self.r = r  # Risk Free rate
        self.mu = mu  # Real-World Drift
        self.T = T # Time to maturity in years
        self.num_paths = num_paths
        self.steps = steps
        self.div = 0.0  # Dividend yield, set to 0 for now, will update later if needed to facilitate

        self.rng = rng if rng is not None else np.random.default_rng(seed)

    def _simulate_paths(self, risk_neutral: bool = True, Z: np.ndarray | None = None, 
                        *, antithetic: bool = False) -> np.ndarray:
        """Simulate stock prices over time using Geometric Brownian Motion."""
        num_paths = self.num_paths
        num_steps = self.steps
        dt = self.T / self.steps # Step size

        if Z is None:
            if antithetic:
                half_paths = (num_paths + 1) // 2
                Z_half = self.rng.standard_normal(size=(num_steps, half_paths))
                Z = np.concatenate((Z_half, -Z_half), axis=1)[:, :num_paths]
            else:
                Z = self.rng.standard_normal(size=(num_steps, num_paths))

        drift_param = self.r if risk_neutral else self.mu
        if drift_param is None:
            raise ValueError("Set r for risk-neutral or mu for real-world simulations before calling _simulate_paths")

        log_returns = (drift_param - self.div - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * Z

        S = np.empty((num_steps + 1, num_paths), dtype=float)
        S[0, :] = self.S_0
        S[1:] = self.S_0 * np.exp(np.cumsum(log_returns, axis=0))

        return S

    def simulate_paths(self, risk_neutral: bool = True, *, antithetic: bool = True) -> np.ndarray:
        """Generate GBM paths; antithetic variates are on by default for variance reduction."""
        return self._simulate_paths(risk_neutral=risk_neutral, antithetic=antithetic)

    def plot_paths(self, num_plots: int = 1, call: bool = True, *, antithetic: bool = True):
        import matplotlib.pyplot as plt
        paths = self._simulate_paths(antithetic=antithetic) # Antithetic Variates Method used while plotting
        plt.figure(figsize=(12, 8))

        if num_plots > 1:
            for i in range(min(num_plots, self.num_paths)):
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

    def european(self, call: bool = True, *, antithetic: bool = True) -> tuple[float, float]:
        """Price a European option via Monte Carlo (returns mean price and standard error)."""
        if self.r is None:
            raise ValueError("Risk-free rate r must be set before pricing under the risk-neutral measure.")

        paths = self._simulate_paths(antithetic=antithetic)
        S_T = paths[-1]
        if call:
            payoffs = np.maximum(S_T - self.X, 0)  # Basic call option payoff equation
        else:
            payoffs = np.maximum(self.X - S_T, 0)  # Basic put option payoff equation

        discounted = np.exp(-self.r * self.T) * payoffs
        return np.mean(discounted), np.std(discounted) / np.sqrt(self.num_paths)

    def american(self, call: bool = True, basis_fn: str = "laguerre", *, antithetic: bool = True,
                 return_diagnostics: bool = False) -> tuple[float, float] | tuple[float, float, dict[str, np.ndarray]]:
        """Price an American option using the Least Squares Monte Carlo method."""
        
        if self.r is None:
            raise ValueError("Risk-free rate r must be set before pricing under the risk-neutral measure.")

        paths = self._simulate_paths(antithetic=antithetic)
        n_steps, n_paths = paths.shape
        dt = self.T / (n_steps - 1)
        discount = np.exp(-self.r * dt)

        if call:
            payoff = np.maximum(paths - self.X, 0)
        else:
            payoff = np.maximum(self.X - paths, 0)

        cashflow = payoff[-1].copy() # Copy of terminal payoffs
        immediate = payoff[:-1].copy()
        continuation_est = np.full((n_steps - 1, n_paths), np.nan)
        exercise_time = np.full(n_paths, -1, dtype=int)

        for t in range(n_steps - 2, -1, -1):
            in_the_money = payoff[t] > 0  # In-the-money paths
            if np.any(in_the_money):
                # Regression on in-the-money paths
                S_t = paths[t, in_the_money]
                Y = cashflow[in_the_money] * discount # One step discounted value of continuing from t to t+1

                if basis_fn == "monomial":
                    basis = np.vander(S_t, N=3, increasing=True) # Monomial basis function of degree 2 built using Vandermonde matrix
                elif basis_fn == "laguerre":
                    x = (S_t / self.X).astype(float) # Normalise stock prices by strike price
                    basis = np.exp(-x[:, None] / 2) * np.polynomial.laguerre.lagvander(x, 3) # Laguerre polynomial basis of degree 2
                elif basis_fn == "hermite":
                    basis = np.polynomial.hermite.hermvander(S_t, 3) # Hermite polynomial basis of degree 2
                else:
                    raise ValueError(f"Unknown basis function: {basis_fn}")
                coeffs, *_ = np.linalg.lstsq(basis, Y, rcond=None)
                continuation = basis @ coeffs # Dot product of basis and coeffs
                exercise = payoff[t, in_the_money]

                continuation_est[t, in_the_money] = continuation

                # Exercise if immediate payoff > continuation value
                exercise_now = exercise > continuation
                in_indices = np.where(in_the_money)[0]
                exercise_paths = in_indices[exercise_now]
                exercise_time[exercise_paths] = t
                cashflow[in_the_money] = np.where(exercise_now, exercise, cashflow[in_the_money] * discount)
            cashflow[~in_the_money] *= discount # Discount out-of-the-money paths

        price = np.mean(cashflow)
        stderr = np.std(cashflow) / np.sqrt(n_paths) # Corrected standard error
        if not return_diagnostics:
            return price, stderr

        exercise_mask = np.zeros((n_steps - 1, n_paths), dtype=bool)
        valid = exercise_time >= 0
        exercise_mask[exercise_time[valid], np.where(valid)[0]] = True
        diagnostics = {
            "paths": paths,
            "time_grid": np.linspace(0.0, self.T, n_steps),
            "exercise_mask": exercise_mask,
            "exercise_time": exercise_time,
            "immediate_payoff": immediate,
            "continuation_estimate": continuation_est,
        }
        return price, stderr, diagnostics

    def plot_american_exercise(self, call: bool = True, basis_fn: str = "laguerre", *, antithetic: bool = True,
                               max_paths: int | None = 500, show_boundary: bool = True,
                               figsize: tuple[float, float] = (10, 6)) -> tuple[float, float]:
        """Plot simulated paths with optimal exercise decisions highlighted."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("plot_american_exercise requires matplotlib to be installed.") from exc

        price, stderr, diagnostics = self.american(call=call, basis_fn=basis_fn, antithetic=antithetic,
                                                   return_diagnostics=True)

        paths = diagnostics["paths"]
        time_grid = diagnostics["time_grid"]
        exercise_mask = diagnostics["exercise_mask"]
        n_steps, n_paths = paths.shape

        if max_paths is None or max_paths >= n_paths:
            path_indices = np.arange(n_paths)
        else:
            path_indices = np.arange(max_paths)

        plt.figure(figsize=figsize)
        for idx in path_indices:
            plt.plot(time_grid, paths[:, idx], color="gray", alpha=0.15, linewidth=0.8)

        mask_subset = exercise_mask[:, path_indices]
        if mask_subset.any():
            t_idx, p_idx = np.nonzero(mask_subset)
            plt.scatter(time_grid[:-1][t_idx], paths[t_idx, path_indices[p_idx]],
                        c="red", s=20, alpha=0.6, label="Exercise decision")

        if show_boundary:
            boundary = np.full(exercise_mask.shape[0], np.nan)
            for t in range(exercise_mask.shape[0]):
                exercised_states = paths[t, exercise_mask[t]]
                if exercised_states.size:
                    boundary[t] = np.mean(exercised_states)
            if not np.all(np.isnan(boundary)):
                plt.plot(time_grid[:-1], boundary, color="red", linewidth=2.0, label="Average exercise level")

        plt.axhline(self.X, color="orange", linestyle="--", linewidth=1.2, label="Strike")
        plt.title("American option exercise profile")
        plt.xlabel("Time")
        plt.ylabel("Underlying price")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        return price, stderr

__all__ = ['MonteCarloPricing']
