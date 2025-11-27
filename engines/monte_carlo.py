from __future__ import annotations

from typing import Literal, Optional

import numpy as np

BasisName = Literal["laguerre", "monomial", "hermite"]

class LSMConfig:
    def __init__(self, *, basis: BasisName = "laguerre", include_all_paths: bool = True,
                 mask_tolerance: float = 0.0) -> None:
        self.basis = basis
        self.include_all_paths = bool(include_all_paths)
        self.mask_tolerance = float(mask_tolerance)


def immediate_payoff(paths: np.ndarray, *, strike: float, call: bool) -> np.ndarray:
    if call:
        return np.maximum(paths - strike, 0.0)
    return np.maximum(strike - paths, 0.0)


def build_mask(paths: np.ndarray, *, strike: float, call: bool, include_all: bool, tolerance: float) -> np.ndarray:
    if include_all:
        return np.ones((paths.shape[0] - 1, paths.shape[1]), dtype=bool)

    payoff = immediate_payoff(paths, strike=strike, call=call)
    mask = payoff[:-1] > 0.0
    if tolerance > 0.0:
        states = paths[:-1]
        mask |= np.abs(states - strike) <= tolerance
    return mask


def lsm_basis(states: np.ndarray, *, strike: float, basis: BasisName) -> np.ndarray:
    flattened = np.ravel(states)
    if flattened.size == 0:
        return np.zeros((0, 3))
    if basis == "monomial":
        return np.vander(flattened, N=3, increasing=True)
    if basis == "hermite":
        return np.polynomial.hermite.hermvander(flattened, 3)
    if basis != "laguerre":
        raise ValueError(f"Unsupported basis '{basis}'.")
    normalized = (flattened / strike).astype(float, copy=False)
    return np.exp(-normalized[:, None] / 2.0) * np.polynomial.laguerre.lagvander(normalized, 3)


def _lsm_cashflows(paths: np.ndarray, *, strike: float, call: bool, rate: float, maturity: float,
                   config: LSMConfig, mask: Optional[np.ndarray] = None, capture_mask: bool = False,
                   capture_diagnostics: bool = False) -> tuple[np.ndarray, Optional[np.ndarray], Optional[dict[str, np.ndarray]]]:
    n_steps, n_paths = paths.shape
    if n_steps < 2:
        raise ValueError("Need at least one time step for American valuation.")
    if maturity <= 0.0:
        raise ValueError("Maturity must be positive for American valuation.")

    dt = maturity / (n_steps - 1)
    discount = np.exp(-rate * dt)

    payoff = immediate_payoff(paths, strike=strike, call=call)
    cashflow = payoff[-1].copy()
    expected = (n_steps - 1, n_paths)
    if mask is not None:
        mask_to_use = np.asarray(mask, dtype=bool)
        if mask_to_use.shape != expected:
            raise ValueError(f"Continuation mask must have shape {expected}, received {mask_to_use.shape}.")
    else:
        mask_to_use = build_mask(paths, strike=strike, call=call, 
                                 include_all=config.include_all_paths,
                                 tolerance=config.mask_tolerance)

    continuation_est = None
    exercise_time = None
    immediate_payoff_slice = None
    if capture_diagnostics:
        continuation_est = np.full((n_steps - 1, n_paths), np.nan)
        exercise_time = np.full(n_paths, -1, dtype=int)
        immediate_payoff_slice = payoff[:-1].copy()

    for t in range(n_steps - 2, -1, -1):
        include = mask_to_use[t]
        if np.any(include):
            states = paths[t, include]
            continuation_targets = cashflow[include] * discount
            basis = lsm_basis(states, strike=strike, basis=config.basis)
            coeffs, *_ = np.linalg.lstsq(basis, continuation_targets, rcond=None)
            continuation = basis @ coeffs
            exercise = payoff[t, include]
            # Only allow exercise when the payoff is strictly positive
            # to avoid spurious "exercise" decisions for out-of-the-money states.
            exercise_now = (exercise > 0.0) & (exercise > continuation)
            idx = np.where(include)[0]
            cashflow[idx] = np.where(exercise_now, exercise, cashflow[idx] * discount)

            if capture_diagnostics:
                continuation_est[t, include] = continuation
                exercise_time[idx[exercise_now]] = t
        cashflow[~include] *= discount

    diagnostics = None
    if capture_diagnostics:
        assert continuation_est is not None and exercise_time is not None and immediate_payoff_slice is not None
        exercise_mask = np.zeros((n_steps - 1, n_paths), dtype=bool)
        valid = exercise_time >= 0
        exercise_mask[exercise_time[valid], np.where(valid)[0]] = True
        diagnostics = {
            "paths": paths,
            "time_grid": np.linspace(0.0, maturity, n_steps),
            "exercise_mask": exercise_mask,
            "exercise_time": exercise_time,
            "immediate_payoff": immediate_payoff_slice,
            "continuation_estimate": continuation_est,
        }

    mask_return = mask_to_use.copy() if capture_mask else None
    return cashflow, mask_return, diagnostics

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

        # If a seed is supplied (and no custom rng), remember it so each pricing call can
        # reset to the same random stream for deterministic results.
        self._seed = None if rng is not None else seed
        self.rng = rng if rng is not None else np.random.default_rng(seed)

    def _reset_rng(self) -> None:
        """Recreate the RNG from the stored seed when one was provided."""
        if self._seed is not None:
            self.rng = np.random.default_rng(self._seed)

    def _simulate_paths(self, risk_neutral: bool = True, Z: np.ndarray | None = None, 
                        *, antithetic: bool = False) -> np.ndarray:
        """Simulate stock prices over time using Geometric Brownian Motion."""
        num_paths = self.num_paths
        num_steps = self.steps
        dt = self.T / self.steps # Step size

        if Z is None:
            self._reset_rng()
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
                 include_all_paths: bool = True, mask_tolerance: float = 0.0,
                 continuation_mask: Optional[np.ndarray] = None, paths: Optional[np.ndarray] = None,
                 return_diagnostics: bool = False, return_mask: bool = False) -> tuple:
        """Price an American option using the Least Squares Monte Carlo method.

        Parameters beyond the classic signature provide hooks for advanced workflows:
        - include_all_paths/mask_tolerance control the regression mask construction.
        - continuation_mask allows supplying a precomputed ITM mask (e.g., from an unbumped run).
        - paths may be pre-simulated using shared random draws.
        - return_mask surfaces the regression mask alongside the diagnostics if requested.
        """

        if self.r is None:
            raise ValueError("Risk-free rate r must be set before pricing under the risk-neutral measure.")

        if paths is None:
            paths = self._simulate_paths(antithetic=antithetic)
        n_steps, n_paths = paths.shape

        config = LSMConfig(basis=basis_fn, include_all_paths=include_all_paths, mask_tolerance=mask_tolerance)
        capture_diag = return_diagnostics
        capture_mask = return_mask
        cashflow, mask_out, diagnostics = _lsm_cashflows(paths, strike=self.X, call=call, rate=self.r, maturity=self.T,
                                                         config=config, mask=continuation_mask, capture_mask=capture_mask,
                                                         capture_diagnostics=capture_diag)

        price = float(np.mean(cashflow))
        ddof = 1 if n_paths > 1 else 0
        stderr = float(np.std(cashflow, ddof=ddof) / np.sqrt(n_paths))
        result: list = [price, stderr]
        if return_diagnostics:
            result.append(diagnostics if diagnostics is not None else {})
        if return_mask:
            if mask_out is None:
                mask_out = build_mask(paths, strike=self.X, call=call, include_all=include_all_paths,
                                      tolerance=mask_tolerance)
            result.append(mask_out)
        return tuple(result)

    def american_cashflows(self, paths: np.ndarray, *, call: bool = True, basis_fn: str = "laguerre",
                            include_all_paths: bool = True, mask_tolerance: float = 0.0,
                            mask: Optional[np.ndarray] = None, return_mask: bool = False,
                            return_diagnostics: bool = False):
        """Direct access to discounted American cashflows for custom workflows."""

        if self.r is None:
            raise ValueError("Risk-free rate r must be set before running American LSM cashflows.")

        config = LSMConfig(basis=basis_fn, include_all_paths=include_all_paths, mask_tolerance=mask_tolerance)
        capture_mask = return_mask
        capture_diag = return_diagnostics
        cashflow, mask_out, diagnostics = _lsm_cashflows(paths, strike=self.X, call=call, rate=self.r, maturity=self.T,
                                                         config=config, mask=mask, capture_mask=capture_mask,
                                                         capture_diagnostics=capture_diag)
        outputs: list = [cashflow]
        if return_diagnostics:
            outputs.append(diagnostics if diagnostics is not None else {})
        if return_mask:
            if mask_out is None:
                mask_out = build_mask(paths, strike=self.X, call=call, include_all=include_all_paths,
                                      tolerance=mask_tolerance)
            outputs.append(mask_out)
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def plot_american_exercise(self, call: bool = True, basis_fn: str = "laguerre", *, antithetic: bool = True,
                               max_paths: int | None = 500, show_boundary: bool = True,
                               figsize: tuple[float, float] = (10, 6)):
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
