import numpy as np
from scipy.stats import norm
from scipy.stats.qmc import Sobol

from engines.monte_carlo import MonteCarloPricing


class QuasiMonteCarloPricing(MonteCarloPricing):
    """Monte Carlo pricer that drives the base class with Sobol (optionally bridged) normals."""

    def __init__(self, S_0: float, X: float, sigma: float, T: float, *, r: float | None = None,
                 mu: float | None = None, num_paths: int = 1024, steps: int = 252,
                 scramble: bool = True, seed: int | None = None,
                 brownian_bridge: bool = False) -> None:
        super().__init__(S_0=S_0, X=X, sigma=sigma, T=T, r=r, mu=mu,
                         num_paths=num_paths, steps=steps,
                         rng=np.random.default_rng(seed), seed=seed)
        self._sobol = Sobol(d=self.steps, scramble=scramble, seed=seed)
        self._use_bridge = brownian_bridge

    def reset_sequence(self) -> None:
        self._sobol.reset()

    def _draw_normals(self, num_paths: int) -> np.ndarray:
        uniforms = self._sobol.random(n=num_paths)
        eps = np.finfo(float).eps
        normals = norm.ppf(np.clip(uniforms, eps, 1.0 - eps)).T
        if self._use_bridge and self.steps > 1:
            normals = self._apply_brownian_bridge(normals)
        return normals

    def _apply_brownian_bridge(self, normals: np.ndarray) -> np.ndarray:
        steps, num_paths = normals.shape
        dt = self.T / steps
        times = np.linspace(0.0, self.T, steps + 1)
        bridged = np.empty_like(normals)

        def fill(path: np.ndarray, left: int, right: int, cursor: int) -> int:
            if right - left <= 1 or cursor >= steps:
                return cursor
            mid = (left + right) // 2
            if path_flags[mid]:
                cursor = fill(path, left, mid, cursor)
                return fill(path, mid, right, cursor)

            t_left, t_mid, t_right = times[left], times[mid], times[right]
            weight_left = (t_right - t_mid) / (t_right - t_left)
            weight_right = (t_mid - t_left) / (t_right - t_left)
            variance = (t_mid - t_left) * (t_right - t_mid) / (t_right - t_left)

            path[mid] = weight_left * path[left] + weight_right * path[right]
            if variance > 0.0:
                path[mid] += np.sqrt(variance) * normals[cursor, path_idx]
            path_flags[mid] = True
            cursor += 1

            cursor = fill(path, left, mid, cursor)
            return fill(path, mid, right, cursor)

        for path_idx in range(num_paths):
            path = np.zeros(steps + 1, dtype=float)
            path[0] = 0.0
            path[-1] = np.sqrt(self.T) * normals[0, path_idx]
            path_flags = np.zeros(steps + 1, dtype=bool)
            path_flags[0] = True
            path_flags[-1] = True
            fill(path, 0, steps, cursor=1)
            bridged[:, path_idx] = np.diff(path) / np.sqrt(dt)

        return bridged

    def _simulate_paths(self, risk_neutral: bool = True, Z: np.ndarray | None = None, *,
                        antithetic: bool = False) -> np.ndarray:
        if Z is None:
            count = (self.num_paths + 1) // 2 if antithetic else self.num_paths
            normals = self._draw_normals(count)
            if antithetic:
                normals = np.concatenate((normals, -normals), axis=1)[:, : self.num_paths]
            Z = normals
        return super()._simulate_paths(risk_neutral=risk_neutral, Z=Z, antithetic=False)
