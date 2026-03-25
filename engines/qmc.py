"""Quasi-Monte Carlo path generation for existing pricers."""
import math
import numpy as np
from scipy.stats import norm
from scipy.stats.qmc import Sobol
from engines.monte_carlo import MonteCarloPricing

def _to_normals(uniforms: np.ndarray) -> np.ndarray:
    eps = np.finfo(float).eps
    return norm.ppf(np.clip(uniforms, eps, 1.0 - eps))


def _radical_inverse(index: int, *, base: int = 2) -> float:
    result = 0.0
    factor = 1.0 / base
    i = int(index)
    while i > 0:
        result += (i % base) * factor
        i //= base
        factor /= base
    return result


def _van_der_corput(n: int, *, base: int = 2, start_index: int = 0) -> np.ndarray:
    seq = np.empty(n, dtype=float)
    for k in range(n):
        seq[k] = _radical_inverse(start_index + k, base=base)
    return seq


def _apply_brownian_bridge(normals: np.ndarray, *, maturity: float) -> np.ndarray:
    """Apply a Brownian bridge reordering to normal increments."""
    steps, num_paths = normals.shape
    if steps <= 1:
        return normals

    dt = maturity / steps
    times = np.linspace(0.0, maturity, steps + 1)
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
        path[-1] = np.sqrt(maturity) * normals[0, path_idx]
        path_flags = np.zeros(steps + 1, dtype=bool)
        path_flags[0] = True
        path_flags[-1] = True
        fill(path, 0, steps, cursor=1)
        bridged[:, path_idx] = np.diff(path) / np.sqrt(dt)

    return bridged


def apply_path_construction(normals: np.ndarray, *, construction: str = "seq",
                            maturity: float) -> np.ndarray:
    """Apply a Brownian path construction to normal increments.

    Supported constructions:
    - "seq": sequential increments (identity transform).
    - "bb": Brownian bridge construction.
    """
    mode = construction.lower()
    if mode == "seq":
        return normals
    if mode == "bb":
        return _apply_brownian_bridge(normals, maturity=maturity)
    raise ValueError("construction must be one of: 'seq', 'bb'.")


def generate_qmc_normals(*, method: str, num_paths: int, steps: int, scramble: bool = False,
                         seed: int | None = None, antithetic: bool = False) -> np.ndarray:
    """Generate (steps, num_paths) normal draws via QMC methods.

    Supported methods: "vdc", "sobol", "scrambled_sobol".
    """
    method = method.lower()
    if steps < 1:
        raise ValueError("steps must be >= 1.")
    if num_paths < 1:
        raise ValueError("num_paths must be >= 1.")

    count = (num_paths + 1) // 2 if antithetic else num_paths

    if method == "vdc":
        start = 0 if seed is None else int(seed) * max(steps * count, 1)
        uniforms_flat = _van_der_corput(steps * count, base=2, start_index=start)
        if scramble:
            rng = np.random.default_rng(seed)
            shift = rng.random()
            uniforms_flat = (uniforms_flat + shift) % 1.0
        uniforms = uniforms_flat.reshape(steps, count)
    elif method in {"sobol", "scrambled_sobol"}:
        use_scramble = bool(scramble or method == "scrambled_sobol")
        engine = Sobol(d=steps, scramble=use_scramble, seed=seed)
        # Sobol balance properties are strongest at powers of two; draw and trim.
        m = int(math.ceil(math.log2(max(1, count))))
        uniforms = engine.random_base2(m=m)[:count].T
    else:
        raise ValueError("method must be one of: 'vdc', 'sobol', 'scrambled_sobol'.")

    normals = _to_normals(uniforms)
    if antithetic:
        normals = np.concatenate((normals, -normals), axis=1)[:, :num_paths]
    return normals


def generate_qmc_paths(*, pricer: MonteCarloPricing, method: str = "sobol",
                       scramble: bool = False, seed: int | None = None,
                       brownian_bridge: bool = False, construction: str | None = None,
                       antithetic: bool = False,
                       risk_neutral: bool = True) -> np.ndarray:
    """Generate GBM paths using QMC normals for an existing pricer."""
    normals = generate_qmc_normals(method=method, num_paths=pricer.num_paths,
                                   steps=pricer.steps, scramble=scramble,
                                   seed=seed, antithetic=antithetic)
    if construction is not None:
        normals = apply_path_construction(normals, construction=construction, maturity=pricer.T)
    elif brownian_bridge:
        normals = _apply_brownian_bridge(normals, maturity=pricer.T)
    return pricer._simulate_paths(risk_neutral=risk_neutral, Z=normals, antithetic=False)


class QuasiMonteCarloPricing(MonteCarloPricing):
    """Monte Carlo pricer that drives the base class with QMC normals."""

    def __init__(self, S_0: float, X: float, sigma: float, T: float, *,
                 r: float | None = None, mu: float | None = None,
                 num_paths: int = 1024, steps: int = 252,
                 method: str = "sobol", scramble: bool = False,
                 seed: int | None = None, brownian_bridge: bool = False) -> None:

        super().__init__(S_0=S_0, X=X, sigma=sigma, T=T, r=r, mu=mu,
                         num_paths=num_paths, steps=steps,
                         rng=np.random.default_rng(seed), seed=seed)
        self._method = method
        self._scramble = scramble
        self._seed = seed
        self._brownian_bridge = brownian_bridge

    def _draw_normals(self, count: int) -> np.ndarray:
        """Return deterministic QMC normals for reuse in CRN-based Greek estimators."""
        normals = generate_qmc_normals(
            method=self._method,
            num_paths=int(count),
            steps=self.steps,
            scramble=self._scramble,
            seed=self._seed,
            antithetic=False,
        )
        if self._brownian_bridge:
            normals = _apply_brownian_bridge(normals, maturity=self.T)
        return normals

    def _simulate_paths(self, risk_neutral: bool = True, Z: np.ndarray | None = None, *,
                        antithetic: bool = False) -> np.ndarray:
        if Z is None:
            if antithetic:
                count = (self.num_paths + 1) // 2
                base = self._draw_normals(count)
                Z = np.concatenate((base, -base), axis=1)[:, :self.num_paths]
            else:
                Z = self._draw_normals(self.num_paths)
        return super()._simulate_paths(risk_neutral=risk_neutral, Z=Z, antithetic=False)


__all__ = [
    "generate_qmc_normals",
    "apply_path_construction",
    "generate_qmc_paths",
    "QuasiMonteCarloPricing",
]
