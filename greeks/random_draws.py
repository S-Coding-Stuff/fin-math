"""Utility helpers for sampling Brownian increments used by multiple Greek estimators."""
import numpy as np
from engines.monte_carlo import MonteCarloPricing

def _maybe_antithetic(normals: np.ndarray, paths: int, *, antithetic: bool) -> np.ndarray:
    if not antithetic:
        return normals
    expanded = np.concatenate((normals, -normals), axis=1)
    return expanded[:, :paths]

def draw_common_normals(pricer: MonteCarloPricing, *, antithetic: bool) -> np.ndarray:
    """Return Brownian shocks sourced from the pricer (QMC-aware if available)."""
    steps = int(pricer.steps)
    paths = int(pricer.num_paths)
    count = (paths + 1) // 2 if antithetic else paths

    draw_fn = getattr(pricer, "_draw_normals", None)
    if callable(draw_fn):
        normals = draw_fn(count)
    else:
        normals = pricer.rng.standard_normal(size=(steps, count))

    if normals.shape != (steps, count):
        if normals.shape == (count, steps):
            normals = normals.T
        else:
            raise ValueError(
                f"Expected normals of shape ({steps},{count}), received {normals.shape}."
            )
    return _maybe_antithetic(normals, paths, antithetic=antithetic)

def draw_independent_normals(steps: int, paths: int, *, antithetic: bool, seed: int | None = None) -> np.ndarray:
    """Return Brownian shocks from an independent RNG (useful for diagnostics)."""
    count = (paths + 1) // 2 if antithetic else paths
    rng = np.random.default_rng(seed)
    normals = rng.standard_normal(size=(steps, count))
    return _maybe_antithetic(normals, paths, antithetic=antithetic)

__all__ = ["draw_common_normals", "draw_independent_normals"]
