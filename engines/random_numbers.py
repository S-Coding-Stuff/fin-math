import numpy as np
from scipy.stats import norm
from scipy.stats.qmc import Sobol


class SobolNormalGenerator:
    def __init__(self, *, dimension: int, scramble: bool = False, seed: int | None = None) -> None:
        if dimension < 1:
            raise ValueError("dimension must be >= 1.")
        self.dimension = int(dimension)
        self.scramble = bool(scramble)
        self.seed = seed
        self._engine = Sobol(d=self.dimension, scramble=self.scramble, seed=self.seed)

    def standard_normal(self, shape: tuple[int, int]) -> np.ndarray:
        if len(shape) != 2:
            raise ValueError("shape must be a 2-tuple of (dimension, num_samples).")
        dimension, num_samples = int(shape[0]), int(shape[1])
        if dimension != self.dimension:
            raise ValueError(f"Requested dimension {dimension} does not match generator dimension {self.dimension}.")
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1.")

        eps = np.finfo(float).eps
        if self.scramble:
            uniforms = self._engine.random(num_samples)
        else:
            m = int(np.ceil(np.log2(max(1, num_samples))))
            uniforms = self._engine.random_base2(m=m)[:num_samples]
        normals = norm.ppf(np.clip(uniforms, eps, 1.0 - eps))
        return normals.T


__all__ = ["SobolNormalGenerator"]
