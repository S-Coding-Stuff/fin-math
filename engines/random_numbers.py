import numpy as np
from statistics import NormalDist

try:
    from scipy.stats import qmc
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    qmc = None


def _is_prime(value: int) -> bool:
    if value <= 1:
        return False
    if value <= 3:
        return True
    if value % 2 == 0 or value % 3 == 0:
        return False
    i = 5
    while i * i <= value:
        if value % i == 0 or value % (i + 2) == 0:
            return False
        i += 6
    return True


def _next_prime(lower_bound: int) -> int:
    candidate = max(2, lower_bound)
    while not _is_prime(candidate):
        candidate += 1
    return candidate


def _to_base_digits(n: int, base: int) -> list[int]:
    if n == 0:
        return [0]
    digits: list[int] = []
    value = n
    while value > 0:
        digits.append(value % base)
        value //= base
    return digits


STANDARD_NORMAL = NormalDist()
NORMAL_INV_CDF = np.vectorize(STANDARD_NORMAL.inv_cdf, otypes=[float])


def _normal_ppf(u: np.ndarray) -> np.ndarray:
    """Transform uniforms to standard normal variates using statistics.NormalDist."""
    return NORMAL_INV_CDF(u)


class FaureSequence:
    """Construct a Faure low-discrepancy sequence of a given dimension."""

    def __init__(self, dimension: int, base: int | None = None, start_index: int = 0):
        if dimension < 1:
            raise ValueError("Faure sequence dimension must be at least 1.")
        if start_index < 0:
            raise ValueError("Faure sequence start_index must be non-negative.")
        self.dimension = int(dimension)
        self.base = _next_prime(self.dimension) if base is None else int(base)
        if self.base < self.dimension:
            raise ValueError("Faure sequence requires base >= dimension.")
        if not _is_prime(self.base):
            raise ValueError("Faure sequence base must be prime.")

        self._start_index = int(start_index)
        self._index = self._start_index
        self._binom: list[list[int]] = [[1]]  # Pascal triangle mod base

    def reset(self) -> None:
        self._index = self._start_index

    def _ensure_binom(self, order: int) -> None:
        while len(self._binom) < order:
            n = len(self._binom)
            prev = self._binom[-1]
            row = [1]
            for k in range(1, n):
                row.append((prev[k - 1] + prev[k]) % self.base)
            row.append(1)
            self._binom.append(row)

    def _transform_digits(self, digits: list[int], dim_index: int) -> list[int]:
        if dim_index == 0:
            return digits.copy()

        m = len(digits)
        transformed = [0] * m
        for r in range(m):
            total = 0
            for k in range(r, m):
                coeff = (self._binom[k][r] * pow(dim_index, k - r, self.base)) % self.base
                total += digits[k] * coeff
            transformed[r] = total % self.base
        return transformed

    def next(self) -> np.ndarray:
        digits = _to_base_digits(self._index, self.base)
        self._ensure_binom(len(digits))

        point = np.empty(self.dimension, dtype=float)
        for dim_index in range(self.dimension):
            digits_transformed = self._transform_digits(digits, dim_index)
            value = 0.0
            for j, digit in enumerate(digits_transformed):
                value += digit / (self.base ** (j + 1))
            point[dim_index] = value

        self._index += 1
        return point

    def generate(self, n: int) -> np.ndarray:
        if n < 0:
            raise ValueError("Number of requested Faure points must be non-negative.")
        points = np.empty((n, self.dimension), dtype=float)
        for i in range(n):
            points[i] = self.next()
        return points


class FaureNormalGenerator:
    """Adapter that mimics np.random.Generator.standard_normal using Faure points."""

    def __init__(
        self,
        dimension: int,
        base: int | None = None,
        *,
        start_index: int = 0,
        clip_eps: float = 1e-12,
    ):
        self.sequence = FaureSequence(dimension=dimension, base=base, start_index=start_index)
        self._clip_eps = float(clip_eps)

    @property
    def base(self) -> int:
        return self.sequence.base

    @property
    def dimension(self) -> int:
        return self.sequence.dimension

    def reset(self) -> None:
        self.sequence.reset()

    def standard_normal(self, size) -> np.ndarray:
        if isinstance(size, int):
            size = (size,)
        else:
            size = tuple(size)

        if len(size) == 1:
            dims = 1
            count = size[0]
            if self.dimension != 1:
                raise ValueError("FaureNormalGenerator dimension mismatch for 1D request.")
        elif len(size) == 2:
            dims, count = size
            if dims > self.dimension:
                raise ValueError("Requested dimension exceeds Faure generator dimension.")
        else:
            raise ValueError("FaureNormalGenerator currently supports 1D or 2D size tuples.")

        uniforms = self.sequence.generate(count)
        if len(size) == 2 and dims < self.dimension:
            uniforms = uniforms[:, :dims]

        clipped = np.clip(uniforms, self._clip_eps, 1.0 - self._clip_eps)
        normals = _normal_ppf(clipped)

        if len(size) == 1:
            return normals[:, 0]

        return normals.T


class SobolNormalGenerator:
    """Sobol' sequence-based quasi-random generator compatible with standard_normal calls."""

    def __init__(
        self,
        dimension: int,
        *,
        scramble: bool = True,
        seed: int | None = None,
        clip_eps: float = 1e-12,
    ):
        if dimension < 1:
            raise ValueError("SobolNormalGenerator dimension must be at least 1.")
        if qmc is None:
            raise ImportError("SobolNormalGenerator requires scipy>=1.7 (scipy.stats.qmc).")

        self._dimension = int(dimension)
        self._sobol = qmc.Sobol(d=self._dimension, scramble=scramble, seed=seed)
        self._clip_eps = float(clip_eps)

    @property
    def dimension(self) -> int:
        return self._dimension

    def reset(self) -> None:
        self._sobol.reset()

    def standard_normal(self, size) -> np.ndarray:
        if isinstance(size, int):
            size = (size,)
        else:
            size = tuple(size)

        if len(size) == 1:
            dims = 1
            count = size[0]
        elif len(size) == 2:
            dims, count = size
            if dims > self._dimension:
                raise ValueError("Requested dimension exceeds Sobol generator dimension.")
        else:
            raise ValueError("SobolNormalGenerator currently supports 1D or 2D size tuples.")

        uniforms = self._sobol.random(count)
        if dims < self._dimension:
            uniforms = uniforms[:, :dims]

        clipped = np.clip(uniforms, self._clip_eps, 1.0 - self._clip_eps)
        normals = _normal_ppf(clipped)

        if len(size) == 1:
            return normals[:, 0]

        return normals.T


__all__ = ["FaureSequence", "FaureNormalGenerator", "SobolNormalGenerator"]
