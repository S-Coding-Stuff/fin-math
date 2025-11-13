"""Monte Carlo finite-difference Greeks.

This module provides the structure for computing option Greeks by bumping model
parameters and reusing common random numbers to reduce variance. It is intended
for development into a robust implementation; methods are stubbed with clear
interfaces and documentation."""

from __future__ import annotations
from typing import Dict, Iterable, Optional

import numpy as np

from engines.monte_carlo import MonteCarloPricing

class BumpConfig:
    """Configuration for finite-difference bump sizes.

    Use absolute bump sizes; a higher-level wrapper may add relative scaling."""
    def __init__(self, S_0: float = 1e-2, sigma: float = 1e-3, r: float = 1e-4, T: float = 1e-4) -> None:
        self.S_0: float = S_0
        self.sigma: float = sigma
        self.r: float = r
        self.T: float = T


class MonteCarloFiniteDifference:
    """Finite-difference Greeks computed from a Monte Carlo pricing engine.

    Parameters
    - pricer: an instance of engines.monte_carlo.MonteCarloPricing (or subclass)
    - call: option type flag (True for call, False for put)
    - antithetic: use antithetic variates in simulations
    - risk_neutral: simulate under risk-neutral (uses r) or real-world (uses mu)
    - bumps: absolute bump sizes per parameter
    - relative: if True, bumps may be interpreted as relative in the final implementation."""

    def __init__(self, pricer: MonteCarloPricing, *, call: bool = True, antithetic: bool = True, risk_neutral: bool = True,
                 bumps: Optional[BumpConfig] = None, relative: bool = False) -> None:
        
        self.pricer = pricer
        self.call = bool(call)
        self.antithetic = bool(antithetic)
        self.risk_neutral = bool(risk_neutral)
        self.bumps = bumps if bumps is not None else BumpConfig()
        self.relative = bool(relative)

    def _common_random_normals(self) -> np.ndarray:
        """Draw and return normal variates to be reused across all bumps.

        Implementation detail: In a variance-reduced FD, all bumped prices should
        use the exact same random shocks. This method handles antithetic
        pairing."""
        steps = int(self.pricer.steps)
        paths = int(self.pricer.num_paths)

        # Prefer a QMC-aware draw if the pricer provides one (e.g., QuasiMonteCarloPricing)
        draw_fn = getattr(self.pricer, "_draw_normals", None)
        if callable(draw_fn):
            count = (paths + 1) // 2 if self.antithetic else paths
            normals = draw_fn(count)  # expected shape: (steps, count)
            if normals.shape != (steps, count):
                # Safety: transpose common alternative shape
                if normals.shape == (count, steps):
                    normals = normals.T
                else:
                    raise ValueError(
                        f"_draw_normals returned shape {normals.shape}, expected ({steps}, {count})."
                    )
            if self.antithetic:
                normals = np.concatenate((normals, -normals), axis=1)[:, :paths]
            return normals

        # Fallback to pseudo-random normals from the pricer's RNG, preserving RNG state
        rng = self.pricer.rng
        state = rng.bit_generator.state
        try:
            if self.antithetic:
                half = (paths + 1) // 2
                z_half = rng.standard_normal(size=(steps, half))
                Z = np.concatenate((z_half, -z_half), axis=1)[:, :paths]
            else:
                Z = rng.standard_normal(size=(steps, paths))
        finally:
            # Restore RNG state so this draw doesn't perturb subsequent randomness elsewhere
            rng.bit_generator.state = state
        return Z

    def _price(
        self,
        *,
        S_0: Optional[float] = None,
        sigma: Optional[float] = None,
        r: Optional[float] = None,
        mu: Optional[float] = None,
        Z: Optional[np.ndarray] = None,
        risk_neutral: Optional[bool] = None,
    ) -> tuple[float, float]:
        """Return MC price (mean, stderr) for the given parameters.

        The default implementation should delegate to the underlying pricer,
        passing in the provided bumps and random numbers. To be implemented."""
        raise NotImplementedError

    def price(self) -> tuple[float, float]:
        """Baseline MC price and standard error with current settings."""
        return self._price()

    def greek(self, kind: str, scheme: str = "central", *, bumps: Optional[Dict[str, float]] = None,
              relative: Optional[bool] = None) -> float:
        """Compute a single Greek by finite differences.

        kind: one of {'delta','gamma','vega','theta','rho'}; extensible later.
        scheme: 'central' (default), 'forward', or 'backward' depending on bounds.
        bumps: optional per-parameter bump size overrides.
        relative: override for interpreting bump sizes as relative."""
        raise NotImplementedError

    def greeks(self, kinds: Optional[Iterable[str]] = None, scheme: str = "central", *, 
               bumps: Optional[Dict[str, float]] = None, relative: Optional[bool] = None) -> Dict[str, float]:
        """Compute multiple Greeks and return a mapping of kind -> value.

        kinds defaults to ['delta','gamma','vega','theta','rho']."""
        raise NotImplementedError

    # Thin convenience wrappers (to be wired to greek())
    def delta(self) -> float:  # dV/dS
        raise NotImplementedError

    def gamma(self) -> float:  # d^2 V/dS^2
        raise NotImplementedError

    def vega(self) -> float:  # dV/dsigma
        raise NotImplementedError

    def theta(self) -> float:  # dV/dT
        raise NotImplementedError

    def rho(self) -> float:  # dV/dr
        raise NotImplementedError


__all__ = ["BumpConfig", "MonteCarloFiniteDifference"]
