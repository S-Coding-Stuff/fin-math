"""Monte Carlo finite-difference Greeks.

This module provides the structure for computing option Greeks by bumping model
parameters and reusing common random numbers to reduce variance. It is intended
for development into a robust implementation; methods are stubbed with clear
interfaces and documentation."""

from typing import Dict, Iterable, Optional
import numpy as np
from engines.monte_carlo import MonteCarloPricing
from greeks.random_draws import draw_common_normals, draw_independent_normals

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
                 bumps: Optional[BumpConfig] = None, relative: bool = False, style: str = "european",
                 basis_fn: str = "laguerre", include_all_paths: bool = True, mask_tolerance: float = 0.0) -> None:
        
        self.pricer = pricer
        self.call = bool(call)
        self.antithetic = bool(antithetic)
        self.risk_neutral = bool(risk_neutral)
        self.bumps = bumps if bumps is not None else BumpConfig()
        self.relative = bool(relative)
        style_norm = str(style).strip().lower()
        if style_norm not in {"european", "american"}:
            raise ValueError("style must be either 'european' or 'american'.")
        self.style = style_norm
        self.basis_fn = str(basis_fn).strip().lower()
        self.include_all_paths = bool(include_all_paths)
        self.mask_tolerance = max(0.0, float(mask_tolerance))
        self._base_bumps: Dict[str, float] = {
            "S_0": float(self.bumps.S_0),
            "sigma": float(self.bumps.sigma),
            "r": float(self.bumps.r),
            "T": float(self.bumps.T),
        }

    def _common_random_normals(self) -> np.ndarray:
        """Normals intended to be shared across bumped valuations."""
        return draw_common_normals(self.pricer, antithetic=self.antithetic)

    def _independent_random_normals(self) -> np.ndarray:
        """Normals that ignore the pricer RNG (useful for diagnostics or stress tests)."""
        steps = int(self.pricer.steps)
        paths = int(self.pricer.num_paths)
        return draw_independent_normals(steps, paths, antithetic=self.antithetic)

    def _price(self, *, S_0: Optional[float] = None, sigma: Optional[float] = None, r: Optional[float] = None,
               mu: Optional[float] = None, T: Optional[float] = None, Z: Optional[np.ndarray] = None,
               risk_neutral: Optional[bool] = None, continuation_mask: Optional[np.ndarray] = None,
               return_itm_mask: bool = False) -> tuple[float, float] | tuple[float, float, np.ndarray]:
        """Monte Carlo price (mean, stderr) for the supplied parameter overrides."""

        pricer = self.pricer
        risk_flag = self.risk_neutral if risk_neutral is None else bool(risk_neutral)

        # Checking over variables 
        overrides: Dict[str, float] = {}
        for name, value in (("S_0", S_0), ("sigma", sigma), ("r", r), ("mu", mu), ("T", T)):
            if value is not None:
                overrides[name] = float(value)

        saved = {name: getattr(pricer, name) for name in ("S_0", "sigma", "r", "mu", "T")}
        try:
            for name, value in overrides.items():
                setattr(pricer, name, value)

            if pricer.S_0 <= 0.0:
                raise ValueError("Spot S_0 must be positive.")
            if pricer.sigma <= 0.0:
                raise ValueError("Volatility sigma must be positive.")
            if pricer.T <= 0.0:
                raise ValueError("Maturity T must be positive.")

            drift = pricer.r if risk_flag else pricer.mu
            if drift is None:
                which = "r" if risk_flag else "mu"
                raise ValueError(f"{which} must be set on the pricer (or via bump) before pricing.")
            if pricer.r is None:
                raise ValueError("Risk-free rate r must be set for discounting.")

            steps = int(pricer.steps)
            paths = int(pricer.num_paths)
            if Z is None:
                Z = self._independent_random_normals()
            if Z.shape != (steps, paths):
                raise ValueError(f"Z must have shape ({steps},{paths}); received {Z.shape}.")

            paths_matrix = pricer._simulate_paths(risk_neutral=risk_flag, Z=Z, antithetic=False)
            mask_used: Optional[np.ndarray] = None
            if self.style == "american":
                mask = None
                if continuation_mask is not None:
                    mask = self._validate_continuation_mask(continuation_mask, paths_matrix)
                discounted_result = pricer.american_cashflows(
                    paths_matrix,
                    call=self.call,
                    basis_fn=self.basis_fn,
                    include_all_paths=self.include_all_paths,
                    mask_tolerance=self.mask_tolerance,
                    mask=mask,
                    return_mask=return_itm_mask,
                )
                if return_itm_mask:
                    discounted, mask_used = discounted_result
                else:
                    discounted = discounted_result
            else:
                discounted = self._discounted_european(paths_matrix)
            mean_price = float(np.mean(discounted))
            ddof = 1 if discounted.size > 1 else 0
            stderr = float(np.std(discounted, ddof=ddof) / np.sqrt(discounted.size))
            if return_itm_mask:
                if mask_used is None:
                    raise RuntimeError("ITM mask capture requested but no mask was produced.")
                return mean_price, stderr, mask_used
            return mean_price, stderr
        finally:
            for name, value in saved.items():
                setattr(pricer, name, value)

    def _discounted_european(self, paths: np.ndarray) -> np.ndarray:
        terminal = paths[-1]
        strike = self.pricer.X
        if self.call:
            payoffs = np.maximum(terminal - strike, 0.0)
        else:
            payoffs = np.maximum(strike - terminal, 0.0)
        discount = np.exp(-self.pricer.r * self.pricer.T)
        return discount * payoffs

    def _validate_continuation_mask(self, mask: np.ndarray, paths: np.ndarray) -> np.ndarray:
        expected = (paths.shape[0] - 1, paths.shape[1])
        if mask.shape != expected:
            raise ValueError(f"Continuation mask must have shape {expected}, received {mask.shape}.")
        return mask.astype(bool, copy=False)

    def price(self) -> tuple[float, float]:
        """Baseline MC price and standard error with current settings."""
        return self._price()

    def greek(self, kind: str, scheme: str = "central", *, bumps: Optional[Dict[str, float]] = None,
        relative: Optional[bool] = None) -> float:
        """Compute a single finite-difference Greek (delta, gamma, vega, theta, rho)."""
        kind_l = kind.lower()
        greek_to_param = {"delta": "S_0", "gamma": "S_0",
                          "vega": "sigma",
                          "theta": "T",
                          "rho": "r"
                          }
        if kind_l not in greek_to_param:
            raise ValueError(f"Unsupported greek '{kind}'.")

        param_name = greek_to_param[kind_l]
        pricer = self.pricer
        base_value = getattr(pricer, param_name)
        if base_value is None:
            raise ValueError(f"Parameter '{param_name}' is not set on the pricing engine.")

        bump_values = dict(self._base_bumps)
        if bumps:
            for key, value in bumps.items():
                if key not in bump_values:
                    raise ValueError(f"Unknown bump key '{key}'. Expected one of {list(bump_values)}.")
                bump_values[key] = float(value)

        bump_value = float(bump_values[param_name])
        rel_flag = self.relative if relative is None else bool(relative)
        if rel_flag:
            bump_value = float(base_value) * bump_value
        if bump_value == 0.0:
            raise ValueError("Bump size must be non-zero.")

        positive_params = {"S_0", "sigma", "T"}

        def ensure_positive(name: str, value: float) -> None:
            if name in positive_params and value <= 0.0:
                raise ValueError(f"Bump results in non-positive {name}. Reduce bump or choose forward/backward scheme.")

        risk_flag = self.risk_neutral
        Z = self._common_random_normals()
        base_itm_flags: Optional[np.ndarray] = None
        base_price_cache: Optional[float] = None
        if self.style == "american":
            base_result = self._price(Z=Z, risk_neutral=risk_flag, return_itm_mask=True)
            base_price_cache, _, base_itm_flags = base_result
            if base_itm_flags is None:
                raise RuntimeError("Failed to capture ITM flags from the base American run.")

        def priced_at(value: float) -> float:
            ensure_positive(param_name, value)
            kwargs = {param_name: value}
            mask = base_itm_flags if self.style == "american" else None
            price_result = self._price(
                Z=Z,
                risk_neutral=risk_flag,
                continuation_mask=mask,
                **kwargs,
            )
            return price_result[0]

        def base_price_value() -> float:
            nonlocal base_price_cache
            if base_price_cache is None:
                base_price_cache = priced_at(base_value)
            return base_price_cache

        scheme_l = scheme.lower()
        h = bump_value

        if kind_l == "gamma":
            if scheme_l != "central":
                raise ValueError("Gamma currently supports only the central difference scheme.")
            up = priced_at(base_value + h)
            mid = base_price_value()
            down = priced_at(base_value - h)
            return (up - 2.0 * mid + down) / (h ** 2)

        # Forward, Backward and Central Difference for Greeks
        if scheme_l == "central":
            up = priced_at(base_value + h)
            down = priced_at(base_value - h)
            return (up - down) / (2.0 * h)
        if scheme_l == "forward":
            up = priced_at(base_value + h)
            mid = base_price_value()
            return (up - mid) / h
        if scheme_l == "backward":
            mid = base_price_value()
            down = priced_at(base_value - h)
            return (mid - down) / h
        raise ValueError("scheme must be one of {'central','forward','backward'}.")

    def greeks(self, kinds: Optional[Iterable[str]] = None, scheme: str = "central", *, 
               bumps: Optional[Dict[str, float]] = None, relative: Optional[bool] = None) -> Dict[str, float]:
        """Compute multiple Greeks and return a mapping of kind -> value.

        kinds defaults to ['delta','gamma','vega','theta','rho']."""
        raise NotImplementedError

    # Thin convenience wrappers (to be wired to greek())
    def delta(self) -> float:  # dV/dS
        return self.greek("delta")

    def gamma(self) -> float:  # d^2 V/dS^2
        return self.greek("gamma")

    def vega(self) -> float:  # dV/dsigma
        return self.greek("vega")

    def theta(self) -> float:  # dV/dT
        return self.greek("theta")

    def rho(self) -> float:  # dV/dr
        return self.greek("rho")


__all__ = ["BumpConfig", "MonteCarloFiniteDifference"]
