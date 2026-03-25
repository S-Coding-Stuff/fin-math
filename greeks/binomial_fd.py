"""Finite-difference Greeks using the binomial pricing model."""
from dataclasses import dataclass
from models.binomial_method import BinomialPricing
@dataclass
class BinomialBumps:
    """Absolute bump sizes for finite-difference estimates."""

    S_0: float = 1e-2
    sigma: float = 1e-4
    r: float = 1e-4
    T: float = 1e-4


class BinomialFiniteDifference:
    """Compute Greeks for American options via binomial bump-and-reprice."""

    def __init__(self, S_0: float, K: float, r: float, sigma: float, T: float, 
                 steps: int = 200, *, call: bool = True, 
                 bumps: BinomialBumps | None = None) -> None:
        self.S_0 = float(S_0)
        self.K = float(K)
        self.r = float(r)
        self.sigma = float(sigma)
        self.T = float(T)
        self.steps = int(steps)
        self.call = bool(call)
        self.bumps = bumps if bumps is not None else BinomialBumps()
        self._price_cache: dict[tuple[float, float, float, float], float] = {}

    def _price(self, *, S_0: float | None = None, r: float | None = None, sigma: float | None = None,
               T: float | None = None) -> float:
        s0_eff = float(S_0 if S_0 is not None else self.S_0)
        r_eff = float(r if r is not None else self.r)
        sigma_eff = float(sigma if sigma is not None else self.sigma)
        t_eff = float(T if T is not None else self.T)
        key = (s0_eff, r_eff, sigma_eff, t_eff)
        cached = self._price_cache.get(key)
        if cached is not None:
            return cached
        pricer = BinomialPricing(
            S_0=s0_eff,
            K=self.K,
            r=r_eff,
            sigma=sigma_eff,
            T=t_eff,
            steps=self.steps,
        )
        value = float(pricer.american(call=self.call))
        self._price_cache[key] = value
        return value

    def price(self) -> float:
        return self._price()

    def delta(self) -> float:
        h = float(self.bumps.S_0)
        S0 = self.S_0
        if h == 0.0:
            raise ValueError("S_0 bump must be non-zero.")
        if S0 - h <= 0.0:
            up = self._price(S_0=S0 + h)
            mid = self._price(S_0=S0)
            return (up - mid) / h # forward difference
        up = self._price(S_0=S0 + h)
        dn = self._price(S_0=S0 - h)
        return (up - dn) / (2.0 * h) # central difference

    def gamma(self) -> float:
        h = float(self.bumps.S_0)
        S0 = self.S_0
        if h == 0.0:
            raise ValueError("S_0 bump must be non-zero.")
        if S0 - h <= 0.0:
            f0 = self._price(S_0=S0)
            f1 = self._price(S_0=S0 + h)
            f2 = self._price(S_0=S0 + 2.0 * h)
            return (f2 - 2.0 * f1 + f0) / (h**2) # forward difference
        up = self._price(S_0=S0 + h)
        mid = self._price(S_0=S0)
        dn = self._price(S_0=S0 - h)
        return (up - 2.0 * mid + dn) / (h**2) # central difference
    
    def vega(self) -> float:
        h = float(self.bumps.sigma)
        sig = self.sigma
        if h == 0.0:
            raise ValueError("sigma bump must be non-zero.")
        if sig - h <= 0.0:
            up = self._price(sigma=sig + h)
            mid = self._price(sigma=sig)
            return (up - mid) / h
        up = self._price(sigma=sig + h)
        dn = self._price(sigma=sig - h)
        return (up - dn) / (2.0 * h)

    def theta(self) -> float:
        h = float(self.bumps.T)
        T = self.T
        if h == 0.0:
            raise ValueError("T bump must be non-zero.")
        if T - h <= 0.0:
            up = self._price(T=T + h)
            mid = self._price(T=T)
            return (up - mid) / h
        up = self._price(T=T + h)
        dn = self._price(T=T - h)
        return (up - dn) / (2.0 * h)

    def rho(self) -> float:
        h = float(self.bumps.r)
        r = self.r
        if h == 0.0:
            raise ValueError("r bump must be non-zero.")
        up = self._price(r=r + h)
        dn = self._price(r=r - h)
        return (up - dn) / (2.0 * h)


__all__ = ["BinomialBumps", "BinomialFiniteDifference"]
