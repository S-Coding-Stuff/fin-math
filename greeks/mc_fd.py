from dataclasses import dataclass
import time
from typing import Callable, Dict, Iterable, Optional

import numpy as np
from engines.monte_carlo import MonteCarloPricing
from greeks.random_draws import draw_common_normals, draw_independent_normals


class BumpConfig:
    """Configuration for finite-difference bump sizes."""

    def __init__(self, S_0: float = 1e-2, sigma: float = 1e-3, r: float = 1e-4, T: float = 1e-4) -> None:
        self.S_0: float = S_0
        self.sigma: float = sigma
        self.r: float = r
        self.T: float = T


@dataclass
class _GreekEvalContext:
    risk_flag: bool
    Z: np.ndarray
    freeze_flag: bool
    frozen_policy: object | None = None
    continuation_mask: Optional[np.ndarray] = None
    base_price: Optional[float] = None


class MonteCarloFiniteDifference:
    """Finite-difference Greeks computed from a Monte Carlo pricing engine."""

    def __init__(
        self,
        pricer: MonteCarloPricing,
        *,
        call: bool = True,
        antithetic: bool = True,
        risk_neutral: bool = True,
        bumps: Optional[BumpConfig] = None,
        relative: bool = False,
        style: str = "european",
        basis_fn: str = "laguerre",
        include_all_paths: bool = True,
        mask_tolerance: float = 0.0,
        nn_kwargs: Optional[dict] = None,
        rlsm_kwargs: Optional[dict] = None,
        train_eval_split: float | None = None,
        freeze_policy: Optional[bool] = None,
        gamma_method: str = "fd",
        gamma_regression_epsilon: float = 5.0,
        gamma_regression_degree: int = 9,
        gamma_regression_samples: int = 128,
    ) -> None:
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
        self.nn_kwargs = {} if nn_kwargs is None else dict(nn_kwargs)
        self.rlsm_kwargs = {} if rlsm_kwargs is None else dict(rlsm_kwargs)
        self.train_eval_split = train_eval_split
        if self.train_eval_split is not None:
            split = float(self.train_eval_split)
            if not (0.0 < split < 1.0):
                raise ValueError("train_eval_split must be strictly between 0 and 1.")
        if freeze_policy is None:
            self.freeze_policy = self.style == "american" and self.basis_fn == "rlsm"
        else:
            self.freeze_policy = bool(freeze_policy)

        gamma_method_l = str(gamma_method).strip().lower()
        if gamma_method_l == "fd" and self.style == "american" and self.basis_fn == "rlsm":
            gamma_method_l = "pde"
        if gamma_method_l not in {"fd", "pde", "regression"}:
            raise ValueError("gamma_method must be one of {'fd','pde','regression'}.")
        self.gamma_method = gamma_method_l
        self.gamma_regression_epsilon = float(gamma_regression_epsilon)
        self.gamma_regression_degree = int(gamma_regression_degree)
        self.gamma_regression_samples = int(gamma_regression_samples)
        if self.gamma_regression_epsilon <= 0.0:
            raise ValueError("gamma_regression_epsilon must be positive.")
        if self.gamma_regression_degree < 2:
            raise ValueError("gamma_regression_degree must be >= 2.")
        if self.gamma_regression_samples < 8:
            raise ValueError("gamma_regression_samples must be >= 8.")

        self._base_bumps: Dict[str, float] = {
            "S_0": float(self.bumps.S_0),
            "sigma": float(self.bumps.sigma),
            "r": float(self.bumps.r),
            "T": float(self.bumps.T),
        }

    def _common_random_normals(self) -> np.ndarray:
        return draw_common_normals(self.pricer, antithetic=self.antithetic)

    def _independent_random_normals(self) -> np.ndarray:
        steps = int(self.pricer.steps)
        paths = int(self.pricer.num_paths)
        return draw_independent_normals(steps, paths, antithetic=self.antithetic)

    def _rlsm_eval_slice(self, cashflows: np.ndarray) -> np.ndarray:
        split = float(self.rlsm_kwargs.get("train_eval_split", 0.5))
        n_paths = cashflows.shape[0]
        train_count = max(1, min(int(n_paths * split), n_paths - 1))
        sample = cashflows[train_count:]
        if sample.size == 0:
            raise ValueError("RLSM evaluation slice is empty; adjust train_eval_split.")
        return sample

    def _price(
        self,
        *,
        S_0: Optional[float] = None,
        sigma: Optional[float] = None,
        r: Optional[float] = None,
        mu: Optional[float] = None,
        T: Optional[float] = None,
        Z: Optional[np.ndarray] = None,
        risk_neutral: Optional[bool] = None,
        continuation_mask: Optional[np.ndarray] = None,
        return_itm_mask: bool = False,
        frozen_policy: object | None = None,
    ) -> tuple[float, float] | tuple[float, float, np.ndarray]:
        """Monte Carlo price (mean, stderr) for supplied parameter overrides."""
        pricer = self.pricer
        risk_flag = self.risk_neutral if risk_neutral is None else bool(risk_neutral)

        overrides: Dict[str, float] = {}
        for name, value in (("S_0", S_0), ("sigma", sigma), ("r", r), ("mu", mu), ("T", T)):
            if value is not None:
                overrides[name] = float(value)

        saved_scalars = {name: getattr(pricer, name) for name in ("S_0", "sigma", "r", "mu", "T")}
        saved_vectors = {
            "_s0_vec": np.array(pricer._s0_vec, copy=True),
            "_sigma_vec": np.array(pricer._sigma_vec, copy=True),
        }

        def _set_override(name: str, value: float) -> None:
            setattr(pricer, name, value)
            if name == "S_0":
                pricer._s0_vec = np.full(pricer.n_assets, value, dtype=float)
            elif name == "sigma":
                pricer._sigma_vec = np.full(pricer.n_assets, value, dtype=float)

        try:
            for name, value in overrides.items():
                _set_override(name, value)

            if float(np.asarray(pricer.S_0, dtype=float).reshape(-1)[0]) <= 0.0:
                raise ValueError("Spot S_0 must be positive.")
            if float(np.asarray(pricer.sigma, dtype=float).reshape(-1)[0]) <= 0.0:
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
                if frozen_policy is not None:
                    discounted = pricer.evaluate_american_policy(frozen_policy, paths_matrix)
                else:
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
                        nn_kwargs=self.nn_kwargs,
                        rlsm_kwargs=self.rlsm_kwargs,
                        train_eval_split=self.train_eval_split,
                    )
                    if return_itm_mask:
                        discounted, mask_used = discounted_result
                    else:
                        discounted = discounted_result
                if self.train_eval_split is not None:
                    split = float(self.train_eval_split)
                    n_paths = discounted.shape[0]
                    train_count = max(1, min(int(n_paths * split), n_paths - 1))
                    discounted_sample = discounted[train_count:]
                elif self.basis_fn == "rlsm":
                    discounted_sample = self._rlsm_eval_slice(discounted)
                else:
                    discounted_sample = discounted
            else:
                discounted = self._discounted_european(paths_matrix)
                discounted_sample = discounted

            mean_price = float(np.mean(discounted_sample))
            ddof = 1 if discounted_sample.size > 1 else 0
            stderr = float(np.std(discounted_sample, ddof=ddof) / np.sqrt(discounted_sample.size))
            if return_itm_mask:
                if mask_used is None:
                    mask_used = np.ones((paths_matrix.shape[0] - 1, paths_matrix.shape[1]), dtype=bool)
                return mean_price, stderr, mask_used
            return mean_price, stderr
        finally:
            for name, value in saved_scalars.items():
                setattr(pricer, name, value)
            pricer._s0_vec = saved_vectors["_s0_vec"]
            pricer._sigma_vec = saved_vectors["_sigma_vec"]

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

    def _resolve_bump_sizes(
        self,
        *,
        bumps: Optional[Dict[str, float]] = None,
        relative: Optional[bool] = None,
    ) -> Dict[str, float]:
        bump_values = dict(self._base_bumps)
        if bumps:
            for key, value in bumps.items():
                if key not in bump_values:
                    raise ValueError(f"Unknown bump key '{key}'. Expected one of {list(bump_values)}.")
                bump_values[key] = float(value)

        rel_flag = self.relative if relative is None else bool(relative)
        resolved: Dict[str, float] = {}
        for name, bump_value in bump_values.items():
            abs_bump = float(bump_value)
            if rel_flag:
                base_value = getattr(self.pricer, name)
                if base_value is None:
                    raise ValueError(f"Parameter '{name}' is not set on the pricing engine.")
                abs_bump = float(base_value) * abs_bump
            if abs_bump == 0.0:
                raise ValueError("Bump size must be non-zero.")
            resolved[name] = abs_bump
        return resolved

    def _prepare_greek_context(self, *, freeze_policy: Optional[bool] = None) -> _GreekEvalContext:
        risk_flag = self.risk_neutral
        Z = self._common_random_normals()
        freeze_flag = self.freeze_policy if freeze_policy is None else bool(freeze_policy)

        context = _GreekEvalContext(risk_flag=risk_flag, Z=Z, freeze_flag=freeze_flag)
        if self.style != "american":
            return context

        if freeze_flag:
            base_paths = self.pricer._simulate_paths(risk_neutral=risk_flag, Z=Z, antithetic=False)
            context.frozen_policy = self.pricer.fit_american_policy(
                call=self.call,
                basis_fn=self.basis_fn,
                include_all_paths=self.include_all_paths,
                mask_tolerance=self.mask_tolerance,
                paths=base_paths,
                rlsm_kwargs=self.rlsm_kwargs,
                train_eval_split=self.train_eval_split,
            )
            return context

        base_price, _, continuation_mask = self._price(
            Z=Z,
            risk_neutral=risk_flag,
            return_itm_mask=True,
        )
        context.base_price = base_price
        context.continuation_mask = continuation_mask
        return context

    def _price_from_context(self, *, name: str, value: float, context: _GreekEvalContext) -> float:
        positive_params = {"S_0", "sigma", "T"}
        if name in positive_params and value <= 0.0:
            raise ValueError(f"Bump results in non-positive {name}. Reduce bump or choose another scheme.")

        kwargs = {name: value}
        result = self._price(
            Z=context.Z,
            risk_neutral=context.risk_flag,
            continuation_mask=context.continuation_mask if self.style == "american" and not context.freeze_flag else None,
            frozen_policy=context.frozen_policy,
            **kwargs,
        )
        return result[0]

    def _base_price_from_context(self, *, name: str, value: float, context: _GreekEvalContext) -> float:
        if context.base_price is None:
            context.base_price = self._price_from_context(name=name, value=value, context=context)
        return context.base_price

    def _compute_greek(
        self,
        kind: str,
        scheme: str = "central",
        *,
        bump_sizes: Dict[str, float],
        context: _GreekEvalContext,
        gamma_method: Optional[str] = None,
    ) -> float:
        kind_l = kind.lower()
        greek_to_param = {
            "delta": "S_0",
            "gamma": "S_0",
            "vega": "sigma",
            "theta": "T",
            "rho": "r",
        }
        if kind_l not in greek_to_param:
            raise ValueError(f"Unsupported greek '{kind}'.")

        param_name = greek_to_param[kind_l]
        base_value = getattr(self.pricer, param_name)
        if base_value is None:
            raise ValueError(f"Parameter '{param_name}' is not set on the pricing engine.")
        base_value = float(base_value)
        bump_value = float(bump_sizes[param_name])

        positive_params = {"S_0", "sigma", "T"}

        def priced_at(name: str, value: float) -> float:
            if name in positive_params and value <= 0.0:
                raise ValueError(f"Bump results in non-positive {name}. Reduce bump or choose another scheme.")
            return self._price_from_context(name=name, value=value, context=context)

        def base_price_value() -> float:
            return self._base_price_from_context(name=param_name, value=base_value, context=context)

        def first_derivative(name: str, base: float, h: float) -> float:
            if h == 0.0:
                raise ValueError("First-derivative bump must be non-zero.")
            scheme_l = scheme.lower()
            if scheme_l == "central" and name in positive_params and (base - h) <= 0.0:
                up = priced_at(name, base + h)
                mid = base_price_value() if name == param_name else priced_at(name, base)
                return (up - mid) / h
            if scheme_l == "central":
                up = priced_at(name, base + h)
                down = priced_at(name, base - h)
                return (up - down) / (2.0 * h)
            if scheme_l == "forward":
                up = priced_at(name, base + h)
                mid = base_price_value() if name == param_name else priced_at(name, base)
                return (up - mid) / h
            if scheme_l == "backward":
                mid = base_price_value() if name == param_name else priced_at(name, base)
                down = priced_at(name, base - h)
                return (mid - down) / h
            raise ValueError("scheme must be one of {'central','forward','backward'}.")

        if kind_l == "gamma":
            gamma_method_l = self.gamma_method if gamma_method is None else str(gamma_method).strip().lower()
            if gamma_method_l not in {"fd", "pde", "regression"}:
                raise ValueError("gamma_method must be one of {'fd','pde','regression'}.")

            if gamma_method_l == "fd":
                if scheme.lower() != "central":
                    raise ValueError("Gamma with finite differences currently supports only scheme='central'.")
                up = priced_at("S_0", base_value + bump_value)
                mid = base_price_value()
                down = priced_at("S_0", base_value - bump_value)
                return (up - 2.0 * mid + down) / (bump_value ** 2)

            if gamma_method_l == "pde":
                if self.pricer.r is None:
                    raise ValueError("PDE gamma requires a scalar risk-free rate r.")
                sigma0 = float(self.pricer.sigma)
                if sigma0 <= 0.0:
                    raise ValueError("PDE gamma requires positive volatility.")
                s0 = float(self.pricer.S_0)
                if s0 <= 0.0:
                    raise ValueError("PDE gamma requires positive spot.")
                r0 = float(self.pricer.r)
                v0 = base_price_value()
                h_s = float(bump_sizes["S_0"])
                h_t = float(bump_sizes["T"])
                delta = first_derivative("S_0", s0, h_s)
                theta = first_derivative("T", float(self.pricer.T), h_t)
                denom = (sigma0 ** 2) * (s0 ** 2)
                return 2.0 * (theta + r0 * (v0 - s0 * delta)) / denom

            eps = self.gamma_regression_epsilon
            n_samples = self.gamma_regression_samples
            degree = min(self.gamma_regression_degree, n_samples - 1)
            rng = np.random.default_rng(getattr(self.pricer, "_seed", None))
            spots = np.clip(base_value + rng.normal(0.0, eps, size=n_samples), 1e-10, None)
            prices = np.array([priced_at("S_0", float(s)) for s in spots], dtype=float)
            x = spots - base_value
            design = np.vander(x, N=degree + 1, increasing=True)
            coeffs, *_ = np.linalg.lstsq(design, prices, rcond=None)
            return float(2.0 * coeffs[2])

        value = first_derivative(param_name, base_value, bump_value)
        if kind_l == "theta":
            return -value
        return value

    def price(self) -> tuple[float, float]:
        return self._price()

    def greek(
        self,
        kind: str,
        scheme: str = "central",
        *,
        bumps: Optional[Dict[str, float]] = None,
        relative: Optional[bool] = None,
        freeze_policy: Optional[bool] = None,
        gamma_method: Optional[str] = None,
    ) -> float:
        """Compute a single Greek (delta, gamma, vega, theta, rho)."""
        context = self._prepare_greek_context(freeze_policy=freeze_policy)
        bump_sizes = self._resolve_bump_sizes(bumps=bumps, relative=relative)
        return self._compute_greek(
            kind,
            scheme=scheme,
            bump_sizes=bump_sizes,
            context=context,
            gamma_method=gamma_method,
        )

    def greeks(
        self,
        kinds: Optional[Iterable[str]] = None,
        scheme: str = "central",
        *,
        bumps: Optional[Dict[str, float]] = None,
        relative: Optional[bool] = None,
        freeze_policy: Optional[bool] = None,
        gamma_method: Optional[str] = None,
        on_start: Optional[Callable[[str], None]] = None,
        on_complete: Optional[Callable[[str, float, float], None]] = None,
    ) -> Dict[str, float]:
        use = list(kinds) if kinds is not None else ["delta", "gamma", "vega", "theta", "rho"]
        context = self._prepare_greek_context(freeze_policy=freeze_policy)
        bump_sizes = self._resolve_bump_sizes(bumps=bumps, relative=relative)
        result: Dict[str, float] = {}
        for kind in use:
            if on_start is not None:
                on_start(kind)
            started = time.perf_counter()
            result[kind] = self._compute_greek(
                kind,
                scheme=scheme,
                bump_sizes=bump_sizes,
                context=context,
                gamma_method=gamma_method,
            )
            if on_complete is not None:
                on_complete(kind, result[kind], float(time.perf_counter() - started))
        return result

    def delta(self) -> float:
        return self.greek("delta")

    def gamma(self) -> float:
        return self.greek("gamma")

    def vega(self) -> float:
        return self.greek("vega")

    def theta(self) -> float:
        return self.greek("theta")

    def rho(self) -> float:
        return self.greek("rho")


__all__ = ["BumpConfig", "MonteCarloFiniteDifference"]
