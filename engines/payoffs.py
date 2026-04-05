"""Shared payoff helpers for multi-asset experiments."""

from typing import Callable
import numpy as np

PayoffFn = Callable[[np.ndarray], np.ndarray]

_PAYOFF_STYLE_ALIASES = {
    "vanilla": "vanilla",
    "arithmetic_basket": "arithmetic_basket",
    "basket_put": "arithmetic_basket",
    "basket_call": "arithmetic_basket",
    "geometric_basket": "geometric_basket_put",
    "geometric_basket_put": "geometric_basket_put",
    "max_basket": "max_call",
    "max_call": "max_call",
    "min_put": "min_put",
}

def _coerce_states(states: np.ndarray) -> np.ndarray:
    arr = np.asarray(states, dtype=float)
    if arr.ndim not in (2, 3):
        raise ValueError("states must be either a 2D or 3D array.")
    return arr

# For arithmetic basket payoffs' weights
def _weighted_basket_value(states: np.ndarray, weights: np.ndarray) -> np.ndarray:
    arr = _coerce_states(states)
    if arr.ndim == 3:
        return np.tensordot(arr, weights, axes=([2], [0]))
    return arr @ weights

def normalise_payoff_style(payoff_style: str) -> str:
    key = str(payoff_style).strip().lower()
    try:
        return _PAYOFF_STYLE_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown payoff_style '{payoff_style}'. Supported styles: {sorted(_PAYOFF_STYLE_ALIASES)}."
        ) from exc

def available_payoff_styles() -> tuple[str, ...]:
    return tuple(sorted(set(_PAYOFF_STYLE_ALIASES.values())))

def arithmetic_basket_payoff(states: np.ndarray, *, strike: float, call: bool, weights: np.ndarray) -> np.ndarray:
    basket = _weighted_basket_value(states, np.asarray(weights, dtype=float))
    if call:
        return np.maximum(basket - strike, 0.0)
    return np.maximum(strike - basket, 0.0)

def geometric_basket_payoff(states: np.ndarray, *, strike: float, call: bool) -> np.ndarray:
    arr = _coerce_states(states)
    basket = np.prod(arr, axis=-1) ** (1.0 / arr.shape[-1])
    if call:
        return np.maximum(basket - strike, 0.0)
    return np.maximum(strike - basket, 0.0)

def max_basket_payoff(states: np.ndarray, *, strike: float, call: bool) -> np.ndarray:
    arr = _coerce_states(states)
    basket = np.max(arr, axis=-1)
    if call:
        return np.maximum(basket - strike, 0.0)
    return np.maximum(strike - basket, 0.0)

def min_basket_payoff(states: np.ndarray, *, strike: float, call: bool) -> np.ndarray:
    arr = _coerce_states(states)
    basket = np.min(arr, axis=-1)
    if call:
        return np.maximum(basket - strike, 0.0)
    return np.maximum(strike - basket, 0.0)

def max_call_payoff(states: np.ndarray, *, strike: float) -> np.ndarray:
    return max_basket_payoff(states, strike=strike, call=True)

def geometric_basket_put_payoff(states: np.ndarray, *, strike: float) -> np.ndarray:
    return geometric_basket_payoff(states, strike=strike, call=False)

def min_put_payoff(states: np.ndarray, *, strike: float) -> np.ndarray:
    return min_basket_payoff(states, strike=strike, call=False)

def make_payoff_fn(*, payoff_style: str, strike: float, call: bool, weights: np.ndarray | None = None) -> PayoffFn | None:
    style = normalise_payoff_style(payoff_style)

    if style == "vanilla":
        return None
    
    if style == "arithmetic_basket":
        if weights is None:
            raise ValueError("weights are required for payoff_style='arithmetic_basket'.")
        resolved_weights = np.asarray(weights, dtype=float)
        return lambda states: arithmetic_basket_payoff(states, strike=strike, call=bool(call), weights=resolved_weights)
    
    if style == "geometric_basket_put":
        if call:
            raise ValueError("payoff_style='geometric_basket_put' is only valid for put payoffs.")
        return lambda states: geometric_basket_put_payoff(states, strike=strike)
    
    if style == "max_call":
        if not call:
            raise ValueError("payoff_style='max_call' is only valid for call payoffs.")
        return lambda states: max_call_payoff(states, strike=strike)
    
    if style == "min_put":
        if call:
            raise ValueError("payoff_style='min_put' is only valid for put payoffs.")
        return lambda states: min_put_payoff(states, strike=strike)

    raise ValueError(f"Unsupported payoff style '{payoff_style}'.")

__all__ = [
    "PayoffFn",
    "arithmetic_basket_payoff",
    "available_payoff_styles",
    "geometric_basket_payoff",
    "geometric_basket_put_payoff",
    "make_payoff_fn",
    "max_basket_payoff",
    "max_call_payoff",
    "min_basket_payoff",
    "min_put_payoff",
    "normalise_payoff_style",
]
