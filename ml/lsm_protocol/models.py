"""Core data models for the LSM-ML evaluation protocol."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OptionScenario:
    """Single option setup used as one evaluation scenario."""

    S0: float
    K: float
    T: float
    call: bool

    @property
    def moneyness(self) -> float:
        return self.S0 / self.K


@dataclass
class StepModel:
    """Regression model fitted at one LSM backward step."""

    kind: str
    model: Any
    degree: int = 0


@dataclass
class LSMPolicy:
    """Fitted exercise policy composed of per-step continuation models."""

    call: bool
    strike: float
    rate: float
    maturity: float
    steps: int
    models: list[StepModel | None]
    fallback_values: list[float]

    @property
    def dt(self) -> float:
        return self.maturity / self.steps

    @property
    def discount_grid(self) -> np.ndarray:
        return np.exp(-self.rate * self.dt * np.arange(self.steps + 1))

    def continuation_value(self, t: int, states: np.ndarray) -> np.ndarray:
        x = np.asarray(states, dtype=float).ravel()
        if t >= self.steps:
            return np.zeros_like(x)

        step_model = self.models[t]
        if step_model is None:
            return np.full(x.shape[0], self.fallback_values[t], dtype=float)

        if step_model.kind == "ols":
            features = np.vander(x, N=step_model.degree + 1, increasing=True)
            coeffs = np.asarray(step_model.model, dtype=float)
            return features @ coeffs

        if step_model.kind in {"svr", "cart"}:
            return np.asarray(step_model.model.predict(x.reshape(-1, 1)), dtype=float)

        raise ValueError(f"Unknown step model kind: {step_model.kind}")
