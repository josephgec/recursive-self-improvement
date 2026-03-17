from __future__ import annotations

"""Mock value head for EPPO training."""

import numpy as np
from dataclasses import dataclass


@dataclass
class ValuePrediction:
    """Value head prediction result."""

    value: float
    confidence: float


@dataclass
class DriftResult:
    """Result of value drift detection."""

    is_drifting: bool
    drift_magnitude: float
    trend: str  # "stable", "increasing", "decreasing"
    window_mean: float
    window_std: float


class MockValueHead:
    """Mock value head network using numpy.

    Predicts state values and detects value drift over training.
    """

    def __init__(self, input_dim: int = 64, seed: int = 42):
        self._rng = np.random.RandomState(seed)
        self._input_dim = input_dim
        self._weights = self._rng.randn(input_dim) * 0.1
        self._bias = 0.0
        self._prediction_history: list[float] = []
        self._drift_threshold = 0.5

    @property
    def prediction_history(self) -> list[float]:
        """History of value predictions."""
        return list(self._prediction_history)

    def predict_value(self, state: np.ndarray) -> float:
        """Predict value for a state.

        Args:
            state: array of shape (input_dim,)

        Returns:
            Predicted scalar value.
        """
        if state.ndim > 1:
            state = state.flatten()[:self._input_dim]

        value = float(np.dot(state[:self._input_dim], self._weights) + self._bias)
        # Add small noise
        value += self._rng.randn() * 0.01
        self._prediction_history.append(value)
        return value

    def predict_batch(self, states: np.ndarray) -> np.ndarray:
        """Predict values for a batch of states.

        Args:
            states: array of shape (batch, input_dim)

        Returns:
            Array of predicted values, shape (batch,).
        """
        values = states @ self._weights + self._bias
        values += self._rng.randn(len(states)) * 0.01
        for v in values:
            self._prediction_history.append(float(v))
        return values

    def detect_drift(self, history: list[float] | None = None, window: int = 20) -> DriftResult:
        """Detect drift in value predictions.

        Args:
            history: Optional explicit history; uses internal if None.
            window: Window size for drift analysis.

        Returns:
            DriftResult with drift analysis.
        """
        h = history if history is not None else self._prediction_history

        if len(h) < window:
            return DriftResult(
                is_drifting=False,
                drift_magnitude=0.0,
                trend="stable",
                window_mean=float(np.mean(h)) if h else 0.0,
                window_std=float(np.std(h)) if h else 0.0,
            )

        recent = h[-window:]
        older = h[-2 * window : -window] if len(h) >= 2 * window else h[: len(h) - window]

        recent_mean = float(np.mean(recent))
        recent_std = float(np.std(recent))
        older_mean = float(np.mean(older))

        drift_magnitude = abs(recent_mean - older_mean)
        is_drifting = drift_magnitude > self._drift_threshold

        if recent_mean > older_mean + self._drift_threshold * 0.5:
            trend = "increasing"
        elif recent_mean < older_mean - self._drift_threshold * 0.5:
            trend = "decreasing"
        else:
            trend = "stable"

        return DriftResult(
            is_drifting=is_drifting,
            drift_magnitude=drift_magnitude,
            trend=trend,
            window_mean=recent_mean,
            window_std=recent_std,
        )

    def update_weights(self, gradient: np.ndarray, lr: float = 0.001) -> None:
        """Mock weight update."""
        self._weights -= lr * gradient[:self._input_dim]
