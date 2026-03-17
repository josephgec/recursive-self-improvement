from __future__ import annotations

"""Per-layer norm tracking."""

import numpy as np
from dataclasses import dataclass


@dataclass
class LayerNormSnapshot:
    """Snapshot of per-layer norms at a point in time."""

    step: int
    norms: list[float]
    mean_norm: float
    norm_variance: float


class LayerNormTracker:
    """Tracks per-layer activation norms over training.

    Monitors for divergence between layers and norm collapse.
    """

    def __init__(self, num_layers: int = 6):
        self._num_layers = num_layers
        self._history: list[LayerNormSnapshot] = []
        self._step = 0

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def history(self) -> list[LayerNormSnapshot]:
        return list(self._history)

    def track(self, activations: list[np.ndarray]) -> LayerNormSnapshot:
        """Track norms for a set of layer activations.

        Args:
            activations: List of numpy arrays, one per layer.

        Returns:
            LayerNormSnapshot with current norm values.
        """
        self._step += 1

        norms = []
        for act in activations:
            norm = float(np.linalg.norm(act))
            norms.append(norm)

        mean_norm = float(np.mean(norms))
        norm_variance = float(np.var(norms))

        snapshot = LayerNormSnapshot(
            step=self._step,
            norms=norms,
            mean_norm=mean_norm,
            norm_variance=norm_variance,
        )
        self._history.append(snapshot)
        return snapshot

    def get_layer_trend(self, layer: int, window: int = 10) -> str:
        """Get the trend for a specific layer.

        Returns "increasing", "decreasing", "stable", or "insufficient_data".
        """
        if len(self._history) < window or layer >= self._num_layers:
            return "insufficient_data"

        recent = [
            s.norms[layer]
            for s in self._history[-window:]
            if layer < len(s.norms)
        ]

        if len(recent) < window:
            return "insufficient_data"

        first_half = np.mean(recent[: window // 2])
        second_half = np.mean(recent[window // 2 :])

        if first_half <= 0:
            return "stable"

        change = (second_half - first_half) / first_half

        if change > 0.05:
            return "increasing"
        elif change < -0.05:
            return "decreasing"
        return "stable"

    def get_divergence(self) -> float:
        """Get current inter-layer norm divergence.

        Returns coefficient of variation across layer norms.
        """
        if not self._history:
            return 0.0

        latest = self._history[-1]
        mean = latest.mean_norm
        if mean <= 0:
            return 0.0

        return float(np.std(latest.norms) / mean)

    def is_collapsing(self, threshold: float = 0.3, window: int = 10) -> bool:
        """Check if norms are collapsing across layers."""
        if len(self._history) < window:
            return False

        recent_means = [s.mean_norm for s in self._history[-window:]]
        first = np.mean(recent_means[: window // 2])
        second = np.mean(recent_means[window // 2 :])

        if first <= 0:
            return False

        decline = (first - second) / first
        return decline > threshold
