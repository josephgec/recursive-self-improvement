"""Alpha schedule strategies for controlling real/synthetic data mixing ratio."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any


class AlphaSchedule(ABC):
    """Base class for alpha schedules.

    An alpha schedule determines the fraction of real data (alpha_t) used
    at each generation *t* of a recursive model lineage.  ``alpha_t = 1``
    means fully real; ``alpha_t = 0`` means fully synthetic.
    """

    @abstractmethod
    def __call__(self, generation: int, total_generations: int) -> float:
        """Return alpha for a given generation.

        Args:
            generation: 0-indexed generation number.
            total_generations: Total number of generations in the lineage.

        Returns:
            Alpha value in [0, 1].
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ConstantAlpha(AlphaSchedule):
    """Return a fixed alpha at every generation."""

    def __init__(self, alpha: float = 0.5) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self._alpha = alpha

    def __call__(self, generation: int, total_generations: int) -> float:
        return self._alpha

    def __repr__(self) -> str:
        return f"ConstantAlpha(alpha={self._alpha})"


class LinearDecay(AlphaSchedule):
    """Linearly decay alpha from ``alpha_0`` to ``alpha_min``.

    At generation 0 the value is ``alpha_0``; at the last generation it is
    ``alpha_min``.  For a single generation, ``alpha_0`` is returned.
    """

    def __init__(self, alpha_0: float = 1.0, alpha_min: float = 0.0) -> None:
        self._alpha_0 = alpha_0
        self._alpha_min = alpha_min

    def __call__(self, generation: int, total_generations: int) -> float:
        if total_generations <= 1:
            return self._alpha_0
        t = generation / (total_generations - 1)
        return self._alpha_0 + t * (self._alpha_min - self._alpha_0)

    def __repr__(self) -> str:
        return (
            f"LinearDecay(alpha_0={self._alpha_0}, alpha_min={self._alpha_min})"
        )


class ExponentialDecay(AlphaSchedule):
    """Exponentially decay alpha: ``alpha_t = alpha_0 * gamma^t``."""

    def __init__(self, alpha_0: float = 1.0, gamma: float = 0.8) -> None:
        self._alpha_0 = alpha_0
        self._gamma = gamma

    def __call__(self, generation: int, total_generations: int) -> float:
        return self._alpha_0 * (self._gamma ** generation)

    def __repr__(self) -> str:
        return (
            f"ExponentialDecay(alpha_0={self._alpha_0}, gamma={self._gamma})"
        )


class ZeroAlpha(AlphaSchedule):
    """Always return alpha=0 (pure synthetic data)."""

    def __call__(self, generation: int, total_generations: int) -> float:
        return 0.0

    def __repr__(self) -> str:
        return "ZeroAlpha()"


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

_SCHEDULE_REGISTRY: dict[str, type[AlphaSchedule]] = {
    "constant": ConstantAlpha,
    "linear": LinearDecay,
    "exponential": ExponentialDecay,
    "zero": ZeroAlpha,
}


def schedule_from_config(config: dict[str, Any]) -> AlphaSchedule:
    """Instantiate an AlphaSchedule from a config dictionary.

    Expected structure (mirrors ``configs/schedules/*.yaml``)::

        {"type": "constant", "alpha": 0.5}
        {"type": "linear", "alpha_0": 1.0, "alpha_min": 0.0}
        {"type": "exponential", "alpha_0": 1.0, "gamma": 0.8}
        {"type": "zero"}

    Args:
        config: Schedule configuration dict.  Must contain a ``"type"`` key.

    Returns:
        An ``AlphaSchedule`` instance.
    """
    schedule_type = config.get("type")
    if schedule_type is None:
        raise ValueError("Schedule config must contain a 'type' key")

    cls = _SCHEDULE_REGISTRY.get(schedule_type)
    if cls is None:
        raise ValueError(
            f"Unknown schedule type '{schedule_type}'. "
            f"Available: {list(_SCHEDULE_REGISTRY.keys())}"
        )

    # Pass all keys except 'type' as kwargs to the constructor.
    kwargs = {k: v for k, v in config.items() if k != "type"}
    return cls(**kwargs)
