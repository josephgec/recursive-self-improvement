"""Clean data reserve management for collapse recovery.

Maintains a pool of verified clean (non-synthetic) data that can be drawn
upon when model collapse is detected or suspected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ReserveStatus:
    """Status of the clean data reserve."""
    total_size: int
    remaining: int
    drawn: int
    fraction_remaining: float
    is_sufficient: bool
    min_reserve_fraction: float

    @property
    def fraction_drawn(self) -> float:
        if self.total_size == 0:
            return 0.0
        return self.drawn / self.total_size


class CleanDataReserve:
    """Manages a reserve of clean (non-synthetic) data.

    Ensures we always maintain a minimum fraction of verified clean data
    for recovery purposes.
    """

    def __init__(
        self,
        initial_size: int = 10000,
        min_reserve_fraction: float = 0.1,
    ):
        if initial_size < 0:
            raise ValueError("initial_size must be non-negative")
        if not 0.0 <= min_reserve_fraction <= 1.0:
            raise ValueError("min_reserve_fraction must be in [0, 1]")

        self._total_size = initial_size
        self._remaining = initial_size
        self._min_reserve_fraction = min_reserve_fraction
        self._draw_history: List[Dict[str, Any]] = []

    def verify_reserve(self) -> ReserveStatus:
        """Check the current state of the data reserve.

        Returns:
            ReserveStatus with current metrics.
        """
        fraction = self._remaining / self._total_size if self._total_size > 0 else 0.0
        return ReserveStatus(
            total_size=self._total_size,
            remaining=self._remaining,
            drawn=self._total_size - self._remaining,
            fraction_remaining=fraction,
            is_sufficient=fraction >= self._min_reserve_fraction,
            min_reserve_fraction=self._min_reserve_fraction,
        )

    def draw(self, n: int) -> List[int]:
        """Draw n samples from the reserve.

        Args:
            n: Number of samples to draw.

        Returns:
            List of sample indices drawn.

        Raises:
            ValueError: If n > remaining or would breach minimum reserve.
        """
        if n < 0:
            raise ValueError("Cannot draw negative samples")
        if n > self._remaining:
            raise ValueError(
                f"Cannot draw {n} samples, only {self._remaining} remaining"
            )

        min_reserve = int(self._total_size * self._min_reserve_fraction)
        if self._remaining - n < min_reserve:
            raise ValueError(
                f"Drawing {n} would breach minimum reserve of {min_reserve}. "
                f"Currently {self._remaining} remaining."
            )

        start = self._total_size - self._remaining
        indices = list(range(start, start + n))
        self._remaining -= n

        self._draw_history.append({
            "n": n,
            "remaining_after": self._remaining,
            "indices_start": start,
            "indices_end": start + n,
        })

        return indices

    def get_remaining(self) -> int:
        """Return number of remaining samples."""
        return self._remaining

    def is_sufficient(self) -> bool:
        """Check if reserve is above minimum threshold."""
        if self._total_size == 0:
            return False
        return (self._remaining / self._total_size) >= self._min_reserve_fraction

    def get_draw_history(self) -> List[Dict[str, Any]]:
        """Return the history of all draws."""
        return list(self._draw_history)

    def replenish(self, n: int) -> None:
        """Add verified clean data back to the reserve.

        Args:
            n: Number of verified clean samples to add.
        """
        if n < 0:
            raise ValueError("Cannot replenish negative samples")
        self._remaining += n
        self._total_size += n
