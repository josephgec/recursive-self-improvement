"""Metrics tracking for REPL pool management."""

import time
import threading
from dataclasses import dataclass, field
from typing import List


@dataclass
class MetricsSummary:
    """Summary of pool metrics.

    Attributes:
        total_acquires: Total number of acquire operations.
        total_releases: Total number of release operations.
        total_wait_time_ms: Total time spent waiting for REPLs.
        avg_wait_time_ms: Average wait time per acquire.
        total_hold_time_ms: Total time REPLs were held.
        avg_hold_time_ms: Average hold time per release.
        peak_concurrent: Peak number of concurrently held REPLs.
    """

    total_acquires: int = 0
    total_releases: int = 0
    total_wait_time_ms: float = 0.0
    avg_wait_time_ms: float = 0.0
    total_hold_time_ms: float = 0.0
    avg_hold_time_ms: float = 0.0
    peak_concurrent: int = 0


class PoolMetricsTracker:
    """Tracks metrics for REPL pool operations.

    Records acquire/release events and computes summary statistics.
    """

    def __init__(self):
        self._acquires: List[float] = []  # timestamps
        self._releases: List[float] = []  # timestamps
        self._wait_times: List[float] = []  # milliseconds
        self._hold_times: List[float] = []  # milliseconds
        self._current_held: int = 0
        self._peak_concurrent: int = 0
        self._lock = threading.Lock()

    def record_acquire(self, wait_time_ms: float = 0.0) -> None:
        """Record a REPL acquire operation.

        Args:
            wait_time_ms: Time spent waiting for the REPL in milliseconds.
        """
        with self._lock:
            self._acquires.append(time.time())
            self._wait_times.append(wait_time_ms)
            self._current_held += 1
            self._peak_concurrent = max(self._peak_concurrent, self._current_held)

    def record_release(self, hold_time_ms: float = 0.0) -> None:
        """Record a REPL release operation.

        Args:
            hold_time_ms: Time the REPL was held in milliseconds.
        """
        with self._lock:
            self._releases.append(time.time())
            self._hold_times.append(hold_time_ms)
            self._current_held = max(0, self._current_held - 1)

    def summary(self) -> MetricsSummary:
        """Get a summary of all tracked metrics.

        Returns:
            MetricsSummary with aggregated statistics.
        """
        with self._lock:
            total_acquires = len(self._acquires)
            total_releases = len(self._releases)
            total_wait = sum(self._wait_times)
            total_hold = sum(self._hold_times)

            return MetricsSummary(
                total_acquires=total_acquires,
                total_releases=total_releases,
                total_wait_time_ms=total_wait,
                avg_wait_time_ms=total_wait / total_acquires if total_acquires else 0,
                total_hold_time_ms=total_hold,
                avg_hold_time_ms=total_hold / total_releases if total_releases else 0,
                peak_concurrent=self._peak_concurrent,
            )

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._acquires.clear()
            self._releases.clear()
            self._wait_times.clear()
            self._hold_times.clear()
            self._current_held = 0
            self._peak_concurrent = 0
