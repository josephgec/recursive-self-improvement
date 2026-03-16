"""LatencyCeilingConstraint: P95 latency must not exceed threshold."""

from __future__ import annotations

from typing import Any, List

from src.constraints.base import Constraint, ConstraintResult, CheckContext


class LatencyCeilingConstraint(Constraint):
    """P95 latency must remain below the ceiling."""

    def __init__(self, p95_threshold_ms: float = 30000) -> None:
        super().__init__(
            name="latency_ceiling",
            description="P95 latency must not exceed threshold",
            category="quality",
            threshold=p95_threshold_ms,
        )

    def check(self, agent_state: Any, context: CheckContext) -> ConstraintResult:
        """Evaluate latency.

        ``agent_state`` must expose:
        * ``get_latency_samples() -> List[float]``  latencies in milliseconds.
        """
        samples = agent_state.get_latency_samples()
        p95 = self._percentile(samples, 95)

        headroom = self._threshold - p95
        satisfied = p95 <= self._threshold

        return ConstraintResult(
            satisfied=satisfied,
            measured_value=p95,
            threshold=self._threshold,
            headroom=headroom,
            details={
                "p95_ms": p95,
                "p50_ms": self._percentile(samples, 50),
                "p99_ms": self._percentile(samples, 99),
                "sample_count": len(samples),
                "direction": "ceiling",
            },
        )

    def headroom(self, measured_value: float) -> float:
        """For ceiling constraints headroom is threshold - measured."""
        return self._threshold - measured_value

    @staticmethod
    def _percentile(data: List[float], pct: float) -> float:
        """Compute percentile without numpy."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * (pct / 100.0)
        f = int(k)
        c = f + 1
        if c >= len(sorted_data):
            return sorted_data[-1]
        d0 = sorted_data[f] * (c - k)
        d1 = sorted_data[c] * (k - f)
        return d0 + d1
