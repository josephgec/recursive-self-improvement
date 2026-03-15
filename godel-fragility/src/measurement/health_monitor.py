"""Monitor agent health during stress testing."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class HealthStatus:
    """Current health status of the agent."""

    is_alive: bool = True
    is_progressing: bool = True
    resource_ok: bool = True
    modification_rate_ok: bool = True
    overall_healthy: bool = True
    details: str = ""
    timestamp: float = field(default_factory=time.monotonic)


class AgentHealthMonitor:
    """Monitor agent health by checking liveness, progress, resources, and modification rate."""

    def __init__(
        self,
        max_stagnant_iterations: int = 10,
        max_modification_rate: float = 5.0,  # modifications per iteration
        max_memory_mb: float = 1024.0,
    ) -> None:
        self._max_stagnant = max_stagnant_iterations
        self._max_mod_rate = max_modification_rate
        self._max_memory = max_memory_mb
        self._history: List[HealthStatus] = []
        self._recent_accuracies: List[float] = []
        self._modification_counts: List[int] = []

    def check(
        self,
        agent: Any,
        iteration: int,
        accuracy: float,
        modification_count: int = 0,
    ) -> HealthStatus:
        """Run all health checks on the agent.

        Args:
            agent: The agent to check.
            iteration: Current iteration number.
            accuracy: Current accuracy.
            modification_count: Number of modifications this iteration.

        Returns:
            HealthStatus with detailed results.
        """
        self._recent_accuracies.append(accuracy)
        self._modification_counts.append(modification_count)

        is_alive = self._check_liveness(agent)
        is_progressing = self._check_progress()
        resource_ok = self._check_resources()
        mod_rate_ok = self._check_modification_rate()

        overall = is_alive and is_progressing and resource_ok and mod_rate_ok

        details_parts = []
        if not is_alive:
            details_parts.append("Agent not responding")
        if not is_progressing:
            details_parts.append(f"Stagnant for {self._max_stagnant}+ iterations")
        if not resource_ok:
            details_parts.append("Resource limits exceeded")
        if not mod_rate_ok:
            details_parts.append("Modification rate too high")

        status = HealthStatus(
            is_alive=is_alive,
            is_progressing=is_progressing,
            resource_ok=resource_ok,
            modification_rate_ok=mod_rate_ok,
            overall_healthy=overall,
            details="; ".join(details_parts) if details_parts else "healthy",
        )

        self._history.append(status)
        return status

    def should_terminate(self) -> bool:
        """Should the stress test be terminated based on health?"""
        if not self._history:
            return False

        latest = self._history[-1]

        # Terminate if agent is dead
        if not latest.is_alive:
            return True

        # Terminate if stagnant for too long
        stagnant_count = 0
        for status in reversed(self._history):
            if not status.is_progressing:
                stagnant_count += 1
            else:
                break
        if stagnant_count >= self._max_stagnant:
            return True

        return False

    def _check_liveness(self, agent: Any) -> bool:
        """Check if the agent is still alive and responsive."""
        try:
            return agent.is_functional()
        except Exception:
            return False

    def _check_progress(self) -> bool:
        """Check if the agent is making progress (not stagnant)."""
        if len(self._recent_accuracies) < self._max_stagnant:
            return True

        recent = self._recent_accuracies[-self._max_stagnant:]
        if not recent:
            return True

        # Stagnant if all recent accuracies are within 1% of each other
        min_acc = min(recent)
        max_acc = max(recent)
        return (max_acc - min_acc) > 0.01

    def _check_resources(self) -> bool:
        """Check resource usage (simplified -- just checks iteration count)."""
        # In a real implementation, would check memory, CPU, etc.
        return True

    def _check_modification_rate(self) -> bool:
        """Check if modifications are happening at a reasonable rate."""
        if len(self._modification_counts) < 3:
            return True

        recent = self._modification_counts[-3:]
        avg_rate = sum(recent) / len(recent)
        return avg_rate <= self._max_mod_rate

    @property
    def history(self) -> List[HealthStatus]:
        return list(self._history)
