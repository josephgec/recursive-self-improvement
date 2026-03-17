"""Circuit breaker for runaway cost control.

Monitors sub-queries, tokens, and costs per query.
Kills operations that exceed safety thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CircuitBreakerState:
    """Current state of the circuit breaker."""
    query_id: str
    sub_queries: int
    tokens: int
    cost: float
    tripped: bool
    trip_reason: Optional[str] = None


class CircuitBreaker:
    """Circuit breaker for runaway query cost control.

    Tracks sub-queries, tokens, and cost per query.
    Trips (kills) if any threshold is exceeded.
    """

    def __init__(
        self,
        max_sub_queries: int = 50,
        max_tokens: int = 1_000_000,
        max_cost_per_query: float = 10.00,
    ):
        self.max_sub_queries = max_sub_queries
        self.max_tokens = max_tokens
        self.max_cost_per_query = max_cost_per_query

        self._current_query: Optional[str] = None
        self._sub_queries: int = 0
        self._tokens: int = 0
        self._cost: float = 0.0
        self._tripped: bool = False
        self._trip_reason: Optional[str] = None
        self._trip_history: List[CircuitBreakerState] = []

    def register_query(self, query_id: str) -> None:
        """Register a new top-level query, resetting counters.

        Args:
            query_id: Unique identifier for the query.
        """
        self._current_query = query_id
        self._sub_queries = 0
        self._tokens = 0
        self._cost = 0.0
        self._tripped = False
        self._trip_reason = None

    def record_sub_query(self, cost: float = 0.0) -> bool:
        """Record a sub-query and check thresholds.

        Args:
            cost: Cost of this sub-query.

        Returns:
            True if still within limits, False if tripped.
        """
        self._sub_queries += 1
        self._cost += cost
        return self.check()

    def record_tokens(self, tokens: int, cost: float = 0.0) -> bool:
        """Record token usage and check thresholds.

        Args:
            tokens: Number of tokens used.
            cost: Cost of these tokens.

        Returns:
            True if still within limits, False if tripped.
        """
        self._tokens += tokens
        self._cost += cost
        return self.check()

    def check(self, kill: bool = False) -> bool:
        """Check all thresholds.

        Args:
            kill: If True, also execute kill on trip.

        Returns:
            True if within limits, False if tripped.
        """
        if self._tripped:
            return False

        if self._sub_queries >= self.max_sub_queries:
            self._trip("max_sub_queries_exceeded", kill)
            return False

        if self._tokens >= self.max_tokens:
            self._trip("max_tokens_exceeded", kill)
            return False

        if self._cost >= self.max_cost_per_query:
            self._trip("max_cost_exceeded", kill)
            return False

        return True

    def _trip(self, reason: str, execute_kill: bool = False) -> None:
        """Trip the circuit breaker."""
        self._tripped = True
        self._trip_reason = reason

        state = CircuitBreakerState(
            query_id=self._current_query or "unknown",
            sub_queries=self._sub_queries,
            tokens=self._tokens,
            cost=self._cost,
            tripped=True,
            trip_reason=reason,
        )
        self._trip_history.append(state)

        if execute_kill:
            self.kill()

    def kill(self) -> Dict[str, Any]:
        """Kill the current operation.

        Returns:
            Dict with kill details.
        """
        self._tripped = True
        return {
            "status": "killed",
            "query_id": self._current_query,
            "reason": self._trip_reason or "manual_kill",
            "sub_queries": self._sub_queries,
            "tokens": self._tokens,
            "cost": self._cost,
        }

    @property
    def is_tripped(self) -> bool:
        return self._tripped

    @property
    def state(self) -> CircuitBreakerState:
        return CircuitBreakerState(
            query_id=self._current_query or "none",
            sub_queries=self._sub_queries,
            tokens=self._tokens,
            cost=self._cost,
            tripped=self._tripped,
            trip_reason=self._trip_reason,
        )

    def get_trip_history(self) -> List[CircuitBreakerState]:
        """Return history of all circuit breaker trips."""
        return list(self._trip_history)
