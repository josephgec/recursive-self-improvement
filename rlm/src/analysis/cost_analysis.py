"""Cost analysis: token usage, latency, cost per query."""

from __future__ import annotations

from typing import Any, Dict, List

from src.core.session import SessionResult


def token_usage(
    results: List[SessionResult],
    tokens_per_iteration: int = 500,
) -> Dict[str, Any]:
    """Estimate total and per-query token usage."""
    if not results:
        return {"total_tokens": 0, "avg_tokens": 0, "per_query": []}

    per_query = [r.total_iterations * tokens_per_iteration for r in results]
    total = sum(per_query)
    return {
        "total_tokens": total,
        "avg_tokens": total / len(results),
        "per_query": per_query,
    }


def latency_analysis(
    results: List[SessionResult],
) -> Dict[str, Any]:
    """Analyze elapsed time across sessions."""
    if not results:
        return {"total_elapsed": 0.0, "avg_elapsed": 0.0, "per_query": []}

    elapsed = [r.elapsed_time for r in results]
    return {
        "total_elapsed": sum(elapsed),
        "avg_elapsed": sum(elapsed) / len(elapsed),
        "min_elapsed": min(elapsed),
        "max_elapsed": max(elapsed),
        "per_query": elapsed,
    }


def cost_per_query(
    results: List[SessionResult],
    cost_per_iteration: float = 0.01,
) -> Dict[str, Any]:
    """Compute cost breakdown per query."""
    if not results:
        return {"total_cost": 0.0, "avg_cost": 0.0, "per_query": []}

    per_query = [r.total_iterations * cost_per_iteration for r in results]
    total = sum(per_query)
    return {
        "total_cost": total,
        "avg_cost": total / len(results),
        "per_query": per_query,
    }
