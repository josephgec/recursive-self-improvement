"""Depth analysis: recursion depth distribution."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from src.core.session import SessionResult


def recursion_depth_distribution(
    results: List[SessionResult],
) -> Dict[str, Any]:
    """Analyze recursion depth across sessions.

    Returns distribution, average, and max depth.
    """
    if not results:
        return {"distribution": {}, "avg_depth": 0.0, "max_depth": 0}

    depths = [r.depth for r in results]
    counter = Counter(depths)
    return {
        "distribution": dict(counter),
        "avg_depth": sum(depths) / len(depths),
        "max_depth": max(depths),
        "total_sessions": len(results),
    }
