"""Depth conditions for the RLM depth experiment."""

from dataclasses import dataclass
from typing import List


@dataclass
class DepthCondition:
    """A single condition in the RLM depth experiment."""

    name: str
    description: str
    depth: int


def build_depth_conditions() -> List[DepthCondition]:
    """Build all 7 depth conditions (depth 0 through 6)."""
    return [
        DepthCondition(
            name=f"depth_{d}",
            description=f"Recursion depth {d}" if d > 0 else "No recursion (baseline)",
            depth=d,
        )
        for d in range(7)
    ]
