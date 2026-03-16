"""Lineage tracker: tracks modification lineage and improvement chains."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class LineageTracker:
    """Tracks the lineage of modifications and improvement chains."""

    def __init__(self):
        self._lineage: List[Dict[str, Any]] = []

    def record_modification(
        self,
        iteration: int,
        candidate_id: str,
        target: str,
        parent_ids: Optional[List[str]] = None,
        accuracy_before: float = 0.0,
        accuracy_after: float = 0.0,
        applied: bool = True,
    ) -> None:
        """Record a modification in the lineage."""
        entry = {
            "iteration": iteration,
            "candidate_id": candidate_id,
            "target": target,
            "parent_ids": parent_ids or [],
            "accuracy_before": accuracy_before,
            "accuracy_after": accuracy_after,
            "applied": applied,
            "improved": accuracy_after > accuracy_before,
        }
        self._lineage.append(entry)

    def get_lineage(self, candidate_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get lineage entries, optionally filtered by candidate_id."""
        if candidate_id is None:
            return list(self._lineage)
        return [e for e in self._lineage if e["candidate_id"] == candidate_id]

    def trace_improvement(self, candidate_id: str) -> List[Dict[str, Any]]:
        """Trace the improvement chain for a candidate through its parents."""
        chain: List[Dict[str, Any]] = []
        visited: set = set()
        queue = [candidate_id]

        while queue:
            cid = queue.pop(0)
            if cid in visited:
                continue
            visited.add(cid)
            entries = self.get_lineage(cid)
            for entry in entries:
                chain.append(entry)
                for parent in entry.get("parent_ids", []):
                    if parent not in visited:
                        queue.append(parent)

        return chain

    @property
    def size(self) -> int:
        return len(self._lineage)

    def clear(self) -> None:
        self._lineage.clear()
