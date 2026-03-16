"""Spawn depth limiting for sandboxed REPL hierarchies."""

import threading
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class DepthStatus:
    """Status of spawn depth tracking.

    Attributes:
        current_depth: Current depth level.
        max_depth: Maximum allowed depth.
        active_spawns: Number of currently active spawned REPLs.
        can_spawn: Whether another spawn is allowed.
    """

    current_depth: int = 0
    max_depth: int = 5
    active_spawns: int = 0
    can_spawn: bool = True


class DepthLimiter:
    """Manages REPL spawn depth to prevent unbounded recursion.

    Tracks the depth of spawned child REPLs and enforces a maximum
    depth limit.
    """

    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        self._depths: Dict[str, int] = {}
        self._active: Dict[str, bool] = {}
        self._lock = threading.Lock()

    def can_spawn(self, parent_id: str) -> bool:
        """Check if a parent REPL can spawn a child.

        Args:
            parent_id: Identifier of the parent REPL.

        Returns:
            True if the spawn would not exceed the depth limit.
        """
        with self._lock:
            current_depth = self._depths.get(parent_id, 0)
            return current_depth < self.max_depth

    def register_spawn(self, parent_id: str, child_id: str) -> int:
        """Register a new child REPL spawn.

        Args:
            parent_id: Identifier of the parent REPL.
            child_id: Identifier of the child REPL.

        Returns:
            The depth of the child REPL.
        """
        with self._lock:
            parent_depth = self._depths.get(parent_id, 0)
            child_depth = parent_depth + 1
            self._depths[child_id] = child_depth
            self._active[child_id] = True
            # Ensure parent is tracked
            if parent_id not in self._depths:
                self._depths[parent_id] = 0
            if parent_id not in self._active:
                self._active[parent_id] = True
            return child_depth

    def register_completion(self, repl_id: str) -> None:
        """Mark a REPL as completed/terminated.

        Args:
            repl_id: Identifier of the completed REPL.
        """
        with self._lock:
            self._active[repl_id] = False

    def get_depth(self, repl_id: str) -> int:
        """Get the current depth of a REPL.

        Args:
            repl_id: Identifier of the REPL.

        Returns:
            Depth level (0 for root REPLs).
        """
        with self._lock:
            return self._depths.get(repl_id, 0)

    def get_status(self, repl_id: str = None) -> DepthStatus:
        """Get depth status.

        Args:
            repl_id: Specific REPL to check, or None for overall status.

        Returns:
            DepthStatus with current tracking information.
        """
        with self._lock:
            if repl_id:
                depth = self._depths.get(repl_id, 0)
                active = sum(
                    1 for rid, is_active in self._active.items()
                    if is_active and self._depths.get(rid, 0) > depth
                )
                return DepthStatus(
                    current_depth=depth,
                    max_depth=self.max_depth,
                    active_spawns=active,
                    can_spawn=depth < self.max_depth,
                )
            else:
                active = sum(1 for v in self._active.values() if v)
                max_current = max(self._depths.values()) if self._depths else 0
                return DepthStatus(
                    current_depth=max_current,
                    max_depth=self.max_depth,
                    active_spawns=active,
                    can_spawn=max_current < self.max_depth,
                )
