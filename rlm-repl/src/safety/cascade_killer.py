"""Cascade killing for REPL process trees."""

import threading
from typing import Callable, Dict, List, Optional, Set

from src.interface.errors import CascadeKillError


class CascadeKiller:
    """Manages parent-child REPL relationships for cascade killing.

    When a parent REPL is killed, all its descendants are also killed.
    Maintains a tree structure of REPL relationships.
    """

    def __init__(self):
        self._children: Dict[str, List[str]] = {}
        self._parent: Dict[str, Optional[str]] = {}
        self._kill_callbacks: Dict[str, Callable] = {}
        self._alive: Dict[str, bool] = {}
        self._lock = threading.Lock()

    def register(
        self,
        repl_id: str,
        parent_id: Optional[str] = None,
        kill_callback: Optional[Callable] = None,
    ) -> None:
        """Register a REPL in the tree.

        Args:
            repl_id: Unique identifier for the REPL.
            parent_id: Parent REPL identifier, or None for root.
            kill_callback: Function to call when killing this REPL.
        """
        with self._lock:
            self._children.setdefault(repl_id, [])
            self._parent[repl_id] = parent_id
            self._alive[repl_id] = True

            if kill_callback:
                self._kill_callbacks[repl_id] = kill_callback

            if parent_id is not None:
                self._children.setdefault(parent_id, [])
                self._children[parent_id].append(repl_id)

    def kill(self, repl_id: str) -> List[str]:
        """Kill a REPL and all its descendants.

        Args:
            repl_id: Identifier of the REPL to kill.

        Returns:
            List of all REPL IDs that were killed.

        Raises:
            CascadeKillError: If killing fails.
        """
        killed = []
        with self._lock:
            if repl_id not in self._alive:
                raise CascadeKillError(repl_id, f"Unknown REPL: {repl_id}")
            self._kill_recursive(repl_id, killed)
        return killed

    def _kill_recursive(self, repl_id: str, killed: List[str]) -> None:
        """Recursively kill a REPL and descendants (must hold lock)."""
        # Kill children first (depth-first)
        for child_id in list(self._children.get(repl_id, [])):
            if self._alive.get(child_id, False):
                self._kill_recursive(child_id, killed)

        # Kill this REPL
        if self._alive.get(repl_id, False):
            callback = self._kill_callbacks.get(repl_id)
            if callback:
                try:
                    callback()
                except Exception:
                    pass
            self._alive[repl_id] = False
            killed.append(repl_id)

    def kill_subtree(self, repl_id: str) -> List[str]:
        """Kill only the descendants of a REPL, not the REPL itself.

        Args:
            repl_id: Identifier of the parent REPL.

        Returns:
            List of descendant REPL IDs that were killed.
        """
        killed = []
        with self._lock:
            for child_id in list(self._children.get(repl_id, [])):
                if self._alive.get(child_id, False):
                    self._kill_recursive(child_id, killed)
        return killed

    def get_descendants(self, repl_id: str) -> List[str]:
        """Get all descendant REPL IDs.

        Args:
            repl_id: Identifier of the root REPL.

        Returns:
            List of all descendant REPL IDs.
        """
        descendants = []
        with self._lock:
            self._collect_descendants(repl_id, descendants)
        return descendants

    def _collect_descendants(self, repl_id: str, result: List[str]) -> None:
        """Recursively collect descendants (must hold lock)."""
        for child_id in self._children.get(repl_id, []):
            result.append(child_id)
            self._collect_descendants(child_id, result)

    def get_depth(self, repl_id: str) -> int:
        """Get the depth of a REPL in the tree.

        Args:
            repl_id: Identifier of the REPL.

        Returns:
            Depth (0 for root).
        """
        with self._lock:
            depth = 0
            current = repl_id
            while self._parent.get(current) is not None:
                depth += 1
                current = self._parent[current]
            return depth

    def is_alive(self, repl_id: str) -> bool:
        """Check if a REPL is alive.

        Args:
            repl_id: Identifier of the REPL.

        Returns:
            True if the REPL is alive.
        """
        with self._lock:
            return self._alive.get(repl_id, False)

    def unregister(self, repl_id: str) -> None:
        """Remove a REPL from the tree.

        Args:
            repl_id: Identifier of the REPL to remove.
        """
        with self._lock:
            # Remove from parent's children list
            parent = self._parent.get(repl_id)
            if parent and parent in self._children:
                self._children[parent] = [
                    c for c in self._children[parent] if c != repl_id
                ]

            # Clean up
            self._children.pop(repl_id, None)
            self._parent.pop(repl_id, None)
            self._kill_callbacks.pop(repl_id, None)
            self._alive.pop(repl_id, None)
