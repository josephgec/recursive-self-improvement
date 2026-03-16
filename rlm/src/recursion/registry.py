"""Session registry: track active sessions and their hierarchy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SessionInfo:
    """Information about a registered session."""
    session_id: str
    depth: int
    parent_id: Optional[str]
    query: str
    status: str = "active"  # active, completed, failed
    result: Any = None
    children: List[str] = field(default_factory=list)


class SessionRegistry:
    """Global registry of active and completed sessions."""

    def __init__(self) -> None:
        self._sessions: Dict[str, SessionInfo] = {}
        self._counter = 0

    def register(
        self,
        query: str,
        depth: int = 0,
        parent_id: Optional[str] = None,
    ) -> str:
        """Register a new session and return its ID."""
        self._counter += 1
        session_id = f"session_{self._counter}"
        info = SessionInfo(
            session_id=session_id,
            depth=depth,
            parent_id=parent_id,
            query=query,
        )
        self._sessions[session_id] = info
        if parent_id and parent_id in self._sessions:
            self._sessions[parent_id].children.append(session_id)
        return session_id

    def get(self, session_id: str) -> Optional[SessionInfo]:
        """Return session info, or None if not found."""
        return self._sessions.get(session_id)

    def update_status(self, session_id: str, status: str, result: Any = None) -> None:
        """Update the status (and optionally result) of a session."""
        if session_id in self._sessions:
            self._sessions[session_id].status = status
            if result is not None:
                self._sessions[session_id].result = result

    def list_active(self) -> List[SessionInfo]:
        """Return all sessions with status 'active'."""
        return [s for s in self._sessions.values() if s.status == "active"]

    def get_tree(self, root_id: Optional[str] = None) -> Dict[str, Any]:
        """Return a nested dict representing the session tree.

        If *root_id* is None, returns trees for all root sessions.
        """
        if root_id is not None:
            return self._build_tree(root_id)
        # All root sessions (no parent)
        roots = [s for s in self._sessions.values() if s.parent_id is None]
        return {r.session_id: self._build_tree(r.session_id) for r in roots}

    def _build_tree(self, session_id: str) -> Dict[str, Any]:
        info = self._sessions.get(session_id)
        if info is None:
            return {}
        return {
            "id": info.session_id,
            "depth": info.depth,
            "query": info.query,
            "status": info.status,
            "children": [self._build_tree(c) for c in info.children],
        }
