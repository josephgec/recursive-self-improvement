"""Audit bridge: logs modifications and rollbacks for audit trail."""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional


class AuditBridge:
    """Maintains an audit trail of all modifications and rollbacks."""

    def __init__(self):
        self._modifications: List[Dict[str, Any]] = []
        self._rollbacks: List[Dict[str, Any]] = []

    def log_modification(
        self,
        candidate_id: str,
        target: str,
        old_code: str,
        new_code: str,
        result: str = "applied",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a modification event."""
        entry = {
            "timestamp": time.time(),
            "type": "modification",
            "candidate_id": candidate_id,
            "target": target,
            "old_code_length": len(old_code),
            "new_code_length": len(new_code),
            "result": result,
            "metadata": metadata or {},
        }
        self._modifications.append(entry)

    def log_rollback(
        self,
        reason: str,
        iteration: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a rollback event."""
        entry = {
            "timestamp": time.time(),
            "type": "rollback",
            "reason": reason,
            "iteration": iteration,
            "metadata": metadata or {},
        }
        self._rollbacks.append(entry)

    def export_history(self) -> Dict[str, Any]:
        """Export the complete audit history."""
        return {
            "modifications": list(self._modifications),
            "rollbacks": list(self._rollbacks),
            "total_modifications": len(self._modifications),
            "total_rollbacks": len(self._rollbacks),
        }

    def export_json(self) -> str:
        """Export audit history as JSON."""
        return json.dumps(self.export_history(), indent=2)

    @property
    def modification_count(self) -> int:
        return len(self._modifications)

    @property
    def rollback_count(self) -> int:
        return len(self._rollbacks)

    @property
    def modifications(self) -> List[Dict[str, Any]]:
        return list(self._modifications)

    @property
    def rollbacks(self) -> List[Dict[str, Any]]:
        return list(self._rollbacks)
