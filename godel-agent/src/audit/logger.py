"""Audit logging for tracking all agent activities."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class AuditLogger:
    """Logs all agent activities to JSON files for safety review."""

    def __init__(
        self,
        log_dir: str = "data/audit_logs",
        log_diffs: bool = True,
        log_reasoning: bool = True,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_diffs = log_diffs
        self.log_reasoning = log_reasoning
        self._entries: list[dict[str, Any]] = []
        self._run_id = str(int(time.time()))

        # Create log directory
        self._run_dir = self.log_dir / self._run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)

        # Symlink latest
        latest = self.log_dir / "latest"
        if latest.is_symlink():
            latest.unlink()
        try:
            latest.symlink_to(self._run_dir.resolve())
        except OSError:
            pass

    def log_iteration(
        self,
        iteration: int,
        accuracy: float,
        total_tasks: int,
        correct_tasks: int,
    ) -> None:
        """Log an iteration result."""
        entry = {
            "type": "iteration",
            "timestamp": time.time(),
            "iteration": iteration,
            "accuracy": accuracy,
            "total_tasks": total_tasks,
            "correct_tasks": correct_tasks,
        }
        self._entries.append(entry)
        self._write_entry(entry)

    def log_deliberation(
        self,
        iteration: int,
        deliberation_data: dict[str, Any],
    ) -> None:
        """Log a deliberation event."""
        entry = {
            "type": "deliberation",
            "timestamp": time.time(),
            "iteration": iteration,
            "data": deliberation_data,
        }
        self._entries.append(entry)
        self._write_entry(entry)

    def log_modification(
        self,
        iteration: int,
        proposal_data: dict[str, Any],
        accepted: bool = True,
    ) -> None:
        """Log a modification event."""
        entry = {
            "type": "modification",
            "timestamp": time.time(),
            "iteration": iteration,
            "proposal": proposal_data,
            "accepted": accepted,
        }
        self._entries.append(entry)
        self._write_entry(entry)

    def log_rollback(self, iteration: int, reason: str) -> None:
        """Log a rollback event."""
        entry = {
            "type": "rollback",
            "timestamp": time.time(),
            "iteration": iteration,
            "reason": reason,
        }
        self._entries.append(entry)
        self._write_entry(entry)

    def get_modification_history(self) -> list[dict[str, Any]]:
        """Get all modification-related log entries."""
        return [
            e for e in self._entries
            if e["type"] in ("modification", "rollback", "deliberation")
        ]

    def export_for_safety_review(self) -> dict[str, Any]:
        """Export all entries in a format suitable for safety review."""
        modifications = [e for e in self._entries if e["type"] == "modification"]
        rollbacks = [e for e in self._entries if e["type"] == "rollback"]
        iterations = [e for e in self._entries if e["type"] == "iteration"]

        return {
            "run_id": self._run_id,
            "total_entries": len(self._entries),
            "total_iterations": len(iterations),
            "total_modifications": len(modifications),
            "total_rollbacks": len(rollbacks),
            "modifications_accepted": sum(1 for m in modifications if m.get("accepted", False)),
            "entries": self._entries,
        }

    def _write_entry(self, entry: dict[str, Any]) -> None:
        """Write a single entry to the log file."""
        log_file = self._run_dir / "audit_log.jsonl"
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError:
            pass

    @property
    def entries(self) -> list[dict[str, Any]]:
        return list(self._entries)

    @property
    def run_dir(self) -> Path:
        return self._run_dir
