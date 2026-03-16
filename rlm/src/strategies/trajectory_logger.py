"""TrajectoryLogger: record and export session trajectories."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from src.core.session import TrajectoryStep, SessionResult


@dataclass
class TrajectoryRecord:
    """A complete trajectory record for export."""
    session_id: Optional[str]
    depth: int
    total_iterations: int
    forced_final: bool
    elapsed_time: float
    steps: List[Dict[str, Any]]
    result: Optional[str]
    timestamp: float = field(default_factory=time.time)


class TrajectoryLogger:
    """Log and export session trajectories."""

    def __init__(self) -> None:
        self._records: List[TrajectoryRecord] = []

    def log_session(self, session_result: SessionResult) -> TrajectoryRecord:
        """Log a completed session result."""
        steps = []
        for step in session_result.trajectory:
            steps.append({
                "iteration": step.iteration,
                "code_blocks": step.code_blocks,
                "has_final": step.has_final,
                "execution_results": [
                    {
                        "code": er.code,
                        "stdout": er.stdout,
                        "success": er.success,
                        "exception": er.exception,
                    }
                    for er in step.execution_results
                ],
            })

        record = TrajectoryRecord(
            session_id=session_result.session_id,
            depth=session_result.depth,
            total_iterations=session_result.total_iterations,
            forced_final=session_result.forced_final,
            elapsed_time=session_result.elapsed_time,
            steps=steps,
            result=str(session_result.result) if session_result.result else None,
        )
        self._records.append(record)
        return record

    def export_trajectory(self, index: int = -1) -> Dict[str, Any]:
        """Export a single trajectory as a dict."""
        if not self._records:
            return {}
        record = self._records[index]
        return {
            "session_id": record.session_id,
            "depth": record.depth,
            "total_iterations": record.total_iterations,
            "forced_final": record.forced_final,
            "elapsed_time": record.elapsed_time,
            "steps": record.steps,
            "result": record.result,
            "timestamp": record.timestamp,
        }

    def export_all(self) -> List[Dict[str, Any]]:
        """Export all recorded trajectories."""
        return [self.export_trajectory(i) for i in range(len(self._records))]

    @property
    def records(self) -> List[TrajectoryRecord]:
        return list(self._records)
