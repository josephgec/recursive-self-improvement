"""File-based experiment tracking backend.

Writes JSONL files under ``data/tracking/{run_name}/``:
  * ``metrics.jsonl``     — generation metrics
  * ``drift.jsonl``       — goal-drift measurements
  * ``constraints.jsonl`` — constraint-preservation reports
  * ``alerts.log``        — alert messages with rich formatting
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tracking.src.tracker import ExperimentTracker

logger = logging.getLogger(__name__)


def _rich_severity(severity: str) -> str:
    """Return a severity label with ANSI-style markers for log readability."""
    markers = {
        "warning": "[WARNING]",
        "critical": "[CRITICAL]",
        "halt": "[HALT]",
    }
    return markers.get(severity.lower(), f"[{severity.upper()}]")


class LocalTracker(ExperimentTracker):
    """JSONL-file-backed experiment tracker.

    Parameters
    ----------
    base_dir : str | Path
        Root directory for tracking data.  Defaults to ``data/tracking``.
    """

    def __init__(self, base_dir: str | Path = "data/tracking") -> None:
        self._base_dir = Path(base_dir)
        self._run_dir: Path | None = None
        self._run_name: str | None = None

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _ensure_dir(self) -> Path:
        if self._run_dir is None:
            raise RuntimeError("Call init_run() before logging.")
        self._run_dir.mkdir(parents=True, exist_ok=True)
        return self._run_dir

    def _append_jsonl(self, filename: str, record: dict[str, Any]) -> None:
        path = self._ensure_dir() / filename
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def _append_log(self, filename: str, line: str) -> None:
        path = self._ensure_dir() / filename
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # -----------------------------------------------------------------
    # ExperimentTracker interface
    # -----------------------------------------------------------------

    def init_run(
        self,
        run_name: str,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        self._run_name = run_name
        self._run_dir = self._base_dir / run_name
        self._run_dir.mkdir(parents=True, exist_ok=True)
        # Write run metadata
        meta = {
            "run_name": run_name,
            "config": config or {},
            "tags": tags or [],
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_path = self._run_dir / "run_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
        logger.info("LocalTracker: run '%s' initialised at %s", run_name, self._run_dir)

    def log_generation(self, generation: int, metrics: dict[str, Any]) -> None:
        record = {
            "generation": generation,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **metrics,
        }
        self._append_jsonl("metrics.jsonl", record)

    def log_drift(self, generation: int, drift: dict[str, Any]) -> None:
        record = {
            "generation": generation,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **drift,
        }
        self._append_jsonl("drift.jsonl", record)

    def log_constraints(self, generation: int, report: dict[str, Any]) -> None:
        record = {
            "generation": generation,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **report,
        }
        self._append_jsonl("constraints.jsonl", record)

    def log_alert(self, alert: dict[str, Any]) -> None:
        self._append_jsonl("alerts.jsonl", alert)
        # Also write human-readable line to alerts.log
        severity = alert.get("severity", "unknown")
        metric = alert.get("metric", "?")
        value = alert.get("value", "?")
        threshold = alert.get("threshold", "?")
        gen = alert.get("generation", "?")
        message = alert.get("message", "")
        ts = alert.get("timestamp", datetime.now(timezone.utc).isoformat())
        line = (
            f"{ts} {_rich_severity(severity)} gen={gen} "
            f"metric={metric} value={value} threshold={threshold} "
            f"| {message}"
        )
        self._append_log("alerts.log", line)

    def finish(self) -> None:
        if self._run_dir is not None:
            finish_record = {
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
            finish_path = self._run_dir / "run_finished.json"
            with open(finish_path, "w", encoding="utf-8") as f:
                json.dump(finish_record, f, indent=2, default=str)
            logger.info("LocalTracker: run '%s' finished.", self._run_name)
        self._run_dir = None
        self._run_name = None
