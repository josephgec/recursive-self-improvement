"""Tests for AuditLogger."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.audit.logger import AuditLogger


@pytest.fixture
def audit_logger(tmp_dir: Path) -> AuditLogger:
    return AuditLogger(log_dir=str(tmp_dir / "audit_logs"))


class TestLogIteration:
    def test_log_iteration(self, audit_logger: AuditLogger) -> None:
        audit_logger.log_iteration(0, 0.75, 10, 7)
        entries = audit_logger.entries
        assert len(entries) == 1
        assert entries[0]["type"] == "iteration"
        assert entries[0]["accuracy"] == 0.75
        assert entries[0]["total_tasks"] == 10
        assert entries[0]["correct_tasks"] == 7

    def test_log_multiple_iterations(self, audit_logger: AuditLogger) -> None:
        for i in range(5):
            audit_logger.log_iteration(i, 0.5 + i * 0.05, 10, 5 + i)
        assert len(audit_logger.entries) == 5


class TestLogModification:
    def test_log_modification_accepted(self, audit_logger: AuditLogger) -> None:
        audit_logger.log_modification(
            3,
            {"target": "prompt_strategy", "description": "Improve prompt"},
            accepted=True,
        )
        entries = audit_logger.entries
        assert len(entries) == 1
        assert entries[0]["type"] == "modification"
        assert entries[0]["accepted"] is True
        assert entries[0]["proposal"]["target"] == "prompt_strategy"

    def test_log_modification_rejected(self, audit_logger: AuditLogger) -> None:
        audit_logger.log_modification(
            3,
            {"target": "reasoning_strategy"},
            accepted=False,
        )
        assert audit_logger.entries[0]["accepted"] is False


class TestLogRollback:
    def test_log_rollback(self, audit_logger: AuditLogger) -> None:
        audit_logger.log_rollback(5, "Validation failed")
        entries = audit_logger.entries
        assert len(entries) == 1
        assert entries[0]["type"] == "rollback"
        assert entries[0]["reason"] == "Validation failed"
        assert entries[0]["iteration"] == 5


class TestLogDeliberation:
    def test_log_deliberation(self, audit_logger: AuditLogger) -> None:
        audit_logger.log_deliberation(
            7,
            {"should_proceed": True, "risk": "low"},
        )
        entries = audit_logger.entries
        assert len(entries) == 1
        assert entries[0]["type"] == "deliberation"
        assert entries[0]["data"]["should_proceed"] is True


class TestGetModificationHistory:
    def test_filters_only_mod_events(self, audit_logger: AuditLogger) -> None:
        audit_logger.log_iteration(0, 0.5, 10, 5)
        audit_logger.log_deliberation(1, {"should_proceed": True})
        audit_logger.log_modification(1, {"target": "prompt_strategy"}, accepted=True)
        audit_logger.log_iteration(1, 0.6, 10, 6)
        audit_logger.log_rollback(2, "Failed validation")

        history = audit_logger.get_modification_history()
        types = [e["type"] for e in history]
        assert "iteration" not in types
        assert "deliberation" in types
        assert "modification" in types
        assert "rollback" in types


class TestExportForSafetyReview:
    def test_export_structure(self, audit_logger: AuditLogger) -> None:
        audit_logger.log_iteration(0, 0.5, 10, 5)
        audit_logger.log_modification(1, {"target": "prompt_strategy"}, accepted=True)
        audit_logger.log_rollback(2, "Failed")

        export = audit_logger.export_for_safety_review()
        assert "run_id" in export
        assert "total_entries" in export
        assert export["total_entries"] == 3
        assert export["total_iterations"] == 1
        assert export["total_modifications"] == 1
        assert export["total_rollbacks"] == 1
        assert export["modifications_accepted"] == 1
        assert "entries" in export


class TestWritesToDisk:
    def test_entries_written_to_jsonl(self, audit_logger: AuditLogger) -> None:
        audit_logger.log_iteration(0, 0.5, 10, 5)
        audit_logger.log_modification(1, {"target": "test"}, accepted=True)

        log_file = audit_logger.run_dir / "audit_log.jsonl"
        assert log_file.exists()

        with open(log_file) as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        assert len(lines) == 2

        entry = json.loads(lines[0])
        assert entry["type"] == "iteration"
