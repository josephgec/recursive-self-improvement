"""Tests for experiment tracker backends (LocalTracker, WandBTracker)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tracking.src.local_backend import LocalTracker
from tracking.src.wandb_backend import WandBTracker


# ---------------------------------------------------------------------------
# LocalTracker tests
# ---------------------------------------------------------------------------

class TestLocalTracker:
    """LocalTracker writes JSONL files to disk."""

    def test_init_run_creates_directory(self, tmp_path: Path) -> None:
        tracker = LocalTracker(base_dir=tmp_path)
        tracker.init_run("test_run")
        assert (tmp_path / "test_run").is_dir()
        assert (tmp_path / "test_run" / "run_meta.json").exists()

    def test_log_three_generations(self, tmp_path: Path) -> None:
        tracker = LocalTracker(base_dir=tmp_path)
        tracker.init_run("gen_test")

        for gen in range(3):
            tracker.log_generation(gen, {"loss": 1.0 - gen * 0.1, "accuracy": 0.5 + gen * 0.1})

        metrics_path = tmp_path / "gen_test" / "metrics.jsonl"
        assert metrics_path.exists()

        lines = metrics_path.read_text().strip().split("\n")
        assert len(lines) == 3

        records = [json.loads(line) for line in lines]
        assert records[0]["generation"] == 0
        assert records[1]["generation"] == 1
        assert records[2]["generation"] == 2
        assert records[0]["loss"] == pytest.approx(1.0)
        assert records[2]["accuracy"] == pytest.approx(0.7)

    def test_log_drift(self, tmp_path: Path) -> None:
        tracker = LocalTracker(base_dir=tmp_path)
        tracker.init_run("drift_test")
        tracker.log_drift(1, {"goal_drift_index": 0.05, "semantic_drift": 0.02})

        drift_path = tmp_path / "drift_test" / "drift.jsonl"
        assert drift_path.exists()
        record = json.loads(drift_path.read_text().strip())
        assert record["goal_drift_index"] == 0.05

    def test_log_constraints(self, tmp_path: Path) -> None:
        tracker = LocalTracker(base_dir=tmp_path)
        tracker.init_run("constraint_test")
        tracker.log_constraints(1, {"all_passed": True, "recommendation": "proceed"})

        path = tmp_path / "constraint_test" / "constraints.jsonl"
        assert path.exists()
        record = json.loads(path.read_text().strip())
        assert record["all_passed"] is True

    def test_log_alert(self, tmp_path: Path) -> None:
        tracker = LocalTracker(base_dir=tmp_path)
        tracker.init_run("alert_test")
        tracker.log_alert({
            "severity": "warning",
            "metric": "gdi",
            "value": 0.2,
            "threshold": 0.15,
            "generation": 3,
            "message": "GDI exceeded",
        })

        # Check JSONL file
        jsonl_path = tmp_path / "alert_test" / "alerts.jsonl"
        assert jsonl_path.exists()
        record = json.loads(jsonl_path.read_text().strip())
        assert record["severity"] == "warning"

        # Check human-readable log
        log_path = tmp_path / "alert_test" / "alerts.log"
        assert log_path.exists()
        content = log_path.read_text()
        assert "[WARNING]" in content
        assert "gdi" in content

    def test_finish_creates_marker(self, tmp_path: Path) -> None:
        tracker = LocalTracker(base_dir=tmp_path)
        tracker.init_run("finish_test")
        tracker.finish()
        assert (tmp_path / "finish_test" / "run_finished.json").exists()

    def test_log_without_init_raises(self, tmp_path: Path) -> None:
        tracker = LocalTracker(base_dir=tmp_path)
        with pytest.raises(RuntimeError, match="init_run"):
            tracker.log_generation(0, {"loss": 1.0})

    def test_full_lifecycle(self, tmp_path: Path) -> None:
        """Full init → log × 3 → finish lifecycle."""
        tracker = LocalTracker(base_dir=tmp_path)
        tracker.init_run("lifecycle", config={"lr": 1e-4}, tags=["test"])

        for gen in range(3):
            tracker.log_generation(gen, {"loss": 1.0 / (gen + 1)})
            tracker.log_drift(gen, {"gdi": gen * 0.01})
            tracker.log_constraints(gen, {"all_passed": True})

        tracker.finish()

        run_dir = tmp_path / "lifecycle"
        assert run_dir.is_dir()
        assert len((run_dir / "metrics.jsonl").read_text().strip().split("\n")) == 3
        assert len((run_dir / "drift.jsonl").read_text().strip().split("\n")) == 3
        assert len((run_dir / "constraints.jsonl").read_text().strip().split("\n")) == 3
        assert (run_dir / "run_meta.json").exists()
        assert (run_dir / "run_finished.json").exists()


# ---------------------------------------------------------------------------
# WandBTracker tests (mocked)
# ---------------------------------------------------------------------------

class TestWandBTracker:
    """WandBTracker wraps wandb calls — all tested via mocks."""

    @patch("tracking.src.wandb_backend._WANDB_AVAILABLE", True)
    @patch("tracking.src.wandb_backend.wandb")
    def test_init_run_calls_wandb_init(self, mock_wandb: MagicMock) -> None:
        tracker = WandBTracker(project="test-proj", entity="test-entity")
        tracker.init_run("run_1", config={"lr": 1e-4}, tags=["test"])

        mock_wandb.init.assert_called_once_with(
            project="test-proj",
            entity="test-entity",
            name="run_1",
            config={"lr": 1e-4},
            tags=["test"],
        )

    @patch("tracking.src.wandb_backend._WANDB_AVAILABLE", True)
    @patch("tracking.src.wandb_backend.wandb")
    def test_log_generation_namespaces(self, mock_wandb: MagicMock) -> None:
        tracker = WandBTracker()
        mock_wandb.init.return_value = MagicMock()
        tracker.init_run("run_2")
        tracker.log_generation(1, {"loss": 0.5, "accuracy": 0.8})

        mock_wandb.log.assert_called_once()
        call_args = mock_wandb.log.call_args
        logged = call_args[0][0]
        assert logged["training/loss"] == 0.5
        assert logged["training/accuracy"] == 0.8
        assert logged["generation"] == 1
        assert call_args[1]["step"] == 1

    @patch("tracking.src.wandb_backend._WANDB_AVAILABLE", True)
    @patch("tracking.src.wandb_backend.wandb")
    def test_log_drift_namespaces(self, mock_wandb: MagicMock) -> None:
        tracker = WandBTracker()
        mock_wandb.init.return_value = MagicMock()
        tracker.init_run("run_3")
        tracker.log_drift(2, {"goal_drift_index": 0.1})

        logged = mock_wandb.log.call_args[0][0]
        assert logged["safety/goal_drift_index"] == 0.1

    @patch("tracking.src.wandb_backend._WANDB_AVAILABLE", True)
    @patch("tracking.src.wandb_backend.wandb")
    def test_log_alert_calls_wandb_alert(self, mock_wandb: MagicMock) -> None:
        mock_wandb.AlertLevel = MagicMock()
        mock_wandb.AlertLevel.WARN = "WARN"
        mock_wandb.AlertLevel.ERROR = "ERROR"

        tracker = WandBTracker()
        mock_wandb.init.return_value = MagicMock()
        tracker.init_run("run_4")
        tracker.log_alert({
            "severity": "WARNING",
            "metric": "gdi",
            "message": "Drift exceeded",
        })
        mock_wandb.alert.assert_called_once()

    @patch("tracking.src.wandb_backend._WANDB_AVAILABLE", True)
    @patch("tracking.src.wandb_backend.wandb")
    def test_finish_calls_wandb_finish(self, mock_wandb: MagicMock) -> None:
        tracker = WandBTracker()
        mock_wandb.init.return_value = MagicMock()
        tracker.init_run("run_5")
        tracker.finish()
        mock_wandb.finish.assert_called_once()

    @patch("tracking.src.wandb_backend._WANDB_AVAILABLE", False)
    def test_noop_when_wandb_unavailable(self) -> None:
        """When wandb is not installed, all methods are no-ops."""
        tracker = WandBTracker()
        # None of these should raise
        tracker.init_run("noop")
        tracker.log_generation(0, {"loss": 1.0})
        tracker.log_drift(0, {"gdi": 0.0})
        tracker.log_constraints(0, {"all_passed": True})
        tracker.log_alert({"severity": "warning", "metric": "x", "message": "y"})
        tracker.finish()
