"""Tests for analysis modules: complexity_tracker, convergence, modification_history, report."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch, MagicMock

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.analysis.complexity_tracker import ComplexityTracker
from src.analysis.convergence import ConvergenceDetector
from src.analysis.modification_history import ModificationHistoryAnalyzer
from src.analysis.report import generate_report
from src.core.state import AgentState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_states(n: int = 5, with_code: bool = False) -> list[AgentState]:
    """Create a list of synthetic AgentState objects."""
    states: list[AgentState] = []
    code = (
        "def choose(task):\n"
        "    if task.domain == 'math':\n"
        "        return 'cot'\n"
        "    return 'direct'\n"
    )
    for i in range(n):
        s = AgentState(
            iteration=i,
            accuracy_history=[0.5 + 0.05 * j for j in range(i + 1)],
            modifications_applied=[{"target": "prompt_strategy"}] * i,
        )
        if with_code:
            # Make code grow across iterations to test explosion
            s.few_shot_selector_code = code * (i + 1)
            s.reasoning_strategy_code = code
        states.append(s)
    return states


def _make_audit_entries() -> list[dict[str, Any]]:
    """Create synthetic audit log entries."""
    entries: list[dict[str, Any]] = []
    for i in range(5):
        entries.append({
            "type": "iteration",
            "iteration": i,
            "accuracy": 0.5 + 0.05 * i,
        })
    entries.append({
        "type": "modification",
        "iteration": 2,
        "proposal": {"target": "prompt_strategy", "description": "Better prompts"},
        "accepted": True,
    })
    entries.append({
        "type": "modification",
        "iteration": 3,
        "proposal": {"target": "reasoning_strategy", "description": "Switch mode"},
        "accepted": False,
    })
    entries.append({
        "type": "rollback",
        "iteration": 3,
        "reason": "Validation failed after strategy change",
    })
    return entries


# ===================================================================
# ComplexityTracker
# ===================================================================

class TestComplexityTracker:
    def test_track_returns_dataframe(self) -> None:
        tracker = ComplexityTracker()
        states = _make_states(3, with_code=True)
        df = tracker.track(states)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "iteration" in df.columns
        assert "num_modifications" in df.columns

    def test_track_without_code(self) -> None:
        tracker = ComplexityTracker()
        states = _make_states(3, with_code=False)
        df = tracker.track(states)
        assert len(df) == 3
        # No code columns should be present since code is empty
        complexity_cols = [c for c in df.columns if c.endswith("_nodes")]
        assert len(complexity_cols) == 0

    def test_track_with_code_adds_complexity_columns(self) -> None:
        tracker = ComplexityTracker()
        states = _make_states(3, with_code=True)
        df = tracker.track(states)
        assert "few_shot_selector_code_nodes" in df.columns
        assert "few_shot_selector_code_cyclomatic" in df.columns
        assert "few_shot_selector_code_nesting" in df.columns
        assert "few_shot_selector_code_loc" in df.columns

    def test_detect_complexity_explosion_false_with_few_states(self) -> None:
        tracker = ComplexityTracker()
        states = _make_states(1, with_code=True)
        assert tracker.detect_complexity_explosion(states) is False

    def test_detect_complexity_explosion_false_for_stable_code(self) -> None:
        tracker = ComplexityTracker()
        code = "def f():\n    return 1\n"
        states = [
            AgentState(iteration=i, few_shot_selector_code=code)
            for i in range(3)
        ]
        assert tracker.detect_complexity_explosion(states) is False

    def test_detect_complexity_explosion_true_for_growing_code(self) -> None:
        tracker = ComplexityTracker()
        code = "def f():\n    return 1\n"
        states = [
            AgentState(iteration=0, few_shot_selector_code=code),
            AgentState(iteration=1, few_shot_selector_code=code * 10),
        ]
        # 10x growth > 5x threshold
        result = tracker.detect_complexity_explosion(states, threshold_ratio=5.0)
        assert result is True

    def test_detect_complexity_explosion_with_custom_threshold(self) -> None:
        tracker = ComplexityTracker()
        code = "def f():\n    return 1\n"
        states = [
            AgentState(iteration=0, few_shot_selector_code=code),
            AgentState(iteration=1, few_shot_selector_code=code * 3),
        ]
        assert tracker.detect_complexity_explosion(states, threshold_ratio=2.0) is True
        assert tracker.detect_complexity_explosion(states, threshold_ratio=50.0) is False

    def test_plot_complexity_vs_performance(self, tmp_path: Any) -> None:
        tracker = ComplexityTracker()
        states = _make_states(4, with_code=True)
        fig = tracker.plot_complexity_vs_performance(states)
        assert fig is not None
        plt.close(fig)

    def test_plot_saves_to_file(self, tmp_path: Any) -> None:
        tracker = ComplexityTracker()
        states = _make_states(4, with_code=True)
        out = str(tmp_path / "complexity.png")
        fig = tracker.plot_complexity_vs_performance(states, output_path=out)
        assert fig is not None
        assert os.path.exists(out)
        plt.close(fig)

    def test_plot_with_empty_accuracy(self, tmp_path: Any) -> None:
        tracker = ComplexityTracker()
        states = [AgentState(iteration=i) for i in range(3)]
        fig = tracker.plot_complexity_vs_performance(states)
        assert fig is not None
        plt.close(fig)


# ===================================================================
# ConvergenceDetector
# ===================================================================

class TestConvergenceDetector:
    def test_not_stagnant_with_too_few_points(self) -> None:
        cd = ConvergenceDetector(window=5)
        assert cd.is_stagnant([0.5, 0.6]) is False

    def test_stagnant_when_flat(self) -> None:
        cd = ConvergenceDetector(window=5, stagnation_threshold=0.01)
        assert cd.is_stagnant([0.7, 0.7, 0.7, 0.7, 0.7]) is True

    def test_not_stagnant_when_varying(self) -> None:
        cd = ConvergenceDetector(window=5, stagnation_threshold=0.01)
        assert cd.is_stagnant([0.5, 0.6, 0.55, 0.65, 0.7]) is False

    def test_stagnant_uses_last_window(self) -> None:
        cd = ConvergenceDetector(window=3, stagnation_threshold=0.02)
        history = [0.1, 0.2, 0.5, 0.5, 0.5]  # last 3 are flat
        assert cd.is_stagnant(history) is True

    def test_is_diverging_with_negative_trend(self) -> None:
        cd = ConvergenceDetector(window=5)
        history = [0.8, 0.75, 0.7, 0.65, 0.6]
        assert cd.is_diverging(history) is True

    def test_not_diverging_with_positive_trend(self) -> None:
        cd = ConvergenceDetector(window=5)
        history = [0.5, 0.55, 0.6, 0.65, 0.7]
        assert cd.is_diverging(history) is False

    def test_not_diverging_with_too_few_points(self) -> None:
        cd = ConvergenceDetector(window=5)
        assert cd.is_diverging([0.5, 0.4]) is False

    def test_compute_trend_positive(self) -> None:
        cd = ConvergenceDetector(window=5)
        history = [0.5, 0.55, 0.6, 0.65, 0.7]
        trend = cd.compute_trend(history)
        assert trend > 0

    def test_compute_trend_negative(self) -> None:
        cd = ConvergenceDetector(window=5)
        history = [0.8, 0.7, 0.6, 0.5, 0.4]
        trend = cd.compute_trend(history)
        assert trend < 0

    def test_compute_trend_empty(self) -> None:
        cd = ConvergenceDetector()
        assert cd.compute_trend([]) == 0.0

    def test_compute_trend_single_point(self) -> None:
        cd = ConvergenceDetector()
        assert cd.compute_trend([0.5]) == 0.0

    def test_compute_trend_custom_n(self) -> None:
        cd = ConvergenceDetector(window=5)
        history = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        trend_3 = cd.compute_trend(history, n=3)
        assert trend_3 > 0

    def test_should_modify_false_with_few_points(self) -> None:
        cd = ConvergenceDetector(window=5)
        assert cd.should_modify([0.5, 0.6]) is False

    def test_should_modify_true_when_stagnant(self) -> None:
        cd = ConvergenceDetector(window=5, stagnation_threshold=0.01)
        assert cd.should_modify([0.7, 0.7, 0.7, 0.7, 0.7]) is True

    def test_should_modify_true_when_diverging(self) -> None:
        cd = ConvergenceDetector(window=5)
        assert cd.should_modify([0.8, 0.7, 0.6, 0.5, 0.4]) is True

    def test_should_modify_false_when_improving(self) -> None:
        cd = ConvergenceDetector(window=5, stagnation_threshold=0.01)
        assert cd.should_modify([0.5, 0.55, 0.6, 0.65, 0.7]) is False


# ===================================================================
# ModificationHistoryAnalyzer
# ===================================================================

class TestModificationHistoryAnalyzer:
    def test_init_without_entries(self) -> None:
        analyzer = ModificationHistoryAnalyzer()
        assert analyzer.success_rate_by_component() == {}

    def test_load_entries(self) -> None:
        analyzer = ModificationHistoryAnalyzer()
        entries = _make_audit_entries()
        analyzer.load_entries(entries)
        rates = analyzer.success_rate_by_component()
        assert "prompt_strategy" in rates

    def test_success_rate_by_component(self) -> None:
        entries = _make_audit_entries()
        analyzer = ModificationHistoryAnalyzer(entries)
        rates = analyzer.success_rate_by_component()
        assert rates["prompt_strategy"] == 1.0  # 1 accepted out of 1
        assert rates["reasoning_strategy"] == 0.0  # 0 accepted out of 1

    def test_success_rate_ignores_non_modification_entries(self) -> None:
        entries = [
            {"type": "iteration", "iteration": 0, "accuracy": 0.5},
            {"type": "modification", "proposal": {"target": "x"}, "accepted": True},
        ]
        analyzer = ModificationHistoryAnalyzer(entries)
        rates = analyzer.success_rate_by_component()
        assert "x" in rates
        assert rates["x"] == 1.0

    def test_convergence_analysis(self) -> None:
        entries = _make_audit_entries()
        analyzer = ModificationHistoryAnalyzer(entries)
        result = analyzer.convergence_analysis()

        assert result["total_iterations"] == 5
        assert result["total_modifications"] == 2
        assert result["total_rollbacks"] == 1
        assert result["acceptance_rate"] == 0.5  # (2-1)/2
        assert len(result["accuracy_trajectory"]) == 5
        assert result["final_accuracy"] == pytest.approx(0.7)
        assert result["best_accuracy"] == pytest.approx(0.7)

    def test_convergence_analysis_empty(self) -> None:
        analyzer = ModificationHistoryAnalyzer([])
        result = analyzer.convergence_analysis()
        assert result["total_iterations"] == 0
        assert result["total_modifications"] == 0
        assert result["final_accuracy"] == 0.0
        assert result["best_accuracy"] == 0.0

    def test_plot_modification_timeline(self) -> None:
        entries = _make_audit_entries()
        analyzer = ModificationHistoryAnalyzer(entries)
        fig = analyzer.plot_modification_timeline()
        assert fig is not None
        plt.close(fig)

    def test_plot_timeline_saves_to_file(self, tmp_path: Any) -> None:
        entries = _make_audit_entries()
        analyzer = ModificationHistoryAnalyzer(entries)
        out = str(tmp_path / "timeline.png")
        fig = analyzer.plot_modification_timeline(output_path=out)
        assert fig is not None
        assert os.path.exists(out)
        plt.close(fig)

    def test_plot_timeline_empty_entries(self) -> None:
        analyzer = ModificationHistoryAnalyzer([])
        fig = analyzer.plot_modification_timeline()
        assert fig is not None
        plt.close(fig)

    def test_to_dataframe(self) -> None:
        entries = _make_audit_entries()
        analyzer = ModificationHistoryAnalyzer(entries)
        df = analyzer.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(entries)

    def test_to_dataframe_empty(self) -> None:
        analyzer = ModificationHistoryAnalyzer([])
        df = analyzer.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ===================================================================
# Report generation
# ===================================================================

class TestGenerateReport:
    def test_basic_report(self) -> None:
        entries = _make_audit_entries()
        report = generate_report(entries)
        assert "# Godel Agent Run Report" in report
        assert "## Summary" in report
        assert "Total iterations: 5" in report
        assert "Total modifications: 2" in report
        assert "Total rollbacks: 1" in report

    def test_report_contains_performance_trajectory(self) -> None:
        entries = _make_audit_entries()
        report = generate_report(entries)
        assert "## Performance Trajectory" in report
        assert "Iteration" in report
        assert "Accuracy" in report

    def test_report_contains_modification_timeline(self) -> None:
        entries = _make_audit_entries()
        report = generate_report(entries)
        assert "## Modification Timeline" in report
        assert "prompt_strategy" in report

    def test_report_contains_success_rates(self) -> None:
        entries = _make_audit_entries()
        report = generate_report(entries)
        assert "## Success Rate by Component" in report

    def test_report_contains_rollbacks(self) -> None:
        entries = _make_audit_entries()
        report = generate_report(entries)
        assert "## Rollbacks" in report
        assert "Validation failed" in report

    def test_report_empty_entries(self) -> None:
        report = generate_report([])
        assert "# Godel Agent Run Report" in report
        assert "Total iterations: 0" in report
        assert "No modifications applied." in report

    def test_report_saves_to_file(self, tmp_path: Any) -> None:
        entries = _make_audit_entries()
        out = str(tmp_path / "report.md")
        report = generate_report(entries, output_path=out)
        assert os.path.exists(out)
        with open(out) as f:
            content = f.read()
        assert content == report

    def test_report_no_rollbacks(self) -> None:
        entries = [
            {"type": "iteration", "iteration": 0, "accuracy": 0.8},
            {"type": "modification", "iteration": 1, "proposal": {"target": "x", "description": "y"}, "accepted": True},
        ]
        report = generate_report(entries)
        assert "## Rollbacks" not in report
