"""Tests for DiffEngine, CodeDiff, BehavioralDiff."""

from __future__ import annotations

import pytest

from src.modification.diff_engine import DiffEngine, CodeDiff, BehavioralDiff


class TestCodeDiff:
    def test_default_values(self) -> None:
        d = CodeDiff()
        assert d.before == ""
        assert d.after == ""
        assert d.unified_diff == ""
        assert d.lines_added == 0
        assert d.lines_removed == 0
        assert d.lines_changed == 0


class TestBehavioralDiff:
    def test_default_values(self) -> None:
        bd = BehavioralDiff()
        assert bd.accuracy_before == 0.0
        assert bd.accuracy_after == 0.0
        assert bd.accuracy_delta == 0.0
        assert bd.tasks_improved == []
        assert bd.tasks_degraded == []
        assert bd.tasks_unchanged == []


class TestDiffEngineComputeDiff:
    def test_identical_code(self) -> None:
        engine = DiffEngine()
        diff = engine.compute_diff("x = 1\n", "x = 1\n")
        assert diff.lines_added == 0
        assert diff.lines_removed == 0
        assert diff.lines_changed == 0
        assert diff.unified_diff == ""

    def test_added_line(self) -> None:
        engine = DiffEngine()
        diff = engine.compute_diff("x = 1\n", "x = 1\ny = 2\n")
        assert diff.lines_added == 1
        assert diff.lines_removed == 0
        assert diff.before == "x = 1\n"
        assert diff.after == "x = 1\ny = 2\n"

    def test_removed_line(self) -> None:
        engine = DiffEngine()
        diff = engine.compute_diff("x = 1\ny = 2\n", "x = 1\n")
        assert diff.lines_removed == 1
        assert diff.lines_added == 0

    def test_changed_line(self) -> None:
        engine = DiffEngine()
        diff = engine.compute_diff("x = 1\n", "x = 2\n")
        assert diff.lines_added == 1
        assert diff.lines_removed == 1
        assert diff.lines_changed == 1

    def test_unified_diff_format(self) -> None:
        engine = DiffEngine()
        diff = engine.compute_diff("x = 1\n", "x = 2\n")
        assert "---" in diff.unified_diff
        assert "+++" in diff.unified_diff
        assert "-x = 1" in diff.unified_diff
        assert "+x = 2" in diff.unified_diff

    def test_multi_line_diff(self) -> None:
        before = "def f():\n    return 1\n"
        after = "def f():\n    x = 2\n    return x\n"
        engine = DiffEngine()
        diff = engine.compute_diff(before, after)
        assert diff.lines_added >= 1
        assert diff.lines_removed >= 1
        assert diff.unified_diff != ""

    def test_empty_to_code(self) -> None:
        engine = DiffEngine()
        diff = engine.compute_diff("", "x = 1\n")
        assert diff.lines_added == 1
        assert diff.lines_removed == 0

    def test_code_to_empty(self) -> None:
        engine = DiffEngine()
        diff = engine.compute_diff("x = 1\n", "")
        assert diff.lines_removed == 1
        assert diff.lines_added == 0


class TestDiffEngineComputeBehavioralDiff:
    def test_identical_results(self) -> None:
        engine = DiffEngine()
        results = [
            {"task_id": "t1", "correct": True},
            {"task_id": "t2", "correct": False},
        ]
        bd = engine.compute_behavioral_diff(results, results)
        assert bd.accuracy_delta == 0.0
        assert bd.tasks_improved == []
        assert bd.tasks_degraded == []
        assert len(bd.tasks_unchanged) == 2

    def test_improvement(self) -> None:
        engine = DiffEngine()
        before = [
            {"task_id": "t1", "correct": False},
            {"task_id": "t2", "correct": True},
        ]
        after = [
            {"task_id": "t1", "correct": True},
            {"task_id": "t2", "correct": True},
        ]
        bd = engine.compute_behavioral_diff(before, after)
        assert bd.accuracy_after > bd.accuracy_before
        assert bd.accuracy_delta > 0
        assert "t1" in bd.tasks_improved
        assert "t2" in bd.tasks_unchanged

    def test_degradation(self) -> None:
        engine = DiffEngine()
        before = [
            {"task_id": "t1", "correct": True},
            {"task_id": "t2", "correct": True},
        ]
        after = [
            {"task_id": "t1", "correct": False},
            {"task_id": "t2", "correct": True},
        ]
        bd = engine.compute_behavioral_diff(before, after)
        assert bd.accuracy_delta < 0
        assert "t1" in bd.tasks_degraded

    def test_empty_results(self) -> None:
        engine = DiffEngine()
        bd = engine.compute_behavioral_diff([], [])
        assert bd.accuracy_before == 0.0
        assert bd.accuracy_after == 0.0
        assert bd.accuracy_delta == 0.0

    def test_auto_generated_task_ids(self) -> None:
        engine = DiffEngine()
        before = [{"correct": True}]
        after = [{"correct": False}]
        bd = engine.compute_behavioral_diff(before, after)
        # task_id auto-generated as "0"
        assert "0" in bd.tasks_degraded

    def test_mixed_task_sets(self) -> None:
        engine = DiffEngine()
        before = [
            {"task_id": "t1", "correct": True},
            {"task_id": "t2", "correct": False},
        ]
        after = [
            {"task_id": "t2", "correct": True},
            {"task_id": "t3", "correct": True},
        ]
        bd = engine.compute_behavioral_diff(before, after)
        # t1 was correct before, not in after -> defaults to False -> degraded
        assert "t1" in bd.tasks_degraded
        # t2 was False, now True -> improved
        assert "t2" in bd.tasks_improved
        # t3 new, defaults to False before, True after -> improved
        assert "t3" in bd.tasks_improved
