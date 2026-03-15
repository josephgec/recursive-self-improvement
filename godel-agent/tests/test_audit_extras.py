"""Tests for audit extras: DiffFormatter, ReasoningTraceCapture, SafetyHooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from src.audit.diff_formatter import DiffFormatter
from src.audit.reasoning_trace import ReasoningTraceCapture, TraceEntry
from src.audit.safety_hooks import SafetyHooks
from src.modification.diff_engine import DiffEngine, CodeDiff


# ===================================================================
# DiffFormatter
# ===================================================================

class TestDiffFormatter:
    def _make_diff(self) -> CodeDiff:
        engine = DiffEngine()
        return engine.compute_diff(
            "def f():\n    return 1\n",
            "def f():\n    x = 2\n    return x\n",
        )

    def test_format_unified_with_precomputed_diff(self) -> None:
        diff = self._make_diff()
        formatter = DiffFormatter()
        output = formatter.format_unified(diff)
        # Since unified_diff is already populated, it should return it directly
        assert "---" in output or "-" in output

    def test_format_unified_without_precomputed_diff(self) -> None:
        diff = CodeDiff(
            before="x = 1\n",
            after="x = 2\n",
            unified_diff="",  # not precomputed
            lines_added=1,
            lines_removed=1,
            lines_changed=1,
        )
        formatter = DiffFormatter()
        output = formatter.format_unified(diff)
        assert "---" in output
        assert "+++" in output
        assert "-x = 1" in output
        assert "+x = 2" in output

    def test_format_summary(self) -> None:
        diff = self._make_diff()
        formatter = DiffFormatter()
        summary = formatter.format_summary(diff)
        assert "Lines added:" in summary
        assert "Lines removed:" in summary
        assert "Lines changed:" in summary

    def test_format_side_by_side(self) -> None:
        formatter = DiffFormatter()
        output = formatter.format_side_by_side("x = 1\ny = 2", "x = 2\ny = 3\nz = 4")
        assert "BEFORE" in output
        assert "AFTER" in output
        assert "x = 1" in output
        assert "x = 2" in output
        # Third line only on the right
        assert "z = 4" in output

    def test_format_side_by_side_empty_before(self) -> None:
        formatter = DiffFormatter()
        output = formatter.format_side_by_side("", "a\nb")
        assert "BEFORE" in output
        assert "a" in output

    def test_format_side_by_side_equal_length(self) -> None:
        formatter = DiffFormatter()
        output = formatter.format_side_by_side("a\nb", "c\nd")
        lines = output.split("\n")
        assert len(lines) == 4  # header + separator + 2 data lines


# ===================================================================
# ReasoningTraceCapture
# ===================================================================

class TestReasoningTraceCapture:
    def test_capture_single_step(self) -> None:
        rtc = ReasoningTraceCapture()
        rtc.capture("analysis", "Analyzing the problem")
        trace = rtc._current_trace
        assert len(trace) == 1
        assert trace[0].step == "analysis"
        assert trace[0].content == "Analyzing the problem"

    def test_capture_with_metadata(self) -> None:
        rtc = ReasoningTraceCapture()
        rtc.capture("compute", "Computing result", confidence=0.9, model="gpt4")
        entry = rtc._current_trace[0]
        assert entry.metadata["confidence"] == 0.9
        assert entry.metadata["model"] == "gpt4"

    def test_end_trace_returns_and_archives(self) -> None:
        rtc = ReasoningTraceCapture()
        rtc.capture("step1", "content1")
        rtc.capture("step2", "content2")
        trace = rtc.end_trace()
        assert len(trace) == 2
        assert len(rtc._current_trace) == 0
        assert len(rtc._traces) == 1

    def test_end_empty_trace(self) -> None:
        rtc = ReasoningTraceCapture()
        trace = rtc.end_trace()
        assert trace == []
        assert len(rtc._traces) == 0

    def test_get_trace_current(self) -> None:
        rtc = ReasoningTraceCapture()
        rtc.capture("s1", "c1")
        # No completed traces, should return current
        trace = rtc.get_trace()
        assert len(trace) == 1

    def test_get_trace_by_index(self) -> None:
        rtc = ReasoningTraceCapture()
        rtc.capture("s1", "c1")
        rtc.end_trace()
        rtc.capture("s2", "c2")
        rtc.end_trace()
        trace = rtc.get_trace(0)
        assert trace[0].step == "s1"
        trace = rtc.get_trace(1)
        assert trace[0].step == "s2"
        trace = rtc.get_trace(-1)
        assert trace[0].step == "s2"

    def test_get_trace_invalid_index(self) -> None:
        rtc = ReasoningTraceCapture()
        rtc.capture("s1", "c1")
        rtc.end_trace()
        trace = rtc.get_trace(99)
        assert trace == []

    def test_get_all_traces(self) -> None:
        rtc = ReasoningTraceCapture()
        rtc.capture("s1", "c1")
        rtc.end_trace()
        rtc.capture("s2", "c2")
        rtc.end_trace()
        all_traces = rtc.get_all_traces()
        assert len(all_traces) == 2

    def test_format_trace_default(self) -> None:
        rtc = ReasoningTraceCapture()
        rtc.capture("analysis", "Looking at the problem carefully")
        rtc.capture("solution", "The answer is 42", confidence=0.95)
        output = rtc.format_trace()
        assert "[1] analysis" in output
        assert "[2] solution" in output
        assert "Looking at the problem" in output
        assert "confidence: 0.95" in output

    def test_format_trace_explicit(self) -> None:
        rtc = ReasoningTraceCapture()
        trace = [
            TraceEntry(step="s1", content="c1"),
            TraceEntry(step="s2", content="c2"),
        ]
        output = rtc.format_trace(trace)
        assert "[1] s1" in output
        assert "[2] s2" in output

    def test_format_empty_trace(self) -> None:
        rtc = ReasoningTraceCapture()
        output = rtc.format_trace()
        assert output == ""

    def test_trace_entry_has_timestamp(self) -> None:
        entry = TraceEntry(step="s", content="c")
        assert entry.timestamp > 0


# ===================================================================
# SafetyHooks
# ===================================================================

class TestSafetyHooks:
    def test_default_init(self) -> None:
        hooks = SafetyHooks()
        assert hooks.max_complexity_ratio == 5.0
        assert hooks.max_modifications_in_window == 3
        assert hooks.window_size == 5

    def test_custom_init(self) -> None:
        hooks = SafetyHooks(max_complexity_ratio=3.0, max_modifications_in_window=2, window_size=10)
        assert hooks.max_complexity_ratio == 3.0
        assert hooks.max_modifications_in_window == 2
        assert hooks.window_size == 10

    def test_check_complexity_bounds_no_baseline(self) -> None:
        hooks = SafetyHooks()
        assert hooks.check_complexity_bounds(100.0) is True

    def test_check_complexity_bounds_zero_baseline(self) -> None:
        hooks = SafetyHooks()
        hooks.set_initial_complexity(0.0)
        assert hooks.check_complexity_bounds(100.0) is True

    def test_check_complexity_bounds_within(self) -> None:
        hooks = SafetyHooks(max_complexity_ratio=5.0)
        hooks.set_initial_complexity(10.0)
        assert hooks.check_complexity_bounds(40.0) is True

    def test_check_complexity_bounds_exceeded(self) -> None:
        hooks = SafetyHooks(max_complexity_ratio=5.0)
        hooks.set_initial_complexity(10.0)
        assert hooks.check_complexity_bounds(60.0) is False

    def test_check_complexity_bounds_exact_boundary(self) -> None:
        hooks = SafetyHooks(max_complexity_ratio=5.0)
        hooks.set_initial_complexity(10.0)
        # ratio = 50/10 = 5.0, not > 5.0
        assert hooks.check_complexity_bounds(50.0) is True

    def test_check_modification_rate_ok(self) -> None:
        hooks = SafetyHooks(max_modifications_in_window=3, window_size=5)

        @dataclass
        class FakeResult:
            modification_applied: bool = False

        results = [FakeResult(False)] * 5
        assert hooks.check_modification_rate(results, 5) is True

    def test_check_modification_rate_too_fast(self) -> None:
        hooks = SafetyHooks(max_modifications_in_window=3, window_size=5)

        @dataclass
        class FakeResult:
            modification_applied: bool = False

        results = [FakeResult(True)] * 5
        assert hooks.check_modification_rate(results, 5) is False

    def test_check_modification_rate_boundary(self) -> None:
        hooks = SafetyHooks(max_modifications_in_window=3, window_size=5)

        @dataclass
        class FakeResult:
            modification_applied: bool = False

        results = [FakeResult(True), FakeResult(True), FakeResult(False), FakeResult(False), FakeResult(False)]
        assert hooks.check_modification_rate(results, 5) is True

    def test_check_modification_rate_windowing(self) -> None:
        hooks = SafetyHooks(max_modifications_in_window=2, window_size=3)

        @dataclass
        class FakeResult:
            modification_applied: bool = False

        # Older entries outside window should not count
        results = [FakeResult(True)] * 5 + [FakeResult(False)] * 3
        assert hooks.check_modification_rate(results, 8) is True

    def test_check_all(self) -> None:
        hooks = SafetyHooks()
        hooks.set_initial_complexity(10.0)

        @dataclass
        class FakeResult:
            modification_applied: bool = False

        results = [FakeResult(False)] * 5
        checks = hooks.check_all(20.0, results, 5)
        assert checks["complexity_bounds"] is True
        assert checks["modification_rate"] is True

    def test_check_all_fails_complexity(self) -> None:
        hooks = SafetyHooks(max_complexity_ratio=2.0)
        hooks.set_initial_complexity(10.0)

        @dataclass
        class FakeResult:
            modification_applied: bool = False

        results = [FakeResult(False)] * 5
        checks = hooks.check_all(100.0, results, 5)
        assert checks["complexity_bounds"] is False
        assert checks["modification_rate"] is True
