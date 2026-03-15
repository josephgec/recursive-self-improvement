"""Tests for RuntimeInspector."""

from __future__ import annotations

from src.core.runtime import RuntimeInspector, FunctionInfo, PerformanceSnapshot


class DummyComponent:
    """Dummy component for testing inspection."""

    name = "dummy"
    value = 42

    def do_something(self, x: int) -> int:
        """Do something with x."""
        return x * 2

    def other_method(self) -> str:
        return "hello"


class TestInspectFunctions:
    def test_inspect_functions_returns_list(self) -> None:
        inspector = RuntimeInspector()
        funcs = inspector.inspect_functions(DummyComponent())
        assert isinstance(funcs, list)
        names = [f.name for f in funcs]
        assert "do_something" in names
        assert "other_method" in names

    def test_function_info_has_source(self) -> None:
        inspector = RuntimeInspector()
        funcs = inspector.inspect_functions(DummyComponent())
        do_something = next(f for f in funcs if f.name == "do_something")
        assert "return x * 2" in do_something.source

    def test_inspect_variables(self) -> None:
        inspector = RuntimeInspector()
        variables = inspector.inspect_variables(DummyComponent())
        assert "name" in variables
        assert "value" in variables


class TestInspectPerformance:
    def test_empty_history(self) -> None:
        inspector = RuntimeInspector()
        perf = inspector.inspect_performance([])
        assert perf.accuracy_last_n == 0.0
        assert perf.trend == 0.0
        assert perf.total_tasks == 0

    def test_positive_trend(self) -> None:
        inspector = RuntimeInspector()
        history = [0.5, 0.55, 0.6, 0.65, 0.7]
        perf = inspector.inspect_performance(history)
        assert perf.trend > 0
        assert perf.accuracy_last_n > 0.5
        assert perf.total_tasks == 5

    def test_negative_trend(self) -> None:
        inspector = RuntimeInspector()
        history = [0.8, 0.75, 0.7, 0.65, 0.6]
        perf = inspector.inspect_performance(history)
        assert perf.trend < 0

    def test_stagnant_trend(self) -> None:
        inspector = RuntimeInspector()
        history = [0.7, 0.7, 0.7, 0.7, 0.7]
        perf = inspector.inspect_performance(history)
        assert abs(perf.trend) < 0.01

    def test_error_types(self) -> None:
        inspector = RuntimeInspector()
        history = [0.5, 0.6]
        errors = [
            {"error_type": "arithmetic"},
            {"error_type": "arithmetic"},
            {"error_type": "logic"},
        ]
        perf = inspector.inspect_performance(history, error_log=errors)
        assert perf.error_types["arithmetic"] == 2
        assert perf.error_types["logic"] == 1

    def test_correct_tasks_count(self) -> None:
        inspector = RuntimeInspector()
        history = [0.8, 0.9, 0.3, 0.6, 0.1]
        perf = inspector.inspect_performance(history)
        assert perf.correct_tasks == 3  # 0.8, 0.9, 0.6 are >= 0.5


class TestSelfReport:
    def test_report_contains_key_info(self) -> None:
        inspector = RuntimeInspector()
        history = [0.6, 0.65, 0.7, 0.68, 0.72]
        report = inspector.generate_self_report(history)
        assert "Self-Report" in report
        assert "Total iterations: 5" in report
        assert "Recent accuracy" in report

    def test_report_includes_modifications(self) -> None:
        inspector = RuntimeInspector()
        history = [0.6, 0.65]
        mods = [
            {"target": "prompt_strategy", "success": True},
            {"target": "reasoning_strategy", "success": False},
        ]
        report = inspector.generate_self_report(history, modifications=mods)
        assert "prompt_strategy" in report
        assert "reasoning_strategy" in report

    def test_report_truncation(self) -> None:
        inspector = RuntimeInspector()
        history = [0.5] * 100
        report = inspector.generate_self_report(history, max_tokens=10)
        # Very short max_tokens should truncate
        assert len(report) <= 60  # 10 tokens * 4 chars + truncation marker


class TestGetFunctionSource:
    def test_get_source_of_function(self) -> None:
        inspector = RuntimeInspector()

        def my_func(x: int) -> int:
            return x + 1

        source = inspector.get_function_source(my_func)
        assert "return x + 1" in source

    def test_get_source_of_builtin(self) -> None:
        inspector = RuntimeInspector()
        source = inspector.get_function_source(len)
        assert source == ""  # builtins have no source
