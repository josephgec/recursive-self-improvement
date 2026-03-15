"""Tests for the fault injector."""

from __future__ import annotations

import ast
import textwrap

import pytest

from src.adversarial.fault_injector import FaultInjector, FaultType, InjectionResult


@pytest.fixture
def injector() -> FaultInjector:
    return FaultInjector(seed=42)


@pytest.fixture
def sample_code() -> str:
    return textwrap.dedent("""\
        def solve(x):
            if x > 0:
                return x * 2
            else:
                return 0
    """)


class TestSyntaxErrors:
    def test_missing_colon(self, injector: FaultInjector, sample_code: str) -> None:
        result = injector.inject_syntax_error(sample_code, "missing_colon")
        assert isinstance(result, InjectionResult)
        assert result.fault_type == FaultType.SYNTAX
        assert result.fault_subtype == "missing_colon"
        assert result.modified_code != result.original_code
        # Should now fail to parse
        with pytest.raises(SyntaxError):
            ast.parse(result.modified_code)

    def test_unmatched_paren(self, injector: FaultInjector, sample_code: str) -> None:
        result = injector.inject_syntax_error(sample_code, "unmatched_paren")
        assert result.fault_type == FaultType.SYNTAX
        assert result.modified_code != result.original_code
        with pytest.raises(SyntaxError):
            ast.parse(result.modified_code)

    def test_bad_indent(self, injector: FaultInjector, sample_code: str) -> None:
        result = injector.inject_syntax_error(sample_code, "bad_indent")
        assert result.fault_type == FaultType.SYNTAX
        assert result.modified_code != result.original_code

    def test_unknown_subtype_raises(self, injector: FaultInjector, sample_code: str) -> None:
        with pytest.raises(ValueError, match="Unknown syntax subtype"):
            injector.inject_syntax_error(sample_code, "nonexistent")


class TestRuntimeErrors:
    def test_name_error(self, injector: FaultInjector, sample_code: str) -> None:
        result = injector.inject_runtime_error(sample_code, "name_error")
        assert result.fault_type == FaultType.RUNTIME
        assert result.fault_subtype == "name_error"
        assert "undefined_variable_xyz" in result.modified_code
        # Should parse fine
        ast.parse(result.modified_code)

    def test_type_error(self, injector: FaultInjector, sample_code: str) -> None:
        result = injector.inject_runtime_error(sample_code, "type_error")
        assert result.fault_type == FaultType.RUNTIME
        assert '"string" + 42' in result.modified_code
        ast.parse(result.modified_code)

    def test_zero_division(self, injector: FaultInjector, sample_code: str) -> None:
        result = injector.inject_runtime_error(sample_code, "zero_division")
        assert result.fault_type == FaultType.RUNTIME
        assert "1 / 0" in result.modified_code
        ast.parse(result.modified_code)

    def test_unknown_subtype_raises(self, injector: FaultInjector, sample_code: str) -> None:
        with pytest.raises(ValueError, match="Unknown runtime subtype"):
            injector.inject_runtime_error(sample_code, "nonexistent")


class TestLogicErrors:
    def test_off_by_one(self, injector: FaultInjector) -> None:
        code = "for i in range(10):\n    print(i)"
        result = injector.inject_logic_error(code, "off_by_one")
        assert result.fault_type == FaultType.LOGIC
        assert result.modified_code != code
        # range(10) should be changed to range(11)
        assert "11" in result.modified_code

    def test_inverted_condition(self, injector: FaultInjector, sample_code: str) -> None:
        result = injector.inject_logic_error(sample_code, "inverted_condition")
        assert result.fault_type == FaultType.LOGIC
        assert result.modified_code != sample_code

    def test_wrong_variable(self, injector: FaultInjector) -> None:
        code = textwrap.dedent("""\
            def compute(a, b):
                result = a + b
                return result
        """)
        result = injector.inject_logic_error(code, "wrong_variable")
        assert result.fault_type == FaultType.LOGIC
        # Variable names should be swapped
        assert result.modified_code != code

    def test_unknown_subtype_raises(self, injector: FaultInjector, sample_code: str) -> None:
        with pytest.raises(ValueError, match="Unknown logic subtype"):
            injector.inject_logic_error(sample_code, "nonexistent")


class TestPerformanceRegression:
    def test_inject_performance_regression(self, injector: FaultInjector, sample_code: str) -> None:
        result = injector.inject_performance_regression(sample_code)
        assert result.fault_type == FaultType.PERFORMANCE
        assert result.fault_subtype == "nested_loop_waste"
        assert "_perf_waste" in result.modified_code
        assert "range(1000)" in result.modified_code
        # Should still parse
        ast.parse(result.modified_code)


class TestSilentCorruption:
    def test_inject_silent_corruption(self, injector: FaultInjector, sample_code: str) -> None:
        result = injector.inject_silent_corruption(sample_code)
        assert result.fault_type == FaultType.SILENT
        assert result.modified_code != result.original_code
        # Should still parse
        ast.parse(result.modified_code)

    def test_silent_corruption_on_return_value(self, injector: FaultInjector) -> None:
        code = "def f():\n    return 42\n"
        result = injector.inject_silent_corruption(code)
        assert "0.99" in result.modified_code or "corrupted" in result.modified_code


class TestInjectionResult:
    def test_injection_result_fields(self, injector: FaultInjector, sample_code: str) -> None:
        result = injector.inject_syntax_error(sample_code, "missing_colon")
        assert result.original_code == sample_code
        assert result.success is True
        assert result.description
