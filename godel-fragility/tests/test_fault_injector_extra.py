"""Additional tests for fault injector to increase coverage."""

from __future__ import annotations

import ast
import textwrap

import pytest

from src.adversarial.fault_injector import FaultInjector, FaultType, InjectionResult


@pytest.fixture
def injector() -> FaultInjector:
    return FaultInjector(seed=42)


# ------------------------------------------------------------------ #
# Edge cases for syntax errors
# ------------------------------------------------------------------ #


class TestSyntaxErrorEdgeCases:
    def test_missing_colon_no_candidates(self, injector: FaultInjector) -> None:
        """Code with no lines ending in colon should use fallback."""
        code = "x = 1\ny = 2\n"
        result = injector.inject_syntax_error(code, "missing_colon")
        assert "def broken(" in result.modified_code

    def test_unmatched_paren_no_parens(self, injector: FaultInjector) -> None:
        """Code with no parens should use fallback."""
        code = "x = 1\ny = 2\n"
        result = injector.inject_syntax_error(code, "unmatched_paren")
        assert "x = (1 + 2" in result.modified_code

    def test_bad_indent_no_indented_lines(self, injector: FaultInjector) -> None:
        """Code with no indented lines should use fallback."""
        code = "x = 1\n"
        result = injector.inject_syntax_error(code, "bad_indent")
        assert "x = 1" in result.modified_code


# ------------------------------------------------------------------ #
# Edge cases for runtime errors
# ------------------------------------------------------------------ #


class TestRuntimeErrorEdgeCases:
    def test_name_error_no_assignment(self, injector: FaultInjector) -> None:
        """Code with no assignment should append error."""
        code = "pass\n"
        result = injector.inject_runtime_error(code, "name_error")
        assert "undefined_variable_xyz" in result.modified_code

    def test_type_error_no_assignment(self, injector: FaultInjector) -> None:
        code = "pass\n"
        result = injector.inject_runtime_error(code, "type_error")
        assert '"string" + 42' in result.modified_code

    def test_zero_division_no_assignment(self, injector: FaultInjector) -> None:
        code = "pass\n"
        result = injector.inject_runtime_error(code, "zero_division")
        assert "1 / 0" in result.modified_code


# ------------------------------------------------------------------ #
# Edge cases for logic errors
# ------------------------------------------------------------------ #


class TestLogicErrorEdgeCases:
    def test_off_by_one_no_range(self, injector: FaultInjector) -> None:
        """No range() call should modify a numeric literal."""
        code = "x = 5\ny = 10\n"
        result = injector.inject_logic_error(code, "off_by_one")
        assert result.modified_code != code
        # One of the numbers should be incremented
        assert "6" in result.modified_code or "11" in result.modified_code

    def test_inverted_condition_all_types(self, injector: FaultInjector) -> None:
        """Test all condition inversions."""
        # True -> False
        result = injector.inject_logic_error("x = True\n", "inverted_condition")
        assert "False" in result.modified_code

        # == -> !=
        result = injector.inject_logic_error("if x == 0:\n    pass\n", "inverted_condition")
        assert "!=" in result.modified_code

    def test_wrong_variable_few_names(self, injector: FaultInjector) -> None:
        """Code with < 2 user-defined names can't swap."""
        code = "x = 1\n"
        result = injector.inject_logic_error(code, "wrong_variable")
        # Only one variable (x), can't swap
        assert result.modified_code == code

    def test_wrong_variable_syntax_error(self, injector: FaultInjector) -> None:
        code = "def broken("
        result = injector.inject_logic_error(code, "wrong_variable")
        assert result.modified_code == code


# ------------------------------------------------------------------ #
# Performance regression edge cases
# ------------------------------------------------------------------ #


class TestPerformanceRegressionEdgeCases:
    def test_no_function_def(self, injector: FaultInjector) -> None:
        """Code with no function def should append regression."""
        code = "x = 1\ny = 2\n"
        result = injector.inject_performance_regression(code)
        assert "_perf_waste" in result.modified_code


# ------------------------------------------------------------------ #
# Silent corruption edge cases
# ------------------------------------------------------------------ #


class TestSilentCorruptionEdgeCases:
    def test_no_return_value(self, injector: FaultInjector) -> None:
        """Code with no return should try variable reassignment."""
        code = textwrap.dedent("""\
            x = 10
            y = 20
        """)
        result = injector.inject_silent_corruption(code)
        assert result.modified_code != result.original_code

    def test_return_none(self, injector: FaultInjector) -> None:
        """'return None' is skipped by the return path; if no modifiable
        variables either, success may be False."""
        code = "def f():\n    return None\n"
        result = injector.inject_silent_corruption(code)
        # Code has 'return None' which is skipped, and no variable assignments
        # to corrupt, so the injector cannot modify anything
        assert isinstance(result, InjectionResult)
        assert result.fault_type == FaultType.SILENT

    def test_no_modifiable_code(self, injector: FaultInjector) -> None:
        """Code with only comments/imports should still produce a result."""
        code = "# just a comment\nimport os\n"
        result = injector.inject_silent_corruption(code)
        assert isinstance(result, InjectionResult)
