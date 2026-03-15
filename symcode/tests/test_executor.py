"""Tests for the code executor."""

from __future__ import annotations

import pytest

from src.verification.executor import SymCodeExecutor


class TestSymCodeExecutor:
    """Test subprocess-based code execution."""

    def setup_method(self):
        self.executor = SymCodeExecutor(timeout=10)

    def test_simple_answer(self):
        """Execute code that sets answer = 2+2."""
        code = 'answer = 2 + 2\nprint(f"Answer: {answer}")\n'
        result = self.executor.execute(code)
        assert result.success is True
        assert result.answer == "4"

    def test_sympy_computation(self):
        """Execute code using SymPy."""
        code = (
            "from sympy import symbols, solve\n"
            "x = symbols('x')\n"
            "sols = solve(x**2 - 4, x)\n"
            "answer = sorted(sols)\n"
            'print(f"Answer: {answer}")\n'
        )
        result = self.executor.execute(code)
        assert result.success is True
        assert result.answer is not None

    def test_syntax_error(self):
        """Execute code with a syntax error."""
        code = "def foo(\n  answer = 1\n"
        result = self.executor.execute(code)
        assert result.success is False
        assert result.error is not None
        assert "Syntax" in result.error.error_type or "syntax" in result.error.message.lower()

    def test_zero_division(self):
        """Execute code that raises ZeroDivisionError."""
        code = "answer = 1 / 0\n"
        result = self.executor.execute(code)
        assert result.success is False
        assert result.error is not None
        assert result.error.error_type == "ZeroDivisionError"

    def test_timeout(self):
        """Execute code that runs forever (should timeout)."""
        executor = SymCodeExecutor(timeout=2)
        code = "import time\ntime.sleep(60)\nanswer = 1\n"
        result = executor.execute(code)
        assert result.success is False
        assert result.timed_out is True

    def test_no_answer_variable(self):
        """Execute code that has no answer variable."""
        code = "result = 42\nprint(result)\n"
        result = self.executor.execute(code)
        # Should succeed (no exception) but answer should be None
        assert result.success is True
        assert result.answer is None

    def test_name_error(self):
        """Execute code with undefined variable."""
        code = "answer = undefined_var + 1\n"
        result = self.executor.execute(code)
        assert result.success is False
        assert result.error is not None
        assert result.error.error_type == "NameError"

    def test_stdout_captured(self):
        """Stdout should be captured."""
        code = 'print("hello world")\nanswer = 99\n'
        result = self.executor.execute(code)
        assert result.success is True
        assert "hello world" in result.stdout
        assert result.answer == "99"

    def test_execution_time_recorded(self):
        """Execution time should be > 0."""
        code = "answer = 1\n"
        result = self.executor.execute(code)
        assert result.execution_time > 0

    def test_import_error(self):
        """Execute code with missing import."""
        code = "import nonexistent_module_xyz\nanswer = 1\n"
        result = self.executor.execute(code)
        assert result.success is False
        assert result.error is not None
        assert result.error.error_type in ("ModuleNotFoundError", "ImportError")

    def test_stderr_captured(self):
        """Stderr from subprocess should be captured."""
        code = "import sys\nsys.stderr.write('warning\\n')\nanswer = 1\n"
        result = self.executor.execute(code)
        assert result.success is True
        # stderr may have content from the warning
        assert result.answer == "1"

    def test_large_output(self):
        """Code producing lots of stdout should still work."""
        code = "for i in range(1000): print(i)\nanswer = 999\n"
        result = self.executor.execute(code)
        assert result.success is True
        assert result.answer == "999"

    def test_custom_python_executable(self):
        """SymCodeExecutor with explicit python path."""
        import sys
        executor = SymCodeExecutor(timeout=10, python_executable=sys.executable)
        code = "answer = 7\n"
        result = executor.execute(code)
        assert result.success is True
        assert result.answer == "7"

    def test_parse_stderr_syntax_error(self):
        """Test _parse_stderr with a typical Python traceback."""
        stderr = (
            'Traceback (most recent call last):\n'
            '  File "test.py", line 5, in <module>\n'
            '    x = 1 / 0\n'
            'ZeroDivisionError: division by zero\n'
        )
        error = self.executor._parse_stderr(stderr)
        assert error.error_type == "ZeroDivisionError"
        assert error.message == "division by zero"
        assert error.line_number == 5

    def test_parse_stderr_no_match(self):
        """Test _parse_stderr with non-standard error output."""
        stderr = "Something went wrong but not a Python error"
        error = self.executor._parse_stderr(stderr)
        assert error.error_type == "UnknownError"
        assert error.line_number is None

    def test_code_that_crashes_wrapper(self):
        """If wrapper results JSON is missing, fallback to stderr parsing."""
        # Code that somehow causes the wrapper to fail writing results
        # (e.g., by exiting before writing)
        code = "import sys; sys.exit(1)\n"
        result = self.executor.execute(code)
        # Should handle gracefully
        assert result.success is False

    def test_answer_variable_complex_type(self):
        """Answer variable as a complex expression."""
        code = "answer = [1, 2, 3]\n"
        result = self.executor.execute(code)
        assert result.success is True
        assert result.answer == "[1, 2, 3]"
