"""Tests for the local in-process REPL sandbox."""

from __future__ import annotations

import platform
import threading
from unittest.mock import patch

import pytest

from repl.src.execution import _Timeout, _run_with_thread_timeout, execute_code
from repl.src.local_repl import LocalREPL
from repl.src.sandbox import ExecutionResult, REPLConfig


@pytest.fixture
def repl() -> LocalREPL:
    config = REPLConfig(timeout_seconds=5, max_recursion_depth=3)
    r = LocalREPL(config=config)
    yield r
    r.shutdown()


class TestBasicExecution:
    def test_assign_and_retrieve(self, repl: LocalREPL) -> None:
        result = repl.execute("x = 42")
        assert result.success
        assert repl.get_variable("x") == 42

    def test_print_output(self, repl: LocalREPL) -> None:
        repl.execute("x = 42")
        result = repl.execute("print(x)")
        assert result.success
        assert result.stdout.strip() == "42"

    def test_list_variables(self, repl: LocalREPL) -> None:
        repl.execute("a = 1\nb = 'hello'")
        variables = repl.list_variables()
        assert variables["a"] == 1
        assert variables["b"] == "hello"

    def test_set_variable(self, repl: LocalREPL) -> None:
        repl.set_variable("y", 99)
        result = repl.execute("print(y)")
        assert result.stdout.strip() == "99"

    def test_execution_result_variables(self, repl: LocalREPL) -> None:
        result = repl.execute("x = 42\ny = 'hello'")
        assert result.variables["x"] == 42
        assert result.variables["y"] == "hello"


class TestTimeout:
    def test_timeout_triggers(self, repl: LocalREPL) -> None:
        config = REPLConfig(timeout_seconds=2, max_recursion_depth=3)
        r = LocalREPL(config=config)
        result = r.execute("import time; time.sleep(600)")
        assert not result.success
        assert result.error_type == "timeout"
        r.shutdown()


class TestChildREPL:
    def test_spawn_child_inherits_variables(self, repl: LocalREPL) -> None:
        repl.execute("x = [1, 2, 3]")
        child = repl.spawn_child()
        assert child.get_variable("x") == [1, 2, 3]
        assert child.depth == repl.depth + 1

    def test_child_mutation_does_not_affect_parent(self, repl: LocalREPL) -> None:
        repl.execute("x = [1, 2, 3]")
        child = repl.spawn_child()
        child.execute("x.append(4)")
        # Parent must be unchanged
        assert repl.get_variable("x") == [1, 2, 3]
        assert child.get_variable("x") == [1, 2, 3, 4]

    def test_max_recursion_depth_exceeded(self, repl: LocalREPL) -> None:
        # Config has max_recursion_depth=3, repl is at depth=0
        c1 = repl.spawn_child()   # depth 1
        c2 = c1.spawn_child()     # depth 2
        c3 = c2.spawn_child()     # depth 3
        with pytest.raises(RecursionError):
            c3.spawn_child()      # depth 4 → exceeds limit


class TestSecurity:
    def test_import_os_blocked(self, repl: LocalREPL) -> None:
        result = repl.execute("import os")
        assert not result.success
        assert result.error_type == "SecurityError"

    def test_dunder_class_blocked(self, repl: LocalREPL) -> None:
        result = repl.execute("x = ''.__class__")
        assert not result.success
        assert result.error_type == "SecurityError"

    def test_safe_math_allowed(self, repl: LocalREPL) -> None:
        result = repl.execute("x = 1 + 1")
        assert result.success
        assert repl.get_variable("x") == 2


class TestReset:
    def test_reset_clears_variables(self, repl: LocalREPL) -> None:
        repl.execute("x = 42")
        assert repl.get_variable("x") == 42
        repl.reset()
        with pytest.raises(KeyError):
            repl.get_variable("x")
        # Can still execute after reset
        result = repl.execute("y = 10")
        assert result.success
        assert repl.get_variable("y") == 10


# ---------------------------------------------------------------------------
# Tests for execution.py: output truncation
# ---------------------------------------------------------------------------

class TestOutputTruncation:
    """Test that long stdout/stderr is truncated to max_output_bytes."""

    def test_stdout_truncated(self) -> None:
        """Output exceeding max_output_bytes is truncated with a marker."""
        ns: dict = {}
        # Generate output that exceeds 100 bytes
        result = execute_code(
            "print('A' * 200)",
            ns,
            timeout=10,
            max_output_bytes=100,
        )
        assert result.success
        assert "[output truncated]" in result.stdout
        # The truncated output should be around max_output_bytes + marker
        assert len(result.stdout) < 250

    def test_stderr_truncated(self) -> None:
        """Stderr exceeding max_output_bytes is truncated."""
        ns: dict = {}
        result = execute_code(
            "import sys; sys.stderr.write('E' * 200)",
            ns,
            timeout=10,
            max_output_bytes=100,
        )
        assert result.success
        assert "[output truncated]" in result.stderr

    def test_output_not_truncated_when_short(self) -> None:
        """Short output is not truncated."""
        ns: dict = {}
        result = execute_code(
            "print('hello')",
            ns,
            timeout=10,
            max_output_bytes=1_000_000,
        )
        assert result.success
        assert "[output truncated]" not in result.stdout
        assert result.stdout.strip() == "hello"


# ---------------------------------------------------------------------------
# Tests for execution.py: error handling
# ---------------------------------------------------------------------------

class TestExecutionErrorHandling:
    """Test error handling paths in execute_code."""

    def test_recursion_error(self) -> None:
        """RecursionError is caught and reported."""
        ns: dict = {}
        result = execute_code(
            "def f(): f()\nf()",
            ns,
            timeout=10,
        )
        assert not result.success
        assert result.error_type == "recursion"

    def test_memory_error(self) -> None:
        """MemoryError is caught and reported."""
        ns: dict = {}
        result = execute_code(
            "raise MemoryError('out of memory')",
            ns,
            timeout=10,
        )
        assert not result.success
        assert result.error_type == "memory"
        assert "out of memory" in result.error_message

    def test_memory_error_empty_message(self) -> None:
        """MemoryError with no message uses fallback."""
        ns: dict = {}
        result = execute_code(
            "raise MemoryError()",
            ns,
            timeout=10,
        )
        assert not result.success
        assert result.error_type == "memory"
        assert "Out of memory" in result.error_message

    def test_generic_exception(self) -> None:
        """Generic exceptions are caught with type name and traceback."""
        ns: dict = {}
        result = execute_code(
            "raise ValueError('bad value')",
            ns,
            timeout=10,
        )
        assert not result.success
        assert result.error_type == "ValueError"
        assert "bad value" in result.error_message

    def test_execution_time_tracked(self) -> None:
        """Execution time is measured in milliseconds."""
        ns: dict = {}
        result = execute_code("x = 1 + 1", ns, timeout=10)
        assert result.success
        assert result.execution_time_ms >= 0

    def test_memory_tracking(self) -> None:
        """Memory usage is tracked (>= 0)."""
        ns: dict = {}
        result = execute_code("x = [0] * 1000", ns, timeout=10)
        assert result.success
        assert result.memory_used_mb >= 0

    def test_variables_captured(self) -> None:
        """User variables are captured, dunder names excluded."""
        ns: dict = {}
        result = execute_code(
            "x = 42\ny = 'hello'\n_private = 1",
            ns,
            timeout=10,
        )
        assert result.success
        assert result.variables.get("x") == 42
        assert result.variables.get("y") == "hello"
        assert "_private" not in result.variables


# ---------------------------------------------------------------------------
# Tests for execution.py: thread-based timeout (non-Unix / non-main-thread)
# ---------------------------------------------------------------------------

class TestThreadTimeout:
    """Test the thread-based timeout fallback path."""

    def test_thread_timeout_triggers(self) -> None:
        """_run_with_thread_timeout raises _Timeout for long-running code."""
        ns: dict = {}
        with pytest.raises(_Timeout, match="timed out"):
            _run_with_thread_timeout(
                "import time; time.sleep(60)",
                ns,
                timeout=0.5,
            )

    def test_thread_timeout_propagates_exception(self) -> None:
        """_run_with_thread_timeout propagates exceptions from code."""
        ns: dict = {}
        with pytest.raises(ValueError, match="test error"):
            _run_with_thread_timeout(
                "raise ValueError('test error')",
                ns,
                timeout=10,
            )

    def test_thread_timeout_success(self) -> None:
        """_run_with_thread_timeout completes normally for fast code."""
        ns: dict = {}
        _run_with_thread_timeout("x = 42", ns, timeout=10)
        assert ns["x"] == 42

    def test_execute_code_uses_thread_path_on_windows(self) -> None:
        """On Windows (mocked), execute_code should use thread timeout."""
        with patch("repl.src.execution.platform.system", return_value="Windows"):
            ns: dict = {}
            result = execute_code("x = 1 + 1", ns, timeout=10)
            assert result.success
            assert ns["x"] == 2

    def test_execute_code_uses_thread_path_off_main_thread(self) -> None:
        """When not on main thread, execute_code uses thread timeout."""
        results = []

        def worker():
            ns: dict = {}
            result = execute_code("x = 99", ns, timeout=10)
            results.append(result)

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=10)
        assert len(results) == 1
        assert results[0].success
