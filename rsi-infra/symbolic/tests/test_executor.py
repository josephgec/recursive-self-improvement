"""Tests for the shared subprocess execution helper."""

from __future__ import annotations

import multiprocessing
import queue
from unittest.mock import MagicMock, patch

import pytest

from symbolic.src.executor import _worker, run_in_subprocess


class TestRunInSubprocess:
    """Test run_in_subprocess with real subprocess execution."""

    def test_simple_code_success(self) -> None:
        """Basic code executes successfully."""
        result = run_in_subprocess("x = 1 + 1", timeout=10)
        assert result["success"] is True
        assert result["variables"]["x"] == 2
        assert result["error"] is None
        assert "execution_time_ms" in result

    def test_with_namespace_setup(self) -> None:
        """Namespace setup code runs before user code."""
        result = run_in_subprocess(
            "y = math.sqrt(16)",
            namespace_setup="import math",
            timeout=10,
        )
        assert result["success"] is True
        assert result["variables"]["y"] == 4.0

    def test_setup_keys_excluded(self) -> None:
        """Variables from setup code are excluded from results."""
        result = run_in_subprocess(
            "z = 42",
            namespace_setup="setup_var = 99",
            timeout=10,
        )
        assert result["success"] is True
        assert "z" in result["variables"]
        assert "setup_var" not in result["variables"]

    def test_dunder_variables_excluded(self) -> None:
        """Variables starting with __ are excluded."""
        result = run_in_subprocess(
            "x = 1\n__hidden = 2",
            timeout=10,
        )
        assert result["success"] is True
        assert "x" in result["variables"]
        assert "__hidden" not in result["variables"]

    def test_unpicklable_variable_uses_repr(self) -> None:
        """Unpicklable values fall back to repr()."""
        result = run_in_subprocess(
            "import threading; lock = threading.Lock()",
            timeout=10,
        )
        assert result["success"] is True
        # lock is unpicklable, should be stored as repr string
        assert "lock" in result["variables"]
        assert isinstance(result["variables"]["lock"], str)

    def test_error_in_code(self) -> None:
        """Errors in user code are reported."""
        result = run_in_subprocess("1 / 0", timeout=10)
        assert result["success"] is False
        assert result["error"] is not None
        assert "ZeroDivisionError" in result["error"]
        assert result["variables"] == {}

    def test_error_in_setup(self) -> None:
        """Errors in setup code are reported."""
        result = run_in_subprocess(
            "x = 1",
            namespace_setup="import nonexistent_module_xyz",
            timeout=10,
        )
        assert result["success"] is False
        assert "ModuleNotFoundError" in result["error"] or "ImportError" in result["error"]

    def test_timeout(self) -> None:
        """Long-running code is killed after timeout."""
        result = run_in_subprocess(
            "import time; time.sleep(60)",
            timeout=1.0,
        )
        assert result["success"] is False
        assert "timed out" in result["error"].lower()
        assert result["execution_time_ms"] > 0

    def test_execution_time_tracked(self) -> None:
        """Execution time is reported in milliseconds."""
        result = run_in_subprocess("x = 42", timeout=10)
        assert "execution_time_ms" in result
        assert result["execution_time_ms"] >= 0

    def test_empty_code(self) -> None:
        """Empty code string succeeds with no variables."""
        result = run_in_subprocess("", timeout=10)
        assert result["success"] is True
        assert result["variables"] == {}

    def test_empty_setup(self) -> None:
        """Empty setup string is fine."""
        result = run_in_subprocess("x = 42", namespace_setup="", timeout=10)
        assert result["success"] is True
        assert result["variables"]["x"] == 42

    def test_multiple_variables(self) -> None:
        """Multiple variables are all captured."""
        result = run_in_subprocess(
            "a = 1\nb = 'hello'\nc = [1, 2, 3]",
            timeout=10,
        )
        assert result["success"] is True
        assert result["variables"]["a"] == 1
        assert result["variables"]["b"] == "hello"
        assert result["variables"]["c"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Tests: _worker function directly (avoids subprocess coverage gap)
# ---------------------------------------------------------------------------

class _SimpleQueue:
    """A simple queue wrapper that mimics multiprocessing.Queue interface
    but works in the same process without a background feeder thread."""

    def __init__(self):
        self._items = []

    def put(self, item):
        self._items.append(item)

    def get_nowait(self):
        if not self._items:
            raise queue.Empty
        return self._items.pop(0)


class TestWorkerDirect:
    """Test _worker by calling it directly in the current process."""

    def _run_worker(self, code: str, setup: str = "") -> dict:
        """Run _worker in-process and return the result dict."""
        q = _SimpleQueue()
        _worker(code, setup, q)
        return q.get_nowait()

    def test_basic_execution(self) -> None:
        result = self._run_worker("x = 42")
        assert result["success"] is True
        assert result["variables"]["x"] == 42
        assert result["error"] is None

    def test_with_namespace_setup(self) -> None:
        result = self._run_worker("y = val + 1", "val = 10")
        assert result["success"] is True
        assert result["variables"]["y"] == 11
        # "val" is from setup, should be excluded
        assert "val" not in result["variables"]

    def test_setup_keys_excluded(self) -> None:
        result = self._run_worker("z = a + 1", "a = 5")
        assert result["success"] is True
        assert "a" not in result["variables"]
        assert result["variables"]["z"] == 6

    def test_dunder_excluded(self) -> None:
        result = self._run_worker("x = 1\n__private = 2")
        assert result["success"] is True
        assert "x" in result["variables"]
        assert "__private" not in result["variables"]

    def test_unpicklable_uses_repr(self) -> None:
        result = self._run_worker("import threading; lk = threading.Lock()")
        assert result["success"] is True
        assert "lk" in result["variables"]
        # Should be a string representation since Lock is unpicklable
        assert isinstance(result["variables"]["lk"], str)

    def test_error_handling(self) -> None:
        result = self._run_worker("raise ValueError('oops')")
        assert result["success"] is False
        assert "ValueError" in result["error"]
        assert "oops" in result["error"]
        assert result["variables"] == {}

    def test_setup_error(self) -> None:
        result = self._run_worker("x = 1", "raise RuntimeError('setup failed')")
        assert result["success"] is False
        assert "RuntimeError" in result["error"]

    def test_empty_setup(self) -> None:
        result = self._run_worker("x = 99", "")
        assert result["success"] is True
        assert result["variables"]["x"] == 99

    def test_no_user_variables(self) -> None:
        result = self._run_worker("pass")
        assert result["success"] is True
        assert result["variables"] == {}


# ---------------------------------------------------------------------------
# Tests: edge case branches in run_in_subprocess
# ---------------------------------------------------------------------------

class TestRunInSubprocessEdgeCases:
    """Test edge case branches using mocks where needed."""

    def test_nonzero_exitcode_empty_queue(self) -> None:
        """When subprocess exits with non-zero code and no result, report error."""
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = False
        mock_proc.exitcode = 1

        mock_queue = MagicMock()
        mock_queue.empty.return_value = True

        mock_ctx = MagicMock()
        mock_ctx.Queue.return_value = mock_queue
        mock_ctx.Process.return_value = mock_proc

        with patch("symbolic.src.executor.multiprocessing.get_context", return_value=mock_ctx):
            result = run_in_subprocess("x = 1", timeout=10)

        assert result["success"] is False
        assert "exited with code 1" in result["error"]
        assert "execution_time_ms" in result

    def test_empty_queue_zero_exitcode(self) -> None:
        """When queue is empty but exitcode is 0, report no result."""
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = False
        mock_proc.exitcode = 0

        mock_queue = MagicMock()
        # exitcode == 0, so the `exitcode != 0 and queue.empty()` branch is
        # skipped.  The next `if result_queue.empty()` check should return True.
        mock_queue.empty.return_value = True

        mock_ctx = MagicMock()
        mock_ctx.Queue.return_value = mock_queue
        mock_ctx.Process.return_value = mock_proc

        with patch("symbolic.src.executor.multiprocessing.get_context", return_value=mock_ctx):
            result = run_in_subprocess("x = 1", timeout=10)

        assert result["success"] is False
        assert "No result returned" in result["error"]
