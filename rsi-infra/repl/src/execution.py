"""Low-level code execution with resource limits."""

from __future__ import annotations

import contextlib
import io
import platform
import signal
import sys
import threading
import time
import traceback
from typing import Any

from repl.src.sandbox import ExecutionResult

# psutil is an optional-but-expected dependency (listed in pyproject.toml).
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


class _Timeout(Exception):
    """Raised when code execution exceeds the time limit."""


def _timeout_handler(signum: int, frame: Any) -> None:  # noqa: ANN401
    raise _Timeout("Execution timed out")


def execute_code(
    code: str,
    namespace: dict[str, Any],
    timeout: float = 300,
    max_output_bytes: int = 1_048_576,
) -> ExecutionResult:
    """Execute *code* inside *namespace* and return an :class:`ExecutionResult`.

    Parameters
    ----------
    code:
        Python source code to execute via ``exec()``.
    namespace:
        The globals dict.  It is mutated in-place so that variable state
        persists across calls.
    timeout:
        Maximum wall-clock seconds before the execution is aborted.
    max_output_bytes:
        Maximum combined bytes of captured stdout + stderr.
    """

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    start_time = time.perf_counter()
    mem_before = _memory_mb()

    result = ExecutionResult()

    try:
        with (
            contextlib.redirect_stdout(stdout_buf),
            contextlib.redirect_stderr(stderr_buf),
        ):
            _run_with_timeout(code, namespace, timeout)

        result.success = True

    except _Timeout:
        result.success = False
        result.error_type = "timeout"
        result.error_message = f"Execution exceeded {timeout}s time limit"

    except RecursionError as exc:
        result.success = False
        result.error_type = "recursion"
        result.error_message = str(exc)

    except MemoryError as exc:
        result.success = False
        result.error_type = "memory"
        result.error_message = str(exc) or "Out of memory"

    except Exception as exc:  # noqa: BLE001
        result.success = False
        result.error_type = type(exc).__name__
        result.error_message = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )

    elapsed = time.perf_counter() - start_time
    mem_after = _memory_mb()

    # Capture output (truncated to max_output_bytes) -----------------------
    raw_stdout = stdout_buf.getvalue()
    raw_stderr = stderr_buf.getvalue()

    if len(raw_stdout.encode()) > max_output_bytes:
        raw_stdout = raw_stdout[: max_output_bytes] + "\n... [output truncated]"
    if len(raw_stderr.encode()) > max_output_bytes:
        raw_stderr = raw_stderr[: max_output_bytes] + "\n... [output truncated]"

    result.stdout = raw_stdout
    result.stderr = raw_stderr
    result.execution_time_ms = round(elapsed * 1000, 2)
    result.memory_used_mb = round(max(mem_after - mem_before, 0), 2)

    # Snapshot user variables (exclude dunder / builtins) -------------------
    result.variables = {
        k: v
        for k, v in namespace.items()
        if not k.startswith("_") and k != "__builtins__"
    }

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_with_timeout(code: str, namespace: dict, timeout: float) -> None:
    """Run *code* with a wall-clock timeout.

    On Unix we use ``signal.SIGALRM`` which is reliable.  On other platforms
    we fall back to a daemon thread + ``threading.Event``.
    """
    if platform.system() != "Windows" and threading.current_thread() is threading.main_thread():
        _run_with_signal_timeout(code, namespace, timeout)
    else:
        _run_with_thread_timeout(code, namespace, timeout)


def _run_with_signal_timeout(code: str, namespace: dict, timeout: float) -> None:
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(int(max(timeout, 1)))
    try:
        exec(code, namespace)  # noqa: S102 -- intentional exec
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _run_with_thread_timeout(code: str, namespace: dict, timeout: float) -> None:
    result_exc: list[BaseException] = []
    done = threading.Event()

    def _target() -> None:
        try:
            exec(code, namespace)  # noqa: S102
        except BaseException as exc:  # noqa: BLE001
            result_exc.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()

    if not done.wait(timeout=timeout):
        raise _Timeout("Execution timed out")

    if result_exc:
        raise result_exc[0]


def _memory_mb() -> float:
    """Return current process RSS in MiB (0 if psutil unavailable)."""
    if not _HAS_PSUTIL:
        return 0.0
    proc = psutil.Process()
    return proc.memory_info().rss / (1024 * 1024)
