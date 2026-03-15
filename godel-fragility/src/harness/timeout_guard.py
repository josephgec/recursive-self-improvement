"""Timeout guard for stress test execution."""

from __future__ import annotations

import signal
import sys
import threading
from contextlib import contextmanager
from typing import Generator, Optional


class TimeoutError(Exception):
    """Raised when execution exceeds the time limit."""


class TimeoutGuard:
    """Context manager that kills execution after N seconds.

    Uses threading-based timeout for cross-platform compatibility.
    Falls back to signal-based on Unix when running in the main thread.
    """

    def __init__(self, seconds: float, message: Optional[str] = None) -> None:
        self.seconds = seconds
        self.message = message or f"Execution timed out after {seconds}s"
        self._timer: Optional[threading.Timer] = None
        self._timed_out = False

    def __enter__(self) -> "TimeoutGuard":
        self._timed_out = False

        if self.seconds <= 0:
            return self

        # Try signal-based timeout first (only works in main thread on Unix)
        if (
            sys.platform != "win32"
            and threading.current_thread() is threading.main_thread()
        ):
            self._use_signal = True
            self._old_handler = signal.signal(signal.SIGALRM, self._signal_handler)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
        else:
            self._use_signal = False
            self._timer = threading.Timer(self.seconds, self._thread_timeout)
            self._timer.daemon = True
            self._timer.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:  # type: ignore[no-untyped-def]
        if self.seconds <= 0:
            return False

        if hasattr(self, "_use_signal") and self._use_signal:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self._old_handler)
        elif self._timer is not None:
            self._timer.cancel()
            self._timer = None

        return False  # Don't suppress exceptions

    def _signal_handler(self, signum: int, frame: object) -> None:
        self._timed_out = True
        raise TimeoutError(self.message)

    def _thread_timeout(self) -> None:
        self._timed_out = True
        # Thread-based timeout can only set a flag;
        # the running code must check `timed_out` periodically.

    @property
    def timed_out(self) -> bool:
        return self._timed_out


@contextmanager
def timeout(seconds: float, message: Optional[str] = None) -> Generator[TimeoutGuard, None, None]:
    """Convenience context manager for timeouts."""
    guard = TimeoutGuard(seconds, message)
    with guard:
        yield guard
