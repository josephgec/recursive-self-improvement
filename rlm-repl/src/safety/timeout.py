"""Timeout enforcement for sandboxed code execution."""

import platform
import threading
from typing import Any, Callable, Optional, Tuple

from src.interface.errors import ExecutionTimeoutError


class TimeoutEnforcer:
    """Enforces execution time limits using signal-based or threading-based timeouts.

    On Unix systems, uses signal.SIGALRM for reliable timeout enforcement.
    Falls back to threading-based timeout on non-Unix or when signals are unavailable.
    """

    def __init__(self, timeout_seconds: float = 30.0):
        self.timeout_seconds = timeout_seconds
        self._use_signals = platform.system() != "Windows"

    def execute_with_timeout(
        self,
        func: Callable[..., Any],
        args: tuple = (),
        kwargs: dict = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """Execute a function with a timeout.

        Args:
            func: Function to execute.
            args: Positional arguments.
            kwargs: Keyword arguments.
            timeout: Override timeout in seconds.

        Returns:
            The function's return value.

        Raises:
            ExecutionTimeoutError: If the function exceeds the timeout.
        """
        kwargs = kwargs or {}
        effective_timeout = timeout if timeout is not None else self.timeout_seconds

        # Signal-based timeout only works from the main thread
        use_signals = (
            self._use_signals
            and threading.current_thread() is threading.main_thread()
        )

        if use_signals:
            return self._execute_with_signal(func, args, kwargs, effective_timeout)
        else:
            return self._execute_with_thread(func, args, kwargs, effective_timeout)

    def _execute_with_signal(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        timeout: float,
    ) -> Any:
        """Execute with signal-based timeout (Unix only)."""
        import signal

        def handler(signum, frame):
            raise ExecutionTimeoutError(timeout)

        old_handler = signal.signal(signal.SIGALRM, handler)
        # Use integer seconds for alarm; use setitimer if available for sub-second
        try:
            if hasattr(signal, "setitimer"):
                signal.setitimer(signal.ITIMER_REAL, timeout)
            else:
                signal.alarm(max(1, int(timeout)))
            try:
                result = func(*args, **kwargs)
            finally:
                if hasattr(signal, "setitimer"):
                    signal.setitimer(signal.ITIMER_REAL, 0)
                else:
                    signal.alarm(0)
            return result
        finally:
            signal.signal(signal.SIGALRM, old_handler)

    def _execute_with_thread(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        timeout: float,
    ) -> Any:
        """Execute with threading-based timeout (cross-platform fallback)."""
        result_container: dict = {"result": None, "exception": None}

        def target():
            try:
                result_container["result"] = func(*args, **kwargs)
            except Exception as e:
                result_container["exception"] = e

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            raise ExecutionTimeoutError(timeout)

        if result_container["exception"] is not None:
            raise result_container["exception"]

        return result_container["result"]
