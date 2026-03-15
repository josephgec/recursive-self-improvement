"""Tests for the timeout guard module."""

from __future__ import annotations

import time

import pytest

from src.harness.timeout_guard import TimeoutError, TimeoutGuard, timeout


# ------------------------------------------------------------------ #
# TimeoutGuard
# ------------------------------------------------------------------ #


class TestTimeoutGuard:
    def test_no_timeout(self) -> None:
        """Code that completes quickly should not timeout."""
        with TimeoutGuard(5.0) as guard:
            x = sum(range(100))
        assert guard.timed_out is False

    def test_zero_timeout_noop(self) -> None:
        """A timeout of 0 should be a no-op."""
        with TimeoutGuard(0) as guard:
            x = sum(range(100))
        assert guard.timed_out is False

    def test_negative_timeout_noop(self) -> None:
        """A negative timeout should be a no-op."""
        with TimeoutGuard(-1) as guard:
            x = sum(range(100))
        assert guard.timed_out is False

    def test_custom_message(self) -> None:
        guard = TimeoutGuard(5.0, message="Custom timeout message")
        assert guard.message == "Custom timeout message"

    def test_default_message(self) -> None:
        guard = TimeoutGuard(10.0)
        assert "10" in guard.message

    def test_timed_out_property_initial(self) -> None:
        guard = TimeoutGuard(5.0)
        assert guard.timed_out is False

    def test_timeout_fires(self) -> None:
        """Signal-based timeout should raise TimeoutError."""
        with pytest.raises(TimeoutError):
            with TimeoutGuard(0.1):
                time.sleep(5)

    def test_guard_cleanup_on_normal_exit(self) -> None:
        """Guard should clean up properly on normal exit."""
        guard = TimeoutGuard(5.0)
        with guard:
            pass
        # Should not raise or leave lingering timers
        assert guard.timed_out is False


# ------------------------------------------------------------------ #
# timeout() context manager
# ------------------------------------------------------------------ #


class TestTimeoutContextManager:
    def test_no_timeout(self) -> None:
        with timeout(5.0) as guard:
            x = sum(range(100))
        assert guard.timed_out is False

    def test_custom_message(self) -> None:
        with timeout(5.0, message="custom") as guard:
            pass
        assert guard.message == "custom"

    def test_timeout_fires(self) -> None:
        with pytest.raises(TimeoutError):
            with timeout(0.1):
                time.sleep(5)
