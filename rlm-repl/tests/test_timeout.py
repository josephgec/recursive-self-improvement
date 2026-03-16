"""Tests for the TimeoutEnforcer."""

import time
import pytest
from src.safety.timeout import TimeoutEnforcer
from src.interface.errors import ExecutionTimeoutError


class TestTimeoutEnforcer:
    """Test timeout enforcement."""

    def setup_method(self):
        self.enforcer = TimeoutEnforcer(timeout_seconds=2.0)

    def test_fast_function_completes(self):
        def fast():
            return 42

        result = self.enforcer.execute_with_timeout(fast)
        assert result == 42

    def test_slow_function_times_out(self):
        def slow():
            time.sleep(10)
            return "should not reach"

        with pytest.raises(ExecutionTimeoutError):
            self.enforcer.execute_with_timeout(slow, timeout=0.5)

    def test_function_with_args(self):
        def add(a, b):
            return a + b

        result = self.enforcer.execute_with_timeout(add, args=(3, 4))
        assert result == 7

    def test_function_with_kwargs(self):
        def greet(name="world"):
            return f"hello {name}"

        result = self.enforcer.execute_with_timeout(
            greet, kwargs={"name": "test"}
        )
        assert result == "hello test"

    def test_override_timeout(self):
        def fast():
            return "ok"

        result = self.enforcer.execute_with_timeout(fast, timeout=10.0)
        assert result == "ok"

    def test_exception_propagated(self):
        def failing():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            self.enforcer.execute_with_timeout(failing)

    def test_signal_timeout(self):
        """Test signal-based timeout specifically."""
        enforcer = TimeoutEnforcer(timeout_seconds=0.5)
        enforcer._use_signals = True

        def slow():
            while True:
                pass

        with pytest.raises(ExecutionTimeoutError):
            enforcer.execute_with_timeout(slow, timeout=0.5)

    def test_thread_timeout(self):
        """Test threading-based timeout specifically."""
        enforcer = TimeoutEnforcer(timeout_seconds=0.5)
        enforcer._use_signals = False

        def slow():
            time.sleep(10)

        with pytest.raises(ExecutionTimeoutError):
            enforcer.execute_with_timeout(slow, timeout=0.5)

    def test_thread_exception_propagated(self):
        """Test that exceptions in threaded execution are propagated."""
        enforcer = TimeoutEnforcer(timeout_seconds=5.0)
        enforcer._use_signals = False

        def failing():
            raise RuntimeError("thread error")

        with pytest.raises(RuntimeError, match="thread error"):
            enforcer.execute_with_timeout(failing)

    def test_timeout_error_attributes(self):
        err = ExecutionTimeoutError(5.0)
        assert err.timeout_seconds == 5.0
        assert "5" in str(err)

    def test_custom_timeout_message(self):
        err = ExecutionTimeoutError(5.0, message="custom msg")
        assert str(err) == "custom msg"
