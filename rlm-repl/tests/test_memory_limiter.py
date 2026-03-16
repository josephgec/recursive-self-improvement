"""Tests for the MemoryLimiter."""

import pytest
from src.safety.memory_limiter import MemoryLimiter, MemoryStatus


class TestMemoryLimiter:
    """Test memory limiting and monitoring."""

    def setup_method(self):
        self.limiter = MemoryLimiter(max_memory_mb=512.0)

    def test_monitor_returns_status(self):
        status = self.limiter.monitor()
        assert isinstance(status, MemoryStatus)
        assert status.limit_mb == 512.0

    def test_current_usage(self):
        usage = self.limiter.get_current_usage_mb()
        assert isinstance(usage, float)
        assert usage >= 0

    def test_status_fields(self):
        status = self.limiter.monitor()
        assert hasattr(status, "current_mb")
        assert hasattr(status, "peak_mb")
        assert hasattr(status, "limit_mb")
        assert hasattr(status, "available_mb")
        assert hasattr(status, "exceeded")

    def test_available_mb_calculated(self):
        status = self.limiter.monitor()
        expected = max(0, status.limit_mb - status.current_mb)
        assert abs(status.available_mb - expected) < 0.1

    def test_reset_peak(self):
        self.limiter.get_current_usage_mb()
        self.limiter.reset_peak()
        assert self.limiter._peak_mb == 0.0

    def test_peak_tracking(self):
        self.limiter.get_current_usage_mb()
        status = self.limiter.monitor()
        assert status.peak_mb >= 0

    def test_set_process_limit(self):
        # This may or may not succeed depending on the platform
        result = self.limiter.set_process_limit(1024.0)
        assert isinstance(result, bool)

    def test_memory_status_dataclass(self):
        status = MemoryStatus(
            current_mb=100.0,
            peak_mb=200.0,
            limit_mb=512.0,
            available_mb=412.0,
            exceeded=False,
        )
        assert status.current_mb == 100.0
        assert not status.exceeded

    def test_exceeded_status(self):
        status = MemoryStatus(
            current_mb=600.0,
            peak_mb=600.0,
            limit_mb=512.0,
            available_mb=0.0,
            exceeded=True,
        )
        assert status.exceeded

    def test_default_max_memory(self):
        limiter = MemoryLimiter()
        assert limiter.max_memory_mb == 512.0

    def test_custom_max_memory(self):
        limiter = MemoryLimiter(max_memory_mb=1024.0)
        assert limiter.max_memory_mb == 1024.0
        status = limiter.monitor()
        assert status.limit_mb == 1024.0
