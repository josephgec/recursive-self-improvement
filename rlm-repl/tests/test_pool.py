"""Tests for the REPLPool."""

import threading
import pytest
from src.pool.pool import REPLPool, PoolMetrics
from src.pool.lifecycle import REPLLifecycle
from src.pool.metrics import PoolMetricsTracker, MetricsSummary
from src.safety.policy import SafetyPolicy


class TestREPLPool:
    """Test REPL pool acquire/release."""

    def test_pool_creation(self):
        pool = REPLPool(size=2)
        assert pool.total == 2
        assert pool.available == 2
        pool.shutdown()

    def test_acquire_and_release(self):
        pool = REPLPool(size=2)
        repl = pool.acquire()
        assert repl is not None
        assert pool.available == 1
        pool.release(repl)
        assert pool.available == 2
        pool.shutdown()

    def test_acquire_all(self):
        pool = REPLPool(size=2)
        r1 = pool.acquire()
        r2 = pool.acquire()
        assert pool.available == 0
        pool.release(r1)
        pool.release(r2)
        pool.shutdown()

    def test_acquire_timeout(self):
        pool = REPLPool(size=1)
        r1 = pool.acquire()
        with pytest.raises(TimeoutError):
            pool.acquire(timeout=0.1)
        pool.release(r1)
        pool.shutdown()

    def test_pool_metrics(self):
        pool = REPLPool(size=2)
        r1 = pool.acquire()
        pool.release(r1)

        metrics = pool.get_metrics()
        assert isinstance(metrics, PoolMetrics)
        assert metrics.total == 2
        assert metrics.total_acquires == 1
        assert metrics.total_releases == 1
        pool.shutdown()

    def test_pool_shutdown(self):
        pool = REPLPool(size=2)
        pool.shutdown()
        with pytest.raises(RuntimeError):
            pool.acquire()

    def test_release_after_shutdown(self):
        pool = REPLPool(size=2)
        repl = pool.acquire()
        pool.shutdown()
        # Should not raise, just destroys the REPL
        pool.release(repl)

    def test_concurrent_access(self):
        pool = REPLPool(size=4)
        results = []
        errors = []

        def worker():
            try:
                repl = pool.acquire(timeout=5.0)
                result = repl.execute("x = 1")
                results.append(result.success)
                pool.release(repl)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert all(results)
        pool.shutdown()

    def test_pool_with_policy(self):
        policy = SafetyPolicy(timeout_seconds=5)
        pool = REPLPool(size=2, policy=policy)
        repl = pool.acquire()
        result = repl.execute("x = 42")
        assert result.success
        pool.release(repl)
        pool.shutdown()


class TestREPLLifecycle:
    """Test REPL lifecycle management."""

    def setup_method(self):
        self.lifecycle = REPLLifecycle()

    def test_create(self):
        repl = self.lifecycle.create()
        assert repl.is_alive()
        repl.shutdown()

    def test_warm(self):
        repl = self.lifecycle.create()
        assert self.lifecycle.warm(repl)
        repl.shutdown()

    def test_recycle(self):
        repl = self.lifecycle.create()
        repl.execute("x = 42")
        recycled = self.lifecycle.recycle(repl)
        assert recycled.is_alive()
        # After recycle, variable should be cleared
        with pytest.raises(KeyError):
            recycled.get_variable("x")
        recycled.shutdown()

    def test_recycle_dead_repl(self):
        repl = self.lifecycle.create()
        repl.shutdown()
        recycled = self.lifecycle.recycle(repl)
        assert recycled.is_alive()
        recycled.shutdown()

    def test_destroy(self):
        repl = self.lifecycle.create()
        self.lifecycle.destroy(repl)
        assert not repl.is_alive()

    def test_health_check_alive(self):
        repl = self.lifecycle.create()
        assert self.lifecycle.health_check(repl)
        repl.shutdown()

    def test_health_check_dead(self):
        repl = self.lifecycle.create()
        repl.shutdown()
        assert not self.lifecycle.health_check(repl)


class TestPoolMetricsTracker:
    """Test metrics tracking."""

    def setup_method(self):
        self.tracker = PoolMetricsTracker()

    def test_record_acquire(self):
        self.tracker.record_acquire(wait_time_ms=10.0)
        summary = self.tracker.summary()
        assert summary.total_acquires == 1
        assert summary.total_wait_time_ms == 10.0

    def test_record_release(self):
        self.tracker.record_release(hold_time_ms=50.0)
        summary = self.tracker.summary()
        assert summary.total_releases == 1
        assert summary.total_hold_time_ms == 50.0

    def test_averages(self):
        self.tracker.record_acquire(wait_time_ms=10.0)
        self.tracker.record_acquire(wait_time_ms=20.0)
        self.tracker.record_release(hold_time_ms=30.0)
        self.tracker.record_release(hold_time_ms=40.0)

        summary = self.tracker.summary()
        assert summary.avg_wait_time_ms == 15.0
        assert summary.avg_hold_time_ms == 35.0

    def test_peak_concurrent(self):
        self.tracker.record_acquire()
        self.tracker.record_acquire()
        self.tracker.record_release()
        self.tracker.record_acquire()

        summary = self.tracker.summary()
        assert summary.peak_concurrent == 2

    def test_reset(self):
        self.tracker.record_acquire()
        self.tracker.record_release()
        self.tracker.reset()
        summary = self.tracker.summary()
        assert summary.total_acquires == 0
        assert summary.total_releases == 0

    def test_empty_summary(self):
        summary = self.tracker.summary()
        assert summary.total_acquires == 0
        assert summary.avg_wait_time_ms == 0
        assert summary.avg_hold_time_ms == 0

    def test_metrics_summary_dataclass(self):
        summary = MetricsSummary(total_acquires=5, total_releases=3)
        assert summary.total_acquires == 5
        assert summary.total_releases == 3
