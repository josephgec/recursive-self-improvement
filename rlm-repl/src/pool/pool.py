"""REPL pool for managing multiple sandbox instances."""

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

from src.interface.base import SandboxREPL
from src.safety.policy import SafetyPolicy
from src.pool.lifecycle import REPLLifecycle
from src.pool.metrics import PoolMetricsTracker

logger = logging.getLogger(__name__)


@dataclass
class PoolMetrics:
    """Snapshot of pool metrics.

    Attributes:
        total: Total REPL instances managed.
        available: Number of available instances.
        in_use: Number of instances currently in use.
        total_acquires: Total acquire operations.
        total_releases: Total release operations.
    """

    total: int = 0
    available: int = 0
    in_use: int = 0
    total_acquires: int = 0
    total_releases: int = 0


class REPLPool:
    """Pool of REPL instances with acquire/release semantics.

    Manages a fixed-size pool of REPL instances for efficient reuse.
    Supports both synchronous and async acquire/release.
    """

    def __init__(
        self,
        size: int = 4,
        policy: Optional[SafetyPolicy] = None,
    ):
        self._size = size
        self._policy = policy or SafetyPolicy()
        self._lifecycle = REPLLifecycle(policy=self._policy)
        self._metrics_tracker = PoolMetricsTracker()
        self._pool: queue.Queue = queue.Queue(maxsize=size)
        self._in_use: set = set()
        self._all_repls: list = []
        self._lock = threading.Lock()
        self._shutdown = False

        # Pre-populate pool
        for _ in range(size):
            repl = self._lifecycle.create()
            self._pool.put(repl)
            self._all_repls.append(repl)

    def acquire(self, timeout: float = 30.0) -> SandboxREPL:
        """Acquire a REPL from the pool.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            A SandboxREPL instance.

        Raises:
            TimeoutError: If no REPL becomes available within the timeout.
            RuntimeError: If the pool has been shut down.
        """
        if self._shutdown:
            raise RuntimeError("Pool has been shut down")

        start = time.time()
        try:
            repl = self._pool.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"No REPL available within {timeout}s")

        wait_ms = (time.time() - start) * 1000

        # Health check and recycle if needed
        if not self._lifecycle.health_check(repl):
            repl = self._lifecycle.recycle(repl)

        with self._lock:
            self._in_use.add(id(repl))

        self._metrics_tracker.record_acquire(wait_ms)
        return repl

    def release(self, repl: SandboxREPL) -> None:
        """Release a REPL back to the pool.

        The REPL is recycled (reset) before being returned to the pool.

        Args:
            repl: The REPL to release.
        """
        if self._shutdown:
            self._lifecycle.destroy(repl)
            return

        hold_start = time.time()

        with self._lock:
            self._in_use.discard(id(repl))

        # Recycle the REPL
        recycled = self._lifecycle.recycle(repl)

        hold_ms = (time.time() - hold_start) * 1000
        self._metrics_tracker.record_release(hold_ms)

        try:
            self._pool.put_nowait(recycled)
        except queue.Full:
            self._lifecycle.destroy(recycled)

    def shutdown(self) -> None:
        """Shut down the pool and destroy all REPLs."""
        self._shutdown = True

        # Drain pool
        while not self._pool.empty():
            try:
                repl = self._pool.get_nowait()
                self._lifecycle.destroy(repl)
            except queue.Empty:
                break

        # Destroy any in-use REPLs
        for repl in self._all_repls:
            try:
                self._lifecycle.destroy(repl)
            except Exception:
                pass

    @property
    def available(self) -> int:
        """Number of available REPLs."""
        return self._pool.qsize()

    @property
    def total(self) -> int:
        """Total pool size."""
        return self._size

    def get_metrics(self) -> PoolMetrics:
        """Get current pool metrics.

        Returns:
            PoolMetrics snapshot.
        """
        summary = self._metrics_tracker.summary()
        with self._lock:
            in_use = len(self._in_use)

        return PoolMetrics(
            total=self._size,
            available=self._pool.qsize(),
            in_use=in_use,
            total_acquires=summary.total_acquires,
            total_releases=summary.total_releases,
        )
