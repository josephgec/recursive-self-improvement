"""Pool of reusable REPL instances for concurrent workloads."""

from __future__ import annotations

import asyncio
from typing import Callable

from repl.src.local_repl import LocalREPL
from repl.src.sandbox import REPLConfig, SandboxREPL


class REPLPool:
    """Manages a fixed-size pool of :class:`SandboxREPL` instances.

    REPLs are pre-created at construction time and handed out via
    :meth:`acquire` / :meth:`release`.  On release the REPL is reset
    so the next consumer gets a clean environment.

    Usage::

        pool = REPLPool(pool_size=4)
        repl = await pool.acquire()
        try:
            result = repl.execute("x = 1 + 1")
        finally:
            await pool.release(repl)
        await pool.shutdown()
    """

    def __init__(
        self,
        pool_size: int = 4,
        config: REPLConfig | None = None,
        factory: Callable[[], SandboxREPL] | None = None,
    ) -> None:
        self._config = config or REPLConfig()
        self._pool_size = pool_size
        self._factory = factory or self._default_factory

        self._queue: asyncio.Queue[SandboxREPL] = asyncio.Queue(maxsize=pool_size)
        self._all: list[SandboxREPL] = []
        self._in_use: set[int] = set()  # ids of checked-out REPLs

        for _ in range(pool_size):
            repl = self._factory()
            self._all.append(repl)
            self._queue.put_nowait(repl)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def acquire(self) -> SandboxREPL:
        """Acquire a REPL from the pool (waits if none available)."""
        repl = await self._queue.get()
        self._in_use.add(id(repl))
        return repl

    async def release(self, repl: SandboxREPL) -> None:
        """Return a REPL to the pool after resetting it."""
        repl.reset()
        self._in_use.discard(id(repl))
        await self._queue.put(repl)

    async def shutdown(self) -> None:
        """Shut down every REPL in the pool."""
        for repl in self._all:
            repl.shutdown()
        self._all.clear()
        # Drain the queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def available(self) -> int:
        """Number of REPLs currently available for checkout."""
        return self._queue.qsize()

    @property
    def in_use(self) -> int:
        """Number of REPLs currently checked out."""
        return len(self._in_use)

    @property
    def pool_size(self) -> int:
        return self._pool_size

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _default_factory(self) -> SandboxREPL:
        return LocalREPL(config=self._config)
