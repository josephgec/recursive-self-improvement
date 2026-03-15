"""High-level REPL client for rsi-infra.

Wraps the pool / local-REPL layer with convenient context managers.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncIterator, Iterator

from repl.src.local_repl import LocalREPL
from repl.src.pool import REPLPool
from repl.src.sandbox import ExecutionResult, REPLConfig, SandboxREPL
from sdk.config import InfraConfig


class REPLClient:
    """Convenient wrapper around :class:`REPLPool` and :class:`SandboxREPL`.

    Usage::

        client = REPLClient.from_config(config)
        async with client.session() as repl:
            result = repl.execute("x = 1 + 2")
            assert result.success
        await client.shutdown()
    """

    def __init__(self, pool: REPLPool) -> None:
        self._pool = pool

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: InfraConfig) -> REPLClient:
        """Create a client backed by a :class:`REPLPool` from *config*."""
        repl_config = config.repl_config
        pool_size = config.repl_pool_size
        backend = config.repl_backend

        if backend == "local":
            factory = lambda: LocalREPL(config=repl_config)
        else:
            # Future: docker / modal factories
            factory = lambda: LocalREPL(config=repl_config)

        pool = REPLPool(pool_size=pool_size, config=repl_config, factory=factory)
        return cls(pool)

    # ------------------------------------------------------------------
    # Context managers
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def session(self) -> AsyncIterator[SandboxREPL]:
        """Acquire a REPL from the pool and release on exit.

        Example::

            async with client.session() as repl:
                repl.execute("x = 42")
        """
        repl = await self._pool.acquire()
        try:
            yield repl
        finally:
            await self._pool.release(repl)

    @contextmanager
    def session_sync(self) -> Iterator[SandboxREPL]:
        """Synchronous version of :meth:`session`.

        Runs the acquire/release in the current (or a new) event loop.
        """
        loop = _get_or_create_loop()
        repl = loop.run_until_complete(self._pool.acquire())
        try:
            yield repl
        finally:
            loop.run_until_complete(self._pool.release(repl))

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    async def execute(self, code: str) -> ExecutionResult:
        """One-shot execute: acquire, run, release."""
        async with self.session() as repl:
            return repl.execute(code)

    def execute_sync(self, code: str) -> ExecutionResult:
        """Synchronous one-shot execute."""
        with self.session_sync() as repl:
            return repl.execute(code)

    @asynccontextmanager
    async def child(self, parent: SandboxREPL) -> AsyncIterator[SandboxREPL]:
        """Spawn a child REPL from *parent*, shut it down on exit.

        The child inherits the parent's variables but is isolated.
        """
        child_repl = parent.spawn_child()
        try:
            yield child_repl
        finally:
            child_repl.shutdown()

    @contextmanager
    def child_sync(self, parent: SandboxREPL) -> Iterator[SandboxREPL]:
        """Synchronous version of :meth:`child`."""
        child_repl = parent.spawn_child()
        try:
            yield child_repl
        finally:
            child_repl.shutdown()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Shut down the underlying pool."""
        await self._pool.shutdown()

    def shutdown_sync(self) -> None:
        """Synchronous pool shutdown."""
        loop = _get_or_create_loop()
        loop.run_until_complete(self._pool.shutdown())

    @property
    def pool(self) -> REPLPool:
        return self._pool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_or_create_loop() -> asyncio.AbstractEventLoop:
    """Return the running event loop or create a new one."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
