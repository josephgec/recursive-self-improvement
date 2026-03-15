"""Tests for the REPL pool."""

from __future__ import annotations

import asyncio

import pytest

from repl.src.pool import REPLPool
from repl.src.sandbox import REPLConfig


@pytest.fixture
def config() -> REPLConfig:
    return REPLConfig(timeout_seconds=5, max_recursion_depth=3)


def _run(coro):
    """Run an async coroutine synchronously (works without pytest-asyncio)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestREPLPool:
    def test_acquire_all(self, config: REPLConfig) -> None:
        async def _test():
            pool = REPLPool(pool_size=2, config=config)
            r1 = await pool.acquire()
            r2 = await pool.acquire()
            assert pool.available == 0
            assert pool.in_use == 2
            await pool.release(r1)
            await pool.release(r2)
            await pool.shutdown()
        _run(_test())

    def test_release_makes_available(self, config: REPLConfig) -> None:
        async def _test():
            pool = REPLPool(pool_size=2, config=config)
            r1 = await pool.acquire()
            _ = await pool.acquire()
            assert pool.available == 0
            await pool.release(r1)
            assert pool.available == 1
            assert pool.in_use == 1
            await pool.shutdown()
        _run(_test())

    def test_released_repl_is_reset(self, config: REPLConfig) -> None:
        async def _test():
            pool = REPLPool(pool_size=1, config=config)
            repl = await pool.acquire()
            repl.execute("x = 42")
            assert repl.get_variable("x") == 42
            await pool.release(repl)

            # Acquire the same REPL again -- it should be clean
            repl2 = await pool.acquire()
            assert repl2.list_variables() == {}
            await pool.release(repl2)
            await pool.shutdown()
        _run(_test())

    def test_pool_size_property(self, config: REPLConfig) -> None:
        async def _test():
            pool = REPLPool(pool_size=3, config=config)
            assert pool.pool_size == 3
            assert pool.available == 3
            assert pool.in_use == 0
            await pool.shutdown()
        _run(_test())

    def test_shutdown(self, config: REPLConfig) -> None:
        async def _test():
            pool = REPLPool(pool_size=2, config=config)
            await pool.shutdown()
            # After shutdown the internal list is empty
            assert pool.available == 0
        _run(_test())
