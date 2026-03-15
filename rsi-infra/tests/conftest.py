"""Shared fixtures for rsi-infra integration tests.

All fixtures use local backends only — no Docker, Modal, or wandb required.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from repl.src.local_repl import LocalREPL
from repl.src.sandbox import REPLConfig
from sdk.config import InfraConfig
from sdk.repl_client import REPLClient
from sdk.symbolic_client import SymbolicClient
from sdk.tracking_client import TrackingClient
from tracking.src.local_backend import LocalTracker

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def local_config() -> InfraConfig:
    """InfraConfig loaded from configs/local.yaml with defaults merged."""
    return InfraConfig.from_yaml(_PROJECT_ROOT / "configs" / "local.yaml")


@pytest.fixture
def ci_config() -> InfraConfig:
    """InfraConfig loaded from configs/ci.yaml (tighter limits)."""
    return InfraConfig.from_yaml(_PROJECT_ROOT / "configs" / "ci.yaml")


# ---------------------------------------------------------------------------
# REPL fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def repl() -> LocalREPL:
    """A fresh LocalREPL with default config."""
    r = LocalREPL()
    yield r
    r.shutdown()


@pytest.fixture
def repl_client(local_config: InfraConfig) -> REPLClient:
    """A REPLClient backed by a LocalREPL pool.

    Creates a fresh event loop so the asyncio.Queue inside REPLPool works
    reliably across tests on Python 3.9.
    """
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    client = REPLClient.from_config(local_config)
    yield client
    # Clean up: shut down pool and close loop
    try:
        loop.run_until_complete(client.shutdown())
    except Exception:
        pass
    loop.close()


# ---------------------------------------------------------------------------
# Symbolic fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def symbolic_client(local_config: InfraConfig) -> SymbolicClient:
    """A SymbolicClient backed by subprocess runners."""
    return SymbolicClient.from_config(local_config)


# ---------------------------------------------------------------------------
# Tracking fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_tracking_dir():
    """Temporary directory for tracking data.  Cleaned up after test."""
    d = tempfile.mkdtemp(prefix="rsi_test_tracking_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def tracking_client(local_config: InfraConfig, tmp_tracking_dir: Path) -> TrackingClient:
    """A TrackingClient that writes to a temp directory."""
    client = TrackingClient.from_config(local_config)
    # Point the tracker at the temp dir
    if isinstance(client.tracker, LocalTracker):
        client.tracker._base_dir = tmp_tracking_dir
    return client
