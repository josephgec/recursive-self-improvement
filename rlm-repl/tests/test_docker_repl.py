"""Tests for the DockerREPL backend."""

import pytest
from src.backends.docker import DockerREPL
from src.safety.policy import SafetyPolicy


class TestDockerREPL:
    """Test DockerREPL with fallback to LocalREPL."""

    def test_creates_with_fallback(self):
        """DockerREPL should fall back to LocalREPL when Docker is unavailable."""
        repl = DockerREPL(use_fallback=True)
        assert repl.is_fallback
        assert repl.is_alive()
        repl.shutdown()

    def test_execute_via_fallback(self):
        repl = DockerREPL(use_fallback=True)
        result = repl.execute("x = 42")
        assert result.success
        assert repl.get_variable("x") == 42
        repl.shutdown()

    def test_variables_via_fallback(self):
        repl = DockerREPL(use_fallback=True)
        repl.set_variable("y", 99)
        assert repl.get_variable("y") == 99
        assert "y" in repl.list_variables()
        repl.shutdown()

    def test_snapshot_via_fallback(self):
        repl = DockerREPL(use_fallback=True)
        repl.execute("x = 1")
        snap = repl.snapshot()
        repl.execute("x = 2")
        repl.restore(snap)
        assert repl.get_variable("x") == 1
        repl.shutdown()

    def test_spawn_child_via_fallback(self):
        repl = DockerREPL(use_fallback=True)
        repl.execute("x = 10")
        child = repl.spawn_child()
        assert child.get_variable("x") == 10
        child.shutdown()
        repl.shutdown()

    def test_reset_via_fallback(self):
        repl = DockerREPL(use_fallback=True)
        repl.execute("x = 1")
        repl.reset()
        with pytest.raises(KeyError):
            repl.get_variable("x")
        repl.shutdown()

    def test_shutdown_via_fallback(self):
        repl = DockerREPL(use_fallback=True)
        repl.shutdown()
        assert not repl.is_alive()

    def test_is_docker_false_without_docker(self):
        repl = DockerREPL(use_fallback=True)
        assert not repl.is_docker
        repl.shutdown()

    def test_with_policy(self):
        policy = SafetyPolicy(timeout_seconds=5)
        repl = DockerREPL(policy=policy, use_fallback=True)
        result = repl.execute("x = 1")
        assert result.success
        repl.shutdown()

    def test_no_fallback_raises(self):
        """When Docker is unavailable and fallback is disabled, should raise."""
        with pytest.raises(RuntimeError):
            DockerREPL(use_fallback=False)
