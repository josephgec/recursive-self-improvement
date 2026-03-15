"""Tests for DockerREPL with fully mocked Docker SDK."""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from repl.src.sandbox import ExecutionResult, REPLConfig


# ---------------------------------------------------------------------------
# Helper: build a mock docker module
# ---------------------------------------------------------------------------

def _make_mock_docker():
    """Create a mock docker module that simulates a working Docker SDK."""
    mock_docker = MagicMock()
    mock_docker.errors = MagicMock()
    mock_docker.errors.DockerException = type("DockerException", (Exception,), {})

    mock_client = MagicMock()
    mock_client.ping.return_value = True

    mock_container = MagicMock()
    mock_docker.from_env.return_value = mock_client
    mock_client.containers.run.return_value = mock_container

    return mock_docker, mock_client, mock_container


# ---------------------------------------------------------------------------
# Tests: Docker unavailable
# ---------------------------------------------------------------------------

class TestDockerUnavailable:
    """DockerREPL raises RuntimeError when Docker is not available."""

    def test_raises_when_docker_sdk_not_installed(self) -> None:
        """When the docker module cannot be imported, constructing DockerREPL raises."""
        # Patch _HAS_DOCKER to False at module level
        with patch("repl.src.docker_repl._HAS_DOCKER", False), \
             patch("repl.src.docker_repl._docker_available", return_value=False):
            from repl.src.docker_repl import DockerREPL
            with pytest.raises(RuntimeError, match="Docker is not available"):
                DockerREPL()

    def test_raises_when_daemon_not_running(self) -> None:
        """When docker SDK is installed but daemon not responding, raises."""
        with patch("repl.src.docker_repl._docker_available", return_value=False):
            from repl.src.docker_repl import DockerREPL
            with pytest.raises(RuntimeError, match="Docker is not available"):
                DockerREPL()


# ---------------------------------------------------------------------------
# Tests: Docker available (mocked)
# ---------------------------------------------------------------------------

class TestDockerREPLWithMockedDocker:
    """Test DockerREPL with a fully mocked Docker SDK."""

    @pytest.fixture
    def docker_repl(self):
        """Create a DockerREPL with mocked Docker SDK."""
        mock_docker, mock_client, mock_container = _make_mock_docker()

        with patch("repl.src.docker_repl._docker_available", return_value=True), \
             patch("repl.src.docker_repl._HAS_DOCKER", True), \
             patch("repl.src.docker_repl.docker", mock_docker):
            from repl.src.docker_repl import DockerREPL
            config = REPLConfig(timeout_seconds=5, max_recursion_depth=3)
            repl = DockerREPL(config=config)
            repl._mock_docker = mock_docker
            repl._mock_client = mock_client
            repl._mock_container = mock_container
            yield repl

    def test_initialization(self, docker_repl) -> None:
        """DockerREPL initializes successfully with mocked Docker."""
        assert docker_repl._container is not None
        assert docker_repl._config.timeout_seconds == 5
        assert docker_repl.depth == 0

    def test_execute_returns_result(self, docker_repl) -> None:
        """execute() parses JSON output from container exec_run."""
        result_json = json.dumps({
            "stdout": "hello\n",
            "stderr": "",
            "success": True,
            "error_type": None,
            "error_message": None,
            "execution_time_ms": 1.23,
            "variables": {"x": "42"},
        })
        docker_repl._container.exec_run.return_value = (
            0,
            (result_json.encode(), b""),
        )

        result = docker_repl.execute("x = 42; print('hello')")
        assert isinstance(result, ExecutionResult)
        assert result.success is True
        assert result.stdout == "hello\n"
        assert result.variables == {"x": "42"}
        assert result.execution_time_ms == 1.23

    def test_execute_handles_json_decode_error(self, docker_repl) -> None:
        """execute() handles malformed JSON output gracefully."""
        docker_repl._container.exec_run.return_value = (
            1,
            (b"not valid json", b"some error"),
        )

        result = docker_repl.execute("bad code")
        assert result.success is False
        assert result.error_type == "ContainerError"
        assert "Failed to parse" in result.error_message

    def test_execute_handles_none_output(self, docker_repl) -> None:
        """execute() handles None stdout/stderr from container."""
        docker_repl._container.exec_run.return_value = (
            0,
            (None, None),
        )

        result = docker_repl.execute("pass")
        assert result.success is False  # Empty output can't be parsed as JSON

    def test_execute_error_in_code(self, docker_repl) -> None:
        """execute() returns error details when code raises inside container."""
        result_json = json.dumps({
            "stdout": "",
            "stderr": "",
            "success": False,
            "error_type": "ZeroDivisionError",
            "error_message": "division by zero",
            "execution_time_ms": 0.5,
            "variables": {},
        })
        docker_repl._container.exec_run.return_value = (
            0,
            (result_json.encode(), b""),
        )

        result = docker_repl.execute("1/0")
        assert result.success is False
        assert result.error_type == "ZeroDivisionError"

    def test_get_set_list_variables(self, docker_repl) -> None:
        """Variable management works via the internal dict."""
        docker_repl.set_variable("x", 42)
        assert docker_repl.get_variable("x") == 42
        assert docker_repl.list_variables() == {"x": 42}

    def test_get_variable_missing_raises_keyerror(self, docker_repl) -> None:
        """get_variable raises KeyError for missing variables."""
        with pytest.raises(KeyError, match="missing"):
            docker_repl.get_variable("missing")

    def test_spawn_child_creates_new_container(self, docker_repl) -> None:
        """spawn_child creates a new DockerREPL with incremented depth."""
        docker_repl.set_variable("x", 100)

        with patch("repl.src.docker_repl._docker_available", return_value=True), \
             patch("repl.src.docker_repl._HAS_DOCKER", True), \
             patch("repl.src.docker_repl.docker", docker_repl._mock_docker):
            child = docker_repl.spawn_child()
            assert child.depth == 1
            assert child.get_variable("x") == 100
            # New container created for child
            assert docker_repl._mock_client.containers.run.call_count >= 2

    def test_spawn_child_max_depth_exceeded(self, docker_repl) -> None:
        """spawn_child raises RecursionError when depth limit exceeded."""
        docker_repl._depth = 3  # max_recursion_depth=3

        with patch("repl.src.docker_repl._docker_available", return_value=True), \
             patch("repl.src.docker_repl._HAS_DOCKER", True), \
             patch("repl.src.docker_repl.docker", docker_repl._mock_docker):
            with pytest.raises(RecursionError, match="exceeded"):
                docker_repl.spawn_child()

    def test_reset_kills_and_recreates_container(self, docker_repl) -> None:
        """reset() kills the container and creates a new one."""
        docker_repl.set_variable("x", 42)
        old_container = docker_repl._container

        docker_repl.reset()

        old_container.kill.assert_called_once()
        assert docker_repl.list_variables() == {}
        # containers.run called again for new container
        assert docker_repl._mock_client.containers.run.call_count >= 2

    def test_reset_tolerates_kill_failure(self, docker_repl) -> None:
        """reset() does not crash if container.kill() raises."""
        docker_repl._container.kill.side_effect = Exception("already dead")
        docker_repl.reset()  # Should not raise

    def test_shutdown_kills_container(self, docker_repl) -> None:
        """shutdown() kills the container and clears variables."""
        docker_repl.set_variable("x", 42)
        docker_repl.shutdown()

        docker_repl._container.kill.assert_called()
        assert docker_repl.list_variables() == {}

    def test_shutdown_tolerates_kill_failure(self, docker_repl) -> None:
        """shutdown() does not crash if container.kill() raises."""
        docker_repl._container.kill.side_effect = Exception("already dead")
        docker_repl.shutdown()  # Should not raise


# ---------------------------------------------------------------------------
# Tests: _docker_available helper
# ---------------------------------------------------------------------------

class TestDockerAvailableHelper:
    """Test the _docker_available() helper function."""

    def test_returns_false_when_no_sdk(self) -> None:
        with patch("repl.src.docker_repl._HAS_DOCKER", False):
            from repl.src.docker_repl import _docker_available
            assert _docker_available() is False

    def test_returns_false_when_ping_fails(self) -> None:
        mock_docker = MagicMock()
        mock_docker.from_env.side_effect = Exception("daemon not running")

        with patch("repl.src.docker_repl._HAS_DOCKER", True), \
             patch("repl.src.docker_repl.docker", mock_docker):
            from repl.src.docker_repl import _docker_available
            assert _docker_available() is False

    def test_returns_true_when_working(self) -> None:
        mock_docker = MagicMock()
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_docker.from_env.return_value = mock_client

        with patch("repl.src.docker_repl._HAS_DOCKER", True), \
             patch("repl.src.docker_repl.docker", mock_docker):
            from repl.src.docker_repl import _docker_available
            assert _docker_available() is True
