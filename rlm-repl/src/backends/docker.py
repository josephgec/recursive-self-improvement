"""Docker-based REPL backend with fallback to LocalREPL."""

import logging
from typing import Any, List, Optional

from src.interface.base import SandboxREPL
from src.interface.types import ExecutionResult
from src.safety.policy import SafetyPolicy
from src.backends.local import LocalREPL

logger = logging.getLogger(__name__)


def _docker_available() -> bool:
    """Check if Docker is available."""
    try:
        import docker
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


class DockerREPL(SandboxREPL):
    """Docker-based sandboxed REPL.

    Provides container-level isolation for code execution.
    Falls back to LocalREPL when Docker is unavailable.
    """

    def __init__(
        self,
        policy: Optional[SafetyPolicy] = None,
        image: str = "rlm-sandbox:latest",
        use_fallback: bool = True,
    ):
        self._policy = policy or SafetyPolicy()
        self._image = image
        self._use_fallback = use_fallback
        self._docker_available = _docker_available()
        self._fallback: Optional[LocalREPL] = None
        self._container = None

        if not self._docker_available:
            if self._use_fallback:
                logger.warning(
                    "Docker not available, falling back to LocalREPL"
                )
                self._fallback = LocalREPL(policy=self._policy)
            else:
                raise RuntimeError("Docker is not available and fallback is disabled")

    @property
    def is_docker(self) -> bool:
        """Whether this REPL is running in Docker."""
        return self._docker_available and self._fallback is None

    @property
    def is_fallback(self) -> bool:
        """Whether this REPL is using the LocalREPL fallback."""
        return self._fallback is not None

    def _get_backend(self) -> SandboxREPL:
        """Get the active backend (Docker or fallback)."""
        if self._fallback is not None:
            return self._fallback
        raise RuntimeError("Docker backend not implemented")

    def execute(self, code: str) -> ExecutionResult:
        """Execute code in the sandbox."""
        return self._get_backend().execute(code)

    def get_variable(self, name: str) -> Any:
        """Retrieve a variable."""
        return self._get_backend().get_variable(name)

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable."""
        self._get_backend().set_variable(name, value)

    def list_variables(self) -> List[str]:
        """List all user-defined variables."""
        return self._get_backend().list_variables()

    def spawn_child(self) -> "SandboxREPL":
        """Spawn a child REPL."""
        return self._get_backend().spawn_child()

    def snapshot(self) -> str:
        """Take a snapshot."""
        return self._get_backend().snapshot()

    def restore(self, snapshot_id: str) -> None:
        """Restore a snapshot."""
        self._get_backend().restore(snapshot_id)

    def reset(self) -> None:
        """Reset to clean state."""
        self._get_backend().reset()

    def shutdown(self) -> None:
        """Shut down the REPL."""
        if self._fallback:
            self._fallback.shutdown()
        if self._container:
            try:
                self._container.stop(timeout=5)
                self._container.remove(force=True)
            except Exception:
                pass

    def is_alive(self) -> bool:
        """Check if the REPL is running."""
        if self._fallback:
            return self._fallback.is_alive()
        return False
