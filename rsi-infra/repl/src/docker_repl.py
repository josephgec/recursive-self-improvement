"""Docker-backed sandboxed REPL.

Provides strong isolation by running code inside a disposable container.
Falls back gracefully when Docker is not installed or the daemon is not
running.
"""

from __future__ import annotations

import json
import textwrap
from typing import Any

from repl.src.sandbox import ExecutionResult, REPLConfig, SandboxREPL
from repl.src.security import SecurityPolicy

# Docker SDK is an optional dependency.
try:
    import docker
    from docker.errors import DockerException
    _HAS_DOCKER = True
except ImportError:
    _HAS_DOCKER = False
    docker = None  # type: ignore[assignment]
    DockerException = Exception  # type: ignore[misc,assignment]


def _docker_available() -> bool:
    """Return True only if the Docker SDK is installed AND the daemon responds."""
    if not _HAS_DOCKER:
        return False
    try:
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


class DockerREPL(SandboxREPL):
    """Sandboxed REPL that executes code inside a Docker container.

    Container configuration:
    * ``network_mode="none"`` -- no network access by default.
    * ``mem_limit`` -- matches ``config.max_memory_mb``.
    * ``read_only=True`` with a ``/tmp`` tmpfs for scratch space.

    If Docker is not available the constructor raises ``RuntimeError`` with
    an actionable message.
    """

    def __init__(
        self,
        config: REPLConfig | None = None,
        depth: int = 0,
        policy: SecurityPolicy | None = None,
        image: str = "python:3.11-slim",
    ) -> None:
        super().__init__(config=config, depth=depth)
        self._policy = policy or SecurityPolicy()
        self._image = image
        self._variables: dict[str, Any] = {}

        if not _docker_available():
            raise RuntimeError(
                "Docker is not available. Either install the Docker SDK "
                "(`pip install docker`) and ensure the Docker daemon is "
                "running, or use backend='local' instead."
            )

        self._client = docker.from_env()
        self._container = self._create_container()

    # ------------------------------------------------------------------
    # Container lifecycle
    # ------------------------------------------------------------------

    def _create_container(self) -> Any:
        network_mode = "none" if not self._config.network_access else "bridge"
        mem_limit = f"{self._config.max_memory_mb}m"
        return self._client.containers.run(
            self._image,
            command="sleep infinity",
            detach=True,
            network_mode=network_mode,
            mem_limit=mem_limit,
            read_only=True,
            tmpfs={"/tmp": "size=256m"},
            remove=True,
        )

    # ------------------------------------------------------------------
    # SandboxREPL interface
    # ------------------------------------------------------------------

    def execute(self, code: str) -> ExecutionResult:
        wrapper = textwrap.dedent(f"""\
            import json, sys, time, traceback

            _code = {code!r}

            _ns = {{}}
            _start = time.perf_counter()
            _stdout = ""
            _stderr = ""
            _success = True
            _error_type = None
            _error_message = None

            import io, contextlib
            _sout = io.StringIO()
            _serr = io.StringIO()
            try:
                with contextlib.redirect_stdout(_sout), contextlib.redirect_stderr(_serr):
                    exec(_code, _ns)
            except Exception as _exc:
                _success = False
                _error_type = type(_exc).__name__
                _error_message = traceback.format_exc()

            _elapsed = (time.perf_counter() - _start) * 1000
            _vars = {{k: repr(v) for k, v in _ns.items() if not k.startswith("_")}}
            print(json.dumps({{
                "stdout": _sout.getvalue(),
                "stderr": _serr.getvalue(),
                "success": _success,
                "error_type": _error_type,
                "error_message": _error_message,
                "execution_time_ms": _elapsed,
                "variables": _vars,
            }}))
        """)

        exit_code, output = self._container.exec_run(
            ["python", "-c", wrapper],
            demux=True,
        )
        stdout_raw = (output[0] or b"").decode()
        stderr_raw = (output[1] or b"").decode()

        try:
            data = json.loads(stdout_raw)
        except json.JSONDecodeError:
            return ExecutionResult(
                stdout=stdout_raw,
                stderr=stderr_raw,
                success=False,
                error_type="ContainerError",
                error_message="Failed to parse container output",
            )

        return ExecutionResult(
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            success=data.get("success", False),
            error_type=data.get("error_type"),
            error_message=data.get("error_message"),
            execution_time_ms=data.get("execution_time_ms", 0),
            variables=data.get("variables", {}),
        )

    def get_variable(self, name: str) -> Any:
        if name not in self._variables:
            raise KeyError(name)
        return self._variables[name]

    def set_variable(self, name: str, value: Any) -> None:
        self._variables[name] = value

    def list_variables(self) -> dict[str, Any]:
        return dict(self._variables)

    def spawn_child(self) -> DockerREPL:
        new_depth = self._depth + 1
        if new_depth > self._config.max_recursion_depth:
            raise RecursionError(
                f"Maximum REPL recursion depth ({self._config.max_recursion_depth}) exceeded"
            )
        child = DockerREPL(
            config=self._config,
            depth=new_depth,
            policy=self._policy,
            image=self._image,
        )
        child._variables = dict(self._variables)
        return child

    def reset(self) -> None:
        self._variables.clear()
        try:
            self._container.kill()
        except Exception:
            pass
        self._container = self._create_container()

    def shutdown(self) -> None:
        self._variables.clear()
        try:
            self._container.kill()
        except Exception:
            pass
