"""Modal-backed sandboxed REPL (stub).

This module defines the interface for a Modal cloud REPL backend.
Modal (https://modal.com) provides serverless GPU containers that are
well-suited for heavy computation, but requires an account and
``MODAL_TOKEN_ID`` / ``MODAL_TOKEN_SECRET`` environment variables.

To implement:
1. ``modal.App`` with a custom image including allowed_packages.
2. ``@app.function(gpu=...)`` for GPU-enabled execution.
3. Serialise namespace via pickle, send to the remote function, exec,
   and return the updated namespace + captured output.

The interface mirrors :class:`SandboxREPL` exactly so it can be used as
a drop-in replacement for :class:`LocalREPL` or :class:`DockerREPL`.
"""

from __future__ import annotations

from typing import Any

from repl.src.sandbox import ExecutionResult, REPLConfig, SandboxREPL


class ModalREPL(SandboxREPL):
    """Stub REPL backend that delegates to Modal serverless containers.

    All methods raise :class:`NotImplementedError` with a helpful message
    until Modal integration is configured.
    """

    def __init__(
        self,
        config: REPLConfig | None = None,
        depth: int = 0,
    ) -> None:
        super().__init__(config=config, depth=depth)
        # Eagerly check so callers fail fast.
        self._check_modal()

    @staticmethod
    def _check_modal() -> None:
        try:
            import modal  # noqa: F401
        except ImportError:
            raise NotImplementedError(
                "Modal is not installed. Install it with `pip install modal` "
                "and set MODAL_TOKEN_ID / MODAL_TOKEN_SECRET environment "
                "variables, then re-implement ModalREPL."
            )
        raise NotImplementedError(
            "ModalREPL is a stub. To enable it:\n"
            "  1. Create a modal.App with a custom image.\n"
            "  2. Implement execute() to ship code + namespace to a "
            "remote @app.function.\n"
            "  3. Handle GPU allocation via config.gpu_enabled.\n"
            "See the module docstring for details."
        )

    # ------------------------------------------------------------------
    # SandboxREPL interface (all raise NotImplementedError)
    # ------------------------------------------------------------------

    def execute(self, code: str) -> ExecutionResult:
        raise NotImplementedError("ModalREPL.execute is not implemented")

    def get_variable(self, name: str) -> Any:
        raise NotImplementedError("ModalREPL.get_variable is not implemented")

    def set_variable(self, name: str, value: Any) -> None:
        raise NotImplementedError("ModalREPL.set_variable is not implemented")

    def list_variables(self) -> dict[str, Any]:
        raise NotImplementedError("ModalREPL.list_variables is not implemented")

    def spawn_child(self) -> ModalREPL:
        raise NotImplementedError("ModalREPL.spawn_child is not implemented")

    def reset(self) -> None:
        raise NotImplementedError("ModalREPL.reset is not implemented")

    def shutdown(self) -> None:
        raise NotImplementedError("ModalREPL.shutdown is not implemented")
