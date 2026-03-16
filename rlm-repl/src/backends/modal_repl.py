"""Modal cloud REPL backend (stub implementation)."""

from typing import Any, List, Optional

from src.interface.base import SandboxREPL
from src.interface.types import ExecutionResult
from src.safety.policy import SafetyPolicy


class ModalREPL(SandboxREPL):
    """Modal cloud-based REPL backend.

    This is a stub implementation. All methods raise NotImplementedError.
    """

    def __init__(self, policy: Optional[SafetyPolicy] = None):
        self._policy = policy or SafetyPolicy()

    def execute(self, code: str) -> ExecutionResult:
        raise NotImplementedError("ModalREPL is not yet implemented")

    def get_variable(self, name: str) -> Any:
        raise NotImplementedError("ModalREPL is not yet implemented")

    def set_variable(self, name: str, value: Any) -> None:
        raise NotImplementedError("ModalREPL is not yet implemented")

    def list_variables(self) -> List[str]:
        raise NotImplementedError("ModalREPL is not yet implemented")

    def spawn_child(self) -> "SandboxREPL":
        raise NotImplementedError("ModalREPL is not yet implemented")

    def snapshot(self) -> str:
        raise NotImplementedError("ModalREPL is not yet implemented")

    def restore(self, snapshot_id: str) -> None:
        raise NotImplementedError("ModalREPL is not yet implemented")

    def reset(self) -> None:
        raise NotImplementedError("ModalREPL is not yet implemented")

    def shutdown(self) -> None:
        raise NotImplementedError("ModalREPL is not yet implemented")

    def is_alive(self) -> bool:
        raise NotImplementedError("ModalREPL is not yet implemented")
