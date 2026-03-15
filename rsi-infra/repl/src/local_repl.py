"""In-process sandboxed REPL using exec() with restricted builtins."""

from __future__ import annotations

import copy
from typing import Any

from repl.src.execution import execute_code
from repl.src.sandbox import ExecutionResult, REPLConfig, SandboxREPL
from repl.src.security import CodeAnalyzer, SafeImportHook, SecurityPolicy, SAFE_BUILTINS


class LocalREPL(SandboxREPL):
    """Sandboxed REPL that runs code in the current process.

    Security is enforced via:
    * Restricted ``__builtins__`` (only :data:`SAFE_BUILTINS`).
    * Static :class:`CodeAnalyzer` checks *before* execution.
    * Time and memory limits delegated to :func:`execute_code`.
    """

    def __init__(
        self,
        config: REPLConfig | None = None,
        depth: int = 0,
        policy: SecurityPolicy | None = None,
    ) -> None:
        super().__init__(config=config, depth=depth)
        self._policy = policy or SecurityPolicy()
        self._analyzer = CodeAnalyzer(self._policy)
        self._globals: dict[str, Any] = self._make_clean_globals()
        self._shutdown = False

    # ------------------------------------------------------------------
    # SandboxREPL interface
    # ------------------------------------------------------------------

    def execute(self, code: str) -> ExecutionResult:
        if self._shutdown:
            return ExecutionResult(
                success=False,
                error_type="ShutdownError",
                error_message="REPL has been shut down",
            )

        # Static security check
        is_safe, reason = self._analyzer.analyze(code)
        if not is_safe:
            return ExecutionResult(
                success=False,
                error_type="SecurityError",
                error_message=reason,
            )

        return execute_code(
            code,
            self._globals,
            timeout=self._config.timeout_seconds,
            max_output_bytes=self._policy.max_output_bytes,
        )

    def get_variable(self, name: str) -> Any:
        if name not in self._globals:
            raise KeyError(name)
        return self._globals[name]

    def set_variable(self, name: str, value: Any) -> None:
        self._globals[name] = value

    def list_variables(self) -> dict[str, Any]:
        return {
            k: v
            for k, v in self._globals.items()
            if not k.startswith("_") and k != "__builtins__"
        }

    def spawn_child(self) -> LocalREPL:
        new_depth = self._depth + 1
        if new_depth > self._config.max_recursion_depth:
            raise RecursionError(
                f"Maximum REPL recursion depth ({self._config.max_recursion_depth}) exceeded"
            )

        child = LocalREPL(
            config=self._config,
            depth=new_depth,
            policy=self._policy,
        )
        # Deep-copy user variables into the child namespace
        for k, v in self.list_variables().items():
            try:
                child._globals[k] = copy.deepcopy(v)
            except Exception:
                # If deep copy fails (e.g. for unpicklable objects), share ref
                child._globals[k] = v
        return child

    def reset(self) -> None:
        self._globals = self._make_clean_globals()

    def shutdown(self) -> None:
        self._globals.clear()
        self._shutdown = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_clean_globals(self) -> dict[str, Any]:
        """Return a fresh globals dict with restricted builtins and a safe
        ``__import__`` that only allows approved packages."""
        builtins = dict(SAFE_BUILTINS)
        builtins["__import__"] = SafeImportHook(self._config.allowed_packages)
        return {"__builtins__": builtins}
