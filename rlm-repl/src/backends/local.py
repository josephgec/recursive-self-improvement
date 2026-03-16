"""Local exec()-based REPL backend with full safety stack."""

import io
import sys
import time
import traceback
import uuid
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, List, Optional

from src.interface.base import SandboxREPL
from src.interface.types import ExecutionResult
from src.interface.errors import (
    ExecutionTimeoutError,
    ForbiddenCodeError,
    RecursionDepthError,
    REPLNotAliveError,
)
from src.safety.ast_scanner import ASTScanner
from src.safety.timeout import TimeoutEnforcer
from src.safety.output_limiter import OutputLimiter
from src.safety.memory_limiter import MemoryLimiter
from src.safety.policy import SafetyPolicy
from src.memory.variable_store import VariableStore
from src.memory.snapshot import SnapshotManager
from src.memory.child_memory import ChildMemoryManager
from src.protocol.final_functions import FinalProtocol


class LocalREPL(SandboxREPL):
    """Local exec()-based sandboxed REPL.

    Executes code in a restricted namespace with:
    - AST scanning for forbidden patterns
    - Timeout enforcement
    - Output size limiting
    - Memory monitoring
    - Variable tracking
    - Snapshot/restore support
    - Child REPL spawning
    """

    # Restricted builtins - only safe operations allowed
    SAFE_BUILTINS = {
        "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes",
        "callable", "chr", "complex", "dict", "divmod", "enumerate",
        "filter", "float", "format", "frozenset", "hash", "hex", "id",
        "int", "isinstance", "issubclass", "iter", "len", "list",
        "map", "max", "min", "next", "object", "oct", "ord", "pow",
        "print", "property", "range", "repr", "reversed", "round",
        "set", "slice", "sorted", "str", "sum", "tuple", "zip",
        "True", "False", "None",
        "ArithmeticError", "AssertionError", "AttributeError",
        "BaseException", "BlockingIOError", "BrokenPipeError",
        "BufferError", "BytesWarning", "ChildProcessError",
        "ConnectionAbortedError", "ConnectionError", "ConnectionRefusedError",
        "ConnectionResetError", "DeprecationWarning", "EOFError",
        "EnvironmentError", "Exception", "FileExistsError",
        "FileNotFoundError", "FloatingPointError", "FutureWarning",
        "GeneratorExit", "IOError", "ImportError", "ImportWarning",
        "IndentationError", "IndexError", "InterruptedError",
        "IsADirectoryError", "KeyError", "KeyboardInterrupt",
        "LookupError", "MemoryError", "ModuleNotFoundError",
        "NameError", "NotADirectoryError", "NotImplementedError",
        "OSError", "OverflowError", "PendingDeprecationWarning",
        "PermissionError", "ProcessLookupError", "RecursionError",
        "ReferenceError", "ResourceWarning", "RuntimeError",
        "RuntimeWarning", "StopAsyncIteration", "StopIteration",
        "SyntaxError", "SyntaxWarning", "SystemError",
        "SystemExit", "TabError", "TimeoutError", "TypeError",
        "UnboundLocalError", "UnicodeDecodeError", "UnicodeEncodeError",
        "UnicodeError", "UnicodeTranslationError", "UnicodeWarning",
        "UserWarning", "ValueError", "Warning", "ZeroDivisionError",
    }

    def __init__(
        self,
        policy: Optional[SafetyPolicy] = None,
        repl_id: Optional[str] = None,
        depth: int = 0,
        parent_id: Optional[str] = None,
    ):
        self._policy = policy or SafetyPolicy()
        self._id = repl_id or str(uuid.uuid4())[:8]
        self._depth = depth
        self._parent_id = parent_id
        self._alive = True

        # Safety components
        self._scanner = ASTScanner(self._policy)
        self._timeout = TimeoutEnforcer(self._policy.timeout_seconds)
        self._output_limiter = OutputLimiter(self._policy.max_output_chars)
        self._memory_limiter = MemoryLimiter(self._policy.max_memory_mb)

        # Memory components
        self._var_store = VariableStore()
        self._snapshot_mgr = SnapshotManager()
        self._child_memory = ChildMemoryManager()

        # Protocol
        self._final_protocol = FinalProtocol()

        # Initialize namespace with restricted builtins
        import builtins
        safe_builtins = {
            name: getattr(builtins, name)
            for name in self.SAFE_BUILTINS
            if hasattr(builtins, name)
        }
        self._namespace: Dict[str, Any] = {"__builtins__": safe_builtins}

        # Inject FINAL protocol
        self._final_protocol.inject(self._namespace)

        # Track children
        self._children: List["LocalREPL"] = []

    @property
    def repl_id(self) -> str:
        return self._id

    @property
    def depth(self) -> int:
        return self._depth

    def execute(self, code: str) -> ExecutionResult:
        """Execute code in the sandbox."""
        if not self._alive:
            raise REPLNotAliveError()

        start_time = time.time()

        # Layer 1: AST scanning
        scan_result = self._scanner.scan(code)
        if not scan_result.safe:
            error_violations = [v for v in scan_result.violations if v.severity == "error"]
            if error_violations:
                raise ForbiddenCodeError(
                    violations=error_violations,
                    message=str(error_violations[0]),
                )

        # Capture pre-execution state for diff
        pre_state = dict(self._namespace)

        # Execute with timeout and capture output
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        error = None
        error_type = None
        killed = False
        kill_reason = None

        def run_code():
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(compile(code, "<sandbox>", "exec"), self._namespace)

        try:
            # Layer 2: Timeout enforcement
            self._timeout.execute_with_timeout(run_code, timeout=self._policy.timeout_seconds)
        except ExecutionTimeoutError as e:
            killed = True
            kill_reason = "timeout"
            error = str(e)
            error_type = "ExecutionTimeoutError"
        except ForbiddenCodeError:
            raise
        except Exception as e:
            error = str(e)
            error_type = type(e).__name__
            # Capture traceback to stderr
            tb = traceback.format_exc()
            stderr_buf.write(tb)

        elapsed_ms = (time.time() - start_time) * 1000

        # Get output
        stdout = stdout_buf.getvalue()
        stderr = stderr_buf.getvalue()

        # Layer 4: Output limiting
        stdout = self._output_limiter.truncate(stdout)
        stderr = self._output_limiter.truncate(stderr)

        # Track memory
        mem_status = self._memory_limiter.monitor()

        # Compute variable diff
        self._var_store.set_namespace(self._namespace)
        diff = self._var_store.diff(pre_state)
        variables_changed = diff.changed

        return ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            error=error,
            error_type=error_type,
            execution_time_ms=elapsed_ms,
            memory_peak_mb=mem_status.peak_mb,
            variables_changed=variables_changed,
            killed=killed,
            kill_reason=kill_reason,
        )

    def get_variable(self, name: str) -> Any:
        """Retrieve a variable from the sandbox."""
        if not self._alive:
            raise REPLNotAliveError()
        if name not in self._namespace:
            raise KeyError(f"Variable '{name}' not found")
        return self._namespace[name]

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable in the sandbox."""
        if not self._alive:
            raise REPLNotAliveError()
        self._namespace[name] = value

    def list_variables(self) -> List[str]:
        """List all user-defined variables."""
        if not self._alive:
            raise REPLNotAliveError()
        self._var_store.set_namespace(self._namespace)
        return self._var_store.list_all()

    def spawn_child(self) -> "LocalREPL":
        """Spawn a child REPL inheriting variables."""
        if not self._alive:
            raise REPLNotAliveError()

        if self._depth + 1 > self._policy.max_spawn_depth:
            raise RecursionDepthError(
                current_depth=self._depth + 1,
                max_depth=self._policy.max_spawn_depth,
            )

        child = LocalREPL(
            policy=self._policy,
            depth=self._depth + 1,
            parent_id=self._id,
        )

        # Copy parent namespace to child
        child_ns = self._child_memory.prepare_child_namespace(self._namespace)
        for name, value in child_ns.items():
            child._namespace[name] = value

        # Re-inject FINAL protocol in child
        self._final_protocol.inject(child._namespace)

        self._children.append(child)
        return child

    def snapshot(self) -> str:
        """Take a snapshot of the current state."""
        if not self._alive:
            raise REPLNotAliveError()
        return self._snapshot_mgr.take(self._namespace)

    def restore(self, snapshot_id: str) -> None:
        """Restore to a previous snapshot."""
        if not self._alive:
            raise REPLNotAliveError()
        restored = self._snapshot_mgr.restore(snapshot_id)
        # Keep builtins and FINAL protocol
        builtins_ref = self._namespace.get("__builtins__")
        self._namespace.clear()
        if builtins_ref is not None:
            self._namespace["__builtins__"] = builtins_ref
        self._namespace.update(restored)
        self._final_protocol.inject(self._namespace)

    def reset(self) -> None:
        """Reset to a clean state."""
        if not self._alive:
            raise REPLNotAliveError()
        builtins_ref = self._namespace.get("__builtins__")
        self._namespace.clear()
        if builtins_ref is not None:
            self._namespace["__builtins__"] = builtins_ref
        self._final_protocol.inject(self._namespace)

    def shutdown(self) -> None:
        """Shut down the REPL."""
        # Shut down children first
        for child in self._children:
            if child.is_alive():
                child.shutdown()
        self._alive = False
        self._namespace.clear()

    def is_alive(self) -> bool:
        """Check if the REPL is running."""
        return self._alive
