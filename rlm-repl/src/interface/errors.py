"""Error hierarchy for the RLM-REPL sandbox."""


class REPLError(Exception):
    """Base exception for all REPL-related errors."""
    pass


class ExecutionTimeoutError(REPLError):
    """Raised when code execution exceeds the time limit."""

    def __init__(self, timeout_seconds: float, message: str = ""):
        self.timeout_seconds = timeout_seconds
        msg = message or f"Execution timed out after {timeout_seconds}s"
        super().__init__(msg)


class MemoryLimitError(REPLError):
    """Raised when code execution exceeds the memory limit."""

    def __init__(self, limit_mb: float, usage_mb: float = 0.0, message: str = ""):
        self.limit_mb = limit_mb
        self.usage_mb = usage_mb
        msg = message or f"Memory limit exceeded: {usage_mb:.1f}MB / {limit_mb:.1f}MB"
        super().__init__(msg)


class ForbiddenCodeError(REPLError):
    """Raised when code contains forbidden constructs detected by AST scanning."""

    def __init__(self, violations: list = None, message: str = ""):
        self.violations = violations or []
        if message:
            msg = message
        elif self.violations:
            details = "; ".join(str(v) for v in self.violations)
            msg = f"Forbidden code detected: {details}"
        else:
            msg = "Forbidden code detected"
        super().__init__(msg)


class RecursionDepthError(REPLError):
    """Raised when REPL spawn depth exceeds the maximum allowed."""

    def __init__(self, current_depth: int, max_depth: int, message: str = ""):
        self.current_depth = current_depth
        self.max_depth = max_depth
        msg = message or f"Spawn depth {current_depth} exceeds maximum {max_depth}"
        super().__init__(msg)


class OutputSizeLimitError(REPLError):
    """Raised when output exceeds the maximum allowed size."""

    def __init__(self, size: int, limit: int, message: str = ""):
        self.size = size
        self.limit = limit
        msg = message or f"Output size {size} exceeds limit {limit}"
        super().__init__(msg)


class REPLNotAliveError(REPLError):
    """Raised when attempting to use a REPL that has been shut down."""

    def __init__(self, message: str = "REPL is not alive"):
        super().__init__(message)


class SerializationError(REPLError):
    """Raised when a variable cannot be serialized or deserialized."""

    def __init__(self, variable_name: str = "", message: str = ""):
        self.variable_name = variable_name
        msg = message or f"Failed to serialize variable: {variable_name}"
        super().__init__(msg)


class CascadeKillError(REPLError):
    """Raised when cascade killing of child REPLs fails."""

    def __init__(self, repl_id: str = "", message: str = ""):
        self.repl_id = repl_id
        msg = message or f"Failed to cascade kill REPL tree from: {repl_id}"
        super().__init__(msg)
