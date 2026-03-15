"""Security policy and static code analysis for the REPL sandbox."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field


# Modules that must never be imported inside the sandbox.
BLOCKED_MODULES: frozenset[str] = frozenset({
    "subprocess", "os", "ctypes", "socket", "http", "urllib", "requests",
})

# String patterns that are blocked even if they appear as attribute access.
BLOCKED_ATTRIBUTE_PATTERNS: frozenset[str] = frozenset({
    "__class__", "__subclasses__", "__mro__", "__globals__",
    "__bases__", "__code__", "__reduce__",
})

# Built-in names that must be blocked.
BLOCKED_BUILTINS: frozenset[str] = frozenset({
    "exec", "eval", "compile", "__import__", "breakpoint",
})

# Built-in names explicitly allowed in the restricted namespace.
SAFE_BUILTINS: dict[str, object] = {
    "print": print,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "type": type,
    "str": str,
    "int": int,
    "float": float,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "bool": bool,
    "bytes": bytes,
    "bytearray": bytearray,
    "repr": repr,
    "hash": hash,
    "id": id,
    "iter": iter,
    "next": next,
    "reversed": reversed,
    "slice": slice,
    "getattr": getattr,
    "setattr": setattr,
    "hasattr": hasattr,
    "callable": callable,
    "chr": chr,
    "ord": ord,
    "hex": hex,
    "oct": oct,
    "bin": bin,
    "any": any,
    "all": all,
    "dir": dir,
    "vars": vars,
    "format": format,
    "property": property,
    "staticmethod": staticmethod,
    "classmethod": classmethod,
    "super": super,
    "object": object,
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "RuntimeError": RuntimeError,
    "StopIteration": StopIteration,
    "NotImplementedError": NotImplementedError,
    "ZeroDivisionError": ZeroDivisionError,
    "ArithmeticError": ArithmeticError,
    "OverflowError": OverflowError,
    "RecursionError": RecursionError,
    "True": True,
    "False": False,
    "None": None,
}


@dataclass
class SecurityPolicy:
    """Configurable security policy for a sandbox."""

    allow_network: bool = False
    allow_filesystem_read: bool = False
    writable_paths: list[str] = field(default_factory=lambda: ["/tmp/repl_workspace"])
    blocked_modules: frozenset[str] = BLOCKED_MODULES
    blocked_builtins: frozenset[str] = BLOCKED_BUILTINS
    max_output_bytes: int = 1_048_576  # 1 MiB


class CodeAnalyzer:
    """Static analysis of Python source code for security violations.

    Uses the ``ast`` module to inspect the code *before* execution so that
    obviously dangerous operations are rejected without ever running them.
    """

    def __init__(self, policy: SecurityPolicy | None = None) -> None:
        self.policy = policy or SecurityPolicy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, code: str) -> tuple[bool, str | None]:
        """Analyse *code* and return ``(is_safe, rejection_reason)``.

        Returns ``(True, None)`` when no violations are found.
        """
        # 1. Try to parse -----------------------------------------------
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            # Syntax errors are not *security* violations -- let execution
            # surface them normally.
            return True, None

        # 2. Walk the AST -----------------------------------------------
        for node in ast.walk(tree):
            safe, reason = self._check_node(node)
            if not safe:
                return False, reason

        # 3. String-level checks (catches things AST doesn't expose) -----
        for pattern in BLOCKED_ATTRIBUTE_PATTERNS:
            if pattern in code:
                return False, f"Access to '{pattern}' is blocked"

        return True, None

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    def _check_node(self, node: ast.AST) -> tuple[bool, str | None]:
        """Return (is_safe, reason) for a single AST node."""

        # --- Import / ImportFrom ----------------------------------------
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in self.policy.blocked_modules:
                    return False, f"Import of '{alias.name}' is blocked"

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top in self.policy.blocked_modules:
                    return False, f"Import from '{node.module}' is blocked"

        # --- Calls (blocked builtins, os.system, open) -------------------
        if isinstance(node, ast.Call):
            func = node.func
            # Direct call: eval(...), exec(...)
            if isinstance(func, ast.Name):
                if func.id in self.policy.blocked_builtins:
                    return False, f"Call to '{func.id}' is blocked"
                if func.id == "open" and not self.policy.allow_filesystem_read:
                    return False, "Call to 'open' is blocked (filesystem access denied)"
            # Attribute call: os.system(...)
            if isinstance(func, ast.Attribute):
                if func.attr == "system":
                    return False, "Call to 'os.system' is blocked"

        return True, None


class SafeImportHook:
    """A callable that replaces ``__import__`` in restricted builtins.

    Only allows importing modules whose top-level package name appears in
    *allowed_packages* or is a standard-library / built-in module.
    """

    # Packages that are always allowed (stdlib essentials used internally).
    _ALWAYS_ALLOWED: frozenset[str] = frozenset({
        "math", "cmath", "decimal", "fractions", "statistics",
        "collections", "itertools", "functools", "operator",
        "string", "re", "json", "copy", "dataclasses",
        "datetime", "time", "random", "hashlib", "hmac",
        "io", "contextlib", "abc", "typing", "types",
        "enum", "textwrap", "pprint",
    })

    def __init__(self, allowed_packages: list[str] | None = None) -> None:
        extras = frozenset(allowed_packages or [])
        # Normalise: z3-solver -> z3
        normalised = frozenset(p.replace("-", "_").split(".")[0] for p in extras)
        self._allowed = self._ALWAYS_ALLOWED | normalised | frozenset({"z3"})

    def __call__(
        self,
        name: str,
        globals: dict | None = None,
        locals: dict | None = None,
        fromlist: tuple = (),
        level: int = 0,
    ):
        top = name.split(".")[0]
        if top in BLOCKED_MODULES:
            raise ImportError(f"Import of '{name}' is blocked by security policy")
        if top not in self._allowed:
            raise ImportError(
                f"Import of '{name}' is not allowed. "
                f"Allowed packages: {sorted(self._allowed)}"
            )
        return __builtins__["__import__"](name, globals, locals, fromlist, level)  # type: ignore[index]
