"""Safety policy configuration for the RLM-REPL sandbox."""

from dataclasses import dataclass, field
from typing import List, Optional

import yaml


@dataclass
class SafetyPolicy:
    """Defines safety constraints for code execution.

    Attributes:
        forbidden_imports: Module names that cannot be imported.
        forbidden_builtins: Builtin function names that cannot be used.
        allow_dunder_access: Whether __dunder__ attribute access is allowed.
        allow_star_imports: Whether 'from X import *' is allowed.
        max_nesting_depth: Maximum AST nesting depth.
        max_output_chars: Maximum output characters before truncation.
        timeout_seconds: Maximum execution time.
        max_memory_mb: Maximum memory usage in megabytes.
        max_spawn_depth: Maximum REPL spawn depth.
        detect_obfuscation: Whether to detect obfuscation patterns.
        detect_infinite_loops: Whether to detect simple infinite loops.
    """

    forbidden_imports: List[str] = field(default_factory=lambda: [
        "os", "sys", "subprocess", "shutil", "socket", "http", "urllib",
        "requests", "ctypes", "signal", "multiprocessing", "threading",
        "importlib", "__import__", "pathlib",
    ])
    forbidden_builtins: List[str] = field(default_factory=lambda: [
        "exec", "eval", "compile", "__import__", "open", "input",
        "breakpoint", "exit", "quit",
    ])
    allow_dunder_access: bool = False
    allow_star_imports: bool = False
    max_nesting_depth: int = 10
    max_output_chars: int = 100000
    timeout_seconds: float = 30.0
    max_memory_mb: float = 512.0
    max_spawn_depth: int = 5
    detect_obfuscation: bool = True
    detect_infinite_loops: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "SafetyPolicy":
        """Load a safety policy from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A SafetyPolicy instance.
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def strict(cls) -> "SafetyPolicy":
        """Create a strict safety policy with minimal permissions."""
        return cls(
            forbidden_imports=[
                "os", "sys", "subprocess", "shutil", "socket", "http", "urllib",
                "requests", "ctypes", "signal", "multiprocessing", "threading",
                "importlib", "__import__", "pathlib", "pickle", "shelve",
                "tempfile", "glob", "fnmatch", "io", "code", "codeop", "compileall",
            ],
            forbidden_builtins=[
                "exec", "eval", "compile", "__import__", "open", "input",
                "breakpoint", "exit", "quit", "globals", "locals", "vars",
                "dir", "getattr", "setattr", "delattr", "hasattr", "type", "super",
            ],
            allow_dunder_access=False,
            allow_star_imports=False,
            max_nesting_depth=6,
            max_output_chars=50000,
            timeout_seconds=15.0,
            max_memory_mb=256.0,
            max_spawn_depth=2,
            detect_obfuscation=True,
            detect_infinite_loops=True,
        )

    @classmethod
    def relaxed(cls) -> "SafetyPolicy":
        """Create a relaxed safety policy for trusted code."""
        return cls(
            forbidden_imports=["subprocess", "ctypes", "signal"],
            forbidden_builtins=["exec", "eval", "compile", "__import__"],
            allow_dunder_access=True,
            allow_star_imports=True,
            max_nesting_depth=20,
            max_output_chars=500000,
            timeout_seconds=120.0,
            max_memory_mb=2048.0,
            max_spawn_depth=10,
            detect_obfuscation=False,
            detect_infinite_loops=False,
        )
