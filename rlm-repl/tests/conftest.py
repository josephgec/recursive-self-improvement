"""Shared fixtures for RLM-REPL tests."""

import sys
import os
import pytest

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.safety.policy import SafetyPolicy
from src.backends.local import LocalREPL


@pytest.fixture
def default_policy():
    """Default safety policy."""
    return SafetyPolicy()


@pytest.fixture
def strict_policy():
    """Strict safety policy."""
    return SafetyPolicy.strict()


@pytest.fixture
def relaxed_policy():
    """Relaxed safety policy."""
    return SafetyPolicy.relaxed()


@pytest.fixture
def local_repl(default_policy):
    """A fresh LocalREPL instance."""
    repl = LocalREPL(policy=default_policy)
    yield repl
    if repl.is_alive():
        repl.shutdown()


@pytest.fixture
def strict_repl(strict_policy):
    """A LocalREPL with strict policy."""
    repl = LocalREPL(policy=strict_policy)
    yield repl
    if repl.is_alive():
        repl.shutdown()


@pytest.fixture
def sample_code():
    """Collection of sample code strings for testing."""
    return {
        "simple_assign": "x = 42",
        "arithmetic": "result = 2 + 3",
        "print_hello": 'print("hello world")',
        "list_comp": "squares = [x**2 for x in range(10)]",
        "function_def": "def add(a, b): return a + b\nresult = add(3, 4)",
        "multiline": "x = 1\ny = 2\nz = x + y",
        "final_call": 'FINAL("the answer")',
        "final_var": 'result = 42\nFINAL_VAR("result")',
        "forbidden_import": "import os",
        "forbidden_eval": "eval('1+1')",
        "dunder_access": 'x = "".__class__',
        "infinite_loop": "while True:\n    pass",
    }
