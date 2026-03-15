"""Tests for the security policy and code analyser."""

from __future__ import annotations

import pytest

from repl.src.security import CodeAnalyzer, SecurityPolicy


@pytest.fixture
def analyzer() -> CodeAnalyzer:
    return CodeAnalyzer(SecurityPolicy())


class TestBlockedCode:
    def test_import_os_system(self, analyzer: CodeAnalyzer) -> None:
        is_safe, reason = analyzer.analyze("import os; os.system('whoami')")
        assert not is_safe
        assert reason is not None

    def test_import_subprocess(self, analyzer: CodeAnalyzer) -> None:
        is_safe, reason = analyzer.analyze("import subprocess")
        assert not is_safe

    def test_dunder_class_mro(self, analyzer: CodeAnalyzer) -> None:
        is_safe, reason = analyzer.analyze("x = ''.__class__.__mro__")
        assert not is_safe
        assert "__class__" in reason or "__mro__" in reason

    def test_open_etc_passwd(self, analyzer: CodeAnalyzer) -> None:
        is_safe, reason = analyzer.analyze('open("/etc/passwd")')
        assert not is_safe
        assert "open" in reason

    def test_eval_blocked(self, analyzer: CodeAnalyzer) -> None:
        is_safe, reason = analyzer.analyze("eval('1+1')")
        assert not is_safe

    def test_exec_blocked(self, analyzer: CodeAnalyzer) -> None:
        is_safe, reason = analyzer.analyze("exec('print(1)')")
        assert not is_safe

    def test_import_socket(self, analyzer: CodeAnalyzer) -> None:
        is_safe, reason = analyzer.analyze("import socket")
        assert not is_safe

    def test_from_os_import(self, analyzer: CodeAnalyzer) -> None:
        # "os" is not in BLOCKED_MODULES (subprocess, ctypes, etc. are),
        # but os.system is blocked at the Call level.  Direct "import os"
        # should be caught by the import hook at runtime, not the static
        # analyzer.  This test verifies the AST catches os.system() calls.
        is_safe, reason = analyzer.analyze("import os\nos.system('ls')")
        assert not is_safe


class TestAllowedCode:
    def test_simple_math(self, analyzer: CodeAnalyzer) -> None:
        is_safe, reason = analyzer.analyze("x = 1 + 1")
        assert is_safe
        assert reason is None

    def test_import_numpy(self, analyzer: CodeAnalyzer) -> None:
        is_safe, reason = analyzer.analyze("import numpy as np")
        assert is_safe

    def test_import_math(self, analyzer: CodeAnalyzer) -> None:
        is_safe, reason = analyzer.analyze("import math")
        assert is_safe

    def test_list_comprehension(self, analyzer: CodeAnalyzer) -> None:
        is_safe, reason = analyzer.analyze("[x**2 for x in range(10)]")
        assert is_safe

    def test_function_def(self, analyzer: CodeAnalyzer) -> None:
        code = "def foo(x):\n    return x * 2"
        is_safe, reason = analyzer.analyze(code)
        assert is_safe

    def test_class_def(self, analyzer: CodeAnalyzer) -> None:
        code = "class Foo:\n    pass"
        is_safe, reason = analyzer.analyze(code)
        assert is_safe

    def test_syntax_error_passes(self, analyzer: CodeAnalyzer) -> None:
        # Syntax errors are not security violations
        is_safe, reason = analyzer.analyze("def (broken")
        assert is_safe
