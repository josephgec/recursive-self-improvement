"""Tests for the ASTScanner - all 10 scan categories."""

import pytest
from src.safety.ast_scanner import ASTScanner, ScanResult, Violation
from src.safety.policy import SafetyPolicy


class TestASTScannerForbiddenImports:
    """Test forbidden import detection."""

    def setup_method(self):
        self.scanner = ASTScanner()

    def test_import_os_blocked(self):
        result = self.scanner.scan("import os")
        assert not result.safe
        assert any(v.category == "forbidden_import" for v in result.violations)

    def test_import_sys_blocked(self):
        result = self.scanner.scan("import sys")
        assert not result.safe

    def test_import_subprocess_blocked(self):
        result = self.scanner.scan("import subprocess")
        assert not result.safe

    def test_from_os_import_blocked(self):
        result = self.scanner.scan("from os import path")
        assert not result.safe

    def test_import_socket_blocked(self):
        result = self.scanner.scan("import socket")
        assert not result.safe

    def test_import_math_allowed(self):
        result = self.scanner.scan("import math")
        assert result.safe

    def test_import_json_allowed(self):
        result = self.scanner.scan("import json")
        assert result.safe

    def test_nested_import_blocked(self):
        result = self.scanner.scan("import os.path")
        assert not result.safe


class TestASTScannerForbiddenBuiltins:
    """Test forbidden builtin detection."""

    def setup_method(self):
        self.scanner = ASTScanner()

    def test_eval_blocked(self):
        result = self.scanner.scan("eval('1+1')")
        assert not result.safe
        assert any(v.category == "forbidden_builtin" for v in result.violations)

    def test_exec_blocked(self):
        result = self.scanner.scan("exec('x = 1')")
        assert not result.safe

    def test_compile_blocked(self):
        result = self.scanner.scan("compile('x=1', '', 'exec')")
        assert not result.safe

    def test_open_blocked(self):
        result = self.scanner.scan("open('file.txt')")
        assert not result.safe

    def test_input_blocked(self):
        result = self.scanner.scan("input('prompt: ')")
        assert not result.safe

    def test_print_allowed(self):
        result = self.scanner.scan("print('hello')")
        assert result.safe

    def test_len_allowed(self):
        result = self.scanner.scan("len([1, 2, 3])")
        assert result.safe


class TestASTScannerDunderAccess:
    """Test dunder attribute access detection."""

    def setup_method(self):
        self.scanner = ASTScanner()

    def test_class_dunder_blocked(self):
        result = self.scanner.scan('x = "".__class__')
        assert not result.safe
        assert any(v.category == "dunder_access" for v in result.violations)

    def test_bases_dunder_blocked(self):
        result = self.scanner.scan('x = object.__bases__')
        assert not result.safe

    def test_subclasses_blocked(self):
        result = self.scanner.scan('x = object.__subclasses__()')
        assert not result.safe

    def test_normal_attribute_allowed(self):
        result = self.scanner.scan("x = [1, 2, 3]\ny = x.append(4)")
        assert result.safe

    def test_dunder_allowed_with_relaxed_policy(self):
        policy = SafetyPolicy(allow_dunder_access=True)
        scanner = ASTScanner(policy)
        result = scanner.scan('x = "".__class__')
        assert result.safe


class TestASTScannerStarImports:
    """Test star import detection."""

    def setup_method(self):
        self.scanner = ASTScanner()

    def test_star_import_blocked(self):
        result = self.scanner.scan("from math import *")
        assert not result.safe
        assert any(v.category == "star_import" for v in result.violations)

    def test_specific_import_allowed(self):
        result = self.scanner.scan("from math import sqrt")
        assert result.safe

    def test_star_import_allowed_with_relaxed_policy(self):
        policy = SafetyPolicy(allow_star_imports=True)
        scanner = ASTScanner(policy)
        result = scanner.scan("from math import *")
        assert result.safe


class TestASTScannerObfuscation:
    """Test obfuscation detection."""

    def setup_method(self):
        self.scanner = ASTScanner()

    def test_chr_exec_blocked(self):
        code = 'exec("".join(chr(x) for x in [112, 114, 105, 110, 116]))'
        result = self.scanner.scan(code)
        assert not result.safe
        has_obfuscation = any(
            v.category in ("obfuscation", "forbidden_builtin")
            for v in result.violations
        )
        assert has_obfuscation

    def test_normal_chr_allowed(self):
        result = self.scanner.scan("x = chr(65)")
        assert result.safe

    def test_obfuscation_disabled(self):
        policy = SafetyPolicy(detect_obfuscation=False)
        scanner = ASTScanner(policy)
        # Still blocked by forbidden_builtin (exec)
        result = scanner.scan('exec(chr(65))')
        assert not result.safe  # exec is still forbidden


class TestASTScannerInfiniteLoops:
    """Test infinite loop detection."""

    def setup_method(self):
        self.scanner = ASTScanner()

    def test_while_true_no_break(self):
        result = self.scanner.scan("while True:\n    pass")
        assert any(v.category == "infinite_loop" for v in result.violations)

    def test_while_true_with_break_allowed(self):
        code = "while True:\n    x = 1\n    break"
        result = self.scanner.scan(code)
        infinite = [v for v in result.violations if v.category == "infinite_loop"]
        assert len(infinite) == 0

    def test_while_condition_allowed(self):
        code = "x = 10\nwhile x > 0:\n    x -= 1"
        result = self.scanner.scan(code)
        infinite = [v for v in result.violations if v.category == "infinite_loop"]
        assert len(infinite) == 0

    def test_infinite_loop_detection_disabled(self):
        policy = SafetyPolicy(detect_infinite_loops=False)
        scanner = ASTScanner(policy)
        result = scanner.scan("while True:\n    pass")
        infinite = [v for v in result.violations if v.category == "infinite_loop"]
        assert len(infinite) == 0


class TestASTScannerNesting:
    """Test excessive nesting detection."""

    def setup_method(self):
        policy = SafetyPolicy(max_nesting_depth=3)
        self.scanner = ASTScanner(policy)

    def test_shallow_nesting_allowed(self):
        code = "if True:\n    x = 1"
        result = self.scanner.scan(code)
        nesting_violations = [v for v in result.violations if v.category == "excessive_nesting"]
        assert len(nesting_violations) == 0

    def test_deep_nesting_blocked(self):
        code = "if True:\n  if True:\n    if True:\n      if True:\n        x = 1"
        result = self.scanner.scan(code)
        nesting_violations = [v for v in result.violations if v.category == "excessive_nesting"]
        assert len(nesting_violations) > 0


class TestASTScannerSyntaxErrors:
    """Test syntax error handling."""

    def setup_method(self):
        self.scanner = ASTScanner()

    def test_syntax_error_caught(self):
        result = self.scanner.scan("def :")
        assert not result.safe
        assert any(v.category == "syntax_error" for v in result.violations)


class TestScanResultProperties:
    """Test ScanResult properties."""

    def test_error_count(self):
        violations = [
            Violation(category="test", message="error", severity="error"),
            Violation(category="test", message="warning", severity="warning"),
            Violation(category="test", message="error2", severity="error"),
        ]
        result = ScanResult(safe=False, violations=violations)
        assert result.error_count == 2
        assert result.warning_count == 1

    def test_violation_str(self):
        v = Violation(category="forbidden_import", message="import os", line=1)
        s = str(v)
        assert "forbidden_import" in s
        assert "line 1" in s

    def test_safe_code(self):
        scanner = ASTScanner()
        result = scanner.scan("x = 1 + 2")
        assert result.safe
        assert result.error_count == 0
