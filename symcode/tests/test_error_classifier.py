"""Tests for the error classifier."""

from __future__ import annotations

import pytest

from src.verification.error_classifier import ErrorCategory, ErrorClassifier
from src.verification.result_types import CodeError


class TestClassify:
    """Test error classification for all error types."""

    def setup_method(self):
        self.classifier = ErrorClassifier()

    # ── Direct mapping tests ─────────────────────────────────────────

    def test_syntax_error(self):
        error = CodeError(error_type="SyntaxError", message="invalid syntax")
        assert self.classifier.classify(error) == ErrorCategory.SYNTAX_BASIC

    def test_indentation_error(self):
        error = CodeError(error_type="IndentationError", message="unexpected indent")
        assert self.classifier.classify(error) == ErrorCategory.SYNTAX_BASIC

    def test_tab_error(self):
        error = CodeError(error_type="TabError", message="inconsistent tab")
        assert self.classifier.classify(error) == ErrorCategory.SYNTAX_BASIC

    def test_import_error(self):
        error = CodeError(error_type="ImportError", message="cannot import name")
        assert self.classifier.classify(error) == ErrorCategory.IMPORT_MISSING

    def test_module_not_found_error(self):
        error = CodeError(
            error_type="ModuleNotFoundError",
            message="No module named 'nonexistent'",
        )
        assert self.classifier.classify(error) == ErrorCategory.IMPORT_MISSING

    def test_name_error(self):
        error = CodeError(error_type="NameError", message="name 'x' is not defined")
        assert self.classifier.classify(error) == ErrorCategory.NAME_UNDEFINED

    def test_type_error(self):
        error = CodeError(error_type="TypeError", message="unsupported operand type")
        assert self.classifier.classify(error) == ErrorCategory.TYPE_MISMATCH

    def test_zero_division_error(self):
        error = CodeError(error_type="ZeroDivisionError", message="division by zero")
        assert self.classifier.classify(error) == ErrorCategory.DIVISION_ZERO

    def test_timeout_error(self):
        error = CodeError(error_type="TimeoutError", message="timed out")
        assert self.classifier.classify(error) == ErrorCategory.TIMEOUT

    def test_value_error(self):
        error = CodeError(error_type="ValueError", message="invalid literal")
        assert self.classifier.classify(error) == ErrorCategory.LOGIC_ERROR

    def test_index_error(self):
        error = CodeError(error_type="IndexError", message="list index out of range")
        assert self.classifier.classify(error) == ErrorCategory.LOGIC_ERROR

    def test_key_error(self):
        error = CodeError(error_type="KeyError", message="'missing_key'")
        assert self.classifier.classify(error) == ErrorCategory.LOGIC_ERROR

    def test_recursion_error(self):
        error = CodeError(error_type="RecursionError", message="maximum recursion depth")
        assert self.classifier.classify(error) == ErrorCategory.LOGIC_ERROR

    def test_overflow_error(self):
        error = CodeError(error_type="OverflowError", message="result too large")
        assert self.classifier.classify(error) == ErrorCategory.LOGIC_ERROR

    def test_runtime_error(self):
        error = CodeError(error_type="RuntimeError", message="something went wrong")
        assert self.classifier.classify(error) == ErrorCategory.LOGIC_ERROR

    # ── AttributeError refinement ────────────────────────────────────

    def test_attribute_error_sympy(self):
        """AttributeError mentioning sympy -> SYMPY_SPECIFIC."""
        error = CodeError(
            error_type="AttributeError",
            message="'Symbol' object has no attribute 'foo'",
        )
        assert self.classifier.classify(error) == ErrorCategory.SYMPY_SPECIFIC

    def test_attribute_error_generic(self):
        """AttributeError not mentioning sympy -> NAME_UNDEFINED."""
        error = CodeError(
            error_type="AttributeError",
            message="'int' object has no attribute 'append'",
        )
        assert self.classifier.classify(error) == ErrorCategory.NAME_UNDEFINED

    def test_attribute_error_with_sympy_in_message(self):
        error = CodeError(
            error_type="AttributeError",
            message="module 'sympy' has no attribute 'badmethod'",
        )
        assert self.classifier.classify(error) == ErrorCategory.SYMPY_SPECIFIC

    # ── Fallback message-based classification ────────────────────────

    def test_unknown_error_syntax_in_message(self):
        error = CodeError(error_type="CustomError", message="syntax issue found")
        assert self.classifier.classify(error) == ErrorCategory.SYNTAX_BASIC

    def test_unknown_error_import_in_message(self):
        error = CodeError(error_type="CustomError", message="failed to import module")
        assert self.classifier.classify(error) == ErrorCategory.IMPORT_MISSING

    def test_unknown_error_not_defined_in_message(self):
        error = CodeError(error_type="CustomError", message="variable not defined")
        assert self.classifier.classify(error) == ErrorCategory.NAME_UNDEFINED

    def test_unknown_error_timeout_in_message(self):
        error = CodeError(error_type="CustomError", message="operation timed out")
        assert self.classifier.classify(error) == ErrorCategory.TIMEOUT

    def test_unknown_error_division_zero_in_message(self):
        error = CodeError(
            error_type="CustomError", message="division by zero encountered"
        )
        assert self.classifier.classify(error) == ErrorCategory.DIVISION_ZERO

    def test_unknown_error_generic_fallback(self):
        error = CodeError(error_type="CustomError", message="something went wrong")
        assert self.classifier.classify(error) == ErrorCategory.LOGIC_ERROR


class TestGetFixHint:
    """Test fix hint retrieval."""

    def setup_method(self):
        self.classifier = ErrorClassifier()

    def test_all_categories_have_hints(self):
        for cat in ErrorCategory:
            hint = self.classifier.get_fix_hint(cat)
            assert isinstance(hint, str)
            assert len(hint) > 0

    def test_syntax_hint(self):
        hint = self.classifier.get_fix_hint(ErrorCategory.SYNTAX_BASIC)
        assert "syntax" in hint.lower()

    def test_timeout_hint(self):
        hint = self.classifier.get_fix_hint(ErrorCategory.TIMEOUT)
        assert "long" in hint.lower() or "simplify" in hint.lower()

    def test_no_answer_hint(self):
        hint = self.classifier.get_fix_hint(ErrorCategory.NO_ANSWER)
        assert "answer" in hint.lower()


class TestClassifyAndHint:
    """Test combined classify + hint."""

    def setup_method(self):
        self.classifier = ErrorClassifier()

    def test_classify_and_hint_returns_tuple(self):
        error = CodeError(error_type="SyntaxError", message="invalid syntax")
        cat, hint = self.classifier.classify_and_hint(error)
        assert cat == ErrorCategory.SYNTAX_BASIC
        assert isinstance(hint, str)
        assert len(hint) > 0

    def test_classify_and_hint_name_error(self):
        error = CodeError(error_type="NameError", message="name 'x' is not defined")
        cat, hint = self.classifier.classify_and_hint(error)
        assert cat == ErrorCategory.NAME_UNDEFINED
        assert "variable" in hint.lower() or "defined" in hint.lower()
