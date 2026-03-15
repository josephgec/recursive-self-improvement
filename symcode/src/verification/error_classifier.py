"""Classify code execution errors into actionable categories."""

from __future__ import annotations

import enum
import re
from typing import Any

from src.verification.result_types import CodeError
from src.utils.logging import get_logger

logger = get_logger("error_classifier")


class ErrorCategory(enum.Enum):
    SYNTAX_BASIC = "syntax_basic"
    IMPORT_MISSING = "import_missing"
    NAME_UNDEFINED = "name_undefined"
    TYPE_MISMATCH = "type_mismatch"
    DIVISION_ZERO = "division_zero"
    TIMEOUT = "timeout"
    LOGIC_ERROR = "logic_error"
    NO_ANSWER = "no_answer"
    SYMPY_SPECIFIC = "sympy_specific"


# Maps error type strings to categories
_ERROR_TYPE_MAP: dict[str, ErrorCategory] = {
    "SyntaxError": ErrorCategory.SYNTAX_BASIC,
    "IndentationError": ErrorCategory.SYNTAX_BASIC,
    "TabError": ErrorCategory.SYNTAX_BASIC,
    "ImportError": ErrorCategory.IMPORT_MISSING,
    "ModuleNotFoundError": ErrorCategory.IMPORT_MISSING,
    "NameError": ErrorCategory.NAME_UNDEFINED,
    "TypeError": ErrorCategory.TYPE_MISMATCH,
    "ZeroDivisionError": ErrorCategory.DIVISION_ZERO,
    "TimeoutError": ErrorCategory.TIMEOUT,
    "AttributeError": ErrorCategory.SYMPY_SPECIFIC,
    "ValueError": ErrorCategory.LOGIC_ERROR,
    "IndexError": ErrorCategory.LOGIC_ERROR,
    "KeyError": ErrorCategory.LOGIC_ERROR,
    "RecursionError": ErrorCategory.LOGIC_ERROR,
    "OverflowError": ErrorCategory.LOGIC_ERROR,
    "RuntimeError": ErrorCategory.LOGIC_ERROR,
}

# Hints keyed by category
_FIX_HINTS: dict[ErrorCategory, str] = {
    ErrorCategory.SYNTAX_BASIC: (
        "Fix the syntax error. Check for missing colons, parentheses, "
        "or incorrect indentation."
    ),
    ErrorCategory.IMPORT_MISSING: (
        "The import failed. Make sure you are importing from the correct "
        "module. For SymPy functions, use 'from sympy import ...'."
    ),
    ErrorCategory.NAME_UNDEFINED: (
        "A variable or function is not defined. Make sure to define all "
        "variables before using them, or import the needed function."
    ),
    ErrorCategory.TYPE_MISMATCH: (
        "There is a type error. Check that operations are applied to "
        "compatible types. Consider converting between int, float, "
        "and SymPy types explicitly."
    ),
    ErrorCategory.DIVISION_ZERO: (
        "Division by zero occurred. Add a check for zero before dividing, "
        "or re-examine your mathematical approach."
    ),
    ErrorCategory.TIMEOUT: (
        "The code took too long. Simplify your approach: avoid brute-force "
        "loops over large ranges, use SymPy's built-in solvers, or "
        "reduce symbolic computation complexity."
    ),
    ErrorCategory.LOGIC_ERROR: (
        "There is a logical error in the code. Review your mathematical "
        "approach and check edge cases."
    ),
    ErrorCategory.NO_ANSWER: (
        "No answer was produced. Make sure to assign the final result "
        "to a variable named 'answer'."
    ),
    ErrorCategory.SYMPY_SPECIFIC: (
        "A SymPy-specific error occurred. Check that you are using the "
        "correct SymPy function names and calling them with valid arguments. "
        "Consult SymPy documentation for the correct API."
    ),
}


class ErrorClassifier:
    """Classify execution errors and provide fix hints."""

    def classify(self, error: CodeError) -> ErrorCategory:
        """Classify a CodeError into an ErrorCategory."""
        error_type = error.error_type

        # Direct mapping
        if error_type in _ERROR_TYPE_MAP:
            cat = _ERROR_TYPE_MAP[error_type]

            # Refine: check if AttributeError is SymPy-specific
            if cat == ErrorCategory.SYMPY_SPECIFIC:
                msg_lower = error.message.lower()
                if "sympy" in msg_lower or "symbol" in msg_lower:
                    return ErrorCategory.SYMPY_SPECIFIC
                # Could be a generic attribute error
                return ErrorCategory.NAME_UNDEFINED

            return cat

        # Fallback: try to infer from message
        msg = error.message.lower()
        if "syntax" in msg:
            return ErrorCategory.SYNTAX_BASIC
        if "import" in msg or "module" in msg:
            return ErrorCategory.IMPORT_MISSING
        if "not defined" in msg:
            return ErrorCategory.NAME_UNDEFINED
        if "timeout" in msg or "timed out" in msg:
            return ErrorCategory.TIMEOUT
        if "division" in msg and "zero" in msg:
            return ErrorCategory.DIVISION_ZERO

        return ErrorCategory.LOGIC_ERROR

    def get_fix_hint(self, category: ErrorCategory) -> str:
        """Get a human-readable fix hint for an error category."""
        return _FIX_HINTS.get(category, "Review the error and fix the code.")

    def classify_and_hint(self, error: CodeError) -> tuple[ErrorCategory, str]:
        """Classify an error and return both category and hint."""
        cat = self.classify(error)
        hint = self.get_fix_hint(cat)
        return cat, hint
