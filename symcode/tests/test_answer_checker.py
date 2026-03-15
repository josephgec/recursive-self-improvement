"""Tests for the answer checker."""

from __future__ import annotations

import pytest

from src.verification.answer_checker import AnswerChecker


class TestAnswerChecker:
    """Test cascading answer equivalence checks."""

    def setup_method(self):
        self.checker = AnswerChecker()

    # ── numeric / fraction equivalence ──────────────────────────────

    def test_half_decimal(self):
        """'1/2' should equal '0.5'."""
        assert self.checker.check("1/2", "0.5") is True

    def test_fraction_latex(self):
        r"""'\frac{5}{12}' should equal '5/12'."""
        assert self.checker.check("\\frac{5}{12}", "5/12") is True

    # ── symbolic equivalence ────────────────────────────────────────

    def test_sqrt_equivalence(self):
        """'sqrt(2)' should equal '2**(1/2)'."""
        assert self.checker.check("sqrt(2)", "2**(1/2)") is True

    # ── set/list equivalence ────────────────────────────────────────

    def test_list_order_independent(self):
        """'[1,3]' should equal '[3,1]' (set match)."""
        assert self.checker.check("[1,3]", "[3,1]") is True

    # ── exact matches ───────────────────────────────────────────────

    def test_exact_integer(self):
        assert self.checker.check("42", "42") is True

    def test_integer_mismatch(self):
        """'42' should NOT equal '43'."""
        assert self.checker.check("42", "43") is False

    # ── detailed check ──────────────────────────────────────────────

    def test_check_detailed_method_exact(self):
        result = self.checker.check_detailed("42", "42")
        assert result.correct is True
        assert result.method == "exact_string"

    def test_check_detailed_method_numeric(self):
        result = self.checker.check_detailed("0.5", "1/2")
        assert result.correct is True
        # Could match via numeric or fraction
        assert result.method in ("numeric", "fraction", "symbolic")

    def test_check_detailed_method_symbolic(self):
        result = self.checker.check_detailed("x**2 + 2*x + 1", "(x+1)**2")
        assert result.correct is True
        assert result.method == "symbolic"

    def test_check_detailed_not_equal(self):
        result = self.checker.check_detailed("42", "43")
        assert result.correct is False
        assert result.method == "none"

    # ── edge cases ──────────────────────────────────────────────────

    def test_whitespace_normalization(self):
        assert self.checker.check("  42  ", "42") is True

    def test_negative_numbers(self):
        assert self.checker.check("-3", "-3") is True
        assert self.checker.check("-3", "3") is False

    def test_pi(self):
        assert self.checker.check("pi", "pi") is True
        assert self.checker.check("3.14159", "pi") is False
