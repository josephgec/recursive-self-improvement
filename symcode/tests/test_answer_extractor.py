"""Tests for answer extraction and normalization."""

from __future__ import annotations

import pytest

from src.pipeline.answer_extractor import AnswerExtractor, ExtractedAnswer


class TestExtractFromExecution:
    """Test extraction from code execution results."""

    def setup_method(self):
        self.extractor = AnswerExtractor()

    # ── namespace priority ───────────────────────────────────────────

    def test_namespace_answer_variable(self):
        """answer variable in namespace has highest priority."""
        result = self.extractor.extract_from_execution(
            namespace={"answer": 42, "x": 10},
            stdout="Answer: 99",
        )
        assert result is not None
        assert result.raw == "42"
        assert result.source == "variable"
        assert result.confidence == 1.0

    def test_namespace_string_answer(self):
        result = self.extractor.extract_from_execution(
            namespace={"answer": "hello"},
        )
        assert result is not None
        assert result.raw == "hello"
        assert result.source == "variable"

    def test_namespace_sympy_answer(self):
        """SymPy expression as answer variable."""
        import sympy
        result = self.extractor.extract_from_execution(
            namespace={"answer": sympy.Rational(1, 2)},
        )
        assert result is not None
        assert result.source == "variable"
        assert "1/2" in result.normalized or "1" in result.normalized

    # ── stdout Answer: pattern ───────────────────────────────────────

    def test_stdout_answer_pattern(self):
        result = self.extractor.extract_from_execution(
            namespace={},
            stdout="Computing...\nAnswer: 42\nDone.",
        )
        assert result is not None
        assert result.raw == "42"
        assert result.source == "stdout"
        assert result.confidence == 0.9

    def test_stdout_answer_with_spaces(self):
        result = self.extractor.extract_from_execution(
            namespace={},
            stdout="Answer:   3/8  \n",
        )
        assert result is not None
        assert result.raw == "3/8"

    # ── stdout last line fallback ────────────────────────────────────

    def test_stdout_last_line_fallback(self):
        result = self.extractor.extract_from_execution(
            namespace={},
            stdout="step 1\nstep 2\n99\n",
        )
        assert result is not None
        assert result.raw == "99"
        assert result.source == "stdout"
        assert result.confidence == 0.5

    def test_stdout_empty_lines_skipped(self):
        result = self.extractor.extract_from_execution(
            namespace={},
            stdout="42\n\n  \n",
        )
        assert result is not None
        assert result.raw == "42"

    # ── no answer found ──────────────────────────────────────────────

    def test_no_namespace_no_stdout(self):
        result = self.extractor.extract_from_execution()
        assert result is None

    def test_empty_namespace_empty_stdout(self):
        result = self.extractor.extract_from_execution(
            namespace={}, stdout="",
        )
        assert result is None

    def test_namespace_without_answer_key(self):
        result = self.extractor.extract_from_execution(
            namespace={"x": 10, "y": 20},
            stdout="",
        )
        assert result is None

    def test_none_namespace_empty_stdout(self):
        result = self.extractor.extract_from_execution(
            namespace=None, stdout="",
        )
        assert result is None


class TestExtractFromProse:
    """Test extraction from prose/CoT text."""

    def setup_method(self):
        self.extractor = AnswerExtractor()

    def test_empty_text(self):
        result = self.extractor.extract_from_prose("")
        assert result is None

    def test_none_text(self):
        # Should handle gracefully (empty check)
        result = self.extractor.extract_from_prose("")
        assert result is None

    # ── boxed extraction ─────────────────────────────────────────────

    def test_boxed_answer(self):
        result = self.extractor.extract_from_prose(
            "After careful work, the answer is \\boxed{42}."
        )
        assert result is not None
        assert result.source == "prose_boxed"
        assert result.confidence == 1.0
        assert "42" in result.normalized

    def test_boxed_fraction(self):
        result = self.extractor.extract_from_prose(
            "Therefore \\boxed{\\frac{5}{12}}."
        )
        assert result is not None
        assert result.source == "prose_boxed"

    # ── "the answer is" pattern ──────────────────────────────────────

    def test_the_answer_is_pattern(self):
        result = self.extractor.extract_from_prose(
            "We compute step by step. The answer is 42."
        )
        assert result is not None
        assert result.source == "prose_text"
        assert result.confidence == 0.7
        assert "42" in result.normalized

    def test_the_final_answer_is_pattern(self):
        result = self.extractor.extract_from_prose(
            "Working through... The final answer is 100."
        )
        assert result is not None
        assert "100" in result.normalized

    def test_answer_equals_pattern(self):
        result = self.extractor.extract_from_prose(
            "Computation yields Answer = 7."
        )
        assert result is not None
        assert "7" in result.normalized

    def test_therefore_pattern(self):
        result = self.extractor.extract_from_prose(
            "Therefore, 25."
        )
        assert result is not None
        assert "25" in result.normalized

    def test_hence_pattern(self):
        result = self.extractor.extract_from_prose(
            "Hence, the answer is 13."
        )
        assert result is not None
        assert "13" in result.normalized

    def test_no_pattern_found(self):
        result = self.extractor.extract_from_prose(
            "I'm not sure about this problem."
        )
        assert result is None


class TestNormalize:
    """Test answer normalization with fractions, lists, sympy."""

    def setup_method(self):
        self.extractor = AnswerExtractor()

    def test_empty_string(self):
        assert self.extractor.normalize("") == ""

    def test_integer(self):
        result = self.extractor.normalize("42")
        assert result == "42"

    def test_fraction_string(self):
        result = self.extractor.normalize("1/2")
        assert result == "1/2"

    def test_decimal_to_fraction(self):
        """0.5 should normalize to 1/2 via sympy."""
        result = self.extractor.normalize("0.5")
        assert result == "1/2"

    def test_sympy_expression(self):
        result = self.extractor.normalize("2**3")
        assert result == "8"

    def test_latex_with_backslash(self):
        """LaTeX expressions should be stripped and re-normalized."""
        result = self.extractor.normalize("\\frac{1}{2}")
        assert "1" in result and "2" in result

    def test_non_numeric_expression(self):
        """Symbolic expressions should be normalized via sympy."""
        result = self.extractor.normalize("x + 1")
        assert "x" in result

    def test_invalid_expression_passthrough(self):
        """Invalid expressions should pass through after normalization."""
        result = self.extractor.normalize("some random text")
        assert result == "some random text"

    def test_list_expression(self):
        result = self.extractor.normalize("[1, 2, 3]")
        assert result is not None

    def test_rational_non_integer(self):
        """Test that non-integer rationals are kept as fractions."""
        result = self.extractor.normalize("3/7")
        assert result == "3/7"
