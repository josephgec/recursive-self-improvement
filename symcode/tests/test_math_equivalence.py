"""Tests for math equivalence utilities."""

from __future__ import annotations

import pytest

from src.utils.math_equivalence import (
    fraction_equal,
    normalize_answer,
    numeric_equal,
    parse_latex_answer,
    symbolic_equal,
)


class TestSymbolicEqual:
    """Test symbolic equality via SymPy."""

    def test_same_expression(self):
        assert symbolic_equal("x**2 + 1", "x**2 + 1") is True

    def test_equivalent_expressions(self):
        assert symbolic_equal("(x+1)**2", "x**2 + 2*x + 1") is True

    def test_different_expressions(self):
        assert symbolic_equal("x**2", "x**3") is False

    def test_numeric_symbolic(self):
        assert symbolic_equal("1/2", "0.5") is True

    def test_sqrt(self):
        assert symbolic_equal("sqrt(2)", "2**(1/2)") is True

    def test_trig_identity(self):
        """sin(x)^2 + cos(x)^2 should equal 1."""
        assert symbolic_equal("sin(x)**2 + cos(x)**2", "1") is True

    def test_invalid_input(self):
        assert symbolic_equal("not_an_expr!!!", "42") is False


class TestNumericEqual:
    """Test numeric equality within tolerance."""

    def test_exact_integers(self):
        assert numeric_equal("42", "42") is True

    def test_close_floats(self):
        assert numeric_equal("0.333333333", "1/3", tol=1e-6) is True

    def test_not_close(self):
        assert numeric_equal("1.0", "2.0") is False

    def test_zero(self):
        assert numeric_equal("0", "0.0") is True

    def test_custom_tolerance(self):
        assert numeric_equal("1.001", "1.0", tol=0.01) is True
        assert numeric_equal("1.1", "1.0", tol=0.01) is False

    def test_invalid_input(self):
        assert numeric_equal("abc", "42") is False


class TestFractionEqual:
    """Test fraction equality."""

    def test_simple_fraction(self):
        assert fraction_equal("1/2", "1/2") is True

    def test_equivalent_fractions(self):
        assert fraction_equal("2/4", "1/2") is True

    def test_decimal_to_fraction(self):
        assert fraction_equal("0.5", "1/2") is True

    def test_different_fractions(self):
        assert fraction_equal("1/3", "1/2") is False


class TestParseLatexAnswer:
    """Test LaTeX answer parsing."""

    def test_boxed_integer(self):
        result = parse_latex_answer("\\boxed{42}")
        assert result is not None
        assert "42" in result

    def test_boxed_fraction(self):
        result = parse_latex_answer("\\boxed{\\frac{5}{12}}")
        assert result is not None
        # Should parse to 5/12
        assert "5" in result and "12" in result

    def test_plain_number(self):
        result = parse_latex_answer("42")
        assert result is not None
        assert "42" in result

    def test_sqrt(self):
        result = parse_latex_answer("\\sqrt{2}")
        assert result is not None


class TestSetEqual:
    """Test set/list equality."""

    def test_same_set(self):
        from src.utils.math_equivalence import set_equal
        assert set_equal("{1, 2, 3}", "{1, 2, 3}") is True

    def test_reordered_set(self):
        from src.utils.math_equivalence import set_equal
        assert set_equal("{3, 1, 2}", "{1, 2, 3}") is True

    def test_different_sets(self):
        from src.utils.math_equivalence import set_equal
        assert set_equal("{1, 2}", "{1, 3}") is False

    def test_different_lengths(self):
        from src.utils.math_equivalence import set_equal
        assert set_equal("{1, 2}", "{1, 2, 3}") is False

    def test_set_with_brackets(self):
        from src.utils.math_equivalence import set_equal
        assert set_equal("[1, 2]", "[2, 1]") is True

    def test_non_set_strings(self):
        from src.utils.math_equivalence import set_equal
        assert set_equal("42", "42") is False  # not set format

    def test_empty_sets(self):
        from src.utils.math_equivalence import set_equal
        assert set_equal("{}", "{}") is True
        assert set_equal("[]", "[]") is True

    def test_symbolic_set_elements(self):
        from src.utils.math_equivalence import set_equal
        assert set_equal("{1/2, 1}", "{1, 1/2}") is True


class TestTryParseExpr:
    """Test _try_parse_expr helper."""

    def test_simple_number(self):
        from src.utils.math_equivalence import _try_parse_expr
        result = _try_parse_expr("42")
        assert result is not None

    def test_fraction(self):
        from src.utils.math_equivalence import _try_parse_expr
        result = _try_parse_expr("3/4")
        assert result is not None

    def test_latex_frac(self):
        from src.utils.math_equivalence import _try_parse_expr
        result = _try_parse_expr("\\frac{1}{2}")
        assert result is not None

    def test_invalid(self):
        from src.utils.math_equivalence import _try_parse_expr
        result = _try_parse_expr("totally invalid !!! @#$")
        assert result is None

    def test_empty(self):
        from src.utils.math_equivalence import _try_parse_expr
        result = _try_parse_expr("")
        assert result is None

    def test_caret_exponent(self):
        from src.utils.math_equivalence import _try_parse_expr
        result = _try_parse_expr("2^3")
        assert result is not None
        import sympy
        assert result == sympy.Integer(8)


class TestNormalizeAnswer:
    """Test answer normalization."""

    def test_strip_whitespace(self):
        assert normalize_answer("  42  ") == "42"

    def test_strip_trailing_period(self):
        assert normalize_answer("42.") == "42"

    def test_strip_dollar_signs(self):
        assert normalize_answer("$42$") == "42"

    def test_normalize_whitespace(self):
        assert normalize_answer("x  +  1") == "x + 1"

    def test_empty_string(self):
        assert normalize_answer("") == ""

    def test_non_string_input(self):
        assert normalize_answer(42) == "42"

    def test_multiple_dollar_signs(self):
        assert normalize_answer("$x + $y$") == "x + y"


class TestNumericEqualEdgeCases:
    """Additional edge cases for numeric_equal."""

    def test_sympy_expressions(self):
        assert numeric_equal("sqrt(2)", "1.41421356", tol=1e-5) is True

    def test_negative_values(self):
        assert numeric_equal("-3", "-3.0") is True

    def test_complex_fallback(self):
        """Non-numeric strings should return False."""
        assert numeric_equal("abc", "def") is False


class TestFractionEqualEdgeCases:
    """Additional edge cases for fraction_equal."""

    def test_integer_as_fraction(self):
        assert fraction_equal("2", "2/1") is True

    def test_negative_fraction(self):
        assert fraction_equal("-1/2", "-1/2") is True

    def test_symbolic_fraction(self):
        """Sympy fallback for expressions."""
        assert fraction_equal("6/4", "3/2") is True

    def test_invalid_fraction(self):
        assert fraction_equal("abc", "def") is False


class TestParseLatexAnswerEdgeCases:
    """Additional edge cases for parse_latex_answer."""

    def test_text_wrapper(self):
        result = parse_latex_answer("\\text{hello}")
        assert result is not None
        assert "hello" in result

    def test_mathrm_wrapper(self):
        result = parse_latex_answer("\\mathrm{cm}")
        assert result is not None
        assert "cm" in result

    def test_left_right_removed(self):
        result = parse_latex_answer("\\left(x\\right)")
        assert result is not None
        assert "left" not in result
        assert "right" not in result

    def test_spacing_commands(self):
        result = parse_latex_answer("x\\,y\\;z\\!w")
        assert result is not None
