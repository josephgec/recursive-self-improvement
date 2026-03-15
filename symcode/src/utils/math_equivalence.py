"""Mathematical equivalence checking utilities."""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any

import sympy


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison.

    Strips whitespace, removes trailing periods, normalizes fraction and
    decimal representations.
    """
    if not isinstance(answer, str):
        answer = str(answer)
    answer = answer.strip()
    # Remove trailing period
    answer = answer.rstrip(".")
    # Remove dollar signs (LaTeX inline)
    answer = answer.replace("$", "")
    # Remove leading/trailing whitespace again
    answer = answer.strip()
    # Normalize whitespace
    answer = re.sub(r"\s+", " ", answer)
    return answer


def parse_latex_answer(latex: str) -> str | None:
    """Parse a LaTeX answer expression into a normalized string.

    Handles \\boxed{}, \\frac{}{}, \\sqrt{}, \\text{}, etc.
    Returns None if parsing fails completely.
    """
    from src.utils.latex_parser import extract_boxed, latex_to_sympy

    s = latex.strip()
    # Try extracting from \boxed first
    boxed = extract_boxed(s)
    if boxed is not None:
        s = boxed

    # Try converting via sympy
    try:
        expr = latex_to_sympy(s)
        if expr is not None:
            return str(expr)
    except Exception:
        pass

    # Fallback: strip common LaTeX formatting
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", " ").replace("\\;", " ").replace("\\!", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)

    # \frac{a}{b} -> a/b
    s = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", s)
    # \sqrt{a} -> sqrt(a)
    s = re.sub(r"\\sqrt\{([^}]*)\}", r"sqrt(\1)", s)

    return normalize_answer(s)


def _try_parse_expr(s: str) -> sympy.Basic | None:
    """Try parsing a string as a SymPy expression."""
    s = normalize_answer(s)
    if not s:
        return None

    # Try direct sympify
    try:
        return sympy.sympify(s, rational=True)
    except Exception:
        pass

    # Try after LaTeX-style transformations
    transformed = s
    transformed = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"((\1)/(\2))", transformed)
    transformed = re.sub(r"\\sqrt\{([^}]*)\}", r"sqrt(\1)", transformed)
    transformed = transformed.replace("^", "**")

    try:
        return sympy.sympify(transformed, rational=True)
    except Exception:
        pass

    return None


def _try_parse_set(s: str) -> set | None:
    """Try parsing a string as a set/list of values."""
    s = normalize_answer(s)
    # Match [a, b, c] or {a, b, c} patterns
    m = re.match(r"^[\[{]\s*(.*?)\s*[\]}]$", s)
    if not m:
        return None
    inner = m.group(1)
    if not inner:
        return set()
    parts = [p.strip() for p in inner.split(",")]
    result = set()
    for p in parts:
        expr = _try_parse_expr(p)
        if expr is not None:
            result.add(expr)
        else:
            result.add(p)
    return result


def symbolic_equal(a: str, b: str) -> bool:
    """Check if two expressions are symbolically equal via SymPy.

    Tries sympify on both, then checks if their difference simplifies to 0.
    """
    expr_a = _try_parse_expr(a)
    expr_b = _try_parse_expr(b)
    if expr_a is None or expr_b is None:
        return False

    try:
        diff = sympy.simplify(expr_a - expr_b)
        return diff == 0
    except Exception:
        return False


def numeric_equal(a: str, b: str, tol: float = 1e-6) -> bool:
    """Check if two expressions are numerically equal within tolerance."""
    try:
        val_a = complex(sympy.N(sympy.sympify(a, rational=True)))
        val_b = complex(sympy.N(sympy.sympify(b, rational=True)))
    except Exception:
        # Fallback: try float directly
        try:
            val_a = complex(float(a))
            val_b = complex(float(b))
        except Exception:
            return False

    try:
        return abs(val_a - val_b) < tol
    except Exception:
        return False


def set_equal(a: str, b: str) -> bool:
    """Check if two set/list expressions contain the same elements."""
    set_a = _try_parse_set(a)
    set_b = _try_parse_set(b)
    if set_a is None or set_b is None:
        return False
    if len(set_a) != len(set_b):
        return False
    # Try to match elements via symbolic equality
    remaining = list(set_b)
    for elem_a in set_a:
        found = False
        for i, elem_b in enumerate(remaining):
            try:
                if elem_a == elem_b:
                    found = True
                    remaining.pop(i)
                    break
                if isinstance(elem_a, sympy.Basic) and isinstance(elem_b, sympy.Basic):
                    if sympy.simplify(elem_a - elem_b) == 0:
                        found = True
                        remaining.pop(i)
                        break
            except Exception:
                continue
        if not found:
            return False
    return True


def fraction_equal(a: str, b: str) -> bool:
    """Check if two fraction/decimal representations are equal."""
    try:
        fa = Fraction(a).limit_denominator(10**12)
        fb = Fraction(b).limit_denominator(10**12)
        return fa == fb
    except (ValueError, ZeroDivisionError):
        pass

    # Try evaluating as expressions then comparing as fractions
    try:
        val_a = sympy.Rational(sympy.sympify(a, rational=True))
        val_b = sympy.Rational(sympy.sympify(b, rational=True))
        return val_a == val_b
    except Exception:
        return False
