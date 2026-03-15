"""LaTeX parsing utilities for math answer extraction."""

from __future__ import annotations

import re

import sympy


def extract_boxed(text: str) -> str | None:
    """Extract content from \\boxed{...} in text.

    Handles nested braces correctly.  Returns None if no \\boxed found.
    """
    idx = text.find("\\boxed{")
    if idx == -1:
        idx = text.find("\\boxed ")
        if idx == -1:
            return None
        # Simple case: \boxed followed by a single token
        rest = text[idx + 7:].strip()
        token = rest.split()[0] if rest else ""
        return token.rstrip(".")

    # Find the matching closing brace
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth != 0:
        # Unmatched braces; return everything after \boxed{
        return text[start:].strip()

    return text[start : i - 1].strip()


def latex_to_sympy(latex: str) -> sympy.Basic | None:
    """Convert a LaTeX math expression to a SymPy expression.

    Returns None on failure.
    """
    if not latex or not latex.strip():
        return None

    s = latex.strip()

    # Remove display math delimiters
    s = s.strip("$")
    s = re.sub(r"^\\\[", "", s)
    s = re.sub(r"\\\]$", "", s)
    s = s.strip()

    # Pre-process common LaTeX patterns into parseable form
    # \frac{a}{b} -> ((a)/(b))
    def replace_frac(m):
        return f"(({m.group(1)})/({m.group(2)}))"

    # Iteratively replace \frac (handles nested)
    for _ in range(5):
        new_s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", replace_frac, s)
        if new_s == s:
            break
        s = new_s

    # \sqrt[n]{x} -> (x)**(1/(n))
    s = re.sub(r"\\sqrt\[([^]]*)\]\{([^{}]*)\}", r"((\2))**(1/(\1))", s)
    # \sqrt{x} -> sqrt(x)
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)

    # \cdot -> *
    s = s.replace("\\cdot", "*")
    s = s.replace("\\times", "*")
    s = s.replace("\\div", "/")

    # Exponents: ^ -> **
    s = s.replace("^", "**")

    # Remove \left, \right
    s = s.replace("\\left", "").replace("\\right", "")

    # \pi -> pi, \infty -> oo
    s = s.replace("\\pi", "pi")
    s = s.replace("\\infty", "oo")
    s = s.replace("\\infinity", "oo")

    # Trig functions
    for fn in ["sin", "cos", "tan", "sec", "csc", "cot",
               "arcsin", "arccos", "arctan", "log", "ln", "exp"]:
        s = s.replace(f"\\{fn}", fn)

    # \ln -> log (SymPy uses log for natural log)
    s = s.replace("ln(", "log(")

    # Remove remaining \text{...} wrappers
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)

    # Remove any remaining backslashes that aren't part of known commands
    s = re.sub(r"\\[a-zA-Z]+", "", s)

    # Clean up
    s = s.replace("{", "(").replace("}", ")")
    s = s.strip()

    if not s:
        return None

    try:
        return sympy.sympify(s, rational=True)
    except Exception:
        return None


def strip_latex_formatting(text: str) -> str:
    """Remove LaTeX formatting from text, preserving mathematical content."""
    s = text.strip()

    # Remove display math delimiters
    s = s.strip("$")
    s = re.sub(r"\\\[", "", s)
    s = re.sub(r"\\\]", "", s)
    s = re.sub(r"\\\(", "", s)
    s = re.sub(r"\\\)", "", s)

    # Replace common LaTeX commands
    s = s.replace("\\cdot", "*")
    s = s.replace("\\times", "*")
    s = s.replace("\\div", "/")
    s = s.replace("\\pi", "pi")
    s = s.replace("\\infty", "inf")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", " ")
    s = s.replace("\\;", " ")
    s = s.replace("\\!", "")

    # \frac{a}{b} -> (a)/(b)
    for _ in range(5):
        new_s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
        if new_s == s:
            break
        s = new_s

    # \sqrt{x} -> sqrt(x)
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)

    # \text{...} -> ...
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)

    # Remove remaining backslash commands
    s = re.sub(r"\\[a-zA-Z]+", "", s)

    # Clean braces
    s = s.replace("{", "").replace("}", "")

    return s.strip()
