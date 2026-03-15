"""Extract and normalize answers from execution output and prose text."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.utils.latex_parser import extract_boxed, latex_to_sympy, strip_latex_formatting
from src.utils.math_equivalence import normalize_answer
from src.utils.logging import get_logger

logger = get_logger("answer_extractor")


@dataclass
class ExtractedAnswer:
    """An extracted answer with its source and normalized form."""

    raw: str
    normalized: str
    source: str  # "variable", "stdout", "prose_boxed", "prose_text"
    confidence: float = 1.0


class AnswerExtractor:
    """Extract answers from code execution results or prose text."""

    # ── extraction from code execution ──────────────────────────────

    def extract_from_execution(
        self,
        namespace: dict[str, Any] | None = None,
        stdout: str = "",
    ) -> ExtractedAnswer | None:
        """Extract answer from code execution results.

        Priority:
        1. `answer` variable in the execution namespace
        2. "Answer: ..." line in stdout
        3. Last print output in stdout
        """
        # 1. Namespace answer variable
        if namespace and "answer" in namespace:
            raw = str(namespace["answer"])
            return ExtractedAnswer(
                raw=raw,
                normalized=self.normalize(raw),
                source="variable",
                confidence=1.0,
            )

        # 2. "Answer: ..." pattern in stdout
        if stdout:
            answer_match = re.search(
                r"Answer:\s*(.+?)$", stdout, re.MULTILINE
            )
            if answer_match:
                raw = answer_match.group(1).strip()
                return ExtractedAnswer(
                    raw=raw,
                    normalized=self.normalize(raw),
                    source="stdout",
                    confidence=0.9,
                )

            # 3. Last print line
            lines = [l.strip() for l in stdout.strip().split("\n") if l.strip()]
            if lines:
                raw = lines[-1]
                return ExtractedAnswer(
                    raw=raw,
                    normalized=self.normalize(raw),
                    source="stdout",
                    confidence=0.5,
                )

        return None

    # ── extraction from prose text ──────────────────────────────────

    def extract_from_prose(self, text: str) -> ExtractedAnswer | None:
        r"""Extract answer from prose/CoT text.

        Looks for:
        1. \boxed{...} notation
        2. "the answer is ..." pattern
        """
        if not text:
            return None

        # 1. \boxed{...}
        boxed = extract_boxed(text)
        if boxed is not None:
            return ExtractedAnswer(
                raw=boxed,
                normalized=self.normalize(boxed),
                source="prose_boxed",
                confidence=1.0,
            )

        # 2. "the answer is ..." patterns
        patterns = [
            r"[Tt]he\s+(?:final\s+)?answer\s+is\s*[:\s]*(.+?)(?:\.|$)",
            r"[Aa]nswer\s*[=:]\s*(.+?)(?:\.|$)",
            r"[Tt]herefore,?\s+(?:the\s+answer\s+is\s+)?(.+?)(?:\.|$)",
            r"[Hh]ence,?\s+(?:the\s+answer\s+is\s+)?(.+?)(?:\.|$)",
        ]
        for pattern in patterns:
            m = re.search(pattern, text, re.MULTILINE)
            if m:
                raw = m.group(1).strip()
                # Clean up common trailing patterns
                raw = re.sub(r"\s*\\?\.$", "", raw)
                return ExtractedAnswer(
                    raw=raw,
                    normalized=self.normalize(raw),
                    source="prose_text",
                    confidence=0.7,
                )

        return None

    # ── normalization ───────────────────────────────────────────────

    def normalize(self, answer: str) -> str:
        """Normalize an answer string for comparison.

        Handles SymPy expressions, fractions, lists, LaTeX.
        """
        if not answer:
            return ""

        s = normalize_answer(answer)

        # Strip LaTeX if present
        if "\\" in s:
            s = strip_latex_formatting(s)
            s = normalize_answer(s)

        # Try to simplify via SymPy
        try:
            import sympy
            expr = sympy.sympify(s, rational=True)
            # For numbers, return simplified string
            if expr.is_number:
                # If it's a rational, show as fraction if not integer
                if expr.is_Rational and not expr.is_Integer:
                    return str(expr)
                return str(expr)
            return str(expr)
        except Exception:
            pass

        return s
