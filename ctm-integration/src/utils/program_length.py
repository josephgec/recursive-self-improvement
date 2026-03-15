"""Program length measurement utilities using AST analysis."""

from __future__ import annotations

import ast
import tokenize
import io
from dataclasses import dataclass
from typing import Optional


@dataclass
class ProgramLength:
    """Measurements of program length/complexity."""

    ast_nodes: int
    lines: int
    tokens: int
    characters: int

    @property
    def total_score(self) -> float:
        """Weighted combination of length measures."""
        return self.ast_nodes * 1.0 + self.lines * 0.5 + self.tokens * 0.1


def measure_program_length(code: str) -> ProgramLength:
    """Measure program length using multiple metrics.

    Args:
        code: Python source code string.

    Returns:
        ProgramLength with ast_nodes, lines, tokens, characters.
    """
    ast_nodes = _count_ast_nodes(code)
    lines = _count_lines(code)
    tokens = _count_tokens(code)
    characters = len(code.strip())

    return ProgramLength(
        ast_nodes=ast_nodes,
        lines=lines,
        tokens=tokens,
        characters=characters,
    )


def _count_ast_nodes(code: str) -> int:
    """Count the number of AST nodes in Python code."""
    try:
        tree = ast.parse(code)
        return sum(1 for _ in ast.walk(tree))
    except SyntaxError:
        # Fallback: estimate from token count
        return _count_tokens(code) // 2


def _count_lines(code: str) -> int:
    """Count non-empty, non-comment lines."""
    count = 0
    for line in code.strip().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            count += 1
    return max(count, 1)


def _count_tokens(code: str) -> int:
    """Count Python tokens in code."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
        # Filter out ENCODING, NEWLINE, NL, ENDMARKER, COMMENT
        meaningful = [
            t
            for t in tokens
            if t.type not in (tokenize.ENCODING, tokenize.NEWLINE, tokenize.NL,
                              tokenize.ENDMARKER, tokenize.COMMENT, tokenize.INDENT,
                              tokenize.DEDENT)
        ]
        return max(len(meaningful), 1)
    except tokenize.TokenError:
        # Fallback: split on whitespace
        return max(len(code.split()), 1)
