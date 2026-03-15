"""Shared metric computation helpers."""

from __future__ import annotations

import ast
import math
from typing import Dict, List, Optional, Sequence, Tuple


def compute_cyclomatic_complexity(code: str) -> int:
    """Compute cyclomatic complexity of Python source code.

    Cyclomatic complexity = number of decision points + 1.
    Decision points: if, elif, for, while, except, with, and, or,
    assert, ternary (IfExp).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return -1

    complexity = 1  # Base complexity

    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.IfExp)):
            complexity += 1
        elif isinstance(node, (ast.For, ast.While)):
            complexity += 1
        elif isinstance(node, ast.ExceptHandler):
            complexity += 1
        elif isinstance(node, ast.With):
            complexity += 1
        elif isinstance(node, ast.Assert):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            # Each 'and'/'or' adds a decision point
            complexity += len(node.values) - 1

    return complexity


def count_ast_nodes(code: str) -> int:
    """Count total AST nodes in Python source code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return -1
    return sum(1 for _ in ast.walk(tree))


def compute_nesting_depth(code: str) -> int:
    """Compute maximum nesting depth of control structures."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return -1

    max_depth = 0

    def _walk(node: ast.AST, depth: int) -> None:
        nonlocal max_depth
        nesting_types = (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.FunctionDef, ast.ClassDef)

        if isinstance(node, nesting_types):
            depth += 1
            max_depth = max(max_depth, depth)

        for child in ast.iter_child_nodes(node):
            _walk(child, depth)

    _walk(tree, 0)
    return max_depth


def count_functions(code: str) -> int:
    """Count the number of function definitions."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return -1
    return sum(1 for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))


def lines_of_code(code: str) -> int:
    """Count non-empty, non-comment lines of code."""
    return sum(
        1 for line in code.split("\n")
        if line.strip() and not line.strip().startswith("#")
    )


def moving_average(values: Sequence[float], window: int = 5) -> List[float]:
    """Compute a simple moving average."""
    if len(values) < window:
        return list(values)
    result = []
    for i in range(len(values) - window + 1):
        avg = sum(values[i:i + window]) / window
        result.append(avg)
    return result


def detect_trend(values: Sequence[float], window: int = 5) -> str:
    """Detect the trend in a sequence of values.

    Returns: 'improving', 'declining', 'flat', 'oscillating', or 'insufficient_data'.
    """
    if len(values) < 3:
        return "insufficient_data"

    ma = moving_average(values, min(window, len(values)))
    if len(ma) < 2:
        return "insufficient_data"

    diffs = [ma[i + 1] - ma[i] for i in range(len(ma) - 1)]

    avg_diff = sum(diffs) / len(diffs)
    sign_changes = sum(
        1 for i in range(len(diffs) - 1)
        if (diffs[i] > 0) != (diffs[i + 1] > 0)
    )

    # Oscillating if many sign changes
    if sign_changes > len(diffs) * 0.6:
        return "oscillating"

    threshold = 0.01
    if avg_diff > threshold:
        return "improving"
    elif avg_diff < -threshold:
        return "declining"
    else:
        return "flat"


def compute_code_similarity(code_a: str, code_b: str) -> float:
    """Compute a simple similarity score (0-1) between two code strings.

    Uses Jaccard similarity on the set of lines.
    """
    lines_a = set(line.strip() for line in code_a.split("\n") if line.strip())
    lines_b = set(line.strip() for line in code_b.split("\n") if line.strip())

    if not lines_a and not lines_b:
        return 1.0
    if not lines_a or not lines_b:
        return 0.0

    intersection = lines_a & lines_b
    union = lines_a | lines_b
    return len(intersection) / len(union)


def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide with a default for zero denominator."""
    if denominator == 0:
        return default
    return numerator / denominator


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a value to [low, high]."""
    return max(low, min(high, value))


def exponential_moving_average(
    values: Sequence[float], alpha: float = 0.3
) -> List[float]:
    """Compute exponential moving average."""
    if not values:
        return []
    result = [values[0]]
    for v in values[1:]:
        result.append(alpha * v + (1 - alpha) * result[-1])
    return result
