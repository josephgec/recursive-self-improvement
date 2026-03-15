"""Tests for shared metric computation helpers."""

from __future__ import annotations

import textwrap

import pytest

from src.utils.metrics import (
    clamp,
    compute_code_similarity,
    compute_cyclomatic_complexity,
    compute_nesting_depth,
    count_ast_nodes,
    count_functions,
    detect_trend,
    exponential_moving_average,
    lines_of_code,
    moving_average,
    safe_division,
)


# ------------------------------------------------------------------ #
# compute_cyclomatic_complexity
# ------------------------------------------------------------------ #


class TestCyclomaticComplexity:
    def test_simple_function(self) -> None:
        code = "def f(x):\n    return x\n"
        # Base complexity = 1, function def = 0 decision points
        assert compute_cyclomatic_complexity(code) == 1

    def test_with_if(self) -> None:
        code = "def f(x):\n    if x > 0:\n        return x\n    return 0\n"
        # 1 base + 1 if = 2
        assert compute_cyclomatic_complexity(code) == 2

    def test_with_for_and_while(self) -> None:
        code = textwrap.dedent("""\
            def f(x):
                for i in range(x):
                    while i > 0:
                        i -= 1
        """)
        # 1 base + 1 for + 1 while = 3
        assert compute_cyclomatic_complexity(code) == 3

    def test_with_except(self) -> None:
        code = textwrap.dedent("""\
            def f():
                try:
                    x = 1
                except ValueError:
                    x = 0
        """)
        # 1 base + 1 except = 2
        assert compute_cyclomatic_complexity(code) == 2

    def test_with_bool_op(self) -> None:
        code = "def f(x, y):\n    if x and y:\n        return True\n"
        # 1 base + 1 if + 1 bool_op (and has 2 values -> 1 decision) = 3
        assert compute_cyclomatic_complexity(code) == 3

    def test_with_assert(self) -> None:
        code = "def f(x):\n    assert x > 0\n"
        # 1 base + 1 assert = 2
        assert compute_cyclomatic_complexity(code) == 2

    def test_with_ternary(self) -> None:
        code = "x = 1 if True else 0\n"
        # 1 base + 1 IfExp = 2
        assert compute_cyclomatic_complexity(code) == 2

    def test_with_with_statement(self) -> None:
        code = "def f():\n    with open('f') as fh:\n        pass\n"
        # 1 base + 1 with = 2
        assert compute_cyclomatic_complexity(code) == 2

    def test_syntax_error(self) -> None:
        code = "def broken("
        assert compute_cyclomatic_complexity(code) == -1


# ------------------------------------------------------------------ #
# count_ast_nodes
# ------------------------------------------------------------------ #


class TestCountAstNodes:
    def test_simple_code(self) -> None:
        code = "x = 1\n"
        count = count_ast_nodes(code)
        assert count > 0

    def test_syntax_error(self) -> None:
        assert count_ast_nodes("def broken(") == -1


# ------------------------------------------------------------------ #
# compute_nesting_depth
# ------------------------------------------------------------------ #


class TestNestingDepth:
    def test_no_nesting(self) -> None:
        code = "x = 1\ny = 2\n"
        assert compute_nesting_depth(code) == 0

    def test_one_level(self) -> None:
        code = "if True:\n    x = 1\n"
        assert compute_nesting_depth(code) == 1

    def test_deep_nesting(self) -> None:
        code = textwrap.dedent("""\
            def f():
                if True:
                    for i in range(10):
                        while True:
                            break
        """)
        # def(1) -> if(2) -> for(3) -> while(4) = 4
        assert compute_nesting_depth(code) == 4

    def test_syntax_error(self) -> None:
        assert compute_nesting_depth("def broken(") == -1


# ------------------------------------------------------------------ #
# count_functions
# ------------------------------------------------------------------ #


class TestCountFunctions:
    def test_no_functions(self) -> None:
        code = "x = 1\n"
        assert count_functions(code) == 0

    def test_multiple_functions(self) -> None:
        code = "def a():\n    pass\ndef b():\n    pass\n"
        assert count_functions(code) == 2

    def test_async_function(self) -> None:
        code = "async def f():\n    pass\n"
        assert count_functions(code) == 1

    def test_syntax_error(self) -> None:
        assert count_functions("def broken(") == -1


# ------------------------------------------------------------------ #
# lines_of_code
# ------------------------------------------------------------------ #


class TestLinesOfCode:
    def test_simple(self) -> None:
        code = "x = 1\ny = 2\n"
        assert lines_of_code(code) == 2

    def test_with_comments_and_blanks(self) -> None:
        code = "# comment\nx = 1\n\n# another\ny = 2\n"
        assert lines_of_code(code) == 2

    def test_empty(self) -> None:
        assert lines_of_code("") == 0


# ------------------------------------------------------------------ #
# moving_average
# ------------------------------------------------------------------ #


class TestMovingAverage:
    def test_short_input(self) -> None:
        values = [1.0, 2.0]
        result = moving_average(values, window=5)
        assert result == [1.0, 2.0]

    def test_normal(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = moving_average(values, window=3)
        assert len(result) == 3
        assert result[0] == pytest.approx(2.0)  # (1+2+3)/3
        assert result[1] == pytest.approx(3.0)  # (2+3+4)/3
        assert result[2] == pytest.approx(4.0)  # (3+4+5)/3


# ------------------------------------------------------------------ #
# detect_trend
# ------------------------------------------------------------------ #


class TestDetectTrend:
    def test_insufficient_data(self) -> None:
        assert detect_trend([0.5, 0.6]) == "insufficient_data"

    def test_improving(self) -> None:
        values = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
        assert detect_trend(values) == "improving"

    def test_declining(self) -> None:
        values = [0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
        assert detect_trend(values) == "declining"

    def test_flat(self) -> None:
        values = [0.50, 0.50, 0.50, 0.50, 0.50, 0.50]
        assert detect_trend(values) == "flat"

    def test_oscillating(self) -> None:
        values = [0.5, 0.8, 0.5, 0.8, 0.5, 0.8, 0.5, 0.8, 0.5, 0.8]
        assert detect_trend(values) == "oscillating"

    def test_ma_single_point(self) -> None:
        # 3 values but window equals length -> 1 MA value -> insufficient
        values = [0.5, 0.6, 0.7]
        result = detect_trend(values, window=3)
        # len(ma) = 1, < 2 -> insufficient_data
        assert result == "insufficient_data"


# ------------------------------------------------------------------ #
# compute_code_similarity
# ------------------------------------------------------------------ #


class TestCodeSimilarity:
    def test_identical(self) -> None:
        code = "x = 1\ny = 2\n"
        assert compute_code_similarity(code, code) == 1.0

    def test_completely_different(self) -> None:
        code_a = "x = 1\ny = 2\n"
        code_b = "a = 10\nb = 20\n"
        sim = compute_code_similarity(code_a, code_b)
        assert sim == 0.0

    def test_partial_overlap(self) -> None:
        code_a = "x = 1\ny = 2\nz = 3\n"
        code_b = "x = 1\ny = 2\nw = 4\n"
        sim = compute_code_similarity(code_a, code_b)
        # 2 common out of 4 unique -> 0.5
        assert sim == pytest.approx(0.5)

    def test_both_empty(self) -> None:
        assert compute_code_similarity("", "") == 1.0

    def test_one_empty(self) -> None:
        assert compute_code_similarity("x = 1\n", "") == 0.0
        assert compute_code_similarity("", "x = 1\n") == 0.0


# ------------------------------------------------------------------ #
# safe_division
# ------------------------------------------------------------------ #


class TestSafeDivision:
    def test_normal(self) -> None:
        assert safe_division(10, 5) == 2.0

    def test_zero_denominator(self) -> None:
        assert safe_division(10, 0) == 0.0

    def test_custom_default(self) -> None:
        assert safe_division(10, 0, default=-1.0) == -1.0


# ------------------------------------------------------------------ #
# clamp
# ------------------------------------------------------------------ #


class TestClamp:
    def test_within_range(self) -> None:
        assert clamp(0.5) == 0.5

    def test_below_low(self) -> None:
        assert clamp(-0.5) == 0.0

    def test_above_high(self) -> None:
        assert clamp(1.5) == 1.0

    def test_custom_range(self) -> None:
        assert clamp(15, low=10, high=20) == 15
        assert clamp(5, low=10, high=20) == 10
        assert clamp(25, low=10, high=20) == 20


# ------------------------------------------------------------------ #
# exponential_moving_average
# ------------------------------------------------------------------ #


class TestExponentialMovingAverage:
    def test_empty(self) -> None:
        assert exponential_moving_average([]) == []

    def test_single_value(self) -> None:
        assert exponential_moving_average([5.0]) == [5.0]

    def test_normal(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = exponential_moving_average(values, alpha=0.5)
        assert len(result) == 5
        assert result[0] == 1.0
        assert result[1] == pytest.approx(1.5)  # 0.5*2 + 0.5*1
        assert result[2] == pytest.approx(2.25)  # 0.5*3 + 0.5*1.5
