"""Tests for AST-based code mutation utilities."""

from __future__ import annotations

import ast
import textwrap

import pytest

from src.utils.code_mutators import (
    add_dead_code,
    add_syntax_error,
    inflate_complexity,
    invert_condition,
    remove_branch,
    swap_variable,
)


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture
def simple_fn() -> str:
    return textwrap.dedent("""\
        def solve(x):
            if x > 0:
                return x * 2
            else:
                return 0
    """)


@pytest.fixture
def multi_fn() -> str:
    return textwrap.dedent("""\
        import math

        def compute(a, b):
            if a == b:
                return 0
            result = a + b
            return result

        def helper(x):
            for i in range(x):
                if i > 5:
                    break
            return i
    """)


# ------------------------------------------------------------------ #
# add_syntax_error
# ------------------------------------------------------------------ #


class TestAddSyntaxError:
    def test_missing_colon(self, simple_fn: str) -> None:
        result = add_syntax_error(simple_fn, "missing_colon")
        assert result != simple_fn
        with pytest.raises(SyntaxError):
            ast.parse(result)

    def test_unmatched_paren(self) -> None:
        code = "def f(x):\n    return x + (1 + 2)\n"
        result = add_syntax_error(code, "unmatched_paren")
        assert result != code
        # The closing paren is removed, so it should be unbalanced
        assert result.count("(") != result.count(")")

    def test_bad_indent(self, simple_fn: str) -> None:
        result = add_syntax_error(simple_fn, "bad_indent")
        assert result != simple_fn

    def test_fallback_for_no_colon(self) -> None:
        code = "x = 1\ny = 2\n"
        result = add_syntax_error(code, "missing_colon")
        # No colon to remove, so fallback appends broken statement
        assert "def broken(" in result

    def test_fallback_for_no_paren(self) -> None:
        code = "x = 1\ny = 2\n"
        result = add_syntax_error(code, "unmatched_paren")
        assert "def broken(" in result

    def test_fallback_for_no_indent(self) -> None:
        code = "x = 1\n"
        result = add_syntax_error(code, "bad_indent")
        assert "def broken(" in result


# ------------------------------------------------------------------ #
# invert_condition
# ------------------------------------------------------------------ #


class TestInvertCondition:
    def test_invert_gt(self) -> None:
        code = "if x > 0:\n    pass\n"
        result = invert_condition(code)
        assert result != code
        # > should become <=
        tree = ast.parse(result)
        # Find the If node and check its comparison
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                assert isinstance(node.ops[0], ast.LtE)
                break

    def test_invert_eq(self) -> None:
        code = "if x == 0:\n    pass\n"
        result = invert_condition(code)
        tree = ast.parse(result)
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                assert isinstance(node.ops[0], ast.NotEq)
                break

    def test_invert_noteq(self) -> None:
        code = "if x != 0:\n    pass\n"
        result = invert_condition(code)
        tree = ast.parse(result)
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                assert isinstance(node.ops[0], ast.Eq)
                break

    def test_invert_lt(self) -> None:
        code = "if x < 10:\n    pass\n"
        result = invert_condition(code)
        tree = ast.parse(result)
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                assert isinstance(node.ops[0], ast.GtE)
                break

    def test_invert_gte(self) -> None:
        code = "if x >= 10:\n    pass\n"
        result = invert_condition(code)
        tree = ast.parse(result)
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                assert isinstance(node.ops[0], ast.Lt)
                break

    def test_invert_lte(self) -> None:
        code = "if x <= 10:\n    pass\n"
        result = invert_condition(code)
        tree = ast.parse(result)
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                assert isinstance(node.ops[0], ast.Gt)
                break

    def test_invert_is(self) -> None:
        code = "if x is None:\n    pass\n"
        result = invert_condition(code)
        tree = ast.parse(result)
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                assert isinstance(node.ops[0], ast.IsNot)
                break

    def test_invert_is_not(self) -> None:
        code = "if x is not None:\n    pass\n"
        result = invert_condition(code)
        tree = ast.parse(result)
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                assert isinstance(node.ops[0], ast.Is)
                break

    def test_no_condition_returns_original(self) -> None:
        code = "x = 1\ny = 2\n"
        result = invert_condition(code)
        assert result == code

    def test_syntax_error_returns_original(self) -> None:
        code = "def broken("
        result = invert_condition(code)
        assert result == code

    def test_result_parses(self, simple_fn: str) -> None:
        result = invert_condition(simple_fn)
        ast.parse(result)  # should not raise


# ------------------------------------------------------------------ #
# swap_variable
# ------------------------------------------------------------------ #


class TestSwapVariable:
    def test_swap(self) -> None:
        code = "a = 1\nb = 2\nresult = a + b\n"
        result = swap_variable(code, "a", "b")
        assert result != code
        # a and b should be swapped
        tree = ast.parse(result)
        names = [
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        ]
        # First assignment target should now be "b" (was "a")
        assert names[0] == "b"

    def test_swap_preserves_syntax(self, simple_fn: str) -> None:
        result = swap_variable(simple_fn, "x", "solve")
        ast.parse(result)  # should not raise

    def test_swap_syntax_error_returns_original(self) -> None:
        code = "def broken("
        result = swap_variable(code, "a", "b")
        assert result == code


# ------------------------------------------------------------------ #
# remove_branch
# ------------------------------------------------------------------ #


class TestRemoveBranch:
    def test_removes_else(self, simple_fn: str) -> None:
        result = remove_branch(simple_fn)
        assert result != simple_fn
        # The else branch should be gone
        tree = ast.parse(result)
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                assert node.orelse == []
                break

    def test_no_branch_returns_same(self) -> None:
        code = "x = 1\ny = 2\n"
        result = remove_branch(code)
        # No if/else to remove, should be equivalent
        assert ast.dump(ast.parse(result)) == ast.dump(ast.parse(code))

    def test_syntax_error_returns_original(self) -> None:
        code = "def broken("
        result = remove_branch(code)
        assert result == code

    def test_result_parses(self, simple_fn: str) -> None:
        result = remove_branch(simple_fn)
        ast.parse(result)


# ------------------------------------------------------------------ #
# add_dead_code
# ------------------------------------------------------------------ #


class TestAddDeadCode:
    def test_adds_dead_code(self, simple_fn: str) -> None:
        result = add_dead_code(simple_fn, lines_to_add=5)
        assert result != simple_fn
        # Should parse fine
        ast.parse(result)
        # Dead code contains "if False"
        assert "False" in result

    def test_dead_code_after_imports(self, multi_fn: str) -> None:
        result = add_dead_code(multi_fn, lines_to_add=3)
        ast.parse(result)
        assert "False" in result

    def test_syntax_error_returns_original(self) -> None:
        code = "def broken("
        result = add_dead_code(code)
        assert result == code

    def test_output_longer_than_input(self, simple_fn: str) -> None:
        result = add_dead_code(simple_fn, lines_to_add=10)
        assert len(result) > len(simple_fn)


# ------------------------------------------------------------------ #
# inflate_complexity
# ------------------------------------------------------------------ #


class TestInflateComplexity:
    def test_inflates_complexity(self, simple_fn: str) -> None:
        result = inflate_complexity(simple_fn, factor=3)
        assert result != simple_fn
        ast.parse(result)
        # Should have more AST nodes
        original_nodes = sum(1 for _ in ast.walk(ast.parse(simple_fn)))
        inflated_nodes = sum(1 for _ in ast.walk(ast.parse(result)))
        assert inflated_nodes > original_nodes

    def test_multiple_functions(self, multi_fn: str) -> None:
        result = inflate_complexity(multi_fn, factor=2)
        ast.parse(result)
        original_nodes = sum(1 for _ in ast.walk(ast.parse(multi_fn)))
        inflated_nodes = sum(1 for _ in ast.walk(ast.parse(result)))
        assert inflated_nodes > original_nodes

    def test_factor_zero_is_identity(self, simple_fn: str) -> None:
        result = inflate_complexity(simple_fn, factor=0)
        # With factor=0, no extra stmts are added
        assert ast.dump(ast.parse(result)) == ast.dump(ast.parse(simple_fn))

    def test_syntax_error_returns_original(self) -> None:
        code = "def broken("
        result = inflate_complexity(code)
        assert result == code
