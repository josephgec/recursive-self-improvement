"""AST-based code mutation utilities for fault injection."""

from __future__ import annotations

import ast
import random
import textwrap
from typing import Optional


def add_syntax_error(code: str, error_type: str = "missing_colon") -> str:
    """Add a syntax error to valid Python code.

    Args:
        code: Valid Python source code.
        error_type: One of 'missing_colon', 'unmatched_paren', 'bad_indent'.

    Returns:
        Code with a syntax error.
    """
    lines = code.split("\n")

    if error_type == "missing_colon":
        for i, line in enumerate(lines):
            if line.rstrip().endswith(":"):
                lines[i] = line.rstrip().rstrip(":")
                return "\n".join(lines)
    elif error_type == "unmatched_paren":
        for i, line in enumerate(lines):
            if "(" in line:
                lines[i] = line.replace(")", "", 1)
                return "\n".join(lines)
    elif error_type == "bad_indent":
        for i, line in enumerate(lines):
            if line.startswith("    ") and line.strip():
                lines[i] = " " + line.lstrip()
                return "\n".join(lines)

    # Fallback: append a broken statement
    lines.append("def broken(")
    return "\n".join(lines)


def invert_condition(code: str) -> str:
    """Invert a boolean condition in the code using AST transformation.

    Finds the first comparison or boolean and inverts it.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    class ConditionInverter(ast.NodeTransformer):
        _inverted = False

        def visit_Compare(self, node: ast.Compare) -> ast.AST:
            if self._inverted:
                return node
            self._inverted = True

            inversion_map = {
                ast.Eq: ast.NotEq,
                ast.NotEq: ast.Eq,
                ast.Lt: ast.GtE,
                ast.GtE: ast.Lt,
                ast.Gt: ast.LtE,
                ast.LtE: ast.Gt,
                ast.Is: ast.IsNot,
                ast.IsNot: ast.Is,
            }

            new_ops = []
            for op in node.ops:
                op_type = type(op)
                if op_type in inversion_map:
                    new_ops.append(inversion_map[op_type]())
                else:
                    new_ops.append(op)
            node.ops = new_ops
            return node

    transformer = ConditionInverter()
    new_tree = transformer.visit(tree)

    if not transformer._inverted:
        return code

    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)


def swap_variable(code: str, var_a: str, var_b: str) -> str:
    """Swap all occurrences of two variable names in the code.

    Uses AST to only swap Name nodes (not attributes, strings, etc.).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    class VariableSwapper(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.Name:
            if node.id == var_a:
                node.id = var_b
            elif node.id == var_b:
                node.id = var_a
            return node

    new_tree = VariableSwapper().visit(tree)
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)


def remove_branch(code: str) -> str:
    """Remove the first else/elif branch from the code.

    This is a logic error that silently changes behavior.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    class BranchRemover(ast.NodeTransformer):
        _removed = False

        def visit_If(self, node: ast.If) -> ast.AST:
            if self._removed:
                return self.generic_visit(node)

            if node.orelse:
                self._removed = True
                node.orelse = []
            return self.generic_visit(node)

    new_tree = BranchRemover().visit(tree)
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)


def add_dead_code(code: str, lines_to_add: int = 10) -> str:
    """Add dead code (unreachable statements) to inflate size without changing behavior."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    dead_code_template = textwrap.dedent("""\
    if False:
        _dead_var_{i} = {i} * 2
        for _j_{i} in range({i}):
            _dead_var_{i} += _j_{i}
    """)

    dead_statements = []
    for i in range(lines_to_add):
        dead_tree = ast.parse(dead_code_template.format(i=i))
        dead_statements.extend(dead_tree.body)

    # Insert dead code after imports but before main code
    insert_point = 0
    for j, node in enumerate(tree.body):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            insert_point = j + 1

    for k, stmt in enumerate(dead_statements):
        tree.body.insert(insert_point + k, stmt)

    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def inflate_complexity(code: str, factor: int = 3) -> str:
    """Inflate the cyclomatic complexity of functions by wrapping bodies
    in redundant conditionals and adding extra branches.

    Args:
        code: Source code string.
        factor: How many redundant branches to add per function.

    Returns:
        Code with inflated complexity.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    class ComplexityInflator(ast.NodeTransformer):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
            self.generic_visit(node)

            extra_stmts = []
            for i in range(factor):
                # Add: if True: pass
                guard = ast.If(
                    test=ast.Compare(
                        left=ast.Constant(value=i),
                        ops=[ast.Lt()],
                        comparators=[ast.Constant(value=i + 1)],
                    ),
                    body=[ast.Pass()],
                    orelse=[ast.Pass()],
                )
                extra_stmts.append(guard)

            node.body = extra_stmts + node.body
            return node

    new_tree = ComplexityInflator().visit(tree)
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)
