"""AST-based code inspection for self-modification."""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FunctionAST:
    """Parsed function information."""

    name: str
    source: str
    ast_node: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    lineno: int = 0
    args: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)


@dataclass
class ComplexityReport:
    """Code complexity metrics."""

    ast_node_count: int = 0
    cyclomatic_complexity: int = 1
    max_nesting: int = 0
    lines_of_code: int = 0


class CodeInspector:
    """Inspects and analyzes code using AST parsing."""

    def parse(self, source: str) -> ast.Module:
        """Parse source code into an AST."""
        return ast.parse(textwrap.dedent(source))

    def get_functions(self, source: str) -> list[FunctionAST]:
        """Extract all function definitions from source code."""
        tree = self.parse(source)
        functions: list[FunctionAST] = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [a.arg for a in node.args.args]
                decorators = []
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Name):
                        decorators.append(dec.id)
                    elif isinstance(dec, ast.Attribute):
                        decorators.append(ast.dump(dec))

                func_source = ast.get_source_segment(source, node) or ""
                functions.append(
                    FunctionAST(
                        name=node.name,
                        source=func_source,
                        ast_node=node,
                        lineno=node.lineno,
                        args=args,
                        decorators=decorators,
                    )
                )

        return functions

    def get_complexity(self, source: str) -> ComplexityReport:
        """Compute complexity metrics for source code."""
        tree = self.parse(source)

        # AST node count
        node_count = sum(1 for _ in ast.walk(tree))

        # Cyclomatic complexity
        cyclomatic = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                cyclomatic += 1
            elif isinstance(node, ast.BoolOp):
                cyclomatic += len(node.values) - 1

        # Max nesting depth
        max_nesting = self._compute_max_nesting(tree)

        # Lines of code (non-empty, non-comment)
        lines = [
            l
            for l in source.strip().split("\n")
            if l.strip() and not l.strip().startswith("#")
        ]

        return ComplexityReport(
            ast_node_count=node_count,
            cyclomatic_complexity=cyclomatic,
            max_nesting=max_nesting,
            lines_of_code=len(lines),
        )

    def _compute_max_nesting(self, tree: ast.AST, depth: int = 0) -> int:
        """Compute maximum nesting depth."""
        max_depth = depth
        nesting_nodes = (
            ast.If, ast.While, ast.For, ast.With, ast.Try,
            ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
        )
        for child in ast.iter_child_nodes(tree):
            if isinstance(child, nesting_nodes):
                child_depth = self._compute_max_nesting(child, depth + 1)
                max_depth = max(max_depth, child_depth)
            else:
                child_depth = self._compute_max_nesting(child, depth)
                max_depth = max(max_depth, child_depth)
        return max_depth

    def find_patterns(self, source: str) -> list[dict[str, Any]]:
        """Find common code patterns that might be improved."""
        tree = self.parse(source)
        patterns: list[dict[str, Any]] = []

        for node in ast.walk(tree):
            # Hardcoded constants in function bodies
            if isinstance(node, (ast.Constant,)):
                if isinstance(node.value, (int, float)) and node.value not in (0, 1, -1, True, False):
                    patterns.append({
                        "type": "hardcoded_constant",
                        "value": node.value,
                        "lineno": node.lineno,
                    })

            # Fixed strategy (return statement with constant string)
            if isinstance(node, ast.Return):
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    patterns.append({
                        "type": "fixed_strategy",
                        "value": node.value.value,
                        "lineno": node.lineno,
                    })

        return patterns

    def compare_ast(self, source_a: str, source_b: str) -> dict[str, Any]:
        """Compare two source code snippets at the AST level."""
        tree_a = self.parse(source_a)
        tree_b = self.parse(source_b)

        nodes_a = [type(n).__name__ for n in ast.walk(tree_a)]
        nodes_b = [type(n).__name__ for n in ast.walk(tree_b)]

        added = []
        removed = []

        counts_a: dict[str, int] = {}
        counts_b: dict[str, int] = {}
        for n in nodes_a:
            counts_a[n] = counts_a.get(n, 0) + 1
        for n in nodes_b:
            counts_b[n] = counts_b.get(n, 0) + 1

        all_types = set(list(counts_a.keys()) + list(counts_b.keys()))
        for t in all_types:
            diff = counts_b.get(t, 0) - counts_a.get(t, 0)
            if diff > 0:
                added.extend([t] * diff)
            elif diff < 0:
                removed.extend([t] * (-diff))

        return {
            "nodes_before": len(nodes_a),
            "nodes_after": len(nodes_b),
            "added_node_types": added,
            "removed_node_types": removed,
            "structurally_identical": ast.dump(tree_a) == ast.dump(tree_b),
        }

    def summarize_for_llm(self, source: str, max_length: int = 300) -> str:
        """Create a concise summary of code for LLM consumption."""
        functions = self.get_functions(source)
        complexity = self.get_complexity(source)
        patterns = self.find_patterns(source)

        lines: list[str] = [
            f"Code summary ({complexity.lines_of_code} lines, complexity={complexity.cyclomatic_complexity}):",
        ]

        for func in functions[:5]:
            args_str = ", ".join(func.args)
            lines.append(f"  - {func.name}({args_str})")

        if patterns:
            pattern_types = set(p["type"] for p in patterns)
            lines.append(f"  Patterns: {', '.join(pattern_types)}")

        summary = "\n".join(lines)
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        return summary

    def get_source(self, obj: Any) -> str:
        """Get source code of a Python object."""
        try:
            return textwrap.dedent(inspect.getsource(obj))
        except (TypeError, OSError):
            return ""
