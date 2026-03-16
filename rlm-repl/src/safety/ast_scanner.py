"""AST-based static analysis scanner for detecting forbidden code patterns."""

import ast
from dataclasses import dataclass, field
from typing import List, Optional, Set

from src.safety.policy import SafetyPolicy


@dataclass
class Violation:
    """A single code violation found by the AST scanner.

    Attributes:
        category: Category of the violation (e.g., 'forbidden_import').
        message: Human-readable description.
        line: Line number where the violation was found.
        col: Column offset.
        severity: Severity level ('error', 'warning').
    """

    category: str
    message: str
    line: int = 0
    col: int = 0
    severity: str = "error"

    def __str__(self) -> str:
        return f"[{self.severity}] {self.category} at line {self.line}: {self.message}"


@dataclass
class ScanResult:
    """Result of scanning code with the AST scanner.

    Attributes:
        safe: Whether the code passed all checks.
        violations: List of violations found.
        code: The original code that was scanned.
    """

    safe: bool
    violations: List[Violation] = field(default_factory=list)
    code: str = ""

    @property
    def error_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "warning")


class ASTScanner:
    """Scans Python code for forbidden patterns using AST analysis.

    Checks performed:
    1. Forbidden imports (import os, from os import ...)
    2. Forbidden builtins (eval, exec, compile, ...)
    3. Dunder access (__class__, __bases__, etc.)
    4. Star imports (from X import *)
    5. Obfuscation detection (chr() + exec patterns)
    6. Infinite loop detection (while True without break)
    7. Excessive nesting depth
    """

    def __init__(self, policy: Optional[SafetyPolicy] = None):
        self.policy = policy or SafetyPolicy()

    def scan(self, code: str) -> ScanResult:
        """Scan code for violations.

        Args:
            code: Python source code to scan.

        Returns:
            ScanResult indicating whether the code is safe.
        """
        violations: List[Violation] = []

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            violations.append(Violation(
                category="syntax_error",
                message=f"Syntax error: {e}",
                line=e.lineno or 0,
                col=e.offset or 0,
            ))
            return ScanResult(safe=False, violations=violations, code=code)

        violations.extend(self._check_forbidden_imports(tree))
        violations.extend(self._check_forbidden_builtins(tree))
        violations.extend(self._check_dunder_access(tree))
        violations.extend(self._check_star_imports(tree))

        if self.policy.detect_obfuscation:
            violations.extend(self._check_obfuscation(tree))

        if self.policy.detect_infinite_loops:
            violations.extend(self._check_infinite_loops(tree))

        violations.extend(self._check_nesting_depth(tree))

        safe = all(v.severity != "error" for v in violations)
        return ScanResult(safe=safe, violations=violations, code=code)

    def _check_forbidden_imports(self, tree: ast.AST) -> List[Violation]:
        """Check for forbidden import statements."""
        violations = []
        forbidden = set(self.policy.forbidden_imports)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_root = alias.name.split(".")[0]
                    if module_root in forbidden or alias.name in forbidden:
                        violations.append(Violation(
                            category="forbidden_import",
                            message=f"Import of '{alias.name}' is forbidden",
                            line=node.lineno,
                            col=node.col_offset,
                        ))
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_root = node.module.split(".")[0]
                    if module_root in forbidden or node.module in forbidden:
                        violations.append(Violation(
                            category="forbidden_import",
                            message=f"Import from '{node.module}' is forbidden",
                            line=node.lineno,
                            col=node.col_offset,
                        ))

        return violations

    def _check_forbidden_builtins(self, tree: ast.AST) -> List[Violation]:
        """Check for forbidden builtin function calls."""
        violations = []
        forbidden = set(self.policy.forbidden_builtins)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in forbidden:
                    violations.append(Violation(
                        category="forbidden_builtin",
                        message=f"Use of builtin '{func.id}' is forbidden",
                        line=node.lineno,
                        col=node.col_offset,
                    ))

        return violations

    def _check_dunder_access(self, tree: ast.AST) -> List[Violation]:
        """Check for dunder attribute access and dunder name references."""
        if self.policy.allow_dunder_access:
            return []

        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if node.attr.startswith("__") and node.attr.endswith("__"):
                    violations.append(Violation(
                        category="dunder_access",
                        message=f"Access to dunder attribute '{node.attr}' is forbidden",
                        line=node.lineno,
                        col=node.col_offset,
                    ))
            elif isinstance(node, ast.Name):
                if (node.id.startswith("__") and node.id.endswith("__")
                        and node.id not in ("__name__", "__doc__")):
                    violations.append(Violation(
                        category="dunder_access",
                        message=f"Reference to dunder name '{node.id}' is forbidden",
                        line=node.lineno,
                        col=node.col_offset,
                    ))

        return violations

    def _check_star_imports(self, tree: ast.AST) -> List[Violation]:
        """Check for star imports (from X import *)."""
        if self.policy.allow_star_imports:
            return []

        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.names and any(alias.name == "*" for alias in node.names):
                    violations.append(Violation(
                        category="star_import",
                        message=f"Star import from '{node.module}' is forbidden",
                        line=node.lineno,
                        col=node.col_offset,
                    ))

        return violations

    def _check_obfuscation(self, tree: ast.AST) -> List[Violation]:
        """Detect obfuscation patterns like chr() + exec."""
        violations = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # Detect exec/eval with chr() or join patterns
                if isinstance(func, ast.Name) and func.id in ("exec", "eval"):
                    # Check if argument uses chr()
                    for arg in node.args:
                        if self._contains_chr_pattern(arg):
                            violations.append(Violation(
                                category="obfuscation",
                                message="Obfuscation detected: chr() used with exec/eval",
                                line=node.lineno,
                                col=node.col_offset,
                            ))
                            break

                # Detect standalone chr + join patterns building code
                if isinstance(func, ast.Attribute) and func.attr == "join":
                    # "".join(chr(x) for x in [...])
                    if self._args_contain_chr(node):
                        violations.append(Violation(
                            category="obfuscation",
                            message="Obfuscation detected: string building with chr()",
                            line=node.lineno,
                            col=node.col_offset,
                            severity="warning",
                        ))

        return violations

    def _contains_chr_pattern(self, node: ast.AST) -> bool:
        """Check if an AST node contains chr() calls."""
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name) and child.func.id == "chr":
                    return True
        return False

    def _args_contain_chr(self, node: ast.Call) -> bool:
        """Check if a Call node's arguments contain chr() patterns."""
        for arg in node.args:
            if self._contains_chr_pattern(arg):
                return True
        return False

    def _check_infinite_loops(self, tree: ast.AST) -> List[Violation]:
        """Detect simple infinite loops (while True without break)."""
        violations = []

        for node in ast.walk(tree):
            if isinstance(node, ast.While):
                # Check for while True
                if isinstance(node.test, ast.Constant) and node.test.value is True:
                    if not self._has_break(node):
                        violations.append(Violation(
                            category="infinite_loop",
                            message="Potential infinite loop: while True without break",
                            line=node.lineno,
                            col=node.col_offset,
                            severity="warning",
                        ))

        return violations

    def _has_break(self, node: ast.While) -> bool:
        """Check if a while loop contains a break statement."""
        for child in ast.walk(node):
            if isinstance(child, ast.Break):
                return True
        return False

    def _check_nesting_depth(self, tree: ast.AST) -> List[Violation]:
        """Check for excessive nesting depth."""
        violations = []
        max_depth = self._get_max_depth(tree)
        if max_depth > self.policy.max_nesting_depth:
            violations.append(Violation(
                category="excessive_nesting",
                message=f"Nesting depth {max_depth} exceeds limit {self.policy.max_nesting_depth}",
                line=1,
                severity="error",
            ))
        return violations

    def _get_max_depth(self, node: ast.AST, current: int = 0) -> int:
        """Recursively compute the maximum nesting depth."""
        nesting_types = (
            ast.If, ast.For, ast.While, ast.With,
            ast.Try, ast.FunctionDef, ast.AsyncFunctionDef,
            ast.ClassDef,
        )
        max_d = current
        for child in ast.iter_child_nodes(node):
            if isinstance(child, nesting_types):
                depth = self._get_max_depth(child, current + 1)
            else:
                depth = self._get_max_depth(child, current)
            max_d = max(max_d, depth)
        return max_d
