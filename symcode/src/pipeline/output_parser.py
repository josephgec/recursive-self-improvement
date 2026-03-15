"""Parse and validate code blocks from LLM responses."""

from __future__ import annotations

import ast
import re
from typing import Any

from src.utils.logging import get_logger

logger = get_logger("output_parser")

# Imports that should never appear in generated code
FORBIDDEN_IMPORTS = frozenset({
    "os",
    "sys",
    "subprocess",
    "shutil",
    "pathlib",
    "socket",
    "http",
    "urllib",
    "requests",
    "pickle",
    "shelve",
    "__import__",
    "importlib",
    "ctypes",
    "signal",
    "multiprocessing",
    "threading",
})


class CodeBlockParser:
    """Extract and validate Python code from LLM output."""

    # Match ```python ... ``` blocks
    _FENCED_RE = re.compile(
        r"```(?:python|py)?\s*\n(.*?)```",
        re.DOTALL,
    )

    def parse(self, text: str) -> str:
        """Extract Python code from text, preferring the last fenced block.

        Falls back to unfenced code detection if no fenced blocks found.
        """
        blocks = self.parse_all(text)
        if blocks:
            # Prefer the last block (often the final/corrected version)
            return blocks[-1].strip()

        # Try unfenced code detection
        return self._extract_unfenced(text)

    def parse_all(self, text: str) -> list[str]:
        """Extract all fenced Python code blocks from text."""
        matches = self._FENCED_RE.findall(text)
        return [m.strip() for m in matches if m.strip()]

    def _extract_unfenced(self, text: str) -> str:
        """Attempt to extract unfenced Python code from text.

        Looks for lines that appear to be Python code (imports, assignments,
        function definitions, etc.).
        """
        lines = text.split("\n")
        code_lines: list[str] = []
        in_code = False

        code_indicators = [
            "import ", "from ", "def ", "class ", "for ", "while ",
            "if ", "elif ", "else:", "try:", "except", "with ",
            "return ", "print(", "answer =", "answer=",
            "result =", "result=",
        ]

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if in_code:
                    code_lines.append(line)
                continue

            is_code = any(stripped.startswith(ind) for ind in code_indicators)
            # Also detect assignments and expressions
            if not is_code and re.match(r"^[a-zA-Z_]\w*\s*=", stripped):
                is_code = True
            if not is_code and re.match(r"^[a-zA-Z_]\w*\(", stripped):
                is_code = True
            # Indented lines after code are likely continuations
            if not is_code and in_code and line.startswith((" ", "\t")):
                is_code = True

            if is_code:
                in_code = True
                code_lines.append(line)
            else:
                if in_code and code_lines:
                    # Stop collecting when we hit non-code
                    break

        result = "\n".join(code_lines).strip()
        return result

    def validate_structure(self, code: str) -> tuple[bool, list[str]]:
        """Validate that code has correct structure.

        Checks:
        - Parses as valid Python AST
        - Contains an `answer` variable assignment
        - Does not import forbidden modules

        Returns:
            (is_valid, list_of_issues)
        """
        issues: list[str] = []

        if not code.strip():
            return False, ["Empty code"]

        # AST parse check
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, [f"Syntax error: {e}"]

        # Check for answer variable assignment
        has_answer = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "answer":
                        has_answer = True
            elif isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Name) and node.target.id == "answer":
                    has_answer = True

        if not has_answer:
            issues.append("No 'answer' variable assignment found")

        # Check for forbidden imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name.split(".")[0]
                    if mod in FORBIDDEN_IMPORTS:
                        issues.append(f"Forbidden import: {mod}")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    mod = node.module.split(".")[0]
                    if mod in FORBIDDEN_IMPORTS:
                        issues.append(f"Forbidden import: {mod}")

        is_valid = has_answer and len(issues) == 0
        return is_valid, issues
