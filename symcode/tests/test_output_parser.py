"""Tests for the output parser."""

from __future__ import annotations

import pytest

from src.pipeline.output_parser import CodeBlockParser


class TestCodeBlockParser:
    """Test extraction and validation of code blocks."""

    def setup_method(self):
        self.parser = CodeBlockParser()

    # ── parse: fenced blocks ────────────────────────────────────────

    def test_single_fenced_block(self):
        text = 'Here is the solution:\n```python\nanswer = 42\nprint(answer)\n```'
        code = self.parser.parse(text)
        assert "answer = 42" in code
        assert "```" not in code

    def test_multiple_fenced_blocks_prefers_last(self):
        text = (
            "First attempt:\n```python\nanswer = 1\n```\n"
            "Actually, correction:\n```python\nanswer = 2\n```"
        )
        code = self.parser.parse(text)
        assert "answer = 2" in code

    def test_fenced_block_without_language(self):
        text = "Solution:\n```\nimport sympy\nanswer = 5\n```"
        code = self.parser.parse(text)
        assert "answer = 5" in code

    def test_parse_all_returns_multiple(self):
        text = "```python\na = 1\n```\n```python\nb = 2\n```"
        blocks = self.parser.parse_all(text)
        assert len(blocks) == 2
        assert "a = 1" in blocks[0]
        assert "b = 2" in blocks[1]

    # ── parse: unfenced code ────────────────────────────────────────

    def test_unfenced_code_detection(self):
        text = (
            "Here is my solution:\n\n"
            "from sympy import *\n"
            "x = symbols('x')\n"
            "answer = solve(x**2 - 4, x)\n"
            "print(answer)\n\n"
            "That gives us the roots."
        )
        code = self.parser.parse(text)
        assert "from sympy import *" in code
        assert "answer = solve" in code

    def test_no_code_returns_empty(self):
        text = "I think the answer is 42. No code needed."
        code = self.parser.parse(text)
        # Should return empty or very minimal
        assert "import" not in code.lower() or code.strip() == ""

    # ── validate_structure ──────────────────────────────────────────

    def test_valid_code_with_answer(self):
        code = "from sympy import *\nx = symbols('x')\nanswer = 42\n"
        is_valid, issues = self.parser.validate_structure(code)
        assert is_valid is True
        assert issues == []

    def test_invalid_no_answer_variable(self):
        code = "from sympy import *\nresult = 42\nprint(result)\n"
        is_valid, issues = self.parser.validate_structure(code)
        assert is_valid is False
        assert any("answer" in issue.lower() for issue in issues)

    def test_invalid_forbidden_import_os(self):
        code = "import os\nanswer = os.getcwd()\n"
        is_valid, issues = self.parser.validate_structure(code)
        assert is_valid is False
        assert any("forbidden" in issue.lower() or "os" in issue.lower() for issue in issues)

    def test_invalid_forbidden_import_subprocess(self):
        code = "import subprocess\nanswer = subprocess.run(['ls'])\n"
        is_valid, issues = self.parser.validate_structure(code)
        assert is_valid is False

    def test_syntax_error(self):
        code = "def foo(\n  answer = 1\n"
        is_valid, issues = self.parser.validate_structure(code)
        assert is_valid is False
        assert any("syntax" in issue.lower() for issue in issues)

    def test_empty_code(self):
        is_valid, issues = self.parser.validate_structure("")
        assert is_valid is False
        assert any("empty" in issue.lower() for issue in issues)

    def test_valid_with_from_import(self):
        code = "from sympy import symbols, solve\nanswer = solve(symbols('x') - 1)\n"
        is_valid, issues = self.parser.validate_structure(code)
        assert is_valid is True
