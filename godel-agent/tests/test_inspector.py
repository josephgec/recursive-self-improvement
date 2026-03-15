"""Tests for CodeInspector."""

from __future__ import annotations

from src.modification.inspector import CodeInspector, FunctionAST, ComplexityReport


SAMPLE_CODE = '''
def add(a, b):
    """Add two numbers."""
    return a + b

def multiply(a, b):
    result = a * b
    return result

class Calculator:
    def compute(self, x, y, op):
        if op == "add":
            return x + y
        elif op == "sub":
            return x - y
        else:
            return 0
'''


SIMPLE_CODE = '''
def choose():
    return "cot"
'''


COMPLEX_CODE = '''
def process(data):
    results = []
    for item in data:
        if item > 0:
            for sub in item:
                if sub % 2 == 0:
                    while sub > 0:
                        results.append(sub)
                        sub -= 1
    return results
'''


class TestParse:
    def test_parse_valid_code(self) -> None:
        inspector = CodeInspector()
        tree = inspector.parse(SAMPLE_CODE)
        assert tree is not None

    def test_parse_empty_code(self) -> None:
        inspector = CodeInspector()
        tree = inspector.parse("")
        assert tree is not None

    def test_parse_invalid_code_raises(self) -> None:
        inspector = CodeInspector()
        import pytest
        with pytest.raises(SyntaxError):
            inspector.parse("def broken(:")


class TestGetFunctions:
    def test_find_all_functions(self) -> None:
        inspector = CodeInspector()
        funcs = inspector.get_functions(SAMPLE_CODE)
        names = [f.name for f in funcs]
        assert "add" in names
        assert "multiply" in names
        assert "compute" in names

    def test_function_args(self) -> None:
        inspector = CodeInspector()
        funcs = inspector.get_functions(SAMPLE_CODE)
        add_func = next(f for f in funcs if f.name == "add")
        assert "a" in add_func.args
        assert "b" in add_func.args


class TestComplexity:
    def test_simple_code_complexity(self) -> None:
        inspector = CodeInspector()
        report = inspector.get_complexity(SIMPLE_CODE)
        assert report.cyclomatic_complexity >= 1
        assert report.lines_of_code >= 2

    def test_complex_code_higher_complexity(self) -> None:
        inspector = CodeInspector()
        simple = inspector.get_complexity(SIMPLE_CODE)
        complex_ = inspector.get_complexity(COMPLEX_CODE)
        assert complex_.cyclomatic_complexity > simple.cyclomatic_complexity
        assert complex_.max_nesting > simple.max_nesting
        assert complex_.ast_node_count > simple.ast_node_count

    def test_nesting_depth(self) -> None:
        inspector = CodeInspector()
        report = inspector.get_complexity(COMPLEX_CODE)
        assert report.max_nesting >= 3  # for, if, for, if, while

    def test_lines_of_code_excludes_comments(self) -> None:
        inspector = CodeInspector()
        code = "# comment\nx = 1\n# another\ny = 2\n"
        report = inspector.get_complexity(code)
        assert report.lines_of_code == 2


class TestFindPatterns:
    def test_find_fixed_strategy(self) -> None:
        inspector = CodeInspector()
        patterns = inspector.find_patterns(SIMPLE_CODE)
        types = [p["type"] for p in patterns]
        assert "fixed_strategy" in types

    def test_find_hardcoded_constant(self) -> None:
        inspector = CodeInspector()
        code = "def f():\n    return x * 42\n"
        patterns = inspector.find_patterns(code)
        constants = [p for p in patterns if p["type"] == "hardcoded_constant"]
        assert len(constants) >= 1
        assert any(p["value"] == 42 for p in constants)


class TestCompareAST:
    def test_identical_code(self) -> None:
        inspector = CodeInspector()
        result = inspector.compare_ast(SIMPLE_CODE, SIMPLE_CODE)
        assert result["structurally_identical"] is True
        assert result["nodes_before"] == result["nodes_after"]

    def test_different_code(self) -> None:
        inspector = CodeInspector()
        code_a = "def f():\n    return 1\n"
        code_b = "def f():\n    x = 1\n    return x + 1\n"
        result = inspector.compare_ast(code_a, code_b)
        assert result["structurally_identical"] is False
        assert result["nodes_after"] > result["nodes_before"]


class TestSummarize:
    def test_summarize_produces_text(self) -> None:
        inspector = CodeInspector()
        summary = inspector.summarize_for_llm(SAMPLE_CODE)
        assert "add" in summary
        assert "lines" in summary.lower() or "complexity" in summary.lower()

    def test_summarize_respects_length(self) -> None:
        inspector = CodeInspector()
        summary = inspector.summarize_for_llm(SAMPLE_CODE, max_length=50)
        assert len(summary) <= 54  # 50 + "..."
