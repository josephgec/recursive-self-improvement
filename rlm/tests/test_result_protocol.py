"""Tests for ResultProtocol."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.core.result_protocol import ResultProtocol, FinalSignal, RLMResult, SignalKind


class TestDetectFinal:
    def test_final_double_quotes(self):
        signal = ResultProtocol.detect_final('FINAL("hello world")')
        assert signal is not None
        assert signal.kind == SignalKind.FINAL
        assert signal.argument == "hello world"

    def test_final_single_quotes(self):
        signal = ResultProtocol.detect_final("FINAL('answer is 42')")
        assert signal is not None
        assert signal.kind == SignalKind.FINAL
        assert signal.argument == "answer is 42"

    def test_final_var_double_quotes(self):
        signal = ResultProtocol.detect_final('FINAL_VAR("result")')
        assert signal is not None
        assert signal.kind == SignalKind.FINAL_VAR
        assert signal.argument == "result"

    def test_final_var_single_quotes(self):
        signal = ResultProtocol.detect_final("FINAL_VAR('my_answer')")
        assert signal is not None
        assert signal.kind == SignalKind.FINAL_VAR
        assert signal.argument == "my_answer"

    def test_no_final(self):
        signal = ResultProtocol.detect_final("print('hello')")
        assert signal is None

    def test_final_in_multiline(self):
        code = """
results = search("test")
answer = results[0]
FINAL("the answer")
"""
        signal = ResultProtocol.detect_final(code)
        assert signal is not None
        assert signal.kind == SignalKind.FINAL
        assert signal.argument == "the answer"

    def test_final_var_preferred_over_final(self):
        # When both are present, FINAL_VAR is checked first (more specific)
        code = 'FINAL_VAR("x")\nFINAL("y")'
        signal = ResultProtocol.detect_final(code)
        assert signal.kind == SignalKind.FINAL_VAR

    def test_final_with_spaces(self):
        signal = ResultProtocol.detect_final('FINAL(  "spaced"  )')
        assert signal is not None
        assert signal.argument == "spaced"

    def test_final_empty_string(self):
        signal = ResultProtocol.detect_final('FINAL("")')
        assert signal is not None
        assert signal.argument == ""


class TestExtractResult:
    def test_extract_final(self):
        signal = FinalSignal(kind=SignalKind.FINAL, argument="hello")
        result = ResultProtocol.extract_result(signal, {})
        assert isinstance(result, RLMResult)
        assert result.value == "hello"
        assert result.source == "FINAL"

    def test_extract_final_var(self):
        signal = FinalSignal(kind=SignalKind.FINAL_VAR, argument="answer")
        repl = {"answer": 42}
        result = ResultProtocol.extract_result(signal, repl)
        assert result.value == 42
        assert result.source == "FINAL_VAR"
        assert result.raw_argument == "answer"

    def test_extract_final_var_undefined(self):
        signal = FinalSignal(kind=SignalKind.FINAL_VAR, argument="missing")
        result = ResultProtocol.extract_result(signal, {})
        assert "<undefined: missing>" in str(result.value)

    def test_result_str(self):
        result = RLMResult(value="test", source="FINAL", raw_argument="test")
        assert str(result) == "test"

    def test_result_str_numeric(self):
        result = RLMResult(value=42, source="FINAL_VAR", raw_argument="x")
        assert str(result) == "42"


class TestInjectProtocolFunctions:
    def test_inject(self):
        repl: dict = {}
        ResultProtocol.inject_protocol_functions(repl)

        assert "FINAL" in repl
        assert "FINAL_VAR" in repl
        assert callable(repl["FINAL"])
        assert callable(repl["FINAL_VAR"])

    def test_final_function_identity(self):
        repl: dict = {}
        ResultProtocol.inject_protocol_functions(repl)

        assert repl["FINAL"]("hello") == "hello"

    def test_final_var_function_identity(self):
        repl: dict = {}
        ResultProtocol.inject_protocol_functions(repl)

        assert repl["FINAL_VAR"]("varname") == "varname"
