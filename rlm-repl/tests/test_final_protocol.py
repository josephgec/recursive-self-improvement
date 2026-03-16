"""Tests for the FINAL protocol."""

import pytest
from src.protocol.final_functions import FinalProtocol, FinalResult
from src.protocol.detector import FinalDetector
from src.protocol.extractor import ResultExtractor
from src.protocol.aggregator import ResultAggregator
from src.protocol.types import FinalSignal
from src.backends.local import LocalREPL
from src.safety.policy import SafetyPolicy


class TestFinalProtocol:
    """Test FINAL/FINAL_VAR protocol functions."""

    def setup_method(self):
        self.protocol = FinalProtocol()

    def test_inject_defines_functions(self):
        ns = {}
        self.protocol.inject(ns)
        assert "FINAL" in ns
        assert "FINAL_VAR" in ns

    def test_final_sets_result(self):
        ns = {}
        self.protocol.inject(ns)
        ns["FINAL"]("the answer")
        result = self.protocol.check_for_result(ns)
        assert result is not None
        assert result.value == "the answer"
        assert result.source == "FINAL"

    def test_final_var_sets_result(self):
        ns = {}
        self.protocol.inject(ns)
        ns["result"] = 42
        ns["FINAL_VAR"]("result")
        result = self.protocol.check_for_result(ns)
        assert result is not None
        assert result.value == 42
        assert result.source == "FINAL_VAR"
        assert result.variable_name == "result"

    def test_final_var_missing_variable(self):
        ns = {}
        self.protocol.inject(ns)
        ns["FINAL_VAR"]("nonexistent")
        result = self.protocol.check_for_result(ns)
        assert result is not None
        assert result.value is None
        assert result.variable_name == "nonexistent"

    def test_no_result(self):
        ns = {}
        self.protocol.inject(ns)
        result = self.protocol.check_for_result(ns)
        assert result is None

    def test_reset(self):
        ns = {}
        self.protocol.inject(ns)
        ns["FINAL"]("value")
        assert self.protocol.check_for_result(ns) is not None
        self.protocol.reset(ns)
        assert self.protocol.check_for_result(ns) is None

    def test_injectable_code(self):
        code = self.protocol.get_injectable_code()
        assert "def FINAL" in code
        assert "def FINAL_VAR" in code

    def test_has_result_property(self):
        result = FinalResult(value="hello", source="FINAL")
        assert result.has_result
        result2 = FinalResult(value=None, source="")
        assert not result2.has_result


class TestFinalProtocolInREPL:
    """Test FINAL protocol integrated with LocalREPL."""

    def setup_method(self):
        self.repl = LocalREPL(policy=SafetyPolicy())
        self.protocol = FinalProtocol()

    def teardown_method(self):
        if self.repl.is_alive():
            self.repl.shutdown()

    def test_final_in_execution(self):
        self.repl.execute('FINAL("the answer is 42")')
        result = self.protocol.check_for_result(self.repl._namespace)
        assert result is not None
        assert result.value == "the answer is 42"

    def test_final_var_in_execution(self):
        self.repl.execute("result = 42")
        self.repl.execute('FINAL_VAR("result")')
        result = self.protocol.check_for_result(self.repl._namespace)
        assert result is not None
        assert result.value == 42

    def test_final_with_complex_value(self):
        self.repl.execute('FINAL({"key": "value", "num": 42})')
        result = self.protocol.check_for_result(self.repl._namespace)
        assert result.value == {"key": "value", "num": 42}


class TestFinalDetector:
    """Test AST-based FINAL detection."""

    def setup_method(self):
        self.detector = FinalDetector()

    def test_detect_final(self):
        signals = self.detector.detect_in_code('FINAL("answer")')
        assert len(signals) == 1
        assert signals[0].signal_type == "FINAL"
        assert signals[0].value == "answer"

    def test_detect_final_var(self):
        signals = self.detector.detect_in_code('FINAL_VAR("result")')
        assert len(signals) == 1
        assert signals[0].signal_type == "FINAL_VAR"
        assert signals[0].value == "result"

    def test_detect_no_final(self):
        signals = self.detector.detect_in_code("x = 42")
        assert len(signals) == 0

    def test_detect_near_misses(self):
        misses = self.detector.detect_near_misses('final("answer")')
        assert len(misses) > 0

    def test_detect_near_miss_result(self):
        misses = self.detector.detect_near_misses('result("answer")')
        assert len(misses) > 0

    def test_no_near_miss_for_correct(self):
        misses = self.detector.detect_near_misses('FINAL("answer")')
        assert len(misses) == 0

    def test_detect_multiple(self):
        code = 'FINAL("one")\nFINAL("two")'
        signals = self.detector.detect_in_code(code)
        assert len(signals) == 2

    def test_detect_syntax_error(self):
        signals = self.detector.detect_in_code("def :")
        assert len(signals) == 0

    def test_final_signal_properties(self):
        sig = FinalSignal(signal_type="FINAL", value="test")
        assert sig.is_final
        assert not sig.is_final_var

        sig2 = FinalSignal(signal_type="FINAL_VAR", value="x")
        assert not sig2.is_final
        assert sig2.is_final_var

    def test_detect_final_with_variable_arg(self):
        signals = self.detector.detect_in_code("FINAL(my_var)")
        assert len(signals) == 1
        assert signals[0].value == "my_var"


class TestResultExtractor:
    """Test result extraction."""

    def setup_method(self):
        self.extractor = ResultExtractor()

    def test_extract_final(self):
        protocol = FinalProtocol()
        ns = {}
        protocol.inject(ns)
        ns["FINAL"]("answer")
        result = self.extractor.extract_from_repl(ns)
        assert result is not None
        assert result.value == "answer"

    def test_serialize_int(self):
        s = self.extractor.serialize_value(42)
        assert s["type"] == "int"
        assert s["value"] == 42

    def test_serialize_str(self):
        s = self.extractor.serialize_value("hello")
        assert s["type"] == "str"
        assert s["value"] == "hello"

    def test_serialize_list(self):
        s = self.extractor.serialize_value([1, 2, 3])
        assert s["type"] == "list"
        assert s["length"] == 3

    def test_serialize_dict(self):
        s = self.extractor.serialize_value({"a": 1})
        assert s["type"] == "dict"
        assert s["length"] == 1

    def test_serialize_none(self):
        s = self.extractor.serialize_value(None)
        assert s["type"] == "NoneType"

    def test_serialize_bool(self):
        s = self.extractor.serialize_value(True)
        assert s["type"] == "bool"
        assert s["value"] is True

    def test_serialize_float(self):
        s = self.extractor.serialize_value(3.14)
        assert s["type"] == "float"

    def test_serialize_numpy(self):
        pytest.importorskip("numpy")
        import numpy as np
        s = self.extractor.serialize_value(np.array([1, 2, 3]))
        assert s["type"] == "numpy.ndarray"
        assert s["shape"] == [3]

    def test_serialize_fallback(self):
        s = self.extractor.serialize_value(object())
        assert s["type"] == "object"

    def test_extract_no_result(self):
        result = self.extractor.extract_from_repl({})
        assert result is None


class TestResultAggregator:
    """Test result aggregation."""

    def setup_method(self):
        self.agg = ResultAggregator()

    def test_collect(self):
        self.agg.collect(FinalResult(value="a", source="FINAL"))
        assert self.agg.count == 1

    def test_concatenate(self):
        self.agg.collect(FinalResult(value="hello", source="FINAL"))
        self.agg.collect(FinalResult(value="world", source="FINAL"))
        result = self.agg.concatenate()
        assert "hello" in result
        assert "world" in result

    def test_vote(self):
        self.agg.collect(FinalResult(value=1, source="FINAL"))
        self.agg.collect(FinalResult(value=2, source="FINAL"))
        self.agg.collect(FinalResult(value=1, source="FINAL"))
        assert self.agg.vote() == 1

    def test_vote_empty(self):
        assert self.agg.vote() is None

    def test_merge_structured(self):
        self.agg.collect(FinalResult(value={"a": 1}, source="FINAL"))
        self.agg.collect(FinalResult(value={"b": 2}, source="FINAL"))
        merged = self.agg.merge_structured()
        assert merged == {"a": 1, "b": 2}

    def test_merge_non_dict_ignored(self):
        self.agg.collect(FinalResult(value="not a dict", source="FINAL"))
        self.agg.collect(FinalResult(value={"a": 1}, source="FINAL"))
        merged = self.agg.merge_structured()
        assert merged == {"a": 1}

    def test_inject_helpers(self):
        ns = {}
        self.agg.inject_helpers(ns)
        assert "collect_result" in ns
        ns["collect_result"]("test_value")
        assert self.agg.count == 1

    def test_clear(self):
        self.agg.collect(FinalResult(value=1, source="FINAL"))
        self.agg.clear()
        assert self.agg.count == 0

    def test_results_property(self):
        self.agg.collect(FinalResult(value=1, source="FINAL"))
        results = self.agg.results
        assert len(results) == 1
        assert results[0].value == 1
