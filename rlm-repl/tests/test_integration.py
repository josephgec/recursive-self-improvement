"""Full stack integration tests: REPL + protocol + safety + pool."""

import pytest
from src.backends.local import LocalREPL
from src.backends.factory import REPLFactory
from src.safety.policy import SafetyPolicy
from src.protocol.final_functions import FinalProtocol
from src.protocol.detector import FinalDetector
from src.protocol.extractor import ResultExtractor
from src.protocol.aggregator import ResultAggregator
from src.pool.pool import REPLPool
from src.memory.variable_store import VariableStore
from src.memory.child_memory import ChildMemoryManager
from src.interface.errors import ForbiddenCodeError


class TestFullStackExecution:
    """Test complete execution pipeline."""

    def test_execute_and_extract_final(self):
        """Full pipeline: execute code, detect FINAL, extract result."""
        repl = LocalREPL()
        protocol = FinalProtocol()
        extractor = ResultExtractor()

        repl.execute("x = 42")
        repl.execute('FINAL(x)')

        result = extractor.extract_from_repl(repl._namespace)
        assert result is not None
        assert result.value == 42
        repl.shutdown()

    def test_execute_final_var_pipeline(self):
        """Full pipeline with FINAL_VAR."""
        repl = LocalREPL()
        extractor = ResultExtractor()

        repl.execute("answer = 'hello world'")
        repl.execute('FINAL_VAR("answer")')

        result = extractor.extract_from_repl(repl._namespace)
        assert result is not None
        assert result.value == "hello world"
        repl.shutdown()

    def test_detect_then_execute(self):
        """Detect FINAL in code before execution."""
        detector = FinalDetector()
        code = 'result = 42\nFINAL(result)'

        signals = detector.detect_in_code(code)
        assert len(signals) > 0
        assert signals[0].signal_type == "FINAL"

        repl = LocalREPL()
        repl.execute(code)

        protocol = FinalProtocol()
        final_result = protocol.check_for_result(repl._namespace)
        assert final_result is not None
        assert final_result.value == 42
        repl.shutdown()

    def test_aggregation_from_multiple_repls(self):
        """Aggregate results from multiple REPL executions."""
        agg = ResultAggregator()
        extractor = ResultExtractor()

        for i in range(3):
            repl = LocalREPL()
            repl.execute(f"result = {i * 10}")
            repl.execute("FINAL(result)")
            result = extractor.extract_from_repl(repl._namespace)
            if result:
                agg.collect(result)
            repl.shutdown()

        assert agg.count == 3
        concat = agg.concatenate()
        assert "0" in concat
        assert "10" in concat
        assert "20" in concat


class TestPoolIntegration:
    """Test pool with full execution pipeline."""

    def test_pool_acquire_execute_release(self):
        pool = REPLPool(size=2)
        repl = pool.acquire()
        result = repl.execute("x = 42")
        assert result.success
        pool.release(repl)
        pool.shutdown()

    def test_pool_with_final_protocol(self):
        pool = REPLPool(size=2)
        extractor = ResultExtractor()

        repl = pool.acquire()
        repl.execute("result = 100")
        repl.execute('FINAL(result)')

        final = extractor.extract_from_repl(repl._namespace)
        assert final is not None
        assert final.value == 100

        pool.release(repl)
        pool.shutdown()

    def test_pool_recycling_clears_state(self):
        pool = REPLPool(size=1)
        repl = pool.acquire()
        repl.execute("x = 42")
        pool.release(repl)

        # Acquire again - should be clean
        repl2 = pool.acquire()
        result = repl2.execute("try:\n    y = x\nexcept NameError:\n    y = -1")
        assert repl2.get_variable("y") == -1
        pool.release(repl2)
        pool.shutdown()

    def test_pool_metrics_after_operations(self):
        pool = REPLPool(size=2)
        r1 = pool.acquire()
        r2 = pool.acquire()
        pool.release(r1)
        pool.release(r2)

        metrics = pool.get_metrics()
        assert metrics.total_acquires == 2
        assert metrics.total_releases == 2
        assert metrics.available == 2
        pool.shutdown()


class TestSafetyWithProtocol:
    """Test that safety stack works with protocol."""

    def test_safe_final(self):
        repl = LocalREPL()
        result = repl.execute('FINAL("safe answer")')
        assert result.success
        repl.shutdown()

    def test_forbidden_before_final(self):
        repl = LocalREPL()
        with pytest.raises(ForbiddenCodeError):
            repl.execute('import os\nFINAL(os.getcwd())')
        repl.shutdown()

    def test_safe_computation_then_final(self):
        repl = LocalREPL()
        repl.execute("data = [1, 2, 3, 4, 5]")
        repl.execute("result = sum(data) / len(data)")
        repl.execute("FINAL(result)")

        protocol = FinalProtocol()
        final = protocol.check_for_result(repl._namespace)
        assert final is not None
        assert final.value == 3.0
        repl.shutdown()


class TestChildREPLIntegration:
    """Test child REPL with full stack."""

    def test_child_inherits_and_computes(self):
        parent = LocalREPL()
        parent.execute("shared_data = [1, 2, 3]")

        child = parent.spawn_child()
        child.execute("result = sum(shared_data)")
        assert child.get_variable("result") == 6

        child.shutdown()
        parent.shutdown()

    def test_child_final_independent(self):
        parent = LocalREPL()
        child = parent.spawn_child()

        child.execute('FINAL("child result")')
        protocol = FinalProtocol()
        child_result = protocol.check_for_result(child._namespace)
        parent_result = protocol.check_for_result(parent._namespace)

        assert child_result is not None
        assert child_result.value == "child result"
        assert parent_result is None

        child.shutdown()
        parent.shutdown()

    def test_snapshot_restore_with_final(self):
        repl = LocalREPL()
        repl.execute("x = 1")
        snap = repl.snapshot()

        repl.execute("x = 2")
        repl.execute("FINAL(x)")

        protocol = FinalProtocol()
        assert protocol.check_for_result(repl._namespace).value == 2

        repl.restore(snap)
        # After restore, FINAL result should be cleared
        assert protocol.check_for_result(repl._namespace) is None
        assert repl.get_variable("x") == 1
        repl.shutdown()


class TestFactoryIntegration:
    """Test REPLFactory integration."""

    def test_create_local(self):
        repl = REPLFactory.create("local")
        result = repl.execute("x = 42")
        assert result.success
        repl.shutdown()

    def test_create_with_policy(self):
        policy = SafetyPolicy(timeout_seconds=5)
        repl = REPLFactory.create("local", policy=policy)
        result = repl.execute("x = 1")
        assert result.success
        repl.shutdown()

    def test_create_unknown_backend(self):
        with pytest.raises(ValueError):
            REPLFactory.create("unknown")

    def test_auto_detect(self):
        repl = REPLFactory.auto_detect()
        result = repl.execute("x = 1")
        assert result.success
        repl.shutdown()

    def test_from_config(self):
        import os
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "dev.yaml",
        )
        repl = REPLFactory.from_config(config_path)
        result = repl.execute("x = 1")
        assert result.success
        repl.shutdown()


class TestChildMemoryIntegration:
    """Test child memory manager."""

    def test_prepare_child_namespace(self):
        mgr = ChildMemoryManager()
        parent_ns = {
            "x": 42,
            "y": "hello",
            "CONTEXT": "secret",
            "__builtins__": {},
            "__name__": "__main__",
        }
        child_ns = mgr.prepare_child_namespace(parent_ns)
        assert "x" in child_ns
        assert "y" in child_ns
        assert "CONTEXT" not in child_ns
        assert "__builtins__" not in child_ns

    def test_load_into_child(self):
        mgr = ChildMemoryManager()
        parent_ns = {"x": 42, "y": "hello", "CONTEXT": "secret"}
        child_ns = {}
        mgr.load_into_child(child_ns, parent_ns, variable_names=["x"])
        assert "x" in child_ns
        assert "y" not in child_ns
        assert "CONTEXT" not in child_ns

    def test_load_all_into_child(self):
        mgr = ChildMemoryManager()
        parent_ns = {"x": 42, "y": "hello", "CONTEXT": "secret"}
        child_ns = {}
        mgr.load_into_child(child_ns, parent_ns)
        assert "x" in child_ns
        assert "y" in child_ns
        assert "CONTEXT" not in child_ns

    def test_exclude_functions_option(self):
        mgr = ChildMemoryManager()

        def my_func():
            return 42

        parent_ns = {"x": 42, "func": my_func}
        child_ns = mgr.prepare_child_namespace(parent_ns, include_functions=False)
        assert "x" in child_ns
        assert "func" not in child_ns


class TestVariableStoreIntegration:
    """Test variable store with REPL."""

    def test_store_tracks_repl_variables(self):
        repl = LocalREPL()
        repl.execute("a = 1")
        repl.execute("b = 2")
        repl.execute("c = a + b")

        variables = repl.list_variables()
        assert "a" in variables
        assert "b" in variables
        assert "c" in variables
        repl.shutdown()

    def test_execution_result_type(self):
        from src.interface.types import ExecutionResult
        result = ExecutionResult(stdout="hello", execution_time_ms=10.5)
        assert result.success
        assert result.stdout == "hello"

        result2 = ExecutionResult(error="bad", error_type="RuntimeError")
        assert not result2.success

        result3 = ExecutionResult(killed=True, kill_reason="timeout")
        assert not result3.success
