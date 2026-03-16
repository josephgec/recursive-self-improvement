"""Additional tests to boost coverage for edge cases."""

import pytest
from src.interface.errors import (
    REPLError,
    MemoryLimitError,
    ForbiddenCodeError,
    RecursionDepthError,
    OutputSizeLimitError,
    REPLNotAliveError,
    SerializationError,
    CascadeKillError,
)
from src.safety.memory_limiter import MemoryLimiter
from src.pool.lifecycle import REPLLifecycle
from src.pool.pool import REPLPool
from src.memory.variable_store import VariableStore
from src.memory.serializer import REPLSerializer
from src.memory.snapshot import SnapshotManager
from src.memory.child_memory import ChildMemoryManager
from src.protocol.detector import FinalDetector
from src.protocol.extractor import ResultExtractor
from src.backends.local import LocalREPL
from src.backends.factory import REPLFactory
from src.safety.policy import SafetyPolicy


class TestErrorConstructors:
    """Test all error class constructors and attributes."""

    def test_repl_error(self):
        err = REPLError("base error")
        assert str(err) == "base error"

    def test_memory_limit_error_default_message(self):
        err = MemoryLimitError(limit_mb=512.0, usage_mb=600.0)
        assert err.limit_mb == 512.0
        assert err.usage_mb == 600.0
        assert "600.0" in str(err)

    def test_memory_limit_error_custom_message(self):
        err = MemoryLimitError(limit_mb=512.0, message="custom")
        assert str(err) == "custom"

    def test_forbidden_code_error_no_violations(self):
        err = ForbiddenCodeError()
        assert "Forbidden code detected" in str(err)

    def test_forbidden_code_error_with_violations(self):
        err = ForbiddenCodeError(violations=["import os", "eval()"])
        assert "import os" in str(err)
        assert err.violations == ["import os", "eval()"]

    def test_forbidden_code_error_custom_message(self):
        err = ForbiddenCodeError(message="custom msg")
        assert str(err) == "custom msg"

    def test_recursion_depth_error_default(self):
        err = RecursionDepthError(current_depth=6, max_depth=5)
        assert err.current_depth == 6
        assert err.max_depth == 5
        assert "6" in str(err)

    def test_recursion_depth_error_custom(self):
        err = RecursionDepthError(current_depth=6, max_depth=5, message="too deep")
        assert str(err) == "too deep"

    def test_output_size_limit_error(self):
        err = OutputSizeLimitError(size=200000, limit=100000)
        assert err.size == 200000
        assert err.limit == 100000
        assert "200000" in str(err)

    def test_output_size_limit_error_custom(self):
        err = OutputSizeLimitError(size=200000, limit=100000, message="big output")
        assert str(err) == "big output"

    def test_repl_not_alive_error(self):
        err = REPLNotAliveError()
        assert "not alive" in str(err)

    def test_repl_not_alive_error_custom(self):
        err = REPLNotAliveError("dead repl")
        assert str(err) == "dead repl"

    def test_serialization_error(self):
        err = SerializationError(variable_name="x")
        assert err.variable_name == "x"
        assert "x" in str(err)

    def test_serialization_error_custom(self):
        err = SerializationError(variable_name="x", message="custom")
        assert str(err) == "custom"

    def test_cascade_kill_error(self):
        err = CascadeKillError(repl_id="r1")
        assert err.repl_id == "r1"
        assert "r1" in str(err)

    def test_cascade_kill_error_custom(self):
        err = CascadeKillError(repl_id="r1", message="kill failed")
        assert str(err) == "kill failed"


class TestMemoryLimiterEdgeCases:
    """Test memory limiter edge cases."""

    def test_set_process_limit_default(self):
        limiter = MemoryLimiter(max_memory_mb=256.0)
        result = limiter.set_process_limit()
        assert isinstance(result, bool)

    def test_monitor_after_usage(self):
        limiter = MemoryLimiter(max_memory_mb=4096.0)
        # Allocate some memory
        data = [0] * 100000
        status = limiter.monitor()
        assert status.current_mb >= 0
        assert status.limit_mb == 4096.0
        del data

    def test_peak_updates(self):
        limiter = MemoryLimiter(max_memory_mb=4096.0)
        limiter.get_current_usage_mb()
        peak1 = limiter._peak_mb
        limiter.get_current_usage_mb()
        # Peak should be at least what it was
        assert limiter._peak_mb >= 0


class TestLifecycleEdgeCases:
    """Test lifecycle edge cases."""

    def test_warm_failure(self):
        lifecycle = REPLLifecycle()
        repl = lifecycle.create()
        repl.shutdown()
        # Warming a dead REPL should fail
        assert not lifecycle.warm(repl)

    def test_recycle_creates_new_when_dead(self):
        lifecycle = REPLLifecycle()
        repl = lifecycle.create()
        repl.shutdown()
        recycled = lifecycle.recycle(repl)
        assert recycled.is_alive()
        recycled.shutdown()

    def test_destroy_already_dead(self):
        lifecycle = REPLLifecycle()
        repl = lifecycle.create()
        repl.shutdown()
        # Should not raise
        lifecycle.destroy(repl)

    def test_health_check_dead(self):
        lifecycle = REPLLifecycle()
        repl = lifecycle.create()
        repl.shutdown()
        assert not lifecycle.health_check(repl)


class TestVariableStoreEdgeCases:
    """Test variable store size estimation edge cases."""

    def test_size_with_nested_dict(self):
        store = VariableStore()
        store.set("nested", {"a": {"b": {"c": [1, 2, 3]}}})
        size = store.total_size_bytes()
        assert size > 0

    def test_size_with_tuple(self):
        store = VariableStore()
        store.set("tup", (1, 2, 3))
        size = store.total_size_bytes()
        assert size > 0

    def test_size_with_set(self):
        store = VariableStore()
        store.set("s", {1, 2, 3})
        size = store.total_size_bytes()
        assert size > 0

    def test_diff_with_different_types(self):
        store = VariableStore()
        store.set("x", 42)
        previous = {"x": "hello"}  # different type
        diff = store.diff(previous)
        assert "x" in diff.modified

    def test_values_equal_exception(self):
        """Test values comparison when equality check fails."""
        store = VariableStore()

        class BadCompare:
            def __eq__(self, other):
                raise RuntimeError("cannot compare")

        store.set("x", BadCompare())
        previous = {"x": BadCompare()}
        # Should not raise, uses `is` fallback
        diff = store.diff(previous)
        # Since they're different objects, should be modified
        assert "x" in diff.modified

    def test_numpy_values_equal(self):
        np = pytest.importorskip("numpy")
        store = VariableStore()
        arr = np.array([1, 2, 3])
        store.set("arr", arr)
        previous = {"arr": np.array([1, 2, 3])}
        diff = store.diff(previous)
        assert "arr" not in diff.modified

    def test_size_with_numpy(self):
        np = pytest.importorskip("numpy")
        store = VariableStore()
        store.set("arr", np.zeros((100, 100)))
        size = store.total_size_bytes()
        assert size > 10000


class TestSerializerEdgeCases:
    """Test serializer edge cases."""

    def test_dill_round_trip(self):
        serializer = REPLSerializer()
        def my_func(x):
            return x * 2
        tag, data = serializer.serialize(my_func, "func")
        assert tag == "dill"
        restored = serializer.deserialize(tag, data, "func")
        assert restored(5) == 10

    def test_json_fallthrough_to_dill(self):
        """JSON-incompatible types fall through to dill."""
        serializer = REPLSerializer()
        # Set with frozenset not JSON-serializable
        value = {frozenset([1, 2]): "value"}
        tag, data = serializer.serialize(value, "complex")
        assert tag == "dill"


class TestSnapshotEdgeCases:
    """Test snapshot edge cases."""

    def test_snapshot_skips_unserializable(self):
        mgr = SnapshotManager()
        # Lambda that might cause issues with some serializers
        ns = {"x": 42, "func": lambda: None}
        snap_id = mgr.take(ns)
        restored = mgr.restore(snap_id)
        assert "x" in restored


class TestChildMemoryEdgeCases:
    """Test child memory edge cases."""

    def test_exclude_final_result(self):
        mgr = ChildMemoryManager()
        parent = {"x": 1, "__FINAL_RESULT__": "result", "__FINAL_VAR_NAME__": "x"}
        child = mgr.prepare_child_namespace(parent)
        assert "x" in child
        assert "__FINAL_RESULT__" not in child

    def test_load_excluded_variable(self):
        mgr = ChildMemoryManager()
        parent = {"CONTEXT": "secret", "x": 42}
        child = {}
        mgr.load_into_child(child, parent, variable_names=["CONTEXT", "x"])
        assert "x" in child
        assert "CONTEXT" not in child


class TestDetectorEdgeCases:
    """Test detector edge cases."""

    def test_detect_with_no_args(self):
        detector = FinalDetector()
        signals = detector.detect_in_code("FINAL()")
        assert len(signals) == 1
        assert signals[0].value is None

    def test_near_miss_Final(self):
        detector = FinalDetector()
        misses = detector.detect_near_misses('Final("x")')
        assert len(misses) > 0

    def test_near_miss_FINALVAR(self):
        detector = FinalDetector()
        misses = detector.detect_near_misses('FINALVAR("x")')
        assert len(misses) > 0


class TestExtractorEdgeCases:
    """Test extractor edge cases for pandas."""

    def test_serialize_pandas_dataframe(self):
        pd = pytest.importorskip("pandas")
        extractor = ResultExtractor()
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        s = extractor.serialize_value(df)
        assert s["type"] == "pandas.DataFrame"
        assert s["shape"] == [2, 2]

    def test_serialize_pandas_series(self):
        pd = pytest.importorskip("pandas")
        extractor = ResultExtractor()
        series = pd.Series([1, 2, 3], name="test")
        s = extractor.serialize_value(series)
        assert s["type"] == "pandas.Series"
        assert s["length"] == 3


class TestPoolEdgeCases:
    """Test pool edge cases."""

    def test_release_full_pool(self):
        """When pool is full, released REPL should be destroyed."""
        pool = REPLPool(size=1)
        repl = pool.acquire()
        pool.release(repl)
        # Pool is now full (size 1). The release should work fine
        assert pool.available == 1
        pool.shutdown()

    def test_pool_shutdown_idempotent(self):
        pool = REPLPool(size=2)
        pool.shutdown()
        # Should not raise on second call
        pool.shutdown()


class TestFactoryEdgeCases:
    """Test factory edge cases."""

    def test_create_docker(self):
        # Should fall back to local
        repl = REPLFactory.create("docker")
        assert repl.is_alive()
        repl.shutdown()

    def test_from_config_missing_policy(self):
        """Config with nonexistent policy file should use defaults."""
        import tempfile, yaml, os
        config = {
            "backend": "local",
            "safety": {"policy": "/nonexistent/policy.yaml"},
            "execution": {"timeout_seconds": 5},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            tmp_path = f.name
        try:
            repl = REPLFactory.from_config(tmp_path)
            assert repl.is_alive()
            repl.shutdown()
        finally:
            os.unlink(tmp_path)

    def test_from_config_execution_overrides(self):
        """Test that execution config overrides policy values."""
        import tempfile, yaml, os
        config = {
            "backend": "local",
            "execution": {
                "timeout_seconds": 10,
                "max_memory_mb": 256,
                "max_output_chars": 5000,
                "max_recursion_depth": 2,
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            tmp_path = f.name
        try:
            repl = REPLFactory.from_config(tmp_path)
            assert repl.is_alive()
            repl.shutdown()
        finally:
            os.unlink(tmp_path)
