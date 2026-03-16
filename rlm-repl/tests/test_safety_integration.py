"""Integration tests for the 6-layer safety stack."""

import os
import pytest
from src.backends.local import LocalREPL
from src.safety.policy import SafetyPolicy
from src.safety.ast_scanner import ASTScanner
from src.safety.output_limiter import OutputLimiter
from src.safety.depth_limiter import DepthLimiter
from src.safety.resource_monitor import ResourceMonitor
from src.interface.errors import (
    ForbiddenCodeError,
    RecursionDepthError,
)


class TestAllSixLayers:
    """Test all 6 safety layers working together."""

    def test_layer1_ast_scanning(self):
        """Layer 1: AST scanner blocks forbidden code."""
        repl = LocalREPL()
        with pytest.raises(ForbiddenCodeError):
            repl.execute("import os")
        repl.shutdown()

    def test_layer2_timeout(self):
        """Layer 2: Timeout kills long-running code."""
        policy = SafetyPolicy(timeout_seconds=0.5)
        repl = LocalREPL(policy=policy)
        result = repl.execute("while True: pass")
        assert result.killed
        assert result.kill_reason == "timeout"
        repl.shutdown()

    def test_layer3_memory_monitoring(self):
        """Layer 3: Memory is monitored during execution."""
        repl = LocalREPL()
        result = repl.execute("x = list(range(1000))")
        assert result.memory_peak_mb >= 0
        repl.shutdown()

    def test_layer4_output_limiting(self):
        """Layer 4: Output is truncated when too large."""
        policy = SafetyPolicy(max_output_chars=100)
        repl = LocalREPL(policy=policy)
        result = repl.execute("print('x' * 500)")
        assert len(result.stdout) < 500
        assert "TRUNCATED" in result.stdout
        repl.shutdown()

    def test_layer5_depth_limiting(self):
        """Layer 5: Spawn depth is limited."""
        policy = SafetyPolicy(max_spawn_depth=1)
        repl = LocalREPL(policy=policy)
        child = repl.spawn_child()
        with pytest.raises(RecursionDepthError):
            child.spawn_child()
        child.shutdown()
        repl.shutdown()

    def test_layer6_cascade_killing(self):
        """Layer 6: Shutting down parent kills children."""
        repl = LocalREPL()
        child = repl.spawn_child()
        grandchild = child.spawn_child()
        assert grandchild.is_alive()
        repl.shutdown()
        assert not child.is_alive()
        assert not grandchild.is_alive()


class TestMaliciousSamples:
    """Test malicious code samples are blocked."""

    def setup_method(self):
        self.policy = SafetyPolicy()
        self.scanner = ASTScanner(self.policy)
        self.samples_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "malicious_samples",
        )

    def _load_sample(self, name):
        path = os.path.join(self.samples_dir, name)
        with open(path, "r") as f:
            return f.read()

    def test_escape_dunder(self):
        code = self._load_sample("escape_dunder.py")
        result = self.scanner.scan(code)
        assert not result.safe

    def test_escape_import(self):
        code = self._load_sample("escape_import.py")
        result = self.scanner.scan(code)
        assert not result.safe

    def test_escape_eval(self):
        code = self._load_sample("escape_eval.py")
        result = self.scanner.scan(code)
        assert not result.safe

    def test_escape_getattr(self):
        code = self._load_sample("escape_getattr.py")
        result = self.scanner.scan(code)
        assert not result.safe

    def test_fork_bomb(self):
        code = self._load_sample("fork_bomb.py")
        result = self.scanner.scan(code)
        assert not result.safe

    def test_network_exfil(self):
        code = self._load_sample("network_exfil.py")
        result = self.scanner.scan(code)
        assert not result.safe

    def test_file_write(self):
        code = self._load_sample("file_write.py")
        result = self.scanner.scan(code)
        assert not result.safe

    def test_infinite_loop_sample(self):
        code = self._load_sample("infinite_loop.py")
        result = self.scanner.scan(code)
        has_warning = any(v.category == "infinite_loop" for v in result.violations)
        assert has_warning

    def test_memory_bomb_sample(self):
        code = self._load_sample("memory_bomb.py")
        result = self.scanner.scan(code)
        has_warning = any(v.category == "infinite_loop" for v in result.violations)
        assert has_warning

    def test_malicious_samples_blocked_in_repl(self):
        """Test that all malicious samples are blocked when executed in LocalREPL."""
        for sample in [
            "escape_dunder.py", "escape_import.py", "escape_eval.py",
            "escape_getattr.py", "fork_bomb.py", "network_exfil.py",
            "file_write.py",
        ]:
            code = self._load_sample(sample)
            repl = LocalREPL(policy=self.policy)
            blocked = False
            try:
                result = repl.execute(code)
                if result.error or result.killed:
                    blocked = True
            except (ForbiddenCodeError, Exception):
                blocked = True
            finally:
                repl.shutdown()
            assert blocked, f"Sample {sample} was NOT blocked!"


class TestOutputLimiter:
    """Test output limiter integration."""

    def test_truncate_long_output(self):
        limiter = OutputLimiter(max_chars=50)
        output = "x" * 200
        truncated = limiter.truncate(output)
        assert len(truncated) < 200
        assert "TRUNCATED" in truncated

    def test_short_output_unchanged(self):
        limiter = OutputLimiter(max_chars=100)
        output = "hello"
        assert limiter.truncate(output) == output

    def test_check_within_limits(self):
        limiter = OutputLimiter(max_chars=100)
        assert limiter.check("hello")
        assert not limiter.check("x" * 200)

    def test_custom_limit(self):
        limiter = OutputLimiter(max_chars=100)
        assert limiter.check("x" * 50, max_chars=60)
        assert not limiter.check("x" * 50, max_chars=10)

    def test_truncate_custom_limit(self):
        limiter = OutputLimiter(max_chars=100)
        result = limiter.truncate("x" * 50, max_chars=10)
        assert result.startswith("x" * 10)
        assert "TRUNCATED" in result


class TestDepthLimiter:
    """Test depth limiter."""

    def test_can_spawn_at_start(self):
        limiter = DepthLimiter(max_depth=3)
        limiter._depths["root"] = 0
        assert limiter.can_spawn("root")

    def test_cannot_spawn_at_max(self):
        limiter = DepthLimiter(max_depth=2)
        limiter._depths["deep"] = 2
        assert not limiter.can_spawn("deep")

    def test_register_spawn(self):
        limiter = DepthLimiter(max_depth=5)
        depth = limiter.register_spawn("root", "child1")
        assert depth == 1

    def test_register_completion(self):
        limiter = DepthLimiter(max_depth=5)
        limiter.register_spawn("root", "child1")
        limiter.register_completion("child1")
        assert not limiter._active.get("child1", True)

    def test_get_depth(self):
        limiter = DepthLimiter(max_depth=5)
        limiter.register_spawn("root", "child1")
        limiter.register_spawn("child1", "child2")
        assert limiter.get_depth("child2") == 2

    def test_get_status(self):
        limiter = DepthLimiter(max_depth=5)
        limiter.register_spawn("root", "child1")
        status = limiter.get_status("root")
        assert status.current_depth == 0
        assert status.max_depth == 5

    def test_get_status_overall(self):
        limiter = DepthLimiter(max_depth=5)
        limiter.register_spawn("root", "child1")
        status = limiter.get_status()
        assert status.active_spawns > 0


class TestResourceMonitor:
    """Test resource monitoring."""

    def test_register_and_status(self):
        monitor = ResourceMonitor()
        monitor.register("repl1")
        status = monitor.get_status("repl1")
        assert status is not None
        assert status.is_alive
        assert status.repl_id == "repl1"

    def test_unregister(self):
        monitor = ResourceMonitor()
        monitor.register("repl1")
        monitor.unregister("repl1")
        status = monitor.get_status("repl1")
        assert not status.is_alive

    def test_record_execution(self):
        monitor = ResourceMonitor()
        monitor.register("repl1")
        monitor.record_execution("repl1", memory_mb=50.0)
        status = monitor.get_status("repl1")
        assert status.execution_count == 1
        assert status.memory_mb == 50.0

    def test_get_all_status(self):
        monitor = ResourceMonitor()
        monitor.register("repl1")
        monitor.register("repl2")
        all_status = monitor.get_all_status()
        assert "repl1" in all_status
        assert "repl2" in all_status

    def test_unknown_repl(self):
        monitor = ResourceMonitor()
        assert monitor.get_status("unknown") is None


class TestSafetyPolicy:
    """Test safety policy configuration."""

    def test_default_policy(self):
        policy = SafetyPolicy()
        assert "os" in policy.forbidden_imports
        assert "eval" in policy.forbidden_builtins
        assert not policy.allow_dunder_access

    def test_strict_policy(self):
        policy = SafetyPolicy.strict()
        assert "pickle" in policy.forbidden_imports
        assert "getattr" in policy.forbidden_builtins
        assert policy.timeout_seconds == 15.0

    def test_relaxed_policy(self):
        policy = SafetyPolicy.relaxed()
        assert policy.allow_dunder_access
        assert policy.allow_star_imports
        assert policy.timeout_seconds == 120.0

    def test_from_yaml(self):
        yaml_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "safety", "default_policy.yaml",
        )
        policy = SafetyPolicy.from_yaml(yaml_path)
        assert "os" in policy.forbidden_imports
        assert policy.timeout_seconds == 30
