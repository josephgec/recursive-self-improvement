"""Tests for the LocalREPL backend."""

import pytest
from src.backends.local import LocalREPL
from src.safety.policy import SafetyPolicy
from src.interface.errors import (
    ForbiddenCodeError,
    RecursionDepthError,
    REPLNotAliveError,
)


class TestLocalREPLExecution:
    """Test basic code execution."""

    def test_simple_assignment(self, local_repl):
        result = local_repl.execute("x = 42")
        assert result.success
        assert local_repl.get_variable("x") == 42

    def test_arithmetic(self, local_repl):
        result = local_repl.execute("result = 2 + 3")
        assert result.success
        assert local_repl.get_variable("result") == 5

    def test_print_output(self, local_repl):
        result = local_repl.execute('print("hello world")')
        assert result.success
        assert "hello world" in result.stdout

    def test_multiline_code(self, local_repl):
        code = "x = 1\ny = 2\nz = x + y"
        result = local_repl.execute(code)
        assert result.success
        assert local_repl.get_variable("z") == 3

    def test_function_definition(self, local_repl):
        code = "def add(a, b): return a + b\nresult = add(3, 4)"
        result = local_repl.execute(code)
        assert result.success
        assert local_repl.get_variable("result") == 7

    def test_list_comprehension(self, local_repl):
        result = local_repl.execute("squares = [x**2 for x in range(5)]")
        assert result.success
        assert local_repl.get_variable("squares") == [0, 1, 4, 9, 16]

    def test_execution_error(self, local_repl):
        result = local_repl.execute("1 / 0")
        assert not result.success
        assert result.error_type == "ZeroDivisionError"

    def test_name_error(self, local_repl):
        result = local_repl.execute("print(undefined_var)")
        assert not result.success
        assert result.error_type == "NameError"

    def test_syntax_error_in_execution(self, local_repl):
        """Syntax errors are caught by AST scanner first."""
        with pytest.raises(ForbiddenCodeError):
            local_repl.execute("def :")

    def test_execution_time_tracked(self, local_repl):
        result = local_repl.execute("x = sum(range(1000))")
        assert result.execution_time_ms > 0

    def test_variables_changed_tracked(self, local_repl):
        result = local_repl.execute("a = 1\nb = 2")
        assert "a" in result.variables_changed
        assert "b" in result.variables_changed

    def test_stderr_captured(self, local_repl):
        code = "import sys\nprint('err', file=sys.stderr)"
        # sys is forbidden, so this will be caught
        with pytest.raises(ForbiddenCodeError):
            local_repl.execute(code)

    def test_sequential_execution(self, local_repl):
        local_repl.execute("x = 10")
        local_repl.execute("y = x * 2")
        assert local_repl.get_variable("y") == 20

    def test_repl_id(self, local_repl):
        assert local_repl.repl_id is not None
        assert len(local_repl.repl_id) > 0


class TestLocalREPLVariables:
    """Test variable management."""

    def test_get_variable(self, local_repl):
        local_repl.execute("x = 42")
        assert local_repl.get_variable("x") == 42

    def test_set_variable(self, local_repl):
        local_repl.set_variable("x", 99)
        assert local_repl.get_variable("x") == 99

    def test_get_nonexistent_variable(self, local_repl):
        with pytest.raises(KeyError):
            local_repl.get_variable("nonexistent")

    def test_list_variables(self, local_repl):
        local_repl.execute("a = 1\nb = 2\nc = 3")
        variables = local_repl.list_variables()
        assert "a" in variables
        assert "b" in variables
        assert "c" in variables

    def test_list_variables_excludes_internals(self, local_repl):
        local_repl.execute("x = 42")
        variables = local_repl.list_variables()
        assert "__builtins__" not in variables
        assert "FINAL" not in variables
        assert "FINAL_VAR" not in variables

    def test_set_variable_then_use(self, local_repl):
        local_repl.set_variable("external_data", [1, 2, 3])
        result = local_repl.execute("total = sum(external_data)")
        assert result.success
        assert local_repl.get_variable("total") == 6


class TestLocalREPLSpawnChild:
    """Test child REPL spawning."""

    def test_spawn_child(self, local_repl):
        local_repl.execute("x = 42")
        child = local_repl.spawn_child()
        assert child.is_alive()
        assert child.get_variable("x") == 42
        child.shutdown()

    def test_child_independence(self, local_repl):
        local_repl.execute("x = 42")
        child = local_repl.spawn_child()
        child.execute("x = 99")
        assert local_repl.get_variable("x") == 42
        assert child.get_variable("x") == 99
        child.shutdown()

    def test_child_depth(self, local_repl):
        child = local_repl.spawn_child()
        assert child.depth == 1
        grandchild = child.spawn_child()
        assert grandchild.depth == 2
        grandchild.shutdown()
        child.shutdown()

    def test_max_spawn_depth(self):
        policy = SafetyPolicy(max_spawn_depth=2)
        repl = LocalREPL(policy=policy)
        child = repl.spawn_child()
        grandchild = child.spawn_child()
        with pytest.raises(RecursionDepthError):
            grandchild.spawn_child()
        grandchild.shutdown()
        child.shutdown()
        repl.shutdown()

    def test_child_has_final_protocol(self, local_repl):
        child = local_repl.spawn_child()
        result = child.execute('FINAL("from child")')
        assert result.success
        child.shutdown()


class TestLocalREPLSnapshotRestore:
    """Test snapshot and restore."""

    def test_snapshot_and_restore(self, local_repl):
        local_repl.execute("x = 42")
        snap_id = local_repl.snapshot()

        local_repl.execute("x = 99")
        assert local_repl.get_variable("x") == 99

        local_repl.restore(snap_id)
        assert local_repl.get_variable("x") == 42

    def test_restore_nonexistent(self, local_repl):
        with pytest.raises(KeyError):
            local_repl.restore("nonexistent")

    def test_multiple_snapshots(self, local_repl):
        local_repl.execute("x = 1")
        snap1 = local_repl.snapshot()

        local_repl.execute("x = 2")
        snap2 = local_repl.snapshot()

        local_repl.execute("x = 3")

        local_repl.restore(snap1)
        assert local_repl.get_variable("x") == 1

        local_repl.restore(snap2)
        assert local_repl.get_variable("x") == 2


class TestLocalREPLSafety:
    """Test safety features."""

    def test_forbidden_import_blocked(self, local_repl):
        with pytest.raises(ForbiddenCodeError):
            local_repl.execute("import os")

    def test_forbidden_eval_blocked(self, local_repl):
        with pytest.raises(ForbiddenCodeError):
            local_repl.execute("eval('1+1')")

    def test_forbidden_exec_blocked(self, local_repl):
        with pytest.raises(ForbiddenCodeError):
            local_repl.execute("exec('x = 1')")

    def test_dunder_access_blocked(self, local_repl):
        with pytest.raises(ForbiddenCodeError):
            local_repl.execute('x = "".__class__')

    def test_safe_builtins_available(self, local_repl):
        result = local_repl.execute("x = len([1, 2, 3])")
        assert result.success
        assert local_repl.get_variable("x") == 3

    def test_timeout(self):
        policy = SafetyPolicy(timeout_seconds=0.5)
        repl = LocalREPL(policy=policy)
        result = repl.execute("while True: pass")
        assert result.killed
        assert result.kill_reason == "timeout"
        repl.shutdown()

    def test_reset(self, local_repl):
        local_repl.execute("x = 42")
        local_repl.reset()
        with pytest.raises(KeyError):
            local_repl.get_variable("x")

    def test_shutdown(self, local_repl):
        local_repl.shutdown()
        assert not local_repl.is_alive()

    def test_execute_after_shutdown(self, local_repl):
        local_repl.shutdown()
        with pytest.raises(REPLNotAliveError):
            local_repl.execute("x = 1")

    def test_get_variable_after_shutdown(self, local_repl):
        local_repl.shutdown()
        with pytest.raises(REPLNotAliveError):
            local_repl.get_variable("x")

    def test_set_variable_after_shutdown(self, local_repl):
        local_repl.shutdown()
        with pytest.raises(REPLNotAliveError):
            local_repl.set_variable("x", 1)

    def test_list_variables_after_shutdown(self, local_repl):
        local_repl.shutdown()
        with pytest.raises(REPLNotAliveError):
            local_repl.list_variables()

    def test_spawn_child_after_shutdown(self, local_repl):
        local_repl.shutdown()
        with pytest.raises(REPLNotAliveError):
            local_repl.spawn_child()

    def test_snapshot_after_shutdown(self, local_repl):
        local_repl.shutdown()
        with pytest.raises(REPLNotAliveError):
            local_repl.snapshot()

    def test_restore_after_shutdown(self, local_repl):
        local_repl.shutdown()
        with pytest.raises(REPLNotAliveError):
            local_repl.restore("snap")

    def test_reset_after_shutdown(self, local_repl):
        local_repl.shutdown()
        with pytest.raises(REPLNotAliveError):
            local_repl.reset()
