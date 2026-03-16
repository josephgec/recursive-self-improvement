"""Tests for ConstraintGate."""

import pytest
from tests.conftest import MockAgent
from src.checker.suite import ConstraintSuite
from src.checker.runner import ConstraintRunner
from src.enforcement.gate import ConstraintGate, GateDecision
from src.constraints.base import CheckContext


class TestConstraintGate:
    """Tests for ConstraintGate."""

    def _make_gate(self) -> ConstraintGate:
        suite = ConstraintSuite()
        runner = ConstraintRunner(suite, parallel=False)
        return ConstraintGate(runner)

    def test_allow_when_pass(self, check_context):
        """Good agent passes the gate."""
        gate = self._make_gate()
        agent = MockAgent()

        decision = gate.check_and_gate(agent, check_context)

        assert decision.allowed is True
        assert decision.verdict.passed is True
        assert "All constraints satisfied" in decision.reason

    def test_reject_when_fail(self, check_context):
        """Bad agent is rejected by the gate."""
        gate = self._make_gate()
        agent = MockAgent(accuracy=0.50)

        decision = gate.check_and_gate(agent, check_context)

        assert decision.allowed is False
        assert decision.verdict.passed is False
        assert "accuracy_floor" in decision.reason

    def test_wrap_modification_allowed(self, check_context):
        """Modification executes when constraints pass."""
        gate = self._make_gate()
        agent = MockAgent()

        executed = []

        def my_modification():
            executed.append(True)
            return "done"

        decision = gate.wrap_modification(my_modification, agent, check_context)

        assert decision.allowed is True
        assert len(executed) == 1
        assert decision.context.metadata.get("modification_result") == "done"

    def test_wrap_modification_rejected(self, check_context):
        """Modification does NOT execute when constraints fail."""
        gate = self._make_gate()
        agent = MockAgent(accuracy=0.50)

        executed = []

        def my_modification():
            executed.append(True)
            return "done"

        decision = gate.wrap_modification(my_modification, agent, check_context)

        assert decision.allowed is False
        assert len(executed) == 0

    def test_no_override(self, check_context):
        """GateDecision is frozen -- no way to override."""
        gate = self._make_gate()
        agent = MockAgent(accuracy=0.50)

        decision = gate.check_and_gate(agent, check_context)
        assert decision.allowed is False

        # Frozen dataclass prevents mutation
        with pytest.raises(AttributeError):
            decision.allowed = True  # type: ignore

    def test_default_context(self):
        """Gate works without explicit context."""
        gate = self._make_gate()
        agent = MockAgent()

        decision = gate.check_and_gate(agent)
        assert decision.allowed is True

    def test_gate_decision_is_frozen(self, check_context):
        """GateDecision fields cannot be modified."""
        gate = self._make_gate()
        agent = MockAgent()
        decision = gate.check_and_gate(agent, check_context)

        with pytest.raises(AttributeError):
            decision.reason = "override"  # type: ignore

    def test_wrap_modification_default_context(self):
        """wrap_modification works without explicit context."""
        gate = self._make_gate()
        agent = MockAgent()

        decision = gate.wrap_modification(lambda: 42, agent)
        assert decision.allowed is True
        assert decision.context.metadata.get("modification_result") == 42
