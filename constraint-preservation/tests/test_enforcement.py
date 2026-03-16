"""Tests for enforcement: rollback, audit logging, hash chain."""

import pytest
from tests.conftest import MockAgent
from src.checker.suite import ConstraintSuite
from src.checker.runner import ConstraintRunner
from src.checker.verdict import SuiteVerdict
from src.enforcement.gate import ConstraintGate
from src.enforcement.rollback_trigger import RollbackTrigger
from src.enforcement.rejection_handler import RejectionHandler
from src.enforcement.audit import ConstraintAuditLog
from src.constraints.base import CheckContext, ConstraintResult


class _MockRollbackManager:
    """Records rollback calls."""

    def __init__(self):
        self.rollbacks = []

    def rollback(self, reason: str) -> None:
        self.rollbacks.append(reason)


class TestRollbackTrigger:
    """Tests for RollbackTrigger."""

    def test_trigger_without_manager(self):
        """Trigger without manager records event but does not execute."""
        trigger = RollbackTrigger()
        event = trigger.trigger("test failure")

        assert event["action"] == "rollback"
        assert event["reason"] == "test failure"
        assert event["rollback_executed"] is False

    def test_trigger_with_manager(self):
        """Trigger with manager executes the rollback."""
        trigger = RollbackTrigger()
        manager = _MockRollbackManager()
        trigger.set_rollback_manager(manager)

        event = trigger.trigger("constraint violation")

        assert event["rollback_executed"] is True
        assert len(manager.rollbacks) == 1
        assert manager.rollbacks[0] == "constraint violation"

    def test_history(self):
        """History records all trigger events."""
        trigger = RollbackTrigger()
        trigger.trigger("reason1")
        trigger.trigger("reason2")

        assert len(trigger.history) == 2
        assert trigger.history[0]["reason"] == "reason1"

    def test_rollback_on_failure_integration(self, check_context):
        """Full flow: gate rejects, rollback is triggered."""
        suite = ConstraintSuite()
        runner = ConstraintRunner(suite, parallel=False)
        gate = ConstraintGate(runner)
        trigger = RollbackTrigger()
        manager = _MockRollbackManager()
        trigger.set_rollback_manager(manager)

        agent = MockAgent(accuracy=0.50)
        decision = gate.check_and_gate(agent, check_context)

        assert decision.allowed is False
        event = trigger.trigger(decision.reason)
        assert event["rollback_executed"] is True
        assert len(manager.rollbacks) == 1


class TestRejectionHandler:
    """Tests for RejectionHandler."""

    def test_format_message(self, check_context):
        """Rejection message contains violation info."""
        results = {
            "accuracy_floor": ConstraintResult(False, 0.75, 0.80, -0.05),
        }
        verdict = SuiteVerdict(passed=False, results=results)
        handler = RejectionHandler()

        msg = handler.handle(verdict, check_context)

        assert "MODIFICATION REJECTED" in msg
        assert "accuracy_floor" in msg
        assert "0.7500" in msg
        assert "0.8000" in msg

    def test_format_message_multiple_violations(self, check_context):
        """Message includes all violated constraints."""
        results = {
            "accuracy_floor": ConstraintResult(False, 0.75, 0.80, -0.05),
            "safety_eval": ConstraintResult(False, 0.90, 1.0, -0.10),
        }
        verdict = SuiteVerdict(passed=False, results=results)
        handler = RejectionHandler()

        msg = handler.handle(verdict, check_context)

        assert "accuracy_floor" in msg
        assert "safety_eval" in msg
        assert "No override" in msg


class TestAuditLog:
    """Tests for ConstraintAuditLog."""

    def _make_verdict(self, passed: bool) -> SuiteVerdict:
        results = {
            "accuracy_floor": ConstraintResult(passed, 0.85 if passed else 0.75, 0.80, 0.05 if passed else -0.05),
        }
        return SuiteVerdict(passed=passed, results=results)

    def test_log_entry(self, check_context):
        """Log creates an entry with a hash."""
        log = ConstraintAuditLog()
        verdict = self._make_verdict(True)

        entry = log.log(verdict, check_context, "allowed")

        assert entry["index"] == 0
        assert entry["decision"] == "allowed"
        assert entry["passed"] is True
        assert "hash" in entry
        assert len(entry["hash"]) == 64

    def test_get_history(self, check_context):
        """History returns all entries."""
        log = ConstraintAuditLog()
        verdict = self._make_verdict(True)
        log.log(verdict, check_context, "allowed")
        log.log(verdict, check_context, "allowed")

        assert len(log.get_history()) == 2

    def test_get_violations(self, check_context):
        """get_violations returns only failed entries."""
        log = ConstraintAuditLog()
        log.log(self._make_verdict(True), check_context, "allowed")
        log.log(self._make_verdict(False), check_context, "rejected")

        violations = log.get_violations()
        assert len(violations) == 1
        assert violations[0]["passed"] is False

    def test_hash_chain_integrity(self, check_context):
        """Hash chain is valid after multiple entries."""
        log = ConstraintAuditLog()
        for i in range(10):
            passed = i % 3 != 0
            verdict = self._make_verdict(passed)
            log.log(verdict, check_context, "allowed" if passed else "rejected")

        assert log.verify_integrity() is True

    def test_tampered_entry_detected(self, check_context):
        """Tampering with an entry breaks the hash chain."""
        log = ConstraintAuditLog()
        log.log(self._make_verdict(True), check_context, "allowed")
        log.log(self._make_verdict(True), check_context, "allowed")

        # Tamper with the first entry
        log._entries[0]["decision"] = "tampered"

        assert log.verify_integrity() is False

    def test_empty_log_integrity(self):
        """Empty log has valid integrity."""
        log = ConstraintAuditLog()
        assert log.verify_integrity() is True

    def test_persist_to_file(self, tmp_path, check_context):
        """Log can persist entries to a JSONL file."""
        log_file = str(tmp_path / "audit.jsonl")
        log = ConstraintAuditLog(log_path=log_file)
        log.log(self._make_verdict(True), check_context, "allowed")

        import json
        with open(log_file) as f:
            lines = f.readlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["decision"] == "allowed"

    def test_chain_hashes_link(self, check_context):
        """Each entry's prev_hash matches the previous entry's hash."""
        log = ConstraintAuditLog()
        log.log(self._make_verdict(True), check_context, "allowed")
        log.log(self._make_verdict(True), check_context, "allowed")

        entries = log.get_history()
        assert entries[0]["prev_hash"] == "0" * 64
        assert entries[1]["prev_hash"] == entries[0]["hash"]
