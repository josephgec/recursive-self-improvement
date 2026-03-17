"""Tests for Criterion 5: Auditability."""

import hashlib

import pytest

from src.criteria.base import Evidence
from src.criteria.auditability import AuditabilityCriterion
from tests.conftest import _build_hash_chain, _build_broken_hash_chain


class TestAuditability:
    """Test the Auditability criterion."""

    def test_complete_logs_pass(self, passing_evidence):
        """Complete audit trail with all logs, traces, and chain passes."""
        criterion = AuditabilityCriterion()
        result = criterion.evaluate(passing_evidence)

        assert result.passed is True
        assert result.confidence == 1.0

        sub = result.details["sub_results"]
        assert sub["modification_log"]["passed"] is True
        assert sub["constraint_log"]["passed"] is True
        assert sub["gdi_log"]["passed"] is True
        assert sub["interp_log"]["passed"] is True
        assert sub["reasoning_traces"]["passed"] is True
        assert sub["hash_chain"]["passed"] is True

    def test_missing_log_fails(self):
        """Missing a required log should fail."""
        audit_trail = {
            "modification_log": [{"entry": 1}],
            "constraint_log": [{"entry": 1}],
            # gdi_log missing
            "interp_log": [{"entry": 1}],
            "reasoning_traces": [{"id": i} for i in range(25)],
            "hash_chain": _build_hash_chain(),
        }
        evidence = Evidence(audit_trail=audit_trail)
        criterion = AuditabilityCriterion()
        result = criterion.evaluate(evidence)

        assert result.passed is False
        sub = result.details["sub_results"]
        assert sub["gdi_log"]["passed"] is False

    def test_empty_log_fails(self):
        """Empty log (present but no entries) should fail."""
        audit_trail = {
            "modification_log": [],  # empty
            "constraint_log": [{"entry": 1}],
            "gdi_log": [{"entry": 1}],
            "interp_log": [{"entry": 1}],
            "reasoning_traces": [{"id": i} for i in range(25)],
            "hash_chain": _build_hash_chain(),
        }
        evidence = Evidence(audit_trail=audit_trail)
        criterion = AuditabilityCriterion()
        result = criterion.evaluate(evidence)

        assert result.passed is False
        sub = result.details["sub_results"]
        assert sub["modification_log"]["passed"] is False

    def test_broken_chain_fails(self):
        """Broken hash chain should fail."""
        audit_trail = {
            "modification_log": [{"entry": 1}],
            "constraint_log": [{"entry": 1}],
            "gdi_log": [{"entry": 1}],
            "interp_log": [{"entry": 1}],
            "reasoning_traces": [{"id": i} for i in range(25)],
            "hash_chain": _build_broken_hash_chain(10, break_at=3),
        }
        evidence = Evidence(audit_trail=audit_trail)
        criterion = AuditabilityCriterion()
        result = criterion.evaluate(evidence)

        assert result.passed is False
        sub = result.details["sub_results"]
        assert sub["hash_chain"]["passed"] is False

    def test_fewer_than_20_traces_fails(self):
        """Fewer than 20 reasoning traces should fail."""
        audit_trail = {
            "modification_log": [{"entry": 1}],
            "constraint_log": [{"entry": 1}],
            "gdi_log": [{"entry": 1}],
            "interp_log": [{"entry": 1}],
            "reasoning_traces": [{"id": i} for i in range(10)],  # only 10
            "hash_chain": _build_hash_chain(),
        }
        evidence = Evidence(audit_trail=audit_trail)
        criterion = AuditabilityCriterion()
        result = criterion.evaluate(evidence)

        assert result.passed is False
        sub = result.details["sub_results"]
        assert sub["reasoning_traces"]["passed"] is False
        assert sub["reasoning_traces"]["count"] == 10

    def test_exactly_20_traces_passes(self):
        """Exactly 20 traces should pass."""
        audit_trail = {
            "modification_log": [{"entry": 1}],
            "constraint_log": [{"entry": 1}],
            "gdi_log": [{"entry": 1}],
            "interp_log": [{"entry": 1}],
            "reasoning_traces": [{"id": i} for i in range(20)],
            "hash_chain": _build_hash_chain(),
        }
        evidence = Evidence(audit_trail=audit_trail)
        criterion = AuditabilityCriterion()
        result = criterion.evaluate(evidence)

        assert result.passed is True

    def test_empty_hash_chain_fails(self):
        """Empty hash chain should fail."""
        audit_trail = {
            "modification_log": [{"entry": 1}],
            "constraint_log": [{"entry": 1}],
            "gdi_log": [{"entry": 1}],
            "interp_log": [{"entry": 1}],
            "reasoning_traces": [{"id": i} for i in range(25)],
            "hash_chain": [],
        }
        evidence = Evidence(audit_trail=audit_trail)
        criterion = AuditabilityCriterion()
        result = criterion.evaluate(evidence)

        assert result.passed is False

    def test_confidence_proportional(self):
        """Confidence should reflect fraction of sub-tests passed."""
        # Only 3 of 6 sub-tests pass (3 logs present, but missing 1 log,
        # insufficient traces, broken chain)
        audit_trail = {
            "modification_log": [{"entry": 1}],
            "constraint_log": [{"entry": 1}],
            "gdi_log": [{"entry": 1}],
            # interp_log missing
            "reasoning_traces": [{"id": i} for i in range(5)],
            "hash_chain": _build_broken_hash_chain(),
        }
        evidence = Evidence(audit_trail=audit_trail)
        criterion = AuditabilityCriterion()
        result = criterion.evaluate(evidence)

        assert result.passed is False
        assert result.confidence == 3.0 / 6.0  # 3 passed out of 6

    def test_hash_chain_verification(self):
        """Verify hash chain verification logic directly."""
        chain = _build_hash_chain(5)
        assert AuditabilityCriterion._verify_hash_chain(chain) is True

        broken = _build_broken_hash_chain(5, 2)
        assert AuditabilityCriterion._verify_hash_chain(broken) is False

        assert AuditabilityCriterion._verify_hash_chain([]) is False
        assert AuditabilityCriterion._verify_hash_chain("not a list") is False

    def test_properties(self):
        """Test criterion properties."""
        criterion = AuditabilityCriterion()
        assert criterion.name == "Auditability"
        assert "audit" in criterion.description.lower()
        assert "audit_trail" in criterion.required_evidence

    def test_margin_is_trace_surplus(self, passing_evidence):
        """Margin should be n_traces - min_threshold."""
        criterion = AuditabilityCriterion(min_reasoning_traces=20)
        result = criterion.evaluate(passing_evidence)
        # Passing evidence has 25 traces
        assert result.margin == 5.0
