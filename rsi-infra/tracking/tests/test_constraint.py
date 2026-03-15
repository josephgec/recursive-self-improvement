"""Tests for constraint preservation."""

from __future__ import annotations

import pytest

from tracking.src.constraint import (
    ConstraintDef,
    ConstraintPreserver,
    ConstraintResult,
    PreservationReport,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def basic_constraints() -> list[ConstraintDef]:
    """A minimal set of constraints for testing."""
    return [
        ConstraintDef(name="accuracy_floor", metric_key="accuracy", threshold=0.6, direction=">=", severity="revert"),
        ConstraintDef(name="safety_eval_pass", metric_key="safety_score", threshold=0.9, direction=">=", severity="halt"),
        ConstraintDef(name="loss_ceiling", metric_key="loss", threshold=5.0, direction="<=", severity="revert"),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConstraintProceed:
    """Healthy metrics → proceed."""

    def test_all_above_threshold_proceed(self, basic_constraints: list[ConstraintDef]) -> None:
        preserver = ConstraintPreserver(basic_constraints)
        metrics = {"accuracy": 0.85, "safety_score": 0.95, "loss": 2.0}
        report = preserver.check(generation=1, metrics=metrics)
        assert report.all_passed is True
        assert report.recommendation == "proceed"
        assert all(r.passed for r in report.results)

    def test_missing_metric_is_skipped(self, basic_constraints: list[ConstraintDef]) -> None:
        """If a metric is not provided, the constraint is skipped (assumed OK)."""
        preserver = ConstraintPreserver(basic_constraints)
        metrics = {"accuracy": 0.85}  # safety_score and loss not provided
        report = preserver.check(generation=1, metrics=metrics)
        assert report.recommendation == "proceed"
        assert len(report.results) == 1  # only accuracy checked


class TestConstraintRevert:
    """Revert-severity violations."""

    def test_accuracy_below_floor_revert(self, basic_constraints: list[ConstraintDef]) -> None:
        preserver = ConstraintPreserver(basic_constraints)
        metrics = {"accuracy": 0.4, "safety_score": 0.95, "loss": 2.0}
        report = preserver.check(generation=2, metrics=metrics)
        assert report.all_passed is False
        assert report.recommendation == "revert"
        # Find the failing constraint
        failing = [r for r in report.results if not r.passed]
        assert len(failing) == 1
        assert failing[0].name == "accuracy_floor"
        assert failing[0].violation_severity == "revert"

    def test_loss_above_ceiling_revert(self, basic_constraints: list[ConstraintDef]) -> None:
        preserver = ConstraintPreserver(basic_constraints)
        metrics = {"accuracy": 0.7, "safety_score": 0.95, "loss": 10.0}
        report = preserver.check(generation=3, metrics=metrics)
        assert report.recommendation == "revert"


class TestConstraintHalt:
    """Halt-severity violations."""

    def test_safety_eval_fails_halt(self, basic_constraints: list[ConstraintDef]) -> None:
        preserver = ConstraintPreserver(basic_constraints)
        metrics = {"accuracy": 0.7, "safety_score": 0.5, "loss": 2.0}
        report = preserver.check(generation=4, metrics=metrics)
        assert report.all_passed is False
        assert report.recommendation == "halt"
        failing = [r for r in report.results if not r.passed]
        assert any(r.violation_severity == "halt" for r in failing)

    def test_halt_takes_precedence_over_revert(self, basic_constraints: list[ConstraintDef]) -> None:
        """When both halt and revert violations exist, halt wins."""
        preserver = ConstraintPreserver(basic_constraints)
        metrics = {"accuracy": 0.3, "safety_score": 0.1, "loss": 20.0}
        report = preserver.check(generation=5, metrics=metrics)
        assert report.recommendation == "halt"


class TestMultipleViolations:
    """Multiple constraint violations at once."""

    def test_multiple_violations_reported(self, basic_constraints: list[ConstraintDef]) -> None:
        preserver = ConstraintPreserver(basic_constraints)
        metrics = {"accuracy": 0.3, "safety_score": 0.5, "loss": 20.0}
        report = preserver.check(generation=6, metrics=metrics)
        failing = [r for r in report.results if not r.passed]
        assert len(failing) == 3  # all three constraints violated


class TestAddConstraintRuntime:
    """Adding constraints at runtime."""

    def test_add_constraint_at_runtime(self) -> None:
        preserver = ConstraintPreserver(constraints=[])
        # No constraints — everything passes
        report = preserver.check(1, {"accuracy": 0.1})
        assert report.recommendation == "proceed"

        # Add a constraint
        preserver.add_constraint(
            name="min_accuracy",
            metric_key="accuracy",
            threshold=0.5,
            direction=">=",
            severity="revert",
        )
        report = preserver.check(2, {"accuracy": 0.1})
        assert report.recommendation == "revert"

    def test_add_constraint_with_strict_direction(self) -> None:
        preserver = ConstraintPreserver(constraints=[])
        preserver.add_constraint(
            name="diversity",
            metric_key="unique_ratio",
            threshold=0.5,
            direction=">",
            severity="revert",
        )
        # Exactly at threshold should fail for strict >
        report = preserver.check(1, {"unique_ratio": 0.5})
        assert report.recommendation == "revert"

        report = preserver.check(2, {"unique_ratio": 0.51})
        assert report.recommendation == "proceed"


class TestConstraintReportFields:
    """Verify report data class fields."""

    def test_generation_is_stored(self, basic_constraints: list[ConstraintDef]) -> None:
        preserver = ConstraintPreserver(basic_constraints)
        report = preserver.check(42, {"accuracy": 0.9, "safety_score": 0.99, "loss": 1.0})
        assert report.generation == 42

    def test_result_values_match(self, basic_constraints: list[ConstraintDef]) -> None:
        preserver = ConstraintPreserver(basic_constraints)
        report = preserver.check(1, {"accuracy": 0.75, "safety_score": 0.92, "loss": 3.5})
        acc_result = next(r for r in report.results if r.name == "accuracy_floor")
        assert acc_result.value == 0.75
        assert acc_result.threshold == 0.6
        assert acc_result.passed is True
        assert acc_result.violation_severity is None
