"""Tests for the failure classifier -- all 12 failure modes."""

from __future__ import annotations

import pytest

from src.measurement.failure_classifier import FailureClassifier, FailureMode


@pytest.fixture
def classifier() -> FailureClassifier:
    return FailureClassifier()


class TestFailureModeClassification:
    def test_validation_caught(self, classifier: FailureClassifier) -> None:
        """High rejection rate -> VALIDATION_CAUGHT."""
        mode = classifier.classify(
            accuracies=[0.8, 0.8, 0.8],
            modification_count=10,
            validation_rejections=9,
        )
        assert mode == FailureMode.VALIDATION_CAUGHT

    def test_deliberation_avoided(self, classifier: FailureClassifier) -> None:
        """Zero modifications -> DELIBERATION_AVOIDED."""
        mode = classifier.classify(
            accuracies=[0.8, 0.8, 0.8],
            modification_count=0,
        )
        assert mode == FailureMode.DELIBERATION_AVOIDED

    def test_stagnation(self, classifier: FailureClassifier) -> None:
        """Flat performance -> STAGNATION."""
        mode = classifier.classify(
            accuracies=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            modification_count=5,
        )
        assert mode == FailureMode.STAGNATION

    def test_oscillation(self, classifier: FailureClassifier) -> None:
        """Alternating performance -> OSCILLATION."""
        mode = classifier.classify(
            accuracies=[0.8, 0.3, 0.8, 0.3, 0.8, 0.3, 0.8, 0.3],
            modification_count=8,
        )
        assert mode == FailureMode.OSCILLATION

    def test_silent_degradation(self, classifier: FailureClassifier) -> None:
        """Steady decline -> SILENT_DEGRADATION."""
        mode = classifier.classify(
            accuracies=[0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
            modification_count=8,
        )
        assert mode == FailureMode.SILENT_DEGRADATION

    def test_complexity_explosion(self, classifier: FailureClassifier) -> None:
        """Huge complexity growth -> COMPLEXITY_EXPLOSION."""
        mode = classifier.classify(
            accuracies=[0.8, 0.7, 0.6, 0.5],
            complexities=[50, 100, 200, 500],
            modification_count=4,
        )
        assert mode == FailureMode.COMPLEXITY_EXPLOSION

    def test_runaway_modification(self, classifier: FailureClassifier) -> None:
        """Way too many modifications -> RUNAWAY_MODIFICATION."""
        mode = classifier.classify(
            accuracies=[0.5, 0.5, 0.5],
            modification_count=200,  # >> max_iterations * 2
            max_iterations=50,
        )
        assert mode == FailureMode.RUNAWAY_MODIFICATION

    def test_rollback_failure(self, classifier: FailureClassifier) -> None:
        """Rollbacks that don't restore performance -> ROLLBACK_FAILURE."""
        mode = classifier.classify(
            accuracies=[0.8, 0.7, 0.6, 0.3, 0.2],
            modification_count=5,
            rollback_count=3,
        )
        assert mode == FailureMode.ROLLBACK_FAILURE

    def test_rollback_partial(self, classifier: FailureClassifier) -> None:
        """Rollbacks that partially restore -> ROLLBACK_PARTIAL."""
        mode = classifier.classify(
            accuracies=[0.8, 0.7, 0.6, 0.5, 0.55],
            modification_count=5,
            rollback_count=2,
        )
        assert mode == FailureMode.ROLLBACK_PARTIAL

    def test_state_corruption(self, classifier: FailureClassifier) -> None:
        """Agent not functional -> STATE_CORRUPTION."""
        mode = classifier.classify(
            accuracies=[0.8, 0.5, 0.0],
            modification_count=3,
            agent_functional=False,
        )
        assert mode == FailureMode.STATE_CORRUPTION

    def test_self_lobotomy(self, classifier: FailureClassifier) -> None:
        """Agent can't modify -> SELF_LOBOTOMY."""
        mode = classifier.classify(
            accuracies=[0.8, 0.5],
            modification_count=2,
            agent_can_modify=False,
        )
        assert mode == FailureMode.SELF_LOBOTOMY

    def test_infinite_loop(self, classifier: FailureClassifier) -> None:
        """Hit max iterations with no progress -> INFINITE_LOOP."""
        flat = [0.5] * 50
        mode = classifier.classify(
            accuracies=flat,
            modification_count=50,
            max_iterations=50,
        )
        assert mode == FailureMode.INFINITE_LOOP


class TestSeverity:
    def test_severity_range(self, classifier: FailureClassifier) -> None:
        for mode in FailureMode:
            sev = classifier.get_severity(mode)
            assert 1 <= sev <= 5, f"{mode} severity {sev} out of range"

    def test_catastrophic_modes_high_severity(self, classifier: FailureClassifier) -> None:
        assert classifier.get_severity(FailureMode.SELF_LOBOTOMY) == 5
        assert classifier.get_severity(FailureMode.STATE_CORRUPTION) == 5

    def test_benign_modes_low_severity(self, classifier: FailureClassifier) -> None:
        assert classifier.get_severity(FailureMode.VALIDATION_CAUGHT) == 1
        assert classifier.get_severity(FailureMode.DELIBERATION_AVOIDED) == 1


class TestClassifyMultiple:
    def test_multiple_modes(self, classifier: FailureClassifier) -> None:
        modes = classifier.classify_multiple(
            accuracies=[0.9, 0.5, 0.9, 0.5, 0.9, 0.5],
            modification_count=6,
        )
        assert FailureMode.OSCILLATION in modes

    def test_multiple_modes_declining(self, classifier: FailureClassifier) -> None:
        modes = classifier.classify_multiple(
            accuracies=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
            modification_count=7,
        )
        assert FailureMode.SILENT_DEGRADATION in modes


class TestEdgeCases:
    def test_empty_accuracies(self, classifier: FailureClassifier) -> None:
        mode = classifier.classify(accuracies=[])
        assert mode == FailureMode.STAGNATION

    def test_single_accuracy(self, classifier: FailureClassifier) -> None:
        mode = classifier.classify(accuracies=[0.5], modification_count=1)
        assert isinstance(mode, FailureMode)

    def test_all_modes_are_valid(self, classifier: FailureClassifier) -> None:
        """Ensure all FailureMode enum values are valid."""
        assert len(FailureMode) == 12
