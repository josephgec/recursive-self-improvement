"""Tests for the recovery tracker."""

from __future__ import annotations

import pytest

from src.measurement.recovery_tracker import RecoveryEvent, RecoveryTracker


class TestRecoveryEvent:
    def test_basic_event(self) -> None:
        event = RecoveryEvent(
            scenario_name="test",
            fault_type="syntax",
            iteration_injected=5,
        )
        assert event.scenario_name == "test"
        assert event.fault_type == "syntax"
        assert event.iteration_injected == 5
        assert not event.was_detected
        assert not event.was_recovered
        assert event.detection_latency is None
        assert event.recovery_latency is None

    def test_detected_event(self) -> None:
        event = RecoveryEvent(
            scenario_name="test",
            fault_type="syntax",
            iteration_injected=5,
            iteration_detected=8,
        )
        assert event.was_detected
        assert event.detection_latency == 3

    def test_recovered_event(self) -> None:
        event = RecoveryEvent(
            scenario_name="test",
            fault_type="syntax",
            iteration_injected=5,
            iteration_detected=8,
            iteration_recovered=10,
            recovery_quality=0.9,
        )
        assert event.was_detected
        assert event.was_recovered
        assert event.detection_latency == 3
        assert event.recovery_latency == 5


class TestRecoveryTracker:
    def test_track_injection(self) -> None:
        tracker = RecoveryTracker()
        event = tracker.track_injection("test", "syntax", iteration=5, complexity=100)
        assert event.scenario_name == "test"
        assert event.fault_type == "syntax"
        assert event.iteration_injected == 5
        assert event.complexity_at_fault == 100

    def test_update_detection(self) -> None:
        tracker = RecoveryTracker()
        tracker.track_injection("test", "syntax", iteration=5)
        tracker.update(
            iteration=8,
            accuracy=0.3,
            detected=True,
            detection_method="accuracy_drop",
        )
        tracker.finalize_event()

        events = tracker.events
        assert len(events) == 1
        assert events[0].was_detected
        assert events[0].detection_method == "accuracy_drop"
        assert events[0].iteration_detected == 8

    def test_update_recovery(self) -> None:
        tracker = RecoveryTracker()
        tracker.track_injection("test", "syntax", iteration=5)
        tracker.update(iteration=8, accuracy=0.3, detected=True, detection_method="drop")
        tracker.update(
            iteration=12,
            accuracy=0.85,
            recovered=True,
            recovery_method="rollback",
            recovery_quality=0.85,
        )
        event = tracker.finalize_event()

        assert event is not None
        assert event.was_recovered
        assert event.recovery_method == "rollback"
        assert event.recovery_quality == 0.85
        assert event.iteration_recovered == 12

    def test_finalize_without_active_event(self) -> None:
        tracker = RecoveryTracker()
        assert tracker.finalize_event() is None

    def test_recovery_rate(self) -> None:
        tracker = RecoveryTracker()

        # Recovered event
        tracker.track_injection("test1", "syntax", iteration=0)
        tracker.update(iteration=2, accuracy=0.3, detected=True)
        tracker.update(iteration=5, accuracy=0.8, recovered=True, recovery_quality=0.8)
        tracker.finalize_event()

        # Not recovered event
        tracker.track_injection("test2", "runtime", iteration=0)
        tracker.update(iteration=2, accuracy=0.2, detected=True)
        tracker.finalize_event()

        assert tracker.get_recovery_rate() == 0.5

    def test_detection_rate(self) -> None:
        tracker = RecoveryTracker()

        # Detected
        tracker.track_injection("test1", "syntax", iteration=0)
        tracker.update(iteration=2, accuracy=0.3, detected=True)
        tracker.finalize_event()

        # Not detected
        tracker.track_injection("test2", "silent", iteration=0)
        tracker.update(iteration=5, accuracy=0.7)
        tracker.finalize_event()

        assert tracker.get_detection_rate() == 0.5

    def test_mean_time_to_detect(self) -> None:
        tracker = RecoveryTracker()

        tracker.track_injection("test1", "a", iteration=0)
        tracker.update(iteration=3, accuracy=0.5, detected=True)
        tracker.finalize_event()

        tracker.track_injection("test2", "b", iteration=0)
        tracker.update(iteration=5, accuracy=0.5, detected=True)
        tracker.finalize_event()

        assert tracker.get_mean_time_to_detect() == 4.0  # (3 + 5) / 2

    def test_mean_time_to_recover(self) -> None:
        tracker = RecoveryTracker()

        tracker.track_injection("test1", "a", iteration=0)
        tracker.update(iteration=4, accuracy=0.8, recovered=True, recovery_quality=0.8)
        tracker.finalize_event()

        tracker.track_injection("test2", "b", iteration=0)
        tracker.update(iteration=8, accuracy=0.8, recovered=True, recovery_quality=0.8)
        tracker.finalize_event()

        assert tracker.get_mean_time_to_recover() == 6.0  # (4 + 8) / 2

    def test_recovery_by_fault_type(self) -> None:
        tracker = RecoveryTracker()

        # Two syntax faults, one recovers
        tracker.track_injection("s1", "syntax", iteration=0)
        tracker.update(iteration=2, accuracy=0.8, recovered=True, recovery_quality=0.8)
        tracker.finalize_event()

        tracker.track_injection("s2", "syntax", iteration=0)
        tracker.update(iteration=5, accuracy=0.3)
        tracker.finalize_event()

        # One runtime fault, recovers
        tracker.track_injection("r1", "runtime", iteration=0)
        tracker.update(iteration=3, accuracy=0.9, recovered=True, recovery_quality=0.9)
        tracker.finalize_event()

        by_type = tracker.recovery_by_fault_type()
        assert by_type["syntax"] == 0.5
        assert by_type["runtime"] == 1.0

    def test_recovery_by_complexity(self) -> None:
        tracker = RecoveryTracker()

        tracker.track_injection("lo", "a", iteration=0, complexity=50)
        tracker.update(iteration=2, accuracy=0.8, recovered=True, recovery_quality=0.8)
        tracker.finalize_event()

        tracker.track_injection("hi", "a", iteration=0, complexity=200)
        tracker.update(iteration=5, accuracy=0.2)
        tracker.finalize_event()

        by_c = tracker.recovery_by_complexity(bins=2)
        assert len(by_c) >= 1  # At least one bin should have data

    def test_empty_tracker(self) -> None:
        tracker = RecoveryTracker()
        assert tracker.get_recovery_rate() == 0.0
        assert tracker.get_detection_rate() == 0.0
        assert tracker.get_mean_time_to_detect() is None
        assert tracker.get_mean_time_to_recover() is None
        assert tracker.recovery_by_fault_type() == {}

    def test_accuracy_tracking(self) -> None:
        tracker = RecoveryTracker()
        tracker.track_injection("test", "a", iteration=0)
        tracker.update(iteration=1, accuracy=0.5)
        tracker.update(iteration=2, accuracy=0.4)
        tracker.update(iteration=3, accuracy=0.6)
        event = tracker.finalize_event()

        assert event is not None
        assert event.accuracies == [0.5, 0.4, 0.6]
