"""Tests for goal drift index computation."""

from __future__ import annotations

import numpy as np
import pytest

from tracking.src.goal_drift import GoalDriftComputer, GoalDriftMeasurement


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def config() -> dict:
    return {
        "safety": {
            "weights": {
                "semantic": 0.3,
                "lexical": 0.2,
                "structural": 0.2,
                "distributional": 0.3,
            },
            "alert_threshold_drift_cosine": 0.15,
        },
    }


@pytest.fixture()
def reference_texts() -> list[str]:
    return [
        "The quick brown fox jumps over the lazy dog.",
        "A fast brown fox leaps across a sleepy hound.",
        "Quick foxes jump over lazy dogs in the park.",
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGoalDriftBasic:
    """Basic drift computation tests."""

    def test_same_texts_give_near_zero_gdi(self, config: dict, reference_texts: list[str]) -> None:
        """When generated texts are the same as reference, GDI should be ~0."""
        computer = GoalDriftComputer(config)
        computer.set_reference(reference_texts)
        m = computer.compute(generation=1, generated_texts=reference_texts)
        assert isinstance(m, GoalDriftMeasurement)
        assert m.goal_drift_index == pytest.approx(0.0, abs=1e-6)
        assert m.semantic_drift == pytest.approx(0.0, abs=1e-6)
        assert m.alert is False

    def test_very_different_texts_give_positive_gdi(
        self, config: dict, reference_texts: list[str]
    ) -> None:
        """When generated texts are very different, GDI should be > 0."""
        computer = GoalDriftComputer(config)
        computer.set_reference(reference_texts)
        different = [
            "Mathematics is the queen of all sciences.",
            "Calculus provides tools for understanding change.",
            "Abstract algebra studies algebraic structures.",
        ]
        m = computer.compute(generation=1, generated_texts=different)
        assert m.goal_drift_index > 0.0
        assert m.semantic_drift > 0.0
        assert m.lexical_drift > 0.0

    def test_compute_without_reference_raises(self, config: dict) -> None:
        """compute() before set_reference() should raise."""
        computer = GoalDriftComputer(config)
        with pytest.raises(RuntimeError, match="set_reference"):
            computer.compute(1, ["hello"])


class TestGoalDriftWeights:
    """Verify that weights affect the composite GDI."""

    def test_weights_are_respected(self, reference_texts: list[str]) -> None:
        """Changing weights should change the composite GDI."""
        cfg_a = {
            "safety": {
                "weights": {"semantic": 1.0, "lexical": 0.0, "structural": 0.0, "distributional": 0.0},
                "alert_threshold_drift_cosine": 0.15,
            },
        }
        cfg_b = {
            "safety": {
                "weights": {"semantic": 0.0, "lexical": 1.0, "structural": 0.0, "distributional": 0.0},
                "alert_threshold_drift_cosine": 0.15,
            },
        }
        different = [
            "Mathematics is the queen of all sciences.",
            "Calculus provides tools for understanding change.",
        ]
        comp_a = GoalDriftComputer(cfg_a)
        comp_a.set_reference(reference_texts)
        m_a = comp_a.compute(1, different)

        comp_b = GoalDriftComputer(cfg_b)
        comp_b.set_reference(reference_texts)
        m_b = comp_b.compute(1, different)

        # The GDI values should differ because the weights emphasise different sub-metrics
        assert m_a.goal_drift_index != pytest.approx(m_b.goal_drift_index, abs=1e-6)


class TestGoalDriftAlert:
    """Alert triggering tests."""

    def test_alert_triggers_above_threshold(self, reference_texts: list[str]) -> None:
        """GDI above threshold should set alert=True."""
        config = {
            "safety": {
                "weights": {"semantic": 0.3, "lexical": 0.2, "structural": 0.2, "distributional": 0.3},
                "alert_threshold_drift_cosine": 0.01,  # very low threshold
            },
        }
        computer = GoalDriftComputer(config)
        computer.set_reference(reference_texts)
        different = [
            "Mathematics is the queen of all sciences.",
            "Calculus provides tools for understanding change.",
        ]
        m = computer.compute(1, different)
        assert m.alert is True

    def test_no_alert_below_threshold(self, config: dict, reference_texts: list[str]) -> None:
        """Identical texts should not trigger an alert."""
        computer = GoalDriftComputer(config)
        computer.set_reference(reference_texts)
        m = computer.compute(1, reference_texts)
        assert m.alert is False


class TestGoalDriftTrajectory:
    """Trajectory accumulation tests."""

    def test_trajectory_accumulates(self, config: dict, reference_texts: list[str]) -> None:
        """Successive calls to compute() should append to the trajectory."""
        computer = GoalDriftComputer(config)
        computer.set_reference(reference_texts)

        for gen in range(5):
            computer.compute(gen, reference_texts)

        traj = computer.get_trajectory()
        assert len(traj) == 5
        assert [m.generation for m in traj] == list(range(5))

    def test_trajectory_is_copy(self, config: dict, reference_texts: list[str]) -> None:
        """get_trajectory() should return a copy, not the internal list."""
        computer = GoalDriftComputer(config)
        computer.set_reference(reference_texts)
        computer.compute(0, reference_texts)
        traj = computer.get_trajectory()
        traj.clear()
        assert len(computer.get_trajectory()) == 1


class TestGoalDriftDistributional:
    """Tests involving token-level distributional drift."""

    def test_distributional_drift_with_identical_dist(
        self, config: dict, reference_texts: list[str]
    ) -> None:
        """Identical token distributions should give distributional_drift ~ 0."""
        dist = np.array([0.1, 0.2, 0.3, 0.4])
        computer = GoalDriftComputer(config)
        computer.set_reference(reference_texts, reference_distribution=dist)
        m = computer.compute(1, reference_texts, token_distribution=dist)
        assert m.distributional_drift == pytest.approx(0.0, abs=1e-6)

    def test_distributional_drift_with_different_dist(
        self, config: dict, reference_texts: list[str]
    ) -> None:
        """Different token distributions should give positive distributional_drift."""
        ref_dist = np.array([0.25, 0.25, 0.25, 0.25])
        cur_dist = np.array([0.9, 0.03, 0.03, 0.04])
        computer = GoalDriftComputer(config)
        computer.set_reference(reference_texts, reference_distribution=ref_dist)
        m = computer.compute(1, reference_texts, token_distribution=cur_dist)
        assert m.distributional_drift > 0.0
