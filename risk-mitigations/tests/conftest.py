"""Shared fixtures for risk mitigation tests.

Provides MockAgent, sample metrics histories, and sample candidates.
All deterministic, no external dependencies.
"""

import pytest
import sys
import os

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class MockAgent:
    """Mock agent for testing adversarial evaluations and staging."""

    def __init__(self, base_score: float = 0.8, fail_tasks: list = None):
        self.base_score = base_score
        self.fail_tasks = set(fail_tasks or [])
        self.state = {"quality": base_score, "modifier": 0.0}

    def score(self, task):
        """Return a deterministic score for a task."""
        task_id = task.get("id", "")
        if task_id in self.fail_tasks:
            return 0.3  # Fail
        return self.base_score


@pytest.fixture
def mock_agent():
    """A mock agent that passes most tasks."""
    return MockAgent(base_score=0.8)


@pytest.fixture
def mock_agent_failing():
    """A mock agent that fails several adversarial tasks."""
    return MockAgent(
        base_score=0.4,
        fail_tasks=["adv_001", "adv_002", "adv_003", "adv_004", "adv_005"],
    )


@pytest.fixture
def healthy_metrics_history():
    """Metrics history showing healthy model (stable entropy, low KL)."""
    return {
        "entropy": [4.0, 3.95, 4.0, 3.98, 4.02, 3.99, 4.01, 3.97, 4.0, 3.98],
        "kl_divergence": [0.1, 0.12, 0.11, 0.13, 0.1, 0.12, 0.11, 0.12, 0.1, 0.11],
        "quality_score": [0.85, 0.86, 0.85, 0.87, 0.86, 0.85, 0.86, 0.87, 0.88, 0.87],
    }


@pytest.fixture
def collapsing_metrics_history():
    """Metrics history showing model collapse (decreasing entropy, rising KL)."""
    return {
        "entropy": [4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.1],
        "kl_divergence": [0.1, 0.3, 0.6, 1.0, 1.5, 2.5, 4.0, 6.0, 10.0],
        "quality_score": [0.85, 0.83, 0.80, 0.75, 0.70, 0.60, 0.45, 0.30, 0.15],
    }


@pytest.fixture
def sudden_collapse_history():
    """Metrics history with sudden collapse (stable then sharp drop)."""
    return {
        "entropy": [4.0, 3.9, 4.0, 3.9, 4.0, 1.0],
        "kl_divergence": [0.1, 0.1, 0.1, 0.1, 0.1, 8.0],
        "quality_score": [0.85, 0.86, 0.85, 0.86, 0.85, 0.30],
    }


@pytest.fixture
def sample_good_candidate():
    """A modification candidate that improves the agent."""
    return {
        "id": "good_candidate_001",
        "changes": {"modifier": 0.05, "version": "1.1"},
        "expected_improvement": 0.05,
    }


@pytest.fixture
def sample_bad_candidate():
    """A modification candidate that degrades the agent."""
    return {
        "id": "bad_candidate_001",
        "changes": {"modifier": -0.5, "version": "1.1-bad"},
        "expected_improvement": -0.5,
    }


@pytest.fixture
def sample_unsafe_candidate():
    """A modification candidate with unsafe changes."""
    return {
        "id": "unsafe_candidate_001",
        "changes": {"__internal": "hack", "destroy": True},
    }


@pytest.fixture
def cost_history():
    """Sample cost history for forecasting."""
    return [2.5, 3.0, 2.8, 3.2, 2.9, 3.1, 2.7, 3.3, 3.0, 2.8]


@pytest.fixture
def constraint_history():
    """Sample constraint binding history."""
    return [
        {"value": 0.88, "threshold": 0.90, "performance": 0.75},
        {"value": 0.91, "threshold": 0.90, "performance": 0.80},
        {"value": 0.89, "threshold": 0.90, "performance": 0.72},
        {"value": 0.92, "threshold": 0.90, "performance": 0.82},
        {"value": 0.90, "threshold": 0.90, "performance": 0.78},
        {"value": 0.93, "threshold": 0.90, "performance": 0.85},
        {"value": 0.91, "threshold": 0.90, "performance": 0.79},
        {"value": 0.89, "threshold": 0.90, "performance": 0.74},
        {"value": 0.90, "threshold": 0.90, "performance": 0.77},
        {"value": 0.92, "threshold": 0.90, "performance": 0.81},
    ]


@pytest.fixture
def experiment_status():
    """Sample experiment status for readiness checking."""
    return {
        "completed": 8,
        "total": 10,
        "results_quality": 0.85,
    }


@pytest.fixture
def writing_status():
    """Sample writing status for readiness checking."""
    return {
        "sections_done": 5,
        "total_sections": 7,
        "quality": 0.75,
    }
