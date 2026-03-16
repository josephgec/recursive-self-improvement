"""Shared test fixtures for GDI tests."""

import json
import os
from typing import Dict, List

import pytest

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def reference_data() -> Dict:
    """Load reference outputs fixture."""
    with open(os.path.join(FIXTURES_DIR, "reference_outputs.json")) as f:
        return json.load(f)


@pytest.fixture
def drifted_data() -> Dict:
    """Load drifted outputs fixture."""
    with open(os.path.join(FIXTURES_DIR, "drifted_outputs.json")) as f:
        return json.load(f)


@pytest.fixture
def collapsed_data() -> Dict:
    """Load collapsed outputs fixture."""
    with open(os.path.join(FIXTURES_DIR, "collapsed_outputs.json")) as f:
        return json.load(f)


@pytest.fixture
def reference_outputs(reference_data) -> List[str]:
    """Reference output strings."""
    return reference_data["outputs"]


@pytest.fixture
def drifted_outputs(drifted_data) -> List[str]:
    """Drifted output strings."""
    return drifted_data["outputs"]


@pytest.fixture
def collapsed_outputs(collapsed_data) -> List[str]:
    """Collapsed output strings."""
    return collapsed_data["outputs"]


@pytest.fixture
def probe_tasks() -> List[str]:
    """Standard probe tasks."""
    return [
        "Explain the concept of recursion in programming.",
        "What are the benefits of test-driven development?",
        "Describe how a hash table works internally.",
        "What is the difference between a stack and a queue?",
        "Explain the observer pattern in software design.",
    ]


class MockAgent:
    """Mock agent that returns predefined outputs."""

    def __init__(self, outputs: List[str]):
        self._outputs = outputs
        self._call_count = 0

    def run(self, task: str) -> str:
        idx = self._call_count % len(self._outputs)
        self._call_count += 1
        return self._outputs[idx]


class DriftingAgent:
    """Mock agent that gradually drifts from reference behavior."""

    def __init__(self, reference_outputs: List[str], drifted_outputs: List[str]):
        self._reference = reference_outputs
        self._drifted = drifted_outputs
        self._drift_level = 0.0
        self._call_count = 0

    def set_drift(self, level: float) -> None:
        """Set drift level (0.0 = reference, 1.0 = fully drifted)."""
        self._drift_level = max(0.0, min(1.0, level))

    def run(self, task: str) -> str:
        idx = self._call_count % len(self._reference)
        self._call_count += 1
        if self._drift_level < 0.5:
            return self._reference[idx]
        return self._drifted[idx]


@pytest.fixture
def mock_agent(reference_outputs):
    """Mock agent returning reference outputs."""
    return MockAgent(reference_outputs)


@pytest.fixture
def drifted_agent(reference_outputs, drifted_outputs):
    """Mock agent that can drift."""
    return DriftingAgent(reference_outputs, drifted_outputs)
