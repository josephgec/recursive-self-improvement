"""Shared test fixtures for godel-fragility tests."""

from __future__ import annotations

import os
import sys
import tempfile
import textwrap
from typing import Any, Dict, List

import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.adversarial.scenario_registry import AdversarialScenario, ScenarioRegistry
from src.harness.controlled_env import MockAgent


# ------------------------------------------------------------------ #
# Mock Agent Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture
def mock_agent() -> MockAgent:
    """Create a fresh MockAgent for testing."""
    return MockAgent(seed=42)


@pytest.fixture
def mock_agent_with_functions() -> MockAgent:
    """Create a MockAgent with multiple installed functions."""
    agent = MockAgent(seed=42)
    agent.install_function(
        "add",
        textwrap.dedent("""\
            def add(a, b):
                return a + b
        """),
    )
    agent.install_function(
        "multiply",
        textwrap.dedent("""\
            def multiply(a, b):
                result = 0
                for i in range(b):
                    result = result + a
                return result
        """),
    )
    agent.install_function(
        "classify",
        textwrap.dedent("""\
            def classify(x):
                if x > 100:
                    return "high"
                elif x > 50:
                    return "medium"
                elif x > 0:
                    return "low"
                else:
                    return "negative"
        """),
    )
    return agent


# ------------------------------------------------------------------ #
# Sample Code Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture
def simple_code() -> str:
    """Simple Python function for testing."""
    return textwrap.dedent("""\
        def solve(x):
            if x > 0:
                return x * 2
            else:
                return 0
    """)


@pytest.fixture
def complex_code() -> str:
    """More complex Python code for testing."""
    return textwrap.dedent("""\
        def process(data):
            result = []
            total = 0
            for item in data:
                if item > 0:
                    value = item * 2
                    if value > 100:
                        value = 100
                    result.append(value)
                    total = total + value
                else:
                    result.append(0)
            if len(result) > 0:
                average = total / len(result)
            else:
                average = 0
            return result, average

        def validate(result, expected):
            if len(result) != len(expected):
                return False
            correct = 0
            for i in range(len(result)):
                if result[i] == expected[i]:
                    correct = correct + 1
            accuracy = correct / len(result) if len(result) > 0 else 0
            return accuracy >= 0.8
    """)


# ------------------------------------------------------------------ #
# Scenario Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture
def sample_scenario() -> AdversarialScenario:
    """Create a simple test scenario."""

    def setup(agent: Any) -> None:
        agent.set_task("Test task")

    def success_criteria(agent: Any) -> bool:
        return agent.get_accuracy() >= 0.5

    return AdversarialScenario(
        name="test_scenario",
        category="test",
        description="A test scenario",
        severity=3,
        setup=setup,
        expected_failure_mode="STAGNATION",
        recovery_expected=True,
        max_iterations=10,
        success_criteria=success_criteria,
    )


@pytest.fixture
def sample_registry(sample_scenario: AdversarialScenario) -> ScenarioRegistry:
    """Create a registry with a few test scenarios."""
    registry = ScenarioRegistry()
    registry.register(sample_scenario)

    # Add a second scenario
    def setup2(agent: Any) -> None:
        agent.set_task("Harder task")

    registry.register(
        AdversarialScenario(
            name="hard_scenario",
            category="test",
            description="A harder test scenario",
            severity=5,
            setup=setup2,
            expected_failure_mode="STATE_CORRUPTION",
            recovery_expected=False,
            max_iterations=5,
        )
    )

    return registry


# ------------------------------------------------------------------ #
# Temp Directory Fixture
# ------------------------------------------------------------------ #


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for test outputs."""
    return str(tmp_path)
