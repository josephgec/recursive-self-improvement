"""Test fixtures for ablation studies."""

import sys
import os
import pytest

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.suites.base import AblationCondition, AblationSuiteResult, ConditionRun
from src.suites.neurosymbolic import NeurosymbolicAblation
from src.suites.godel import GodelAgentAblation
from src.suites.soar import SOARAblation
from src.suites.rlm import RLMAblation
from src.execution.runner import AblationRunner, MockPipeline, MOCK_ACCURACY_MAP


@pytest.fixture
def mock_pipeline():
    """Create a MockPipeline with default accuracy map."""
    return MockPipeline(noise_std=0.02)


@pytest.fixture
def runner(mock_pipeline):
    """Create an AblationRunner with the mock pipeline."""
    return AblationRunner(pipeline_runner=mock_pipeline)


@pytest.fixture
def neurosymbolic_suite():
    return NeurosymbolicAblation()


@pytest.fixture
def godel_suite():
    return GodelAgentAblation()


@pytest.fixture
def soar_suite():
    return SOARAblation()


@pytest.fixture
def rlm_suite():
    return RLMAblation()


@pytest.fixture
def all_suites():
    return [
        NeurosymbolicAblation(),
        GodelAgentAblation(),
        SOARAblation(),
        RLMAblation(),
    ]


@pytest.fixture
def sample_conditions():
    """Sample conditions for testing."""
    return [
        AblationCondition(name="full", description="Full pipeline", is_full=True),
        AblationCondition(name="ablated_a", description="Without A"),
        AblationCondition(name="ablated_b", description="Without B"),
    ]


@pytest.fixture
def sample_result():
    """Create a sample AblationSuiteResult for testing."""
    result = AblationSuiteResult(suite_name="Test Suite")
    # Full condition: high accuracy
    result.condition_runs["full"] = [
        ConditionRun(condition_name="full", repetition=i, accuracy=0.85 + 0.01 * (i - 2), seed=42 + i)
        for i in range(5)
    ]
    # Ablated condition: lower accuracy
    result.condition_runs["ablated"] = [
        ConditionRun(condition_name="ablated", repetition=i, accuracy=0.72 + 0.01 * (i - 2), seed=42 + i)
        for i in range(5)
    ]
    return result


@pytest.fixture
def neurosymbolic_result(runner, neurosymbolic_suite):
    """Run neurosymbolic suite and return result."""
    return runner.run_suite(neurosymbolic_suite, repetitions=5, seed=42)


@pytest.fixture
def godel_result(runner, godel_suite):
    """Run godel suite and return result."""
    return runner.run_suite(godel_suite, repetitions=5, seed=42)


@pytest.fixture
def soar_result(runner, soar_suite):
    """Run SOAR suite and return result."""
    return runner.run_suite(soar_suite, repetitions=5, seed=42)


@pytest.fixture
def rlm_result(runner, rlm_suite):
    """Run RLM suite and return result."""
    return runner.run_suite(rlm_suite, repetitions=5, seed=42)
