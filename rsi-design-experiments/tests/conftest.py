"""Shared test fixtures and mock pipeline."""

import pytest

from src.harness.controlled_pipeline import MockPipeline, ControlledPipeline
from src.experiments.modification_frequency import ModificationFrequencyExperiment
from src.experiments.hindsight_target import HindsightTargetExperiment
from src.experiments.rlm_depth import RLMDepthExperiment
from src.experiments.base import ExperimentResult, ConditionResult
from src.conditions.frequency_conditions import ModificationFrequencyPolicy
from src.conditions.hindsight_conditions import HindsightTargetPolicy


@pytest.fixture
def mock_pipeline():
    """Return a fresh MockPipeline with seed 42."""
    return MockPipeline(seed=42)


@pytest.fixture
def controlled_pipeline():
    """Return a ControlledPipeline."""
    return ControlledPipeline(seed=42)


@pytest.fixture
def frequency_experiment():
    """Return a ModificationFrequencyExperiment with debug config."""
    return ModificationFrequencyExperiment({"iterations_per_condition": 10})


@pytest.fixture
def hindsight_experiment():
    """Return a HindsightTargetExperiment with debug config."""
    return HindsightTargetExperiment({"iterations_per_condition": 10})


@pytest.fixture
def depth_experiment():
    """Return an RLMDepthExperiment with debug config."""
    return RLMDepthExperiment({"iterations_per_condition": 10})


@pytest.fixture
def sample_eval_tasks():
    """Return sample evaluation tasks for testing."""
    return [
        {"id": i, "type": "math" if i % 3 == 0 else "code" if i % 3 == 1 else "reasoning"}
        for i in range(20)
    ]


@pytest.fixture
def sample_experiment_result():
    """Create a sample ExperimentResult for testing analysis."""
    result = ExperimentResult(
        experiment_name="test_experiment",
        conditions=["cond_a", "cond_b", "cond_c"],
        repetitions=3,
    )
    result.per_condition_results = {
        "cond_a": [
            ConditionResult(
                condition_name="cond_a",
                final_accuracy=0.85,
                stability_score=0.9,
                total_cost=1.0,
                composite_score=0.80,
                generalization_score=0.8,
                accuracy_trajectory=[0.5, 0.6, 0.7, 0.8, 0.85],
            ),
            ConditionResult(
                condition_name="cond_a",
                final_accuracy=0.87,
                stability_score=0.88,
                total_cost=1.1,
                composite_score=0.82,
                generalization_score=0.82,
                accuracy_trajectory=[0.5, 0.62, 0.72, 0.82, 0.87],
            ),
            ConditionResult(
                condition_name="cond_a",
                final_accuracy=0.83,
                stability_score=0.92,
                total_cost=0.9,
                composite_score=0.78,
                generalization_score=0.78,
                accuracy_trajectory=[0.5, 0.58, 0.68, 0.78, 0.83],
            ),
        ],
        "cond_b": [
            ConditionResult(
                condition_name="cond_b",
                final_accuracy=0.75,
                stability_score=0.95,
                total_cost=0.5,
                composite_score=0.70,
                generalization_score=0.7,
                accuracy_trajectory=[0.5, 0.55, 0.6, 0.7, 0.75],
            ),
            ConditionResult(
                condition_name="cond_b",
                final_accuracy=0.73,
                stability_score=0.93,
                total_cost=0.6,
                composite_score=0.68,
                generalization_score=0.68,
                accuracy_trajectory=[0.5, 0.53, 0.58, 0.68, 0.73],
            ),
            ConditionResult(
                condition_name="cond_b",
                final_accuracy=0.77,
                stability_score=0.97,
                total_cost=0.4,
                composite_score=0.72,
                generalization_score=0.72,
                accuracy_trajectory=[0.5, 0.57, 0.62, 0.72, 0.77],
            ),
        ],
        "cond_c": [
            ConditionResult(
                condition_name="cond_c",
                final_accuracy=0.60,
                stability_score=0.98,
                total_cost=0.2,
                composite_score=0.55,
                generalization_score=0.55,
                accuracy_trajectory=[0.5, 0.52, 0.55, 0.58, 0.60],
            ),
            ConditionResult(
                condition_name="cond_c",
                final_accuracy=0.62,
                stability_score=0.96,
                total_cost=0.25,
                composite_score=0.57,
                generalization_score=0.57,
                accuracy_trajectory=[0.5, 0.54, 0.57, 0.60, 0.62],
            ),
            ConditionResult(
                condition_name="cond_c",
                final_accuracy=0.58,
                stability_score=0.99,
                total_cost=0.15,
                composite_score=0.53,
                generalization_score=0.53,
                accuracy_trajectory=[0.5, 0.51, 0.53, 0.56, 0.58],
            ),
        ],
    }
    return result


@pytest.fixture
def sample_nonsignificant_result():
    """Create a result where conditions are very similar (non-significant)."""
    result = ExperimentResult(
        experiment_name="nonsignificant_test",
        conditions=["cond_x", "cond_y"],
        repetitions=3,
    )
    result.per_condition_results = {
        "cond_x": [
            ConditionResult(condition_name="cond_x", composite_score=0.80),
            ConditionResult(condition_name="cond_x", composite_score=0.81),
            ConditionResult(condition_name="cond_x", composite_score=0.79),
        ],
        "cond_y": [
            ConditionResult(condition_name="cond_y", composite_score=0.80),
            ConditionResult(condition_name="cond_y", composite_score=0.80),
            ConditionResult(condition_name="cond_y", composite_score=0.81),
        ],
    }
    return result
