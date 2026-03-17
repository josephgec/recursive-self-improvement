"""Shared test fixtures for reward hacking tests."""

import numpy as np
import pytest

from src.eppo.config import EPPOConfig
from src.eppo.policy import MockPolicy
from src.eppo.value_head import MockValueHead
from src.eppo.trainer import EPPOTrainer


class MockModel:
    """Simple mock model for testing."""

    def __init__(self, input_dim: int = 64, output_dim: int = 100, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = self.rng.randn(input_dim, output_dim) * 0.1

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return x @ self.weights

    def get_activations(self, num_layers: int = 6) -> list[np.ndarray]:
        return [self.rng.randn(self.input_dim) for _ in range(num_layers)]


@pytest.fixture
def mock_model():
    """Provide a mock model."""
    return MockModel()


@pytest.fixture
def eppo_config():
    """Provide default EPPO config."""
    return EPPOConfig(
        learning_rate=0.001,
        entropy_coeff=0.01,
        entropy_mode="coefficient",
        decay_rate=0.99,
        min_beta=0.001,
        epochs=5,
        batch_size=16,
        hidden_dim=32,
        vocab_size=50,
    )


@pytest.fixture
def eppo_trainer(eppo_config):
    """Provide configured EPPO trainer."""
    return EPPOTrainer(eppo_config)


@pytest.fixture
def mock_policy():
    """Provide a mock policy."""
    return MockPolicy(input_dim=32, vocab_size=50, seed=42)


@pytest.fixture
def mock_value_head():
    """Provide a mock value head."""
    return MockValueHead(input_dim=32, seed=42)


@pytest.fixture
def sample_batch():
    """Provide a sample training batch."""
    rng = np.random.RandomState(42)
    bs = 16
    dim = 32
    vocab = 50
    return {
        "states": rng.randn(bs, dim).astype(np.float32),
        "actions": rng.randint(0, vocab, size=bs),
        "rewards": rng.randn(bs).astype(np.float32),
        "old_log_probs": rng.randn(bs, vocab).astype(np.float32) * 0.1,
    }


@pytest.fixture
def sample_rewards():
    """Provide sample reward sequences."""
    rng = np.random.RandomState(42)
    return rng.randn(100).astype(np.float64)


@pytest.fixture
def sample_accuracies():
    """Provide sample accuracy sequences."""
    rng = np.random.RandomState(42)
    return 0.5 + rng.randn(100) * 0.1


@pytest.fixture
def sample_activations():
    """Provide sample layer activations."""
    rng = np.random.RandomState(42)
    return [rng.randn(64) for _ in range(6)]


@pytest.fixture
def declining_activations():
    """Provide activations that decline over time."""
    rng = np.random.RandomState(42)
    history = []
    for step in range(30):
        scale = max(0.1, 1.0 - 0.03 * step)
        acts = [rng.randn(64) * scale for _ in range(6)]
        history.append(acts)
    return history


@pytest.fixture
def green_histories():
    """Provide histories that produce all-green safety package."""
    return {
        "gdi": {
            "governance_review": True,
            "deployment_checks_passed": True,
            "impact_assessed": True,
        },
        "constraint": {
            "reward_bounded": True,
            "entropy_above_min": True,
            "energy_stable": True,
        },
        "interp": {
            "energy_interpretable": True,
            "homogenization_checked": True,
            "activations_tracked": True,
        },
        "reward": {
            "no_divergence": True,
            "no_shortcuts": True,
            "no_gaming": True,
        },
    }


@pytest.fixture
def failing_histories():
    """Provide histories with unresolved failures."""
    return {
        "gdi": {
            "governance_review": True,
            "deployment_checks_passed": True,
            "impact_assessed": True,
            "unresolved_failures": ["safety_review_pending"],
        },
        "constraint": {
            "reward_bounded": False,
            "entropy_above_min": True,
            "energy_stable": False,
            "unresolved_failures": ["energy_collapse"],
        },
        "interp": {
            "energy_interpretable": False,
            "homogenization_checked": True,
            "activations_tracked": True,
            "unresolved_failures": ["interpretation_gap"],
        },
        "reward": {
            "no_divergence": False,
            "no_shortcuts": False,
            "no_gaming": True,
            "unresolved_failures": ["reward_exploit"],
        },
    }
