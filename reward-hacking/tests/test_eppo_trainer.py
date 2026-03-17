"""Tests for EPPO trainer."""

import numpy as np
import pytest

from src.eppo.config import EPPOConfig
from src.eppo.trainer import EPPOTrainer, EPPOStepResult, EPPOEpochResult


class TestEPPOTrainer:
    """Test EPPO training functionality."""

    def test_train_step_returns_result(self, eppo_trainer, sample_batch):
        """A single train step returns an EPPOStepResult."""
        result = eppo_trainer.train_step(sample_batch)
        assert isinstance(result, EPPOStepResult)
        assert result.step == 1

    def test_train_step_combined_loss(self, eppo_trainer, sample_batch):
        """Combined loss = policy_loss - beta * entropy."""
        result = eppo_trainer.train_step(sample_batch)
        expected = result.policy_loss - result.entropy_bonus
        assert abs(result.combined_loss - expected) < 1e-6

    def test_train_10_steps(self, eppo_trainer, sample_batch):
        """Training 10 steps increments global step."""
        for i in range(10):
            result = eppo_trainer.train_step(sample_batch)
            assert result.step == i + 1
        assert eppo_trainer.global_step == 10

    def test_entropy_stays_above_min(self, eppo_trainer):
        """Entropy stays above a reasonable minimum over 10 steps."""
        min_entropy = float("inf")
        for _ in range(10):
            result = eppo_trainer.train_step(eppo_trainer._make_random_batch())
            min_entropy = min(min_entropy, result.entropy)
        # Entropy should be positive for a softmax over vocab
        assert min_entropy > 0

    def test_train_epoch(self, eppo_trainer):
        """train_epoch produces an EPPOEpochResult with summaries."""
        result = eppo_trainer.train_epoch(num_steps=5)
        assert isinstance(result, EPPOEpochResult)
        assert result.epoch == 0
        assert len(result.steps) == 5
        assert result.mean_entropy > 0
        assert result.final_beta > 0

    def test_multiple_epochs(self, eppo_trainer):
        """Multiple epochs increment correctly."""
        for i in range(3):
            result = eppo_trainer.train_epoch(num_steps=5)
            assert result.epoch == i
        assert len(eppo_trainer.epoch_results) == 3
        assert eppo_trainer.global_step == 15

    def test_beta_decays(self, eppo_trainer):
        """Beta decays over steps in coefficient mode."""
        initial_beta = eppo_trainer.entropy_bonus.current_beta
        eppo_trainer.train_epoch(num_steps=10)
        final_beta = eppo_trainer.entropy_bonus.current_beta
        assert final_beta < initial_beta

    def test_training_summary(self, eppo_trainer):
        """get_training_summary returns correct metrics."""
        eppo_trainer.train_epoch(num_steps=5)
        summary = eppo_trainer.get_training_summary()
        assert summary["epochs"] == 1
        assert summary["total_steps"] == 5
        assert "final_entropy" in summary
        assert "entropy_history" in summary

    def test_custom_data(self, eppo_trainer, sample_batch):
        """Training with custom data works."""
        data = [sample_batch] * 3
        result = eppo_trainer.train_epoch(data=data)
        assert len(result.steps) == 3

    def test_config_validation(self):
        """Config validation catches errors."""
        config = EPPOConfig(learning_rate=-1, clip_epsilon=2.0)
        errors = config.validate()
        assert len(errors) >= 2

    def test_config_from_dict(self):
        """Config from_dict ignores unknown keys."""
        d = {"learning_rate": 0.01, "unknown_key": 42}
        config = EPPOConfig.from_dict(d)
        assert config.learning_rate == 0.01

    def test_target_mode_training(self):
        """Target mode adjusts beta to reach entropy target."""
        config = EPPOConfig(
            entropy_mode="target",
            entropy_target=3.0,
            entropy_coeff=0.01,
            adaptive_alpha=0.1,
            hidden_dim=32,
            vocab_size=50,
        )
        trainer = EPPOTrainer(config)
        initial_beta = trainer.entropy_bonus.current_beta

        for _ in range(20):
            trainer.train_step(trainer._make_random_batch())

        # Beta should have changed from initial in target mode
        final_beta = trainer.entropy_bonus.current_beta
        assert final_beta != initial_beta
