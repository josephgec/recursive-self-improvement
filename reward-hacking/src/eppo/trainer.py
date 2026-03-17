from __future__ import annotations

"""EPPO Trainer: Entropy-Penalized Proximal Policy Optimization."""

import numpy as np
from dataclasses import dataclass, field

from .config import EPPOConfig
from .entropy_bonus import EntropyBonus
from .policy import MockPolicy
from .value_head import MockValueHead


@dataclass
class EPPOStepResult:
    """Result from a single training step."""

    policy_loss: float
    value_loss: float
    entropy: float
    entropy_bonus: float
    combined_loss: float  # policy_loss - beta * entropy
    beta: float
    step: int
    clipped_fraction: float


@dataclass
class EPPOEpochResult:
    """Result from a training epoch."""

    epoch: int
    steps: list[EPPOStepResult] = field(default_factory=list)
    mean_policy_loss: float = 0.0
    mean_value_loss: float = 0.0
    mean_entropy: float = 0.0
    mean_combined_loss: float = 0.0
    final_beta: float = 0.0

    def compute_summaries(self) -> None:
        """Compute summary statistics from step results."""
        if not self.steps:
            return
        self.mean_policy_loss = float(np.mean([s.policy_loss for s in self.steps]))
        self.mean_value_loss = float(np.mean([s.value_loss for s in self.steps]))
        self.mean_entropy = float(np.mean([s.entropy for s in self.steps]))
        self.mean_combined_loss = float(np.mean([s.combined_loss for s in self.steps]))
        self.final_beta = self.steps[-1].beta


class EPPOTrainer:
    """Entropy-Penalized PPO Trainer.

    Combines standard PPO with an entropy bonus that either decays
    over time (coefficient mode) or adapts to maintain a target
    entropy level (target mode).
    """

    def __init__(self, config: EPPOConfig | None = None):
        self.config = config or EPPOConfig()
        self.policy = MockPolicy(
            input_dim=self.config.hidden_dim,
            vocab_size=self.config.vocab_size,
        )
        self.value_head = MockValueHead(input_dim=self.config.hidden_dim)
        self.entropy_bonus = EntropyBonus(
            initial_beta=self.config.entropy_coeff,
            mode=self.config.entropy_mode,
            decay_rate=self.config.decay_rate,
            min_beta=self.config.min_beta,
            entropy_target=self.config.entropy_target,
            adaptive_alpha=self.config.adaptive_alpha,
        )
        self._global_step = 0
        self._epoch_results: list[EPPOEpochResult] = []
        self._rng = np.random.RandomState(42)

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def epoch_results(self) -> list[EPPOEpochResult]:
        return list(self._epoch_results)

    def train_step(self, batch: dict) -> EPPOStepResult:
        """Execute a single training step.

        Args:
            batch: dict with keys 'states', 'actions', 'rewards', 'old_log_probs'
                   All values are numpy arrays.

        Returns:
            EPPOStepResult with loss components.
        """
        states = batch["states"]
        rewards = batch["rewards"]
        old_log_probs = batch["old_log_probs"]

        # Forward pass through policy
        logits = self.policy.forward(states)
        entropy = self.policy.compute_entropy(logits)
        new_log_probs = self.policy.get_log_probs(logits)

        # Compute PPO surrogate loss
        # Use mean log prob as approximation for mock
        if new_log_probs.ndim > 1:
            new_lp = new_log_probs.mean(axis=-1)
        else:
            new_lp = new_log_probs.mean()

        if old_log_probs.ndim > 1:
            old_lp = old_log_probs.mean(axis=-1)
        else:
            old_lp = old_log_probs.mean()

        ratio = np.exp(new_lp - old_lp)

        # Compute advantages from rewards and value predictions
        values = self.value_head.predict_batch(states) if states.ndim > 1 else np.array([self.value_head.predict_value(states)])
        advantages = rewards - values[:len(rewards)]

        # PPO clipped objective
        surr1 = ratio * advantages
        surr2 = np.clip(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
        policy_loss = float(-np.mean(np.minimum(surr1, surr2)))

        # Value loss
        value_loss = float(np.mean((values[:len(rewards)] - rewards) ** 2))

        # Entropy bonus
        beta = self.entropy_bonus.current_beta
        entropy_bonus_val = self.entropy_bonus.compute(entropy)

        # Combined loss: PPO loss - beta * entropy (we want to maximize entropy)
        combined_loss = policy_loss - entropy_bonus_val

        # Clipped fraction
        clipped = np.abs(ratio - 1.0) > self.config.clip_epsilon
        clipped_fraction = float(np.mean(clipped)) if isinstance(clipped, np.ndarray) else float(clipped)

        # Mock gradient update
        self.policy.perturb_weights(scale=self.config.learning_rate)

        # Step entropy bonus
        self.entropy_bonus.step(entropy)

        self._global_step += 1

        return EPPOStepResult(
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy=entropy,
            entropy_bonus=entropy_bonus_val,
            combined_loss=combined_loss,
            beta=beta,
            step=self._global_step,
            clipped_fraction=clipped_fraction,
        )

    def train_epoch(self, data: list[dict] | None = None, num_steps: int = 10) -> EPPOEpochResult:
        """Train for one epoch.

        Args:
            data: Optional list of batches. If None, generates random batches.
            num_steps: Number of steps if data is None.

        Returns:
            EPPOEpochResult with summary statistics.
        """
        epoch_num = len(self._epoch_results)
        result = EPPOEpochResult(epoch=epoch_num)

        if data is None:
            data = [self._make_random_batch() for _ in range(num_steps)]

        for batch in data:
            step_result = self.train_step(batch)
            result.steps.append(step_result)

        result.compute_summaries()
        self._epoch_results.append(result)
        return result

    def _make_random_batch(self) -> dict:
        """Generate a random training batch."""
        bs = self.config.batch_size
        dim = self.config.hidden_dim
        return {
            "states": self._rng.randn(bs, dim).astype(np.float32),
            "actions": self._rng.randint(0, self.config.vocab_size, size=bs),
            "rewards": self._rng.randn(bs).astype(np.float32),
            "old_log_probs": self._rng.randn(bs, self.config.vocab_size).astype(np.float32) * 0.1,
        }

    def get_training_summary(self) -> dict:
        """Get summary of all training so far."""
        if not self._epoch_results:
            return {"epochs": 0, "total_steps": 0}

        return {
            "epochs": len(self._epoch_results),
            "total_steps": self._global_step,
            "final_entropy": self._epoch_results[-1].mean_entropy,
            "final_beta": self._epoch_results[-1].final_beta,
            "final_policy_loss": self._epoch_results[-1].mean_policy_loss,
            "final_combined_loss": self._epoch_results[-1].mean_combined_loss,
            "entropy_history": [e.mean_entropy for e in self._epoch_results],
            "beta_history": [e.final_beta for e in self._epoch_results],
        }
