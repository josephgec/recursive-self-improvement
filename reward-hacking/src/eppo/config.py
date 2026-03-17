from __future__ import annotations

"""EPPO configuration dataclass."""

from dataclasses import dataclass, field


@dataclass
class EPPOConfig:
    """Configuration for Entropy-Penalized PPO training."""

    learning_rate: float = 0.0003
    entropy_coeff: float = 0.01
    entropy_target: float = 1.5
    entropy_mode: str = "coefficient"  # "coefficient" or "target"
    decay_rate: float = 0.995
    min_beta: float = 0.001
    clip_epsilon: float = 0.2
    epochs: int = 10
    batch_size: int = 32
    adaptive_alpha: float = 0.1
    gamma: float = 0.99
    gae_lambda: float = 0.95
    value_loss_coeff: float = 0.5
    max_grad_norm: float = 0.5
    num_layers: int = 4
    hidden_dim: int = 64
    vocab_size: int = 100

    def validate(self) -> list[str]:
        """Validate configuration, returning list of errors."""
        errors = []
        if self.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        if self.entropy_coeff < 0:
            errors.append("entropy_coeff must be non-negative")
        if self.entropy_mode not in ("coefficient", "target"):
            errors.append(f"Unknown entropy_mode: {self.entropy_mode}")
        if self.clip_epsilon <= 0 or self.clip_epsilon >= 1:
            errors.append("clip_epsilon must be in (0, 1)")
        if self.min_beta < 0:
            errors.append("min_beta must be non-negative")
        if self.decay_rate <= 0 or self.decay_rate > 1:
            errors.append("decay_rate must be in (0, 1]")
        return errors

    @classmethod
    def from_dict(cls, d: dict) -> "EPPOConfig":
        """Create config from dictionary, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)
