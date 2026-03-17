from __future__ import annotations

"""Mock policy network for EPPO training."""

import numpy as np
from dataclasses import dataclass


@dataclass
class PolicyOutput:
    """Output from a policy forward pass."""

    logits: np.ndarray
    entropy: float
    log_probs: np.ndarray


class MockPolicy:
    """Mock policy network using numpy.

    Produces logits from random weights and tracks entropy over time.
    """

    def __init__(self, input_dim: int = 64, vocab_size: int = 100, seed: int = 42):
        self._rng = np.random.RandomState(seed)
        self._input_dim = input_dim
        self._vocab_size = vocab_size
        # Random weight matrix for mock forward pass
        self._weights = self._rng.randn(input_dim, vocab_size) * 0.1
        self._bias = np.zeros(vocab_size)
        self._entropy_history: list[float] = []
        self._step_count = 0

    @property
    def entropy_history(self) -> list[float]:
        """History of entropy values from forward passes."""
        return list(self._entropy_history)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass: input -> logits.

        Args:
            inputs: array of shape (batch, input_dim) or (input_dim,)

        Returns:
            logits of shape (batch, vocab_size) or (vocab_size,)
        """
        single = inputs.ndim == 1
        if single:
            inputs = inputs.reshape(1, -1)

        logits = inputs @ self._weights + self._bias
        # Add small noise to simulate training updates
        logits += self._rng.randn(*logits.shape) * 0.01

        if single:
            logits = logits[0]

        return logits

    def compute_entropy(self, logits: np.ndarray) -> float:
        """Compute entropy of softmax distribution over logits.

        Args:
            logits: shape (batch, vocab_size) or (vocab_size,)

        Returns:
            Mean entropy across batch.
        """
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)

        # Stable softmax
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp_logits = np.exp(shifted)
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

        # Entropy: -sum(p * log(p)), with epsilon for numerical stability
        eps = 1e-10
        entropy = -np.sum(probs * np.log(probs + eps), axis=-1)
        mean_entropy = float(np.mean(entropy))

        self._entropy_history.append(mean_entropy)
        return mean_entropy

    def get_log_probs(self, logits: np.ndarray) -> np.ndarray:
        """Compute log probabilities from logits.

        Args:
            logits: shape (batch, vocab_size) or (vocab_size,)

        Returns:
            Log probabilities, same shape as logits.
        """
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)

        shifted = logits - logits.max(axis=-1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
        log_probs = shifted - log_sum_exp
        return log_probs

    def forward_full(self, inputs: np.ndarray) -> PolicyOutput:
        """Full forward pass returning logits, entropy, and log_probs."""
        logits = self.forward(inputs)
        entropy = self.compute_entropy(logits)
        log_probs = self.get_log_probs(logits)
        return PolicyOutput(logits=logits, entropy=entropy, log_probs=log_probs)

    def update_weights(self, gradient: np.ndarray, lr: float = 0.001) -> None:
        """Mock weight update (gradient descent step)."""
        self._weights -= lr * gradient
        self._step_count += 1

    def perturb_weights(self, scale: float = 0.01) -> None:
        """Add random perturbation to weights (simulates training)."""
        self._weights += self._rng.randn(*self._weights.shape) * scale
        self._step_count += 1
