from __future__ import annotations

"""Entropy bonus computation for EPPO.

Supports two modes:
- coefficient: decays beta over time
- target: adaptively adjusts beta to maintain entropy target
"""

from dataclasses import dataclass


@dataclass
class EntropyBonusState:
    """Snapshot of entropy bonus state."""

    beta: float
    mode: str
    step_count: int
    entropy_target: float


class EntropyBonus:
    """Manages entropy bonus coefficient for EPPO training.

    In coefficient mode, beta decays multiplicatively each step.
    In target mode, beta adapts to push entropy toward a target value.
    """

    def __init__(
        self,
        initial_beta: float = 0.01,
        mode: str = "coefficient",
        decay_rate: float = 0.995,
        min_beta: float = 0.001,
        entropy_target: float = 1.5,
        adaptive_alpha: float = 0.1,
    ):
        self._initial_beta = initial_beta
        self._beta = initial_beta
        self._mode = mode
        self._decay_rate = decay_rate
        self._min_beta = min_beta
        self._entropy_target = entropy_target
        self._adaptive_alpha = adaptive_alpha
        self._step_count = 0
        self._history: list[float] = []

    @property
    def current_beta(self) -> float:
        """Current entropy bonus coefficient."""
        return self._beta

    @property
    def mode(self) -> str:
        """Current operating mode."""
        return self._mode

    @property
    def step_count(self) -> int:
        """Number of steps taken."""
        return self._step_count

    @property
    def history(self) -> list[float]:
        """History of beta values."""
        return list(self._history)

    def compute(self, entropy: float) -> float:
        """Compute entropy bonus value: beta * entropy."""
        return self._beta * entropy

    def step(self, current_entropy: float) -> float:
        """Update beta based on current entropy and return new beta.

        In coefficient mode: beta *= decay_rate, floored at min_beta.
        In target mode: beta increases if entropy < target, decreases otherwise.
        """
        self._step_count += 1
        self._history.append(self._beta)

        if self._mode == "coefficient":
            self._beta = max(self._beta * self._decay_rate, self._min_beta)
        elif self._mode == "target":
            # Adaptive: increase beta when entropy is too low, decrease when too high
            entropy_gap = self._entropy_target - current_entropy
            self._beta = max(
                self._beta + self._adaptive_alpha * entropy_gap,
                self._min_beta,
            )
        else:
            raise ValueError(f"Unknown mode: {self._mode}")

        return self._beta

    def reset(self) -> None:
        """Reset to initial state."""
        self._beta = self._initial_beta
        self._step_count = 0
        self._history.clear()

    def get_state(self) -> EntropyBonusState:
        """Get current state snapshot."""
        return EntropyBonusState(
            beta=self._beta,
            mode=self._mode,
            step_count=self._step_count,
            entropy_target=self._entropy_target,
        )
