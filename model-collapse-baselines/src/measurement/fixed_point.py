"""Fixed-point (Q*) detection for the model-collapse trajectory.

Detects when the recursive fine-tuning process has converged --
i.e., when additional generations no longer change the model's output
distribution.  This is the "fixed point" Q* of the iterative map
M_{t-1} -> M_t.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceReport:
    """Summary of convergence behaviour."""

    converged: bool
    generation_converged: int | None  # None if not yet converged
    kl_at_convergence: float | None
    diversity_at_convergence: float | None
    total_entropy_lost: float  # entropy[0] - entropy[latest]
    kl_trajectory: list[float] = field(default_factory=list)
    diversity_trajectory: list[float] = field(default_factory=list)
    entropy_trajectory: list[float] = field(default_factory=list)


class FixedPointDetector:
    """Detect when the model-collapse trajectory has reached a fixed point.

    A fixed point is declared when both KL divergence change and
    diversity change between successive generations stay below their
    respective tolerances for ``patience`` consecutive generations.

    Usage::

        detector = FixedPointDetector(patience=3)
        for gen in range(num_generations):
            metrics = measure(model)
            reached_qstar = detector.update(gen, metrics)
            if reached_qstar:
                print(f"Converged at generation {gen}")
                break
        report = detector.get_convergence_report()

    Args:
        patience: Number of consecutive stable generations required
            before declaring convergence.
        kl_tolerance: Maximum allowed absolute change in KL divergence
            between successive generations.
        diversity_tolerance: Maximum allowed absolute change in
            distinct-2 (or similar diversity metric) between
            successive generations.
    """

    def __init__(
        self,
        patience: int = 3,
        kl_tolerance: float = 0.01,
        diversity_tolerance: float = 0.01,
    ) -> None:
        self._patience = patience
        self._kl_tolerance = kl_tolerance
        self._diversity_tolerance = diversity_tolerance

        # State.
        self._kl_history: list[float] = []
        self._diversity_history: list[float] = []
        self._entropy_history: list[float] = []
        self._consecutive_stable: int = 0
        self._converged: bool = False
        self._convergence_generation: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, generation: int, metrics: dict[str, Any]) -> bool:
        """Feed metrics from one generation and check for convergence.

        Args:
            generation: The generation index (0-based).
            metrics: Dictionary containing at least:
                - ``"kl_divergence"`` (float): KL(P || Q_t)
                - ``"diversity"`` (float): A diversity metric (e.g. distinct-2)
                - ``"entropy"`` (float, optional): Mean predictive entropy

        Returns:
            ``True`` if the fixed point Q* has been reached.
        """
        if self._converged:
            return True

        kl = float(metrics.get("kl_divergence", 0.0))
        diversity = float(metrics.get("diversity", 0.0))
        entropy = float(metrics.get("entropy", 0.0))

        self._kl_history.append(kl)
        self._diversity_history.append(diversity)
        self._entropy_history.append(entropy)

        # Need at least two points to compare.
        if len(self._kl_history) < 2:
            return False

        kl_delta = abs(kl - self._kl_history[-2])
        div_delta = abs(diversity - self._diversity_history[-2])

        kl_stable = kl_delta < self._kl_tolerance
        div_stable = div_delta < self._diversity_tolerance

        if kl_stable and div_stable:
            self._consecutive_stable += 1
        else:
            self._consecutive_stable = 0

        if self._consecutive_stable >= self._patience:
            self._converged = True
            self._convergence_generation = generation
            logger.info(
                "Fixed point Q* detected at generation %d "
                "(KL=%.4f, diversity=%.4f)",
                generation,
                kl,
                diversity,
            )
            return True

        return False

    def get_convergence_report(self) -> ConvergenceReport:
        """Return a summary of the convergence trajectory.

        Returns:
            A ``ConvergenceReport`` with trajectories and convergence
            information.
        """
        # Total entropy lost = entropy at gen 0 minus latest.
        if len(self._entropy_history) >= 2:
            total_entropy_lost = self._entropy_history[0] - self._entropy_history[-1]
        else:
            total_entropy_lost = 0.0

        kl_at_conv = None
        div_at_conv = None
        if self._converged and self._convergence_generation is not None:
            idx = self._convergence_generation
            if idx < len(self._kl_history):
                kl_at_conv = self._kl_history[idx]
            # Fall back to last recorded value.
            if kl_at_conv is None and self._kl_history:
                kl_at_conv = self._kl_history[-1]

            if idx < len(self._diversity_history):
                div_at_conv = self._diversity_history[idx]
            if div_at_conv is None and self._diversity_history:
                div_at_conv = self._diversity_history[-1]

        return ConvergenceReport(
            converged=self._converged,
            generation_converged=self._convergence_generation,
            kl_at_convergence=kl_at_conv,
            diversity_at_convergence=div_at_conv,
            total_entropy_lost=total_entropy_lost,
            kl_trajectory=list(self._kl_history),
            diversity_trajectory=list(self._diversity_history),
            entropy_trajectory=list(self._entropy_history),
        )

    def reset(self) -> None:
        """Reset all internal state."""
        self._kl_history.clear()
        self._diversity_history.clear()
        self._entropy_history.clear()
        self._consecutive_stable = 0
        self._converged = False
        self._convergence_generation = None
