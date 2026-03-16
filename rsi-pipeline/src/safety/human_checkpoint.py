"""Human checkpoint: pauses for human review."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from src.pipeline.state import PipelineState


class HumanCheckpoint:
    """Manages human review checkpoints in the pipeline."""

    def __init__(
        self,
        review_interval: int = 10,
        auto_approve_mode: bool = False,
        reviewer: Optional[Callable] = None,
    ):
        self._review_interval = review_interval
        self._auto_approve = auto_approve_mode
        self._reviewer = reviewer
        self._reviews: list = []

    def should_pause(self, state: PipelineState) -> bool:
        """Check if the pipeline should pause for human review."""
        if state.safety.emergency_stop:
            return True
        if state.iteration > 0 and state.iteration % self._review_interval == 0:
            return True
        return False

    def present_review(self, state: PipelineState) -> Dict[str, Any]:
        """Present the current state for review."""
        review = {
            "iteration": state.iteration,
            "accuracy": state.performance.accuracy,
            "gdi_score": state.safety.gdi_score,
            "car_score": state.safety.car_score,
            "consecutive_rollbacks": state.safety.consecutive_rollbacks,
            "violations": state.safety.violations,
            "status": state.status,
        }
        self._reviews.append(review)
        return review

    def auto_approve(self, state: PipelineState) -> bool:
        """Auto-approve for testing. Returns True if approved."""
        if self._auto_approve:
            return True
        if self._reviewer:
            return self._reviewer(state)
        return False

    @property
    def reviews(self) -> list:
        return list(self._reviews)

    @property
    def review_count(self) -> int:
        return len(self._reviews)
