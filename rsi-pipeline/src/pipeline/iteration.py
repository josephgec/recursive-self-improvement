"""Iteration and pipeline result dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class IterationResult:
    """Result of a single pipeline iteration."""
    iteration: int = 0
    improved: bool = False
    candidate: Optional[Dict[str, Any]] = None
    accuracy_before: float = 0.0
    accuracy_after: float = 0.0
    modification: Optional[Dict[str, Any]] = None
    safety_verdict: str = "pass"  # pass, fail, emergency
    rolled_back: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def accuracy_delta(self) -> float:
        return self.accuracy_after - self.accuracy_before

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineResult:
    """Result of a complete pipeline run."""
    total_iterations: int = 0
    successful_improvements: int = 0
    rollbacks: int = 0
    emergency_stops: int = 0
    final_accuracy: float = 0.0
    initial_accuracy: float = 0.0
    iteration_results: List[IterationResult] = field(default_factory=list)
    converged: bool = False
    reason_stopped: str = ""

    @property
    def improvement_rate(self) -> float:
        if self.total_iterations == 0:
            return 0.0
        return self.successful_improvements / self.total_iterations

    @property
    def total_accuracy_gain(self) -> float:
        return self.final_accuracy - self.initial_accuracy

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["improvement_rate"] = self.improvement_rate
        d["total_accuracy_gain"] = self.total_accuracy_gain
        return d
