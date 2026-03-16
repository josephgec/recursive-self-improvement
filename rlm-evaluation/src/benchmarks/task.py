"""Core task and result dataclasses for evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvalTask:
    """A single evaluation task."""

    task_id: str
    benchmark: str
    query: str
    context: str
    expected_answer: str
    category: str = "general"
    context_tokens: int = 0
    difficulty: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.context_tokens == 0 and self.context:
            # Rough token estimate: ~4 chars per token
            self.context_tokens = len(self.context) // 4

    def with_context_size(self, target_tokens: int) -> "EvalTask":
        """Return a copy of this task with context padded/trimmed to target size."""
        target_chars = target_tokens * 4
        current_chars = len(self.context)

        if current_chars >= target_chars:
            new_context = self.context[:target_chars]
        else:
            padding = " filler_text" * ((target_chars - current_chars) // 12 + 1)
            new_context = self.context + padding[:target_chars - current_chars]

        return EvalTask(
            task_id=f"{self.task_id}_ctx{target_tokens}",
            benchmark=self.benchmark,
            query=self.query,
            context=new_context,
            expected_answer=self.expected_answer,
            category=self.category,
            context_tokens=target_tokens,
            difficulty=self.difficulty,
            metadata={**self.metadata, "original_task_id": self.task_id},
        )


@dataclass
class EvalResult:
    """Result of evaluating a single task."""

    task_id: str
    benchmark: str
    answer: str
    correct: bool
    trajectory: List[str] = field(default_factory=list)
    strategy_detected: str = "unknown"
    cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    num_calls: int = 1
    latency_ms: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "task_id": self.task_id,
            "benchmark": self.benchmark,
            "answer": self.answer,
            "correct": self.correct,
            "trajectory": self.trajectory,
            "strategy_detected": self.strategy_detected,
            "cost": self.cost,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "num_calls": self.num_calls,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalResult":
        """Deserialize from dictionary."""
        return cls(**data)
