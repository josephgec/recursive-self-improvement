"""Individual in the evolutionary population."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Individual:
    """A candidate solution (program) in the evolutionary population."""

    code: str
    individual_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    operator: str = "unknown"
    fitness: float = 0.0
    train_accuracy: float = 0.0
    test_accuracy: float = 0.0
    pixel_accuracy: float = 0.0
    simplicity_score: float = 0.0
    consistency_score: float = 0.0
    compile_error: Optional[str] = None
    runtime_errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    evaluated: bool = False

    @property
    def is_valid(self) -> bool:
        """Whether the program compiles and runs without errors."""
        return self.compile_error is None and not self.runtime_errors

    @property
    def code_length(self) -> int:
        """Length of the program code."""
        return len(self.code)

    @property
    def line_count(self) -> int:
        """Number of lines in the program."""
        return len(self.code.strip().splitlines())

    def copy(self) -> Individual:
        """Create a copy of this individual with a new ID."""
        return Individual(
            code=self.code,
            generation=self.generation,
            parent_ids=[self.individual_id],
            operator=self.operator,
            fitness=self.fitness,
            train_accuracy=self.train_accuracy,
            test_accuracy=self.test_accuracy,
            pixel_accuracy=self.pixel_accuracy,
            simplicity_score=self.simplicity_score,
            consistency_score=self.consistency_score,
            compile_error=self.compile_error,
            runtime_errors=list(self.runtime_errors),
            metadata=dict(self.metadata),
            evaluated=self.evaluated,
        )

    def summary(self) -> str:
        """Short summary string."""
        return (
            f"Individual({self.individual_id[:8]}, gen={self.generation}, "
            f"fitness={self.fitness:.3f}, acc={self.train_accuracy:.3f})"
        )

    def __repr__(self) -> str:
        return self.summary()
