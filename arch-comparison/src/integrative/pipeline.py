"""Integrative pipeline: LNN-style constrained decoding with no external solvers."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from src.integrative.constrained_decoder import Constraint, ConstrainedDecoder, ConstrainedOutput
from src.integrative.logical_loss import LogicalLoss


@dataclass
class IntegrativeResult:
    """Result from the integrative pipeline."""
    answer: str
    correct: bool = False
    constrained_output: Optional[ConstrainedOutput] = None
    logical_loss: float = 0.0
    constraints_applied: int = 0
    constraint_violations: int = 0
    total_time: float = 0.0
    metadata: dict = field(default_factory=dict)


class IntegrativePipeline:
    """Integrative pipeline: all reasoning happens inside the model.

    Uses constrained decoding to enforce logical/arithmetic rules
    during generation. No external solver calls.
    """

    def __init__(
        self,
        model: Optional[Callable] = None,
        constraint_weight: float = 0.3,
    ) -> None:
        self.decoder = ConstrainedDecoder(model=model, constraint_weight=constraint_weight)
        self.loss_fn = LogicalLoss()

        # Register default constraints
        self.decoder.add_constraint(Constraint(
            constraint_type="arithmetic",
            pattern=r"\d+\s*[\+\-\*\/]\s*\d+\s*=\s*\d+",
            weight=1.0,
        ))
        self.decoder.add_constraint(Constraint(
            constraint_type="logical",
            pattern=r"(\w+)\s+is\s+(true|false)",
            weight=0.8,
        ))

    def solve(self, problem: str) -> IntegrativeResult:
        """Solve a problem using constrained decoding.

        No external tools are called — everything happens inside the model
        with constraint enforcement.
        """
        start = time.monotonic()

        # Generate with constraints
        output: ConstrainedOutput = self.decoder.generate(problem)

        # Compute logical loss
        loss = self.loss_fn.compute(output.text, context=problem)

        # Extract answer
        answer = self._extract_answer(output.text)

        elapsed = time.monotonic() - start
        return IntegrativeResult(
            answer=answer,
            constrained_output=output,
            logical_loss=loss,
            constraints_applied=len(output.constraints_applied),
            constraint_violations=output.constraint_violations,
            total_time=elapsed,
        )

    def _extract_answer(self, text: str) -> str:
        """Extract the final answer from generated text."""
        match = re.search(r"the answer is\s+(.+?)[\.\n]", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", text)
        if match:
            return match.group(1).strip()

        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        return lines[-1] if lines else text.strip()
