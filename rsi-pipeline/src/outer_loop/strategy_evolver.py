"""Strategy evolver: generates improvement candidates using mock LLM."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

from src.pipeline.state import PipelineState


@dataclass
class Candidate:
    """A proposed code modification candidate."""
    candidate_id: str = ""
    target: str = "default"
    proposed_code: str = ""
    description: str = ""
    parent_ids: List[str] = field(default_factory=list)
    operator: str = "mutate"
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.candidate_id:
            content = f"{self.target}:{self.proposed_code}:{time.time()}"
            self.candidate_id = hashlib.sha256(content.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StrategyEvolver:
    """Generates improvement candidates using SOAR-style evolution.

    Uses a pluggable LLM (mock for tests) to propose code modifications.
    """

    def __init__(
        self,
        llm: Optional[Callable] = None,
        operators: Optional[List[str]] = None,
        targets: Optional[List[str]] = None,
    ):
        self._llm = llm or self._default_mock_llm
        self._operators = operators or ["mutate", "crossover", "refine"]
        self._targets = targets or ["strategy_evolver", "candidate_pool", "empirical_gate"]
        self._generation_count = 0

    def generate_candidates(self, state: PipelineState, n: int = 5) -> List[Candidate]:
        """Generate n improvement candidates for the current state."""
        candidates = []
        for i in range(n):
            operator = self._operators[i % len(self._operators)]
            target = self._targets[i % len(self._targets)]
            proposed = self._llm(state.agent_code.code, target, operator)
            candidate = Candidate(
                target=target,
                proposed_code=proposed,
                description=f"Improvement via {operator} on {target} (gen {self._generation_count})",
                parent_ids=[state.state_id] if state.state_id else [],
                operator=operator,
                score=0.0,
            )
            candidates.append(candidate)
        self._generation_count += 1
        return candidates

    @staticmethod
    def _default_mock_llm(current_code: str, target: str, operator: str) -> str:
        """Mock LLM that produces simple code mutations."""
        if operator == "mutate":
            return current_code + f"\n# Mutated {target}"
        elif operator == "crossover":
            return current_code + f"\n# Crossover {target}"
        elif operator == "refine":
            return current_code + f"\n# Refined {target}"
        return current_code + f"\n# {operator} {target}"
