"""Orchestrates all synthesis strategies to produce training pairs."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from src.collection.trajectory import SearchTrajectory


@dataclass
class TrainingPair:
    """Universal exchange format between synthesis and fine-tuning."""

    pair_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    strategy: str = ""
    task_id: str = ""
    prompt: str = ""
    completion: str = ""
    quality_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "strategy": self.strategy,
            "task_id": self.task_id,
            "prompt": self.prompt,
            "completion": self.completion,
            "quality_score": self.quality_score,
            "metadata": dict(self.metadata),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingPair":
        return cls(
            pair_id=data.get("pair_id", str(uuid.uuid4())[:12]),
            strategy=data.get("strategy", ""),
            task_id=data.get("task_id", ""),
            prompt=data.get("prompt", ""),
            completion=data.get("completion", ""),
            quality_score=data.get("quality_score", 0.0),
            metadata=data.get("metadata", {}),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
        )


class SynthesisStrategy(Protocol):
    """Protocol for synthesis strategies."""

    name: str

    def generate(self, trajectories: List[SearchTrajectory]) -> List[TrainingPair]:
        ...


class Synthesizer:
    """Orchestrates multiple synthesis strategies with configurable weights."""

    def __init__(
        self,
        strategies: Optional[List[Any]] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self._strategies: List[Any] = strategies or []
        self._weights: Dict[str, float] = weights or {}
        self._pairs: List[TrainingPair] = []

    def register_strategy(self, strategy: Any, weight: float = 1.0) -> None:
        """Register a synthesis strategy with a weight."""
        self._strategies.append(strategy)
        self._weights[strategy.name] = weight

    @property
    def strategies(self) -> List[Any]:
        return list(self._strategies)

    @property
    def strategy_names(self) -> List[str]:
        return [s.name for s in self._strategies]

    def synthesize(self, trajectories: List[SearchTrajectory]) -> List[TrainingPair]:
        """Run all strategies and combine results weighted by strategy weights."""
        all_pairs: List[TrainingPair] = []

        for strategy in self._strategies:
            weight = self._weights.get(strategy.name, 1.0)
            if weight <= 0:
                continue
            pairs = strategy.generate(trajectories)
            # Scale quality scores by weight
            for pair in pairs:
                pair.quality_score = pair.quality_score * weight
            all_pairs.extend(pairs)

        self._pairs = all_pairs
        return all_pairs

    @property
    def pairs(self) -> List[TrainingPair]:
        return list(self._pairs)

    def summary(self) -> Dict[str, Any]:
        """Return summary of synthesized pairs by strategy."""
        by_strategy: Dict[str, int] = {}
        for p in self._pairs:
            by_strategy[p.strategy] = by_strategy.get(p.strategy, 0) + 1
        return {
            "total_pairs": len(self._pairs),
            "by_strategy": by_strategy,
            "strategies_registered": len(self._strategies),
        }
