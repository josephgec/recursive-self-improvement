"""Data loader for fine-tuning training data."""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from src.synthesis.synthesizer import TrainingPair


class DataLoader:
    """Load and prepare training data for fine-tuning."""

    def __init__(self, seed: int = 42):
        self._pairs: List[TrainingPair] = []
        self._seed = seed

    def load_from_jsonl(self, filepath: str) -> List[Dict[str, Any]]:
        """Load raw JSONL data."""
        items = []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    def load_pairs(self, pairs: List[TrainingPair]) -> None:
        """Load training pairs directly."""
        self._pairs = list(pairs)

    def load_from_dicts(self, data: List[Dict[str, Any]]) -> List[TrainingPair]:
        """Load training pairs from dictionaries."""
        pairs = [TrainingPair.from_dict(d) for d in data]
        self._pairs.extend(pairs)
        return pairs

    @property
    def pairs(self) -> List[TrainingPair]:
        return list(self._pairs)

    @property
    def count(self) -> int:
        return len(self._pairs)

    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> Tuple[List[TrainingPair], List[TrainingPair], List[TrainingPair]]:
        """Split data into train/val/test sets."""
        rng = random.Random(self._seed)
        shuffled = list(self._pairs)
        rng.shuffle(shuffled)

        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train = shuffled[:train_end]
        val = shuffled[train_end:val_end]
        test = shuffled[val_end:]

        return train, val, test

    def get_batches(self, batch_size: int = 8) -> List[List[TrainingPair]]:
        """Split data into batches."""
        batches = []
        for i in range(0, len(self._pairs), batch_size):
            batches.append(self._pairs[i : i + batch_size])
        return batches

    def filter_by_strategy(self, strategy: str) -> List[TrainingPair]:
        """Filter pairs by strategy name."""
        return [p for p in self._pairs if p.strategy == strategy]

    def filter_by_quality(self, min_score: float = 0.0) -> List[TrainingPair]:
        """Filter pairs by minimum quality score."""
        return [p for p in self._pairs if p.quality_score >= min_score]

    def stats(self) -> Dict[str, Any]:
        """Return statistics about the loaded data."""
        if not self._pairs:
            return {"total": 0}

        by_strategy: Dict[str, int] = {}
        total_tokens = 0
        for p in self._pairs:
            by_strategy[p.strategy] = by_strategy.get(p.strategy, 0) + 1
            total_tokens += p.total_tokens

        qualities = [p.quality_score for p in self._pairs]
        return {
            "total": len(self._pairs),
            "by_strategy": by_strategy,
            "total_tokens": total_tokens,
            "avg_quality": round(sum(qualities) / len(qualities), 4),
            "min_quality": round(min(qualities), 4),
            "max_quality": round(max(qualities), 4),
        }
