"""Data quality analysis for training pairs."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from src.synthesis.synthesizer import TrainingPair


class DataQualityAnalyzer:
    """Analyze quality characteristics of synthesized training data."""

    def __init__(self, pairs: List[TrainingPair] = None):
        self._pairs: List[TrainingPair] = pairs or []

    def load(self, pairs: List[TrainingPair]) -> None:
        self._pairs = list(pairs)

    @property
    def count(self) -> int:
        return len(self._pairs)

    def quality_distribution(self, n_bins: int = 10) -> Dict[str, int]:
        """Distribution of quality scores across bins."""
        bins: Counter = Counter()
        for p in self._pairs:
            bucket = min(n_bins - 1, int(p.quality_score * n_bins))
            low = bucket / n_bins
            high = (bucket + 1) / n_bins
            bins[f"{low:.1f}-{high:.1f}"] = bins.get(f"{low:.1f}-{high:.1f}", 0) + 1
        return dict(sorted(bins.items()))

    def token_stats(self) -> Dict[str, Any]:
        """Statistics on token counts."""
        if not self._pairs:
            return {"prompt_mean": 0, "completion_mean": 0, "total_mean": 0}

        prompt_tokens = [p.prompt_tokens for p in self._pairs]
        comp_tokens = [p.completion_tokens for p in self._pairs]
        total_tokens = [p.total_tokens for p in self._pairs]

        return {
            "prompt_mean": round(sum(prompt_tokens) / len(prompt_tokens), 1),
            "prompt_min": min(prompt_tokens),
            "prompt_max": max(prompt_tokens),
            "completion_mean": round(sum(comp_tokens) / len(comp_tokens), 1),
            "completion_min": min(comp_tokens),
            "completion_max": max(comp_tokens),
            "total_mean": round(sum(total_tokens) / len(total_tokens), 1),
            "total_tokens": sum(total_tokens),
        }

    def strategy_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """Quality breakdown by strategy."""
        by_strat: Dict[str, List[TrainingPair]] = {}
        for p in self._pairs:
            by_strat.setdefault(p.strategy, []).append(p)

        result = {}
        for strat, pairs in by_strat.items():
            qualities = [p.quality_score for p in pairs]
            result[strat] = {
                "count": len(pairs),
                "avg_quality": round(sum(qualities) / len(qualities), 4),
                "min_quality": round(min(qualities), 4),
                "max_quality": round(max(qualities), 4),
            }
        return result

    def completeness_check(self) -> Dict[str, int]:
        """Check for common data quality issues."""
        issues = {
            "empty_prompt": 0,
            "empty_completion": 0,
            "zero_tokens": 0,
            "zero_quality": 0,
            "missing_task_id": 0,
        }
        for p in self._pairs:
            if not p.prompt.strip():
                issues["empty_prompt"] += 1
            if not p.completion.strip():
                issues["empty_completion"] += 1
            if p.total_tokens == 0:
                issues["zero_tokens"] += 1
            if p.quality_score == 0.0:
                issues["zero_quality"] += 1
            if not p.task_id:
                issues["missing_task_id"] += 1
        return issues

    def full_report(self) -> Dict[str, Any]:
        """Generate a complete data quality report."""
        return {
            "total_pairs": self.count,
            "quality_distribution": self.quality_distribution(),
            "token_stats": self.token_stats(),
            "strategy_breakdown": self.strategy_breakdown(),
            "completeness": self.completeness_check(),
        }
