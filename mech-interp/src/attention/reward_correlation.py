"""Detect reward-correlated attention heads."""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np

from src.probing.extractor import HeadStats


@dataclass
class RewardCorrelatedHead:
    """An attention head correlated with reward signals."""
    layer: int
    head: int
    correlation: float
    p_value_approx: float
    trend: str  # "increasing", "decreasing", "stable"
    n_samples: int


class RewardCorrelationDetector:
    """Detect heads whose behavior correlates with reward signals."""

    def __init__(self, correlation_threshold: float = 0.5,
                 min_samples: int = 5):
        self.correlation_threshold = correlation_threshold
        self.min_samples = min_samples
        # (layer, head) -> list of (head_stat_value, reward)
        self.pairs: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}

    def collect_pair(self, head_stats: List[HeadStats], reward: float) -> None:
        """Collect a (head_stats, reward) pair for correlation analysis."""
        for hs in head_stats:
            key = (hs.layer, hs.head)
            if key not in self.pairs:
                self.pairs[key] = []
            # Use entropy as the head statistic
            self.pairs[key].append((hs.entropy, reward))

    def compute_correlations(self) -> Dict[Tuple[int, int], float]:
        """Compute correlation between head entropy and reward for each head."""
        correlations = {}
        for key, pairs in self.pairs.items():
            if len(pairs) < self.min_samples:
                continue
            entropies = np.array([p[0] for p in pairs])
            rewards = np.array([p[1] for p in pairs])
            if np.std(entropies) < 1e-10 or np.std(rewards) < 1e-10:
                correlations[key] = 0.0
                continue
            corr = np.corrcoef(entropies, rewards)[0, 1]
            correlations[key] = float(corr) if not np.isnan(corr) else 0.0
        return correlations

    def detect_reward_correlated(self) -> List[RewardCorrelatedHead]:
        """Find heads with significant reward correlation."""
        correlations = self.compute_correlations()
        result = []
        for key, corr in sorted(correlations.items()):
            if abs(corr) >= self.correlation_threshold:
                pairs = self.pairs[key]
                n = len(pairs)
                # Approximate p-value using t-test
                if n > 2 and abs(corr) < 1.0:
                    t_stat = corr * np.sqrt((n - 2) / (1 - corr**2))
                    # Rough p-value approximation
                    p_approx = 2.0 * np.exp(-0.5 * t_stat**2) / np.sqrt(2 * np.pi)
                else:
                    p_approx = 0.0 if abs(corr) >= 1.0 else 1.0

                trend = self._compute_trend(pairs)
                result.append(RewardCorrelatedHead(
                    layer=key[0],
                    head=key[1],
                    correlation=corr,
                    p_value_approx=float(p_approx),
                    trend=trend,
                    n_samples=n,
                ))
        return result

    def monitor_correlation_trend(self, window: int = 10) -> Dict[Tuple[int, int], str]:
        """Monitor how correlations change over recent window."""
        trends = {}
        for key, pairs in self.pairs.items():
            if len(pairs) < window:
                continue
            recent = pairs[-window:]
            trend = self._compute_trend(recent)
            trends[key] = trend
        return trends

    def _compute_trend(self, pairs: List[Tuple[float, float]]) -> str:
        """Classify trend of correlation pairs."""
        if len(pairs) < 3:
            return "stable"
        rewards = [p[1] for p in pairs]
        # Simple linear trend
        x = np.arange(len(rewards), dtype=float)
        mean_x = x.mean()
        mean_y = np.mean(rewards)
        num = np.sum((x - mean_x) * (np.array(rewards) - mean_y))
        den = np.sum((x - mean_x)**2)
        if abs(den) < 1e-10:
            return "stable"
        slope = num / den
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        return "stable"
