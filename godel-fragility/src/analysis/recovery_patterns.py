"""Analyze recovery patterns from fault injection events."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.measurement.recovery_tracker import RecoveryEvent, RecoveryTracker
from src.utils.metrics import safe_division


class RecoveryPatternAnalyzer:
    """Analyze patterns in how the agent recovers from faults."""

    def cluster_recovery_patterns(
        self, events: List[RecoveryEvent], n_clusters: int = 3
    ) -> Dict[str, List[RecoveryEvent]]:
        """Cluster recovery events by their pattern.

        Clusters based on detection latency, recovery quality, and fault type.
        Uses simple rule-based clustering (not k-means) for robustness.
        """
        clusters: Dict[str, List[RecoveryEvent]] = {
            "quick_recovery": [],
            "slow_recovery": [],
            "no_recovery": [],
        }

        for event in events:
            if not event.was_recovered:
                clusters["no_recovery"].append(event)
            elif event.detection_latency is not None and event.detection_latency <= 3:
                clusters["quick_recovery"].append(event)
            else:
                clusters["slow_recovery"].append(event)

        return clusters

    def recovery_strategy_effectiveness(
        self, events: List[RecoveryEvent]
    ) -> Dict[str, Dict[str, float]]:
        """Analyze which recovery strategies are most effective.

        Returns:
            Dict mapping recovery_method -> {count, avg_quality, avg_latency}.
        """
        by_method: Dict[str, List[RecoveryEvent]] = defaultdict(list)

        for event in events:
            method = event.recovery_method or "none"
            by_method[method].append(event)

        results: Dict[str, Dict[str, float]] = {}

        for method, method_events in by_method.items():
            qualities = [
                e.recovery_quality for e in method_events if e.was_recovered
            ]
            latencies = [
                e.recovery_latency for e in method_events
                if e.recovery_latency is not None
            ]

            results[method] = {
                "count": float(len(method_events)),
                "recovery_rate": safe_division(
                    sum(1 for e in method_events if e.was_recovered),
                    len(method_events),
                ),
                "avg_quality": (
                    sum(qualities) / len(qualities) if qualities else 0.0
                ),
                "avg_latency": (
                    sum(latencies) / len(latencies) if latencies else float("inf")
                ),
            }

        return results

    def time_to_recovery_distribution(
        self, events: List[RecoveryEvent]
    ) -> Dict[str, Any]:
        """Compute statistics on time-to-recovery.

        Returns:
            Dict with mean, median, std, percentiles.
        """
        latencies = [
            e.recovery_latency for e in events
            if e.recovery_latency is not None
        ]

        if not latencies:
            return {
                "mean": None,
                "median": None,
                "std": None,
                "p25": None,
                "p75": None,
                "p95": None,
                "count": 0,
            }

        arr = np.array(latencies, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
            "count": len(latencies),
        }

    def plot_recovery_trajectories(
        self,
        events: List[RecoveryEvent],
        output_path: Optional[str] = None,
    ) -> Optional[Any]:
        """Plot accuracy trajectories for recovery events."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        if not events:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: individual trajectories
        ax1 = axes[0]
        for event in events[:20]:  # Limit for readability
            if event.accuracies:
                color = "green" if event.was_recovered else "red"
                alpha = 0.6 if event.was_recovered else 0.3
                ax1.plot(
                    event.accuracies,
                    color=color,
                    alpha=alpha,
                    linewidth=1,
                )

        ax1.set_xlabel("Iteration (relative to injection)")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Recovery Trajectories")
        ax1.set_ylim(-0.05, 1.05)

        # Right: recovery rate by fault type
        ax2 = axes[1]
        by_type: Dict[str, List[bool]] = defaultdict(list)
        for event in events:
            by_type[event.fault_type].append(event.was_recovered)

        types = sorted(by_type.keys())
        rates = [
            safe_division(sum(by_type[t]), len(by_type[t]))
            for t in types
        ]
        colors = ["green" if r > 0.5 else "orange" if r > 0.2 else "red" for r in rates]

        ax2.barh(types, rates, color=colors, alpha=0.7)
        ax2.set_xlabel("Recovery Rate")
        ax2.set_title("Recovery Rate by Fault Type")
        ax2.set_xlim(0, 1)

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150)

        return fig
