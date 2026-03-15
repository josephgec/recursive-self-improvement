"""Analysis of modification history."""

from __future__ import annotations

from typing import Any

import pandas as pd


class ModificationHistoryAnalyzer:
    """Analyzes the history of self-modifications."""

    def __init__(self, audit_entries: list[dict[str, Any]] | None = None) -> None:
        self._entries = audit_entries or []

    def load_entries(self, entries: list[dict[str, Any]]) -> None:
        """Load audit log entries."""
        self._entries = entries

    def success_rate_by_component(self) -> dict[str, float]:
        """Calculate modification success rate by target component."""
        component_stats: dict[str, dict[str, int]] = {}

        for entry in self._entries:
            if entry.get("type") != "modification":
                continue
            proposal = entry.get("proposal", {})
            target = proposal.get("target", "unknown")
            accepted = entry.get("accepted", False)

            if target not in component_stats:
                component_stats[target] = {"total": 0, "accepted": 0}
            component_stats[target]["total"] += 1
            if accepted:
                component_stats[target]["accepted"] += 1

        result: dict[str, float] = {}
        for target, stats in component_stats.items():
            total = stats["total"]
            result[target] = stats["accepted"] / total if total > 0 else 0.0
        return result

    def convergence_analysis(self) -> dict[str, Any]:
        """Analyze convergence patterns in modifications."""
        iterations = [e for e in self._entries if e.get("type") == "iteration"]
        modifications = [e for e in self._entries if e.get("type") == "modification"]
        rollbacks = [e for e in self._entries if e.get("type") == "rollback"]

        accuracies = [e.get("accuracy", 0.0) for e in iterations]

        return {
            "total_iterations": len(iterations),
            "total_modifications": len(modifications),
            "total_rollbacks": len(rollbacks),
            "acceptance_rate": (
                (len(modifications) - len(rollbacks)) / len(modifications)
                if modifications else 0.0
            ),
            "accuracy_trajectory": accuracies,
            "final_accuracy": accuracies[-1] if accuracies else 0.0,
            "best_accuracy": max(accuracies) if accuracies else 0.0,
        }

    def plot_modification_timeline(
        self, output_path: str | None = None
    ) -> Any:
        """Plot a timeline of modifications. Returns matplotlib figure."""
        try:
            import matplotlib.pyplot as plt

            iterations = [e for e in self._entries if e.get("type") == "iteration"]
            modifications = [e for e in self._entries if e.get("type") == "modification"]
            rollbacks = [e for e in self._entries if e.get("type") == "rollback"]

            fig, ax = plt.subplots(figsize=(12, 6))

            # Accuracy line
            if iterations:
                iters = [e.get("iteration", i) for i, e in enumerate(iterations)]
                accs = [e.get("accuracy", 0.0) for e in iterations]
                ax.plot(iters, accs, "b-o", label="Accuracy", markersize=4)

            # Modification markers
            for mod in modifications:
                it = mod.get("iteration", 0)
                accepted = mod.get("accepted", False)
                color = "green" if accepted else "red"
                ax.axvline(x=it, color=color, alpha=0.3, linestyle="--")

            # Rollback markers
            for rb in rollbacks:
                it = rb.get("iteration", 0)
                ax.axvline(x=it, color="red", alpha=0.5, linestyle=":")

            ax.set_xlabel("Iteration")
            ax.set_ylabel("Accuracy")
            ax.set_title("Modification Timeline")
            ax.legend()

            if output_path:
                fig.savefig(output_path, dpi=150, bbox_inches="tight")

            return fig

        except ImportError:
            return None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert entries to a pandas DataFrame."""
        return pd.DataFrame(self._entries)
