"""Track code complexity over time."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.core.state import AgentState
from src.modification.inspector import CodeInspector


class ComplexityTracker:
    """Tracks code complexity across agent states."""

    def __init__(self) -> None:
        self.inspector = CodeInspector()
        self._history: list[dict[str, Any]] = []

    def track(self, states: list[AgentState]) -> pd.DataFrame:
        """Compute complexity metrics for a list of states."""
        records: list[dict[str, Any]] = []
        for state in states:
            record: dict[str, Any] = {
                "iteration": state.iteration,
                "state_id": state.state_id,
                "num_modifications": len(state.modifications_applied),
            }

            # Measure code complexity if there's code to measure
            for code_field in ("few_shot_selector_code", "reasoning_strategy_code"):
                code = getattr(state, code_field, "")
                if code:
                    try:
                        complexity = self.inspector.get_complexity(code)
                        record[f"{code_field}_nodes"] = complexity.ast_node_count
                        record[f"{code_field}_cyclomatic"] = complexity.cyclomatic_complexity
                        record[f"{code_field}_nesting"] = complexity.max_nesting
                        record[f"{code_field}_loc"] = complexity.lines_of_code
                    except Exception:
                        pass

            records.append(record)

        self._history = records
        return pd.DataFrame(records)

    def detect_complexity_explosion(
        self,
        states: list[AgentState],
        threshold_ratio: float = 5.0,
    ) -> bool:
        """Detect if complexity has exploded relative to initial state."""
        if len(states) < 2:
            return False

        df = self.track(states)
        complexity_cols = [c for c in df.columns if c.endswith("_nodes")]

        for col in complexity_cols:
            values = df[col].dropna()
            if len(values) >= 2 and values.iloc[0] > 0:
                ratio = values.iloc[-1] / values.iloc[0]
                if ratio > threshold_ratio:
                    return True
        return False

    def plot_complexity_vs_performance(
        self,
        states: list[AgentState],
        output_path: str | None = None,
    ) -> Any:
        """Plot complexity metrics against performance."""
        try:
            import matplotlib.pyplot as plt

            df = self.track(states)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Accuracy over iterations
            accuracies = []
            for state in states:
                if state.accuracy_history:
                    accuracies.append(state.accuracy_history[-1])
                else:
                    accuracies.append(0.0)

            ax1.plot(range(len(accuracies)), accuracies, "b-o", markersize=4)
            ax1.set_xlabel("State Index")
            ax1.set_ylabel("Accuracy")
            ax1.set_title("Performance Over Time")

            # Modifications count
            mods = [len(s.modifications_applied) for s in states]
            ax2.bar(range(len(mods)), mods, alpha=0.7)
            ax2.set_xlabel("State Index")
            ax2.set_ylabel("Cumulative Modifications")
            ax2.set_title("Modification Count")

            plt.tight_layout()

            if output_path:
                fig.savefig(output_path, dpi=150, bbox_inches="tight")

            return fig

        except ImportError:
            return None
