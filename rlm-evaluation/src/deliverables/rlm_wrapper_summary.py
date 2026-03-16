"""RLM Wrapper summary for Phase 2b deliverables."""

from __future__ import annotations

from typing import Dict, List, Optional

from src.benchmarks.task import EvalResult


class RLMWrapperSummary:
    """Generate summary of the RLM wrapper capabilities and performance."""

    def __init__(
        self,
        rlm_results: Optional[List[EvalResult]] = None,
    ) -> None:
        self.rlm_results = rlm_results or []

    def generate(self) -> str:
        """Generate the RLM wrapper summary document."""
        sections = [
            self._header(),
            self._architecture_section(),
            self._performance_section(),
            self._strategy_section(),
            self._limitations_section(),
        ]
        return "\n\n".join(sections)

    def _header(self) -> str:
        return "# RLM Wrapper Summary\n\nRecursive Language Model wrapper for long-context task solving."

    def _architecture_section(self) -> str:
        lines = [
            "## Architecture",
            "",
            "The RLM wrapper enables LLMs to handle arbitrarily long contexts by:",
            "- Breaking context into manageable chunks",
            "- Using code-based tools (grep, head, split) to navigate",
            "- Spawning sub-sessions for complex queries",
            "- Aggregating results across multiple passes",
        ]
        return "\n".join(lines)

    def _performance_section(self) -> str:
        if not self.rlm_results:
            return "## Performance\n\nNo results available."

        total = len(self.rlm_results)
        correct = sum(1 for r in self.rlm_results if r.correct)
        accuracy = correct / total if total > 0 else 0
        avg_cost = sum(r.cost for r in self.rlm_results) / total if total > 0 else 0

        lines = [
            "## Performance",
            "",
            f"- Tasks evaluated: {total}",
            f"- Accuracy: {accuracy:.1%}",
            f"- Average cost per task: ${avg_cost:.4f}",
            f"- Total cost: ${sum(r.cost for r in self.rlm_results):.4f}",
        ]
        return "\n".join(lines)

    def _strategy_section(self) -> str:
        if not self.rlm_results:
            return "## Emergent Strategies\n\nNo data available."

        strategies: Dict[str, int] = {}
        for r in self.rlm_results:
            s = r.strategy_detected or "unknown"
            strategies[s] = strategies.get(s, 0) + 1

        lines = [
            "## Emergent Strategies",
            "",
            "The RLM demonstrates emergent strategy selection:",
        ]
        for s, count in sorted(strategies.items(), key=lambda x: -x[1]):
            lines.append(f"- {s}: {count} tasks ({count / len(self.rlm_results):.0%})")
        return "\n".join(lines)

    def _limitations_section(self) -> str:
        lines = [
            "## Limitations",
            "",
            "- Higher cost per task compared to single-shot approaches",
            "- Latency increases with context size",
            "- Strategy selection is not always optimal",
            "- Complex reasoning chains can propagate errors",
        ]
        return "\n".join(lines)
