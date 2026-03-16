"""Comprehensive markdown report generator."""

from __future__ import annotations

from typing import Dict, List, Optional

from src.benchmarks.task import EvalResult
from src.comparison.cost_model import CostModel, CostComparison
from src.comparison.head_to_head import HeadToHeadComparator, HeadToHeadReport
from src.strategies.emergence_analyzer import EmergenceAnalyzer, EmergenceReport
from src.analysis.cost_breakdown import CostBreakdownAnalysis
from src.analysis.strategy_landscape import StrategyLandscape
from src.analysis.trajectory_visualizer import TrajectoryVisualizer


class ReportGenerator:
    """Generate a comprehensive markdown evaluation report."""

    def __init__(self) -> None:
        self.cost_model = CostModel()
        self.h2h = HeadToHeadComparator()
        self.emergence = EmergenceAnalyzer()
        self.cost_analysis = CostBreakdownAnalysis()
        self.landscape = StrategyLandscape()
        self.visualizer = TrajectoryVisualizer()

    def generate(
        self,
        rlm_results: List[EvalResult],
        standard_results: List[EvalResult],
        task_categories: Optional[Dict[str, str]] = None,
    ) -> str:
        """Generate the full report.

        Args:
            rlm_results: Results from RLM system.
            standard_results: Results from standard system.
            task_categories: Optional mapping of task_id -> category.

        Returns:
            Markdown string.
        """
        task_categories = task_categories or {}

        # Compute analyses
        h2h_report = self.h2h.compare(rlm_results, standard_results, task_categories)
        cost_comparison = self.cost_model.compare_systems(rlm_results, standard_results)
        emergence_report = self.emergence.analyze(rlm_results, task_categories)

        sections = [
            self._header(),
            self._overview(rlm_results, standard_results),
            self._head_to_head(h2h_report),
            self._emergence(emergence_report),
            self._cost(cost_comparison, rlm_results, task_categories),
            self._strategies(rlm_results, task_categories),
            self._examples(rlm_results),
            self._conclusions(h2h_report, emergence_report),
        ]

        return "\n\n".join(sections)

    def _header(self) -> str:
        return "# RLM Evaluation Report\n\nComprehensive analysis of RLM vs Standard LLM performance."

    def _overview(
        self,
        rlm_results: List[EvalResult],
        standard_results: List[EvalResult],
    ) -> str:
        rlm_acc = sum(1 for r in rlm_results if r.correct) / len(rlm_results) if rlm_results else 0
        std_acc = sum(1 for r in standard_results if r.correct) / len(standard_results) if standard_results else 0

        return (
            f"## Overview\n\n"
            f"- RLM tasks evaluated: {len(rlm_results)}\n"
            f"- Standard tasks evaluated: {len(standard_results)}\n"
            f"- RLM accuracy: {rlm_acc:.1%}\n"
            f"- Standard accuracy: {std_acc:.1%}"
        )

    def _head_to_head(self, report: HeadToHeadReport) -> str:
        return f"## Head-to-Head Comparison\n\n{report.summary()}"

    def _emergence(self, report: EmergenceReport) -> str:
        return f"## Emergence Analysis\n\n{report.summary()}"

    def _cost(
        self,
        comparison: CostComparison,
        results: List[EvalResult],
        categories: Dict[str, str],
    ) -> str:
        cost_table = self.cost_analysis.cost_summary_table(results, categories)
        return f"## Cost Analysis\n\n{comparison.summary()}\n\n{cost_table}"

    def _strategies(
        self,
        results: List[EvalResult],
        categories: Dict[str, str],
    ) -> str:
        table = self.landscape.distribution_table(results, categories)
        return f"## Strategy Distribution\n\n```\n{table}\n```"

    def _examples(self, results: List[EvalResult]) -> str:
        selected = self.visualizer.select_representative(results, num_examples=3)
        rendered = self.visualizer.render_batch(selected, max_show=3)
        return f"## Example Trajectories\n\n```\n{rendered}\n```"

    def _conclusions(
        self,
        h2h: HeadToHeadReport,
        emergence: EmergenceReport,
    ) -> str:
        lines = [
            "## Conclusions",
            "",
            f"- RLM win rate: {h2h.rlm_win_rate:.1%}",
            f"- Strategy adaptation score: {emergence.adaptation_score:.2f}",
            f"- Grep-before-read rate: {emergence.grep_before_read_rate:.1%}",
        ]
        if h2h.advantage_2x:
            lines.append("- 2x accuracy advantage confirmed for long-context tasks")
        return "\n".join(lines)
