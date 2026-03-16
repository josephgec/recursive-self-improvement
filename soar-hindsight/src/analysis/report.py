"""Generate comprehensive reports from SOAR loop runs."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from src.analysis.data_quality import DataQualityAnalyzer
from src.analysis.iteration_dynamics import IterationDynamicsAnalyzer
from src.analysis.transfer_analysis import TransferAnalyzer
from src.synthesis.synthesizer import TrainingPair


class ReportGenerator:
    """Generate comprehensive reports from SOAR loop execution."""

    def __init__(self, output_dir: str = "data/reports"):
        self.output_dir = output_dir

    def generate(
        self,
        pairs: Optional[List[TrainingPair]] = None,
        iteration_history: Optional[List[Dict[str, Any]]] = None,
        eval_results: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """Generate a full report combining all analyses."""
        report: Dict[str, Any] = {"sections": {}}

        # Data quality
        if pairs:
            dqa = DataQualityAnalyzer(pairs)
            report["sections"]["data_quality"] = dqa.full_report()

        # Iteration dynamics
        if iteration_history:
            ida = IterationDynamicsAnalyzer()
            ida.load_history(iteration_history)
            report["sections"]["iteration_dynamics"] = ida.full_report()

        # Transfer analysis
        if pairs:
            ta = TransferAnalyzer()
            ta.load_pairs(pairs)
            if eval_results:
                ta.load_eval_results(eval_results)
            report["sections"]["transfer"] = ta.summary()

        # Summary
        report["summary"] = self._create_summary(report["sections"])

        return report

    def save(self, report: Dict[str, Any], filename: str = "report.json") -> str:
        """Save report to JSON file."""
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)
        return filepath

    def format_text(self, report: Dict[str, Any]) -> str:
        """Format report as human-readable text."""
        lines = ["=" * 60, "SOAR Hindsight Report", "=" * 60, ""]

        summary = report.get("summary", {})
        lines.append(f"Total training pairs: {summary.get('total_pairs', 'N/A')}")
        lines.append(f"Total iterations: {summary.get('total_iterations', 'N/A')}")
        lines.append(f"Final solve rate: {summary.get('final_solve_rate', 'N/A')}")
        lines.append(f"Converged: {summary.get('converged', 'N/A')}")
        lines.append("")

        # Data quality section
        dq = report.get("sections", {}).get("data_quality", {})
        if dq:
            lines.append("-" * 40)
            lines.append("Data Quality")
            lines.append("-" * 40)
            ts = dq.get("token_stats", {})
            lines.append(f"  Avg prompt tokens: {ts.get('prompt_mean', 'N/A')}")
            lines.append(f"  Avg completion tokens: {ts.get('completion_mean', 'N/A')}")
            sb = dq.get("strategy_breakdown", {})
            for strat, info in sb.items():
                lines.append(f"  {strat}: {info.get('count', 0)} pairs, avg quality {info.get('avg_quality', 0)}")
            lines.append("")

        # Iteration dynamics section
        itr = report.get("sections", {}).get("iteration_dynamics", {})
        if itr:
            lines.append("-" * 40)
            lines.append("Iteration Dynamics")
            lines.append("-" * 40)
            conv = itr.get("convergence", {})
            lines.append(f"  Converged: {conv.get('converged', 'N/A')}")
            lines.append(f"  Total improvement: {conv.get('total_improvement', 'N/A')}")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _create_summary(sections: Dict[str, Any]) -> Dict[str, Any]:
        """Create a top-level summary from section data."""
        summary: Dict[str, Any] = {}

        dq = sections.get("data_quality", {})
        if dq:
            summary["total_pairs"] = dq.get("total_pairs", 0)

        itr = sections.get("iteration_dynamics", {})
        if itr:
            summary["total_iterations"] = itr.get("n_iterations", 0)
            conv = itr.get("convergence", {})
            summary["converged"] = conv.get("converged", False)
            summary["final_solve_rate"] = conv.get("final_solve_rate", 0.0)

        return summary
