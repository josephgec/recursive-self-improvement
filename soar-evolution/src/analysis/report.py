"""Report generation for search results."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.analysis.search_dynamics import SearchDynamicsAnalyzer
from src.analysis.operator_effectiveness import OperatorEffectivenessAnalyzer
from src.analysis.task_difficulty import TaskDifficultyAnalyzer
from src.search.engine import SearchResult


class ReportGenerator:
    """Generates comprehensive reports from search results."""

    def __init__(
        self,
        dynamics_analyzer: Optional[SearchDynamicsAnalyzer] = None,
        operator_analyzer: Optional[OperatorEffectivenessAnalyzer] = None,
        difficulty_analyzer: Optional[TaskDifficultyAnalyzer] = None,
    ):
        self.dynamics = dynamics_analyzer or SearchDynamicsAnalyzer()
        self.operators = operator_analyzer or OperatorEffectivenessAnalyzer()
        self.difficulty = difficulty_analyzer or TaskDifficultyAnalyzer()

    def generate_search_report(
        self,
        result: SearchResult,
        task_id: str = "unknown",
    ) -> Dict[str, Any]:
        """Generate a report for a single search run."""
        report: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "solved": result.solved,
            "best_fitness": result.best_fitness,
            "generations": result.generations_run,
            "total_evaluations": result.total_evaluations,
            "elapsed_seconds": result.elapsed_seconds,
            "stop_reason": result.stop_reason,
        }

        if result.best_individual:
            report["best_program"] = {
                "code": result.best_individual.code,
                "fitness": result.best_individual.fitness,
                "train_accuracy": result.best_individual.train_accuracy,
                "test_accuracy": result.best_individual.test_accuracy,
                "generation": result.best_individual.generation,
            }

        if result.history:
            report["fitness_trajectory"] = [
                h.get("best_fitness", 0.0) for h in result.history
            ]

        return report

    def generate_benchmark_report(
        self,
        results: Dict[str, SearchResult],
    ) -> Dict[str, Any]:
        """Generate a report for a benchmark run across multiple tasks."""
        task_results = {}
        total_solved = 0
        total_time = 0.0

        for task_id, result in results.items():
            task_results[task_id] = {
                "solved": result.solved,
                "best_fitness": result.best_fitness,
                "generations": result.generations_run,
                "time": result.elapsed_seconds,
            }
            if result.solved:
                total_solved += 1
            total_time += result.elapsed_seconds

        return {
            "timestamp": datetime.now().isoformat(),
            "num_tasks": len(results),
            "num_solved": total_solved,
            "solve_rate": total_solved / max(len(results), 1),
            "total_time": total_time,
            "avg_time_per_task": total_time / max(len(results), 1),
            "tasks": task_results,
        }

    def generate_full_report(
        self,
        results: Dict[str, SearchResult],
    ) -> Dict[str, Any]:
        """Generate a comprehensive report including all analyses."""
        report = self.generate_benchmark_report(results)

        report["dynamics"] = self.dynamics.summary()
        report["operators"] = self.operators.summary()
        report["difficulty"] = self.difficulty.summary()

        return report

    def save_report(
        self,
        report: Dict[str, Any],
        path: str,
    ) -> None:
        """Save report to JSON file."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)

    def format_text_report(self, report: Dict[str, Any]) -> str:
        """Format report as human-readable text."""
        lines = ["=" * 60]
        lines.append("SOAR-Evolution Search Report")
        lines.append("=" * 60)
        lines.append("")

        lines.append(f"Timestamp: {report.get('timestamp', 'N/A')}")
        lines.append(f"Tasks: {report.get('num_tasks', 'N/A')}")
        lines.append(f"Solved: {report.get('num_solved', 'N/A')}")
        lines.append(f"Solve Rate: {report.get('solve_rate', 0):.1%}")
        lines.append(f"Total Time: {report.get('total_time', 0):.1f}s")
        lines.append("")

        if "tasks" in report:
            lines.append("Task Results:")
            lines.append("-" * 40)
            for task_id, task_data in report["tasks"].items():
                status = "SOLVED" if task_data.get("solved") else "UNSOLVED"
                fitness = task_data.get("best_fitness", 0)
                lines.append(
                    f"  {task_id}: {status} (fitness={fitness:.4f})"
                )

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)
