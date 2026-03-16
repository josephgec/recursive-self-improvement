"""Task difficulty analysis based on search performance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.arc.difficulty import DifficultyEstimate, estimate_difficulty
from src.arc.grid import ARCTask
from src.search.engine import SearchResult


@dataclass
class TaskAnalysis:
    """Combined analysis of task difficulty and search performance."""

    task_id: str
    estimated_difficulty: DifficultyEstimate
    actual_solved: bool = False
    generations_to_solve: Optional[int] = None
    best_fitness_achieved: float = 0.0
    total_evaluations: int = 0
    search_time: float = 0.0

    @property
    def difficulty_accuracy(self) -> Optional[float]:
        """How well the estimated difficulty predicted actual performance."""
        if self.generations_to_solve is None:
            return None
        # Compare estimated difficulty with actual generations needed
        actual_difficulty = min(self.generations_to_solve / 100.0, 1.0)
        return 1.0 - abs(self.estimated_difficulty.score - actual_difficulty)


class TaskDifficultyAnalyzer:
    """Analyzes task difficulty based on structural features and search results."""

    def __init__(self):
        self._analyses: Dict[str, TaskAnalysis] = {}

    def analyze_task(
        self,
        task: ARCTask,
        search_result: Optional[SearchResult] = None,
    ) -> TaskAnalysis:
        """Analyze a task's difficulty."""
        estimate = estimate_difficulty(task)

        analysis = TaskAnalysis(
            task_id=task.task_id,
            estimated_difficulty=estimate,
        )

        if search_result:
            analysis.actual_solved = search_result.solved
            analysis.best_fitness_achieved = search_result.best_fitness
            analysis.total_evaluations = search_result.total_evaluations
            analysis.search_time = search_result.elapsed_seconds

            if search_result.solved and search_result.history:
                # Find generation where fitness reached target
                for entry in search_result.history:
                    if entry.get("best_fitness", 0) >= 0.99:
                        analysis.generations_to_solve = entry.get("generation", 0)
                        break

        self._analyses[task.task_id] = analysis
        return analysis

    def get_analysis(self, task_id: str) -> Optional[TaskAnalysis]:
        """Get analysis for a specific task."""
        return self._analyses.get(task_id)

    def rank_by_difficulty(self) -> List[TaskAnalysis]:
        """Rank tasks by estimated difficulty."""
        return sorted(
            self._analyses.values(),
            key=lambda a: a.estimated_difficulty.score,
        )

    def rank_by_actual_difficulty(self) -> List[TaskAnalysis]:
        """Rank tasks by actual search performance (hardest first)."""
        return sorted(
            self._analyses.values(),
            key=lambda a: a.best_fitness_achieved,
        )

    def difficulty_correlation(self) -> Optional[float]:
        """Compute correlation between estimated and actual difficulty."""
        analyses = [
            a for a in self._analyses.values()
            if a.best_fitness_achieved > 0
        ]
        if len(analyses) < 2:
            return None

        estimated = [a.estimated_difficulty.score for a in analyses]
        actual = [1.0 - a.best_fitness_achieved for a in analyses]

        # Pearson correlation
        n = len(estimated)
        mean_e = sum(estimated) / n
        mean_a = sum(actual) / n

        num = sum((e - mean_e) * (a - mean_a) for e, a in zip(estimated, actual))
        den_e = sum((e - mean_e) ** 2 for e in estimated) ** 0.5
        den_a = sum((a - mean_a) ** 2 for a in actual) ** 0.5

        if den_e * den_a == 0:
            return 0.0

        return num / (den_e * den_a)

    def summary(self) -> Dict[str, Any]:
        """Generate difficulty analysis summary."""
        if not self._analyses:
            return {"num_tasks": 0}

        solved = sum(1 for a in self._analyses.values() if a.actual_solved)
        return {
            "num_tasks": len(self._analyses),
            "num_solved": solved,
            "solve_rate": solved / len(self._analyses),
            "avg_estimated_difficulty": (
                sum(a.estimated_difficulty.score for a in self._analyses.values())
                / len(self._analyses)
            ),
            "avg_best_fitness": (
                sum(a.best_fitness_achieved for a in self._analyses.values())
                / len(self._analyses)
            ),
            "difficulty_correlation": self.difficulty_correlation(),
        }

    def clear(self) -> None:
        """Clear all analyses."""
        self._analyses.clear()
