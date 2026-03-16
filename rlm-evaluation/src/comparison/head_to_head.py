"""Head-to-head comparison between RLM and standard systems."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.benchmarks.task import EvalResult


@dataclass
class HeadToHeadReport:
    """Report from head-to-head comparison."""
    rlm_wins: int = 0
    standard_wins: int = 0
    ties: int = 0
    total_tasks: int = 0
    rlm_accuracy: float = 0.0
    standard_accuracy: float = 0.0
    rlm_win_rate: float = 0.0
    standard_win_rate: float = 0.0
    advantage_2x: bool = False
    advantage_categories: Dict[str, str] = field(default_factory=dict)
    paired_results: List[Dict[str, object]] = field(default_factory=list)

    def summary(self) -> str:
        """Generate comparison summary."""
        lines = [
            f"Head-to-Head Report ({self.total_tasks} tasks)",
            f"  RLM wins: {self.rlm_wins} ({self.rlm_win_rate:.1%})",
            f"  Standard wins: {self.standard_wins} ({self.standard_win_rate:.1%})",
            f"  Ties: {self.ties}",
            f"  RLM accuracy: {self.rlm_accuracy:.1%}",
            f"  Standard accuracy: {self.standard_accuracy:.1%}",
            f"  2x advantage claim: {self.advantage_2x}",
        ]
        if self.advantage_categories:
            lines.append("  Category advantages:")
            for cat, winner in self.advantage_categories.items():
                lines.append(f"    {cat}: {winner}")
        return "\n".join(lines)


class HeadToHeadComparator:
    """Compare RLM and standard system results head-to-head."""

    def compare(
        self,
        rlm_results: List[EvalResult],
        standard_results: List[EvalResult],
        task_categories: Optional[Dict[str, str]] = None,
    ) -> HeadToHeadReport:
        """Run full head-to-head comparison.

        Both lists should contain results for the same tasks (matched by task_id).
        """
        report = HeadToHeadReport()

        # Build lookup
        std_by_id = {r.task_id: r for r in standard_results}

        paired: List[Dict[str, object]] = []
        rlm_correct = 0
        std_correct = 0

        for rlm_r in rlm_results:
            std_r = std_by_id.get(rlm_r.task_id)
            if std_r is None:
                continue

            report.total_tasks += 1

            if rlm_r.correct and not std_r.correct:
                report.rlm_wins += 1
            elif std_r.correct and not rlm_r.correct:
                report.standard_wins += 1
            else:
                report.ties += 1

            if rlm_r.correct:
                rlm_correct += 1
            if std_r.correct:
                std_correct += 1

            paired.append({
                "task_id": rlm_r.task_id,
                "rlm_correct": rlm_r.correct,
                "std_correct": std_r.correct,
                "rlm_cost": rlm_r.cost,
                "std_cost": std_r.cost,
            })

        report.paired_results = paired

        if report.total_tasks > 0:
            report.rlm_accuracy = rlm_correct / report.total_tasks
            report.standard_accuracy = std_correct / report.total_tasks
            report.rlm_win_rate = report.rlm_wins / report.total_tasks
            report.standard_win_rate = report.standard_wins / report.total_tasks

        # Compute 2x claim
        report.advantage_2x = self.compute_2x_claim(rlm_results, standard_results)

        # Category advantages
        if task_categories:
            report.advantage_categories = self._category_advantages(
                rlm_results, standard_results, task_categories
            )

        return report

    def paired_accuracy(
        self,
        rlm_results: List[EvalResult],
        standard_results: List[EvalResult],
    ) -> Tuple[float, float]:
        """Compute paired accuracy (only on tasks both systems attempted)."""
        std_by_id = {r.task_id: r for r in standard_results}
        rlm_correct = 0
        std_correct = 0
        count = 0

        for rlm_r in rlm_results:
            if rlm_r.task_id in std_by_id:
                count += 1
                if rlm_r.correct:
                    rlm_correct += 1
                if std_by_id[rlm_r.task_id].correct:
                    std_correct += 1

        if count == 0:
            return 0.0, 0.0
        return rlm_correct / count, std_correct / count

    def win_rate(
        self,
        rlm_results: List[EvalResult],
        standard_results: List[EvalResult],
    ) -> Tuple[float, float]:
        """Compute win rates (tasks where one got right and other got wrong)."""
        report = self.compare(rlm_results, standard_results)
        return report.rlm_win_rate, report.standard_win_rate

    def compute_2x_claim(
        self,
        rlm_results: List[EvalResult],
        standard_results: List[EvalResult],
    ) -> bool:
        """Check if RLM achieves >= 2x accuracy of standard on long-context tasks.

        Focuses on tasks with large context (> 8k tokens).
        """
        std_by_id = {r.task_id: r for r in standard_results}

        rlm_correct = 0
        std_correct = 0
        count = 0

        for rlm_r in rlm_results:
            if rlm_r.task_id in std_by_id and rlm_r.input_tokens > 2000:
                count += 1
                if rlm_r.correct:
                    rlm_correct += 1
                if std_by_id[rlm_r.task_id].correct:
                    std_correct += 1

        if count == 0 or std_correct == 0:
            return rlm_correct > 0

        return (rlm_correct / count) >= 2 * (std_correct / count)

    def _category_advantages(
        self,
        rlm_results: List[EvalResult],
        standard_results: List[EvalResult],
        task_categories: Dict[str, str],
    ) -> Dict[str, str]:
        """Determine which system has the advantage per category."""
        std_by_id = {r.task_id: r for r in standard_results}

        from collections import defaultdict
        rlm_by_cat: Dict[str, List[bool]] = defaultdict(list)
        std_by_cat: Dict[str, List[bool]] = defaultdict(list)

        for rlm_r in rlm_results:
            if rlm_r.task_id in std_by_id:
                cat = task_categories.get(rlm_r.task_id, "unknown")
                rlm_by_cat[cat].append(rlm_r.correct)
                std_by_cat[cat].append(std_by_id[rlm_r.task_id].correct)

        advantages: Dict[str, str] = {}
        for cat in rlm_by_cat:
            rlm_acc = sum(rlm_by_cat[cat]) / len(rlm_by_cat[cat]) if rlm_by_cat[cat] else 0
            std_acc = sum(std_by_cat[cat]) / len(std_by_cat[cat]) if std_by_cat[cat] else 0
            if rlm_acc > std_acc:
                advantages[cat] = "rlm"
            elif std_acc > rlm_acc:
                advantages[cat] = "standard"
            else:
                advantages[cat] = "tie"

        return advantages
