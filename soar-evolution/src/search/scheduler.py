"""Budget allocation for evolutionary search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BudgetAllocation:
    """Budget allocation for a single task."""

    task_id: str
    max_generations: int
    max_evaluations: int
    priority: float = 1.0
    used_generations: int = 0
    used_evaluations: int = 0

    @property
    def generations_remaining(self) -> int:
        return max(0, self.max_generations - self.used_generations)

    @property
    def evaluations_remaining(self) -> int:
        return max(0, self.max_evaluations - self.used_evaluations)

    @property
    def fraction_used(self) -> float:
        gen_frac = self.used_generations / max(self.max_generations, 1)
        eval_frac = self.used_evaluations / max(self.max_evaluations, 1)
        return max(gen_frac, eval_frac)


class BudgetScheduler:
    """Manages computational budget across tasks and generations."""

    def __init__(
        self,
        max_generations: int = 100,
        max_evaluations: int = 10000,
    ):
        self.max_generations = max_generations
        self.max_evaluations = max_evaluations
        self._allocations: Dict[str, BudgetAllocation] = {}
        self._global_used_generations = 0
        self._global_used_evaluations = 0

    def has_budget(self, current_generation: int) -> bool:
        """Check if there's budget remaining."""
        return current_generation < self.max_generations

    def allocate_task(
        self,
        task_id: str,
        priority: float = 1.0,
        max_generations: Optional[int] = None,
        max_evaluations: Optional[int] = None,
    ) -> BudgetAllocation:
        """Allocate budget for a specific task."""
        allocation = BudgetAllocation(
            task_id=task_id,
            max_generations=max_generations or self.max_generations,
            max_evaluations=max_evaluations or self.max_evaluations,
            priority=priority,
        )
        self._allocations[task_id] = allocation
        return allocation

    def record_generation(self, task_id: str) -> None:
        """Record that a generation was used for a task."""
        self._global_used_generations += 1
        if task_id in self._allocations:
            self._allocations[task_id].used_generations += 1

    def record_evaluations(self, task_id: str, count: int) -> None:
        """Record evaluations used for a task."""
        self._global_used_evaluations += count
        if task_id in self._allocations:
            self._allocations[task_id].used_evaluations += count

    def get_allocation(self, task_id: str) -> Optional[BudgetAllocation]:
        """Get the budget allocation for a task."""
        return self._allocations.get(task_id)

    def should_continue(self, task_id: str) -> bool:
        """Check if a specific task should continue searching."""
        alloc = self._allocations.get(task_id)
        if alloc is None:
            return True
        return (
            alloc.generations_remaining > 0
            and alloc.evaluations_remaining > 0
        )

    def redistribute(self, completed_tasks: List[str]) -> None:
        """Redistribute budget from completed tasks to active ones."""
        freed_gens = 0
        freed_evals = 0

        for task_id in completed_tasks:
            alloc = self._allocations.get(task_id)
            if alloc:
                freed_gens += alloc.generations_remaining
                freed_evals += alloc.evaluations_remaining

        # Distribute equally among remaining tasks
        active_tasks = [
            tid
            for tid in self._allocations
            if tid not in completed_tasks
            and self._allocations[tid].generations_remaining > 0
        ]

        if active_tasks and freed_gens > 0:
            bonus_gens = freed_gens // len(active_tasks)
            bonus_evals = freed_evals // len(active_tasks)
            for tid in active_tasks:
                self._allocations[tid].max_generations += bonus_gens
                self._allocations[tid].max_evaluations += bonus_evals

    def summary(self) -> Dict:
        """Return budget usage summary."""
        return {
            "global_generations_used": self._global_used_generations,
            "global_evaluations_used": self._global_used_evaluations,
            "tasks": {
                tid: {
                    "used_gens": a.used_generations,
                    "remaining_gens": a.generations_remaining,
                    "fraction_used": a.fraction_used,
                }
                for tid, a in self._allocations.items()
            },
        }
