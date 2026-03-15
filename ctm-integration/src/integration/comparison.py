"""Comparison of augmented vs. standard prompting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.integration.augmented_prompt import AugmentedPromptBuilder
from src.library.store import RuleStore


@dataclass
class ComparisonResult:
    """Result of comparing augmented vs. standard prompt for a single task."""

    task: str
    augmented_prompt: str
    standard_prompt: str
    augmented_length: int = 0
    standard_length: int = 0
    rules_included: int = 0

    @property
    def length_ratio(self) -> float:
        if self.standard_length == 0:
            return 0.0
        return self.augmented_length / self.standard_length


@dataclass
class ComparisonAnalysis:
    """Analysis of multiple comparison results."""

    results: List[ComparisonResult] = field(default_factory=list)
    avg_length_ratio: float = 0.0
    avg_rules_included: float = 0.0
    tasks_with_rules: int = 0
    total_tasks: int = 0

    @property
    def augmentation_rate(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return self.tasks_with_rules / self.total_tasks


class AugmentationComparison:
    """Compares augmented prompts with standard prompts across multiple tasks."""

    def __init__(
        self,
        builder: Optional[AugmentedPromptBuilder] = None,
        store: Optional[RuleStore] = None,
    ) -> None:
        self.store = store or RuleStore()
        self.builder = builder or AugmentedPromptBuilder(store=self.store)

    def run_comparison(
        self, tasks: List[str]
    ) -> List[ComparisonResult]:
        """Run comparison for a list of tasks.

        Args:
            tasks: List of task descriptions.

        Returns:
            List of ComparisonResult objects.
        """
        results = []

        for task in tasks:
            augmented = self.builder.build_prompt(task)
            standard = self.builder.build_standard_prompt(task)

            # Count rules included
            rules_count = augmented.count("### Rule")

            result = ComparisonResult(
                task=task,
                augmented_prompt=augmented,
                standard_prompt=standard,
                augmented_length=len(augmented),
                standard_length=len(standard),
                rules_included=rules_count,
            )
            results.append(result)

        return results

    def analyze(
        self, results: List[ComparisonResult]
    ) -> ComparisonAnalysis:
        """Analyze comparison results.

        Args:
            results: List of comparison results.

        Returns:
            ComparisonAnalysis with aggregate statistics.
        """
        if not results:
            return ComparisonAnalysis()

        total = len(results)
        length_ratios = [r.length_ratio for r in results]
        rules_counts = [r.rules_included for r in results]
        tasks_with = sum(1 for r in results if r.rules_included > 0)

        return ComparisonAnalysis(
            results=results,
            avg_length_ratio=sum(length_ratios) / total,
            avg_rules_included=sum(rules_counts) / total,
            tasks_with_rules=tasks_with,
            total_tasks=total,
        )
