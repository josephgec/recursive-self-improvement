"""Qualitative analysis: select and annotate notable examples."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.benchmarks.registry import BenchmarkResult


@dataclass
class AnnotatedExample:
    """An annotated qualitative example."""
    task_id: str
    benchmark: str
    iteration: int
    category: str
    predicted: Any
    expected: Any
    correct: bool
    annotation: str
    tags: List[str] = field(default_factory=list)


def select_examples(
    results_by_iteration: Dict[int, List[BenchmarkResult]],
    max_examples: int = 10,
) -> List[AnnotatedExample]:
    """Select notable examples for qualitative analysis.

    Selects examples that changed correctness between iterations.
    """
    examples: List[AnnotatedExample] = []
    iterations = sorted(results_by_iteration.keys())

    if len(iterations) < 2:
        # Just pick some examples from the only iteration
        if iterations:
            for r in results_by_iteration[iterations[0]][:max_examples]:
                examples.append(AnnotatedExample(
                    task_id=r.task_id,
                    benchmark=r.benchmark,
                    iteration=iterations[0],
                    category="baseline",
                    predicted=r.predicted_answer,
                    expected=r.expected_answer,
                    correct=r.correct,
                    annotation="Initial baseline result",
                    tags=["baseline"],
                ))
        return examples

    # Find tasks that changed between first and last iteration
    first_results = {r.task_id: r for r in results_by_iteration[iterations[0]]}
    last_results = {r.task_id: r for r in results_by_iteration[iterations[-1]]}

    for task_id in first_results:
        if task_id in last_results:
            first = first_results[task_id]
            last = last_results[task_id]

            if not first.correct and last.correct:
                examples.append(AnnotatedExample(
                    task_id=task_id,
                    benchmark=last.benchmark,
                    iteration=iterations[-1],
                    category="improvement",
                    predicted=last.predicted_answer,
                    expected=last.expected_answer,
                    correct=True,
                    annotation="Task solved after RSI iterations",
                    tags=["improvement", "positive"],
                ))
            elif first.correct and not last.correct:
                examples.append(AnnotatedExample(
                    task_id=task_id,
                    benchmark=last.benchmark,
                    iteration=iterations[-1],
                    category="regression",
                    predicted=last.predicted_answer,
                    expected=last.expected_answer,
                    correct=False,
                    annotation="Task regressed after RSI iterations",
                    tags=["regression", "negative"],
                ))

        if len(examples) >= max_examples:
            break

    return examples[:max_examples]


def annotate(
    example: AnnotatedExample,
    annotation: str,
    tags: Optional[List[str]] = None,
) -> AnnotatedExample:
    """Add annotation to an example."""
    example.annotation = annotation
    if tags:
        example.tags.extend(tags)
    return example
