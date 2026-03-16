"""OOLONG benchmark: long-context evaluation tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils.context_generators import generate_haystack, generate_document_collection


@dataclass
class OOLONGTask:
    """A single OOLONG benchmark task."""
    task_id: str
    category: str  # "retrieval", "aggregation", "reasoning"
    query: str
    context: str
    expected_answer: str
    difficulty: str = "medium"  # easy, medium, hard
    metadata: Dict[str, Any] = field(default_factory=dict)


class OOLONGBenchmark:
    """Built-in OOLONG-style benchmark with 20+ tasks across categories."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._tasks: Optional[List[OOLONGTask]] = None

    @property
    def tasks(self) -> List[OOLONGTask]:
        if self._tasks is None:
            self._tasks = self._build_tasks()
        return self._tasks

    def get_by_category(self, category: str) -> List[OOLONGTask]:
        """Return tasks filtered by category."""
        return [t for t in self.tasks if t.category == category]

    def get_by_difficulty(self, difficulty: str) -> List[OOLONGTask]:
        """Return tasks filtered by difficulty."""
        return [t for t in self.tasks if t.difficulty == difficulty]

    def _build_tasks(self) -> List[OOLONGTask]:
        tasks: List[OOLONGTask] = []

        # --- Retrieval tasks ---
        for i in range(8):
            needle = f"SECRET_CODE_{i}: alpha-{100 + i}"
            haystack = generate_haystack(
                needle=needle,
                haystack_size=5000 + i * 1000,
                position=0.1 * (i + 1),
                seed=self.seed + i,
            )
            tasks.append(OOLONGTask(
                task_id=f"oolong_retrieval_{i + 1}",
                category="retrieval",
                query=f"Find the line containing SECRET_CODE_{i} and report its value.",
                context=haystack,
                expected_answer=f"alpha-{100 + i}",
                difficulty=["easy", "medium", "hard"][min(i // 3, 2)],
                metadata={"needle": needle, "position": 0.1 * (i + 1)},
            ))

        # --- Aggregation tasks ---
        docs = generate_document_collection(num_docs=10, doc_size=500, seed=self.seed)
        full_text = "\n\n---\n\n".join(d["body"] for d in docs)
        for i in range(7):
            if i < 3:
                query = f"How many documents mention 'implementation'?"
                expected = str(sum(1 for d in docs if "implementation" in d["body"].lower()))
            elif i < 5:
                query = f"List all document titles."
                expected = "\n".join(d["title"] for d in docs)
            else:
                query = f"Count the total number of words across all documents."
                expected = str(sum(len(d["body"].split()) for d in docs))
            tasks.append(OOLONGTask(
                task_id=f"oolong_aggregation_{i + 1}",
                category="aggregation",
                query=query,
                context=full_text,
                expected_answer=expected,
                difficulty=["easy", "medium", "hard"][min(i // 3, 2)],
            ))

        # --- Reasoning tasks ---
        reasoning_contexts = [
            (
                "Alice is taller than Bob. Bob is taller than Charlie. "
                "Dave is shorter than Charlie. Eve is taller than Alice.",
                "Who is the tallest person?",
                "Eve",
            ),
            (
                "The price of apples is $2. The price of bananas is twice the "
                "price of apples. The price of cherries is $1 more than bananas.",
                "What is the total cost of 1 apple, 1 banana, and 1 cherry?",
                "11",
            ),
            (
                "Server A has 100 requests/sec. Server B has 150 requests/sec. "
                "Server C has 200 requests/sec. Load balancer distributes evenly.",
                "What is the total throughput in requests/sec?",
                "450",
            ),
            (
                "Temperature readings: Mon=72F, Tue=68F, Wed=75F, Thu=70F, Fri=73F, "
                "Sat=80F, Sun=77F.",
                "What is the average temperature for the week (rounded)?",
                "74",
            ),
            (
                "Team scores: Alpha=85, Beta=92, Gamma=78, Delta=95, Epsilon=88.",
                "Which team has the highest score?",
                "Delta",
            ),
            (
                "Project timeline: Phase1=2weeks, Phase2=3weeks, Phase3=1week, "
                "Phase4=4weeks. Phases are sequential.",
                "How many weeks total for the project?",
                "10",
            ),
        ]
        for i, (ctx, query, expected) in enumerate(reasoning_contexts):
            padded = generate_haystack(ctx, haystack_size=3000, seed=self.seed + 100 + i)
            tasks.append(OOLONGTask(
                task_id=f"oolong_reasoning_{i + 1}",
                category="reasoning",
                query=query,
                context=padded,
                expected_answer=expected,
                difficulty=["easy", "medium", "hard"][min(i // 2, 2)],
            ))

        return tasks

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)
