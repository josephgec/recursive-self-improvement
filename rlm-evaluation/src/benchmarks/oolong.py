"""OOLONG benchmark: long-context evaluation tasks across multiple categories."""

from __future__ import annotations

import hashlib
import random
from typing import List

from src.benchmarks.task import EvalTask


class OOLONGBenchmark:
    """OOLONG benchmark with retrieval, aggregation, reasoning, and counting tasks."""

    name = "oolong"

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def load(self) -> List[EvalTask]:
        """Load all OOLONG tasks."""
        tasks: List[EvalTask] = []
        tasks.extend(self._retrieval_tasks())
        tasks.extend(self._aggregation_tasks())
        tasks.extend(self._reasoning_tasks())
        tasks.extend(self._counting_tasks())
        return tasks

    def _make_context(self, entries: List[str], seed_val: int = 0) -> str:
        """Build a long context from entries with filler."""
        rng = random.Random(self.seed + seed_val)
        filler_sentences = [
            "The weather was mild that day.",
            "Several participants joined the meeting.",
            "The report was filed on time.",
            "No anomalies were detected in the data.",
            "The system operated within normal parameters.",
            "A new version was released last quarter.",
            "The team completed the sprint goals.",
            "Documentation was updated accordingly.",
        ]
        blocks = []
        for entry in entries:
            num_filler = rng.randint(5, 15)
            filler = " ".join(rng.choices(filler_sentences, k=num_filler))
            blocks.append(filler + "\n" + entry + "\n")
        rng.shuffle(blocks)
        return "\n".join(blocks)

    def _retrieval_tasks(self) -> List[EvalTask]:
        """Generate retrieval tasks: find specific facts in long context."""
        tasks = []
        people = [
            ("Alice", "Berlin", "engineer"),
            ("Bob", "Tokyo", "designer"),
            ("Carol", "London", "manager"),
            ("Dave", "Paris", "analyst"),
            ("Eve", "Sydney", "researcher"),
        ]
        entries = [f"Employee record: {name} lives in {city} and works as a {role}."
                   for name, city, role in people]
        context = self._make_context(entries, seed_val=1)

        for i, (name, city, role) in enumerate(people):
            tasks.append(EvalTask(
                task_id=f"oolong_retrieval_{i}",
                benchmark="oolong",
                query=f"Where does {name} live?",
                context=context,
                expected_answer=city,
                category="retrieval",
                difficulty="easy",
            ))

        # Additional retrieval tasks
        facts = [
            ("Project Alpha started on 2024-01-15.", "When did Project Alpha start?", "2024-01-15"),
            ("The server capacity is 128 TB.", "What is the server capacity?", "128 TB"),
            ("The CEO is named Dr. Sarah Chen.", "Who is the CEO?", "Dr. Sarah Chen"),
        ]
        fact_entries = [f[0] for f in facts]
        fact_context = self._make_context(fact_entries, seed_val=2)
        for i, (_, question, answer) in enumerate(facts):
            tasks.append(EvalTask(
                task_id=f"oolong_retrieval_fact_{i}",
                benchmark="oolong",
                query=question,
                context=fact_context,
                expected_answer=answer,
                category="retrieval",
                difficulty="easy",
            ))
        return tasks

    def _aggregation_tasks(self) -> List[EvalTask]:
        """Generate aggregation tasks: combine information from multiple locations."""
        tasks = []
        rng = random.Random(self.seed + 10)

        # Sales aggregation
        regions = ["North", "South", "East", "West", "Central"]
        sales = {r: rng.randint(100, 500) for r in regions}
        entries = [f"Region {r} reported quarterly sales of ${v}k." for r, v in sales.items()]
        context = self._make_context(entries, seed_val=10)

        tasks.append(EvalTask(
            task_id="oolong_agg_total_sales",
            benchmark="oolong",
            query="What are the total quarterly sales across all regions (in $k)?",
            context=context,
            expected_answer=str(sum(sales.values())),
            category="aggregation",
            difficulty="medium",
        ))
        tasks.append(EvalTask(
            task_id="oolong_agg_max_region",
            benchmark="oolong",
            query="Which region had the highest quarterly sales?",
            context=context,
            expected_answer=max(sales, key=lambda r: sales[r]),
            category="aggregation",
            difficulty="medium",
        ))

        # Team size aggregation
        teams = {"Engineering": 25, "Marketing": 12, "Sales": 18, "Support": 15, "Research": 8}
        team_entries = [f"Department {t} has {s} members." for t, s in teams.items()]
        team_context = self._make_context(team_entries, seed_val=11)

        tasks.append(EvalTask(
            task_id="oolong_agg_total_employees",
            benchmark="oolong",
            query="How many total employees are there across all departments?",
            context=team_context,
            expected_answer=str(sum(teams.values())),
            category="aggregation",
            difficulty="medium",
        ))
        tasks.append(EvalTask(
            task_id="oolong_agg_smallest_dept",
            benchmark="oolong",
            query="Which department has the fewest members?",
            context=team_context,
            expected_answer="Research",
            category="aggregation",
            difficulty="medium",
        ))
        return tasks

    def _reasoning_tasks(self) -> List[EvalTask]:
        """Generate reasoning tasks: derive conclusions from scattered evidence."""
        tasks = []

        # Multi-hop reasoning
        clues = [
            "Alice manages the team that Bob is on.",
            "Bob works in the Engineering department.",
            "The Engineering department is on the 5th floor.",
            "The 5th floor has a break room with a coffee machine.",
        ]
        context = self._make_context(clues, seed_val=20)

        tasks.append(EvalTask(
            task_id="oolong_reason_multi_hop",
            benchmark="oolong",
            query="What floor does Alice's team work on?",
            context=context,
            expected_answer="5th floor",
            category="reasoning",
            difficulty="hard",
        ))

        # Timeline reasoning
        events = [
            "Project X was approved on March 1.",
            "Development started 2 weeks after approval.",
            "The first milestone was reached 4 weeks after development started.",
            "Testing began 1 week after the first milestone.",
        ]
        context2 = self._make_context(events, seed_val=21)

        tasks.append(EvalTask(
            task_id="oolong_reason_timeline",
            benchmark="oolong",
            query="How many weeks after approval did testing begin?",
            context=context2,
            expected_answer="7",
            category="reasoning",
            difficulty="hard",
        ))

        # Logical deduction
        rules = [
            "All senior engineers have access to the production server.",
            "Dave is a senior engineer.",
            "Only people with production access can deploy code.",
            "Eve does not have production access.",
        ]
        context3 = self._make_context(rules, seed_val=22)

        tasks.append(EvalTask(
            task_id="oolong_reason_logic_1",
            benchmark="oolong",
            query="Can Dave deploy code?",
            context=context3,
            expected_answer="yes",
            category="reasoning",
            difficulty="hard",
        ))
        tasks.append(EvalTask(
            task_id="oolong_reason_logic_2",
            benchmark="oolong",
            query="Can Eve deploy code?",
            context=context3,
            expected_answer="no",
            category="reasoning",
            difficulty="hard",
        ))
        return tasks

    def _counting_tasks(self) -> List[EvalTask]:
        """Generate counting tasks: count occurrences in long context."""
        tasks = []
        rng = random.Random(self.seed + 30)

        # Count mentions
        names = ["Alice", "Bob", "Carol"]
        mention_counts = {n: rng.randint(3, 8) for n in names}
        lines = []
        for name, count in mention_counts.items():
            for _ in range(count):
                lines.append(f"{name} attended the meeting on {rng.choice(['Monday', 'Tuesday', 'Wednesday'])}.")
        rng.shuffle(lines)
        context = self._make_context(lines, seed_val=30)

        for name, count in mention_counts.items():
            tasks.append(EvalTask(
                task_id=f"oolong_count_{name.lower()}",
                benchmark="oolong",
                query=f"How many times is {name} mentioned as attending a meeting?",
                context=context,
                expected_answer=str(count),
                category="counting",
                difficulty="medium",
            ))

        # Count unique items
        colors = ["red", "blue", "green", "yellow", "purple"]
        items = [f"Item #{rng.randint(1000,9999)} is colored {rng.choice(colors)}." for _ in range(20)]
        unique_colors = len(set(c.split("colored ")[1].rstrip(".") for c in items))
        item_context = self._make_context(items, seed_val=31)

        tasks.append(EvalTask(
            task_id="oolong_count_colors",
            benchmark="oolong",
            query="How many unique colors are mentioned in the item list?",
            context=item_context,
            expected_answer=str(unique_colors),
            category="counting",
            difficulty="medium",
        ))
        return tasks


def _content_hash(s: str) -> str:
    """Create a short hash of content for IDs."""
    return hashlib.md5(s.encode()).hexdigest()[:8]
