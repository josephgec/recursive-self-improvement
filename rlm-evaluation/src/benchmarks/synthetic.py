"""Synthetic task generator with controlled context sizes."""

from __future__ import annotations

import random
from typing import List, Optional

from src.benchmarks.task import EvalTask


class SyntheticTaskGenerator:
    """Generates synthetic tasks with precise control over context size and content."""

    name = "synthetic"

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def load(self) -> List[EvalTask]:
        """Load default synthetic tasks."""
        tasks: List[EvalTask] = []
        tasks.extend(self.needle_in_haystack(context_tokens=4000))
        tasks.extend(self.multi_needle(context_tokens=4000, num_needles=3))
        tasks.extend(self.counting(context_tokens=4000))
        tasks.extend(self.distributed_reasoning(context_tokens=4000))
        return tasks

    def needle_in_haystack(
        self,
        context_tokens: int = 4000,
        num_tasks: int = 3,
        needle_positions: Optional[List[float]] = None,
    ) -> List[EvalTask]:
        """Generate needle-in-haystack tasks.

        A unique fact is placed at a specific position in a large context
        of irrelevant text. The query asks for that fact.
        """
        if needle_positions is None:
            needle_positions = [0.25, 0.5, 0.75]

        rng = random.Random(self.seed + 100)
        tasks = []

        needles = [
            ("The secret code is ALPHA-7749.", "What is the secret code?", "ALPHA-7749"),
            ("The meeting is scheduled for Room 42B.", "Which room is the meeting in?", "Room 42B"),
            ("The password for the vault is diamond-eagle.", "What is the vault password?", "diamond-eagle"),
            ("The launch date is set for November 15th.", "When is the launch date?", "November 15th"),
            ("The project budget is $3.2 million.", "What is the project budget?", "$3.2 million"),
        ]

        filler_sentences = [
            "The quarterly report showed steady progress across all departments.",
            "Team members are encouraged to submit their feedback by Friday.",
            "New guidelines have been issued regarding remote work policies.",
            "The infrastructure upgrade is proceeding according to schedule.",
            "Cross-functional collaboration has improved communication flow.",
            "Performance metrics indicate a positive trend this quarter.",
            "The training program has been well received by participants.",
            "Budget allocations for next year are currently under review.",
        ]

        for i in range(min(num_tasks, len(needles))):
            needle_text, question, answer = needles[i]
            position = needle_positions[i % len(needle_positions)]

            target_chars = context_tokens * 4
            num_filler = target_chars // 60
            filler_lines = [rng.choice(filler_sentences) for _ in range(num_filler)]

            insert_pos = int(len(filler_lines) * position)
            filler_lines.insert(insert_pos, needle_text)

            context = "\n".join(filler_lines)
            if len(context) > target_chars:
                context = context[:target_chars]

            tasks.append(EvalTask(
                task_id=f"synthetic_needle_{i}",
                benchmark="synthetic",
                query=question,
                context=context,
                expected_answer=answer,
                category="needle_in_haystack",
                context_tokens=context_tokens,
                difficulty="easy",
                metadata={"needle_position": position},
            ))

        return tasks

    def multi_needle(
        self,
        context_tokens: int = 4000,
        num_needles: int = 3,
        num_tasks: int = 2,
    ) -> List[EvalTask]:
        """Generate multi-needle tasks: find and combine multiple facts."""
        rng = random.Random(self.seed + 200)
        tasks = []

        needle_sets = [
            {
                "needles": [
                    "Agent Alpha is stationed in Berlin.",
                    "Agent Beta is stationed in Tokyo.",
                    "Agent Gamma is stationed in Cairo.",
                ],
                "query": "List all agent locations.",
                "answer": "Berlin, Tokyo, Cairo",
            },
            {
                "needles": [
                    "Server A has 32 GB RAM.",
                    "Server B has 64 GB RAM.",
                    "Server C has 16 GB RAM.",
                ],
                "query": "What is the total RAM across all servers?",
                "answer": "112",
            },
        ]

        filler_sentences = [
            "Standard operating procedures were followed.",
            "The system check completed without errors.",
            "All personnel reported on time.",
            "Equipment maintenance was performed as scheduled.",
            "Communication channels are operating normally.",
        ]

        for i in range(min(num_tasks, len(needle_sets))):
            ns = needle_sets[i]
            actual_needles = ns["needles"][:num_needles]

            target_chars = context_tokens * 4
            num_filler = target_chars // 50
            filler_lines = [rng.choice(filler_sentences) for _ in range(num_filler)]

            for j, needle in enumerate(actual_needles):
                pos = int(len(filler_lines) * (j + 1) / (len(actual_needles) + 1))
                filler_lines.insert(pos, needle)

            context = "\n".join(filler_lines)
            if len(context) > target_chars:
                context = context[:target_chars]

            tasks.append(EvalTask(
                task_id=f"synthetic_multi_needle_{i}",
                benchmark="synthetic",
                query=ns["query"],
                context=context,
                expected_answer=ns["answer"],
                category="multi_needle",
                context_tokens=context_tokens,
                difficulty="medium",
                metadata={"num_needles": len(actual_needles)},
            ))

        return tasks

    def counting(
        self,
        context_tokens: int = 4000,
        num_tasks: int = 2,
    ) -> List[EvalTask]:
        """Generate counting tasks: count specific items in context."""
        rng = random.Random(self.seed + 300)
        tasks = []

        scenarios = [
            {
                "item": "ERROR",
                "template": "Log entry: {level} - {msg}",
                "levels": ["ERROR", "WARNING", "INFO", "DEBUG"],
                "messages": [
                    "Connection timeout",
                    "Processing request",
                    "Cache miss",
                    "Request completed",
                    "Disk space low",
                ],
            },
            {
                "item": "CRITICAL",
                "template": "Alert: {level} - {msg}",
                "levels": ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                "messages": [
                    "System overload",
                    "Network failure",
                    "Service degradation",
                    "Minor latency spike",
                ],
            },
        ]

        for i, scenario in enumerate(scenarios[:num_tasks]):
            target_chars = context_tokens * 4
            num_entries = target_chars // 40

            lines = []
            target_count = 0
            for _ in range(num_entries):
                level = rng.choice(scenario["levels"])
                msg = rng.choice(scenario["messages"])
                lines.append(scenario["template"].format(level=level, msg=msg))
                if level == scenario["item"]:
                    target_count += 1

            context = "\n".join(lines)
            if len(context) > target_chars:
                context = context[:target_chars]
                # Recount after truncation
                target_count = context.count(scenario["item"])

            tasks.append(EvalTask(
                task_id=f"synthetic_counting_{i}",
                benchmark="synthetic",
                query=f"How many {scenario['item']} entries are in the log?",
                context=context,
                expected_answer=str(target_count),
                category="counting",
                context_tokens=context_tokens,
                difficulty="medium",
                metadata={"target_item": scenario["item"], "count": target_count},
            ))

        return tasks

    def distributed_reasoning(
        self,
        context_tokens: int = 4000,
        num_tasks: int = 2,
    ) -> List[EvalTask]:
        """Generate distributed reasoning tasks: pieces of a puzzle scattered in context."""
        rng = random.Random(self.seed + 400)
        tasks = []

        puzzles = [
            {
                "clues": [
                    "Clue 1: The treasure is not in the mountains.",
                    "Clue 2: The treasure is east of the river.",
                    "Clue 3: The only location east of the river that is not in the mountains is the forest.",
                ],
                "query": "Where is the treasure?",
                "answer": "the forest",
            },
            {
                "clues": [
                    "Fact: The train departs every 30 minutes starting at 6:00 AM.",
                    "Fact: Alice arrives at the station at 7:20 AM.",
                    "Fact: The ride takes 45 minutes.",
                ],
                "query": "What time does Alice arrive at her destination?",
                "answer": "8:15 AM",
            },
        ]

        filler_sentences = [
            "The archives contain extensive records from previous decades.",
            "Standard maintenance procedures are documented in section 4.",
            "Field operations continue as planned.",
            "Resource allocation meets current requirements.",
            "Personnel assignments have been confirmed.",
        ]

        for i, puzzle in enumerate(puzzles[:num_tasks]):
            target_chars = context_tokens * 4
            num_filler = target_chars // 55
            filler_lines = [rng.choice(filler_sentences) for _ in range(num_filler)]

            for j, clue in enumerate(puzzle["clues"]):
                pos = int(len(filler_lines) * (j + 1) / (len(puzzle["clues"]) + 1))
                filler_lines.insert(pos, clue)

            context = "\n".join(filler_lines)
            if len(context) > target_chars:
                context = context[:target_chars]

            tasks.append(EvalTask(
                task_id=f"synthetic_reasoning_{i}",
                benchmark="synthetic",
                query=puzzle["query"],
                context=context,
                expected_answer=puzzle["answer"],
                category="distributed_reasoning",
                context_tokens=context_tokens,
                difficulty="hard",
                metadata={"num_clues": len(puzzle["clues"])},
            ))

        return tasks
