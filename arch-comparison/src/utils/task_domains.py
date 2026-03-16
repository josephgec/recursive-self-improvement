"""Task domains: built-in tasks across multiple domains."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Task:
    """A single evaluation task."""
    task_id: str
    domain: str
    problem: str
    expected_answer: str
    difficulty: str = "medium"  # easy, medium, hard
    metadata: dict = field(default_factory=dict)


class MultiDomainTaskLoader:
    """Loads built-in tasks across multiple domains.

    Each domain has 20+ tasks covering a range of difficulties.
    """

    def __init__(self) -> None:
        self._domains: Dict[str, List[Task]] = {
            "arithmetic": self._build_arithmetic_tasks(),
            "algebra": self._build_algebra_tasks(),
            "logic": self._build_logic_tasks(),
            "probability": self._build_probability_tasks(),
        }

    def load_domain(self, domain: str) -> List[Task]:
        """Load all tasks for a domain.

        Args:
            domain: Domain name (arithmetic, algebra, logic, probability).

        Returns:
            List of Task objects.
        """
        if domain not in self._domains:
            raise ValueError(f"Unknown domain: {domain}. Available: {list(self._domains.keys())}")
        return list(self._domains[domain])

    def load_cross_domain(self, domains: Optional[List[str]] = None) -> List[Task]:
        """Load tasks from multiple domains.

        Args:
            domains: List of domain names. If None, loads all.

        Returns:
            Combined list of tasks.
        """
        domains = domains or list(self._domains.keys())
        tasks: List[Task] = []
        for domain in domains:
            tasks.extend(self.load_domain(domain))
        return tasks

    def load_paired_perturbations(
        self, domain: str
    ) -> Tuple[List[Task], List[Task]]:
        """Load paired original and perturbed tasks.

        Args:
            domain: Domain name.

        Returns:
            Tuple of (original_tasks, perturbed_tasks).
        """
        originals = self.load_domain(domain)
        perturbed = []
        for task in originals:
            p = Task(
                task_id=f"{task.task_id}_perturbed",
                domain=task.domain,
                problem=self._rephrase(task.problem),
                expected_answer=task.expected_answer,
                difficulty=task.difficulty,
                metadata={**task.metadata, "perturbation": "rephrase"},
            )
            perturbed.append(p)
        return originals, perturbed

    @property
    def available_domains(self) -> List[str]:
        return list(self._domains.keys())

    @staticmethod
    def _rephrase(problem: str) -> str:
        """Simple rephrasing: swap "What is" -> "Calculate", etc."""
        rephrases = [
            ("What is", "Calculate"),
            ("Compute", "Find the value of"),
            ("Solve", "Determine the solution to"),
            ("Find", "Determine"),
            ("Calculate", "What is"),
            ("Is it true that", "Determine whether"),
            ("What is the probability", "Find the probability"),
        ]
        for old, new in rephrases:
            if old.lower() in problem.lower():
                # Case-insensitive replacement
                idx = problem.lower().index(old.lower())
                return problem[:idx] + new + problem[idx + len(old):]
        return f"Please answer: {problem}"

    @staticmethod
    def _build_arithmetic_tasks() -> List[Task]:
        """Build arithmetic tasks."""
        tasks = []
        # Simple addition
        pairs = [
            (3, 4, 7), (12, 15, 27), (99, 1, 100), (256, 128, 384),
            (7, 8, 15), (45, 55, 100), (33, 67, 100), (11, 22, 33),
            (100, 200, 300), (17, 23, 40),
        ]
        for i, (a, b, ans) in enumerate(pairs):
            tasks.append(Task(
                task_id=f"arith_add_{i}",
                domain="arithmetic",
                problem=f"What is {a} + {b}?",
                expected_answer=str(ans),
                difficulty="easy" if ans < 100 else "medium",
            ))

        # Multiplication
        mul_pairs = [
            (3, 7, 21), (12, 5, 60), (8, 9, 72), (15, 4, 60),
            (25, 4, 100), (6, 11, 66), (13, 7, 91), (20, 15, 300),
            (9, 9, 81), (14, 3, 42),
        ]
        for i, (a, b, ans) in enumerate(mul_pairs):
            tasks.append(Task(
                task_id=f"arith_mul_{i}",
                domain="arithmetic",
                problem=f"Compute {a} * {b}.",
                expected_answer=str(ans),
                difficulty="easy" if ans < 100 else "medium",
            ))

        # Subtraction
        sub_pairs = [(10, 3, 7), (100, 45, 55), (50, 25, 25), (200, 87, 113)]
        for i, (a, b, ans) in enumerate(sub_pairs):
            tasks.append(Task(
                task_id=f"arith_sub_{i}",
                domain="arithmetic",
                problem=f"What is {a} - {b}?",
                expected_answer=str(ans),
                difficulty="easy",
            ))

        return tasks

    @staticmethod
    def _build_algebra_tasks() -> List[Task]:
        """Build algebra tasks."""
        tasks = []
        # Linear equations
        equations = [
            ("x + 3 = 7", "4"), ("x + 5 = 12", "7"), ("x + 10 = 25", "15"),
            ("2 * x = 10", "5"), ("3 * x = 21", "7"), ("5 * x = 45", "9"),
            ("x + 1 = 1", "0"), ("x + 0 = 42", "42"), ("x + 100 = 200", "100"),
            ("4 * x = 20", "5"),
        ]
        for i, (eq, ans) in enumerate(equations):
            tasks.append(Task(
                task_id=f"alg_linear_{i}",
                domain="algebra",
                problem=f"Solve for x: {eq}",
                expected_answer=ans,
                difficulty="easy" if i < 5 else "medium",
            ))

        # Expression evaluation
        exprs = [
            ("2 + 3 * 4", "14"), ("(2 + 3) * 4", "20"),
            ("10 - 2 * 3", "4"), ("100 - 50 + 25", "75"),
            ("5 * 5 - 10", "15"), ("3 * 3 + 4 * 4", "25"),
            ("2 * (10 + 5)", "30"), ("(8 - 3) * (6 + 1)", "35"),
            ("15 - 3 * 2", "9"), ("7 + 8 * 2", "23"),
        ]
        for i, (expr, ans) in enumerate(exprs):
            tasks.append(Task(
                task_id=f"alg_expr_{i}",
                domain="algebra",
                problem=f"Calculate: {expr}",
                expected_answer=ans,
                difficulty="medium",
            ))

        return tasks

    @staticmethod
    def _build_logic_tasks() -> List[Task]:
        """Build logic tasks."""
        tasks = []
        logic_problems = [
            ("If P implies Q, and P is true, what is Q?", "true"),
            ("If P implies Q, and Q is false, what is P?", "false"),
            ("If A and B are both true, what is A or B?", "true"),
            ("If A is true and B is false, what is A and B?", "false"),
            ("Is it true that: if true then true?", "true"),
            ("Is it true that: if false then true?", "true"),
            ("Is it true that: if true then false?", "false"),
            ("If not P is true, what is P?", "false"),
            ("If P or Q is true, and P is false, what is Q?", "true"),
            ("If P and Q is false, and P is true, what is Q?", "false"),
            ("Is the formula 'P and not P' satisfiable?", "false"),
            ("Is the formula 'P or not P' always true?", "true"),
            ("If all cats are animals, and Tom is a cat, is Tom an animal?", "true"),
            ("If no fish can fly, and Nemo is a fish, can Nemo fly?", "false"),
            ("If A implies B, and B implies C, and A is true, what is C?", "true"),
            ("Is the formula 'true and true' satisfiable?", "true"),
            ("Is the formula 'false or false' satisfiable?", "false"),
            ("If P iff Q, and P is true, what is Q?", "true"),
            ("If exactly one of A, B is true, and A is true, what is B?", "false"),
            ("Is 'if P then P' a tautology?", "true"),
        ]
        for i, (problem, answer) in enumerate(logic_problems):
            tasks.append(Task(
                task_id=f"logic_{i}",
                domain="logic",
                problem=problem,
                expected_answer=answer,
                difficulty="easy" if i < 7 else "medium" if i < 15 else "hard",
            ))
        return tasks

    @staticmethod
    def _build_probability_tasks() -> List[Task]:
        """Build probability tasks."""
        tasks = []
        prob_problems = [
            ("What is the probability of flipping heads on a fair coin?", "0.5"),
            ("What is the probability of rolling a 6 on a fair die?", "0.167"),
            ("What is the probability of rolling an even number on a fair die?", "0.5"),
            ("If P(A)=0.3 and P(B)=0.4 and A,B independent, what is P(A and B)?", "0.12"),
            ("If P(A)=0.3 and P(B)=0.4 and A,B independent, what is P(A or B)?", "0.58"),
            ("What is the probability of getting 2 heads in 2 fair coin flips?", "0.25"),
            ("What is the probability of getting at least 1 head in 2 fair coin flips?", "0.75"),
            ("If P(A)=0.5, what is P(not A)?", "0.5"),
            ("What is the probability of drawing a red card from a standard deck?", "0.5"),
            ("What is the probability of drawing an ace from a standard deck?", "0.077"),
            ("If events are mutually exclusive with P(A)=0.3, P(B)=0.4, what is P(A or B)?", "0.7"),
            ("What is the expected value of a fair die roll?", "3.5"),
            ("If you flip 3 coins, what is P(all heads)?", "0.125"),
            ("What is P(at least one 6) in 2 rolls of a fair die?", "0.306"),
            ("If P(A|B)=0.8 and P(B)=0.5, what is P(A and B)?", "0.4"),
            ("What is the variance of a fair coin flip (0 or 1)?", "0.25"),
            ("If P(A)=1, what is P(A or B) for any B?", "1.0"),
            ("If P(A)=0 and P(B)=0.5, what is P(A and B)?", "0.0"),
            ("What is the expected number of rolls to get a 6?", "6"),
            ("If P(A)=0.6, P(B|A)=0.5, what is P(A and B)?", "0.3"),
        ]
        for i, (problem, answer) in enumerate(prob_problems):
            tasks.append(Task(
                task_id=f"prob_{i}",
                domain="probability",
                problem=problem,
                expected_answer=answer,
                difficulty="easy" if i < 5 else "medium" if i < 15 else "hard",
            ))
        return tasks
