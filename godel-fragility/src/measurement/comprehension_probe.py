"""Probe whether the agent (or an LLM) can comprehend its own code."""

from __future__ import annotations

import ast
import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class ComprehensionQuestion:
    """A question about a piece of code."""

    question_type: str  # 'output_prediction', 'branch_tracing', 'variable_tracking'
    question: str
    correct_answer: str
    code_snippet: str
    difficulty: int = 1  # 1-5


@dataclass
class ComprehensionResult:
    """Result of probing code comprehension."""

    questions: List[ComprehensionQuestion]
    answers: List[str]
    scores: List[float]  # 0 or 1 for each question
    overall_score: float = 0.0
    complexity: int = 0
    code_hash: str = ""

    def __post_init__(self) -> None:
        if self.scores:
            self.overall_score = sum(self.scores) / len(self.scores)


class ComprehensionProbe:
    """Generate comprehension questions about code and score answers.

    Uses a mock LLM by default -- questions are generated deterministically
    from the code AST.
    """

    def __init__(self, seed: int = 42, llm: Optional[Callable[..., str]] = None) -> None:
        self.rng = random.Random(seed)
        self._llm = llm or self._mock_llm

    def probe(self, code: str, complexity: int = 0) -> ComprehensionResult:
        """Generate questions about `code` and score mock answers.

        Args:
            code: Python source code to probe.
            complexity: Reported complexity of the code.

        Returns:
            ComprehensionResult with questions, answers, and scores.
        """
        questions = self._generate_questions(code)
        answers = []
        scores = []

        for q in questions:
            answer = self._llm(q.question, q.code_snippet)
            answers.append(answer)
            score = self._score_answer(answer, q.correct_answer)
            scores.append(score)

        code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]
        return ComprehensionResult(
            questions=questions,
            answers=answers,
            scores=scores,
            complexity=complexity,
            code_hash=code_hash,
        )

    def _generate_questions(self, code: str) -> List[ComprehensionQuestion]:
        """Generate a diverse set of questions from the code AST."""
        questions: List[ComprehensionQuestion] = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return [
                ComprehensionQuestion(
                    question_type="output_prediction",
                    question="Does this code have a syntax error?",
                    correct_answer="yes",
                    code_snippet=code[:200],
                    difficulty=1,
                )
            ]

        questions.extend(self._output_prediction_questions(tree, code))
        questions.extend(self._branch_tracing_questions(tree, code))
        questions.extend(self._variable_tracking_questions(tree, code))

        if not questions:
            questions.append(
                ComprehensionQuestion(
                    question_type="output_prediction",
                    question="Is this valid Python code?",
                    correct_answer="yes",
                    code_snippet=code[:200],
                    difficulty=1,
                )
            )

        return questions

    def _output_prediction_questions(
        self, tree: ast.AST, code: str
    ) -> List[ComprehensionQuestion]:
        """Generate questions about what the code outputs."""
        questions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Ask about the return type/value for simple functions
                has_return = any(
                    isinstance(child, ast.Return) and child.value is not None
                    for child in ast.walk(node)
                )
                if has_return:
                    # Find a return with a constant
                    for child in ast.walk(node):
                        if isinstance(child, ast.Return) and isinstance(
                            child.value, ast.Constant
                        ):
                            questions.append(
                                ComprehensionQuestion(
                                    question_type="output_prediction",
                                    question=(
                                        f"What does function '{node.name}' return "
                                        f"when it reaches the constant return?"
                                    ),
                                    correct_answer=str(child.value.value),
                                    code_snippet=ast.get_source_segment(code, node) or code[:200],
                                    difficulty=2,
                                )
                            )
                            break

        return questions[:3]

    def _branch_tracing_questions(
        self, tree: ast.AST, code: str
    ) -> List[ComprehensionQuestion]:
        """Generate questions about which branches execute."""
        questions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Check if the condition is a constant
                if isinstance(node.test, ast.Constant):
                    branch = "true" if node.test.value else "false"
                    questions.append(
                        ComprehensionQuestion(
                            question_type="branch_tracing",
                            question="Which branch executes when the condition is a constant?",
                            correct_answer=branch,
                            code_snippet=ast.get_source_segment(code, node) or code[:200],
                            difficulty=1,
                        )
                    )
                elif isinstance(node.test, ast.Compare):
                    # For comparisons with constants on both sides
                    if (
                        isinstance(node.test.left, ast.Constant)
                        and len(node.test.comparators) == 1
                        and isinstance(node.test.comparators[0], ast.Constant)
                    ):
                        left = node.test.left.value
                        right = node.test.comparators[0].value
                        op = node.test.ops[0]
                        result = self._eval_compare(left, op, right)
                        branch = "true" if result else "false"
                        questions.append(
                            ComprehensionQuestion(
                                question_type="branch_tracing",
                                question=f"Does {left} {self._op_str(op)} {right}?",
                                correct_answer=branch,
                                code_snippet=ast.get_source_segment(code, node) or code[:200],
                                difficulty=2,
                            )
                        )

        return questions[:3]

    def _variable_tracking_questions(
        self, tree: ast.AST, code: str
    ) -> List[ComprehensionQuestion]:
        """Generate questions about variable values after assignments."""
        questions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if (
                    len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Constant)
                ):
                    var_name = node.targets[0].id
                    value = node.value.value
                    questions.append(
                        ComprehensionQuestion(
                            question_type="variable_tracking",
                            question=f"What is the value of '{var_name}' after this assignment?",
                            correct_answer=str(value),
                            code_snippet=ast.get_source_segment(code, node) or code[:200],
                            difficulty=1,
                        )
                    )

        return questions[:3]

    def _eval_compare(self, left: Any, op: ast.cmpop, right: Any) -> bool:
        """Evaluate a comparison operator."""
        dispatch = {
            ast.Eq: lambda a, b: a == b,
            ast.NotEq: lambda a, b: a != b,
            ast.Lt: lambda a, b: a < b,
            ast.LtE: lambda a, b: a <= b,
            ast.Gt: lambda a, b: a > b,
            ast.GtE: lambda a, b: a >= b,
        }
        fn = dispatch.get(type(op))
        if fn is None:
            return False
        try:
            return fn(left, right)
        except TypeError:
            return False

    def _op_str(self, op: ast.cmpop) -> str:
        """Get string representation of a comparison operator."""
        mapping = {
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
        }
        return mapping.get(type(op), "?")

    def _mock_llm(self, question: str, code_snippet: str) -> str:
        """Mock LLM that gives somewhat reasonable answers.

        For testing purposes, it returns the correct answer with probability
        inversely proportional to code length (simulating comprehension loss).
        """
        # Simulate decreasing comprehension with code length
        code_len = len(code_snippet)
        # Above ~500 chars, start getting things wrong
        accuracy = max(0.1, 1.0 - (code_len / 1000.0))

        if self.rng.random() < accuracy:
            # Find the correct answer from context (cheat for mock)
            # In a real probe, this would call an actual LLM
            return "__CORRECT__"
        else:
            return "__INCORRECT__"

    def _score_answer(self, answer: str, correct: str) -> float:
        """Score an answer against the correct answer."""
        if answer == "__CORRECT__":
            return 1.0
        if answer == "__INCORRECT__":
            return 0.0

        # Normalize and compare
        answer_norm = answer.strip().lower()
        correct_norm = correct.strip().lower()

        if answer_norm == correct_norm:
            return 1.0

        # Partial credit for numeric answers
        try:
            a_val = float(answer_norm)
            c_val = float(correct_norm)
            if c_val != 0:
                relative_error = abs(a_val - c_val) / abs(c_val)
                if relative_error < 0.1:
                    return 0.5
        except (ValueError, ZeroDivisionError):
            pass

        return 0.0
