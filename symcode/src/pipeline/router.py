"""Task routing: classify math problems and decide SymCode vs prose."""

from __future__ import annotations

import enum
import json
import re
from dataclasses import dataclass, field
from typing import Any

from src.utils.logging import get_logger

logger = get_logger("router")

# ── Task type taxonomy ──────────────────────────────────────────────


class TaskType(enum.Enum):
    ALGEBRA = "algebra"
    GEOMETRY = "geometry"
    NUMBER_THEORY = "number_theory"
    COMBINATORICS = "combinatorics"
    CALCULUS = "calculus"
    LOGIC = "logic"
    PROBABILITY = "probability"
    GENERAL = "general"


# Map MATH dataset subjects to our TaskTypes
MATH_SUBJECT_MAP: dict[str, TaskType] = {
    "algebra": TaskType.ALGEBRA,
    "counting_and_probability": TaskType.PROBABILITY,
    "geometry": TaskType.GEOMETRY,
    "intermediate_algebra": TaskType.ALGEBRA,
    "number_theory": TaskType.NUMBER_THEORY,
    "prealgebra": TaskType.ALGEBRA,
    "precalculus": TaskType.CALCULUS,
}


@dataclass
class RoutingDecision:
    """Result of routing a math problem."""

    task_type: TaskType
    use_symcode: bool
    reasoning: str = ""
    confidence: float = 1.0


# ── Keyword tables ──────────────────────────────────────────────────

_KEYWORD_MAP: dict[str, TaskType] = {}

_ALGEBRA_KW = [
    "solve", "equation", "polynomial", "factor", "root", "quadratic",
    "linear", "variable", "expression", "simplify", "expand", "inequality",
    "system of equations", "matrix", "determinant", "eigenvalue",
]
_GEOMETRY_KW = [
    "triangle", "circle", "angle", "perimeter", "area", "volume",
    "polygon", "rectangle", "sphere", "cylinder", "tangent line",
    "inscribed", "circumscribed", "diagonal", "parallel", "perpendicular",
    "congruent", "similar triangles",
]
_NUMBER_THEORY_KW = [
    "divisor", "prime", "gcd", "lcm", "modular", "congruence",
    "remainder", "divisible", "euler", "totient", "fermat",
    "coprime", "modulo", "mod ",
]
_COMBINATORICS_KW = [
    "how many ways", "permutation", "combination", "choose",
    "arrange", "counting", "pigeonhole", "inclusion-exclusion",
    "binomial coefficient",
]
_CALCULUS_KW = [
    "derivative", "integral", "differentiate", "integrate", "limit",
    "convergent", "divergent", "series", "taylor", "maclaurin",
    "partial derivative", "gradient",
]
_LOGIC_KW = [
    "prove", "proof", "if and only if", "contrapositive",
    "induction", "contradiction", "necessary and sufficient",
]
_PROBABILITY_KW = [
    "probability", "expected value", "random", "dice", "coin",
    "fair", "independent", "conditional", "bayes",
]

for _kw in _ALGEBRA_KW:
    _KEYWORD_MAP[_kw] = TaskType.ALGEBRA
for _kw in _GEOMETRY_KW:
    _KEYWORD_MAP[_kw] = TaskType.GEOMETRY
for _kw in _NUMBER_THEORY_KW:
    _KEYWORD_MAP[_kw] = TaskType.NUMBER_THEORY
for _kw in _COMBINATORICS_KW:
    _KEYWORD_MAP[_kw] = TaskType.COMBINATORICS
for _kw in _CALCULUS_KW:
    _KEYWORD_MAP[_kw] = TaskType.CALCULUS
for _kw in _LOGIC_KW:
    _KEYWORD_MAP[_kw] = TaskType.LOGIC
for _kw in _PROBABILITY_KW:
    _KEYWORD_MAP[_kw] = TaskType.PROBABILITY

# Task types that default to prose rather than SymCode
_PROSE_TYPES: set[TaskType] = {TaskType.GEOMETRY}


class TaskRouter:
    """Route math problems to the appropriate pipeline."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    # ── heuristic routing ───────────────────────────────────────────

    def heuristic_route(
        self,
        problem: str,
        metadata: dict[str, Any] | None = None,
    ) -> RoutingDecision:
        """Route a problem using keywords and optional metadata.

        If metadata contains a 'subject' key (e.g. from MATH dataset),
        that takes priority for task-type classification.
        """
        metadata = metadata or {}
        task_type: TaskType | None = None

        # 1. Check metadata subject (from MATH dataset)
        subject = metadata.get("subject", "").lower().strip()
        if subject and subject in MATH_SUBJECT_MAP:
            task_type = MATH_SUBJECT_MAP[subject]

        # 2. Check metadata for explicit task_type override
        explicit_type = metadata.get("task_type", "").lower().strip()
        if explicit_type:
            for tt in TaskType:
                if tt.value == explicit_type:
                    task_type = tt
                    break

        # 3. Keyword heuristic on problem text
        if task_type is None:
            problem_lower = problem.lower()
            scores: dict[TaskType, int] = {}
            for kw, tt in _KEYWORD_MAP.items():
                if kw in problem_lower:
                    scores[tt] = scores.get(tt, 0) + 1
            if scores:
                task_type = max(scores, key=scores.get)  # type: ignore[arg-type]

        if task_type is None:
            task_type = TaskType.GENERAL

        # 4. Decide SymCode vs prose
        # Check metadata override
        use_symcode_override = metadata.get("use_symcode")
        if use_symcode_override is not None:
            use_symcode = bool(use_symcode_override)
        else:
            use_symcode = task_type not in _PROSE_TYPES

        reasoning = (
            f"Classified as {task_type.value} "
            f"({'from metadata' if subject or explicit_type else 'via keywords'}). "
            f"{'SymCode' if use_symcode else 'Prose'} pipeline selected."
        )

        return RoutingDecision(
            task_type=task_type,
            use_symcode=use_symcode,
            reasoning=reasoning,
        )

    # ── LLM-based routing (optional, requires API) ─────────────────

    def llm_route(
        self,
        problem: str,
        llm_client: Any = None,
        router_prompt: str = "",
    ) -> RoutingDecision:
        """Route a problem using an LLM for classification.

        Falls back to heuristic_route if no client provided or on error.
        """
        if llm_client is None:
            return self.heuristic_route(problem)

        try:
            messages = [
                {"role": "system", "content": router_prompt},
                {"role": "user", "content": problem},
            ]
            response = llm_client.chat.completions.create(
                model=self.config.get("model", {}).get("name", "gpt-4o"),
                messages=messages,
                temperature=0.0,
                max_tokens=256,
            )
            content = response.choices[0].message.content.strip()
            data = json.loads(content)

            task_type_str = data.get("task_type", "general")
            task_type = TaskType.GENERAL
            for tt in TaskType:
                if tt.value == task_type_str:
                    task_type = tt
                    break

            use_symcode = data.get("use_symcode", True)
            reasoning = data.get("reasoning", "")

            return RoutingDecision(
                task_type=task_type,
                use_symcode=use_symcode,
                reasoning=reasoning,
            )
        except Exception as e:
            logger.warning(f"LLM routing failed ({e}), falling back to heuristic")
            return self.heuristic_route(problem)
