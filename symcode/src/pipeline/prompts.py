"""Prompt assembly for SymCode and prose pipelines."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from src.pipeline.router import TaskType
from src.utils.logging import get_logger

logger = get_logger("prompts")


class PromptManager:
    """Load prompt templates and build chat-format messages."""

    def __init__(self, prompts_dir: str | Path | None = None):
        if prompts_dir is None:
            prompts_dir = Path(__file__).resolve().parents[2] / "configs" / "prompts"
        self.prompts_dir = Path(prompts_dir)
        self._cache: dict[str, str] = {}
        self._fewshot_cache: list[dict[str, Any]] | None = None

    # ── loading helpers ─────────────────────────────────────────────

    def _load_text(self, filename: str) -> str:
        """Load a text prompt file, with caching."""
        if filename in self._cache:
            return self._cache[filename]
        path = self.prompts_dir / filename
        text = path.read_text(encoding="utf-8").strip()
        self._cache[filename] = text
        return text

    def _load_fewshot(self) -> list[dict[str, Any]]:
        """Load few-shot examples from YAML."""
        if self._fewshot_cache is not None:
            return self._fewshot_cache
        path = self.prompts_dir / "symcode_fewshot.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self._fewshot_cache = data.get("examples", [])
        return self._fewshot_cache

    # ── system prompts ──────────────────────────────────────────────

    @property
    def symcode_system(self) -> str:
        return self._load_text("symcode_system.txt")

    @property
    def retry_system(self) -> str:
        return self._load_text("retry_system.txt")

    @property
    def prose_system(self) -> str:
        return self._load_text("prose_system.txt")

    @property
    def router_system(self) -> str:
        return self._load_text("router_system.txt")

    # ── few-shot selection ──────────────────────────────────────────

    def get_fewshot_examples(
        self,
        category: str | TaskType | None = None,
        max_examples: int = 3,
    ) -> list[dict[str, Any]]:
        """Select few-shot examples, optionally filtered by category."""
        examples = self._load_fewshot()
        if category is not None:
            cat_str = category.value if isinstance(category, TaskType) else category
            filtered = [e for e in examples if e.get("category") == cat_str]
            if filtered:
                examples = filtered
        return examples[:max_examples]

    # ── prompt builders ─────────────────────────────────────────────

    def build_symcode_prompt(
        self,
        problem: str,
        task_type: TaskType | None = None,
        max_fewshot: int = 3,
    ) -> list[dict[str, str]]:
        """Build an OpenAI chat-format message list for SymCode generation.

        Includes system prompt, category-specific few-shot examples,
        and the user problem.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.symcode_system},
        ]

        # Add few-shot examples
        examples = self.get_fewshot_examples(
            category=task_type, max_examples=max_fewshot
        )
        for ex in examples:
            messages.append(
                {"role": "user", "content": f"Problem: {ex['problem']}"}
            )
            code_block = f"```python\n{ex['code'].strip()}\n```"
            messages.append({"role": "assistant", "content": code_block})

        # The actual problem
        messages.append({"role": "user", "content": f"Problem: {problem}"})
        return messages

    def build_retry_prompt(
        self,
        problem: str,
        previous_code: str,
        error_feedback: str,
        attempt: int = 1,
        max_attempts: int = 3,
    ) -> list[dict[str, str]]:
        """Build a retry prompt with error context.

        Includes the retry system prompt, original problem, previous code,
        and structured error feedback.
        """
        retry_context = (
            f"PROBLEM:\n{problem}\n\n"
            f"YOUR PREVIOUS CODE (attempt {attempt}/{max_attempts}):\n"
            f"```python\n{previous_code}\n```\n\n"
            f"ERROR FEEDBACK:\n{error_feedback}"
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.symcode_system},
            {"role": "user", "content": f"Problem: {problem}"},
            {"role": "assistant", "content": f"```python\n{previous_code}\n```"},
            {
                "role": "user",
                "content": f"{self.retry_system}\n\n{retry_context}",
            },
        ]
        return messages

    def build_prose_prompt(self, problem: str) -> list[dict[str, str]]:
        """Build a prose CoT prompt for the baseline solver."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.prose_system},
            {"role": "user", "content": f"Problem: {problem}"},
        ]
        return messages
