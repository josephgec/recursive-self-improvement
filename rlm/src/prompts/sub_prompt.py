"""Sub-prompt builder for nested RLM sessions (depth >= 1)."""

from __future__ import annotations

from src.core.context_loader import ContextMeta
from src.prompts.templates import (
    REPL_INTRO,
    FINAL_INSTRUCTIONS,
    SUB_TEMPLATE,
    BUDGET_WARNING,
)


class SubPromptBuilder:
    """Build the system prompt for a sub-query RLM session."""

    def build(self, context_meta: ContextMeta, query: str, depth: int = 1) -> str:
        """Build the sub-query prompt."""
        return SUB_TEMPLATE.format(
            repl_intro=REPL_INTRO,
            final_instructions=FINAL_INSTRUCTIONS,
            depth=depth,
            query=query,
            context_type=context_meta.context_type,
            context_length=context_meta.size_chars,
            context_tokens=context_meta.size_tokens,
        )

    @staticmethod
    def budget_warning(remaining: int) -> str:
        """Return a budget warning message."""
        return BUDGET_WARNING.format(remaining=remaining)
