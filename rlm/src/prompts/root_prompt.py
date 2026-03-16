"""Root prompt builder for RLM sessions at depth 0."""

from __future__ import annotations

from src.core.context_loader import ContextMeta
from src.prompts.templates import (
    REPL_INTRO,
    FINAL_INSTRUCTIONS,
    SUB_QUERY_INSTRUCTIONS,
    ROOT_TEMPLATE,
    BUDGET_WARNING,
)


class RootPromptBuilder:
    """Build the system prompt for a root-level (depth 0) RLM session."""

    def build(self, context_meta: ContextMeta, depth: int = 0) -> str:
        """Build the root prompt incorporating context metadata."""
        sub_query_section = SUB_QUERY_INSTRUCTIONS if depth == 0 else ""
        return ROOT_TEMPLATE.format(
            repl_intro=REPL_INTRO,
            final_instructions=FINAL_INSTRUCTIONS,
            sub_query_section=sub_query_section,
            query="{query}",  # placeholder — filled later
            context_type=context_meta.context_type,
            context_length=context_meta.size_chars,
            context_tokens=context_meta.size_tokens,
            num_lines=context_meta.num_lines,
        )

    def build_for_sub_query(self, context_meta: ContextMeta, query: str) -> str:
        """Build a prompt for a sub-query (depth > 0)."""
        return ROOT_TEMPLATE.format(
            repl_intro=REPL_INTRO,
            final_instructions=FINAL_INSTRUCTIONS,
            sub_query_section="",
            query=query,
            context_type=context_meta.context_type,
            context_length=context_meta.size_chars,
            context_tokens=context_meta.size_tokens,
            num_lines=context_meta.num_lines,
        )

    @staticmethod
    def budget_warning(remaining: int) -> str:
        """Return a budget warning message."""
        return BUDGET_WARNING.format(remaining=remaining)
