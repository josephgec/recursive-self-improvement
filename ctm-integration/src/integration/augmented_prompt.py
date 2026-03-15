"""Augmented prompt builder that incorporates verified rules from the library."""

from __future__ import annotations

from typing import List, Optional

from src.library.index import RuleIndex
from src.library.rule import VerifiedRule
from src.library.store import RuleStore


class AugmentedPromptBuilder:
    """Builds prompts augmented with relevant verified rules from the library.

    Retrieves relevant rules for a given task and formats them
    as context for an LLM prompt.
    """

    def __init__(
        self,
        store: Optional[RuleStore] = None,
        index: Optional[RuleIndex] = None,
        max_rules: int = 5,
    ) -> None:
        """Initialize the prompt builder.

        Args:
            store: Rule store to retrieve rules from.
            index: Rule index for retrieval. Built from store if not provided.
            max_rules: Maximum number of rules to include in context.
        """
        self.store = store or RuleStore()
        self.index = index or RuleIndex(self.store)
        self.max_rules = max_rules

    def build_prompt(
        self,
        task: str,
        library: Optional[RuleStore] = None,
    ) -> str:
        """Build an augmented prompt with relevant rules.

        Args:
            task: The task description.
            library: Optional alternative library to use.

        Returns:
            Augmented prompt string.
        """
        # Use provided library or default
        if library:
            index = RuleIndex(library)
        else:
            index = self.index

        # Retrieve relevant rules
        rules = self._retrieve_rules(task, index)

        # Format prompt
        if rules:
            rules_context = self._format_rules_for_context(rules)
            prompt = (
                f"## Relevant Verified Rules\n\n"
                f"{rules_context}\n\n"
                f"## Task\n\n"
                f"{task}\n\n"
                f"Use the verified rules above as reference when solving this task. "
                f"You may adapt, combine, or extend them as needed."
            )
        else:
            prompt = self.build_standard_prompt(task)

        return prompt

    def build_standard_prompt(self, task: str) -> str:
        """Build a standard (non-augmented) prompt.

        Args:
            task: The task description.

        Returns:
            Standard prompt string.
        """
        return f"## Task\n\n{task}\n\nSolve the task above."

    def _retrieve_rules(
        self, task: str, index: Optional[RuleIndex] = None
    ) -> List[VerifiedRule]:
        """Retrieve relevant rules for a task.

        Args:
            task: Task description to match against.
            index: Rule index to search in.

        Returns:
            List of relevant verified rules.
        """
        idx = index or self.index
        return idx.retrieve(task, k=self.max_rules)

    def _format_rules_for_context(self, rules: List[VerifiedRule]) -> str:
        """Format rules as context for inclusion in a prompt.

        Args:
            rules: List of verified rules.

        Returns:
            Formatted string representation of the rules.
        """
        formatted_parts = []

        for i, rule in enumerate(rules, 1):
            tags_str = ", ".join(rule.tags) if rule.tags else "none"
            part = (
                f"### Rule {i}: {rule.description}\n"
                f"- **Domain**: {rule.domain}\n"
                f"- **Accuracy**: {rule.accuracy:.1%}\n"
                f"- **BDM Complexity**: {rule.bdm_score:.2f}\n"
                f"- **Tags**: {tags_str}\n"
                f"\n```python\n{rule.source_code}\n```\n"
            )
            formatted_parts.append(part)

        return "\n".join(formatted_parts)
