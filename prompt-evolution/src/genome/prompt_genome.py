"""Core genome representation for prompt evolution."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PromptSection:
    """A single section of a system prompt."""

    section_type: str
    content: str
    token_count: int = 0
    is_mutable: bool = True

    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = self._estimate_tokens(self.content)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count as roughly words * 1.3."""
        words = len(text.split())
        return max(1, int(words * 1.3))

    def copy(self) -> "PromptSection":
        return PromptSection(
            section_type=self.section_type,
            content=self.content,
            token_count=self.token_count,
            is_mutable=self.is_mutable,
        )


@dataclass
class PromptGenome:
    """A complete system prompt represented as a genome of sections."""

    genome_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    sections: Dict[str, PromptSection] = field(default_factory=dict)
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    operator: str = "init"
    fitness: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def to_system_prompt(self) -> str:
        """Assemble sections into a complete system prompt string."""
        from src.genome.sections import SECTION_ORDER

        parts = []
        # Use canonical order, then append any extra sections
        ordered_keys = []
        for key in SECTION_ORDER:
            if key in self.sections:
                ordered_keys.append(key)
        for key in self.sections:
            if key not in ordered_keys:
                ordered_keys.append(key)

        for key in ordered_keys:
            section = self.sections[key]
            parts.append(f"## {section.section_type}\n{section.content}")

        return "\n\n".join(parts)

    def get_section(self, section_type: str) -> Optional[PromptSection]:
        """Get a section by type."""
        return self.sections.get(section_type)

    def set_section(self, section_type: str, content: str, is_mutable: bool = True):
        """Set or update a section."""
        self.sections[section_type] = PromptSection(
            section_type=section_type,
            content=content,
            is_mutable=is_mutable,
        )

    def total_tokens(self) -> int:
        """Total estimated token count across all sections."""
        return sum(s.token_count for s in self.sections.values())

    def copy(self) -> "PromptGenome":
        """Deep copy this genome with a new ID."""
        new_genome = PromptGenome(
            genome_id=str(uuid.uuid4())[:8],
            sections={k: v.copy() for k, v in self.sections.items()},
            generation=self.generation,
            parent_ids=list(self.parent_ids),
            operator=self.operator,
            fitness=self.fitness,
            metadata=dict(self.metadata),
        )
        return new_genome

    @classmethod
    def from_string(cls, prompt_text: str, genome_id: Optional[str] = None) -> "PromptGenome":
        """Parse a system prompt string into a PromptGenome.

        Expects sections delimited by '## SectionType' headers.
        If no headers found, treats entire text as 'identity' section.
        """
        gid = genome_id or str(uuid.uuid4())[:8]
        genome = cls(genome_id=gid)

        lines = prompt_text.strip().split("\n")
        current_section = None
        current_lines: List[str] = []

        for line in lines:
            if line.startswith("## "):
                if current_section is not None:
                    content = "\n".join(current_lines).strip()
                    genome.set_section(current_section, content)
                current_section = line[3:].strip()
                current_lines = []
            else:
                current_lines.append(line)

        # Handle last section
        if current_section is not None:
            content = "\n".join(current_lines).strip()
            genome.set_section(current_section, content)
        elif lines:
            # No section headers found - treat as identity
            genome.set_section("identity", prompt_text.strip())

        return genome

    def similarity(self, other: "PromptGenome") -> float:
        """Compute similarity to another genome (0-1)."""
        from src.genome.similarity import genome_similarity
        return genome_similarity(self, other)
