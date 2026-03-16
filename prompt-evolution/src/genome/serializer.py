"""Serialize and deserialize PromptGenome to/from JSON."""

from __future__ import annotations

import json
from typing import Any, Dict

from src.genome.prompt_genome import PromptGenome, PromptSection


def serialize_genome(genome: PromptGenome) -> str:
    """Serialize a PromptGenome to a JSON string."""
    data = _genome_to_dict(genome)
    return json.dumps(data, indent=2)


def deserialize_genome(json_str: str) -> PromptGenome:
    """Deserialize a JSON string back into a PromptGenome."""
    data = json.loads(json_str)
    return _dict_to_genome(data)


def _genome_to_dict(genome: PromptGenome) -> Dict[str, Any]:
    """Convert genome to a serializable dictionary."""
    sections_data = {}
    for name, section in genome.sections.items():
        sections_data[name] = {
            "section_type": section.section_type,
            "content": section.content,
            "token_count": section.token_count,
            "is_mutable": section.is_mutable,
        }
    return {
        "genome_id": genome.genome_id,
        "sections": sections_data,
        "generation": genome.generation,
        "parent_ids": genome.parent_ids,
        "operator": genome.operator,
        "fitness": genome.fitness,
        "metadata": genome.metadata,
    }


def _dict_to_genome(data: Dict[str, Any]) -> PromptGenome:
    """Convert a dictionary back into a PromptGenome."""
    genome = PromptGenome(
        genome_id=data["genome_id"],
        generation=data.get("generation", 0),
        parent_ids=data.get("parent_ids", []),
        operator=data.get("operator", "deserialized"),
        fitness=data.get("fitness", 0.0),
        metadata=data.get("metadata", {}),
    )
    for name, sec_data in data.get("sections", {}).items():
        genome.sections[name] = PromptSection(
            section_type=sec_data["section_type"],
            content=sec_data["content"],
            token_count=sec_data.get("token_count", 0),
            is_mutable=sec_data.get("is_mutable", True),
        )
    return genome
