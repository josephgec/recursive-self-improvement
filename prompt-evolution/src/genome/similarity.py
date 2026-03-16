"""Similarity metrics for prompt genomes."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.genome.prompt_genome import PromptGenome


def section_similarity(text_a: str, text_b: str) -> float:
    """Compute word-overlap similarity between two text sections.

    Uses Jaccard similarity on word sets.
    Returns a float in [0, 1].
    """
    if not text_a.strip() and not text_b.strip():
        return 1.0
    if not text_a.strip() or not text_b.strip():
        return 0.0

    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())

    if not words_a and not words_b:
        return 1.0

    intersection = words_a & words_b
    union = words_a | words_b

    if not union:
        return 1.0

    return len(intersection) / len(union)


def genome_similarity(genome_a: "PromptGenome", genome_b: "PromptGenome") -> float:
    """Compute weighted similarity between two genomes.

    Computes per-section word overlap weighted by section importance.
    Returns a float in [0, 1].
    """
    from src.genome.sections import section_importance_weights

    weights = section_importance_weights()

    all_sections = set(genome_a.sections.keys()) | set(genome_b.sections.keys())
    if not all_sections:
        return 1.0

    total_weight = 0.0
    weighted_sim = 0.0

    for section_name in all_sections:
        w = weights.get(section_name, 0.1)
        total_weight += w

        text_a = ""
        text_b = ""
        if section_name in genome_a.sections:
            text_a = genome_a.sections[section_name].content
        if section_name in genome_b.sections:
            text_b = genome_b.sections[section_name].content

        sim = section_similarity(text_a, text_b)
        weighted_sim += w * sim

    if total_weight == 0:
        return 1.0

    return weighted_sim / total_weight
