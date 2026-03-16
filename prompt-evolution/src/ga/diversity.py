"""Diversity maintenance for populations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional

from src.genome.similarity import genome_similarity

if TYPE_CHECKING:
    from src.genome.prompt_genome import PromptGenome


def population_diversity(genomes: List["PromptGenome"]) -> float:
    """Compute average pairwise diversity in a population.

    Returns 1 - average_similarity, so higher = more diverse.
    Range: [0, 1].
    """
    if len(genomes) <= 1:
        return 1.0

    total_sim = 0.0
    count = 0

    for i in range(len(genomes)):
        for j in range(i + 1, len(genomes)):
            total_sim += genome_similarity(genomes[i], genomes[j])
            count += 1

    if count == 0:
        return 1.0

    avg_similarity = total_sim / count
    return 1.0 - avg_similarity


def maintain_diversity(
    genomes: List["PromptGenome"],
    threshold: float = 0.3,
    inject_fn: Optional[Callable[[], "PromptGenome"]] = None,
) -> List["PromptGenome"]:
    """Maintain population diversity by injecting random individuals.

    If diversity drops below threshold, replaces the most similar
    individual with a new random one.

    Args:
        genomes: Current population
        threshold: Minimum acceptable diversity
        inject_fn: Function to create a new random genome

    Returns:
        Updated population list.
    """
    if not genomes or inject_fn is None:
        return genomes

    current_diversity = population_diversity(genomes)

    if current_diversity >= threshold:
        return genomes

    # Find the most similar pair and replace the weaker one
    max_sim = -1.0
    replace_idx = -1

    for i in range(len(genomes)):
        for j in range(i + 1, len(genomes)):
            sim = genome_similarity(genomes[i], genomes[j])
            if sim > max_sim:
                max_sim = sim
                # Replace the one with lower fitness
                if genomes[i].fitness <= genomes[j].fitness:
                    replace_idx = i
                else:
                    replace_idx = j

    if replace_idx >= 0:
        new_genome = inject_fn()
        genomes[replace_idx] = new_genome

    return genomes
