"""Prompt analysis: section evolution, patterns, diffs."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.genome.prompt_genome import PromptGenome
from src.genome.similarity import section_similarity
from src.genome.sections import DEFAULT_SECTIONS, SECTION_ORDER


def section_evolution(
    genomes_by_generation: Dict[int, List[PromptGenome]],
) -> Dict[str, List[float]]:
    """Track how each section changes across generations.

    Returns dict mapping section_name to list of avg similarity-to-original per generation.
    """
    result: Dict[str, List[float]] = {s: [] for s in SECTION_ORDER}

    for gen in sorted(genomes_by_generation.keys()):
        genomes = genomes_by_generation[gen]

        for section_name in SECTION_ORDER:
            default_text = DEFAULT_SECTIONS.get(section_name, "")
            sims = []
            for genome in genomes:
                if section_name in genome.sections:
                    sim = section_similarity(
                        genome.sections[section_name].content,
                        default_text,
                    )
                    sims.append(sim)

            avg_sim = sum(sims) / max(len(sims), 1) if sims else 1.0
            result[section_name].append(avg_sim)

    return result


def common_patterns(genomes: List[PromptGenome]) -> Dict[str, List[str]]:
    """Identify common word patterns in high-fitness genomes.

    Returns dict mapping section_name to list of common words/phrases.
    """
    patterns: Dict[str, Dict[str, int]] = {s: {} for s in SECTION_ORDER}

    for genome in genomes:
        for section_name in SECTION_ORDER:
            if section_name not in genome.sections:
                continue
            content = genome.sections[section_name].content.lower()
            words = content.split()
            for word in words:
                word = word.strip(".,;:!?")
                if len(word) > 3:
                    patterns[section_name][word] = (
                        patterns[section_name].get(word, 0) + 1
                    )

    result: Dict[str, List[str]] = {}
    for section_name, word_counts in patterns.items():
        sorted_words = sorted(
            word_counts.items(), key=lambda x: x[1], reverse=True
        )
        result[section_name] = [w for w, c in sorted_words[:10] if c > 1]

    return result


def diff_vs_baseline(
    genome: PromptGenome,
) -> Dict[str, Tuple[float, str, str]]:
    """Compare an evolved genome against the default baseline.

    Returns dict mapping section_name to (similarity, evolved_text, baseline_text).
    """
    diffs = {}
    for section_name in SECTION_ORDER:
        baseline = DEFAULT_SECTIONS.get(section_name, "")
        evolved = ""
        if section_name in genome.sections:
            evolved = genome.sections[section_name].content

        sim = section_similarity(evolved, baseline)
        diffs[section_name] = (sim, evolved, baseline)

    return diffs


def transferable_insights(
    genomes: List[PromptGenome],
    min_fitness: float = 0.6,
) -> List[str]:
    """Extract transferable insights from high-fitness genomes.

    Returns list of insight strings.
    """
    insights = []
    high_fitness = [g for g in genomes if g.fitness >= min_fitness]

    if not high_fitness:
        return ["No high-fitness genomes found."]

    # Check for common methodology patterns
    methodology_words: Dict[str, int] = {}
    for genome in high_fitness:
        if "methodology" in genome.sections:
            for word in genome.sections["methodology"].content.lower().split():
                word = word.strip(".,;:!?")
                if len(word) > 4:
                    methodology_words[word] = methodology_words.get(word, 0) + 1

    common = [w for w, c in methodology_words.items() if c >= len(high_fitness) // 2 + 1]
    if common:
        insights.append(
            f"Common methodology terms in top prompts: {', '.join(common[:5])}"
        )

    # Check average section lengths
    avg_tokens = {}
    for genome in high_fitness:
        for name, section in genome.sections.items():
            if name not in avg_tokens:
                avg_tokens[name] = []
            avg_tokens[name].append(section.token_count)

    for name, tokens in avg_tokens.items():
        avg = sum(tokens) / len(tokens)
        insights.append(f"Optimal '{name}' section length: ~{avg:.0f} tokens")

    return insights
