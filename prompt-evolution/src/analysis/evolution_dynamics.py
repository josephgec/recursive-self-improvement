"""Evolution dynamics analysis: fitness trajectory, convergence, plots."""

from __future__ import annotations

from typing import Dict, List, Optional

from src.ga.engine import EvolutionResult, GenerationResult


def fitness_trajectory(result: EvolutionResult) -> List[Dict]:
    """Extract fitness trajectory data from evolution results.

    Returns list of dicts with generation, best_fitness, avg_fitness, diversity.
    """
    trajectory = []
    for gen_result in result.generation_results:
        trajectory.append({
            "generation": gen_result.generation,
            "best_fitness": gen_result.best_fitness,
            "avg_fitness": gen_result.avg_fitness,
            "diversity": gen_result.diversity,
        })
    return trajectory


def operator_contribution(result: EvolutionResult) -> Dict[str, Dict]:
    """Analyze contribution of each operator across generations.

    Returns dict mapping operator type to stats.
    """
    contributions = {
        "mutation": {"count": 0, "total_generations": 0},
        "crossover": {"count": 0, "total_generations": 0},
        "elitism": {"count": 0, "total_generations": 0},
    }

    for gen_result in result.generation_results:
        contributions["mutation"]["count"] += gen_result.num_mutations
        contributions["crossover"]["count"] += gen_result.num_crossovers
        contributions["elitism"]["count"] += gen_result.num_elites
        for key in contributions:
            contributions[key]["total_generations"] += 1

    # Compute averages
    for key in contributions:
        total_gens = contributions[key]["total_generations"]
        if total_gens > 0:
            contributions[key]["avg_per_generation"] = (
                contributions[key]["count"] / total_gens
            )
        else:
            contributions[key]["avg_per_generation"] = 0.0

    return contributions


def convergence_speed(result: EvolutionResult, threshold: float = 0.9) -> Optional[int]:
    """Determine how many generations to reach a fitness threshold.

    Returns generation number or None if not reached.
    """
    for gen_result in result.generation_results:
        if gen_result.best_fitness >= threshold:
            return gen_result.generation
    return None


def plot_fitness_ascii(result: EvolutionResult, width: int = 50) -> str:
    """Generate ASCII plot of fitness trajectory.

    Returns a text-based plot.
    """
    if not result.fitness_history:
        return "No fitness history available."

    max_fitness = max(result.fitness_history)
    min_fitness = min(result.fitness_history)
    fitness_range = max_fitness - min_fitness if max_fitness > min_fitness else 1.0

    lines = ["Fitness Trajectory", "=" * (width + 15)]

    for i, fitness in enumerate(result.fitness_history):
        normalized = (fitness - min_fitness) / fitness_range if fitness_range > 0 else 0.5
        bar_len = int(normalized * width)
        bar = "#" * bar_len
        lines.append(f"Gen {i + 1:3d} |{bar:<{width}s}| {fitness:.4f}")

    return "\n".join(lines)
