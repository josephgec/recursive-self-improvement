"""Report generation: comprehensive markdown report."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.core.session import SessionResult
from src.evaluation.metrics import RLMMetrics
from src.analysis.trajectory_analysis import (
    strategy_by_task_type,
    efficiency_by_strategy,
    example_trajectories,
)
from src.analysis.cost_analysis import token_usage, latency_analysis, cost_per_query
from src.analysis.depth_analysis import recursion_depth_distribution


def generate_report(
    results: List[SessionResult],
    expected: Optional[List[str]] = None,
    task_types: Optional[List[str]] = None,
    title: str = "RLM Evaluation Report",
) -> str:
    """Generate a comprehensive markdown report from session results."""
    metrics = RLMMetrics()
    lines: List[str] = []
    lines.append(f"# {title}\n")

    # Summary
    lines.append("## Summary\n")
    lines.append(f"- Total sessions: {len(results)}")
    lines.append(f"- Total iterations: {sum(r.total_iterations for r in results)}")
    forced = sum(1 for r in results if r.forced_final)
    lines.append(f"- Forced finals: {forced}")
    lines.append("")

    # Accuracy
    if expected:
        acc = metrics.accuracy(results, expected, exact=False)
        lines.append("## Accuracy\n")
        lines.append(f"- Overall: {acc.value:.2%}")
        lines.append(f"- Correct: {acc.details.get('correct', 0)} / {acc.details.get('n', 0)}")
        lines.append("")

    # Cost
    lines.append("## Cost\n")
    cost = cost_per_query(results)
    tokens = token_usage(results)
    lines.append(f"- Avg cost/query: ${cost['avg_cost']:.4f}")
    lines.append(f"- Total cost: ${cost['total_cost']:.4f}")
    lines.append(f"- Avg tokens/query: {tokens['avg_tokens']:.0f}")
    lines.append("")

    # Latency
    lines.append("## Latency\n")
    lat = latency_analysis(results)
    lines.append(f"- Avg elapsed: {lat['avg_elapsed']:.3f}s")
    if results:
        lines.append(f"- Min elapsed: {lat['min_elapsed']:.3f}s")
        lines.append(f"- Max elapsed: {lat['max_elapsed']:.3f}s")
    lines.append("")

    # Strategy distribution
    lines.append("## Strategies\n")
    strat = metrics.strategy_distribution(results)
    for s, count in strat.details.get("distribution", {}).items():
        lines.append(f"- {s}: {count}")
    lines.append("")

    # Efficiency by strategy
    lines.append("## Efficiency by Strategy\n")
    eff = efficiency_by_strategy(results)
    for s, info in eff.items():
        lines.append(
            f"- {s}: avg_iters={info['avg_iterations']:.1f}, "
            f"avg_elapsed={info['avg_elapsed']:.3f}s, n={info['count']}"
        )
    lines.append("")

    # Depth distribution
    lines.append("## Recursion Depth\n")
    depth = recursion_depth_distribution(results)
    for d, count in depth.get("distribution", {}).items():
        lines.append(f"- Depth {d}: {count} sessions")
    lines.append("")

    # Strategy by task type
    if task_types:
        lines.append("## Strategy by Task Type\n")
        sbt = strategy_by_task_type(results, task_types)
        for tt, dist in sbt.items():
            lines.append(f"### {tt}")
            for s, count in dist.items():
                lines.append(f"- {s}: {count}")
            lines.append("")

    # Example trajectories
    lines.append("## Example Trajectories\n")
    examples = example_trajectories(results, max_examples=3)
    for i, ex in enumerate(examples):
        lines.append(f"### Example {i + 1}")
        lines.append(f"- Strategy: {ex['strategy']} (confidence={ex['confidence']:.2f})")
        lines.append(f"- Iterations: {ex['iterations']}")
        lines.append(f"- Forced final: {ex['forced_final']}")
        if ex['code_samples']:
            lines.append(f"- First code block: `{ex['code_samples'][0][:80]}...`")
        lines.append("")

    return "\n".join(lines)
