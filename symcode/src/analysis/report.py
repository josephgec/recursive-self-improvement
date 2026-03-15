"""Generate a markdown analysis report from benchmark results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.analysis.difficulty_scaling import DifficultyScaler
from src.analysis.error_taxonomy import ErrorTaxonomist, TAXONOMY
from src.analysis.retry_analysis import RetryAnalyzer
from src.benchmarks.comparison import ComparisonAnalyzer
from src.benchmarks.metrics import MetricsComputer
from src.benchmarks.runner import BenchmarkResult
from src.utils.logging import get_logger

logger = get_logger("analysis.report")


def generate_report(
    benchmark_result: BenchmarkResult,
    output_path: str | Path | None = None,
    title: str = "SymCode Benchmark Report",
) -> str:
    """Generate a comprehensive markdown report.

    Args:
        benchmark_result: Results from BenchmarkRunner.run().
        output_path: Optional path to write the report.
        title: Report title.

    Returns:
        The report as a markdown string.
    """
    sections: list[str] = []

    # ── Header ──────────────────────────────────────────────────────
    sections.append(f"# {title}\n")

    # ── Executive summary ───────────────────────────────────────────
    sections.append(_executive_summary(benchmark_result))

    # ── Comparison table ────────────────────────────────────────────
    if benchmark_result.symcode_results and benchmark_result.prose_results:
        sections.append(_comparison_section(benchmark_result))

    # ── Per-subject breakdown ───────────────────────────────────────
    if benchmark_result.symcode_results:
        sections.append(_per_subject_section(benchmark_result))

    # ── Difficulty scaling ──────────────────────────────────────────
    if benchmark_result.symcode_results:
        sections.append(_difficulty_section(benchmark_result))

    # ── Retry analysis ──────────────────────────────────────────────
    if benchmark_result.symcode_results:
        sections.append(_retry_section(benchmark_result))

    # ── Error taxonomy ──────────────────────────────────────────────
    if benchmark_result.symcode_results:
        sections.append(_error_taxonomy_section(benchmark_result))

    # ── Qualitative examples ────────────────────────────────────────
    if benchmark_result.symcode_results and benchmark_result.prose_results:
        sections.append(_qualitative_section(benchmark_result))

    # ── Recommendations ─────────────────────────────────────────────
    sections.append(_recommendations(benchmark_result))

    report = "\n".join(sections)

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
        logger.info("Report written to %s", path)

    return report


# ── Section builders ────────────────────────────────────────────────


def _executive_summary(result: BenchmarkResult) -> str:
    """Build the executive summary section."""
    lines = [
        "## Executive Summary\n",
        f"- **Total problems**: {result.num_problems}",
        f"- **Total time**: {result.total_time:.1f}s",
    ]

    if result.symcode_results:
        lines.append(
            f"- **SymCode accuracy**: {result.symcode_accuracy:.1%}"
        )
    if result.prose_results:
        lines.append(
            f"- **Prose accuracy**: {result.prose_accuracy:.1%}"
        )
    if result.symcode_results and result.prose_results:
        delta = (result.symcode_accuracy - result.prose_accuracy) * 100
        direction = "improvement" if delta > 0 else "regression"
        lines.append(
            f"- **Delta**: {delta:+.1f} percentage points ({direction})"
        )

    lines.append("")
    return "\n".join(lines)


def _comparison_section(result: BenchmarkResult) -> str:
    """Build the head-to-head comparison section."""
    analyzer = ComparisonAnalyzer(result.symcode_results, result.prose_results)
    summary = analyzer.summary()
    stat = analyzer.statistical_test()

    lines = [
        "## Head-to-Head Comparison\n",
        "| Metric | SymCode | Prose |",
        "|--------|---------|-------|",
        f"| Accuracy | {summary['symcode_accuracy']:.1%} | {summary['prose_accuracy']:.1%} |",
        f"| Delta (pp) | {summary['delta_pp']:+.1f} | - |",
        "",
        f"- **McNemar's p-value**: {stat['p_value']:.4f}" if stat['p_value'] is not None else "- McNemar's test: N/A",
        f"- **Significant (p<0.05)**: {'Yes' if stat.get('significant_005') else 'No'}",
        f"- SymCode-only correct: {summary['symcode_only_correct']}",
        f"- Prose-only correct: {summary['prose_only_correct']}",
        f"- Both correct: {summary['both_correct']}",
        f"- Both wrong: {summary['both_wrong']}",
        "",
    ]
    return "\n".join(lines)


def _per_subject_section(result: BenchmarkResult) -> str:
    """Build the per-subject breakdown section."""
    sym_by_subject = MetricsComputer.accuracy_by_subject(result.symcode_results)

    lines = [
        "## Per-Subject Breakdown\n",
        "| Subject | Accuracy | Correct | Total |",
        "|---------|----------|---------|-------|",
    ]

    for subject, data in sym_by_subject.items():
        lines.append(
            f"| {subject} | {data['accuracy']:.1%} | "
            f"{data['correct']} | {data['total']} |"
        )

    lines.append("")
    return "\n".join(lines)


def _difficulty_section(result: BenchmarkResult) -> str:
    """Build the difficulty scaling section."""
    scaler = DifficultyScaler()
    curve = scaler.compute_scaling_curve(result.symcode_results)
    hyp = scaler.test_scaling_hypothesis(result.symcode_results)

    lines = [
        "## Difficulty Scaling\n",
        "| Difficulty | Accuracy | Count |",
        "|------------|----------|-------|",
    ]

    for d, a, c in zip(curve.difficulties, curve.accuracies, curve.counts):
        lines.append(f"| {d} | {a:.1%} | {c} |")

    lines.append("")
    if hyp.get("spearman_rho") is not None:
        lines.append(
            f"- **Spearman rho**: {hyp['spearman_rho']:.3f} "
            f"(p={hyp['p_value']:.4f})"
        )
        lines.append(
            f"- **Monotonically decreasing**: "
            f"{'Yes' if hyp['monotonically_decreasing'] else 'No'}"
        )
    else:
        lines.append(f"- {hyp.get('message', 'Insufficient data')}")

    lines.append("")
    return "\n".join(lines)


def _retry_section(result: BenchmarkResult) -> str:
    """Build the retry analysis section."""
    effectiveness = MetricsComputer.retry_effectiveness(result.symcode_results)
    trajectory = RetryAnalyzer.correction_trajectory(result.symcode_results)

    lines = [
        "## Retry / Self-Correction Analysis\n",
        f"- **First-attempt accuracy**: {effectiveness['first_attempt_accuracy']:.1%}",
        f"- **Final accuracy (after retries)**: {effectiveness['final_accuracy']:.1%}",
        f"- **Recovery rate**: {effectiveness['recovery_rate']:.1%}",
        f"- **Avg attempts (correct)**: {effectiveness['avg_attempts_when_correct']:.2f}",
        f"- **Avg attempts (wrong)**: {effectiveness['avg_attempts_when_wrong']:.2f}",
        "",
    ]

    if trajectory["cumulative_accuracy"]:
        lines.append("### Cumulative Accuracy by Attempt\n")
        lines.append("| Attempt | Accuracy | Marginal Gain |")
        lines.append("|---------|----------|---------------|")
        for i, (acc, gain) in enumerate(
            zip(trajectory["cumulative_accuracy"], trajectory["marginal_gain"])
        ):
            lines.append(f"| {i + 1} | {acc:.1%} | {gain:+.1%} |")
        lines.append("")

    return "\n".join(lines)


def _error_taxonomy_section(result: BenchmarkResult) -> str:
    """Build the error taxonomy section."""
    taxonomist = ErrorTaxonomist()
    report = taxonomist.analyze(result.symcode_results)

    lines = [
        "## Error Taxonomy\n",
        f"- **Total failures**: {report.total_failures} / {report.total_problems}",
        "",
    ]

    for cat, info in TAXONOMY.items():
        total = report.category_totals.get(cat, 0)
        if total == 0:
            continue

        lines.append(f"### {info['description']} ({total} problems)\n")
        lines.append("| Subcategory | Count |")
        lines.append("|-------------|-------|")

        for subcat, desc in info["subcategories"].items():
            count = report.counts.get(cat, {}).get(subcat, 0)
            if count > 0:
                lines.append(f"| {desc} | {count} |")

        lines.append("")

    return "\n".join(lines)


def _qualitative_section(result: BenchmarkResult) -> str:
    """Build the qualitative examples section."""
    analyzer = ComparisonAnalyzer(result.symcode_results, result.prose_results)
    examples = analyzer.qualitative_examples(max_examples=3)

    lines = ["## Qualitative Examples\n"]

    if examples["symcode_wins"]:
        lines.append("### SymCode Wins (correct where prose failed)\n")
        for ex in examples["symcode_wins"]:
            lines.append(
                f"- **Problem**: {ex['problem'][:150]}...\n"
                f"  - Expected: `{ex['expected_answer']}`\n"
                f"  - SymCode: `{ex['symcode_answer']}` (in {ex['symcode_attempts']} attempts)\n"
                f"  - Prose: `{ex['prose_answer']}`\n"
            )

    if examples["prose_wins"]:
        lines.append("### Prose Wins (correct where SymCode failed)\n")
        for ex in examples["prose_wins"]:
            lines.append(
                f"- **Problem**: {ex['problem'][:150]}...\n"
                f"  - Expected: `{ex['expected_answer']}`\n"
                f"  - SymCode: `{ex['symcode_answer']}`\n"
                f"  - Prose: `{ex['prose_answer']}`\n"
            )

    if examples["both_fail"]:
        lines.append("### Both Failed\n")
        for ex in examples["both_fail"]:
            lines.append(
                f"- **Problem**: {ex['problem'][:150]}...\n"
                f"  - Expected: `{ex['expected_answer']}`\n"
                f"  - SymCode: `{ex['symcode_answer']}`\n"
                f"  - Prose: `{ex['prose_answer']}`\n"
            )

    lines.append("")
    return "\n".join(lines)


def _recommendations(result: BenchmarkResult) -> str:
    """Generate recommendations based on the results."""
    lines = ["## Recommendations\n"]

    if result.symcode_results:
        effectiveness = MetricsComputer.retry_effectiveness(result.symcode_results)
        errors = MetricsComputer.error_distribution(result.symcode_results)

        # Recommendation 1: retry effectiveness
        if effectiveness["recovery_rate"] < 0.3:
            lines.append(
                "1. **Improve self-correction**: Recovery rate is low "
                f"({effectiveness['recovery_rate']:.0%}). Consider richer "
                "feedback with more context about the mathematical approach."
            )
        else:
            lines.append(
                "1. **Self-correction is effective**: Recovery rate is "
                f"{effectiveness['recovery_rate']:.0%}. The retry loop "
                "is working well."
            )

        # Recommendation 2: common errors
        if errors:
            top_error = max(errors, key=errors.get)  # type: ignore[arg-type]
            lines.append(
                f"2. **Address {top_error}**: Most common error type "
                f"({errors[top_error]} occurrences). Consider adding "
                "targeted few-shot examples or prompt engineering."
            )

        # Recommendation 3: difficulty scaling
        if result.symcode_accuracy < 0.5:
            lines.append(
                "3. **Overall accuracy needs improvement**: Consider "
                "using a stronger base model or improving prompt design."
            )
        else:
            lines.append(
                "3. **Good baseline accuracy**: Focus on closing the gap "
                "on harder problems."
            )

    if result.symcode_results and result.prose_results:
        delta = result.symcode_accuracy - result.prose_accuracy
        if delta < 0:
            lines.append(
                "4. **SymCode underperforms prose**: Investigate whether "
                "code generation errors are the bottleneck. Consider "
                "hybrid approaches."
            )
        elif delta < 0.10:
            lines.append(
                "4. **Marginal improvement**: The gap is small. Focus on "
                "subjects where SymCode has the largest advantage."
            )
        else:
            lines.append(
                f"4. **Strong improvement ({delta*100:.1f}pp)**: SymCode "
                "shows clear benefits. Consider deploying for supported "
                "task types."
            )

    lines.append("")
    return "\n".join(lines)
