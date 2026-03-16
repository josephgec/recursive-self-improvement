"""Final report: consolidated Phase 1a report."""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.deliverables.symcode_summary import package_symcode_results
from src.deliverables.bdm_summary import package_bdm_results


def generate_phase1a_report(
    results_dir: str = "data/output",
) -> str:
    """Generate the consolidated Phase 1a report.

    Args:
        results_dir: Directory containing result files.

    Returns:
        Markdown-formatted report.
    """
    symcode = package_symcode_results(results_dir)
    bdm = package_bdm_results(results_dir)

    sections = []
    sections.append("# Phase 1a: Architecture Comparison Report\n")
    sections.append("## Overview\n")
    sections.append(
        "This report compares three architectures for recursive self-improvement:\n"
        "1. **Hybrid (SymCode)**: LLM + external solvers via tool-calling\n"
        "2. **Integrative (BDM)**: LNN-style constrained decoding\n"
        "3. **Prose Baseline**: Plain LLM with no augmentation\n"
    )

    # SymCode section
    sections.append("## SymCode (Hybrid Architecture)\n")
    sections.append(f"{symcode['description']}\n")
    sections.append("### Key Findings\n")
    for finding in symcode.get("key_findings", []):
        sections.append(f"- {finding}")
    sections.append("")
    _add_metrics_section(sections, symcode.get("metrics", {}))

    # BDM section
    sections.append("\n## BDM (Integrative Architecture)\n")
    sections.append(f"{bdm['description']}\n")
    sections.append("### Key Findings\n")
    for finding in bdm.get("key_findings", []):
        sections.append(f"- {finding}")
    sections.append("")
    _add_metrics_section(sections, bdm.get("metrics", {}))

    # Comparison
    sections.append("\n## Comparative Analysis\n")
    sections.append("### Generalization\n")
    sym_gen = symcode.get("metrics", {}).get("generalization", {})
    bdm_gen = bdm.get("metrics", {}).get("generalization", {})
    sections.append(
        f"| Metric | SymCode | BDM |\n"
        f"|--------|---------|-----|\n"
        f"| In-domain acc. | {sym_gen.get('in_domain_accuracy', 'N/A')} | {bdm_gen.get('in_domain_accuracy', 'N/A')} |\n"
        f"| Out-of-domain acc. | {sym_gen.get('out_of_domain_accuracy', 'N/A')} | {bdm_gen.get('out_of_domain_accuracy', 'N/A')} |\n"
        f"| Gen. gap | {sym_gen.get('generalization_gap', 'N/A')} | {bdm_gen.get('generalization_gap', 'N/A')} |\n"
    )

    # Recommendations
    sections.append("\n## Recommendations\n")
    sections.append(
        "1. **For maximum accuracy**: Use the hybrid architecture with tool-calling\n"
        "2. **For maximum robustness**: Use the integrative architecture with constraints\n"
        "3. **For RSI suitability**: Hybrid offers best modularity and verifiability\n"
        "4. **For cost efficiency**: Prose baseline is cheapest but least capable\n"
    )

    return "\n".join(sections)


def _add_metrics_section(sections: list, metrics: dict) -> None:
    """Add a metrics section to the report."""
    if not metrics:
        return
    sections.append("### Metrics\n")
    for axis, axis_metrics in metrics.items():
        sections.append(f"**{axis.capitalize()}**\n")
        if isinstance(axis_metrics, dict):
            for metric, value in axis_metrics.items():
                sections.append(f"- {metric}: {value}")
        else:
            sections.append(f"- {axis}: {axis_metrics}")
        sections.append("")
