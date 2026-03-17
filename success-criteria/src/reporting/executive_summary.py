"""Executive summary — one-page stakeholder summary."""

from __future__ import annotations

from src.verdict.verdict import FinalVerdict, VerdictCategory


class ExecutiveSummary:
    """Generates a one-page executive summary for stakeholders."""

    def generate(self, verdict: FinalVerdict) -> str:
        """Generate a concise executive summary in markdown format."""
        lines = []

        # Header
        lines.append("# Month-18 GO/NO-GO Executive Summary")
        lines.append("")

        # Verdict banner
        emoji_map = {
            VerdictCategory.SUCCESS: "GO",
            VerdictCategory.PARTIAL: "CONDITIONAL",
            VerdictCategory.NOT_MET: "NO-GO",
        }
        banner = emoji_map.get(verdict.category, "UNKNOWN")
        lines.append(f"## Decision: {banner}")
        lines.append("")
        lines.append(
            f"**{verdict.n_passed} of {verdict.n_total}** pre-registered "
            f"criteria met (confidence: {verdict.overall_confidence:.0%})"
        )
        lines.append("")

        # Criteria summary table
        lines.append("## Criteria Results")
        lines.append("")
        lines.append("| # | Criterion | Result | Confidence | Margin |")
        lines.append("|---|-----------|--------|------------|--------|")

        for i, result in enumerate(verdict.criteria_results, 1):
            status = "PASS" if result.passed else "FAIL"
            lines.append(
                f"| {i} | {result.criterion_name} | {status} | "
                f"{result.confidence:.0%} | {result.margin:+.2f} |"
            )
        lines.append("")

        # Key findings
        lines.append("## Key Findings")
        lines.append("")
        if verdict.is_go:
            lines.append(
                "All success criteria have been met. The project "
                "demonstrates sustained improvement, paradigm-specific "
                "gains, safe GDI bounds, peer-reviewed publications, "
                "and complete auditability."
            )
        else:
            lines.append("### Passed")
            for r in verdict.passed_criteria:
                lines.append(f"- **{r.criterion_name}**: margin = {r.margin:+.2f}")
            lines.append("")
            lines.append("### Failed")
            for r in verdict.failed_criteria:
                lines.append(f"- **{r.criterion_name}**: margin = {r.margin:+.2f}")

        lines.append("")

        # Rationale
        lines.append("## Rationale")
        lines.append("")
        lines.append(verdict.rationale)
        lines.append("")

        return "\n".join(lines)
