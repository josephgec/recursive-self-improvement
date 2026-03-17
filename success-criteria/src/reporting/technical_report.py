"""Technical report — 8-section markdown report."""

from __future__ import annotations

from typing import Any, Dict

from src.criteria.base import Evidence
from src.verdict.verdict import FinalVerdict


class TechnicalReport:
    """Generates a detailed technical report in markdown."""

    def generate(self, verdict: FinalVerdict, evidence: Evidence) -> str:
        """Generate an 8-section technical report."""
        sections = [
            self._section_1_introduction(),
            self._section_2_methodology(verdict),
            self._section_3_evidence_overview(evidence),
            self._section_4_criterion_results(verdict),
            self._section_5_statistical_analysis(verdict),
            self._section_6_safety_assessment(evidence),
            self._section_7_verdict(verdict),
            self._section_8_conclusions(verdict),
        ]
        return "\n\n".join(sections)

    def _section_1_introduction(self) -> str:
        return (
            "# Technical Report: Month-18 GO/NO-GO Evaluation\n\n"
            "## 1. Introduction\n\n"
            "This report presents the results of the Month-18 GO/NO-GO "
            "evaluation against 5 pre-registered success criteria. "
            "The evaluation follows the pre-registered analysis plan "
            "with no post-hoc modifications to thresholds or methodology."
        )

    def _section_2_methodology(self, verdict: FinalVerdict) -> str:
        lines = [
            "## 2. Methodology\n",
            f"- **Criteria evaluated**: {verdict.n_total}",
            "- **Statistical tests**: Mann-Kendall trend test, paired t-tests",
            "- **Safety metric**: Guardrail Divergence Index (GDI)",
            "- **Integrity**: SHA-256 hash chain verification",
            "- **Decision rule**: SUCCESS requires all criteria passed; "
            "PARTIAL requires 3-4; NOT_MET for 2 or fewer",
        ]
        return "\n".join(lines)

    def _section_3_evidence_overview(self, evidence: Evidence) -> str:
        curve = evidence.get_improvement_curve()
        n_pubs = len(evidence.publications)
        n_gdi = len(evidence.get_gdi_readings())

        lines = [
            "## 3. Evidence Overview\n",
            f"- **Phases**: {len(curve)} phase scores collected",
            f"- **Improvement curve**: {curve}",
            f"- **Publications**: {n_pubs} papers in registry",
            f"- **GDI readings**: {n_gdi} measurements",
            f"- **Phases monitored**: {evidence.get_phases_monitored()}",
        ]
        return "\n".join(lines)

    def _section_4_criterion_results(self, verdict: FinalVerdict) -> str:
        lines = ["## 4. Criterion-by-Criterion Results\n"]
        for i, result in enumerate(verdict.criteria_results, 1):
            status = "PASS" if result.passed else "FAIL"
            lines.append(f"### 4.{i}. {result.criterion_name} [{status}]\n")
            lines.append(f"- **Measured**: {result.measured_value}")
            lines.append(f"- **Threshold**: {result.threshold}")
            lines.append(f"- **Margin**: {result.margin:+.3f}")
            lines.append(f"- **Confidence**: {result.confidence:.2f}")
            lines.append(f"- **Methodology**: {result.methodology}")
            if result.caveats:
                lines.append(f"- **Caveats**: {'; '.join(result.caveats)}")
            lines.append("")

        return "\n".join(lines)

    def _section_5_statistical_analysis(
        self, verdict: FinalVerdict
    ) -> str:
        lines = ["## 5. Statistical Analysis\n"]
        for result in verdict.criteria_results:
            lines.append(f"### {result.criterion_name}")
            for ev in result.supporting_evidence:
                lines.append(f"- {ev}")
            lines.append("")
        return "\n".join(lines)

    def _section_6_safety_assessment(self, evidence: Evidence) -> str:
        readings = evidence.get_gdi_readings()
        if readings:
            max_gdi = max(r.get("gdi", 0) for r in readings)
            mean_gdi = sum(r.get("gdi", 0) for r in readings) / len(readings)
        else:
            max_gdi = 0
            mean_gdi = 0

        lines = [
            "## 6. Safety Assessment\n",
            f"- **Max GDI observed**: {max_gdi:.3f}",
            f"- **Mean GDI**: {mean_gdi:.3f}",
            f"- **Total readings**: {len(readings)}",
            f"- **Phases monitored**: {evidence.get_phases_monitored()}",
        ]
        return "\n".join(lines)

    def _section_7_verdict(self, verdict: FinalVerdict) -> str:
        lines = [
            "## 7. Verdict\n",
            f"**{verdict.category.value}**: {verdict.rationale}",
            "",
            f"- Criteria passed: {verdict.n_passed}/{verdict.n_total}",
            f"- Overall confidence: {verdict.overall_confidence:.2f}",
        ]
        return "\n".join(lines)

    def _section_8_conclusions(self, verdict: FinalVerdict) -> str:
        lines = [
            "## 8. Conclusions and Next Steps\n",
            verdict.rationale,
            "",
        ]
        if verdict.is_go:
            lines.append(
                "The project has met all pre-registered success criteria. "
                "Recommend proceeding to the next phase of development."
            )
        else:
            lines.append(
                "The project has not fully met all success criteria. "
                "Remediation is needed before a final GO decision."
            )
            for r in verdict.failed_criteria:
                lines.append(
                    f"- **{r.criterion_name}**: needs improvement "
                    f"(margin={r.margin:+.2f})"
                )
        return "\n".join(lines)
