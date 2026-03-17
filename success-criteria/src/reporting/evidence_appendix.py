"""Evidence appendix — detailed evidence listing."""

from __future__ import annotations

import json
from typing import Any, Dict

from src.criteria.base import Evidence


class EvidenceAppendix:
    """Generates a detailed evidence appendix in markdown."""

    def generate(self, evidence: Evidence) -> str:
        """Generate the evidence appendix."""
        sections = [
            self._header(),
            self._phase_data(evidence),
            self._safety_data(evidence),
            self._publication_data(evidence),
            self._audit_data(evidence),
        ]
        return "\n\n".join(sections)

    def _header(self) -> str:
        return (
            "# Evidence Appendix\n\n"
            "This appendix contains all evidence used in the "
            "Month-18 GO/NO-GO evaluation."
        )

    def _phase_data(self, evidence: Evidence) -> str:
        lines = ["## A. Phase Data\n"]
        for phase_name in [
            "phase_0", "phase_1", "phase_2", "phase_3", "phase_4"
        ]:
            data = getattr(evidence, phase_name)
            lines.append(f"### {phase_name}\n")
            lines.append(f"- Score: {data.get('score', 'N/A')}")
            lines.append(
                f"- Collapse score: {data.get('collapse_score', 'N/A')}"
            )
            ablations = data.get("ablations", {})
            if ablations:
                lines.append("- Ablations:")
                for paradigm, values in ablations.items():
                    lines.append(
                        f"  - {paradigm}: with={values.get('with', 'N/A')}, "
                        f"without={values.get('without', 'N/A')}"
                    )
            lines.append("")
        return "\n".join(lines)

    def _safety_data(self, evidence: Evidence) -> str:
        lines = ["## B. Safety Data\n"]
        readings = evidence.get_gdi_readings()
        lines.append(f"Total GDI readings: {len(readings)}\n")
        lines.append("| # | Timestamp | GDI | Status | Phase |")
        lines.append("|---|-----------|-----|--------|-------|")
        for i, r in enumerate(readings, 1):
            lines.append(
                f"| {i} | {r.get('timestamp', 'N/A')} | "
                f"{r.get('gdi', 0):.3f} | {r.get('status', 'N/A')} | "
                f"{r.get('phase', 'N/A')} |"
            )
        return "\n".join(lines)

    def _publication_data(self, evidence: Evidence) -> str:
        lines = ["## C. Publications\n"]
        lines.append("| Title | Venue | Status | Year |")
        lines.append("|-------|-------|--------|------|")
        for pub in evidence.publications:
            lines.append(
                f"| {pub.get('title', 'N/A')} | "
                f"{pub.get('venue', 'N/A')} | "
                f"{pub.get('status', 'N/A')} | "
                f"{pub.get('year', 'N/A')} |"
            )
        return "\n".join(lines)

    def _audit_data(self, evidence: Evidence) -> str:
        audit = evidence.audit_trail
        lines = ["## D. Audit Trail\n"]

        for log_name in [
            "modification_log", "constraint_log", "gdi_log", "interp_log"
        ]:
            log_data = audit.get(log_name, [])
            lines.append(f"### {log_name}")
            lines.append(f"Entries: {len(log_data)}\n")
            for entry in log_data[:5]:  # Show first 5
                lines.append(f"- {json.dumps(entry)}")
            if len(log_data) > 5:
                lines.append(f"- ... ({len(log_data) - 5} more entries)")
            lines.append("")

        traces = audit.get("reasoning_traces", [])
        lines.append(f"### Reasoning Traces")
        lines.append(f"Total: {len(traces)}\n")

        chain = audit.get("hash_chain", [])
        lines.append(f"### Hash Chain")
        lines.append(f"Length: {len(chain)}")

        return "\n".join(lines)
