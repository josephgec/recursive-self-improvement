"""Reproducibility packager — bundles everything for reproducibility."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from src.criteria.base import Evidence
from src.reporting.executive_summary import ExecutiveSummary
from src.reporting.technical_report import TechnicalReport
from src.reporting.evidence_appendix import EvidenceAppendix
from src.verdict.verdict import FinalVerdict


class ReproducibilityPackager:
    """Packages evidence, verdict, and reports for reproducibility."""

    def package(
        self,
        evidence: Evidence,
        verdict: FinalVerdict,
        output_dir: str,
    ) -> Dict[str, str]:
        """Create a reproducibility package in the output directory.

        Args:
            evidence: The evidence used.
            verdict: The final verdict.
            output_dir: Directory to write outputs to.

        Returns:
            Dict mapping artifact names to file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        artifacts: Dict[str, str] = {}

        # 1. Evidence snapshot
        evidence_path = os.path.join(output_dir, "evidence.json")
        evidence_data = self._serialize_evidence(evidence)
        self._write_json(evidence_path, evidence_data)
        artifacts["evidence"] = evidence_path

        # 2. Verdict
        verdict_path = os.path.join(output_dir, "verdict.json")
        verdict_data = self._serialize_verdict(verdict)
        self._write_json(verdict_path, verdict_data)
        artifacts["verdict"] = verdict_path

        # 3. Executive summary
        exec_path = os.path.join(output_dir, "executive_summary.md")
        exec_summary = ExecutiveSummary().generate(verdict)
        self._write_text(exec_path, exec_summary)
        artifacts["executive_summary"] = exec_path

        # 4. Technical report
        tech_path = os.path.join(output_dir, "technical_report.md")
        tech_report = TechnicalReport().generate(verdict, evidence)
        self._write_text(tech_path, tech_report)
        artifacts["technical_report"] = tech_path

        # 5. Evidence appendix
        appendix_path = os.path.join(output_dir, "evidence_appendix.md")
        appendix = EvidenceAppendix().generate(evidence)
        self._write_text(appendix_path, appendix)
        artifacts["evidence_appendix"] = appendix_path

        # 6. Manifest
        manifest_path = os.path.join(output_dir, "manifest.json")
        self._write_json(manifest_path, {
            "artifacts": artifacts,
            "verdict_category": verdict.category.value,
            "n_passed": verdict.n_passed,
            "n_total": verdict.n_total,
        })
        artifacts["manifest"] = manifest_path

        return artifacts

    def _serialize_evidence(self, evidence: Evidence) -> Dict[str, Any]:
        """Serialize evidence to a JSON-compatible dict."""
        return {
            "phase_0": evidence.phase_0,
            "phase_1": evidence.phase_1,
            "phase_2": evidence.phase_2,
            "phase_3": evidence.phase_3,
            "phase_4": evidence.phase_4,
            "safety": evidence.safety,
            "publications": evidence.publications,
            "audit_trail": evidence.audit_trail,
        }

    def _serialize_verdict(self, verdict: FinalVerdict) -> Dict[str, Any]:
        """Serialize verdict to a JSON-compatible dict."""
        return {
            "category": verdict.category.value,
            "n_passed": verdict.n_passed,
            "n_total": verdict.n_total,
            "overall_confidence": verdict.overall_confidence,
            "rationale": verdict.rationale,
            "criteria_results": [
                {
                    "criterion_name": r.criterion_name,
                    "passed": r.passed,
                    "confidence": r.confidence,
                    "margin": r.margin,
                    "methodology": r.methodology,
                }
                for r in verdict.criteria_results
            ],
        }

    @staticmethod
    def _write_json(path: str, data: Any) -> None:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @staticmethod
    def _write_text(path: str, text: str) -> None:
        with open(path, "w") as f:
            f.write(text)
