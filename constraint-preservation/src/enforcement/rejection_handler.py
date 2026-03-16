"""RejectionHandler: handles rejected modifications with structured messages."""

from __future__ import annotations

from typing import Any

from src.checker.verdict import SuiteVerdict
from src.constraints.base import CheckContext


class RejectionHandler:
    """Format and handle constraint-violation rejections."""

    def handle(self, verdict: SuiteVerdict, context: CheckContext) -> str:
        """Process a rejection and return a formatted message."""
        return self.format_rejection_message(verdict, context)

    @staticmethod
    def format_rejection_message(verdict: SuiteVerdict, context: CheckContext) -> str:
        """Build a human-readable rejection message."""
        lines = [
            "MODIFICATION REJECTED",
            f"  Modification type: {context.modification_type}",
            f"  Description: {context.modification_description}",
            "",
            "Constraint violations:",
        ]

        for name, result in verdict.violations.items():
            lines.append(f"  [{name}]")
            lines.append(f"    Measured: {result.measured_value:.4f}")
            lines.append(f"    Threshold: {result.threshold:.4f}")
            lines.append(f"    Headroom: {result.headroom:.4f}")
            if result.details:
                for k, v in result.details.items():
                    lines.append(f"    {k}: {v}")

        lines.append("")
        lines.append("The modification cannot proceed. No override is available.")
        return "\n".join(lines)
