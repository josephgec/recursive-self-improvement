"""Drift characterization for GDI results."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..composite.gdi import GDIResult


@dataclass
class DriftCharacterization:
    """Characterization of the type of drift detected."""
    drift_type: str
    confidence: float
    description: str
    signals_involved: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DriftCharacterizer:
    """Characterizes the type of drift from GDI results.

    Drift types:
    - reasoning_shift: Primarily semantic drift
    - style_shift: Primarily lexical drift
    - structural_shift: Primarily structural drift
    - collapse: All signals elevated significantly
    - comprehensive: Multiple signals elevated moderately
    """

    def characterize(self, gdi_result: GDIResult) -> DriftCharacterization:
        """Characterize the type of drift in a GDI result.

        Args:
            gdi_result: GDI computation result.

        Returns:
            DriftCharacterization with type and description.
        """
        scores = {
            "semantic": gdi_result.semantic_score,
            "lexical": gdi_result.lexical_score,
            "structural": gdi_result.structural_score,
            "distributional": gdi_result.distributional_score,
        }

        # Determine which signals are elevated
        elevated = {k: v for k, v in scores.items() if v > 0.3}
        high = {k: v for k, v in scores.items() if v > 0.6}

        # Collapse: all or most signals very high
        if len(high) >= 3:
            return DriftCharacterization(
                drift_type="collapse",
                confidence=min(1.0, gdi_result.composite_score / 0.7),
                description="Severe multi-dimensional drift indicating potential collapse.",
                signals_involved=list(high.keys()),
            )

        # Comprehensive: multiple signals moderately elevated
        if len(elevated) >= 3:
            return DriftCharacterization(
                drift_type="comprehensive",
                confidence=min(1.0, gdi_result.composite_score / 0.5),
                description="Broad drift across multiple dimensions.",
                signals_involved=list(elevated.keys()),
            )

        # Single-signal dominance
        if scores["semantic"] > 0.3 and scores["semantic"] > max(
            scores["lexical"], scores["structural"], scores["distributional"]
        ):
            return DriftCharacterization(
                drift_type="reasoning_shift",
                confidence=scores["semantic"],
                description="Drift primarily in semantic content / reasoning patterns.",
                signals_involved=["semantic"],
            )

        if scores["lexical"] > 0.3 and scores["lexical"] > max(
            scores["semantic"], scores["structural"], scores["distributional"]
        ):
            return DriftCharacterization(
                drift_type="style_shift",
                confidence=scores["lexical"],
                description="Drift primarily in vocabulary and word choice.",
                signals_involved=["lexical"],
            )

        if scores["structural"] > 0.3 and scores["structural"] > max(
            scores["semantic"], scores["lexical"], scores["distributional"]
        ):
            return DriftCharacterization(
                drift_type="structural_shift",
                confidence=scores["structural"],
                description="Drift primarily in text structure and syntax.",
                signals_involved=["structural"],
            )

        if scores["distributional"] > 0.3:
            return DriftCharacterization(
                drift_type="distributional_shift",
                confidence=scores["distributional"],
                description="Drift primarily in token distribution patterns.",
                signals_involved=["distributional"],
            )

        # Minimal drift
        return DriftCharacterization(
            drift_type="minimal",
            confidence=1.0 - gdi_result.composite_score,
            description="No significant drift pattern detected.",
            signals_involved=[],
        )
