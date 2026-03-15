"""BDM calibration: tests BDM scoring on known-complexity strings."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.bdm.scorer import BDMScorer


@dataclass
class CalibrationResult:
    """Result for a single calibration test case."""

    name: str
    data: str
    bdm_score: float
    expected_ordering: str  # "low", "medium", "high"
    baselines: Dict[str, float] = field(default_factory=dict)


@dataclass
class CalibrationReport:
    """Full calibration report."""

    results: List[CalibrationResult] = field(default_factory=list)
    ordering_correct: bool = False
    summary: str = ""

    @property
    def num_tests(self) -> int:
        return len(self.results)


class BDMCalibrator:
    """Calibrates BDM scoring by testing against strings of known complexity.

    Tests three categories:
    - Constant strings (low complexity): "000...0", "111...1"
    - Periodic strings (medium complexity): "010101...", "001001..."
    - Random strings (high complexity): random binary strings
    """

    def __init__(self, scorer: Optional[BDMScorer] = None) -> None:
        self.scorer = scorer or BDMScorer()

    def run_calibration(self, length: int = 48, seed: int = 42) -> CalibrationReport:
        """Run the full calibration suite.

        Args:
            length: Length of test strings.
            seed: Random seed for reproducibility.

        Returns:
            CalibrationReport with all results.
        """
        rng = random.Random(seed)
        results = []

        # --- Constant strings (low complexity) ---
        constant_strings = [
            ("constant_zeros", "0" * length),
            ("constant_ones", "1" * length),
        ]

        for name, data in constant_strings:
            score = self.scorer.score(data)
            baselines = self.scorer.compare_to_baselines(data)
            results.append(
                CalibrationResult(
                    name=name,
                    data=data,
                    bdm_score=score.bdm_value,
                    expected_ordering="low",
                    baselines=baselines,
                )
            )

        # --- Periodic strings (medium complexity) ---
        periodic_strings = [
            ("periodic_01", ("01" * (length // 2))[:length]),
            ("periodic_001", ("001" * (length // 3 + 1))[:length]),
            ("periodic_0011", ("0011" * (length // 4 + 1))[:length]),
        ]

        for name, data in periodic_strings:
            score = self.scorer.score(data)
            baselines = self.scorer.compare_to_baselines(data)
            results.append(
                CalibrationResult(
                    name=name,
                    data=data,
                    bdm_score=score.bdm_value,
                    expected_ordering="medium",
                    baselines=baselines,
                )
            )

        # --- Random strings (high complexity) ---
        for i in range(3):
            data = "".join(rng.choice("01") for _ in range(length))
            name = f"random_{i}"
            score = self.scorer.score(data)
            baselines = self.scorer.compare_to_baselines(data)
            results.append(
                CalibrationResult(
                    name=name,
                    data=data,
                    bdm_score=score.bdm_value,
                    expected_ordering="high",
                    baselines=baselines,
                )
            )

        # Check ordering: constant < periodic < random (on average)
        low_scores = [r.bdm_score for r in results if r.expected_ordering == "low"]
        med_scores = [r.bdm_score for r in results if r.expected_ordering == "medium"]
        high_scores = [r.bdm_score for r in results if r.expected_ordering == "high"]

        avg_low = sum(low_scores) / len(low_scores) if low_scores else 0
        avg_med = sum(med_scores) / len(med_scores) if med_scores else 0
        avg_high = sum(high_scores) / len(high_scores) if high_scores else 0

        ordering_correct = avg_low <= avg_med <= avg_high

        summary_lines = [
            f"Calibration Report ({len(results)} tests)",
            f"  Constant (low):  avg BDM = {avg_low:.2f}",
            f"  Periodic (med):  avg BDM = {avg_med:.2f}",
            f"  Random (high):   avg BDM = {avg_high:.2f}",
            f"  Ordering correct: {ordering_correct}",
        ]

        return CalibrationReport(
            results=results,
            ordering_correct=ordering_correct,
            summary="\n".join(summary_lines),
        )
