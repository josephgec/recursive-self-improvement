"""RSI suitability assessment: scores architectures for recursive self-improvement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.evaluation.benchmark_suite import BenchmarkResults


@dataclass
class RSIAssessment:
    """Assessment of RSI suitability for a system."""
    system: str
    modularity: float = 0.0
    verifiability: float = 0.0
    composability: float = 0.0
    contamination_resistance: float = 0.0
    transparency: float = 0.0
    overall_score: float = 0.0
    recommendation: str = ""
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class RSISuitabilityAssessor:
    """Assesses how suitable each architecture is for recursive self-improvement.

    RSI dimensions:
    - Modularity: can components be independently improved?
    - Verifiability: can improvements be verified?
    - Composability: can improvements compose?
    - Contamination resistance: does self-modification avoid regression?
    - Transparency: is the improvement process inspectable?
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.weights = weights or {
            "modularity": 0.2,
            "verifiability": 0.25,
            "composability": 0.2,
            "contamination_resistance": 0.15,
            "transparency": 0.2,
        }

    def assess(self, results: BenchmarkResults) -> Dict[str, RSIAssessment]:
        """Assess RSI suitability for all systems.

        Args:
            results: BenchmarkResults from benchmark suite.

        Returns:
            Dict mapping system name to RSIAssessment.
        """
        assessments: Dict[str, RSIAssessment] = {}

        systems = set(
            list(results.generalization.keys())
            + list(results.interpretability.keys())
            + list(results.robustness.keys())
        )

        for system in systems:
            modularity = self._score_modularity(system, results)
            verifiability = self._score_verifiability(system, results)
            composability = self._score_composability(system, results)
            contamination = self._score_contamination_resistance(system, results)
            transparency = self._score_transparency(system, results)

            overall = (
                self.weights["modularity"] * modularity
                + self.weights["verifiability"] * verifiability
                + self.weights["composability"] * composability
                + self.weights["contamination_resistance"] * contamination
                + self.weights["transparency"] * transparency
            )

            strengths, weaknesses = self._identify_strengths_weaknesses(
                system, modularity, verifiability, composability,
                contamination, transparency,
            )

            recommendation = self._make_recommendation(system, overall)

            assessments[system] = RSIAssessment(
                system=system,
                modularity=modularity,
                verifiability=verifiability,
                composability=composability,
                contamination_resistance=contamination,
                transparency=transparency,
                overall_score=overall,
                recommendation=recommendation,
                strengths=strengths,
                weaknesses=weaknesses,
            )

        return assessments

    def _score_modularity(self, system: str, results: BenchmarkResults) -> float:
        """Score modularity: can components be independently improved?

        Hybrid: high (LLM, tools, integrator are separate modules)
        Integrative: medium (constraints + model are coupled)
        Prose: low (monolithic)
        """
        base_scores = {"hybrid": 0.9, "integrative": 0.6, "prose": 0.2}
        return base_scores.get(system, 0.3)

    def _score_verifiability(self, system: str, results: BenchmarkResults) -> float:
        """Score verifiability: can improvements be verified?

        Uses interpretability results if available.
        """
        interp = results.interpretability.get(system)
        if interp:
            return interp.step_verifiability
        base_scores = {"hybrid": 0.85, "integrative": 0.5, "prose": 0.15}
        return base_scores.get(system, 0.3)

    def _score_composability(self, system: str, results: BenchmarkResults) -> float:
        """Score composability: can improvements compose?

        Hybrid: high (add tools, swap LLM)
        Integrative: medium (add constraints)
        Prose: low (retraining only)
        """
        gen = results.generalization.get(system)
        if gen and gen.transfer_ratio > 0.5:
            bonus = 0.1
        else:
            bonus = 0.0

        base_scores = {"hybrid": 0.85, "integrative": 0.55, "prose": 0.25}
        return min(1.0, base_scores.get(system, 0.3) + bonus)

    def _score_contamination_resistance(
        self, system: str, results: BenchmarkResults
    ) -> float:
        """Score contamination resistance: does modification avoid regression?

        Uses robustness results if available.
        """
        robust = results.robustness.get(system)
        if robust:
            return robust.consistency
        base_scores = {"hybrid": 0.7, "integrative": 0.8, "prose": 0.4}
        return base_scores.get(system, 0.3)

    def _score_transparency(self, system: str, results: BenchmarkResults) -> float:
        """Score transparency: is the improvement process inspectable?

        Hybrid: high (tool calls are logged)
        Integrative: medium (constraints are documented)
        Prose: low (opaque)
        """
        interp = results.interpretability.get(system)
        if interp:
            return interp.readability
        base_scores = {"hybrid": 0.8, "integrative": 0.5, "prose": 0.2}
        return base_scores.get(system, 0.3)

    def _identify_strengths_weaknesses(
        self,
        system: str,
        modularity: float,
        verifiability: float,
        composability: float,
        contamination: float,
        transparency: float,
    ) -> tuple:
        """Identify strengths and weaknesses."""
        dimensions = {
            "modularity": modularity,
            "verifiability": verifiability,
            "composability": composability,
            "contamination_resistance": contamination,
            "transparency": transparency,
        }
        strengths = [d for d, s in dimensions.items() if s >= 0.7]
        weaknesses = [d for d, s in dimensions.items() if s < 0.4]
        return strengths, weaknesses

    def _make_recommendation(self, system: str, overall: float) -> str:
        """Make a recommendation based on overall score."""
        if overall >= 0.7:
            return f"{system} is highly suitable for RSI applications."
        elif overall >= 0.5:
            return f"{system} has moderate RSI suitability; address weaknesses."
        else:
            return f"{system} has limited RSI suitability; significant architectural changes needed."
