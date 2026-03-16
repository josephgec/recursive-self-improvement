"""Dual verifier: runs empirical and compactness gates, then Pareto filter."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.outer_loop.strategy_evolver import Candidate
from src.pipeline.state import PipelineState
from src.verification.empirical_gate import EmpiricalGate, EmpiricalResult
from src.verification.compactness_gate import CompactnessGate, CompactnessResult
from src.verification.pareto_filter import ParetoFilter
from src.verification.verification_cache import VerificationCache


@dataclass
class VerifiedCandidate:
    """A candidate that has passed both verification gates."""
    candidate: Candidate = field(default_factory=Candidate)
    empirical: EmpiricalResult = field(default_factory=EmpiricalResult)
    compactness: CompactnessResult = field(default_factory=CompactnessResult)
    pareto_optimal: bool = False
    combined_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate.candidate_id,
            "empirical": self.empirical.to_dict(),
            "compactness": self.compactness.to_dict(),
            "pareto_optimal": self.pareto_optimal,
            "combined_score": self.combined_score,
        }


class DualVerifier:
    """Verifies candidates through empirical and compactness gates, then Pareto filter."""

    def __init__(
        self,
        empirical_gate: Optional[EmpiricalGate] = None,
        compactness_gate: Optional[CompactnessGate] = None,
        pareto_filter: Optional[ParetoFilter] = None,
        cache: Optional[VerificationCache] = None,
    ):
        self._empirical = empirical_gate or EmpiricalGate()
        self._compactness = compactness_gate or CompactnessGate()
        self._pareto = pareto_filter or ParetoFilter()
        self._cache = cache or VerificationCache()

    def verify_all(self, candidates: List[Candidate], state: PipelineState) -> List[VerifiedCandidate]:
        """Verify all candidates through both gates and Pareto filter.

        Returns verified candidates sorted by combined score (descending).
        """
        verified: List[VerifiedCandidate] = []

        for candidate in candidates:
            # Check cache
            if self._cache.has(candidate.candidate_id):
                cached = self._cache.get(candidate.candidate_id)
                if cached is not None:
                    verified.append(cached)
                    continue

            # Empirical gate
            emp_result = self._empirical.evaluate(candidate, state)
            if not emp_result.passed:
                continue

            # Compactness gate
            comp_result = self._compactness.evaluate(candidate, emp_result)
            if not comp_result.passed:
                continue

            # Compute combined score
            combined = emp_result.accuracy * 0.7 + (1.0 - comp_result.complexity_ratio) * 0.3

            vc = VerifiedCandidate(
                candidate=candidate,
                empirical=emp_result,
                compactness=comp_result,
                combined_score=combined,
            )

            self._cache.put(candidate.candidate_id, vc)
            verified.append(vc)

        if not verified:
            return []

        # Apply Pareto filter
        pareto_input = [
            {
                "idx": i,
                "accuracy": vc.empirical.accuracy,
                "bdm_score": vc.compactness.bdm_score,
            }
            for i, vc in enumerate(verified)
        ]
        pareto_optimal = self._pareto.filter(pareto_input)
        pareto_indices = {p["idx"] for p in pareto_optimal}

        for i, vc in enumerate(verified):
            vc.pareto_optimal = i in pareto_indices

        # Sort by combined score descending
        verified.sort(key=lambda v: v.combined_score, reverse=True)
        return verified
