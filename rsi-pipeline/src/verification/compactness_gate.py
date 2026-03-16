"""Compactness gate: BDM scoring of code complexity."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.outer_loop.strategy_evolver import Candidate
from src.verification.empirical_gate import EmpiricalResult


@dataclass
class CompactnessResult:
    """Result of compactness evaluation."""
    candidate_id: str = ""
    passed: bool = False
    bdm_score: float = 0.0
    complexity_ratio: float = 0.0
    code_length: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "passed": self.passed,
            "bdm_score": self.bdm_score,
            "complexity_ratio": self.complexity_ratio,
            "code_length": self.code_length,
            "error": self.error,
        }


class CompactnessGate:
    """Evaluates code compactness using BDM (Block Decomposition Method) scoring."""

    def __init__(self, max_bdm_score: float = 500, complexity_weight: float = 0.3):
        self._max_bdm_score = max_bdm_score
        self._complexity_weight = complexity_weight

    def evaluate(self, candidate: Candidate, empirical_result: EmpiricalResult) -> CompactnessResult:
        """Evaluate compactness of a candidate's code."""
        code = candidate.proposed_code
        if not code:
            return CompactnessResult(
                candidate_id=candidate.candidate_id,
                passed=False,
                error="empty_code",
            )

        bdm_score = self._compute_bdm(code)
        code_length = len(code)
        complexity_ratio = bdm_score / max(code_length, 1)
        passed = bdm_score <= self._max_bdm_score

        return CompactnessResult(
            candidate_id=candidate.candidate_id,
            passed=passed,
            bdm_score=bdm_score,
            complexity_ratio=complexity_ratio,
            code_length=code_length,
        )

    def _compute_bdm(self, code: str) -> float:
        """Compute a BDM-inspired complexity score for code.

        This is a simplified approximation:
        - Base: Shannon entropy of the code
        - Penalty for unique tokens
        - Bonus for repeated patterns (compressibility)
        """
        if not code:
            return 0.0

        # Shannon entropy
        freq: dict = {}
        for ch in code:
            freq[ch] = freq.get(ch, 0) + 1
        total = len(code)
        entropy = 0.0
        for count in freq.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        # Unique token count (words)
        tokens = code.split()
        unique_tokens = len(set(tokens))

        # BDM approximation: entropy * length_factor + unique_token_penalty
        length_factor = math.log2(max(len(code), 2))
        bdm = entropy * length_factor + unique_tokens * self._complexity_weight

        return round(bdm, 2)
