"""EntropyFloorConstraint: minimum output diversity measured by entropy metrics."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List

from src.constraints.base import Constraint, ConstraintResult, CheckContext


class EntropyFloorConstraint(Constraint):
    """Output diversity must remain above a minimum entropy threshold."""

    def __init__(self, threshold: float = 3.5) -> None:
        super().__init__(
            name="entropy_floor",
            description="Minimum output diversity measured by token entropy",
            category="quality",
            threshold=threshold,
        )

    def check(self, agent_state: Any, context: CheckContext) -> ConstraintResult:
        """Evaluate entropy.

        ``agent_state`` must expose:
        * ``generate_probe_outputs(probes) -> List[str]``
        """
        from src.evaluation.diversity_probes import DiversityProbes

        probes = DiversityProbes()
        probe_inputs = probes.generate_probes()
        outputs = agent_state.generate_probe_outputs(probe_inputs)

        metrics = self._compute_diversity_metrics(outputs)
        token_entropy = metrics["token_entropy"]

        headroom = self.headroom(token_entropy)
        return ConstraintResult(
            satisfied=token_entropy >= self._threshold,
            measured_value=token_entropy,
            threshold=self._threshold,
            headroom=headroom,
            details=metrics,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _compute_diversity_metrics(outputs: List[str]) -> Dict[str, float]:
        """Compute token_entropy, distinct_n, self_bleu, vocab_size."""
        all_tokens: List[str] = []
        for output in outputs:
            all_tokens.extend(output.lower().split())

        # token entropy
        token_entropy = _shannon_entropy(all_tokens)

        # distinct-1 and distinct-2
        if all_tokens:
            unigrams = set(all_tokens)
            bigrams = set(zip(all_tokens[:-1], all_tokens[1:]))
            distinct_1 = len(unigrams) / len(all_tokens)
            distinct_2 = len(bigrams) / max(len(all_tokens) - 1, 1)
        else:
            distinct_1 = 0.0
            distinct_2 = 0.0

        # self-BLEU approximation (lower is more diverse)
        self_bleu = _approx_self_bleu(outputs)

        vocab_size = len(set(all_tokens))

        return {
            "token_entropy": token_entropy,
            "distinct_1": distinct_1,
            "distinct_2": distinct_2,
            "self_bleu": self_bleu,
            "vocab_size": vocab_size,
        }


def _shannon_entropy(tokens: List[str]) -> float:
    """Compute Shannon entropy in bits."""
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _approx_self_bleu(outputs: List[str]) -> float:
    """Approximate self-BLEU using unigram overlap (lower = more diverse)."""
    if len(outputs) < 2:
        return 0.0

    scores: List[float] = []
    for i, ref in enumerate(outputs):
        ref_tokens = set(ref.lower().split())
        if not ref_tokens:
            continue
        for j, hyp in enumerate(outputs):
            if i == j:
                continue
            hyp_tokens = set(hyp.lower().split())
            if not hyp_tokens:
                continue
            overlap = len(ref_tokens & hyp_tokens)
            scores.append(overlap / len(hyp_tokens))

    return sum(scores) / len(scores) if scores else 0.0
