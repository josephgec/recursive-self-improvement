"""DiversityProbes: generate probe inputs and compute entropy metrics."""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List


class DiversityProbes:
    """Generate probe inputs for measuring output diversity."""

    def generate_probes(self) -> List[str]:
        """Return a list of diverse probe inputs."""
        return [
            "Write a short poem about the ocean.",
            "Describe a sunset in three sentences.",
            "Tell me a story about a lost dog.",
            "Explain quantum physics in simple terms.",
            "Write a haiku about autumn.",
            "Describe your ideal vacation.",
            "What would you do if you could fly?",
            "Write a limerick about a programmer.",
            "Explain why the sky is blue.",
            "Describe the taste of chocolate.",
            "Write a dialogue between a cat and a dog.",
            "Explain machine learning to a child.",
            "Describe what music sounds like.",
            "Write a letter to your future self.",
            "Explain the concept of time.",
        ]

    @staticmethod
    def generate_probe_outputs(agent_state, probes: List[str]) -> List[str]:
        """Helper: use agent to generate outputs for probes."""
        return agent_state.generate_probe_outputs(probes)

    @staticmethod
    def compute_entropy(outputs: List[str]) -> Dict[str, float]:
        """Compute diversity metrics for a list of outputs."""
        all_tokens: List[str] = []
        for output in outputs:
            all_tokens.extend(output.lower().split())

        if not all_tokens:
            return {
                "token_entropy": 0.0,
                "distinct_1": 0.0,
                "distinct_2": 0.0,
                "vocab_size": 0,
            }

        # Shannon entropy
        counts = Counter(all_tokens)
        total = len(all_tokens)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        unigrams = set(all_tokens)
        bigrams = set(zip(all_tokens[:-1], all_tokens[1:]))

        return {
            "token_entropy": entropy,
            "distinct_1": len(unigrams) / total,
            "distinct_2": len(bigrams) / max(total - 1, 1),
            "vocab_size": len(unigrams),
        }
