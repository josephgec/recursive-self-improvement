"""KL divergence estimation for model-collapse detection.

Compares a model's output distribution Q against a reference distribution P
(typically the empirical unigram distribution from real training data).
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class KLResult:
    """Results from KL divergence estimation."""

    kl_p_q: float  # KL(P || Q) -- how well Q approximates P
    kl_q_p: float  # KL(Q || P) -- how well P approximates Q
    js_divergence: float  # Jensen-Shannon divergence (symmetric)
    total_variation: float  # Total variation distance


class KLDivergenceEstimator:
    """Estimate KL divergence between a reference distribution and a model.

    The reference distribution P is typically the empirical unigram token
    distribution from real data (computed by ``RealDataLoader.get_token_distribution``).
    The model distribution Q is estimated from the model's output.
    """

    def __init__(
        self,
        reference_distribution: np.ndarray,
        tokenizer,
        num_bins: int | None = None,
    ) -> None:
        """Initialize with a reference distribution.

        Args:
            reference_distribution: 1-D array of shape ``(vocab_size,)``
                summing to 1.  This is P.
            tokenizer: Tokenizer for converting texts to token IDs.
            num_bins: Not used for token-level KL (kept for API
                compatibility).  The "bins" are naturally the vocabulary
                entries.
        """
        self._reference = np.array(reference_distribution, dtype=np.float64)
        self._tokenizer = tokenizer
        self._num_bins = num_bins
        self._vocab_size = len(self._reference)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_kl(self, model, eval_texts: list[str], batch_size: int = 8) -> KLResult:
        """Estimate divergence between reference P and model Q.

        Uses teacher-forcing to obtain the model's average unigram
        predictive distribution.

        Args:
            model: A HuggingFace causal LM.
            eval_texts: Evaluation texts to compute model logits on.
            batch_size: Batch size for processing.

        Returns:
            A ``KLResult`` with KL, JS, and TV distances.
        """
        import torch

        model.eval()
        device = next(model.parameters()).device

        # Accumulate the model's average token distribution.
        q_counts = np.zeros(self._vocab_size, dtype=np.float64)

        for start in range(0, len(eval_texts), batch_size):
            batch_texts = eval_texts[start : start + batch_size]
            inputs = self._tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits  # (B, L, V)

            # Average the softmax probabilities across all valid positions.
            probs = torch.softmax(logits, dim=-1)  # (B, L, V)
            attention_mask = inputs.get("attention_mask")

            if attention_mask is not None:
                # Expand mask for broadcasting.
                mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
                masked_probs = probs * mask
                # Sum across batch and sequence.
                summed = masked_probs.sum(dim=(0, 1)).cpu().numpy()
                total_positions = mask.sum().item()
            else:
                summed = probs.sum(dim=(0, 1)).cpu().numpy()
                total_positions = probs.shape[0] * probs.shape[1]

            # Truncate or pad if model vocab != reference vocab.
            min_v = min(len(summed), self._vocab_size)
            q_counts[:min_v] += summed[:min_v]

            del inputs, outputs, logits, probs

        # Normalize to get Q.
        total = q_counts.sum()
        if total > 0:
            q = q_counts / total
        else:
            q = np.ones(self._vocab_size, dtype=np.float64) / self._vocab_size

        return self._compute_divergences(self._reference, q)

    def estimate_from_generated_text(self, texts: list[str]) -> KLResult:
        """Estimate divergence from pre-generated text.

        Builds Q from the empirical unigram distribution of tokenized
        generated text, then compares against P.

        Args:
            texts: List of generated text strings.

        Returns:
            A ``KLResult`` with KL, JS, and TV distances.
        """
        q = self.compute_token_distribution(texts)
        return self._compute_divergences(self._reference, q)

    def compute_token_distribution(self, texts: list[str]) -> np.ndarray:
        """Compute empirical unigram distribution from texts.

        Args:
            texts: List of text strings.

        Returns:
            1-D array of shape ``(vocab_size,)`` summing to 1, with
            Laplace smoothing applied.
        """
        counts = np.zeros(self._vocab_size, dtype=np.float64)

        for text in texts:
            token_ids = self._tokenizer.encode(text, add_special_tokens=False)
            for tid in token_ids:
                if 0 <= tid < self._vocab_size:
                    counts[tid] += 1

        # Laplace smoothing.
        counts += 1.0
        return counts / counts.sum()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_divergences(self, p: np.ndarray, q: np.ndarray) -> KLResult:
        """Compute KL(P||Q), KL(Q||P), JS, and TV from two distributions.

        Both *p* and *q* must be normalized probability distributions of
        the same length.  Laplace smoothing is applied to avoid log(0).
        """
        # Apply Laplace smoothing to prevent log(0).
        eps = 1e-10
        p_smooth = p + eps
        p_smooth = p_smooth / p_smooth.sum()
        q_smooth = q + eps
        q_smooth = q_smooth / q_smooth.sum()

        # KL(P || Q) = sum P * log(P / Q)
        kl_p_q = float(np.sum(p_smooth * np.log(p_smooth / q_smooth)))

        # KL(Q || P) = sum Q * log(Q / P)
        kl_q_p = float(np.sum(q_smooth * np.log(q_smooth / p_smooth)))

        # Jensen-Shannon divergence: JS = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
        m = 0.5 * (p_smooth + q_smooth)
        js = 0.5 * float(np.sum(p_smooth * np.log(p_smooth / m)))
        js += 0.5 * float(np.sum(q_smooth * np.log(q_smooth / m)))

        # Total variation: TV = 0.5 * sum |P - Q|
        tv = 0.5 * float(np.sum(np.abs(p_smooth - q_smooth)))

        return KLResult(
            kl_p_q=kl_p_q,
            kl_q_p=kl_q_p,
            js_divergence=js,
            total_variation=tv,
        )
