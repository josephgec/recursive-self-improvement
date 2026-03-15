"""Entropy measurement for model-collapse detection.

Computes per-position predictive entropy (how uncertain the model is)
and sequence-level n-gram entropy (how diverse the generated text is).
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class TokenEntropyResult:
    """Results from per-position predictive entropy measurement."""

    mean: float
    median: float
    std: float
    low_entropy_fraction: float  # fraction of positions with H < 1.0
    entropy_histogram: np.ndarray  # binned histogram of entropy values


@dataclass
class SequenceEntropyResult:
    """Results from sequence-level n-gram entropy."""

    unigram: float
    bigram: float
    trigram: float


@dataclass
class ConditionalEntropyTrajectory:
    """Per-position entropy averaged across texts."""

    position_entropies: np.ndarray  # shape (max_len,)
    num_texts: int
    max_length: int


class EntropyMeasurer:
    """Measure predictive and sequence entropy for model-collapse tracking.

    Predictive entropy uses teacher-forcing: feed the real text through
    the model and measure H(p) = -sum p(v) log p(v) at each position.
    Sequence entropy measures the diversity of generated text itself via
    n-gram frequency distributions.
    """

    def __init__(self, batch_size: int = 8) -> None:
        self._batch_size = batch_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def token_entropy(self, model, eval_texts: list[str], tokenizer) -> TokenEntropyResult:
        """Compute per-position predictive entropy H(p) = -sum p(v) log p(v).

        Uses teacher-forcing: the model sees the real tokens and we read
        off the predictive distribution at each position.

        Args:
            model: A HuggingFace causal LM (``AutoModelForCausalLM``).
            eval_texts: List of evaluation text strings.
            tokenizer: The tokenizer corresponding to *model*.

        Returns:
            A ``TokenEntropyResult`` with mean, median, std, and histogram.
        """
        import torch

        model.eval()
        device = next(model.parameters()).device
        all_entropies: list[float] = []

        for start in range(0, len(eval_texts), self._batch_size):
            batch_texts = eval_texts[start : start + self._batch_size]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits  # (batch, seq_len, vocab)

            # Convert logits to log-probabilities.
            log_probs = torch.log_softmax(logits, dim=-1)  # (B, L, V)

            # Entropy at each position: H = -sum p * log(p)
            probs = torch.exp(log_probs)
            entropies = -(probs * log_probs).sum(dim=-1)  # (B, L)

            # Mask out padding positions.
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                # Shift: logits at position i predict token i+1, so
                # entropy at position i is meaningful for positions 0..L-2.
                # We keep all non-padded positions for simplicity.
                entropies = entropies * attention_mask.float()
                # Collect non-padded entropies.
                for b in range(entropies.shape[0]):
                    seq_len = int(attention_mask[b].sum().item())
                    # Exclude last position (no next token to predict).
                    valid_len = max(seq_len - 1, 1)
                    pos_entropies = entropies[b, :valid_len].cpu().numpy()
                    all_entropies.extend(pos_entropies.tolist())
            else:
                for b in range(entropies.shape[0]):
                    pos_entropies = entropies[b, :-1].cpu().numpy()
                    all_entropies.extend(pos_entropies.tolist())

            del inputs, outputs, logits, log_probs, probs, entropies

        arr = np.array(all_entropies, dtype=np.float64)
        histogram, _ = np.histogram(arr, bins=50, range=(0, float(max(arr.max(), 1.0))))

        return TokenEntropyResult(
            mean=float(np.mean(arr)),
            median=float(np.median(arr)),
            std=float(np.std(arr)),
            low_entropy_fraction=float(np.mean(arr < 1.0)),
            entropy_histogram=histogram,
        )

    def sequence_entropy(self, texts: list[str]) -> SequenceEntropyResult:
        """Compute unigram, bigram, and trigram entropy of generated text.

        Uses simple whitespace tokenization (not model tokenization) to
        measure the diversity of surface-form text.

        Args:
            texts: List of generated text strings.

        Returns:
            A ``SequenceEntropyResult`` with unigram, bigram, and trigram
            entropy values.
        """
        # Collect all tokens across texts.
        all_tokens: list[str] = []
        for text in texts:
            all_tokens.extend(text.lower().split())

        if not all_tokens:
            return SequenceEntropyResult(unigram=0.0, bigram=0.0, trigram=0.0)

        unigram_h = self._ngram_entropy(all_tokens, n=1)
        bigram_h = self._ngram_entropy(all_tokens, n=2)
        trigram_h = self._ngram_entropy(all_tokens, n=3)

        return SequenceEntropyResult(
            unigram=unigram_h,
            bigram=bigram_h,
            trigram=trigram_h,
        )

    def conditional_entropy_trajectory(
        self,
        model,
        eval_texts: list[str],
        tokenizer,
        max_length: int = 512,
    ) -> ConditionalEntropyTrajectory:
        """Compute per-position entropy averaged across texts.

        This gives a "trajectory" showing how entropy evolves along the
        sequence, averaged over all evaluation texts.

        Args:
            model: A HuggingFace causal LM.
            eval_texts: List of evaluation text strings.
            tokenizer: The tokenizer corresponding to *model*.
            max_length: Maximum sequence length to consider.

        Returns:
            A ``ConditionalEntropyTrajectory`` with per-position averages.
        """
        import torch

        model.eval()
        device = next(model.parameters()).device

        entropy_sums = np.zeros(max_length, dtype=np.float64)
        entropy_counts = np.zeros(max_length, dtype=np.int64)

        for start in range(0, len(eval_texts), self._batch_size):
            batch_texts = eval_texts[start : start + self._batch_size]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

            log_probs = torch.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)
            entropies = -(probs * log_probs).sum(dim=-1)  # (B, L)

            attention_mask = inputs.get("attention_mask")
            entropies_np = entropies.cpu().numpy()

            for b in range(entropies_np.shape[0]):
                if attention_mask is not None:
                    seq_len = int(attention_mask[b].sum().item())
                else:
                    seq_len = entropies_np.shape[1]
                valid_len = min(seq_len, max_length)
                entropy_sums[:valid_len] += entropies_np[b, :valid_len]
                entropy_counts[:valid_len] += 1

            del inputs, outputs, logits, log_probs, probs, entropies

        # Average, avoiding division by zero.
        mask = entropy_counts > 0
        position_entropies = np.zeros(max_length, dtype=np.float64)
        position_entropies[mask] = entropy_sums[mask] / entropy_counts[mask]

        # Trim trailing zeros.
        effective_len = int(np.max(np.where(mask)[0]) + 1) if mask.any() else 0
        position_entropies = position_entropies[:effective_len]

        return ConditionalEntropyTrajectory(
            position_entropies=position_entropies,
            num_texts=len(eval_texts),
            max_length=effective_len,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _ngram_entropy(tokens: list[str], n: int) -> float:
        """Compute Shannon entropy of the n-gram distribution.

        H = -sum_g p(g) * log2(p(g))

        Args:
            tokens: List of tokens (words).
            n: N-gram order (1=unigram, 2=bigram, etc.).

        Returns:
            Entropy in bits.
        """
        if len(tokens) < n:
            return 0.0

        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        counts = Counter(ngrams)
        total = sum(counts.values())

        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy
