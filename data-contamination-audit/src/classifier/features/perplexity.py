"""Perplexity-based features for detecting AI-generated text.

Uses a reference language model (GPT-2 124M by default) to compute
per-document perplexity statistics.  Lower perplexity signals more
predictable text — a strong indicator of machine generation.
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class PerplexityScorer:
    """Score text perplexity under a reference causal language model.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier for the reference LM.
    device:
        ``"auto"`` picks CUDA when available, else CPU.
    stride:
        Sliding-window stride (in tokens) for handling long documents.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "auto",
        stride: int = 512,
    ) -> None:
        self.model_name = model_name
        self.stride = stride

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info("Loading tokenizer and model %s on %s", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # GPT-2 context window
        self.max_length: int = self.model.config.n_positions  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def score(self, text: str) -> float:
        """Return the perplexity of *text* under the reference LM.

        Uses a sliding window with ``self.stride`` to handle texts longer
        than the model context window.

        Lower perplexity means the model finds the text more predictable.
        """
        encodings = self.tokenizer(text, return_tensors="pt")
        input_ids: torch.Tensor = encodings.input_ids  # type: ignore[assignment]
        seq_len = input_ids.size(1)

        if seq_len == 0:
            return float("inf")

        nlls: list[float] = []
        prev_end = 0

        for begin in range(0, seq_len, self.stride):
            end = min(begin + self.max_length, seq_len)
            # Determine target start — only score new tokens in the window
            target_begin = max(begin, prev_end)

            ids = input_ids[:, begin:end].to(self.device)

            with torch.no_grad():
                outputs = self.model(ids, labels=ids)

            # The model's built-in loss averages over all positions after
            # the first, but we need finer control for the sliding window.
            # Re-derive the per-token NLL for the target region only.
            logits = outputs.logits  # (1, L, V)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = ids[:, 1:].contiguous()

            loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
            token_nll = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )  # (L-1,)

            # Offset into the window for the target region
            offset = target_begin - begin
            # The shift operation means position i in token_nll corresponds
            # to predicting token (begin + i + 1).  We want tokens from
            # target_begin+1 .. end, so offset = target_begin - begin.
            target_nll = token_nll[offset:]
            nlls.extend(target_nll.tolist())

            prev_end = end
            if end == seq_len:
                break

        if not nlls:
            return float("inf")

        avg_nll = sum(nlls) / len(nlls)
        perplexity = math.exp(avg_nll)
        return perplexity

    def score_batch(self, texts: list[str]) -> list[float]:
        """Score a list of texts sequentially.

        For simplicity this iterates; the sliding-window approach makes
        true batching complex.
        """
        return [self.score(t) for t in texts]

    # ------------------------------------------------------------------
    # Burstiness / chunk-level features
    # ------------------------------------------------------------------

    def compute_features(
        self,
        text: str,
        chunk_size: int = 256,
    ) -> dict[str, float]:
        """Compute all perplexity-derived features for a document.

        Returns
        -------
        dict with keys:
            - ``perplexity_mean``: overall document perplexity
            - ``perplexity_std``: standard deviation across chunks
            - ``perplexity_burstiness``: coefficient of variation (std / mean)
        """
        overall_ppl = self.score(text)

        # Split into roughly equal chunks by token count for burstiness.
        encodings = self.tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids[0]  # type: ignore[index]
        n_tokens = len(input_ids)

        if n_tokens < chunk_size * 2:
            # Not enough tokens for meaningful burstiness — return zeros.
            return {
                "perplexity_mean": overall_ppl,
                "perplexity_std": 0.0,
                "perplexity_burstiness": 0.0,
            }

        chunk_ppls: list[float] = []
        for start in range(0, n_tokens, chunk_size):
            end = min(start + chunk_size, n_tokens)
            if end - start < 16:
                # Ignore very short trailing chunks.
                continue
            chunk_text = self.tokenizer.decode(input_ids[start:end])
            ppl = self.score(chunk_text)
            if math.isfinite(ppl):
                chunk_ppls.append(ppl)

        if len(chunk_ppls) < 2:
            return {
                "perplexity_mean": overall_ppl,
                "perplexity_std": 0.0,
                "perplexity_burstiness": 0.0,
            }

        mean_ppl = sum(chunk_ppls) / len(chunk_ppls)
        var_ppl = sum((p - mean_ppl) ** 2 for p in chunk_ppls) / len(chunk_ppls)
        std_ppl = math.sqrt(var_ppl)
        burstiness = std_ppl / mean_ppl if mean_ppl > 0 else 0.0

        return {
            "perplexity_mean": overall_ppl,
            "perplexity_std": std_ppl,
            "perplexity_burstiness": burstiness,
        }
