"""Goal Drift Index (GDI) — measures how far generated outputs drift from
the reference goal established at generation 0.

Composite GDI is a weighted sum of four sub-metrics:
  * semantic  – cosine distance of mean word-frequency vectors
  * lexical   – Jensen-Shannon divergence of word distributions
  * structural – sentence-length distribution divergence (JS)
  * distributional – KL divergence of token-level distributions
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance
from scipy.spatial.distance import jensenshannon
from scipy.special import kl_div as _scipy_kl  # element-wise KL


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GoalDriftMeasurement:
    """Single drift measurement for one generation."""

    generation: int
    timestamp: str
    # semantic drift (cosine distance of mean word-freq vectors)
    semantic_drift: float
    semantic_drift_max: float
    # lexical drift (Jensen-Shannon divergence of word distributions)
    lexical_drift: float
    lexical_top_k_overlap: float
    # structural drift (sentence-length distribution divergence)
    structural_drift: float
    mean_depth_change: float
    # distributional drift (KL divergence of token distributions)
    distributional_drift: float
    # composite
    goal_drift_index: float
    alert: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _word_counts(texts: list[str]) -> Counter:
    """Aggregate word counts across all texts (lowercased, alphanumeric only)."""
    counter: Counter = Counter()
    for text in texts:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        counter.update(tokens)
    return counter


def _counts_to_vec(counts: Counter, vocab: list[str]) -> np.ndarray:
    """Convert Counter to a numpy vector ordered by *vocab*."""
    vec = np.array([float(counts.get(w, 0)) for w in vocab], dtype=np.float64)
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


def _sentence_lengths(texts: list[str]) -> list[int]:
    """Split texts into sentences and return lengths (word counts)."""
    lengths: list[int] = []
    for text in texts:
        sentences = re.split(r"[.!?]+", text)
        for s in sentences:
            words = s.strip().split()
            if words:
                lengths.append(len(words))
    return lengths


def _length_distribution(lengths: list[int], max_len: int = 100) -> np.ndarray:
    """Histogram of sentence lengths, normalised to a probability distribution."""
    bins = np.zeros(max_len + 1, dtype=np.float64)
    for l in lengths:
        idx = min(l, max_len)
        bins[idx] += 1.0
    total = bins.sum()
    if total > 0:
        bins /= total
    # add tiny epsilon to avoid log(0)
    bins += 1e-12
    bins /= bins.sum()
    return bins


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q), element-wise then summed.  Both must be positive and sum to 1."""
    # Ensure positivity
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GoalDriftComputer:
    """Computes the Goal Drift Index across generations.

    Parameters
    ----------
    config : dict
        Must contain a ``safety`` sub-dict with ``weights`` (semantic, lexical,
        structural, distributional) and ``alert_threshold_drift_cosine``.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        safety = config.get("safety", config)  # accept flat or nested
        weights = safety.get("weights", {})
        self._w_semantic: float = weights.get("semantic", 0.3)
        self._w_lexical: float = weights.get("lexical", 0.2)
        self._w_structural: float = weights.get("structural", 0.2)
        self._w_distributional: float = weights.get("distributional", 0.3)
        self._alert_threshold: float = safety.get("alert_threshold_drift_cosine", 0.15)

        # reference baselines (set by set_reference)
        self._ref_word_counts: Counter | None = None
        self._ref_vocab: list[str] | None = None
        self._ref_vec: np.ndarray | None = None
        self._ref_sent_dist: np.ndarray | None = None
        self._ref_token_dist: np.ndarray | None = None
        self._ref_mean_sent_len: float = 0.0

        self._trajectory: list[GoalDriftMeasurement] = []

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def set_reference(
        self,
        reference_texts: list[str],
        reference_distribution: np.ndarray | None = None,
    ) -> None:
        """Capture generation-0 baselines."""
        self._ref_word_counts = _word_counts(reference_texts)
        self._ref_vocab = list(self._ref_word_counts.keys())
        self._ref_vec = _counts_to_vec(self._ref_word_counts, self._ref_vocab)

        sent_lens = _sentence_lengths(reference_texts)
        self._ref_sent_dist = _length_distribution(sent_lens)
        self._ref_mean_sent_len = float(np.mean(sent_lens)) if sent_lens else 0.0

        if reference_distribution is not None:
            self._ref_token_dist = np.asarray(reference_distribution, dtype=np.float64)
        else:
            self._ref_token_dist = None

    def compute(
        self,
        generation: int,
        generated_texts: list[str],
        token_distribution: np.ndarray | None = None,
    ) -> GoalDriftMeasurement:
        """Compute drift metrics for *generation* against the reference."""
        if self._ref_word_counts is None:
            raise RuntimeError("Call set_reference() before compute().")

        # --- semantic drift (cosine distance of word-freq vectors) ---
        cur_counts = _word_counts(generated_texts)
        # build unified vocab
        unified_vocab = list(set(self._ref_vocab or []) | set(cur_counts.keys()))
        ref_v = _counts_to_vec(self._ref_word_counts, unified_vocab)
        cur_v = _counts_to_vec(cur_counts, unified_vocab)
        if ref_v.sum() == 0 or cur_v.sum() == 0:
            semantic = 0.0
        else:
            semantic = float(cosine_distance(ref_v, cur_v))
            if math.isnan(semantic):
                semantic = 0.0
        semantic_max = semantic  # single-sample max == value

        # --- lexical drift (Jensen-Shannon divergence) ---
        lexical = float(jensenshannon(ref_v, cur_v) ** 2)  # JSD² like scipy returns sqrt
        # Actually jensenshannon returns sqrt(JSD), so squaring gives JSD.
        if math.isnan(lexical):
            lexical = 0.0
        # top-k overlap
        k = 20
        ref_top = set(self._ref_word_counts.most_common(k))
        cur_top = set(cur_counts.most_common(k))
        ref_top_words = {w for w, _ in ref_top}
        cur_top_words = {w for w, _ in cur_top}
        if ref_top_words:
            top_k_overlap = len(ref_top_words & cur_top_words) / len(ref_top_words)
        else:
            top_k_overlap = 1.0

        # --- structural drift (sentence-length distribution divergence) ---
        cur_sent_lens = _sentence_lengths(generated_texts)
        cur_sent_dist = _length_distribution(cur_sent_lens)
        structural = float(jensenshannon(self._ref_sent_dist, cur_sent_dist) ** 2)
        if math.isnan(structural):
            structural = 0.0
        cur_mean_sent = float(np.mean(cur_sent_lens)) if cur_sent_lens else 0.0
        mean_depth_change = cur_mean_sent - self._ref_mean_sent_len

        # --- distributional drift (KL divergence of token distributions) ---
        distributional = 0.0
        if token_distribution is not None and self._ref_token_dist is not None:
            cur_td = np.asarray(token_distribution, dtype=np.float64)
            # align sizes
            max_len = max(len(self._ref_token_dist), len(cur_td))
            ref_td = np.zeros(max_len, dtype=np.float64)
            ref_td[: len(self._ref_token_dist)] = self._ref_token_dist
            c_td = np.zeros(max_len, dtype=np.float64)
            c_td[: len(cur_td)] = cur_td
            distributional = _kl_divergence(c_td, ref_td)

        # --- composite GDI ---
        gdi = (
            self._w_semantic * semantic
            + self._w_lexical * lexical
            + self._w_structural * structural
            + self._w_distributional * distributional
        )

        alert = gdi > self._alert_threshold

        measurement = GoalDriftMeasurement(
            generation=generation,
            timestamp=datetime.now(timezone.utc).isoformat(),
            semantic_drift=semantic,
            semantic_drift_max=semantic_max,
            lexical_drift=lexical,
            lexical_top_k_overlap=top_k_overlap,
            structural_drift=structural,
            mean_depth_change=mean_depth_change,
            distributional_drift=distributional,
            goal_drift_index=gdi,
            alert=alert,
        )
        self._trajectory.append(measurement)
        return measurement

    def get_trajectory(self) -> list[GoalDriftMeasurement]:
        """Return all measurements so far."""
        return list(self._trajectory)
