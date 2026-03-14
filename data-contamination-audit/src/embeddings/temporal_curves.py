"""Temporal contamination curves: similarity evolution over time bins.

Computes per-bin similarity metrics across a temporally-binned corpus and
identifies the inflection point where similarity growth accelerates — a
signal that training-data contamination may have increased.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.common_crawl import Document
from src.embeddings.encoder import DocumentEncoder
from src.embeddings.similarity import (
    corpus_mean_similarity,
    cross_corpus_similarity,
    pairwise_cosine_similarity,
)

logger = logging.getLogger(__name__)


def compute_temporal_curve(
    corpus: dict[str, list[Document]],
    encoder: DocumentEncoder,
    reference_bin: str | None = None,
) -> pd.DataFrame:
    """Build a DataFrame of similarity metrics for each temporal bin.

    Parameters
    ----------
    corpus:
        Mapping from bin label (e.g. ``"2013"``, ``"2020"``) to the
        documents that fall in that bin.
    encoder:
        A :class:`DocumentEncoder` used to embed each bin's documents.
    reference_bin:
        The bin against which cross-corpus similarity is measured.  If
        ``None``, the lexicographically smallest bin is used (typically
        the earliest year).

    Returns
    -------
    pd.DataFrame
        Sorted by ``bin`` with columns:

        - **bin** — time-bin label
        - **mean_similarity** — intra-bin mean pairwise cosine similarity
        - **cross_similarity_to_reference** — mean cosine similarity
          between this bin and the reference bin
        - **n_documents** — number of documents in the bin
        - **similarity_p25** — 25th-percentile of pairwise similarities
        - **similarity_p75** — 75th-percentile of pairwise similarities
    """
    if not corpus:
        logger.warning("compute_temporal_curve called with an empty corpus")
        return pd.DataFrame(
            columns=[
                "bin",
                "mean_similarity",
                "cross_similarity_to_reference",
                "n_documents",
                "similarity_p25",
                "similarity_p75",
            ]
        )

    sorted_bins = sorted(corpus.keys())
    if reference_bin is None:
        reference_bin = sorted_bins[0]
        logger.info("Using %r as the reference bin", reference_bin)

    if reference_bin not in corpus:
        raise ValueError(
            f"reference_bin {reference_bin!r} not found in corpus keys: "
            f"{sorted_bins}"
        )

    # --- Embed every bin ------------------------------------------------
    bin_embeddings: dict[str, np.ndarray] = {}
    for bin_label in sorted_bins:
        docs = corpus[bin_label]
        logger.info(
            "Encoding bin %r (%d documents)", bin_label, len(docs)
        )
        bin_embeddings[bin_label] = encoder.encode(docs)

    ref_emb = bin_embeddings[reference_bin]

    # --- Compute per-bin metrics ----------------------------------------
    rows: list[dict] = []
    for bin_label in sorted_bins:
        emb = bin_embeddings[bin_label]
        n_docs = emb.shape[0]

        mean_sim = corpus_mean_similarity(emb)

        cross_sim = cross_corpus_similarity(emb, ref_emb)

        # Percentiles
        if n_docs >= 2:
            sims = pairwise_cosine_similarity(emb)
            p25 = float(np.percentile(sims, 25))
            p75 = float(np.percentile(sims, 75))
        else:
            p25 = mean_sim
            p75 = mean_sim

        rows.append(
            {
                "bin": bin_label,
                "mean_similarity": mean_sim,
                "cross_similarity_to_reference": cross_sim,
                "n_documents": n_docs,
                "similarity_p25": p25,
                "similarity_p75": p75,
            }
        )
        logger.info(
            "Bin %s: mean_sim=%.4f  cross_sim=%.4f  n=%d",
            bin_label,
            mean_sim,
            cross_sim,
            n_docs,
        )

    df = pd.DataFrame(rows).sort_values("bin").reset_index(drop=True)
    logger.info(
        "Temporal curve complete: %d bins, %d total documents",
        len(df),
        int(df["n_documents"].sum()),
    )
    return df


def detect_inflection_point(curve: pd.DataFrame) -> str:
    """Identify the bin where the similarity increase accelerates most sharply.

    The algorithm:

    1. Smooth the ``mean_similarity`` column with a 3-bin moving average.
    2. Compute the first derivative (finite differences).
    3. Compute the second derivative (acceleration).
    4. Return the bin label whose second derivative is the largest — i.e.
       the point of greatest *acceleration* in similarity growth.

    Parameters
    ----------
    curve:
        A DataFrame as returned by :func:`compute_temporal_curve`,
        expected to be sorted by ``bin``.

    Returns
    -------
    str
        The bin label at the inflection point.

    Raises
    ------
    ValueError
        If the curve has fewer than 3 rows (insufficient for a second
        derivative).
    """
    if len(curve) < 3:
        raise ValueError(
            f"Need at least 3 bins to compute an inflection point, "
            f"got {len(curve)}"
        )

    similarity = curve["mean_similarity"].values.astype(np.float64)

    # 3-bin moving average (centred).  We use min_periods=1 at edges so the
    # output length matches the input, but only interior points will have
    # meaningful second derivatives.
    smoothed = (
        pd.Series(similarity)
        .rolling(window=3, center=True, min_periods=1)
        .mean()
        .values
    )

    # First derivative (forward differences, length n-1).
    first_deriv = np.diff(smoothed)

    # Second derivative (length n-2).
    second_deriv = np.diff(first_deriv)

    # The i-th element of second_deriv corresponds to the (i+1)-th bin in
    # the original curve (the centre of the three-point stencil).
    max_idx = int(np.argmax(second_deriv))
    inflection_bin_idx = max_idx + 1  # map back to curve index

    inflection_bin = str(curve.iloc[inflection_bin_idx]["bin"])
    logger.info(
        "Inflection point detected at bin %r (second-derivative index %d, "
        "value %.6f)",
        inflection_bin,
        max_idx,
        second_deriv[max_idx],
    )
    return inflection_bin
