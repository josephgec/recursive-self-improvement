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


def compute_per_source_curves(
    corpus: dict[str, list[Document]],
    encoder: DocumentEncoder,
    reference_bin: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute temporal curves for each document source independently.

    Parameters
    ----------
    corpus:
        Mapping from bin label to documents (as used by
        :func:`compute_temporal_curve`).
    encoder:
        A :class:`DocumentEncoder` for embedding.
    reference_bin:
        The bin against which cross-corpus similarity is measured.  If
        ``None``, the lexicographically smallest bin is used.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from source name (e.g. ``"wikipedia"``) to a curve
        DataFrame.  Bins with fewer than 5 documents for a given source
        are skipped for that source.
    """
    # Split corpus by source within each bin
    from collections import defaultdict

    source_corpora: dict[str, dict[str, list[Document]]] = defaultdict(dict)
    for bin_label, docs in corpus.items():
        by_source: dict[str, list[Document]] = defaultdict(list)
        for doc in docs:
            by_source[doc.source].append(doc)
        for source_name, source_docs in by_source.items():
            if len(source_docs) >= 5:
                source_corpora[source_name][bin_label] = source_docs
            else:
                logger.info(
                    "Skipping source %r in bin %r: only %d documents (< 5)",
                    source_name,
                    bin_label,
                    len(source_docs),
                )

    result: dict[str, pd.DataFrame] = {}
    for source_name, sub_corpus in sorted(source_corpora.items()):
        logger.info(
            "Computing temporal curve for source %r (%d bins)",
            source_name,
            len(sub_corpus),
        )
        curve = compute_temporal_curve(sub_corpus, encoder, reference_bin=reference_bin)
        result[source_name] = curve

    return result


def detect_inflection_with_ci(
    curve: pd.DataFrame,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> dict:
    """Identify the inflection point with a bootstrap confidence interval.

    Parameters
    ----------
    curve:
        A DataFrame as returned by :func:`compute_temporal_curve`,
        expected to be sorted by ``bin``.
    n_bootstrap:
        Number of bootstrap iterations.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys:

        - **bin_label** — the modal inflection bin label
        - **second_derivative** — second derivative value at the inflection
        - **ci_lower** — bin label at the 5th percentile of bootstrap
          inflection points
        - **ci_upper** — bin label at the 95th percentile of bootstrap
          inflection points
        - **confidence** — fraction of bootstrap samples that agree with the
          modal inflection point

    Raises
    ------
    ValueError
        If the curve has fewer than 3 rows.
    """
    if len(curve) < 3:
        raise ValueError(
            f"Need at least 3 bins for inflection detection, got {len(curve)}"
        )

    similarity = curve["mean_similarity"].values.astype(np.float64)
    bins = curve["bin"].astype(str).values

    # Compute the canonical inflection point and its second-derivative value
    smoothed = (
        pd.Series(similarity)
        .rolling(window=3, center=True, min_periods=1)
        .mean()
        .values
    )
    first_deriv = np.diff(smoothed)
    second_deriv = np.diff(first_deriv)
    max_idx = int(np.argmax(second_deriv))
    inflection_bin_idx = max_idx + 1
    inflection_second_deriv = float(second_deriv[max_idx])

    # Bootstrap: add noise, recompute inflection point each time
    rng = np.random.default_rng(seed)
    sim_range = similarity.max() - similarity.min()
    noise_std = 0.01 * sim_range if sim_range > 0 else 0.001

    bootstrap_indices: list[int] = []
    for _ in range(n_bootstrap):
        noisy = similarity + rng.normal(0.0, noise_std, size=len(similarity))
        sm = (
            pd.Series(noisy)
            .rolling(window=3, center=True, min_periods=1)
            .mean()
            .values
        )
        fd = np.diff(sm)
        sd = np.diff(fd)
        bi = int(np.argmax(sd)) + 1
        bootstrap_indices.append(bi)

    bootstrap_indices_arr = np.array(bootstrap_indices)

    # Modal inflection index
    from collections import Counter

    counts = Counter(bootstrap_indices)
    modal_idx = counts.most_common(1)[0][0]
    confidence = counts[modal_idx] / n_bootstrap

    # CI bounds: 5th and 95th percentile of bootstrap inflection indices
    ci_lower_idx = int(np.percentile(bootstrap_indices_arr, 5))
    ci_upper_idx = int(np.percentile(bootstrap_indices_arr, 95))

    # Clamp indices to valid range
    ci_lower_idx = max(0, min(ci_lower_idx, len(bins) - 1))
    ci_upper_idx = max(0, min(ci_upper_idx, len(bins) - 1))
    modal_idx = max(0, min(modal_idx, len(bins) - 1))

    result = {
        "bin_label": str(bins[modal_idx]),
        "second_derivative": inflection_second_deriv,
        "ci_lower": str(bins[ci_lower_idx]),
        "ci_upper": str(bins[ci_upper_idx]),
        "confidence": confidence,
    }

    logger.info(
        "Inflection with CI: bin=%s, CI=[%s, %s], confidence=%.1f%%",
        result["bin_label"],
        result["ci_lower"],
        result["ci_upper"],
        result["confidence"] * 100,
    )
    return result
