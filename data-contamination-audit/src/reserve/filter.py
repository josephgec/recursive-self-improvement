"""Reserve filtering: select documents with high human-authorship probability.

Uses a contamination classifier (either raw or calibrated) to score each
document and retains only those whose predicted probability of being
human-authored meets or exceeds a configurable threshold.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.data.common_crawl import Document

logger = logging.getLogger(__name__)


def filter_to_reserve(
    documents: list[Document],
    classifier,
    feature_matrix: pd.DataFrame,
    threshold: float = 0.90,
) -> list[Document]:
    """Keep only documents whose predicted human-authorship probability is
    at or above *threshold*.

    Parameters
    ----------
    documents:
        Source documents to filter.
    classifier:
        Any object exposing a ``predict_proba(features) -> np.ndarray``
        method that returns shape ``(n, 2)`` with columns
        ``[p_human, p_synthetic]``.  Both
        :class:`~src.classifier.model.ContaminationClassifier` and
        :class:`~src.classifier.calibration.CalibratedClassifier` satisfy
        this interface.
    feature_matrix:
        Feature matrix aligned row-wise with *documents*.
    threshold:
        Minimum ``p_human`` required to enter the reserve.

    Returns
    -------
    list[Document]
        Filtered documents, each with ``metadata["authenticity_score"]``
        set to its ``p_human`` value.
    """
    if len(documents) != len(feature_matrix):
        raise ValueError(
            f"documents ({len(documents)}) and feature_matrix "
            f"({len(feature_matrix)}) must have the same length"
        )

    probas = classifier.predict_proba(feature_matrix)  # (n, 2)

    reserve: list[Document] = []
    for doc, p_human in zip(documents, probas[:, 0]):
        p_human_float = float(p_human)
        doc.metadata["authenticity_score"] = p_human_float
        if p_human_float >= threshold:
            reserve.append(doc)

    logger.info(
        "Reserve filter: %d / %d documents pass threshold %.2f",
        len(reserve),
        len(documents),
        threshold,
    )
    return reserve


def compute_alpha_t(full_corpus_size: int, reserve_size: int) -> float:
    """Compute alpha_t, the proportion of authentic data in the corpus.

    Parameters
    ----------
    full_corpus_size:
        Total number of documents in the audited corpus.
    reserve_size:
        Number of documents that passed the reserve filter.

    Returns
    -------
    float
        ``reserve_size / full_corpus_size``.  Returns 0.0 when the corpus
        is empty.
    """
    if full_corpus_size == 0:
        return 0.0
    return reserve_size / full_corpus_size
