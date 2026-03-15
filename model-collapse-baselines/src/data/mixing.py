"""Data mixing utilities for blending real and synthetic corpora."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset

logger = logging.getLogger(__name__)


def mix_data(
    real_data: "Dataset",
    synthetic_data: "Dataset",
    alpha: float,
    total_size: int,
    seed: int = 42,
) -> "Dataset":
    """Mix *real_data* and *synthetic_data* at ratio *alpha*.

    ``alpha`` controls the fraction of real data in the final dataset:

    * ``alpha = 1.0`` -- pure real data
    * ``alpha = 0.0`` -- pure synthetic data
    * ``alpha = 0.6`` -- 60 % real, 40 % synthetic

    If a source corpus is smaller than the requested count, rows are sampled
    **with replacement** and a warning is emitted.  Otherwise, sampling is
    **without replacement**.

    A ``"source"`` column (``"real"`` / ``"synthetic"``) is added before
    returning.

    Args:
        real_data: HuggingFace Dataset of real documents.
        synthetic_data: HuggingFace Dataset of synthetic documents.
        alpha: Fraction of real data in [0, 1].
        total_size: Desired number of rows in the output.
        seed: Random seed for reproducibility.

    Returns:
        A shuffled ``Dataset`` of size *total_size* with a ``"source"``
        column.
    """
    from datasets import Dataset, concatenate_datasets

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    n_real = round(alpha * total_size)
    n_synthetic = total_size - n_real

    parts: list[Dataset] = []

    if n_real > 0:
        real_sample = _sample_from(real_data, n_real, seed, label="real")
        real_sample = real_sample.add_column(
            "source", ["real"] * len(real_sample)
        )
        parts.append(real_sample)

    if n_synthetic > 0:
        synth_sample = _sample_from(
            synthetic_data, n_synthetic, seed + 1, label="synthetic"
        )
        synth_sample = synth_sample.add_column(
            "source", ["synthetic"] * len(synth_sample)
        )
        parts.append(synth_sample)

    if len(parts) == 0:
        # Edge case: total_size == 0
        return Dataset.from_dict({"text": [], "source": []})

    mixed = concatenate_datasets(parts)
    mixed = mixed.shuffle(seed=seed)
    return mixed


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _sample_from(
    dataset: "Dataset",
    n: int,
    seed: int,
    label: str,
) -> "Dataset":
    """Sample *n* rows from *dataset*, with or without replacement."""
    import numpy as np

    if n <= len(dataset):
        # Without replacement.
        return dataset.shuffle(seed=seed).select(range(n))

    # With replacement -- warn the caller.
    warnings.warn(
        f"Requested {n} {label} samples but source has only {len(dataset)}; "
        f"sampling with replacement.",
        stacklevel=3,
    )
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=n, replace=True).tolist()
    return dataset.select(indices)
