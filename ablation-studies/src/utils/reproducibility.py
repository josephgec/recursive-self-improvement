"""Reproducibility utilities: seed management, config hashing."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Dict


def set_global_seed(seed: int) -> None:
    """Set the global random seed for reproducibility.

    Sets Python's built-in random module seed.
    """
    random.seed(seed)


def get_config_hash(config: Dict[str, Any]) -> str:
    """Compute a deterministic hash of a configuration dictionary.

    Useful for caching and verifying that configs haven't changed.

    Args:
        config: Configuration dictionary.

    Returns:
        Hex digest of the config hash.
    """
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def verify_seed_consistency(seed: int, n_samples: int = 10) -> bool:
    """Verify that seed produces consistent random sequences.

    Generates two sequences with the same seed and checks they match.
    """
    rng1 = random.Random(seed)
    seq1 = [rng1.random() for _ in range(n_samples)]

    rng2 = random.Random(seed)
    seq2 = [rng2.random() for _ in range(n_samples)]

    return seq1 == seq2


def generate_run_id(suite_name: str, seed: int, repetitions: int) -> str:
    """Generate a unique run identifier."""
    key = f"{suite_name}:{seed}:{repetitions}"
    return hashlib.md5(key.encode()).hexdigest()[:12]
