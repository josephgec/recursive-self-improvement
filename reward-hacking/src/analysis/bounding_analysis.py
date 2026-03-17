from __future__ import annotations

"""Analysis utilities for reward bounding results."""

import numpy as np

from ..bounding.process_reward import ShapedReward


def analyze_bounding(history: list[ShapedReward]) -> dict:
    """Analyze reward bounding effectiveness.

    Args:
        history: List of ShapedReward results.

    Returns:
        Dictionary with analysis metrics.
    """
    if not history:
        return {"status": "no_data"}

    raw_values = [sr.raw for sr in history]
    final_values = [sr.final for sr in history]
    clipped_count = sum(1 for sr in history if sr.was_clipped)
    bounded_count = sum(1 for sr in history if sr.was_delta_bounded)

    raw_arr = np.array(raw_values)
    final_arr = np.array(final_values)

    # Compression ratio
    raw_range = float(np.ptp(raw_arr))
    final_range = float(np.ptp(final_arr))
    compression = 1.0 - (final_range / raw_range) if raw_range > 0 else 0.0

    return {
        "num_rewards": len(history),
        "raw": {
            "mean": float(np.mean(raw_arr)),
            "std": float(np.std(raw_arr)),
            "min": float(np.min(raw_arr)),
            "max": float(np.max(raw_arr)),
        },
        "shaped": {
            "mean": float(np.mean(final_arr)),
            "std": float(np.std(final_arr)),
            "min": float(np.min(final_arr)),
            "max": float(np.max(final_arr)),
        },
        "clipping": {
            "count": clipped_count,
            "fraction": clipped_count / len(history),
        },
        "delta_bounding": {
            "count": bounded_count,
            "fraction": bounded_count / len(history),
        },
        "compression_ratio": compression,
        "effective": compression > 0 or (clipped_count + bounded_count) > 0,
    }
