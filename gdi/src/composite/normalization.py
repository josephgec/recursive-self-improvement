"""Per-signal normalization for GDI composite scoring."""

from typing import Dict, Optional


# Per-signal calibration parameters: (scale_factor, cap)
_DEFAULT_CALIBRATION = {
    "semantic": (2.0, 1.0),      # raw / 0.5 capped at 1.0
    "lexical": (2.0, 1.0),       # raw / 0.5 capped at 1.0
    "structural": (2.0, 1.0),    # raw / 0.5 capped at 1.0
    "distributional": (2.0, 1.0),  # raw / 0.5 capped at 1.0
}


def normalize_signal(
    raw: float,
    signal_type: str,
    calibration: Optional[Dict[str, tuple]] = None,
) -> float:
    """Normalize a raw signal score using per-signal calibration.

    Args:
        raw: Raw signal score.
        signal_type: Signal type identifier (semantic, lexical, etc.).
        calibration: Optional per-signal calibration parameters.
                     Dict mapping signal_type to (scale_factor, cap).

    Returns:
        Normalized score in [0, cap].
    """
    cal = calibration or _DEFAULT_CALIBRATION

    if signal_type in cal:
        scale, cap = cal[signal_type]
    else:
        scale, cap = (2.0, 1.0)

    return min(cap, max(0.0, raw * scale))
