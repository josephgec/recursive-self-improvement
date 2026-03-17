"""Decorator for monitoring model internals during function execution."""

import functools
from typing import Any, Callable, Optional

from src.probing.probe_set import ProbeSet
from src.probing.extractor import ActivationExtractor, ActivationSnapshot
from src.probing.diff import ActivationDiff


def monitor_internals(model: Any = None,
                      probe_set: Optional[ProbeSet] = None,
                      on_diff: Optional[Callable] = None):
    """Decorator that monitors model internals before/after function execution.

    Usage:
        @monitor_internals(model=my_model)
        def train_step(model, data):
            ...

    Args:
        model: Model to monitor
        probe_set: ProbeSet for activation extraction
        on_diff: Callback called with (before_snapshot, after_snapshot, diff_result)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            m = model
            if m is None and args:
                m = args[0]
            if m is None:
                return func(*args, **kwargs)

            ps = probe_set or ProbeSet()
            extractor = ActivationExtractor(m)
            differ = ActivationDiff()

            # Before
            probes = ps.get_all()
            before_snap = extractor.extract(probes)

            # Execute
            result = func(*args, **kwargs)

            # After
            after_snap = extractor.extract(probes)

            # Diff
            diff_result = differ.compute(before_snap, after_snap)

            # Callback
            if on_diff is not None:
                on_diff(before_snap, after_snap, diff_result)

            # Attach to result if dict
            if isinstance(result, dict):
                result["_interp_diff"] = diff_result.to_dict()

            return result

        wrapper._monitored = True
        return wrapper
    return decorator
