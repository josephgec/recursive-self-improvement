"""Drift tracking decorator for automated GDI monitoring."""

import functools
from typing import Any, Callable, List, Optional

from ..composite.gdi import GoalDriftIndex, GDIResult
from ..reference.store import ReferenceStore


class GDIDriftError(Exception):
    """Raised when GDI reaches red alert level."""

    def __init__(self, result: GDIResult, message: str = ""):
        self.result = result
        if not message:
            message = (
                f"GDI drift alert: {result.alert_level} "
                f"(score={result.composite_score:.3f})"
            )
        super().__init__(message)


def track_drift(
    gdi: GoalDriftIndex,
    probe_tasks: List[str],
    ref_store: ReferenceStore,
    agent_extractor: Optional[Callable[..., Any]] = None,
    raise_on_red: bool = True,
) -> Callable:
    """Decorator that auto-computes GDI after decorated function.

    Args:
        gdi: GoalDriftIndex instance.
        probe_tasks: List of probe task strings.
        ref_store: Reference store.
        agent_extractor: Function to extract agent from decorated function's args.
                        If None, first argument is used as agent.
        raise_on_red: If True, raise GDIDriftError on red alert.

    Returns:
        Decorator function.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Run the original function
            result = func(*args, **kwargs)

            # Extract agent
            if agent_extractor:
                agent = agent_extractor(*args, **kwargs)
            elif args:
                agent = args[0]
            else:
                return result

            # Run probe tasks
            current_outputs = [agent.run(task) for task in probe_tasks]

            # Load reference
            ref_data = ref_store.load()
            reference_outputs = ref_data.get("outputs", [])

            # Compute GDI
            gdi_result = gdi.compute(current_outputs, reference_outputs)

            # Store result on the function
            if not hasattr(wrapper, '_gdi_results'):
                wrapper._gdi_results = []
            wrapper._gdi_results.append(gdi_result)

            # Raise on red
            if raise_on_red and gdi_result.alert_level == "red":
                raise GDIDriftError(gdi_result)

            return result

        wrapper._gdi_results = []
        return wrapper

    return decorator
