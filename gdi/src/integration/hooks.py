"""GDI integration hooks for automated drift checking."""

from typing import Any, Callable, Dict, List, Optional

from ..composite.gdi import GoalDriftIndex, GDIResult
from ..reference.store import ReferenceStore


class GDIHooks:
    """Hooks for integrating GDI checks into agent workflows.

    Provides lifecycle hooks that run probe tasks and compute GDI
    after various agent events.
    """

    def __init__(
        self,
        gdi: GoalDriftIndex,
        probe_tasks: List[str],
        ref_store: ReferenceStore,
        on_result: Optional[Callable[[GDIResult], None]] = None,
    ):
        """Initialize hooks.

        Args:
            gdi: GoalDriftIndex instance.
            probe_tasks: List of probe task strings.
            ref_store: Reference store for loading baseline outputs.
            on_result: Optional callback invoked with each GDI result.
        """
        self.gdi = gdi
        self.probe_tasks = probe_tasks
        self.ref_store = ref_store
        self.on_result = on_result
        self._results: List[GDIResult] = []

    def _run_check(self, agent: Any, event: str) -> GDIResult:
        """Run probe tasks and compute GDI.

        Args:
            agent: Agent with run(task) method.
            event: Event that triggered this check.

        Returns:
            GDIResult.
        """
        current_outputs = [agent.run(task) for task in self.probe_tasks]

        ref_data = self.ref_store.load()
        reference_outputs = ref_data.get("outputs", [])

        result = self.gdi.compute(current_outputs, reference_outputs)
        result.metadata["trigger_event"] = event
        self._results.append(result)

        if self.on_result:
            self.on_result(result)

        return result

    def after_modification(self, agent: Any) -> GDIResult:
        """Check GDI after agent code modification.

        Args:
            agent: The modified agent.

        Returns:
            GDIResult.
        """
        return self._run_check(agent, "after_modification")

    def after_training(self, agent: Any) -> GDIResult:
        """Check GDI after agent training update.

        Args:
            agent: The trained agent.

        Returns:
            GDIResult.
        """
        return self._run_check(agent, "after_training")

    def after_library_update(self, agent: Any) -> GDIResult:
        """Check GDI after library/dependency update.

        Args:
            agent: The agent with updated libraries.

        Returns:
            GDIResult.
        """
        return self._run_check(agent, "after_library_update")

    def periodic_check(self, agent: Any) -> GDIResult:
        """Periodic GDI check.

        Args:
            agent: The agent to check.

        Returns:
            GDIResult.
        """
        return self._run_check(agent, "periodic_check")

    @property
    def results(self) -> List[GDIResult]:
        """Get all recorded results."""
        return self._results
