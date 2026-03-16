"""Bridge connecting GDI to the Phase 3.1 orchestrator."""

from typing import Any, Callable, Dict, List, Optional

from ..composite.gdi import GoalDriftIndex, GDIResult
from ..reference.store import ReferenceStore
from ..alerting.alert_manager import AlertManager
from .phase_adapters import PhaseAdapter


class PipelineBridge:
    """Connects GDI monitoring to the Phase 3.1 orchestrator.

    Provides a unified interface for the orchestrator to:
    - Check drift before/after pipeline stages
    - Get go/no-go decisions
    - Report drift metrics
    """

    def __init__(
        self,
        gdi: GoalDriftIndex,
        ref_store: ReferenceStore,
        probe_tasks: List[str],
        alert_manager: Optional[AlertManager] = None,
    ):
        """Initialize bridge.

        Args:
            gdi: GoalDriftIndex instance.
            ref_store: Reference store.
            probe_tasks: Probe tasks for drift checking.
            alert_manager: Optional alert manager for notifications.
        """
        self.gdi = gdi
        self.ref_store = ref_store
        self.probe_tasks = probe_tasks
        self.alert_manager = alert_manager
        self._iteration = 0
        self._results: List[GDIResult] = []

    def check_drift(self, agent: Any) -> GDIResult:
        """Run a drift check and return the result.

        Args:
            agent: Agent with run(task) method.

        Returns:
            GDIResult.
        """
        current_outputs = [agent.run(task) for task in self.probe_tasks]

        ref_data = self.ref_store.load()
        reference_outputs = ref_data.get("outputs", [])

        result = self.gdi.compute(current_outputs, reference_outputs)
        self._iteration += 1
        result.metadata["iteration"] = self._iteration
        self._results.append(result)

        if self.alert_manager:
            self.alert_manager.process(result, self._iteration)

        return result

    def is_safe_to_proceed(self, agent: Any) -> bool:
        """Check if it's safe to proceed with the next pipeline stage.

        Returns True if GDI is green or yellow.
        """
        result = self.check_drift(agent)
        return result.alert_level in ("green", "yellow")

    def get_metrics(self) -> Dict[str, Any]:
        """Get current GDI metrics for the orchestrator.

        Returns:
            Dictionary of metrics.
        """
        if not self._results:
            return {"status": "no_data"}

        latest = self._results[-1]
        return {
            "composite_score": latest.composite_score,
            "alert_level": latest.alert_level,
            "trend": latest.trend,
            "iteration": self._iteration,
            "semantic": latest.semantic_score,
            "lexical": latest.lexical_score,
            "structural": latest.structural_score,
            "distributional": latest.distributional_score,
        }

    @property
    def results(self) -> List[GDIResult]:
        """Get all recorded results."""
        return self._results
