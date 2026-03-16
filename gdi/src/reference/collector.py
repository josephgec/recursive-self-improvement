"""Reference output collector for GDI."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ReferenceOutputs:
    """Collection of reference outputs from probe tasks."""
    outputs: List[str]
    task_outputs: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReferenceCollector:
    """Collects reference outputs by running probe tasks on an agent.

    Each probe task is run multiple times to capture output variability.
    """

    def __init__(self, samples_per_task: int = 3):
        """Initialize collector.

        Args:
            samples_per_task: Number of times to run each probe task.
        """
        self.samples_per_task = samples_per_task

    def collect(
        self, agent: Any, probe_tasks: List[str]
    ) -> ReferenceOutputs:
        """Collect reference outputs from agent.

        Args:
            agent: Agent with a run(task) -> str method.
            probe_tasks: List of probe task strings.

        Returns:
            ReferenceOutputs with all collected outputs.
        """
        all_outputs: List[str] = []
        task_outputs: Dict[str, List[str]] = {}

        for task in probe_tasks:
            task_results = []
            for _ in range(self.samples_per_task):
                output = agent.run(task)
                task_results.append(output)
                all_outputs.append(output)
            task_outputs[task] = task_results

        return ReferenceOutputs(
            outputs=all_outputs,
            task_outputs=task_outputs,
            metadata={
                "num_tasks": len(probe_tasks),
                "samples_per_task": self.samples_per_task,
                "total_outputs": len(all_outputs),
            },
        )
