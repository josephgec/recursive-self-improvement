"""Human-gated reference updater for GDI."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .collector import ReferenceCollector, ReferenceOutputs
from .store import ReferenceStore


@dataclass
class UpdateProposal:
    """A proposed update to reference outputs."""
    new_outputs: ReferenceOutputs
    reason: str
    proposed_by: str = "system"
    approved: bool = False
    approved_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReferenceUpdater:
    """Human-gated reference output updater.

    Proposes reference updates that must be explicitly approved
    before being applied.
    """

    def __init__(
        self,
        store: ReferenceStore,
        collector: ReferenceCollector,
    ):
        """Initialize updater.

        Args:
            store: Reference store for persistence.
            collector: Reference collector for gathering outputs.
        """
        self.store = store
        self.collector = collector
        self._pending_proposals: List[UpdateProposal] = []

    def propose_update(
        self,
        agent: Any,
        probe_tasks: List[str],
        reason: str = "scheduled_update",
    ) -> UpdateProposal:
        """Propose a reference update by collecting new outputs.

        Args:
            agent: Agent to collect outputs from.
            probe_tasks: Probe tasks to run.
            reason: Reason for the update.

        Returns:
            UpdateProposal awaiting approval.
        """
        new_outputs = self.collector.collect(agent, probe_tasks)
        proposal = UpdateProposal(
            new_outputs=new_outputs,
            reason=reason,
        )
        self._pending_proposals.append(proposal)
        return proposal

    def approve_update(
        self,
        proposal: UpdateProposal,
        approved_by: str = "human",
    ) -> None:
        """Approve and apply a reference update.

        Args:
            proposal: The update proposal to approve.
            approved_by: Identity of the approver.
        """
        proposal.approved = True
        proposal.approved_by = approved_by

        data = {
            "outputs": proposal.new_outputs.outputs,
            "task_outputs": proposal.new_outputs.task_outputs,
            "metadata": {
                **proposal.new_outputs.metadata,
                "reason": proposal.reason,
                "approved_by": approved_by,
            },
        }
        self.store.update(data)

    @property
    def pending_proposals(self) -> List[UpdateProposal]:
        """Get list of pending (unapproved) proposals."""
        return [p for p in self._pending_proposals if not p.approved]
