"""Snapshot management for REPL state persistence."""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.memory.serializer import REPLSerializer
from src.interface.errors import SerializationError


@dataclass
class REPLSnapshot:
    """A point-in-time snapshot of a REPL's state.

    Attributes:
        snapshot_id: Unique identifier for the snapshot.
        timestamp: Unix timestamp when the snapshot was taken.
        variables: Serialized variable data.
        metadata: Additional metadata about the snapshot.
    """

    snapshot_id: str = ""
    timestamp: float = 0.0
    variables: Dict[str, tuple] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SnapshotManager:
    """Manages snapshots of REPL state for save/restore.

    Serializes all variables and stores them for later restoration.
    """

    def __init__(self, serializer: Optional[REPLSerializer] = None):
        self._serializer = serializer or REPLSerializer()
        self._snapshots: Dict[str, REPLSnapshot] = {}

    def take(self, namespace: Dict[str, Any], metadata: Dict[str, Any] = None) -> str:
        """Take a snapshot of the current namespace.

        Args:
            namespace: The variable namespace to snapshot.
            metadata: Optional metadata to attach.

        Returns:
            Snapshot identifier string.
        """
        snapshot_id = str(uuid.uuid4())[:8]
        serialized: Dict[str, tuple] = {}

        for name, value in namespace.items():
            if name.startswith("__") and name.endswith("__"):
                continue
            try:
                tag, data = self._serializer.serialize(value, name)
                serialized[name] = (tag, data)
            except SerializationError:
                # Skip unserializable variables
                pass

        snapshot = REPLSnapshot(
            snapshot_id=snapshot_id,
            timestamp=time.time(),
            variables=serialized,
            metadata=metadata or {},
        )
        self._snapshots[snapshot_id] = snapshot
        return snapshot_id

    def restore(self, snapshot_id: str) -> Dict[str, Any]:
        """Restore a snapshot.

        Args:
            snapshot_id: Identifier of the snapshot to restore.

        Returns:
            Namespace dictionary with deserialized variables.

        Raises:
            KeyError: If the snapshot does not exist.
        """
        if snapshot_id not in self._snapshots:
            raise KeyError(f"Snapshot '{snapshot_id}' not found")

        snapshot = self._snapshots[snapshot_id]
        namespace: Dict[str, Any] = {}

        for name, (tag, data) in snapshot.variables.items():
            try:
                namespace[name] = self._serializer.deserialize(tag, data, name)
            except SerializationError:
                pass

        return namespace

    def size_bytes(self, snapshot_id: str) -> int:
        """Get the size of a snapshot in bytes.

        Args:
            snapshot_id: Identifier of the snapshot.

        Returns:
            Total bytes of serialized data.

        Raises:
            KeyError: If the snapshot does not exist.
        """
        if snapshot_id not in self._snapshots:
            raise KeyError(f"Snapshot '{snapshot_id}' not found")

        snapshot = self._snapshots[snapshot_id]
        return sum(len(data) for _, data in snapshot.variables.values())

    def list_snapshots(self) -> list:
        """List all snapshot IDs.

        Returns:
            List of snapshot identifier strings.
        """
        return list(self._snapshots.keys())

    def delete(self, snapshot_id: str) -> None:
        """Delete a snapshot.

        Args:
            snapshot_id: Identifier of the snapshot to delete.

        Raises:
            KeyError: If the snapshot does not exist.
        """
        if snapshot_id not in self._snapshots:
            raise KeyError(f"Snapshot '{snapshot_id}' not found")
        del self._snapshots[snapshot_id]
