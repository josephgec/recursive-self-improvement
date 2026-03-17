"""Artifact registry for tracking evidence artifacts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Artifact:
    """A registered evidence artifact."""

    artifact_id: str
    name: str
    artifact_type: str
    data: Any
    registered_at: str = ""
    sha256: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.registered_at:
            self.registered_at = datetime.utcnow().isoformat()
        if not self.sha256:
            self.sha256 = hashlib.sha256(
                str(self.data).encode("utf-8")
            ).hexdigest()


class ArtifactRegistry:
    """Registry for managing evidence artifacts."""

    def __init__(self) -> None:
        self._artifacts: Dict[str, Artifact] = {}

    def register(
        self,
        artifact_id: str,
        name: str,
        artifact_type: str,
        data: Any,
        metadata: Dict[str, Any] | None = None,
    ) -> Artifact:
        """Register a new artifact.

        Raises ValueError if artifact_id already exists.
        """
        if artifact_id in self._artifacts:
            raise ValueError(
                f"Artifact '{artifact_id}' already registered"
            )

        artifact = Artifact(
            artifact_id=artifact_id,
            name=name,
            artifact_type=artifact_type,
            data=data,
            metadata=metadata or {},
        )
        self._artifacts[artifact_id] = artifact
        return artifact

    def get(self, artifact_id: str) -> Optional[Artifact]:
        """Get an artifact by ID. Returns None if not found."""
        return self._artifacts.get(artifact_id)

    def list_all(self) -> List[Artifact]:
        """List all registered artifacts."""
        return list(self._artifacts.values())

    def list_by_type(self, artifact_type: str) -> List[Artifact]:
        """List artifacts filtered by type."""
        return [
            a for a in self._artifacts.values()
            if a.artifact_type == artifact_type
        ]

    def remove(self, artifact_id: str) -> bool:
        """Remove an artifact. Returns True if found and removed."""
        if artifact_id in self._artifacts:
            del self._artifacts[artifact_id]
            return True
        return False

    def verify_integrity(self, artifact_id: str) -> bool:
        """Verify that an artifact's data hasn't changed since registration."""
        artifact = self._artifacts.get(artifact_id)
        if artifact is None:
            return False

        current_hash = hashlib.sha256(
            str(artifact.data).encode("utf-8")
        ).hexdigest()
        return current_hash == artifact.sha256

    def count(self) -> int:
        """Return number of registered artifacts."""
        return len(self._artifacts)
