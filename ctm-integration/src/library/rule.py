"""Verified rule dataclass for the rule library."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VerifiedRule:
    """A verified rule stored in the library.

    Attributes:
        rule_id: Unique identifier for this rule.
        domain: Problem domain (e.g., "math", "string", "general").
        description: Human-readable description of what the rule does.
        source_code: Python source code implementing the rule.
        accuracy: Verified accuracy on test examples (0.0 to 1.0).
        bdm_score: BDM complexity score of the source code.
        mdl_score: Minimum Description Length score.
        tags: List of tags for categorization and retrieval.
        examples_count: Number of examples the rule was verified against.
        created_at: Timestamp of rule creation.
        generation: Which synthesis iteration produced this rule.
        metadata: Additional metadata.
    """

    rule_id: str
    domain: str
    description: str
    source_code: str
    accuracy: float = 0.0
    bdm_score: float = 0.0
    mdl_score: float = 0.0
    tags: List[str] = field(default_factory=list)
    examples_count: int = 0
    created_at: float = field(default_factory=time.time)
    generation: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def code_hash(self) -> str:
        """SHA-256 hash of the source code for deduplication."""
        return hashlib.sha256(self.source_code.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "rule_id": self.rule_id,
            "domain": self.domain,
            "description": self.description,
            "source_code": self.source_code,
            "accuracy": self.accuracy,
            "bdm_score": self.bdm_score,
            "mdl_score": self.mdl_score,
            "tags": self.tags,
            "examples_count": self.examples_count,
            "created_at": self.created_at,
            "generation": self.generation,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerifiedRule":
        """Create from dictionary."""
        return cls(
            rule_id=data["rule_id"],
            domain=data.get("domain", "general"),
            description=data.get("description", ""),
            source_code=data["source_code"],
            accuracy=data.get("accuracy", 0.0),
            bdm_score=data.get("bdm_score", 0.0),
            mdl_score=data.get("mdl_score", 0.0),
            tags=data.get("tags", []),
            examples_count=data.get("examples_count", 0),
            created_at=data.get("created_at", time.time()),
            generation=data.get("generation", 0),
            metadata=data.get("metadata", {}),
        )
