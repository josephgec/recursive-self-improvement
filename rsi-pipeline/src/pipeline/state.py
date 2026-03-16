"""Pipeline state management with serialization."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class AgentCodeSnapshot:
    """Snapshot of the agent's modifiable code."""
    code: str = ""
    version: int = 0
    target: str = "default"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentCodeSnapshot":
        return cls(**d)


@dataclass
class PerformanceRecord:
    """Record of agent performance metrics."""
    accuracy: float = 0.0
    test_pass_rate: float = 0.0
    complexity_score: float = 0.0
    entropy: float = 1.0
    iteration: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PerformanceRecord":
        return cls(**d)


@dataclass
class SafetyStatus:
    """Current safety status of the pipeline."""
    gdi_score: float = 0.0
    car_score: float = 1.0
    constraints_satisfied: bool = True
    consecutive_rollbacks: int = 0
    emergency_stop: bool = False
    violations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SafetyStatus":
        return cls(**d)


@dataclass
class PipelineState:
    """Complete state of the RSI pipeline."""
    iteration: int = 0
    status: str = "initialized"  # initialized, running, paused, stopped, emergency
    agent_code: AgentCodeSnapshot = field(default_factory=AgentCodeSnapshot)
    original_code: AgentCodeSnapshot = field(default_factory=AgentCodeSnapshot)
    performance: PerformanceRecord = field(default_factory=PerformanceRecord)
    safety: SafetyStatus = field(default_factory=SafetyStatus)
    performance_history: List[PerformanceRecord] = field(default_factory=list)
    modification_history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def state_id(self) -> str:
        """Compute a hash-based state identifier."""
        content = json.dumps({
            "iteration": self.iteration,
            "code": self.agent_code.code,
            "version": self.agent_code.version,
            "accuracy": self.performance.accuracy,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_json(self) -> str:
        """Serialize state to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "iteration": self.iteration,
            "status": self.status,
            "agent_code": self.agent_code.to_dict(),
            "original_code": self.original_code.to_dict(),
            "performance": self.performance.to_dict(),
            "safety": self.safety.to_dict(),
            "performance_history": [p.to_dict() for p in self.performance_history],
            "modification_history": self.modification_history,
            "metadata": self.metadata,
        }

    @classmethod
    def from_json(cls, json_str: str) -> "PipelineState":
        """Deserialize state from JSON string."""
        d = json.loads(json_str)
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineState":
        """Create state from dictionary."""
        return cls(
            iteration=d.get("iteration", 0),
            status=d.get("status", "initialized"),
            agent_code=AgentCodeSnapshot.from_dict(d["agent_code"]) if "agent_code" in d else AgentCodeSnapshot(),
            original_code=AgentCodeSnapshot.from_dict(d["original_code"]) if "original_code" in d else AgentCodeSnapshot(),
            performance=PerformanceRecord.from_dict(d["performance"]) if "performance" in d else PerformanceRecord(),
            safety=SafetyStatus.from_dict(d["safety"]) if "safety" in d else SafetyStatus(),
            performance_history=[PerformanceRecord.from_dict(p) for p in d.get("performance_history", [])],
            modification_history=d.get("modification_history", []),
            metadata=d.get("metadata", {}),
        )

    def save(self, path: str) -> None:
        """Save state to file."""
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "PipelineState":
        """Load state from file."""
        with open(path, "r") as f:
            return cls.from_json(f.read())
