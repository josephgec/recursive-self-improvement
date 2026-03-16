"""Resource monitoring for sandboxed REPL instances."""

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from src.safety.memory_limiter import MemoryLimiter


@dataclass
class ResourceStatus:
    """Resource usage status for a REPL instance.

    Attributes:
        repl_id: Identifier of the REPL.
        memory_mb: Current memory usage in megabytes.
        uptime_seconds: Time since REPL creation.
        execution_count: Number of executions performed.
        is_alive: Whether the REPL is still running.
        last_activity: Timestamp of last activity.
    """

    repl_id: str = ""
    memory_mb: float = 0.0
    uptime_seconds: float = 0.0
    execution_count: int = 0
    is_alive: bool = True
    last_activity: float = 0.0


class ResourceMonitor:
    """Monitors resource usage across REPL instances.

    Tracks memory usage, execution counts, and uptime for all
    registered REPL instances.
    """

    def __init__(self):
        self._repls: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    def register(self, repl_id: str) -> None:
        """Register a REPL for monitoring.

        Args:
            repl_id: Unique identifier for the REPL.
        """
        with self._lock:
            self._repls[repl_id] = {
                "created_at": time.time(),
                "execution_count": 0,
                "last_activity": time.time(),
                "is_alive": True,
                "memory_mb": 0.0,
            }

    def unregister(self, repl_id: str) -> None:
        """Unregister a REPL from monitoring.

        Args:
            repl_id: Identifier of the REPL to remove.
        """
        with self._lock:
            if repl_id in self._repls:
                self._repls[repl_id]["is_alive"] = False

    def record_execution(self, repl_id: str, memory_mb: float = 0.0) -> None:
        """Record an execution event for a REPL.

        Args:
            repl_id: Identifier of the REPL.
            memory_mb: Memory used during execution.
        """
        with self._lock:
            if repl_id in self._repls:
                self._repls[repl_id]["execution_count"] += 1
                self._repls[repl_id]["last_activity"] = time.time()
                self._repls[repl_id]["memory_mb"] = memory_mb

    def get_status(self, repl_id: str) -> Optional[ResourceStatus]:
        """Get the resource status for a specific REPL.

        Args:
            repl_id: Identifier of the REPL.

        Returns:
            ResourceStatus or None if not registered.
        """
        with self._lock:
            info = self._repls.get(repl_id)
            if info is None:
                return None

            now = time.time()
            return ResourceStatus(
                repl_id=repl_id,
                memory_mb=info["memory_mb"],
                uptime_seconds=now - info["created_at"],
                execution_count=info["execution_count"],
                is_alive=info["is_alive"],
                last_activity=info["last_activity"],
            )

    def get_all_status(self) -> Dict[str, ResourceStatus]:
        """Get resource status for all registered REPLs.

        Returns:
            Dictionary mapping REPL IDs to their ResourceStatus.
        """
        result = {}
        with self._lock:
            for repl_id in self._repls:
                info = self._repls[repl_id]
                now = time.time()
                result[repl_id] = ResourceStatus(
                    repl_id=repl_id,
                    memory_mb=info["memory_mb"],
                    uptime_seconds=now - info["created_at"],
                    execution_count=info["execution_count"],
                    is_alive=info["is_alive"],
                    last_activity=info["last_activity"],
                )
        return result
