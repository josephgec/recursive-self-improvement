"""Memory limiting and monitoring for sandboxed code execution."""

import os
import platform
from dataclasses import dataclass
from typing import Optional


@dataclass
class MemoryStatus:
    """Current memory status.

    Attributes:
        current_mb: Current memory usage in megabytes.
        peak_mb: Peak memory usage in megabytes.
        limit_mb: Memory limit in megabytes.
        available_mb: Available memory before limit.
        exceeded: Whether the limit has been exceeded.
    """

    current_mb: float = 0.0
    peak_mb: float = 0.0
    limit_mb: float = 0.0
    available_mb: float = 0.0
    exceeded: bool = False


class MemoryLimiter:
    """Manages memory limits for sandboxed code execution.

    Uses resource module on Unix systems to set process memory limits.
    Provides monitoring and status tracking on all platforms.
    """

    def __init__(self, max_memory_mb: float = 512.0):
        self.max_memory_mb = max_memory_mb
        self._peak_mb: float = 0.0
        self._has_resource = False
        try:
            import resource
            self._resource = resource
            self._has_resource = True
        except ImportError:
            self._resource = None

    def set_process_limit(self, limit_mb: Optional[float] = None) -> bool:
        """Set the process memory limit.

        Args:
            limit_mb: Memory limit in megabytes. Uses default if not specified.

        Returns:
            True if the limit was set successfully.
        """
        effective_limit = limit_mb or self.max_memory_mb

        if not self._has_resource:
            return False

        try:
            limit_bytes = int(effective_limit * 1024 * 1024)
            self._resource.setrlimit(
                self._resource.RLIMIT_AS,
                (limit_bytes, limit_bytes),
            )
            return True
        except (ValueError, OSError):
            return False

    def get_current_usage_mb(self) -> float:
        """Get current memory usage in megabytes."""
        if self._has_resource:
            try:
                usage = self._resource.getrusage(self._resource.RUSAGE_SELF)
                # maxrss is in kilobytes on Linux, bytes on macOS
                if platform.system() == "Darwin":
                    current_mb = usage.ru_maxrss / (1024 * 1024)
                else:
                    current_mb = usage.ru_maxrss / 1024
                self._peak_mb = max(self._peak_mb, current_mb)
                return current_mb
            except Exception:
                pass

        # Fallback: try psutil-like approach via /proc
        try:
            pid = os.getpid()
            with open(f"/proc/{pid}/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        kb = int(line.split()[1])
                        mb = kb / 1024
                        self._peak_mb = max(self._peak_mb, mb)
                        return mb
        except (FileNotFoundError, PermissionError, ValueError):
            pass

        return 0.0

    def monitor(self) -> MemoryStatus:
        """Get current memory status.

        Returns:
            MemoryStatus with current usage information.
        """
        current = self.get_current_usage_mb()
        self._peak_mb = max(self._peak_mb, current)

        return MemoryStatus(
            current_mb=current,
            peak_mb=self._peak_mb,
            limit_mb=self.max_memory_mb,
            available_mb=max(0, self.max_memory_mb - current),
            exceeded=current > self.max_memory_mb,
        )

    def reset_peak(self) -> None:
        """Reset peak memory tracking."""
        self._peak_mb = 0.0
