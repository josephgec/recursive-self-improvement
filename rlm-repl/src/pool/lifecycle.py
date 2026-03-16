"""REPL lifecycle management."""

import logging
from typing import Optional

from src.interface.base import SandboxREPL
from src.backends.local import LocalREPL
from src.safety.policy import SafetyPolicy

logger = logging.getLogger(__name__)


class REPLLifecycle:
    """Manages the lifecycle of REPL instances.

    Handles creation, warming, recycling, destroying, and health checking.
    """

    def __init__(self, policy: Optional[SafetyPolicy] = None):
        self._policy = policy or SafetyPolicy()

    def create(self) -> SandboxREPL:
        """Create a new REPL instance.

        Returns:
            A fresh SandboxREPL instance.
        """
        return LocalREPL(policy=self._policy)

    def warm(self, repl: SandboxREPL) -> bool:
        """Warm up a REPL by executing a trivial operation.

        Args:
            repl: The REPL to warm up.

        Returns:
            True if warming succeeded.
        """
        try:
            result = repl.execute("_ = None")
            return result.success
        except Exception:
            logger.warning("Failed to warm REPL")
            return False

    def recycle(self, repl: SandboxREPL) -> SandboxREPL:
        """Recycle a REPL by resetting it.

        If reset fails, creates a new instance.

        Args:
            repl: The REPL to recycle.

        Returns:
            A recycled or new SandboxREPL instance.
        """
        try:
            if repl.is_alive():
                repl.reset()
                return repl
        except Exception:
            pass

        # Create new instance
        try:
            repl.shutdown()
        except Exception:
            pass
        return self.create()

    def destroy(self, repl: SandboxREPL) -> None:
        """Destroy a REPL instance.

        Args:
            repl: The REPL to destroy.
        """
        try:
            repl.shutdown()
        except Exception:
            logger.warning("Error during REPL shutdown")

    def health_check(self, repl: SandboxREPL) -> bool:
        """Check if a REPL is healthy and responsive.

        Args:
            repl: The REPL to check.

        Returns:
            True if the REPL is healthy.
        """
        try:
            if not repl.is_alive():
                return False
            result = repl.execute("1 + 1")
            return result.success
        except Exception:
            return False
