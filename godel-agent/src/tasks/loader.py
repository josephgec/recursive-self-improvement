"""Task suite loader that loads tasks by domain name."""

from __future__ import annotations

from src.core.executor import Task
from src.tasks.math_tasks import MathTaskLoader
from src.tasks.code_tasks import CodeTaskLoader


class TaskSuiteLoader:
    """Loads task suites by domain name."""

    _loaders = {
        "math": MathTaskLoader,
        "code": CodeTaskLoader,
    }

    @classmethod
    def load(cls, domain: str = "all") -> list[Task]:
        """Load tasks by domain name.

        Args:
            domain: "math", "code", or "all" for everything.
        """
        if domain == "all":
            tasks: list[Task] = []
            for loader_cls in cls._loaders.values():
                tasks.extend(loader_cls().load())
            return tasks

        loader_cls = cls._loaders.get(domain)
        if loader_cls is None:
            raise ValueError(
                f"Unknown domain '{domain}'. Available: {list(cls._loaders.keys())}"
            )
        return loader_cls().load()

    @classmethod
    def available_domains(cls) -> list[str]:
        """List available task domains."""
        return list(cls._loaders.keys()) + ["all"]
