"""Runtime inspection for self-awareness and introspection."""

from __future__ import annotations

import inspect
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class FunctionInfo:
    """Metadata about a function."""

    name: str
    module: str
    source: str
    signature: str
    doc: str = ""
    line_count: int = 0


@dataclass
class PerformanceSnapshot:
    """Snapshot of recent performance metrics."""

    accuracy_last_n: float = 0.0
    trend: float = 0.0  # slope of linear regression
    error_types: dict[str, int] = field(default_factory=dict)
    total_tasks: int = 0
    correct_tasks: int = 0


class RuntimeInspector:
    """Inspects agent internals for self-awareness."""

    def __init__(self) -> None:
        self._tracked_functions: dict[str, Callable[..., Any]] = {}

    def inspect_variables(self, obj: Any) -> dict[str, Any]:
        """Inspect the public attributes of an object."""
        result: dict[str, Any] = {}
        for name in dir(obj):
            if name.startswith("_"):
                continue
            try:
                val = getattr(obj, name)
                if not callable(val):
                    result[name] = repr(val)
            except Exception:
                result[name] = "<inaccessible>"
        return result

    def inspect_functions(self, obj: Any) -> list[FunctionInfo]:
        """List all callable methods on an object with their metadata."""
        functions: list[FunctionInfo] = []
        for name in dir(obj):
            if name.startswith("_"):
                continue
            try:
                attr = getattr(obj, name)
                if callable(attr):
                    source = self.get_function_source(attr)
                    sig = str(inspect.signature(attr)) if inspect.ismethod(attr) or inspect.isfunction(attr) else "()"
                    doc = inspect.getdoc(attr) or ""
                    functions.append(
                        FunctionInfo(
                            name=name,
                            module=getattr(attr, "__module__", ""),
                            source=source,
                            signature=sig,
                            doc=doc,
                            line_count=source.count("\n") + 1 if source else 0,
                        )
                    )
            except Exception:
                continue
        return functions

    def inspect_performance(
        self,
        accuracy_history: list[float],
        n: int = 5,
        error_log: list[dict[str, Any]] | None = None,
    ) -> PerformanceSnapshot:
        """Analyze recent performance with trend detection."""
        if not accuracy_history:
            return PerformanceSnapshot()

        recent = accuracy_history[-n:]
        accuracy_last_n = float(np.mean(recent))

        # Linear regression for trend
        trend = 0.0
        if len(recent) >= 2:
            x = np.arange(len(recent), dtype=float)
            y = np.array(recent, dtype=float)
            coeffs = np.polyfit(x, y, 1)
            trend = float(coeffs[0])

        # Error type counts
        error_types: dict[str, int] = {}
        if error_log:
            for entry in error_log:
                etype = entry.get("error_type", "unknown")
                error_types[etype] = error_types.get(etype, 0) + 1

        total = len(accuracy_history)
        correct = sum(1 for a in accuracy_history if a >= 0.5)

        return PerformanceSnapshot(
            accuracy_last_n=accuracy_last_n,
            trend=trend,
            error_types=error_types,
            total_tasks=total,
            correct_tasks=correct,
        )

    def generate_self_report(
        self,
        accuracy_history: list[float],
        modifications: list[dict[str, Any]] | None = None,
        max_tokens: int = 500,
    ) -> str:
        """Generate a concise textual self-report under max_tokens words."""
        perf = self.inspect_performance(accuracy_history)
        lines: list[str] = [
            "=== Agent Self-Report ===",
            f"Total iterations: {perf.total_tasks}",
            f"Correct: {perf.correct_tasks}/{perf.total_tasks}",
            f"Recent accuracy (last 5): {perf.accuracy_last_n:.3f}",
            f"Trend: {perf.trend:+.4f}",
        ]

        if modifications:
            lines.append(f"Modifications applied: {len(modifications)}")
            recent_mods = modifications[-3:]
            for m in recent_mods:
                target = m.get("target", "unknown")
                success = m.get("success", False)
                lines.append(f"  - {target}: {'accepted' if success else 'rolled back'}")

        if perf.error_types:
            lines.append("Error types:")
            for etype, count in sorted(perf.error_types.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"  - {etype}: {count}")

        report = "\n".join(lines)
        # Truncate to approximate token limit (rough: 1 token ~ 4 chars)
        max_chars = max_tokens * 4
        if len(report) > max_chars:
            report = report[:max_chars] + "\n[truncated]"
        return report

    def get_function_source(self, func: Callable[..., Any]) -> str:
        """Get the source code of a function."""
        try:
            source = inspect.getsource(func)
            return textwrap.dedent(source)
        except (TypeError, OSError):
            return ""

    def track_function(self, name: str, func: Callable[..., Any]) -> None:
        """Register a function for tracking."""
        self._tracked_functions[name] = func

    def get_tracked_functions(self) -> dict[str, FunctionInfo]:
        """Get info about all tracked functions."""
        result: dict[str, FunctionInfo] = {}
        for name, func in self._tracked_functions.items():
            source = self.get_function_source(func)
            sig = str(inspect.signature(func)) if inspect.isfunction(func) else "()"
            doc = inspect.getdoc(func) or ""
            result[name] = FunctionInfo(
                name=name,
                module=getattr(func, "__module__", ""),
                source=source,
                signature=sig,
                doc=doc,
                line_count=source.count("\n") + 1 if source else 0,
            )
        return result
