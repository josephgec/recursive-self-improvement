"""Track recovery events during stress testing."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.utils.metrics import safe_division


@dataclass
class RecoveryEvent:
    """A single fault-injection -> detection -> recovery event."""

    scenario_name: str
    fault_type: str
    iteration_injected: int
    iteration_detected: Optional[int] = None
    iteration_recovered: Optional[int] = None
    detection_method: Optional[str] = None
    recovery_method: Optional[str] = None
    recovery_quality: float = 0.0  # 0-1: how well did it recover?
    accuracies: List[float] = field(default_factory=list)
    time_to_detect: Optional[float] = None  # seconds
    time_to_recover: Optional[float] = None  # seconds
    complexity_at_fault: int = 0
    _inject_time: float = field(default=0.0, repr=False)
    _detect_time: Optional[float] = field(default=None, repr=False)

    @property
    def was_detected(self) -> bool:
        return self.iteration_detected is not None

    @property
    def was_recovered(self) -> bool:
        return self.iteration_recovered is not None

    @property
    def detection_latency(self) -> Optional[int]:
        """Iterations between injection and detection."""
        if self.iteration_detected is None:
            return None
        return self.iteration_detected - self.iteration_injected

    @property
    def recovery_latency(self) -> Optional[int]:
        """Iterations between injection and recovery."""
        if self.iteration_recovered is None:
            return None
        return self.iteration_recovered - self.iteration_injected


class RecoveryTracker:
    """Tracks fault injection, detection, and recovery across scenarios."""

    def __init__(self) -> None:
        self._events: List[RecoveryEvent] = []
        self._active_event: Optional[RecoveryEvent] = None

    def track_injection(
        self,
        scenario_name: str,
        fault_type: str,
        iteration: int,
        complexity: int = 0,
    ) -> RecoveryEvent:
        """Start tracking a new injection event."""
        if self._active_event is not None:
            # Finalize previous event without recovery
            self._events.append(self._active_event)

        event = RecoveryEvent(
            scenario_name=scenario_name,
            fault_type=fault_type,
            iteration_injected=iteration,
            complexity_at_fault=complexity,
            _inject_time=time.monotonic(),
        )
        self._active_event = event
        return event

    def update(
        self,
        iteration: int,
        accuracy: float,
        detected: bool = False,
        detection_method: Optional[str] = None,
        recovered: bool = False,
        recovery_method: Optional[str] = None,
        recovery_quality: float = 0.0,
    ) -> None:
        """Update the active event with new information."""
        if self._active_event is None:
            return

        self._active_event.accuracies.append(accuracy)

        if detected and self._active_event.iteration_detected is None:
            self._active_event.iteration_detected = iteration
            self._active_event.detection_method = detection_method
            now = time.monotonic()
            self._active_event._detect_time = now
            self._active_event.time_to_detect = now - self._active_event._inject_time

        if recovered and self._active_event.iteration_recovered is None:
            self._active_event.iteration_recovered = iteration
            self._active_event.recovery_method = recovery_method
            self._active_event.recovery_quality = recovery_quality
            now = time.monotonic()
            self._active_event.time_to_recover = now - self._active_event._inject_time

    def finalize_event(self) -> Optional[RecoveryEvent]:
        """Finalize the active event and add it to history."""
        if self._active_event is None:
            return None
        event = self._active_event
        self._events.append(event)
        self._active_event = None
        return event

    @property
    def events(self) -> List[RecoveryEvent]:
        return list(self._events)

    def get_recovery_rate(self) -> float:
        """Fraction of injected faults that were recovered from."""
        if not self._events:
            return 0.0
        recovered = sum(1 for e in self._events if e.was_recovered)
        return safe_division(recovered, len(self._events))

    def get_detection_rate(self) -> float:
        """Fraction of injected faults that were detected."""
        if not self._events:
            return 0.0
        detected = sum(1 for e in self._events if e.was_detected)
        return safe_division(detected, len(self._events))

    def get_mean_time_to_detect(self) -> Optional[float]:
        """Mean detection latency in iterations."""
        latencies = [
            e.detection_latency for e in self._events
            if e.detection_latency is not None
        ]
        if not latencies:
            return None
        return sum(latencies) / len(latencies)

    def get_mean_time_to_recover(self) -> Optional[float]:
        """Mean recovery latency in iterations."""
        latencies = [
            e.recovery_latency for e in self._events
            if e.recovery_latency is not None
        ]
        if not latencies:
            return None
        return sum(latencies) / len(latencies)

    def recovery_by_fault_type(self) -> Dict[str, float]:
        """Recovery rate broken down by fault type."""
        by_type: Dict[str, List[bool]] = defaultdict(list)
        for event in self._events:
            by_type[event.fault_type].append(event.was_recovered)
        return {
            ft: safe_division(sum(recoveries), len(recoveries))
            for ft, recoveries in by_type.items()
        }

    def recovery_by_complexity(self, bins: int = 5) -> Dict[str, float]:
        """Recovery rate broken down by complexity bins."""
        if not self._events:
            return {}

        complexities = [e.complexity_at_fault for e in self._events]
        min_c, max_c = min(complexities), max(complexities)
        if min_c == max_c:
            return {f"{min_c}": self.get_recovery_rate()}

        bin_size = (max_c - min_c) / bins
        result: Dict[str, float] = {}

        for b in range(bins):
            low = min_c + b * bin_size
            high = low + bin_size
            label = f"{int(low)}-{int(high)}"
            events_in_bin = [
                e for e in self._events
                if low <= e.complexity_at_fault < high or (b == bins - 1 and e.complexity_at_fault == high)
            ]
            if events_in_bin:
                recovered = sum(1 for e in events_in_bin if e.was_recovered)
                result[label] = safe_division(recovered, len(events_in_bin))

        return result
