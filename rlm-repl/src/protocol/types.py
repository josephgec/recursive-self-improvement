"""Protocol types for the FINAL protocol."""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class FinalSignal:
    """A FINAL signal detected in code or execution.

    Attributes:
        signal_type: Type of signal ('FINAL' or 'FINAL_VAR').
        value: The value passed to FINAL() or the variable name for FINAL_VAR().
        line: Line number where the signal was detected.
        raw_code: The raw code containing the signal.
    """

    signal_type: str  # "FINAL" or "FINAL_VAR"
    value: Any = None
    line: int = 0
    raw_code: str = ""

    @property
    def is_final(self) -> bool:
        return self.signal_type == "FINAL"

    @property
    def is_final_var(self) -> bool:
        return self.signal_type == "FINAL_VAR"
