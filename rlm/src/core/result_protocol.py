"""Result protocol: FINAL("text") and FINAL_VAR("name") patterns."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class SignalKind(Enum):
    FINAL = "FINAL"
    FINAL_VAR = "FINAL_VAR"


@dataclass
class FinalSignal:
    """Represents a detected FINAL or FINAL_VAR signal."""
    kind: SignalKind
    argument: str  # literal text for FINAL, variable name for FINAL_VAR


@dataclass
class RLMResult:
    """The final result of an RLM session."""
    value: Any
    source: str  # "FINAL" or "FINAL_VAR"
    raw_argument: str

    def __str__(self) -> str:
        return str(self.value)


# Regex patterns
_FINAL_PATTERN = re.compile(r'FINAL\(\s*(?:"([^"]*?)"|\'([^\']*?)\')\s*\)')
_FINAL_VAR_PATTERN = re.compile(r'FINAL_VAR\(\s*(?:"([^"]*?)"|\'([^\']*?)\')\s*\)')


class ResultProtocol:
    """Detect and extract FINAL signals from LLM-generated code."""

    @staticmethod
    def detect_final(code: str) -> Optional[FinalSignal]:
        """Check if *code* contains a FINAL(...) or FINAL_VAR(...) call.

        Returns the first signal found, or ``None``.
        """
        # Check FINAL_VAR first (more specific)
        m = _FINAL_VAR_PATTERN.search(code)
        if m:
            arg = m.group(1) if m.group(1) is not None else m.group(2)
            return FinalSignal(kind=SignalKind.FINAL_VAR, argument=arg)

        m = _FINAL_PATTERN.search(code)
        if m:
            arg = m.group(1) if m.group(1) is not None else m.group(2)
            return FinalSignal(kind=SignalKind.FINAL, argument=arg)

        return None

    @staticmethod
    def extract_result(signal: FinalSignal, repl: Dict[str, Any]) -> RLMResult:
        """Given a FinalSignal, extract the result value.

        For ``FINAL("text")`` the value is the literal text.
        For ``FINAL_VAR("name")`` the value is looked up in the REPL namespace.
        """
        if signal.kind == SignalKind.FINAL:
            return RLMResult(
                value=signal.argument,
                source="FINAL",
                raw_argument=signal.argument,
            )
        # FINAL_VAR
        var_name = signal.argument
        value = repl.get(var_name, f"<undefined: {var_name}>")
        return RLMResult(
            value=value,
            source="FINAL_VAR",
            raw_argument=var_name,
        )

    @staticmethod
    def inject_protocol_functions(repl: Dict[str, Any]) -> None:
        """Inject FINAL and FINAL_VAR as callable markers into the REPL.

        They are implemented as identity functions so the code runs without
        error, but the *detect_final* check happens before execution.
        """

        def _final(value: Any) -> Any:
            return value

        def _final_var(name: str) -> str:
            return name

        repl["FINAL"] = _final
        repl["FINAL_VAR"] = _final_var
