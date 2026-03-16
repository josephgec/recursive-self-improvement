"""FINAL protocol implementation for result extraction."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class FinalResult:
    """Result extracted via the FINAL protocol.

    Attributes:
        value: The final result value.
        source: How the result was obtained ('FINAL', 'FINAL_VAR').
        variable_name: Name of the variable (for FINAL_VAR).
    """

    value: Any = None
    source: str = ""
    variable_name: str = ""

    @property
    def has_result(self) -> bool:
        return self.value is not None


class FinalProtocol:
    """Implements the FINAL/FINAL_VAR protocol for result extraction.

    Provides injectable code that defines FINAL() and FINAL_VAR() functions
    in the REPL namespace. These functions signal that the code has produced
    a final result.

    Usage in sandboxed code:
        FINAL("the answer is 42")
        FINAL_VAR("result")  # extracts the value of 'result' variable
    """

    FINAL_RESULT_KEY = "__FINAL_RESULT__"
    FINAL_VAR_KEY = "__FINAL_VAR_NAME__"

    def get_injectable_code(self) -> str:
        """Get Python code that defines FINAL and FINAL_VAR functions.

        Returns:
            Python source code string.
        """
        return '''
def FINAL(value):
    """Signal a final result value."""
    globals()["__FINAL_RESULT__"] = value

def FINAL_VAR(name):
    """Signal that a variable contains the final result."""
    globals()["__FINAL_VAR_NAME__"] = name
'''

    def inject(self, namespace: Dict[str, Any]) -> None:
        """Inject FINAL protocol functions into a namespace.

        Creates closure-based FINAL and FINAL_VAR functions that
        directly reference the namespace dict, avoiding any need
        for globals() in the restricted sandbox.

        Args:
            namespace: The namespace to inject into.
        """
        ns = namespace

        def FINAL(value):
            """Signal a final result value."""
            ns["__FINAL_RESULT__"] = value

        def FINAL_VAR(name):
            """Signal that a variable contains the final result."""
            ns["__FINAL_VAR_NAME__"] = name

        namespace["FINAL"] = FINAL
        namespace["FINAL_VAR"] = FINAL_VAR

    def check_for_result(self, namespace: Dict[str, Any]) -> Optional[FinalResult]:
        """Check if a FINAL result has been signaled.

        Args:
            namespace: The namespace to check.

        Returns:
            FinalResult if a result was signaled, None otherwise.
        """
        # Check for direct FINAL() call
        if self.FINAL_RESULT_KEY in namespace:
            return FinalResult(
                value=namespace[self.FINAL_RESULT_KEY],
                source="FINAL",
            )

        # Check for FINAL_VAR() call
        if self.FINAL_VAR_KEY in namespace:
            var_name = namespace[self.FINAL_VAR_KEY]
            if var_name in namespace:
                return FinalResult(
                    value=namespace[var_name],
                    source="FINAL_VAR",
                    variable_name=var_name,
                )
            else:
                return FinalResult(
                    value=None,
                    source="FINAL_VAR",
                    variable_name=var_name,
                )

        return None

    def reset(self, namespace: Dict[str, Any]) -> None:
        """Clear any FINAL signals from the namespace.

        Args:
            namespace: The namespace to reset.
        """
        namespace.pop(self.FINAL_RESULT_KEY, None)
        namespace.pop(self.FINAL_VAR_KEY, None)
