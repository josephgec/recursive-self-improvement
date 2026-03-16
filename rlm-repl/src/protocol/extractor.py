"""Result extraction from REPL namespaces."""

from typing import Any, Dict, Optional

from src.protocol.final_functions import FinalProtocol, FinalResult


class ResultExtractor:
    """Extracts results from REPL execution.

    Handles type-aware serialization of results including
    special handling for numpy arrays, pandas DataFrames, etc.
    """

    def __init__(self, protocol: Optional[FinalProtocol] = None):
        self._protocol = protocol or FinalProtocol()

    def extract_from_repl(self, namespace: Dict[str, Any]) -> Optional[FinalResult]:
        """Extract a FINAL result from a REPL namespace.

        Args:
            namespace: The REPL namespace to check.

        Returns:
            FinalResult if found, None otherwise.
        """
        return self._protocol.check_for_result(namespace)

    def serialize_value(self, value: Any) -> Dict[str, Any]:
        """Serialize a value in a type-aware manner.

        Args:
            value: The value to serialize.

        Returns:
            Dictionary with 'type', 'value', and optionally 'repr' keys.
        """
        if value is None:
            return {"type": "NoneType", "value": None, "repr": "None"}

        if isinstance(value, bool):
            return {"type": "bool", "value": value, "repr": str(value)}

        if isinstance(value, int):
            return {"type": "int", "value": value, "repr": str(value)}

        if isinstance(value, float):
            return {"type": "float", "value": value, "repr": str(value)}

        if isinstance(value, str):
            return {"type": "str", "value": value, "repr": repr(value)}

        if isinstance(value, list):
            return {
                "type": "list",
                "value": [self.serialize_value(item) for item in value],
                "repr": repr(value),
                "length": len(value),
            }

        if isinstance(value, dict):
            return {
                "type": "dict",
                "value": {
                    str(k): self.serialize_value(v) for k, v in value.items()
                },
                "repr": repr(value),
                "length": len(value),
            }

        # Numpy array
        try:
            import numpy as np
            if isinstance(value, np.ndarray):
                return {
                    "type": "numpy.ndarray",
                    "value": value.tolist(),
                    "repr": repr(value),
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
        except ImportError:
            pass

        # Pandas DataFrame
        try:
            import pandas as pd
            if isinstance(value, pd.DataFrame):
                return {
                    "type": "pandas.DataFrame",
                    "value": value.to_dict(orient="records"),
                    "repr": repr(value),
                    "shape": list(value.shape),
                    "columns": list(value.columns),
                }
            if isinstance(value, pd.Series):
                return {
                    "type": "pandas.Series",
                    "value": value.tolist(),
                    "repr": repr(value),
                    "length": len(value),
                    "name": str(value.name),
                }
        except ImportError:
            pass

        # Fallback
        return {
            "type": type(value).__name__,
            "value": str(value),
            "repr": repr(value),
        }
