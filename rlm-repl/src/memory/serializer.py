"""Serialization for REPL variable snapshots."""

import json
from typing import Any, Optional, Tuple

from src.interface.errors import SerializationError


class REPLSerializer:
    """Serializes and deserializes REPL variables.

    Uses fast paths for common types (JSON for primitives) and
    falls back to dill for complex objects.
    """

    # Type tags for the serialization format
    TAG_JSON = "json"
    TAG_NUMPY = "numpy"
    TAG_DILL = "dill"

    def serialize(self, value: Any, name: str = "") -> Tuple[str, bytes]:
        """Serialize a value.

        Args:
            value: The value to serialize.
            name: Variable name (for error messages).

        Returns:
            Tuple of (type_tag, serialized_bytes).

        Raises:
            SerializationError: If serialization fails.
        """
        # Fast path: JSON-serializable primitives
        if isinstance(value, (int, float, str, bool, type(None), list, dict)):
            try:
                data = json.dumps(value).encode("utf-8")
                return (self.TAG_JSON, data)
            except (TypeError, ValueError):
                pass  # Fall through to dill

        # Numpy arrays
        try:
            import numpy as np
            if isinstance(value, np.ndarray):
                import io
                buf = io.BytesIO()
                np.save(buf, value)
                return (self.TAG_NUMPY, buf.getvalue())
        except ImportError:
            pass

        # Dill fallback
        try:
            import dill
            data = dill.dumps(value)
            return (self.TAG_DILL, data)
        except Exception as e:
            raise SerializationError(
                variable_name=name,
                message=f"Failed to serialize '{name}': {e}",
            )

    def deserialize(self, tag: str, data: bytes, name: str = "") -> Any:
        """Deserialize a value.

        Args:
            tag: Type tag indicating the serialization format.
            data: Serialized bytes.
            name: Variable name (for error messages).

        Returns:
            The deserialized value.

        Raises:
            SerializationError: If deserialization fails.
        """
        try:
            if tag == self.TAG_JSON:
                return json.loads(data.decode("utf-8"))
            elif tag == self.TAG_NUMPY:
                import numpy as np
                import io
                buf = io.BytesIO(data)
                return np.load(buf, allow_pickle=False)
            elif tag == self.TAG_DILL:
                import dill
                return dill.loads(data)
            else:
                raise SerializationError(
                    variable_name=name,
                    message=f"Unknown serialization tag: {tag}",
                )
        except SerializationError:
            raise
        except Exception as e:
            raise SerializationError(
                variable_name=name,
                message=f"Failed to deserialize '{name}': {e}",
            )

    def round_trip(self, value: Any, name: str = "") -> Any:
        """Serialize and immediately deserialize a value.

        Args:
            value: The value to round-trip.
            name: Variable name.

        Returns:
            The deserialized value.
        """
        tag, data = self.serialize(value, name)
        return self.deserialize(tag, data, name)

    def can_serialize(self, value: Any) -> bool:
        """Check if a value can be serialized.

        Args:
            value: The value to check.

        Returns:
            True if the value can be serialized.
        """
        try:
            self.serialize(value, "")
            return True
        except SerializationError:
            return False
