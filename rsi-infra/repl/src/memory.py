"""Persistent variable memory for REPL sessions.

Provides serialisation with fast-paths for numpy arrays and pandas
DataFrames, size tracking, and LRU-style eviction.
"""

from __future__ import annotations

import copy
import io
import pickle
import sys
from typing import Any


def _has_numpy() -> bool:
    try:
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


def _has_pandas() -> bool:
    try:
        import pandas  # noqa: F401
        return True
    except ImportError:
        return False


class REPLMemory:
    """Key-value store for REPL variables with serialisation and eviction.

    Internally each value is stored *serialised* so that size accounting is
    accurate and ``clone()`` is cheap.
    """

    def __init__(self, max_size_bytes: int = 256 * 1024 * 1024) -> None:
        self._store: dict[str, bytes] = {}
        self._max_size_bytes = max_size_bytes

    # ------------------------------------------------------------------
    # Public API – bulk operations
    # ------------------------------------------------------------------

    def save(self, namespace: dict[str, Any]) -> None:
        """Persist every key/value in *namespace*."""
        for name, value in namespace.items():
            self.save_variable(name, value)

    def load(self) -> dict[str, Any]:
        """Deserialise all stored variables and return as a dict."""
        return {name: self.load_variable(name) for name in self._store}

    # ------------------------------------------------------------------
    # Public API – single variable
    # ------------------------------------------------------------------

    def save_variable(self, name: str, value: Any) -> None:
        """Serialise and store a single variable.

        If the total memory budget would be exceeded, the largest entry is
        evicted first (repeatedly) until the new value fits.
        """
        data = self._serialise(value)
        # Evict until there is room (or the store is empty)
        while (
            self._total_bytes() + len(data)
            - len(self._store.get(name, b""))
            > self._max_size_bytes
            and self._store
        ):
            self.evict_largest()
        self._store[name] = data

    def load_variable(self, name: str) -> Any:
        if name not in self._store:
            raise KeyError(name)
        return self._deserialise(self._store[name])

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def clone(self) -> REPLMemory:
        """Return an independent deep copy of this memory store."""
        new = REPLMemory(max_size_bytes=self._max_size_bytes)
        new._store = copy.deepcopy(self._store)
        return new

    def size_bytes(self) -> int:
        """Total serialised size of all stored variables."""
        return self._total_bytes()

    def evict_largest(self) -> str | None:
        """Remove and return the name of the largest stored variable.

        Returns ``None`` when the store is empty.
        """
        if not self._store:
            return None
        largest = max(self._store, key=lambda k: len(self._store[k]))
        del self._store[largest]
        return largest

    def summary(self) -> dict[str, int]:
        """Return ``{name: size_in_bytes}`` for every stored variable."""
        return {name: len(data) for name, data in self._store.items()}

    def __contains__(self, name: str) -> bool:
        return name in self._store

    def __len__(self) -> int:
        return len(self._store)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialise(value: Any) -> bytes:
        """Serialise *value*, using format-specific fast-paths where possible."""
        # numpy ndarray → np.save into a BytesIO
        if _has_numpy():
            import numpy as np
            if isinstance(value, np.ndarray):
                buf = io.BytesIO()
                np.save(buf, value)
                return buf.getvalue()

        # pandas DataFrame → parquet bytes (if pyarrow available) or pickle
        if _has_pandas():
            import pandas as pd
            if isinstance(value, pd.DataFrame):
                try:
                    buf = io.BytesIO()
                    value.to_parquet(buf, engine="pyarrow")
                    return buf.getvalue()
                except ImportError:
                    # pyarrow not installed, fall through to pickle
                    pass

        return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _deserialise(data: bytes) -> Any:
        """Deserialise bytes produced by :meth:`_serialise`."""
        # Detect numpy .npy format (magic bytes: \x93NUMPY)
        if data[:6] == b"\x93NUMPY":
            if _has_numpy():
                import numpy as np
                buf = io.BytesIO(data)
                return np.load(buf)

        # Detect parquet format (magic bytes: PAR1)
        if data[:4] == b"PAR1":
            if _has_pandas():
                import pandas as pd
                buf = io.BytesIO(data)
                return pd.read_parquet(buf, engine="pyarrow")

        return pickle.loads(data)  # noqa: S301

    def _total_bytes(self) -> int:
        return sum(len(v) for v in self._store.values())
