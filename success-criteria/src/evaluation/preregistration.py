"""Preregistration verification — ensures thresholds haven't changed."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict


class PreregistrationVerifier:
    """Verifies that evaluation thresholds match pre-registered values."""

    def __init__(self, registered_hash: str = ""):
        self._registered_hash = registered_hash

    def verify_thresholds_unchanged(
        self, config_hash: str
    ) -> Dict[str, Any]:
        """Verify that the current config hash matches registered hash.

        Args:
            config_hash: SHA-256 hash of the current threshold config.

        Returns:
            Dict with verification result.
        """
        if not self._registered_hash:
            return {
                "verified": False,
                "reason": "No pre-registered hash available",
                "current_hash": config_hash,
            }

        matches = config_hash == self._registered_hash
        return {
            "verified": matches,
            "registered_hash": self._registered_hash,
            "current_hash": config_hash,
            "reason": (
                "Hashes match" if matches
                else "Config has been modified since pre-registration"
            ),
        }

    @staticmethod
    def compute_hash(config: Dict[str, Any]) -> str:
        """Compute deterministic hash of a config dictionary."""
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def compute_file_hash(filepath: str) -> str:
        """Compute SHA-256 hash of a file."""
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def register(self, config: Dict[str, Any]) -> str:
        """Register a config and store its hash. Returns the hash."""
        h = self.compute_hash(config)
        self._registered_hash = h
        return h

    @property
    def registered_hash(self) -> str:
        return self._registered_hash
