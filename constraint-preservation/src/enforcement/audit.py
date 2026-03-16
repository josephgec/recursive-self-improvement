"""ConstraintAuditLog: append-only audit log with SHA-256 hash chain."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Optional

from src.checker.verdict import SuiteVerdict
from src.constraints.base import CheckContext


class ConstraintAuditLog:
    """Append-only audit log with SHA-256 hash chain for tamper detection."""

    def __init__(self, log_path: Optional[str] = None) -> None:
        self._log_path = log_path
        self._entries: List[Dict[str, Any]] = []
        self._prev_hash: str = "0" * 64  # genesis hash

    def log(
        self,
        verdict: SuiteVerdict,
        context: CheckContext,
        decision: str,
    ) -> Dict[str, Any]:
        """Append an entry to the audit log.

        Returns the created entry (including its hash).
        """
        entry = {
            "index": len(self._entries),
            "timestamp": time.time(),
            "decision": decision,
            "passed": verdict.passed,
            "modification_type": context.modification_type,
            "modification_description": context.modification_description,
            "violations": list(verdict.violations.keys()),
            "results_summary": {
                name: {
                    "satisfied": r.satisfied,
                    "measured_value": r.measured_value,
                    "threshold": r.threshold,
                    "headroom": r.headroom,
                }
                for name, r in verdict.results.items()
            },
            "prev_hash": self._prev_hash,
        }

        # Compute hash over the entry (excluding the hash field itself)
        entry_bytes = json.dumps(entry, sort_keys=True).encode()
        entry_hash = hashlib.sha256(entry_bytes).hexdigest()
        entry["hash"] = entry_hash

        self._prev_hash = entry_hash
        self._entries.append(entry)

        # Persist if path is set
        if self._log_path:
            self._persist(entry)

        return entry

    def get_history(self) -> List[Dict[str, Any]]:
        """Return all log entries."""
        return list(self._entries)

    def get_violations(self) -> List[Dict[str, Any]]:
        """Return only entries where the verdict failed."""
        return [e for e in self._entries if not e["passed"]]

    def verify_integrity(self) -> bool:
        """Verify the SHA-256 hash chain.

        Returns True if all hashes are valid and the chain is unbroken.
        """
        if not self._entries:
            return True

        prev_hash = "0" * 64
        for entry in self._entries:
            # Re-derive the hash
            entry_copy = {k: v for k, v in entry.items() if k != "hash"}
            if entry_copy.get("prev_hash") != prev_hash:
                return False
            entry_bytes = json.dumps(entry_copy, sort_keys=True).encode()
            computed_hash = hashlib.sha256(entry_bytes).hexdigest()
            if computed_hash != entry["hash"]:
                return False
            prev_hash = computed_hash

        return True

    def _persist(self, entry: Dict[str, Any]) -> None:
        """Append entry to the log file as a JSON line."""
        with open(self._log_path, "a") as f:  # type: ignore[arg-type]
            f.write(json.dumps(entry) + "\n")
