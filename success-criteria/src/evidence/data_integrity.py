"""Data integrity verification for evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class IntegrityReport:
    """Report from integrity verification."""

    overall_valid: bool
    hash_chain_valid: bool
    preregistration_valid: bool
    timestamp_valid: bool
    issues: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """One-line summary."""
        status = "VALID" if self.overall_valid else "INVALID"
        n_issues = len(self.issues)
        return f"Integrity: {status} ({n_issues} issues found)"


class DataIntegrityVerifier:
    """Verifies integrity of evidence data."""

    def verify(
        self,
        evidence: Any,
        preregistration_hash: str = "",
    ) -> IntegrityReport:
        """Verify all integrity checks on the evidence.

        Args:
            evidence: The Evidence object to verify.
            preregistration_hash: Expected hash of pre-registration config.

        Returns:
            IntegrityReport with results of all checks.
        """
        issues: List[str] = []
        details: Dict[str, Any] = {}

        # Check 1: Hash chain
        hash_chain_valid = self._check_hash_chain(evidence, issues, details)

        # Check 2: Preregistration hash
        prereg_valid = self._check_preregistration(
            preregistration_hash, issues, details
        )

        # Check 3: Timestamp ordering
        timestamp_valid = self._check_timestamps(evidence, issues, details)

        overall = hash_chain_valid and prereg_valid and timestamp_valid

        return IntegrityReport(
            overall_valid=overall,
            hash_chain_valid=hash_chain_valid,
            preregistration_valid=prereg_valid,
            timestamp_valid=timestamp_valid,
            issues=issues,
            details=details,
        )

    def _check_hash_chain(
        self,
        evidence: Any,
        issues: List[str],
        details: Dict[str, Any],
    ) -> bool:
        """Verify hash chain in audit trail."""
        audit = getattr(evidence, "audit_trail", {})
        chain = audit.get("hash_chain", [])

        if not chain:
            issues.append("No hash chain found in audit trail")
            details["hash_chain"] = {"status": "missing"}
            return False

        prev_hash = "genesis"
        for i, entry in enumerate(chain):
            data = entry.get("data", "")
            expected = entry.get("hash", "")
            computed = hashlib.sha256(
                f"{prev_hash}{data}".encode("utf-8")
            ).hexdigest()

            if computed != expected:
                issues.append(
                    f"Hash chain broken at entry {i}: "
                    f"expected {expected[:16]}..., got {computed[:16]}..."
                )
                details["hash_chain"] = {
                    "status": "broken",
                    "broken_at": i,
                }
                return False
            prev_hash = computed

        details["hash_chain"] = {
            "status": "valid",
            "length": len(chain),
        }
        return True

    def _check_preregistration(
        self,
        expected_hash: str,
        issues: List[str],
        details: Dict[str, Any],
    ) -> bool:
        """Verify preregistration config hasn't changed."""
        if not expected_hash:
            # No hash provided — skip check but note it
            details["preregistration"] = {"status": "skipped", "reason": "no hash provided"}
            return True

        # Compute hash of the thresholds config
        actual_hash = self.compute_config_hash()
        valid = actual_hash == expected_hash

        if not valid:
            issues.append(
                f"Preregistration hash mismatch: "
                f"expected {expected_hash[:16]}..., got {actual_hash[:16]}..."
            )

        details["preregistration"] = {
            "status": "valid" if valid else "mismatch",
            "expected": expected_hash,
            "actual": actual_hash,
        }
        return valid

    def _check_timestamps(
        self,
        evidence: Any,
        issues: List[str],
        details: Dict[str, Any],
    ) -> bool:
        """Check that phase timestamps are in order."""
        phases = ["phase_0", "phase_1", "phase_2", "phase_3", "phase_4"]
        timestamps = []

        for phase_name in phases:
            phase_data = getattr(evidence, phase_name, {})
            ts = phase_data.get("timestamp", "")
            if ts:
                timestamps.append((phase_name, ts))

        if len(timestamps) < 2:
            details["timestamps"] = {"status": "insufficient_data"}
            return True

        valid = True
        for i in range(len(timestamps) - 1):
            if timestamps[i][1] > timestamps[i + 1][1]:
                issues.append(
                    f"Timestamp out of order: {timestamps[i][0]} "
                    f"({timestamps[i][1]}) > {timestamps[i + 1][0]} "
                    f"({timestamps[i + 1][1]})"
                )
                valid = False

        details["timestamps"] = {
            "status": "valid" if valid else "out_of_order",
            "checked": len(timestamps),
        }
        return valid

    @staticmethod
    def compute_config_hash(config_path: str = "") -> str:
        """Compute SHA-256 hash of a config file or default content."""
        if config_path:
            try:
                with open(config_path, "rb") as f:
                    return hashlib.sha256(f.read()).hexdigest()
            except FileNotFoundError:
                return hashlib.sha256(b"missing").hexdigest()

        # Default: hash of the canonical threshold values
        canonical = json.dumps(
            {
                "sustained_improvement": {
                    "trend_alpha": 0.05,
                    "min_total_gain_pp": 5.0,
                    "min_collapse_divergence_pp": 10.0,
                },
                "paradigm_improvement": {
                    "alpha": 0.05,
                    "min_effects": {
                        "symcode": 5.0,
                        "godel": 2.0,
                        "soar": 5.0,
                        "rlm": 10.0,
                    },
                },
                "gdi_bounds": {
                    "max_gdi": 0.50,
                    "max_consecutive_yellow": 5,
                },
                "publication_acceptance": {
                    "min_accepted": 2,
                    "min_tier_1_or_2": 1,
                },
                "auditability": {
                    "min_reasoning_traces": 20,
                },
            },
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
