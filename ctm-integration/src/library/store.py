"""JSON-based persistent storage for verified rules."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from src.library.rule import VerifiedRule


class RuleStore:
    """Persistent store for verified rules using JSON files.

    Supports CRUD operations, domain filtering, and deduplication.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        """Initialize the rule store.

        Args:
            path: Path to the JSON file for persistence.
                  If None, operates in-memory only.
        """
        self.path = path
        self._rules: Dict[str, VerifiedRule] = {}

        if path and os.path.exists(path):
            self._load()

    @property
    def size(self) -> int:
        """Number of rules in the store."""
        return len(self._rules)

    def add(self, rule: VerifiedRule) -> None:
        """Add a rule to the store.

        Args:
            rule: The verified rule to add.
        """
        self._rules[rule.rule_id] = rule
        self._save()

    def get(self, rule_id: str) -> Optional[VerifiedRule]:
        """Get a rule by its ID.

        Args:
            rule_id: The rule identifier.

        Returns:
            The VerifiedRule, or None if not found.
        """
        return self._rules.get(rule_id)

    def remove(self, rule_id: str) -> bool:
        """Remove a rule from the store.

        Args:
            rule_id: The rule identifier.

        Returns:
            True if the rule was removed, False if not found.
        """
        if rule_id in self._rules:
            del self._rules[rule_id]
            self._save()
            return True
        return False

    def list_all(self) -> List[VerifiedRule]:
        """List all rules in the store.

        Returns:
            List of all verified rules.
        """
        return list(self._rules.values())

    def list_by_domain(self, domain: str) -> List[VerifiedRule]:
        """List rules filtered by domain.

        Args:
            domain: The domain to filter by.

        Returns:
            List of rules in the specified domain.
        """
        return [r for r in self._rules.values() if r.domain == domain]

    def deduplicate(self) -> int:
        """Remove duplicate rules based on source code hash.

        Keeps the rule with the highest accuracy when duplicates are found.

        Returns:
            Number of duplicates removed.
        """
        seen_hashes: Dict[str, VerifiedRule] = {}
        to_remove = []

        for rule in self._rules.values():
            h = rule.code_hash
            if h in seen_hashes:
                existing = seen_hashes[h]
                # Keep the one with higher accuracy
                if rule.accuracy > existing.accuracy:
                    to_remove.append(existing.rule_id)
                    seen_hashes[h] = rule
                else:
                    to_remove.append(rule.rule_id)
            else:
                seen_hashes[h] = rule

        removed = 0
        for rule_id in to_remove:
            if rule_id in self._rules:
                del self._rules[rule_id]
                removed += 1

        if removed > 0:
            self._save()

        return removed

    def _save(self) -> None:
        """Save rules to disk."""
        if not self.path:
            return

        os.makedirs(
            os.path.dirname(self.path) if os.path.dirname(self.path) else ".",
            exist_ok=True,
        )

        data = {
            "rules": [r.to_dict() for r in self._rules.values()]
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        """Load rules from disk."""
        if not self.path or not os.path.exists(self.path):
            return

        with open(self.path, "r") as f:
            data = json.load(f)

        for rule_data in data.get("rules", []):
            rule = VerifiedRule.from_dict(rule_data)
            self._rules[rule.rule_id] = rule
