"""Library evolution: pruning, quality measurement, and improvement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.library.rule import VerifiedRule
from src.library.store import RuleStore


@dataclass
class LibraryMetrics:
    """Metrics for evaluating library quality."""

    total_rules: int = 0
    unique_domains: int = 0
    avg_accuracy: float = 0.0
    avg_bdm_score: float = 0.0
    avg_mdl_score: float = 0.0
    coverage: float = 0.0  # fraction of domains with at least one good rule
    quality_score: float = 0.0  # combined metric

    def to_dict(self) -> Dict[str, float]:
        return {
            "total_rules": float(self.total_rules),
            "unique_domains": float(self.unique_domains),
            "avg_accuracy": self.avg_accuracy,
            "avg_bdm_score": self.avg_bdm_score,
            "avg_mdl_score": self.avg_mdl_score,
            "coverage": self.coverage,
            "quality_score": self.quality_score,
        }


class LibraryEvolver:
    """Evolves the rule library over time.

    Manages pruning of low-quality rules and measurement of library quality.
    """

    def __init__(
        self,
        store: Optional[RuleStore] = None,
        min_accuracy: float = 0.5,
        max_rules_per_domain: int = 20,
    ) -> None:
        self.store = store or RuleStore()
        self.min_accuracy = min_accuracy
        self.max_rules_per_domain = max_rules_per_domain
        self._metrics_history: List[LibraryMetrics] = []

    def evolve(self, new_rules: List[VerifiedRule]) -> LibraryMetrics:
        """Add new rules and evolve the library.

        Args:
            new_rules: New rules to consider adding.

        Returns:
            Updated library metrics.
        """
        # Add qualifying rules
        for rule in new_rules:
            if rule.accuracy >= self.min_accuracy:
                self.store.add(rule)

        # Deduplicate
        self.store.deduplicate()

        # Prune
        self.prune()

        # Measure quality
        metrics = self.measure_library_quality()
        self._metrics_history.append(metrics)

        return metrics

    def prune(self) -> int:
        """Prune low-quality rules from the library.

        Removes rules below minimum accuracy threshold and enforces
        per-domain limits by keeping only the best rules.

        Returns:
            Number of rules pruned.
        """
        all_rules = self.store.list_all()
        to_remove = []

        # Remove below-threshold rules
        for rule in all_rules:
            if rule.accuracy < self.min_accuracy:
                to_remove.append(rule.rule_id)

        # Enforce per-domain limits
        domains: Dict[str, List[VerifiedRule]] = {}
        for rule in all_rules:
            if rule.rule_id in to_remove:
                continue
            if rule.domain not in domains:
                domains[rule.domain] = []
            domains[rule.domain].append(rule)

        for domain, rules in domains.items():
            if len(rules) > self.max_rules_per_domain:
                # Sort by accuracy (descending), then by MDL (ascending)
                sorted_rules = sorted(
                    rules,
                    key=lambda r: (-r.accuracy, r.mdl_score),
                )
                # Remove excess rules
                for rule in sorted_rules[self.max_rules_per_domain:]:
                    to_remove.append(rule.rule_id)

        pruned = 0
        for rule_id in to_remove:
            if self.store.remove(rule_id):
                pruned += 1

        return pruned

    def measure_library_quality(
        self, known_domains: Optional[List[str]] = None
    ) -> LibraryMetrics:
        """Measure the quality of the current library.

        Args:
            known_domains: Optional list of all known domains for coverage calculation.

        Returns:
            LibraryMetrics.
        """
        all_rules = self.store.list_all()

        if not all_rules:
            return LibraryMetrics()

        total = len(all_rules)
        domains = set(r.domain for r in all_rules)
        accuracies = [r.accuracy for r in all_rules]
        bdm_scores = [r.bdm_score for r in all_rules]
        mdl_scores = [r.mdl_score for r in all_rules]

        avg_accuracy = sum(accuracies) / total
        avg_bdm = sum(bdm_scores) / total if bdm_scores else 0.0
        avg_mdl = sum(mdl_scores) / total if mdl_scores else 0.0

        # Coverage: fraction of known domains with at least one rule above threshold
        if known_domains:
            covered = sum(
                1 for d in known_domains
                if any(r.domain == d and r.accuracy >= self.min_accuracy for r in all_rules)
            )
            coverage = covered / len(known_domains)
        else:
            coverage = 1.0  # assume full coverage if no domain list provided

        # Quality score: weighted combination
        quality = avg_accuracy * 0.5 + coverage * 0.3 + (1.0 / (1.0 + avg_mdl)) * 0.2

        return LibraryMetrics(
            total_rules=total,
            unique_domains=len(domains),
            avg_accuracy=avg_accuracy,
            avg_bdm_score=avg_bdm,
            avg_mdl_score=avg_mdl,
            coverage=coverage,
            quality_score=quality,
        )

    @property
    def metrics_history(self) -> List[LibraryMetrics]:
        """Return the history of library metrics."""
        return list(self._metrics_history)
