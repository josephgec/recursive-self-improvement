"""Feedback loop: feeding synthesis results back to improve generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.library.rule import VerifiedRule
from src.library.store import RuleStore
from src.synthesis.pareto_selector import ScoredRule


class FeedbackLoop:
    """Manages feedback from synthesis results to improve future generation.

    Tracks successful patterns, failure patterns, and provides context
    for the next generation round.
    """

    def __init__(self, store: Optional[RuleStore] = None) -> None:
        self.store = store or RuleStore()
        self._successful_patterns: List[Dict[str, Any]] = []
        self._failure_patterns: List[Dict[str, Any]] = []
        self._iteration_history: List[Dict[str, Any]] = []

    def feed_synthesis_results(
        self,
        scored_rules: List[ScoredRule],
        iteration: int = 0,
    ) -> None:
        """Process synthesis results and extract patterns.

        Args:
            scored_rules: Results from Pareto selection.
            iteration: Current iteration number.
        """
        iteration_data = {
            "iteration": iteration,
            "total_rules": len(scored_rules),
            "pareto_rules": sum(1 for r in scored_rules if r.is_pareto_optimal),
            "best_accuracy": max((r.accuracy for r in scored_rules), default=0.0),
            "avg_complexity": (
                sum(r.bdm_complexity for r in scored_rules) / len(scored_rules)
                if scored_rules else 0.0
            ),
        }
        self._iteration_history.append(iteration_data)

        # Extract patterns from good rules
        for sr in scored_rules:
            if sr.accuracy >= 0.8 and sr.is_pareto_optimal:
                self._successful_patterns.append({
                    "domain": sr.rule.domain,
                    "variant": sr.rule.prompt_variant,
                    "accuracy": sr.accuracy,
                    "complexity": sr.bdm_complexity,
                    "code_snippet": sr.rule.source_code[:200],
                })

                # Add to library
                verified = VerifiedRule(
                    rule_id=sr.rule.rule_id,
                    domain=sr.rule.domain,
                    description=sr.rule.description,
                    source_code=sr.rule.source_code,
                    accuracy=sr.accuracy,
                    bdm_score=sr.bdm_complexity,
                    mdl_score=sr.mdl_score,
                    generation=sr.rule.generation,
                )
                self.store.add(verified)

            elif sr.accuracy < 0.3:
                self._failure_patterns.append({
                    "domain": sr.rule.domain,
                    "variant": sr.rule.prompt_variant,
                    "accuracy": sr.accuracy,
                })

    def get_context_for_generation(self) -> Dict[str, Any]:
        """Get context information for the next generation round.

        Returns:
            Dictionary with context for guiding generation.
        """
        context: Dict[str, Any] = {
            "successful_variants": [],
            "avoid_variants": [],
            "best_domains": [],
            "iteration_count": len(self._iteration_history),
        }

        # Find most successful prompt variants
        variant_successes: Dict[str, int] = {}
        for pattern in self._successful_patterns:
            v = pattern.get("variant", "unknown")
            variant_successes[v] = variant_successes.get(v, 0) + 1

        context["successful_variants"] = sorted(
            variant_successes.keys(),
            key=lambda v: variant_successes[v],
            reverse=True,
        )

        # Find variants to avoid
        variant_failures: Dict[str, int] = {}
        for pattern in self._failure_patterns:
            v = pattern.get("variant", "unknown")
            variant_failures[v] = variant_failures.get(v, 0) + 1

        context["avoid_variants"] = sorted(
            variant_failures.keys(),
            key=lambda v: variant_failures[v],
            reverse=True,
        )

        # Track improvement trend
        if len(self._iteration_history) >= 2:
            recent = self._iteration_history[-1]
            previous = self._iteration_history[-2]
            context["accuracy_improving"] = (
                recent["best_accuracy"] > previous["best_accuracy"]
            )
        else:
            context["accuracy_improving"] = None

        return context

    @property
    def successful_patterns(self) -> List[Dict[str, Any]]:
        return list(self._successful_patterns)

    @property
    def failure_patterns(self) -> List[Dict[str, Any]]:
        return list(self._failure_patterns)

    @property
    def iteration_history(self) -> List[Dict[str, Any]]:
        return list(self._iteration_history)
