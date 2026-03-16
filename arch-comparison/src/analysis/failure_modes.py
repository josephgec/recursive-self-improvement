"""Failure mode analysis: categorize and compare failures across systems."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.evaluation.benchmark_suite import BenchmarkResults


@dataclass
class FailureCategory:
    """A category of failures."""
    name: str
    count: int = 0
    examples: List[Dict[str, Any]] = field(default_factory=list)
    description: str = ""


@dataclass
class FailureModeReport:
    """Report on failure modes."""
    system: str
    categories: List[FailureCategory] = field(default_factory=list)
    total_failures: int = 0
    failure_rate: float = 0.0
    metadata: dict = field(default_factory=dict)


class FailureModeAnalyzer:
    """Analyzes failure modes across different systems."""

    def analyze(self, results: BenchmarkResults) -> Dict[str, FailureModeReport]:
        """Analyze failure modes for all systems.

        Args:
            results: BenchmarkResults from benchmark suite.

        Returns:
            Dict mapping system name to FailureModeReport.
        """
        reports: Dict[str, FailureModeReport] = {}

        for system, gen_result in results.generalization.items():
            all_results = gen_result.in_domain_results + gen_result.out_of_domain_results
            failures = [r for r in all_results if not r.get("correct", True)]
            total = len(all_results)

            if system == "hybrid":
                categories = self._categorize_hybrid_failures(failures)
            elif system == "integrative":
                categories = self._categorize_integrative_failures(failures)
            else:
                categories = self._categorize_prose_failures(failures)

            reports[system] = FailureModeReport(
                system=system,
                categories=categories,
                total_failures=len(failures),
                failure_rate=len(failures) / max(total, 1),
            )

        return reports

    def _categorize_hybrid_failures(
        self, failures: List[Dict[str, Any]]
    ) -> List[FailureCategory]:
        """Categorize hybrid system failures."""
        tool_errors = FailureCategory(
            name="tool_error",
            description="Tool call returned an error or unexpected result",
        )
        integration_errors = FailureCategory(
            name="integration_error",
            description="LLM failed to correctly integrate tool output",
        )
        no_tool_call = FailureCategory(
            name="no_tool_call",
            description="LLM failed to invoke an appropriate tool",
        )

        for f in failures:
            predicted = f.get("predicted", "")
            if "error" in predicted.lower():
                tool_errors.count += 1
                tool_errors.examples.append(f)
            elif "unknown" in predicted.lower():
                no_tool_call.count += 1
                no_tool_call.examples.append(f)
            else:
                integration_errors.count += 1
                integration_errors.examples.append(f)

        return [tool_errors, integration_errors, no_tool_call]

    def _categorize_integrative_failures(
        self, failures: List[Dict[str, Any]]
    ) -> List[FailureCategory]:
        """Categorize integrative system failures."""
        constraint_miss = FailureCategory(
            name="constraint_miss",
            description="Constraint not applied when it should have been",
        )
        wrong_correction = FailureCategory(
            name="wrong_correction",
            description="Constraint applied but corrected to wrong value",
        )
        general_error = FailureCategory(
            name="general_error",
            description="General reasoning error",
        )

        for f in failures:
            predicted = f.get("predicted", "")
            if "unknown" in predicted.lower():
                constraint_miss.count += 1
                constraint_miss.examples.append(f)
            else:
                general_error.count += 1
                general_error.examples.append(f)

        return [constraint_miss, wrong_correction, general_error]

    def _categorize_prose_failures(
        self, failures: List[Dict[str, Any]]
    ) -> List[FailureCategory]:
        """Categorize prose baseline failures."""
        computation = FailureCategory(
            name="computation_error",
            description="Incorrect computation in prose reasoning",
        )
        logic_error = FailureCategory(
            name="logic_error",
            description="Logical error in reasoning chain",
        )
        other = FailureCategory(
            name="other",
            description="Other failure modes",
        )

        for f in failures:
            domain = f.get("domain", "")
            if domain in ("arithmetic", "algebra"):
                computation.count += 1
                computation.examples.append(f)
            elif domain in ("logic", "probability"):
                logic_error.count += 1
                logic_error.examples.append(f)
            else:
                other.count += 1
                other.examples.append(f)

        return [computation, logic_error, other]

    def find_complementary_strengths(
        self, results: BenchmarkResults
    ) -> Dict[str, List[str]]:
        """Find tasks where one system succeeds and another fails.

        Returns dict mapping (system_a, system_b) pairs to task_ids
        where a succeeds and b fails.
        """
        system_results: Dict[str, Dict[str, bool]] = {}
        for system, gen_result in results.generalization.items():
            task_correctness: Dict[str, bool] = {}
            for r in gen_result.in_domain_results + gen_result.out_of_domain_results:
                task_correctness[r["task_id"]] = r["correct"]
            system_results[system] = task_correctness

        complements: Dict[str, List[str]] = {}
        systems = list(system_results.keys())
        for i, a in enumerate(systems):
            for b in systems[i + 1:]:
                key = f"{a}_beats_{b}"
                tasks = [
                    tid for tid in system_results[a]
                    if system_results[a].get(tid, False)
                    and not system_results[b].get(tid, False)
                ]
                complements[key] = tasks

                key_rev = f"{b}_beats_{a}"
                tasks_rev = [
                    tid for tid in system_results[b]
                    if system_results[b].get(tid, False)
                    and not system_results[a].get(tid, False)
                ]
                complements[key_rev] = tasks_rev

        return complements
