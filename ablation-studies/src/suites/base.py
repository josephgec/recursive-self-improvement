"""Base classes for ablation suites."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class AblationCondition:
    """A single ablation condition (e.g., 'no_rollback')."""

    name: str
    description: str
    pipeline_config: Dict[str, Any] = field(default_factory=dict)
    is_full: bool = False

    def __repr__(self) -> str:
        return f"AblationCondition(name={self.name!r}, is_full={self.is_full})"


@dataclass
class ConditionRun:
    """Result of running one condition once."""

    condition_name: str
    repetition: int
    accuracy: float
    metrics: Dict[str, float] = field(default_factory=dict)
    seed: int = 0

    @property
    def score(self) -> float:
        """Primary score (accuracy)."""
        return self.accuracy


@dataclass
class PaperAssets:
    """Collection of publication-ready assets."""

    tables: Dict[str, str] = field(default_factory=dict)
    figures: Dict[str, Any] = field(default_factory=dict)
    narrative: str = ""
    appendix: str = ""

    @property
    def has_content(self) -> bool:
        return bool(self.tables or self.figures or self.narrative)


@dataclass
class AblationSuiteResult:
    """Complete results for an ablation suite."""

    suite_name: str
    condition_runs: Dict[str, List[ConditionRun]] = field(default_factory=dict)
    paper_assets: Optional[PaperAssets] = None

    def get_scores(self, condition_name: str) -> List[float]:
        """Get all scores for a condition."""
        if condition_name not in self.condition_runs:
            return []
        return [run.score for run in self.condition_runs[condition_name]]

    def get_mean_score(self, condition_name: str) -> float:
        """Get mean score for a condition."""
        scores = self.get_scores(condition_name)
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def get_all_condition_names(self) -> List[str]:
        """Get all condition names."""
        return list(self.condition_runs.keys())

    def best_condition(self) -> str:
        """Return the condition with the highest mean score."""
        names = self.get_all_condition_names()
        if not names:
            return ""
        return max(names, key=lambda n: self.get_mean_score(n))


class AblationSuite(abc.ABC):
    """Abstract base class for ablation suites."""

    @abc.abstractmethod
    def get_conditions(self) -> List[AblationCondition]:
        """Return all conditions for this suite."""
        ...

    @abc.abstractmethod
    def build_pipeline(self, condition: AblationCondition) -> Dict[str, Any]:
        """Build a pipeline config for the given condition."""
        ...

    @abc.abstractmethod
    def get_benchmarks(self) -> List[str]:
        """Return benchmark names used by this suite."""
        ...

    @abc.abstractmethod
    def get_paper_name(self) -> str:
        """Return the name used in paper references."""
        ...

    @abc.abstractmethod
    def get_key_comparisons(self) -> List[Tuple[str, str]]:
        """Return pairs of conditions for key comparisons."""
        ...

    def get_full_condition(self) -> Optional[AblationCondition]:
        """Return the full (non-ablated) condition."""
        for c in self.get_conditions():
            if c.is_full:
                return c
        return None

    def run(self, repetitions: int = 5, seed: int = 42,
            pipeline_runner: Any = None) -> AblationSuiteResult:
        """Run all conditions with the given runner."""
        from src.execution.runner import AblationRunner

        if pipeline_runner is None:
            runner = AblationRunner()
        else:
            runner = AblationRunner(pipeline_runner=pipeline_runner)
        return runner.run_suite(self, repetitions=repetitions, seed=seed)

    def analyze(self, result: AblationSuiteResult) -> Dict[str, Any]:
        """Run statistical analysis on results."""
        from src.analysis.statistical_tests import PublicationStatistics

        stats = PublicationStatistics()
        analyses = {}
        for a_name, b_name in self.get_key_comparisons():
            a_scores = result.get_scores(a_name)
            b_scores = result.get_scores(b_name)
            if a_scores and b_scores:
                analyses[(a_name, b_name)] = stats.pairwise_comparison(
                    a_scores, b_scores, a_name, b_name
                )
        return analyses

    def generate_paper_assets(self, result: AblationSuiteResult) -> PaperAssets:
        """Generate publication-ready tables, figures, and narrative."""
        from src.publication.latex_tables import LaTeXTableGenerator
        from src.publication.narrative import NarrativeGenerator

        assets = PaperAssets()

        table_gen = LaTeXTableGenerator()
        assets.tables["main_results"] = table_gen.main_results_table(result)

        analyses = self.analyze(result)
        if analyses:
            assets.tables["pairwise"] = table_gen.pairwise_comparison_table(
                list(analyses.values())
            )

        narr_gen = NarrativeGenerator()
        assets.narrative = narr_gen.generate_results_section(
            result, analyses, self.get_paper_name()
        )

        return assets
