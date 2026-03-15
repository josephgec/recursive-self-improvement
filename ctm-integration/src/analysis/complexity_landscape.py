"""Complexity landscape analysis across domains."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.bdm.scorer import BDMScorer


@dataclass
class DomainComplexity:
    """Complexity statistics for a single domain."""

    domain: str
    samples: int
    avg_bdm: float
    min_bdm: float
    max_bdm: float
    std_bdm: float
    avg_entropy: float


class ComplexityLandscapeAnalyzer:
    """Analyzes BDM complexity across different domains and data types."""

    def __init__(self, scorer: Optional[BDMScorer] = None) -> None:
        self.scorer = scorer or BDMScorer()

    def map_bdm_across_domains(
        self, domain_data: Dict[str, List[str]]
    ) -> List[DomainComplexity]:
        """Map BDM complexity across multiple domains.

        Args:
            domain_data: Mapping from domain name to list of data samples.

        Returns:
            List of DomainComplexity for each domain.
        """
        results = []

        for domain, samples in domain_data.items():
            if not samples:
                continue

            bdm_scores = []
            entropies = []

            for sample in samples:
                score = self.scorer.score(sample)
                bdm_scores.append(score.bdm_value)

                baselines = self.scorer.compare_to_baselines(sample)
                entropies.append(baselines.get("shannon_entropy", 0.0))

            avg_bdm = sum(bdm_scores) / len(bdm_scores)
            min_bdm = min(bdm_scores)
            max_bdm = max(bdm_scores)

            # Standard deviation
            variance = sum((x - avg_bdm) ** 2 for x in bdm_scores) / len(bdm_scores)
            std_bdm = variance ** 0.5

            avg_entropy = sum(entropies) / len(entropies)

            results.append(
                DomainComplexity(
                    domain=domain,
                    samples=len(samples),
                    avg_bdm=avg_bdm,
                    min_bdm=min_bdm,
                    max_bdm=max_bdm,
                    std_bdm=std_bdm,
                    avg_entropy=avg_entropy,
                )
            )

        return results

    def plot_landscape(
        self,
        domain_complexities: List[DomainComplexity],
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """Plot the complexity landscape across domains.

        Args:
            domain_complexities: List of domain complexity results.
            output_path: Path to save the plot.

        Returns:
            Path to saved plot, or None.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            domains = [dc.domain for dc in domain_complexities]
            avg_bdms = [dc.avg_bdm for dc in domain_complexities]
            std_bdms = [dc.std_bdm for dc in domain_complexities]
            avg_entropies = [dc.avg_entropy for dc in domain_complexities]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # BDM by domain
            x_pos = range(len(domains))
            ax1.bar(x_pos, avg_bdms, yerr=std_bdms, capsize=5, alpha=0.7)
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(domains, rotation=45, ha="right")
            ax1.set_ylabel("Average BDM Score")
            ax1.set_title("BDM Complexity by Domain")
            ax1.grid(True, alpha=0.3)

            # BDM vs Entropy
            ax2.scatter(avg_entropies, avg_bdms, s=100)
            for i, domain in enumerate(domains):
                ax2.annotate(domain, (avg_entropies[i], avg_bdms[i]),
                           fontsize=8, ha="center", va="bottom")
            ax2.set_xlabel("Average Shannon Entropy")
            ax2.set_ylabel("Average BDM Score")
            ax2.set_title("BDM vs. Shannon Entropy")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            if output_path:
                os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
                fig.savefig(output_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                return output_path
            else:
                plt.close(fig)
                return None
        except ImportError:
            return None
