"""Analyze the complexity ceiling -- where the agent loses comprehension."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SigmoidFit:
    """Parameters of a fitted sigmoid curve: y = L / (1 + exp(-k*(x - x0))) + b."""

    L: float = 1.0  # max value
    k: float = -0.01  # steepness (negative = decreasing)
    x0: float = 100.0  # midpoint (the "ceiling")
    b: float = 0.0  # baseline
    r_squared: float = 0.0  # goodness of fit


@dataclass
class CeilingAnalysis:
    """Results of complexity ceiling analysis."""

    ceiling_estimate: Optional[float] = None
    sigmoid_fit: Optional[SigmoidFit] = None
    cliff_detected: bool = False
    cliff_location: Optional[float] = None
    decline_characterization: str = "unknown"  # 'gradual', 'cliff', 'plateau_then_cliff'
    safe_operating_range: Optional[Tuple[float, float]] = None
    data_points: int = 0
    complexity_range: Optional[Tuple[float, float]] = None


class ComplexityCeilingAnalyzer:
    """Analyze how performance degrades with increasing complexity."""

    def analyze(
        self,
        complexities: List[float],
        accuracies: List[float],
        comprehension_scores: Optional[List[float]] = None,
    ) -> CeilingAnalysis:
        """Run full ceiling analysis.

        Args:
            complexities: Code complexity values.
            accuracies: Corresponding accuracy values.
            comprehension_scores: Optional comprehension probe scores.

        Returns:
            CeilingAnalysis with all findings.
        """
        result = CeilingAnalysis()
        result.data_points = len(complexities)

        if len(complexities) < 3:
            return result

        result.complexity_range = (min(complexities), max(complexities))

        # Fit sigmoid
        result.sigmoid_fit = self.fit_sigmoid(complexities, accuracies)

        # Estimate ceiling from sigmoid midpoint
        if result.sigmoid_fit and result.sigmoid_fit.r_squared > 0.3:
            result.ceiling_estimate = result.sigmoid_fit.x0

        # Detect cliff
        cliff = self.detect_cliff(complexities, accuracies)
        result.cliff_detected = cliff is not None
        result.cliff_location = cliff

        # Characterize decline
        result.decline_characterization = self.characterize_decline(
            complexities, accuracies
        )

        # Compute safe range
        result.safe_operating_range = self.compute_safe_operating_range(
            complexities, accuracies
        )

        # If no sigmoid ceiling, use cliff or simple threshold
        if result.ceiling_estimate is None:
            if cliff is not None:
                result.ceiling_estimate = cliff
            else:
                # Find where accuracy first drops below 0.5
                for c, a in sorted(zip(complexities, accuracies)):
                    if a < 0.5:
                        result.ceiling_estimate = c
                        break

        return result

    def fit_sigmoid(
        self, complexities: List[float], accuracies: List[float]
    ) -> Optional[SigmoidFit]:
        """Fit a decreasing sigmoid to complexity vs accuracy data.

        Uses simple grid search since scipy.optimize may not be available
        in all environments.
        """
        if len(complexities) < 5:
            return None

        x = np.array(complexities, dtype=float)
        y = np.array(accuracies, dtype=float)

        # Normalize x for numerical stability
        x_min, x_max = x.min(), x.max()
        if x_max == x_min:
            return None

        best_fit = None
        best_r2 = -float("inf")

        # Grid search over parameters
        L_range = [max(y) - min(y)] if max(y) != min(y) else [1.0]
        b_range = [min(y)]

        for L in L_range:
            for b in b_range:
                for x0 in np.linspace(x_min, x_max, 10):
                    for k in [-0.001, -0.005, -0.01, -0.02, -0.05, -0.1]:
                        # Compute predicted values
                        with np.errstate(over="ignore"):
                            exponent = -k * (x - x0)
                            exponent = np.clip(exponent, -500, 500)
                            y_pred = L / (1 + np.exp(exponent)) + b

                        # R-squared
                        ss_res = np.sum((y - y_pred) ** 2)
                        ss_tot = np.sum((y - np.mean(y)) ** 2)
                        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

                        if r2 > best_r2:
                            best_r2 = r2
                            best_fit = SigmoidFit(
                                L=float(L),
                                k=float(k),
                                x0=float(x0),
                                b=float(b),
                                r_squared=float(r2),
                            )

        return best_fit

    def detect_cliff(
        self,
        complexities: List[float],
        accuracies: List[float],
        threshold: float = 0.2,
    ) -> Optional[float]:
        """Detect a sudden drop (cliff) in accuracy.

        Args:
            complexities: Complexity values.
            accuracies: Accuracy values.
            threshold: Minimum drop to consider a cliff.

        Returns:
            Complexity at which the cliff occurs, or None.
        """
        if len(complexities) < 3:
            return None

        # Sort by complexity
        pairs = sorted(zip(complexities, accuracies))
        sorted_c = [p[0] for p in pairs]
        sorted_a = [p[1] for p in pairs]

        # Look for the largest single-step drop
        max_drop = 0.0
        cliff_idx = None

        for i in range(len(sorted_a) - 1):
            drop = sorted_a[i] - sorted_a[i + 1]
            if drop > max_drop:
                max_drop = drop
                cliff_idx = i

        if max_drop >= threshold and cliff_idx is not None:
            return sorted_c[cliff_idx]

        return None

    def characterize_decline(
        self, complexities: List[float], accuracies: List[float]
    ) -> str:
        """Characterize how accuracy declines with complexity.

        Returns:
            One of: 'gradual', 'cliff', 'plateau_then_cliff', 'no_decline', 'insufficient_data'.
        """
        if len(complexities) < 5:
            return "insufficient_data"

        pairs = sorted(zip(complexities, accuracies))
        sorted_a = [p[1] for p in pairs]

        # Check for no decline
        if sorted_a[-1] >= sorted_a[0] * 0.9:
            return "no_decline"

        # Check for cliff
        cliff = self.detect_cliff(complexities, accuracies, threshold=0.2)

        # Check for plateau then cliff
        mid = len(sorted_a) // 2
        first_half_var = max(sorted_a[:mid]) - min(sorted_a[:mid]) if mid > 0 else 0
        second_half_drop = sorted_a[mid] - sorted_a[-1] if mid > 0 else 0

        if first_half_var < 0.1 and second_half_drop > 0.2:
            return "plateau_then_cliff"

        if cliff is not None:
            return "cliff"

        return "gradual"

    def compute_safe_operating_range(
        self,
        complexities: List[float],
        accuracies: List[float],
        min_accuracy: float = 0.7,
    ) -> Optional[Tuple[float, float]]:
        """Compute the complexity range where accuracy stays above a threshold.

        Args:
            complexities: Complexity values.
            accuracies: Accuracy values.
            min_accuracy: Minimum acceptable accuracy.

        Returns:
            (min_complexity, max_complexity) range, or None.
        """
        if not complexities:
            return None

        pairs = sorted(zip(complexities, accuracies))
        safe = [c for c, a in pairs if a >= min_accuracy]

        if not safe:
            return None

        return (min(safe), max(safe))

    def plot_ceiling_analysis(
        self,
        analysis: CeilingAnalysis,
        complexities: List[float],
        accuracies: List[float],
        output_path: Optional[str] = None,
    ) -> Optional[Any]:
        """Plot the complexity ceiling analysis."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        # Scatter plot of data
        ax.scatter(complexities, accuracies, alpha=0.5, label="Observed", color="blue")

        # Plot sigmoid fit if available
        if analysis.sigmoid_fit and analysis.sigmoid_fit.r_squared > 0.1:
            sf = analysis.sigmoid_fit
            x_smooth = np.linspace(min(complexities), max(complexities), 200)
            with np.errstate(over="ignore"):
                exponent = -sf.k * (x_smooth - sf.x0)
                exponent = np.clip(exponent, -500, 500)
                y_smooth = sf.L / (1 + np.exp(exponent)) + sf.b
            ax.plot(x_smooth, y_smooth, "r-", label=f"Sigmoid fit (R²={sf.r_squared:.2f})")

        # Mark ceiling
        if analysis.ceiling_estimate is not None:
            ax.axvline(
                x=analysis.ceiling_estimate,
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"Ceiling: {analysis.ceiling_estimate:.0f}",
            )

        # Mark safe range
        if analysis.safe_operating_range:
            lo, hi = analysis.safe_operating_range
            ax.axvspan(lo, hi, alpha=0.1, color="green", label="Safe range")

        ax.set_xlabel("Code Complexity (AST nodes)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Complexity Ceiling Analysis")
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150)

        return fig
