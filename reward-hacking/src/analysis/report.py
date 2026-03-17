from __future__ import annotations

"""Full report generation combining all analysis modules."""

from ..eppo.trainer import EPPOEpochResult
from ..bounding.process_reward import ShapedReward
from ..energy.energy_tracker import EnergyMeasurement
from .eppo_analysis import analyze_eppo_training
from .bounding_analysis import analyze_bounding
from .energy_analysis import analyze_energy


def generate_full_report(
    epoch_results: list[EPPOEpochResult] | None = None,
    shaped_rewards: list[ShapedReward] | None = None,
    energy_measurements: list[EnergyMeasurement] | None = None,
) -> str:
    """Generate a full analysis report.

    Args:
        epoch_results: EPPO training epoch results.
        shaped_rewards: Shaped reward history.
        energy_measurements: Energy measurement history.

    Returns:
        Markdown report string.
    """
    lines = ["# Reward Hacking Defense Analysis Report", ""]

    # EPPO Analysis
    lines.append("## EPPO Training Analysis")
    lines.append("")
    if epoch_results:
        eppo = analyze_eppo_training(epoch_results)
        lines.append(f"- Epochs: {eppo['num_epochs']}")
        lines.append(f"- Final Entropy: {eppo['entropy']['final']:.4f}")
        lines.append(f"- Entropy Slope: {eppo['entropy']['slope']:.6f}")
        lines.append(f"- Final Beta: {eppo['beta']['final']:.6f}")
        lines.append(f"- Healthy: {'Yes' if eppo['healthy'] else 'No'}")
    else:
        lines.append("No EPPO training data available.")
    lines.append("")

    # Bounding Analysis
    lines.append("## Reward Bounding Analysis")
    lines.append("")
    if shaped_rewards:
        bounding = analyze_bounding(shaped_rewards)
        lines.append(f"- Rewards Processed: {bounding['num_rewards']}")
        lines.append(f"- Clipping Rate: {bounding['clipping']['fraction']:.2%}")
        lines.append(f"- Delta Bounding Rate: {bounding['delta_bounding']['fraction']:.2%}")
        lines.append(f"- Compression Ratio: {bounding['compression_ratio']:.4f}")
        lines.append(f"- Effective: {'Yes' if bounding['effective'] else 'No'}")
    else:
        lines.append("No bounding data available.")
    lines.append("")

    # Energy Analysis
    lines.append("## Energy Analysis")
    lines.append("")
    if energy_measurements:
        energy = analyze_energy(energy_measurements)
        lines.append(f"- Measurements: {energy['num_measurements']}")
        lines.append(f"- Final Energy: {energy['total_energy']['final']:.4f}")
        lines.append(f"- Energy Slope: {energy['total_energy']['slope']:.6f}")
        lines.append(f"- Stable: {'Yes' if energy['stable'] else 'No'}")
        lines.append(f"- Declining: {'Yes' if energy['declining'] else 'No'}")
    else:
        lines.append("No energy data available.")
    lines.append("")

    return "\n".join(lines)
