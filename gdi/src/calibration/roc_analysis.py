"""ROC curve analysis for GDI threshold calibration."""

from typing import List, Tuple


def compute_roc_curve(
    scores: List[float],
    labels: List[int],
) -> Tuple[List[float], List[float], List[float]]:
    """Compute ROC curve from scores and binary labels.

    Args:
        scores: GDI scores (higher = more drift).
        labels: Binary labels (1 = unhealthy/drifted, 0 = healthy).

    Returns:
        Tuple of (fpr_list, tpr_list, thresholds).
    """
    if not scores or not labels:
        return [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]

    # Sort by score descending
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])

    total_pos = sum(labels)
    total_neg = len(labels) - total_pos

    if total_pos == 0 or total_neg == 0:
        return [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]

    fpr_list = [0.0]
    tpr_list = [0.0]
    thresholds = [paired[0][0] + 0.01]

    tp = 0
    fp = 0

    for score, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / total_pos)
        fpr_list.append(fp / total_neg)
        thresholds.append(score)

    return fpr_list, tpr_list, thresholds


def compute_auc(
    fpr_list: List[float], tpr_list: List[float]
) -> float:
    """Compute Area Under the ROC Curve using trapezoidal rule.

    Args:
        fpr_list: False positive rates.
        tpr_list: True positive rates.

    Returns:
        AUC value in [0, 1].
    """
    auc = 0.0
    for i in range(1, len(fpr_list)):
        dx = fpr_list[i] - fpr_list[i - 1]
        y_avg = (tpr_list[i] + tpr_list[i - 1]) / 2
        auc += dx * y_avg
    return max(0.0, min(1.0, auc))


def find_best_threshold(
    fpr_list: List[float],
    tpr_list: List[float],
    thresholds: List[float],
) -> float:
    """Find the best threshold by maximizing Youden's J statistic.

    J = TPR - FPR (maximized at optimal threshold).

    Args:
        fpr_list: False positive rates.
        tpr_list: True positive rates.
        thresholds: Corresponding threshold values.

    Returns:
        Optimal threshold value.
    """
    best_j = -1.0
    best_threshold = 0.5

    for fpr, tpr, thresh in zip(fpr_list, tpr_list, thresholds):
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_threshold = thresh

    return best_threshold
