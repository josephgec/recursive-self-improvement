"""Attention subsystem: head extraction, specialization, reward correlation, role tracking."""

from src.attention.head_extractor import HeadExtractor
from src.attention.specialization import HeadSpecializationTracker, HeadShift, HeadRoleChange, HeadTrackingResult
from src.attention.reward_correlation import RewardCorrelationDetector, RewardCorrelatedHead
from src.attention.role_tracker import HeadRoleTracker
from src.attention.dead_head_detector import DeadHeadDetector

__all__ = [
    "HeadExtractor",
    "HeadSpecializationTracker", "HeadShift", "HeadRoleChange", "HeadTrackingResult",
    "RewardCorrelationDetector", "RewardCorrelatedHead",
    "HeadRoleTracker",
    "DeadHeadDetector",
]
