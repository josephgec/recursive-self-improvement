"""ARC domain: grid representation, task loading, evaluation."""

from src.arc.grid import Grid, GridDiff, ARCTask
from src.arc.loader import ARCLoader
from src.arc.evaluator import ProgramEvaluator
from src.arc.visualizer import GridVisualizer
from src.arc.difficulty import estimate_difficulty

__all__ = [
    "Grid", "GridDiff", "ARCTask",
    "ARCLoader", "ProgramEvaluator",
    "GridVisualizer", "estimate_difficulty",
]
