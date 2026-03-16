"""Benchmark task definitions and loaders."""

from src.benchmarks.task import EvalTask, EvalResult
from src.benchmarks.registry import BenchmarkRegistry
from src.benchmarks.oolong import OOLONGBenchmark
from src.benchmarks.locodiff import LoCoDiffBenchmark
from src.benchmarks.synthetic import SyntheticTaskGenerator

__all__ = [
    "EvalTask",
    "EvalResult",
    "BenchmarkRegistry",
    "OOLONGBenchmark",
    "LoCoDiffBenchmark",
    "SyntheticTaskGenerator",
]
