"""FINAL protocol for result extraction in the RLM-REPL sandbox."""

from src.protocol.final_functions import FinalProtocol, FinalResult
from src.protocol.detector import FinalDetector
from src.protocol.extractor import ResultExtractor
from src.protocol.aggregator import ResultAggregator
from src.protocol.types import FinalSignal

__all__ = [
    "FinalProtocol",
    "FinalResult",
    "FinalDetector",
    "ResultExtractor",
    "ResultAggregator",
    "FinalSignal",
]
