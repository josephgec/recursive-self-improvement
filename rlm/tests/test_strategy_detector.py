"""Tests for StrategyDetector."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.strategies.detector import StrategyDetector, Strategy, StrategyClassification
from src.core.session import TrajectoryStep
from src.core.code_executor import CodeBlockResult


def _make_step(code_blocks: list[str], iteration: int = 1) -> TrajectoryStep:
    """Helper to create a TrajectoryStep."""
    return TrajectoryStep(
        iteration=iteration,
        llm_response="",
        code_blocks=code_blocks,
        execution_results=[],
        has_final=False,
    )


class TestStrategyDetector:
    def test_empty_trajectory(self):
        detector = StrategyDetector()
        result = detector.classify([])
        assert result.strategy == Strategy.DIRECT

    def test_direct_strategy(self):
        detector = StrategyDetector()
        trajectory = [_make_step(["x = 42\nFINAL('42')"])]
        result = detector.classify(trajectory)
        assert result.strategy == Strategy.DIRECT

    def test_peek_then_grep(self):
        detector = StrategyDetector()
        trajectory = [
            _make_step(["preview = peek(0, 100)"]),
            _make_step(["results = grep('pattern')"]),
            _make_step(["FINAL('found')"]),
        ]
        result = detector.classify(trajectory)
        assert result.strategy == Strategy.PEEK_THEN_GREP

    def test_grep_only(self):
        detector = StrategyDetector()
        trajectory = [
            _make_step(["results = search('query')"]),
            _make_step(["FINAL('found')"]),
        ]
        result = detector.classify(trajectory)
        assert result.strategy == Strategy.PEEK_THEN_GREP

    def test_map_reduce(self):
        detector = StrategyDetector()
        trajectory = [
            _make_step(["chunks = chunk(2000)"]),
            _make_step(["for c in chunks: process(c)"]),
            _make_step(["FINAL('aggregated')"]),
        ]
        result = detector.classify(trajectory)
        assert result.strategy == Strategy.MAP_REDUCE

    def test_hierarchical(self):
        detector = StrategyDetector()
        trajectory = [
            _make_step(["r = rlm_sub_query(query='test', context=ctx)"]),
            _make_step(["FINAL(r)"]),
        ]
        result = detector.classify(trajectory)
        assert result.strategy == Strategy.HIERARCHICAL

    def test_hierarchical_with_sub_sessions(self):
        detector = StrategyDetector()
        trajectory = [_make_step(["x = 1"])]
        result = detector.classify(trajectory, sub_sessions=["session1"])
        assert result.strategy == Strategy.HIERARCHICAL

    def test_iterative_refinement(self):
        detector = StrategyDetector()
        trajectory = [
            _make_step(["x = CONTEXT[:100]"], 1),
            _make_step(["y = x.split()"], 2),
            _make_step(["z = len(y)"], 3),
            _make_step(["FINAL(str(z))"], 4),
        ]
        result = detector.classify(trajectory)
        assert result.strategy == Strategy.ITERATIVE_REFINEMENT

    def test_classification_str(self):
        cls = StrategyClassification(
            strategy=Strategy.DIRECT, confidence=0.9, evidence=["test"]
        )
        assert "direct" in str(cls)
        assert "0.90" in str(cls)

    def test_confidence_range(self):
        detector = StrategyDetector()
        trajectory = [_make_step(["peek(0, 100)"]), _make_step(["grep('x')"]),
                       _make_step(["FINAL('y')"])]
        result = detector.classify(trajectory)
        assert 0.0 <= result.confidence <= 1.0

    def test_evidence_non_empty(self):
        detector = StrategyDetector()
        result = detector.classify([_make_step(["x = 1"])])
        assert len(result.evidence) > 0
