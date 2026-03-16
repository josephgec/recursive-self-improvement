"""Tests for strategy classifier."""

import pytest

from src.strategies.classifier import StrategyClassifier, StrategyType, StrategyClassification
from src.strategies.code_pattern_detector import CodePatternDetector, CodePattern


class TestCodePatternDetector:
    """Test code pattern detection."""

    def test_detect_peek_pattern(self):
        detector = CodePatternDetector()
        patterns = detector.detect_patterns(["head -100 context.txt"])
        types = [p.pattern_type for p in patterns]
        assert "peek" in types

    def test_detect_grep_pattern(self):
        detector = CodePatternDetector()
        patterns = detector.detect_patterns(["grep -n 'target' file.txt"])
        types = [p.pattern_type for p in patterns]
        assert "grep" in types

    def test_detect_chunk_pattern(self):
        detector = CodePatternDetector()
        patterns = detector.detect_patterns(["split -l 100 file.txt chunk_"])
        types = [p.pattern_type for p in patterns]
        assert "chunk" in types

    def test_detect_sub_query_pattern(self):
        detector = CodePatternDetector()
        patterns = detector.detect_patterns(["python analyze_part1.py"])
        types = [p.pattern_type for p in patterns]
        assert "sub_query" in types

    def test_detect_loop_pattern(self):
        detector = CodePatternDetector()
        patterns = detector.detect_patterns(["for f in chunk_*; do process $f; done"])
        types = [p.pattern_type for p in patterns]
        assert "loop" in types

    def test_detect_aggregate_pattern(self):
        detector = CodePatternDetector()
        patterns = detector.detect_patterns(["result = aggregate(partial_results)"])
        types = [p.pattern_type for p in patterns]
        assert "aggregate" in types

    def test_detect_final_pattern(self):
        detector = CodePatternDetector()
        patterns = detector.detect_patterns(["echo 'final answer'"])
        types = [p.pattern_type for p in patterns]
        assert "final" in types

    def test_pattern_sequence(self):
        detector = CodePatternDetector()
        trajectory = [
            "head -100 file.txt",
            "grep 'target' file.txt",
            "echo 'answer'",
        ]
        seq = detector.pattern_sequence(trajectory)
        assert "peek" in seq
        assert "grep" in seq
        assert "final" in seq

    def test_has_pattern(self):
        detector = CodePatternDetector()
        trajectory = ["grep 'x' file.txt"]
        assert detector.has_pattern(trajectory, "grep")
        assert not detector.has_pattern(trajectory, "chunk")

    def test_pattern_counts(self):
        detector = CodePatternDetector()
        trajectory = [
            "grep 'a' file.txt",
            "grep 'b' file.txt",
            "echo 'result'",
        ]
        counts = detector.pattern_counts(trajectory)
        assert counts.get("grep", 0) == 2
        assert counts.get("final", 0) == 1

    def test_empty_trajectory(self):
        detector = CodePatternDetector()
        patterns = detector.detect_patterns([])
        assert len(patterns) == 0


class TestStrategyClassifier:
    """Test strategy classification for each of the 6 strategies."""

    def test_classify_direct(self):
        classifier = StrategyClassifier()
        trajectory = [
            "cat context.txt",
            "echo 'answer'",
        ]
        result = classifier.classify(trajectory)
        assert result.strategy == StrategyType.DIRECT
        assert result.confidence > 0.3

    def test_classify_peek_then_grep(self):
        classifier = StrategyClassifier()
        trajectory = [
            "# Peek at structure\nhead -100 context.txt",
            "# Search for info\ngrep -n 'target' context.txt",
            "# Read section\nsed -n '50,60p' context.txt",
            "echo 'answer'",
        ]
        result = classifier.classify(trajectory)
        assert result.strategy == StrategyType.PEEK_THEN_GREP
        assert result.confidence > 0.5

    def test_classify_iterative_search(self):
        classifier = StrategyClassifier()
        trajectory = [
            "grep -c 'target' context.txt",
            "for line in context: if target in line: count += 1",
            "while more_data: count += scan_next()",
            "echo 'count'",
        ]
        result = classifier.classify(trajectory)
        assert result.strategy == StrategyType.ITERATIVE_SEARCH
        assert result.confidence > 0.5

    def test_classify_map_reduce(self):
        classifier = StrategyClassifier()
        trajectory = [
            "split -l 100 context.txt chunk_",
            "for f in chunk_*; do grep -c 'data' $f; done",
            "for chunk in chunks: process(chunk)",
            "result = aggregate(partial_results)",
            "echo 'total'",
        ]
        result = classifier.classify(trajectory)
        assert result.strategy == StrategyType.MAP_REDUCE
        assert result.confidence > 0.5

    def test_classify_hierarchical(self):
        classifier = StrategyClassifier()
        trajectory = [
            "head -50 context.txt",
            "python analyze_clue_1.py",
            "python analyze_clue_2.py",
            "python synthesize.py",
            "echo 'result'",
        ]
        result = classifier.classify(trajectory)
        assert result.strategy == StrategyType.HIERARCHICAL
        assert result.confidence > 0.5

    def test_classify_hybrid(self):
        classifier = StrategyClassifier()
        trajectory = [
            "head -100 context.txt",
            "grep 'key' context.txt",
            "split -l 50 context.txt chunk_",
            "python analyze.py",
            "for item in items: process(item)",
            "result = aggregate(all_results)",
            "echo 'final'",
        ]
        result = classifier.classify(trajectory)
        # Hybrid should be detected when multiple strategies overlap
        assert result.strategy in (StrategyType.HYBRID, StrategyType.MAP_REDUCE,
                                    StrategyType.HIERARCHICAL)

    def test_classification_has_evidence(self):
        classifier = StrategyClassifier()
        trajectory = ["head -100 file.txt", "grep 'x' file.txt"]
        result = classifier.classify(trajectory)
        assert isinstance(result.evidence, dict)
        assert isinstance(result.pattern_sequence, list)

    def test_classify_batch(self):
        classifier = StrategyClassifier()
        trajectories = [
            ["cat file.txt", "echo 'a'"],
            ["head -10 file.txt", "grep 'x' file.txt"],
        ]
        results = classifier.classify_batch(trajectories)
        assert len(results) == 2
        assert all(isinstance(r, StrategyClassification) for r in results)

    def test_strategy_name_property(self):
        classifier = StrategyClassifier()
        trajectory = ["cat file.txt", "echo 'answer'"]
        result = classifier.classify(trajectory)
        assert result.strategy_name == result.strategy.value

    def test_empty_trajectory_defaults_to_direct(self):
        classifier = StrategyClassifier()
        result = classifier.classify([])
        assert result.strategy == StrategyType.DIRECT
