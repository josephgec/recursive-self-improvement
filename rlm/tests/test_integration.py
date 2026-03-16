"""Integration tests: end-to-end scenarios."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.core.completion import completion, RLMCompletionAPI
from src.core.session import RLMSession, SessionResult
from src.core.config import load_config, merge_configs, DEFAULT_CONFIG
from src.recursion.depth_controller import DepthController
from src.strategies.detector import StrategyDetector, Strategy
from src.strategies.trajectory_logger import TrajectoryLogger
from src.evaluation.synthetic import SyntheticLongContextGenerator
from src.evaluation.metrics import RLMMetrics
from src.analysis.report import generate_report
from src.analysis.trajectory_analysis import (
    strategy_by_task_type,
    efficiency_by_strategy,
    example_trajectories,
    strategy_by_context_size,
)
from src.analysis.cost_analysis import token_usage, latency_analysis, cost_per_query
from src.analysis.depth_analysis import recursion_depth_distribution
from src.analysis.comparison import RLMComparisonAnalyzer
from src.recursion.aggregator import ResultAggregator
from src.recursion.registry import SessionRegistry
from tests.conftest import (
    MockLLM,
    MockLLMImmediate,
    MockLLMNeverFinal,
    MockLLMChunking,
    MockLLMSubQuery,
)


class TestNeedleInHaystack:
    """End-to-end: find a needle in a haystack using grep strategy."""

    def test_needle_search(self):
        gen = SyntheticLongContextGenerator()
        task = gen.needle_in_haystack(
            needle="SECRET_PASSWORD: alpha_bravo_42",
            haystack_size=5000,
            position=0.5,
        )

        llm = MockLLM()
        result = completion(
            prompt="Find the SECRET_PASSWORD in the text",
            context=task.context,
            llm=llm,
        )

        assert isinstance(result, str)
        assert len(result) > 0

    def test_needle_strategy_detection(self):
        gen = SyntheticLongContextGenerator()
        task = gen.needle_in_haystack()

        llm = MockLLM()
        session = RLMSession(llm=llm, max_iterations=5)
        sr = session.run(query="Find the secret", context=task.context)

        detector = StrategyDetector()
        cls = detector.classify(sr.trajectory)
        # MockLLM uses peek then grep
        assert cls.strategy in (Strategy.PEEK_THEN_GREP, Strategy.DIRECT, Strategy.ITERATIVE_REFINEMENT)


class TestCountingMapReduce:
    """End-to-end: counting task using map-reduce (chunking) strategy."""

    def test_counting_with_chunks(self):
        gen = SyntheticLongContextGenerator()
        task = gen.counting_task(target_word="specific_marker", count=7)

        llm = MockLLMChunking()
        session = RLMSession(llm=llm, max_iterations=5)
        sr = session.run(query=task.query, context=task.context)

        assert sr.result is not None
        assert sr.total_iterations >= 1

    def test_chunking_strategy_detected(self):
        llm = MockLLMChunking()
        session = RLMSession(llm=llm, max_iterations=5)
        sr = session.run(query="Count words", context="word " * 100)

        detector = StrategyDetector()
        cls = detector.classify(sr.trajectory)
        assert cls.strategy == Strategy.MAP_REDUCE


class TestBudgetEnforcement:
    """End-to-end: budget enforcement prevents infinite loops."""

    def test_budget_enforced(self):
        llm = MockLLMNeverFinal()
        session = RLMSession(llm=llm, max_iterations=3, forced_final=True)
        sr = session.run(query="test", context="test context")

        assert sr.total_iterations == 3
        assert sr.forced_final
        assert sr.result is not None

    def test_budget_with_no_forced(self):
        llm = MockLLMNeverFinal()
        session = RLMSession(llm=llm, max_iterations=2, forced_final=False)
        sr = session.run(query="test", context="test context")

        assert sr.total_iterations == 2
        assert sr.result is None


class TestSubQueryIntegration:
    """End-to-end: sub-query spawning."""

    def test_sub_query_session(self):
        llm = MockLLMSubQuery()
        dc = DepthController(max_depth=3, max_iterations=5)
        session = RLMSession(llm=llm, max_iterations=5, depth_controller=dc)
        sr = session.run(
            query="Find the secret in this long text",
            context="The secret is: HIDDEN_42\n" + "filler " * 200,
        )

        assert sr.result is not None
        assert sr.total_iterations >= 1


class TestTrajectoryLogging:
    """End-to-end: trajectory logging and export."""

    def test_log_and_export(self):
        llm = MockLLM()
        session = RLMSession(llm=llm, max_iterations=5)
        sr = session.run(query="test", context="test context here")

        logger = TrajectoryLogger()
        record = logger.log_session(sr)
        assert record.total_iterations == sr.total_iterations

        exported = logger.export_trajectory()
        assert "steps" in exported
        assert exported["total_iterations"] == sr.total_iterations

    def test_export_all(self):
        llm = MockLLM()
        logger = TrajectoryLogger()
        for i in range(3):
            llm.reset()
            session = RLMSession(llm=llm, max_iterations=5)
            sr = session.run(query=f"test {i}", context="ctx")
            logger.log_session(sr)

        all_exported = logger.export_all()
        assert len(all_exported) == 3


class TestAnalysis:
    """End-to-end: analysis and reporting."""

    def _make_results(self, n: int = 5) -> list[SessionResult]:
        results = []
        for i in range(n):
            llm = MockLLM()
            session = RLMSession(llm=llm, max_iterations=5)
            sr = session.run(query=f"Find item {i}", context=f"item {i} value {i * 10}")
            results.append(sr)
        return results

    def test_trajectory_analysis(self):
        results = self._make_results(3)
        task_types = ["retrieval"] * 3

        sbt = strategy_by_task_type(results, task_types)
        assert "retrieval" in sbt

        eff = efficiency_by_strategy(results)
        assert len(eff) > 0

        examples = example_trajectories(results, max_examples=2)
        assert len(examples) <= 2

    def test_strategy_by_context_size(self):
        results = self._make_results(3)
        sizes = [100, 5000, 50000]
        sbc = strategy_by_context_size(results, sizes)
        assert len(sbc) > 0

    def test_cost_analysis(self):
        results = self._make_results(3)
        tokens = token_usage(results)
        assert tokens["total_tokens"] > 0

        latency = latency_analysis(results)
        assert latency["avg_elapsed"] >= 0

        cost = cost_per_query(results)
        assert cost["avg_cost"] >= 0

    def test_depth_analysis(self):
        results = self._make_results(3)
        dist = recursion_depth_distribution(results)
        assert dist["total_sessions"] == 3

    def test_comparison_analyzer(self):
        results_a = self._make_results(2)
        results_b = self._make_results(2)
        expected = ["0", "10"]

        analyzer = RLMComparisonAnalyzer()
        acc_comp = analyzer.accuracy_comparison(results_a, results_b, expected)
        assert "difference" in acc_comp

        cost_comp = analyzer.cost_comparison(results_a, results_b)
        assert "difference" in cost_comp

    def test_context_scaling_comparison(self):
        results_a = {1000: self._make_results(1)}
        results_b = {1000: self._make_results(1)}
        expected = {1000: ["0"]}

        analyzer = RLMComparisonAnalyzer()
        scaling = analyzer.context_scaling_comparison(
            results_a, results_b, expected
        )
        assert "sizes" in scaling

    def test_generate_report(self):
        results = self._make_results(3)
        expected = ["0", "10", "20"]
        task_types = ["retrieval", "aggregation", "reasoning"]

        report = generate_report(results, expected, task_types)
        assert "# RLM Evaluation Report" in report
        assert "Summary" in report
        assert "Cost" in report


class TestConfig:
    def test_load_default(self):
        config = load_config()
        assert "model" in config
        assert "max_iterations" in config

    def test_merge_configs(self):
        override = {"max_iterations": 20, "recursion": {"max_depth": 5}}
        merged = merge_configs(DEFAULT_CONFIG, override)
        assert merged["max_iterations"] == 20
        assert merged["recursion"]["max_depth"] == 5
        # Original values should be preserved
        assert merged["model"] == "mock"

    def test_load_nonexistent_file(self):
        config = load_config("/nonexistent/path.yaml")
        assert config == DEFAULT_CONFIG


class TestAggregator:
    def test_concatenate(self):
        results = ["a", "b", "c"]
        assert ResultAggregator.concatenate(results) == "a\nb\nc"

    def test_concatenate_custom_separator(self):
        results = ["x", "y"]
        assert ResultAggregator.concatenate(results, ", ") == "x, y"

    def test_vote(self):
        results = ["a", "b", "a", "a", "b"]
        assert ResultAggregator.vote(results) == "a"

    def test_vote_empty(self):
        assert ResultAggregator.vote([]) is None

    def test_merge_structured(self):
        results = [{"a": 1}, {"b": 2}]
        merged = ResultAggregator.merge_structured(results)
        assert merged["a"] == 1
        assert merged["b"] == 2

    def test_merge_with_strings(self):
        results = ["hello", {"a": 1}]
        merged = ResultAggregator.merge_structured(results)
        assert "a" in merged

    def test_merge_with_json_string(self):
        results = ['{"key": "val"}']
        merged = ResultAggregator.merge_structured(results)
        assert merged["key"] == "val"

    def test_inject_helpers(self):
        repl: dict = {}
        ResultAggregator.inject_helpers(repl)
        assert "aggregate_concat" in repl
        assert "aggregate_vote" in repl
        assert "aggregate_merge" in repl


class TestRegistry:
    def test_register_and_get(self):
        reg = SessionRegistry()
        sid = reg.register("test query", depth=0)
        info = reg.get(sid)
        assert info is not None
        assert info.query == "test query"
        assert info.depth == 0

    def test_list_active(self):
        reg = SessionRegistry()
        reg.register("q1")
        reg.register("q2")
        active = reg.list_active()
        assert len(active) == 2

    def test_update_status(self):
        reg = SessionRegistry()
        sid = reg.register("q1")
        reg.update_status(sid, "completed", result="done")
        info = reg.get(sid)
        assert info.status == "completed"
        assert info.result == "done"

    def test_parent_child(self):
        reg = SessionRegistry()
        parent = reg.register("parent")
        child = reg.register("child", depth=1, parent_id=parent)
        parent_info = reg.get(parent)
        assert child in parent_info.children

    def test_get_tree(self):
        reg = SessionRegistry()
        parent = reg.register("parent")
        reg.register("child1", depth=1, parent_id=parent)
        reg.register("child2", depth=1, parent_id=parent)
        tree = reg.get_tree(parent)
        assert tree["id"] == parent
        assert len(tree["children"]) == 2

    def test_get_all_trees(self):
        reg = SessionRegistry()
        reg.register("root1")
        reg.register("root2")
        trees = reg.get_tree()
        assert len(trees) == 2

    def test_get_nonexistent(self):
        reg = SessionRegistry()
        assert reg.get("nonexistent") is None

    def test_update_nonexistent(self):
        reg = SessionRegistry()
        # Should not raise
        reg.update_status("nonexistent", "completed")


class TestUtilityModules:
    """Test utility functions."""

    def test_token_counter(self):
        from src.utils.token_counter import estimate_tokens, count_tokens
        assert estimate_tokens("") == 0
        assert estimate_tokens("hello world") > 0
        assert count_tokens("") == 0
        assert count_tokens("hello world") > 0

    def test_context_generators(self):
        from src.utils.context_generators import generate_haystack, generate_document_collection
        haystack = generate_haystack("needle", haystack_size=1000)
        assert "needle" in haystack

        docs = generate_document_collection(num_docs=5)
        assert len(docs) == 5
        assert all("title" in d and "body" in d for d in docs)


class TestHelpersDirectly:
    """Test helper functions directly."""

    def test_peek_helpers(self):
        from src.strategies.peek_helpers import peek, make_peek
        repl = {"CONTEXT": "Hello World Test"}
        assert peek(repl, 0, 5) == "Hello"

        fn = make_peek(repl)
        assert fn(6, 5) == "World"

    def test_grep_helpers(self):
        from src.strategies.grep_helpers import grep, search, make_grep
        repl = {"CONTEXT": "line1 hello\nline2 world\nline3 hello again"}
        results = grep(repl, "hello")
        assert len(results) == 2

        results = search(repl, "world")
        assert len(results) == 1

        grep_fn, search_fn = make_grep(repl)
        assert len(grep_fn("hello")) == 2
        assert len(search_fn("world")) == 1

    def test_grep_with_context_lines(self):
        from src.strategies.grep_helpers import grep
        repl = {"CONTEXT": "a\nb\nc\nd\ne"}
        results = grep(repl, "c", context_lines=1)
        assert len(results) == 1
        assert "b" in results[0]
        assert "d" in results[0]

    def test_grep_invalid_regex(self):
        from src.strategies.grep_helpers import grep
        repl = {"CONTEXT": "test [bracket"}
        results = grep(repl, "[bracket")
        assert len(results) >= 0  # Should not raise

    def test_chunk_helpers(self):
        from src.strategies.chunk_helpers import chunk, count_lines, make_chunk
        repl = {"CONTEXT": "a\nb\nc"}
        assert count_lines(repl) == 3

        chunks = chunk(repl, chunk_size=2, overlap=0)
        assert len(chunks) > 0

        chunk_fn, cl_fn = make_chunk(repl)
        assert cl_fn() == 3

    def test_chunk_empty_context(self):
        from src.strategies.chunk_helpers import count_lines
        repl = {"CONTEXT": ""}
        assert count_lines(repl) == 0


class TestPromptsAndTemplates:
    """Test prompt building."""

    def test_root_prompt(self):
        from src.prompts.root_prompt import RootPromptBuilder
        from src.core.context_loader import ContextMeta
        builder = RootPromptBuilder()
        meta = ContextMeta(
            context_type="str", size_chars=100, size_tokens=25, num_lines=5
        )
        prompt = builder.build(meta)
        assert "REPL" in prompt or "CONTEXT" in prompt

    def test_root_prompt_sub_query(self):
        from src.prompts.root_prompt import RootPromptBuilder
        from src.core.context_loader import ContextMeta
        builder = RootPromptBuilder()
        meta = ContextMeta(
            context_type="str", size_chars=100, size_tokens=25, num_lines=5
        )
        prompt = builder.build_for_sub_query(meta, "test query")
        assert "test query" in prompt

    def test_sub_prompt(self):
        from src.prompts.sub_prompt import SubPromptBuilder
        from src.core.context_loader import ContextMeta
        builder = SubPromptBuilder()
        meta = ContextMeta(
            context_type="str", size_chars=100, size_tokens=25, num_lines=5
        )
        prompt = builder.build(meta, "sub question", depth=2)
        assert "sub question" in prompt
        assert "2" in prompt

    def test_budget_warning(self):
        from src.prompts.root_prompt import RootPromptBuilder
        from src.prompts.sub_prompt import SubPromptBuilder
        w1 = RootPromptBuilder.budget_warning(2)
        w2 = SubPromptBuilder.budget_warning(1)
        assert "2" in w1
        assert "1" in w2

    def test_templates_exist(self):
        from src.prompts.templates import (
            REPL_INTRO, FINAL_INSTRUCTIONS, SUB_QUERY_INSTRUCTIONS,
            BUDGET_WARNING, ROOT_TEMPLATE, SUB_TEMPLATE,
        )
        assert "CONTEXT" in REPL_INTRO
        assert "FINAL" in FINAL_INSTRUCTIONS
        assert "rlm_sub_query" in SUB_QUERY_INSTRUCTIONS
        assert "{remaining}" in BUDGET_WARNING
        assert "{query}" in ROOT_TEMPLATE
        assert "{depth}" in SUB_TEMPLATE
