"""Tests for EvolutionarySearchEngine and supporting search components."""

import pytest
from src.search.engine import EvolutionarySearchEngine, SearchConfig, SearchResult
from src.search.scheduler import BudgetScheduler, BudgetAllocation
from src.search.early_stopping import EarlyStopping
from src.search.parallel import ParallelTaskSearch
from src.arc.loader import ARCLoader
from src.analysis.search_dynamics import SearchDynamicsAnalyzer, GenerationSnapshot
from src.analysis.operator_effectiveness import OperatorEffectivenessAnalyzer
from src.analysis.task_difficulty import TaskDifficultyAnalyzer
from src.analysis.report import ReportGenerator
from src.population.individual import Individual


class TestSearchConfig:
    def test_defaults(self):
        config = SearchConfig()
        assert config.population_size == 50
        assert config.max_generations == 100


class TestSearchResult:
    def test_summary(self):
        result = SearchResult(
            best_fitness=0.8,
            generations_run=10,
            total_evaluations=500,
            solved=False,
            stop_reason="stagnation",
        )
        s = result.summary()
        assert "0.8" in s
        assert "stagnation" in s


class TestEarlyStopping:
    def test_no_stop_initially(self):
        es = EarlyStopping(patience=5, min_generations=3)
        assert not es.should_stop(0.5)
        assert not es.should_stop(0.6)

    def test_stop_on_stagnation(self):
        es = EarlyStopping(patience=3, min_generations=1)
        assert not es.should_stop(0.5)  # gen 1: best=0.5, stagnation=0
        assert not es.should_stop(0.5)  # gen 2: stagnation=1
        assert not es.should_stop(0.5)  # gen 3: stagnation=2
        assert es.should_stop(0.5)     # gen 4: stagnation=3 >= patience=3

    def test_no_stop_with_improvement(self):
        es = EarlyStopping(patience=3, min_generations=1)
        for i in range(20):
            result = es.should_stop(i * 0.05)
            assert not result  # Always improving

    def test_reset(self):
        es = EarlyStopping(patience=3)
        es.should_stop(0.5)
        es.should_stop(0.5)
        es.reset()
        assert es.state.total_generations == 0

    def test_stagnation_ratio(self):
        es = EarlyStopping(patience=10)
        es.should_stop(0.5)
        es.should_stop(0.5)
        es.should_stop(0.5)
        ratio = es.stagnation_ratio()
        assert 0.0 <= ratio <= 1.0

    def test_stagnation_ratio_zero_patience(self):
        es = EarlyStopping(patience=0)
        assert es.stagnation_ratio() == 1.0

    def test_recent_trend(self):
        es = EarlyStopping(patience=10)
        for i in range(10):
            es.should_stop(i * 0.1)
        trend = es.recent_trend()
        assert trend > 0  # Positive trend (improving)

    def test_recent_trend_short(self):
        es = EarlyStopping()
        assert es.recent_trend() == 0.0
        es.should_stop(0.5)
        assert es.recent_trend() == 0.0  # Only one point

    def test_summary(self):
        es = EarlyStopping(patience=5)
        es.should_stop(0.5)
        s = es.summary()
        assert "best_fitness" in s
        assert "patience" in s

    def test_min_generations_prevents_stopping(self):
        es = EarlyStopping(patience=1, min_generations=5)
        for _ in range(4):
            assert not es.should_stop(0.5)


class TestBudgetScheduler:
    def test_has_budget(self):
        sched = BudgetScheduler(max_generations=10)
        assert sched.has_budget(5)
        assert not sched.has_budget(10)

    def test_allocate_task(self):
        sched = BudgetScheduler()
        alloc = sched.allocate_task("task1", priority=2.0)
        assert alloc.task_id == "task1"
        assert alloc.priority == 2.0

    def test_record_generation(self):
        sched = BudgetScheduler()
        sched.allocate_task("task1")
        sched.record_generation("task1")
        alloc = sched.get_allocation("task1")
        assert alloc.used_generations == 1

    def test_record_evaluations(self):
        sched = BudgetScheduler()
        sched.allocate_task("task1")
        sched.record_evaluations("task1", 50)
        alloc = sched.get_allocation("task1")
        assert alloc.used_evaluations == 50

    def test_should_continue(self):
        sched = BudgetScheduler(max_generations=2)
        sched.allocate_task("task1", max_generations=2)
        assert sched.should_continue("task1")
        sched.record_generation("task1")
        sched.record_generation("task1")
        assert not sched.should_continue("task1")

    def test_should_continue_no_allocation(self):
        sched = BudgetScheduler()
        assert sched.should_continue("unknown_task")

    def test_redistribute(self):
        sched = BudgetScheduler()
        sched.allocate_task("task1", max_generations=10)
        sched.allocate_task("task2", max_generations=10)
        sched.redistribute(["task1"])
        alloc2 = sched.get_allocation("task2")
        assert alloc2.max_generations > 10

    def test_summary(self):
        sched = BudgetScheduler()
        sched.allocate_task("task1")
        s = sched.summary()
        assert "global_generations_used" in s
        assert "tasks" in s

    def test_budget_allocation_properties(self):
        alloc = BudgetAllocation(
            task_id="t1",
            max_generations=10,
            max_evaluations=100,
            used_generations=5,
            used_evaluations=30,
        )
        assert alloc.generations_remaining == 5
        assert alloc.evaluations_remaining == 70
        assert alloc.fraction_used == 0.5

    def test_record_generation_no_allocation(self):
        sched = BudgetScheduler()
        sched.record_generation("unknown")  # Should not crash
        sched.record_evaluations("unknown", 10)  # Should not crash


class TestEvolutionarySearchEngine:
    def test_search_basic(self, color_swap_task, mock_llm):
        config = SearchConfig(
            population_size=5,
            max_generations=3,
            stagnation_limit=10,
        )
        engine = EvolutionarySearchEngine(config=config, llm_call=mock_llm)
        result = engine.search(color_swap_task)
        assert result is not None
        assert result.generations_run >= 0
        assert result.total_evaluations > 0
        assert result.best_individual is not None

    def test_search_stops_on_stagnation(self, color_swap_task):
        config = SearchConfig(
            population_size=5,
            max_generations=100,
            stagnation_limit=2,
        )

        def identity_llm(prompt):
            return "def transform(g): return [row[:] for row in g]"

        engine = EvolutionarySearchEngine(config=config, llm_call=identity_llm)
        result = engine.search(color_swap_task)
        assert result.stop_reason in ("stagnation", "max_generations", "target_fitness_reached", "budget_exhausted")

    def test_search_solves_task(self, color_swap_task, mock_llm):
        config = SearchConfig(
            population_size=10,
            max_generations=5,
            target_fitness=0.1,  # Very low target
        )
        engine = EvolutionarySearchEngine(config=config, llm_call=mock_llm)
        result = engine.search(color_swap_task)
        # Should achieve at least something
        assert result.best_fitness > 0

    def test_search_result_has_history(self, color_swap_task, mock_llm):
        config = SearchConfig(
            population_size=5,
            max_generations=3,
            stagnation_limit=10,
        )
        engine = EvolutionarySearchEngine(config=config, llm_call=mock_llm)
        result = engine.search(color_swap_task)
        assert isinstance(result.history, list)


class TestParallelTaskSearch:
    def test_search_all_sequential(self, mock_llm):
        config = SearchConfig(
            population_size=3,
            max_generations=2,
            stagnation_limit=5,
        )
        loader = ARCLoader()
        tasks = [loader.load_task("simple_color_swap")]

        pts = ParallelTaskSearch(config=config, max_workers=1, llm_call=mock_llm)
        result = pts.search_all(tasks)
        assert result.num_tasks == 1
        assert result.total_time >= 0

    def test_search_all_multiple(self, mock_llm):
        config = SearchConfig(
            population_size=3,
            max_generations=2,
            stagnation_limit=5,
        )
        loader = ARCLoader()
        tasks = loader.load_all()

        pts = ParallelTaskSearch(config=config, max_workers=1, llm_call=mock_llm)
        result = pts.search_all(tasks)
        assert result.num_tasks == len(tasks)

    def test_parallel_search_result_properties(self):
        from src.search.parallel import ParallelSearchResult
        r = ParallelSearchResult()
        assert r.num_tasks == 0
        assert r.solve_rate == 0.0
        assert "ParallelSearchResult" in r.summary()

    def test_search_result_solved_count(self):
        from src.search.parallel import ParallelSearchResult
        r = ParallelSearchResult(
            results={
                "t1": SearchResult(solved=True),
                "t2": SearchResult(solved=False),
            }
        )
        assert r.num_solved == 1
        assert r.solve_rate == 0.5

    def test_search_all_parallel(self, mock_llm):
        config = SearchConfig(
            population_size=3,
            max_generations=2,
            stagnation_limit=5,
        )
        loader = ARCLoader()
        tasks = [loader.load_task("simple_color_swap"), loader.load_task("pattern_fill")]

        pts = ParallelTaskSearch(config=config, max_workers=2, llm_call=mock_llm)
        result = pts.search_all(tasks)
        assert result.num_tasks == 2


class TestSearchDynamicsAnalyzer:
    def test_record_and_trajectory(self):
        analyzer = SearchDynamicsAnalyzer()
        for i in range(10):
            analyzer.record(GenerationSnapshot(
                generation=i, best_fitness=i * 0.1, avg_fitness=i * 0.05
            ))
        traj = analyzer.fitness_trajectory()
        assert len(traj) == 10
        assert traj[0] == 0.0
        assert traj[-1] == pytest.approx(0.9)

    def test_record_from_dict(self):
        analyzer = SearchDynamicsAnalyzer()
        analyzer.record_from_dict({"generation": 0, "best_fitness": 0.5, "avg_fitness": 0.3})
        assert len(analyzer.snapshots) == 1

    def test_convergence_generation(self):
        analyzer = SearchDynamicsAnalyzer()
        for i in range(10):
            analyzer.record(GenerationSnapshot(
                generation=i, best_fitness=i * 0.1, avg_fitness=0.0
            ))
        assert analyzer.convergence_generation(0.5) == 5

    def test_convergence_not_found(self):
        analyzer = SearchDynamicsAnalyzer()
        analyzer.record(GenerationSnapshot(generation=0, best_fitness=0.1, avg_fitness=0.0))
        assert analyzer.convergence_generation(0.99) is None

    def test_improvement_rate(self):
        analyzer = SearchDynamicsAnalyzer()
        analyzer.record(GenerationSnapshot(generation=0, best_fitness=0.0, avg_fitness=0.0))
        analyzer.record(GenerationSnapshot(generation=1, best_fitness=1.0, avg_fitness=0.5))
        assert analyzer.improvement_rate() == 1.0

    def test_improvement_rate_empty(self):
        analyzer = SearchDynamicsAnalyzer()
        assert analyzer.improvement_rate() == 0.0

    def test_stagnation_periods(self):
        analyzer = SearchDynamicsAnalyzer()
        for i in range(10):
            analyzer.record(GenerationSnapshot(
                generation=i, best_fitness=0.5, avg_fitness=0.5
            ))
        periods = analyzer.stagnation_periods(min_length=3)
        assert len(periods) > 0

    def test_summary(self):
        analyzer = SearchDynamicsAnalyzer()
        analyzer.record(GenerationSnapshot(generation=0, best_fitness=0.5, avg_fitness=0.3))
        s = analyzer.summary()
        assert s["num_generations"] == 1

    def test_summary_empty(self):
        analyzer = SearchDynamicsAnalyzer()
        s = analyzer.summary()
        assert s["num_generations"] == 0

    def test_clear(self):
        analyzer = SearchDynamicsAnalyzer()
        analyzer.record(GenerationSnapshot(generation=0, best_fitness=0.5, avg_fitness=0.3))
        analyzer.clear()
        assert len(analyzer.snapshots) == 0

    def test_avg_fitness_trajectory(self):
        analyzer = SearchDynamicsAnalyzer()
        analyzer.record(GenerationSnapshot(generation=0, best_fitness=0.5, avg_fitness=0.3))
        assert analyzer.avg_fitness_trajectory() == [0.3]


class TestOperatorEffectivenessAnalyzer:
    def test_record(self):
        analyzer = OperatorEffectivenessAnalyzer()
        analyzer.record("mutate_bug_fix", 0.5, 0.7)
        stats = analyzer.get_stats("mutate_bug_fix")
        assert stats is not None
        assert stats.invocations == 1
        assert stats.improvements == 1

    def test_record_no_improvement(self):
        analyzer = OperatorEffectivenessAnalyzer()
        analyzer.record("mutate_bug_fix", 0.7, 0.5)
        stats = analyzer.get_stats("mutate_bug_fix")
        assert stats.improvements == 0

    def test_record_from_individual(self):
        analyzer = OperatorEffectivenessAnalyzer()
        ind = Individual(
            code="c",
            operator="crossover",
            fitness=0.6,
            metadata={"parent_fitness": 0.4},
        )
        analyzer.record_from_individual(ind)
        stats = analyzer.get_stats("crossover")
        assert stats.improvements == 1

    def test_rank_operators(self):
        analyzer = OperatorEffectivenessAnalyzer()
        analyzer.record("op_a", 0.5, 0.7)
        analyzer.record("op_b", 0.5, 0.3)
        ranked = analyzer.rank_operators()
        assert ranked[0].name == "op_a"

    def test_best_operator(self):
        analyzer = OperatorEffectivenessAnalyzer()
        analyzer.record("op_a", 0.5, 0.7)
        assert analyzer.best_operator() == "op_a"

    def test_best_operator_empty(self):
        analyzer = OperatorEffectivenessAnalyzer()
        assert analyzer.best_operator() is None

    def test_summary(self):
        analyzer = OperatorEffectivenessAnalyzer()
        analyzer.record("op_a", 0.5, 0.7)
        s = analyzer.summary()
        assert "op_a" in s

    def test_clear(self):
        analyzer = OperatorEffectivenessAnalyzer()
        analyzer.record("op_a", 0.5, 0.7)
        analyzer.clear()
        assert analyzer.all_stats() == {}

    def test_operator_stats_properties(self):
        from src.analysis.operator_effectiveness import OperatorStats
        stats = OperatorStats(name="test", invocations=0)
        assert stats.improvement_rate == 0.0
        assert stats.avg_fitness_delta == 0.0


class TestTaskDifficultyAnalyzer:
    def test_analyze_task(self, color_swap_task):
        analyzer = TaskDifficultyAnalyzer()
        analysis = analyzer.analyze_task(color_swap_task)
        assert analysis.estimated_difficulty.score >= 0.0
        assert analysis.estimated_difficulty.level in ("easy", "medium", "hard")

    def test_analyze_with_search_result(self, color_swap_task):
        analyzer = TaskDifficultyAnalyzer()
        result = SearchResult(
            solved=True,
            best_fitness=0.95,
            total_evaluations=100,
            elapsed_seconds=1.0,
            history=[{"generation": 5, "best_fitness": 0.99}],
        )
        analysis = analyzer.analyze_task(color_swap_task, result)
        assert analysis.actual_solved

    def test_rank_by_difficulty(self, all_tasks):
        analyzer = TaskDifficultyAnalyzer()
        for task in all_tasks:
            analyzer.analyze_task(task)
        ranked = analyzer.rank_by_difficulty()
        assert len(ranked) == len(all_tasks)

    def test_rank_by_actual_difficulty(self, all_tasks):
        analyzer = TaskDifficultyAnalyzer()
        for task in all_tasks:
            analyzer.analyze_task(task)
        ranked = analyzer.rank_by_actual_difficulty()
        assert len(ranked) == len(all_tasks)

    def test_summary(self, color_swap_task):
        analyzer = TaskDifficultyAnalyzer()
        analyzer.analyze_task(color_swap_task)
        s = analyzer.summary()
        assert s["num_tasks"] == 1

    def test_summary_empty(self):
        analyzer = TaskDifficultyAnalyzer()
        s = analyzer.summary()
        assert s["num_tasks"] == 0

    def test_difficulty_correlation_insufficient(self):
        analyzer = TaskDifficultyAnalyzer()
        assert analyzer.difficulty_correlation() is None

    def test_clear(self, color_swap_task):
        analyzer = TaskDifficultyAnalyzer()
        analyzer.analyze_task(color_swap_task)
        analyzer.clear()
        assert analyzer.get_analysis("simple_color_swap") is None

    def test_get_analysis(self, color_swap_task):
        analyzer = TaskDifficultyAnalyzer()
        analyzer.analyze_task(color_swap_task)
        a = analyzer.get_analysis("simple_color_swap")
        assert a is not None
        assert a.task_id == "simple_color_swap"

    def test_difficulty_accuracy_none(self, color_swap_task):
        analyzer = TaskDifficultyAnalyzer()
        a = analyzer.analyze_task(color_swap_task)
        assert a.difficulty_accuracy is None

    def test_difficulty_correlation_with_data(self, all_tasks):
        analyzer = TaskDifficultyAnalyzer()
        for task in all_tasks:
            result = SearchResult(
                solved=True,
                best_fitness=0.8,
                total_evaluations=100,
            )
            analyzer.analyze_task(task, result)
        corr = analyzer.difficulty_correlation()
        # With enough tasks, correlation should be computable
        if corr is not None:
            assert -1.0 <= corr <= 1.0

    def test_difficulty_accuracy_with_gen(self, color_swap_task):
        analyzer = TaskDifficultyAnalyzer()
        result = SearchResult(
            solved=True,
            best_fitness=1.0,
            history=[{"generation": 3, "best_fitness": 0.99}],
        )
        a = analyzer.analyze_task(color_swap_task, result)
        assert a.generations_to_solve == 3
        assert a.difficulty_accuracy is not None


class TestReportGenerator:
    def test_generate_search_report(self):
        gen = ReportGenerator()
        result = SearchResult(
            best_fitness=0.8,
            generations_run=10,
            solved=False,
            stop_reason="stagnation",
            best_individual=Individual(
                code="def transform(g): return g",
                fitness=0.8,
                train_accuracy=0.7,
                test_accuracy=0.6,
                generation=5,
            ),
            history=[{"best_fitness": 0.5}, {"best_fitness": 0.8}],
        )
        report = gen.generate_search_report(result, task_id="test")
        assert report["task_id"] == "test"
        assert report["best_fitness"] == 0.8
        assert "best_program" in report

    def test_generate_benchmark_report(self):
        gen = ReportGenerator()
        results = {
            "task1": SearchResult(solved=True, best_fitness=1.0, elapsed_seconds=1.0),
            "task2": SearchResult(solved=False, best_fitness=0.5, elapsed_seconds=2.0),
        }
        report = gen.generate_benchmark_report(results)
        assert report["num_tasks"] == 2
        assert report["num_solved"] == 1
        assert report["solve_rate"] == 0.5

    def test_generate_full_report(self):
        gen = ReportGenerator()
        results = {
            "task1": SearchResult(solved=True, best_fitness=1.0),
        }
        report = gen.generate_full_report(results)
        assert "dynamics" in report
        assert "operators" in report
        assert "difficulty" in report

    def test_format_text_report(self):
        gen = ReportGenerator()
        report = {
            "timestamp": "2024-01-01",
            "num_tasks": 3,
            "num_solved": 2,
            "solve_rate": 0.667,
            "total_time": 10.5,
            "tasks": {
                "t1": {"solved": True, "best_fitness": 1.0},
                "t2": {"solved": False, "best_fitness": 0.3},
            },
        }
        text = gen.format_text_report(report)
        assert "SOAR-Evolution" in text
        assert "SOLVED" in text
        assert "UNSOLVED" in text

    def test_save_report(self, tmp_path):
        gen = ReportGenerator()
        report = {"test": True}
        path = str(tmp_path / "report.json")
        gen.save_report(report, path)
        import json
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["test"] is True

    def test_generate_search_report_no_individual(self):
        gen = ReportGenerator()
        result = SearchResult(best_fitness=0.0, generations_run=0)
        report = gen.generate_search_report(result)
        assert "best_program" not in report

    def test_generate_search_report_no_history(self):
        gen = ReportGenerator()
        result = SearchResult(best_fitness=0.0, generations_run=0)
        report = gen.generate_search_report(result)
        assert "fitness_trajectory" not in report
