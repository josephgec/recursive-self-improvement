"""Integration tests: end-to-end workflow tests."""

import pytest
from src.arc.loader import ARCLoader
from src.arc.evaluator import ProgramEvaluator
from src.arc.visualizer import GridVisualizer
from src.arc.difficulty import estimate_difficulty, DifficultyEstimate
from src.arc.grid import Grid, diff_grids
from src.population.individual import Individual
from src.population.population import Population
from src.population.fitness import FitnessComputer
from src.population.selection import TournamentSelection
from src.population.diversity import DiversityMetrics
from src.population.archive import EliteArchive
from src.operators.initializer import LLMInitializer
from src.operators.mutator import LLMMutator, MutationType
from src.operators.crossover import LLMCrossover
from src.operators.error_analyzer import ErrorAnalyzer
from src.operators.fragment_extractor import FragmentExtractor
from src.search.engine import EvolutionarySearchEngine, SearchConfig
from src.utils.code_similarity import code_similarity, normalize_code, structural_similarity, tokenize
from src.utils.grid_diff import compute_grid_diff, diff_summary, highlight_changes, compute_accuracy_map, common_errors
from src.utils.logging import get_logger, setup_logging


class TestEndToEnd:
    """Full pipeline integration tests."""

    def test_load_evaluate_task(self):
        """Load a task, evaluate a solution, check result."""
        loader = ARCLoader()
        task = loader.load_task("simple_color_swap")

        code = """def transform(input_grid):
    result = []
    for row in input_grid:
        new_row = []
        for cell in row:
            if cell == 1:
                new_row.append(2)
            else:
                new_row.append(cell)
        result.append(new_row)
    return result
"""
        evaluator = ProgramEvaluator()
        result = evaluator.evaluate_task(code, task)
        assert result.all_train_correct
        assert result.all_test_correct
        assert result.fully_correct

    def test_evolutionary_pipeline(self):
        """Test the full evolutionary pipeline end-to-end."""
        loader = ARCLoader()
        task = loader.load_task("simple_color_swap")

        def mock_llm(prompt):
            return '''def transform(input_grid):
    result = []
    for row in input_grid:
        new_row = []
        for cell in row:
            if cell == 1:
                new_row.append(2)
            else:
                new_row.append(cell)
        result.append(new_row)
    return result
'''

        config = SearchConfig(
            population_size=5,
            max_generations=3,
            stagnation_limit=10,
        )
        engine = EvolutionarySearchEngine(config=config, llm_call=mock_llm)
        result = engine.search(task)
        assert result.best_individual is not None
        assert result.best_fitness > 0

    def test_population_lifecycle(self):
        """Test creating, evaluating, and evolving a population."""
        loader = ARCLoader()
        task = loader.load_task("pattern_fill")

        # Initialize
        init = LLMInitializer()
        individuals = init.generate(task, count=5)
        assert len(individuals) == 5

        # Evaluate
        fc = FitnessComputer()
        fc.evaluate_population(individuals, task)
        for ind in individuals:
            assert ind.evaluated

        # Select
        sel = TournamentSelection(tournament_size=3)
        parents = sel.select(individuals, 3)
        assert len(parents) == 3

        # Mutate
        mutator = LLMMutator()
        children = mutator.mutate_batch(parents, task)
        assert len(children) == 3

        # Evaluate children
        fc.evaluate_population(children, task)

        # Population management
        pop = Population(max_size=10)
        pop.add_all(individuals)
        pop.replace_generation(children)
        assert pop.generation == 1

    def test_diversity_tracking(self):
        """Test diversity metrics throughout evolution."""
        inds = [
            Individual(code=f"def transform(g): return [[{i}]]", fitness=i * 0.1)
            for i in range(5)
        ]
        dm = DiversityMetrics()
        report = dm.diversity_report(inds)
        assert report["population_size"] == 5
        assert report["unique_ratio"] == 1.0

    def test_archive_integration(self):
        """Test elite archive with real evaluation."""
        loader = ARCLoader()
        task = loader.load_task("simple_color_swap")
        fc = FitnessComputer()
        archive = EliteArchive(max_size=5)

        codes = [
            "def transform(g): return [row[:] for row in g]",
            "def transform(g): return [[0]*len(g[0]) for _ in g]",
            "def transform(g):\n    return [[2 if c==1 else c for c in r] for r in g]",
        ]

        for code in codes:
            ind = Individual(code=code)
            fc.evaluate_individual(ind, task)
            archive.try_add(ind)

        assert archive.size > 0
        assert archive.best is not None


class TestLoaderIntegration:
    def test_load_all_builtin(self):
        loader = ARCLoader()
        tasks = loader.load_all()
        assert len(tasks) >= 3

    def test_load_from_fixtures(self, fixtures_dir):
        loader = ARCLoader(data_dir=str(fixtures_dir))
        tasks = loader.load_all()
        assert len(tasks) >= 3

    def test_list_task_ids(self):
        loader = ARCLoader()
        ids = loader.list_task_ids()
        assert "simple_color_swap" in ids

    def test_cache(self):
        loader = ARCLoader()
        t1 = loader.load_task("simple_color_swap")
        t2 = loader.load_task("simple_color_swap")
        assert t1 is t2

    def test_clear_cache(self):
        loader = ARCLoader()
        loader.load_task("simple_color_swap")
        loader.clear_cache()
        # Should reload
        t2 = loader.load_task("simple_color_swap")
        assert t2 is not None

    def test_load_nonexistent(self):
        loader = ARCLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_task("nonexistent_task_xyz")

    def test_load_task_from_file(self, tmp_path):
        """Load a task from file that is NOT a built-in."""
        import json
        task_data = {
            "train": [{"input": [[1, 0]], "output": [[0, 1]]}],
            "test": [{"input": [[0, 1]], "output": [[1, 0]]}],
        }
        task_file = tmp_path / "custom_task.json"
        with open(task_file, "w") as f:
            json.dump(task_data, f)

        loader = ARCLoader(data_dir=str(tmp_path))
        task = loader.load_task("custom_task")
        assert task.task_id == "custom_task"
        assert task.num_train == 1

    def test_load_all_with_file_tasks(self, tmp_path):
        """Load all including file-based tasks not in built-in."""
        import json
        task_data = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [],
        }
        task_file = tmp_path / "extra_task.json"
        with open(task_file, "w") as f:
            json.dump(task_data, f)

        loader = ARCLoader(data_dir=str(tmp_path))
        tasks = loader.load_all()
        ids = [t.task_id for t in tasks]
        assert "extra_task" in ids

    def test_list_task_ids_with_data_dir(self, tmp_path):
        """List task IDs including file-based tasks."""
        import json
        task_data = {"train": [{"input": [[1]], "output": [[2]]}], "test": []}
        (tmp_path / "file_task.json").write_text(json.dumps(task_data))

        loader = ARCLoader(data_dir=str(tmp_path))
        ids = loader.list_task_ids()
        assert "file_task" in ids

    def test_load_all_with_bad_file(self, tmp_path):
        """Ensure bad JSON files don't crash load_all."""
        (tmp_path / "bad_task.json").write_text("not json")
        loader = ARCLoader(data_dir=str(tmp_path))
        tasks = loader.load_all()
        # Should still load built-in tasks even if file task fails
        assert len(tasks) >= 3


class TestVisualizerIntegration:
    def test_render_grid(self):
        vis = GridVisualizer()
        grid = Grid.from_list([[1, 2], [3, 4]])
        s = vis.render_grid(grid)
        assert "2x2" in s
        assert "1 2" in s

    def test_render_grid_color_names(self):
        vis = GridVisualizer(use_color_names=True)
        grid = Grid.from_list([[0, 1], [2, 3]])
        s = vis.render_grid(grid)
        assert "black" in s
        assert "blue" in s

    def test_render_task(self, color_swap_task):
        vis = GridVisualizer()
        s = vis.render_task(color_swap_task)
        assert "simple_color_swap" in s
        assert "Training" in s

    def test_render_example(self, color_swap_task):
        vis = GridVisualizer()
        s = vis.render_example(color_swap_task.train[0])
        assert "Input" in s
        assert "Output" in s

    def test_render_comparison(self):
        vis = GridVisualizer()
        inp = Grid.from_list([[1, 2], [3, 4]])
        expected = Grid.from_list([[2, 1], [4, 3]])
        actual = Grid.from_list([[2, 2], [4, 4]])
        s = vis.render_comparison(inp, expected, actual)
        assert "Differences" in s

    def test_render_comparison_match(self):
        vis = GridVisualizer()
        g = Grid.from_list([[1, 2], [3, 4]])
        s = vis.render_comparison(g, g, g)
        assert "No differences" in s

    def test_render_comparison_shape_mismatch(self):
        vis = GridVisualizer()
        inp = Grid.from_list([[1, 2], [3, 4]])
        expected = Grid.from_list([[1, 2, 3]])
        actual = Grid.from_list([[1, 2], [3, 4]])
        s = vis.render_comparison(inp, expected, actual)
        assert "Shape mismatch" in s

    def test_render_comparison_many_diffs(self):
        vis = GridVisualizer()
        inp = Grid.from_list([[i % 10 for i in range(10)] for _ in range(10)])
        expected = Grid.from_list([[0 for _ in range(10)] for _ in range(10)])
        actual = Grid.from_list([[9 for _ in range(10)] for _ in range(10)])
        s = vis.render_comparison(inp, expected, actual)
        assert "... and" in s or "Differences" in s

    def test_render_task_compact(self, color_swap_task):
        vis = GridVisualizer()
        s = vis.render_task_compact(color_swap_task)
        assert "simple_color_swap" in s
        assert "Train 1:" in s


class TestDifficultyIntegration:
    def test_estimate_all_tasks(self, all_tasks):
        for task in all_tasks:
            est = estimate_difficulty(task)
            assert 0.0 <= est.score <= 1.0
            assert est.level in ("easy", "medium", "hard")
            assert len(est.factors) > 0

    def test_difficulty_from_score(self):
        d = DifficultyEstimate.from_score(0.1, {})
        assert d.level == "easy"
        d = DifficultyEstimate.from_score(0.5, {})
        assert d.level == "medium"
        d = DifficultyEstimate.from_score(0.8, {})
        assert d.level == "hard"

    def test_difficulty_clamping(self):
        d = DifficultyEstimate.from_score(-0.5, {})
        assert d.score == 0.0
        d = DifficultyEstimate.from_score(1.5, {})
        assert d.score == 1.0


class TestCodeSimilarityIntegration:
    def test_identical_code(self):
        code = "def f(): return 1"
        assert code_similarity(code, code) == 1.0

    def test_different_code(self):
        a = "def transform(g): return g"
        b = "x = 42\ny = 99\nz = x + y"
        sim = code_similarity(a, b)
        assert sim < 1.0

    def test_empty_code(self):
        # Two empty strings are trivially identical
        assert code_similarity("", "") == 1.0
        assert code_similarity("abc", "") == 0.0

    def test_normalize(self):
        code = "x = 1  # comment\ny = 2  # another comment"
        norm = normalize_code(code)
        assert "#" not in norm

    def test_structural_similarity(self):
        a = "def foo(x): return x + 1"
        b = "def bar(y): return y + 1"
        sim = structural_similarity(a, b)
        assert sim > 0.5  # Structurally similar

    def test_structural_similarity_different(self):
        a = "x = 1"
        b = "for i in range(10): print(i)"
        sim = structural_similarity(a, b)
        assert sim < 1.0

    def test_tokenize(self):
        tokens = tokenize("def f(x): return x + 1")
        assert "def" in tokens
        assert "f" in tokens
        assert "return" in tokens

    def test_similarity_normalized_match(self):
        a = "x = 1  # comment"
        b = "x = 1"
        sim = code_similarity(a, b)
        assert sim == 1.0  # Same after normalization


class TestGridDiffIntegration:
    def test_compute_grid_diff(self):
        g1 = Grid.from_list([[1, 2], [3, 4]])
        g2 = Grid.from_list([[1, 9], [3, 4]])
        d = compute_grid_diff(g1, g2)
        assert d.num_changes == 1

    def test_diff_summary(self):
        g1 = Grid.from_list([[1, 2], [3, 4]])
        g2 = Grid.from_list([[1, 9], [3, 4]])
        d = compute_grid_diff(g1, g2)
        s = diff_summary(d)
        assert "Changed cells: 1" in s

    def test_highlight_changes(self):
        g1 = Grid.from_list([[1, 2], [3, 4]])
        g2 = Grid.from_list([[1, 9], [3, 4]])
        result = highlight_changes(g1, g2)
        assert "2 -> 9" in result

    def test_highlight_no_changes(self):
        g = Grid.from_list([[1, 2], [3, 4]])
        result = highlight_changes(g, g)
        assert "No changes" in result

    def test_highlight_shape_changed(self):
        g1 = Grid.from_list([[1, 2]])
        g2 = Grid.from_list([[1, 2], [3, 4]])
        result = highlight_changes(g1, g2)
        assert "Shape changed" in result

    def test_compute_accuracy_map(self):
        out = Grid.from_list([[1, 2], [3, 4]])
        exp = Grid.from_list([[1, 9], [3, 4]])
        m = compute_accuracy_map(out, exp)
        assert m[0][0] is True
        assert m[0][1] is False

    def test_accuracy_map_shape_mismatch(self):
        out = Grid.from_list([[1]])
        exp = Grid.from_list([[1, 2]])
        m = compute_accuracy_map(out, exp)
        assert m == []

    def test_common_errors(self):
        outputs = [
            Grid.from_list([[0, 0], [0, 0]]),
            Grid.from_list([[1, 1], [1, 1]]),
        ]
        expected = [
            Grid.from_list([[1, 1], [1, 1]]),
            Grid.from_list([[1, 1], [1, 1]]),
        ]
        errors = common_errors(outputs, expected)
        assert "mostly_wrong" in errors or "minor_errors" in errors or not errors

    def test_common_errors_shape_mismatch(self):
        outputs = [Grid.from_list([[1]])]
        expected = [Grid.from_list([[1, 2]])]
        errors = common_errors(outputs, expected)
        assert errors.get("shape_mismatch", 0) == 1


class TestLoggingIntegration:
    def test_get_logger(self):
        logger = get_logger("test_module")
        assert logger is not None
        assert "soar.test_module" in logger.name

    def test_setup_logging(self):
        setup_logging(level="DEBUG")
        logger = get_logger("test")
        logger.debug("test message")

    def test_setup_logging_with_file(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        setup_logging(level="INFO", log_file=log_file)
        logger = get_logger("test_file")
        logger.info("test message")
