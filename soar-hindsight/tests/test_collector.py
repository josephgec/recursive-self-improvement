"""Tests for trajectory collection, database, indexer, and statistics."""

import json
import os
import tempfile

import pytest

from src.collection.collector import TrajectoryCollector
from src.collection.database import TrajectoryDatabase
from src.collection.indexer import TrajectoryIndexer
from src.collection.statistics import CorpusStatistics
from src.collection.trajectory import (
    IndividualRecord,
    ImprovementStep,
    SearchTrajectory,
    TaskSpec,
)


class TestTaskSpec:
    def test_to_dict_from_dict(self, task_spec):
        d = task_spec.to_dict()
        restored = TaskSpec.from_dict(d)
        assert restored.task_id == task_spec.task_id
        assert restored.description == task_spec.description
        assert restored.difficulty == task_spec.difficulty
        assert restored.tags == task_spec.tags

    def test_defaults(self):
        ts = TaskSpec(task_id="t", description="d")
        assert ts.difficulty == "medium"
        assert ts.tags == []
        assert ts.test_cases == []
        assert ts.constraints == {}


class TestIndividualRecord:
    def test_is_solved(self):
        ind = IndividualRecord(fitness=1.0)
        assert ind.is_solved is True
        ind2 = IndividualRecord(fitness=0.5)
        assert ind2.is_solved is False

    def test_to_dict_from_dict(self, individual_records):
        ind = individual_records[0]
        d = ind.to_dict()
        restored = IndividualRecord.from_dict(d)
        assert restored.individual_id == ind.individual_id
        assert restored.fitness == ind.fitness
        assert restored.error == ind.error
        assert restored.operator == ind.operator

    def test_defaults(self):
        ind = IndividualRecord()
        assert ind.generation == 0
        assert ind.fitness == 0.0
        assert ind.parent_ids == []
        assert ind.error is None


class TestImprovementStep:
    def test_fitness_delta(self):
        step = ImprovementStep(fitness_before=0.3, fitness_after=0.7)
        assert step.fitness_delta == pytest.approx(0.4)

    def test_is_improvement(self):
        step = ImprovementStep(fitness_before=0.3, fitness_after=0.7)
        assert step.is_improvement is True
        step2 = ImprovementStep(fitness_before=0.7, fitness_after=0.3)
        assert step2.is_improvement is False

    def test_to_dict_from_dict(self):
        step = ImprovementStep(
            operator="mutation",
            code_before="x=1",
            code_after="x=2",
            fitness_before=0.3,
            fitness_after=0.8,
            error_before="err",
            error_after=None,
        )
        d = step.to_dict()
        restored = ImprovementStep.from_dict(d)
        assert restored.operator == "mutation"
        assert restored.fitness_delta == pytest.approx(0.5)
        assert restored.error_before == "err"
        assert restored.error_after is None


class TestSearchTrajectory:
    def test_best_individual(self, solved_trajectory):
        best = solved_trajectory.best_individual
        assert best is not None
        assert best.fitness == 1.0

    def test_best_individual_empty(self):
        traj = SearchTrajectory()
        assert traj.best_individual is None

    def test_solved_individuals(self, solved_trajectory):
        solved = solved_trajectory.solved_individuals
        assert len(solved) == 2

    def test_failed_individuals(self, solved_trajectory):
        failed = solved_trajectory.failed_individuals
        assert len(failed) == 1
        assert failed[0].error is not None

    def test_generations(self, solved_trajectory):
        gens = solved_trajectory.generations
        assert len(gens) == 2
        assert len(gens[0]) == 2  # gen 0
        assert len(gens[1]) == 2  # gen 1

    def test_add_individual(self):
        traj = SearchTrajectory()
        ind = IndividualRecord(fitness=0.5, generation=0)
        traj.add_individual(ind)
        assert traj.best_fitness == 0.5
        assert traj.total_generations == 1
        assert not traj.solved

        ind2 = IndividualRecord(fitness=1.0, generation=1)
        traj.add_individual(ind2)
        assert traj.best_fitness == 1.0
        assert traj.solved is True
        assert traj.total_generations == 2

    def test_extract_improvement_chain(self, solved_trajectory):
        chain = solved_trajectory.extract_improvement_chain()
        assert len(chain) >= 1
        for step in chain:
            assert step.fitness_after > step.fitness_before

    def test_to_dict_from_dict(self, solved_trajectory):
        d = solved_trajectory.to_dict()
        restored = SearchTrajectory.from_dict(d)
        assert restored.trajectory_id == solved_trajectory.trajectory_id
        assert restored.best_fitness == solved_trajectory.best_fitness
        assert restored.solved == solved_trajectory.solved
        assert len(restored.individuals) == len(solved_trajectory.individuals)
        assert restored.task.task_id == solved_trajectory.task.task_id

    def test_from_dict_no_task(self):
        d = {"trajectory_id": "no-task", "task": None, "individuals": []}
        traj = SearchTrajectory.from_dict(d)
        assert traj.task is None

    def test_generations_empty(self):
        traj = SearchTrajectory()
        assert traj.generations == []


class TestTrajectoryCollector:
    def test_collect_from_file(self, solved_trajectory_file):
        collector = TrajectoryCollector()
        traj = collector.collect_from_file(solved_trajectory_file)
        assert traj is not None
        assert traj.solved is True
        assert collector.count == 1

    def test_collect_from_directory(self, fixtures_dir):
        collector = TrajectoryCollector(trajectory_dir=fixtures_dir)
        collected = collector.collect_from_directory()
        assert len(collected) == 3
        assert collector.count == 3

    def test_collect_from_dicts(self):
        collector = TrajectoryCollector()
        data = [
            {
                "trajectory_id": "t1",
                "task": {"task_id": "tid", "description": "test"},
                "individuals": [
                    {"code": "x", "fitness": 0.5, "generation": 0, "operator": "init"}
                ],
                "best_fitness": 0.5,
            }
        ]
        collected = collector.collect_from_dicts(data)
        assert len(collected) == 1
        assert collector.count == 1

    def test_min_fitness_filter(self, fixtures_dir):
        collector = TrajectoryCollector(trajectory_dir=fixtures_dir, min_fitness=0.5)
        collected = collector.collect_from_directory()
        for t in collected:
            assert t.best_fitness >= 0.5

    def test_get_solved_trajectories(self, fixtures_dir):
        collector = TrajectoryCollector(trajectory_dir=fixtures_dir)
        collector.collect_from_directory()
        solved = collector.get_solved_trajectories()
        assert len(solved) == 1
        assert all(t.solved for t in solved)

    def test_get_partial_trajectories(self, fixtures_dir):
        collector = TrajectoryCollector(trajectory_dir=fixtures_dir)
        collector.collect_from_directory()
        partial = collector.get_partial_trajectories(min_fitness=0.3, max_fitness=1.0)
        assert len(partial) >= 1

    def test_get_failed_trajectories(self, fixtures_dir):
        collector = TrajectoryCollector(trajectory_dir=fixtures_dir)
        collector.collect_from_directory()
        failed = collector.get_failed_trajectories()
        assert len(failed) >= 1

    def test_get_crossover_candidates(self, fixtures_dir):
        collector = TrajectoryCollector(trajectory_dir=fixtures_dir)
        collector.collect_from_directory()
        crossover = collector.get_crossover_candidates()
        assert len(crossover) >= 1

    def test_extract_improvement_chains(self, fixtures_dir):
        collector = TrajectoryCollector(trajectory_dir=fixtures_dir)
        collector.collect_from_directory()
        chains = collector.extract_improvement_chains()
        assert len(chains) >= 1

    def test_clear(self, fixtures_dir):
        collector = TrajectoryCollector(trajectory_dir=fixtures_dir)
        collector.collect_from_directory()
        assert collector.count > 0
        collector.clear()
        assert collector.count == 0

    def test_summary(self, fixtures_dir):
        collector = TrajectoryCollector(trajectory_dir=fixtures_dir)
        collector.collect_from_directory()
        s = collector.summary()
        assert s["total"] > 0
        assert "solved" in s
        assert "avg_fitness" in s

    def test_summary_empty(self):
        collector = TrajectoryCollector()
        s = collector.summary()
        assert s["total"] == 0
        assert s["avg_fitness"] == 0.0

    def test_collect_nonexistent_dir(self):
        collector = TrajectoryCollector(trajectory_dir="/nonexistent/path")
        result = collector.collect_from_directory()
        assert result == []

    def test_trajectories_property(self, fixtures_dir):
        collector = TrajectoryCollector(trajectory_dir=fixtures_dir)
        collector.collect_from_directory()
        trajs = collector.trajectories
        assert isinstance(trajs, list)
        assert len(trajs) == collector.count


class TestTrajectoryDatabase:
    def test_store_and_load(self, solved_trajectory):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = TrajectoryDatabase(db_dir=tmpdir)
            tid = db.store(solved_trajectory)
            assert tid == solved_trajectory.trajectory_id

            loaded = db.load(tid)
            assert loaded is not None
            assert loaded.trajectory_id == tid
            assert loaded.best_fitness == solved_trajectory.best_fitness

    def test_load_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = TrajectoryDatabase(db_dir=tmpdir)
            result = db.load("nonexistent")
            assert result is None

    def test_delete(self, solved_trajectory):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = TrajectoryDatabase(db_dir=tmpdir)
            db.store(solved_trajectory)
            assert db.count() == 1
            db.delete(solved_trajectory.trajectory_id)
            assert db.count() == 0

    def test_list_ids(self, solved_trajectory, partial_trajectory):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = TrajectoryDatabase(db_dir=tmpdir)
            db.store(solved_trajectory)
            db.store(partial_trajectory)
            ids = db.list_ids()
            assert len(ids) == 2

    def test_query_by_fitness(self, solved_trajectory, partial_trajectory):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = TrajectoryDatabase(db_dir=tmpdir)
            db.store(solved_trajectory)
            db.store(partial_trajectory)
            high = db.query_by_fitness(min_fitness=0.8)
            assert solved_trajectory.trajectory_id in high

    def test_query_by_solved(self, solved_trajectory, partial_trajectory):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = TrajectoryDatabase(db_dir=tmpdir)
            db.store(solved_trajectory)
            db.store(partial_trajectory)
            solved_ids = db.query_by_solved(True)
            assert solved_trajectory.trajectory_id in solved_ids
            unsolved_ids = db.query_by_solved(False)
            assert partial_trajectory.trajectory_id in unsolved_ids

    def test_query_by_task(self, solved_trajectory):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = TrajectoryDatabase(db_dir=tmpdir)
            db.store(solved_trajectory)
            result = db.query_by_task(solved_trajectory.task.task_id)
            assert solved_trajectory.trajectory_id in result

    def test_query_custom(self, solved_trajectory, partial_trajectory):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = TrajectoryDatabase(db_dir=tmpdir)
            db.store(solved_trajectory)
            db.store(partial_trajectory)
            result = db.query(lambda m: m.get("n_individuals", 0) > 2)
            assert len(result) >= 1

    def test_stats(self, solved_trajectory, partial_trajectory):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = TrajectoryDatabase(db_dir=tmpdir)
            db.store(solved_trajectory)
            db.store(partial_trajectory)
            s = db.stats()
            assert s["total"] == 2
            assert "avg_fitness" in s

    def test_stats_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = TrajectoryDatabase(db_dir=tmpdir)
            s = db.stats()
            assert s["total"] == 0

    def test_initialize_existing(self, solved_trajectory):
        with tempfile.TemporaryDirectory() as tmpdir:
            db1 = TrajectoryDatabase(db_dir=tmpdir)
            db1.store(solved_trajectory)

            # Initialize new db instance and load manifest
            db2 = TrajectoryDatabase(db_dir=tmpdir)
            db2.initialize()
            assert db2.count() == 1

    def test_cache(self, solved_trajectory):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = TrajectoryDatabase(db_dir=tmpdir)
            db.store(solved_trajectory)
            # Load from cache (second call)
            loaded1 = db.load(solved_trajectory.trajectory_id)
            loaded2 = db.load(solved_trajectory.trajectory_id)
            assert loaded1 is loaded2  # same object from cache


class TestTrajectoryIndexer:
    def test_index_and_lookup(self, solved_trajectory):
        indexer = TrajectoryIndexer()
        indexer.index(solved_trajectory)
        assert indexer.total_indexed() == 1

    def test_index_many(self, all_trajectories):
        indexer = TrajectoryIndexer()
        count = indexer.index_many(all_trajectories)
        assert count == 3
        assert indexer.total_indexed() == 3

    def test_lookup_by_task(self, solved_trajectory):
        indexer = TrajectoryIndexer()
        indexer.index(solved_trajectory)
        result = indexer.lookup_by_task(solved_trajectory.task.task_id)
        assert solved_trajectory.trajectory_id in result

    def test_lookup_by_fitness(self, all_trajectories):
        indexer = TrajectoryIndexer()
        indexer.index_many(all_trajectories)
        high = indexer.lookup_by_fitness(min_fitness=0.9)
        assert len(high) >= 1

    def test_lookup_by_operator(self, solved_trajectory):
        indexer = TrajectoryIndexer()
        indexer.index(solved_trajectory)
        result = indexer.lookup_by_operator("mutation")
        assert len(result) >= 1

    def test_lookup_solved(self, all_trajectories):
        indexer = TrajectoryIndexer()
        indexer.index_many(all_trajectories)
        solved = indexer.lookup_solved(True)
        assert len(solved) >= 1
        unsolved = indexer.lookup_solved(False)
        assert len(unsolved) >= 1

    def test_all_tasks(self, all_trajectories):
        indexer = TrajectoryIndexer()
        indexer.index_many(all_trajectories)
        tasks = indexer.all_tasks()
        assert len(tasks) >= 1

    def test_all_operators(self, all_trajectories):
        indexer = TrajectoryIndexer()
        indexer.index_many(all_trajectories)
        ops = indexer.all_operators()
        assert len(ops) >= 1

    def test_summary(self, all_trajectories):
        indexer = TrajectoryIndexer()
        indexer.index_many(all_trajectories)
        s = indexer.summary()
        assert s["total_indexed"] == 3
        assert "unique_tasks" in s
        assert "fitness_distribution" in s

    def test_clear(self, all_trajectories):
        indexer = TrajectoryIndexer()
        indexer.index_many(all_trajectories)
        indexer.clear()
        assert indexer.total_indexed() == 0

    def test_no_duplicate_indexing(self, solved_trajectory):
        indexer = TrajectoryIndexer()
        indexer.index(solved_trajectory)
        indexer.index(solved_trajectory)
        assert indexer.total_indexed() == 1

    def test_fitness_bucket(self):
        assert TrajectoryIndexer._fitness_bucket(0.0) == "0.0-0.1"
        assert TrajectoryIndexer._fitness_bucket(0.55) == "0.5-0.6"
        assert TrajectoryIndexer._fitness_bucket(1.0) == "1.0-1.0"

    def test_index_no_task(self):
        traj = SearchTrajectory(trajectory_id="no-task")
        indexer = TrajectoryIndexer()
        indexer.index(traj)
        assert indexer.total_indexed() == 1
        assert indexer.all_tasks() == []


class TestCorpusStatistics:
    def test_basic_stats(self, all_trajectories):
        stats = CorpusStatistics(all_trajectories)
        assert stats.total == 3
        assert stats.solved_count == 1
        assert 0.0 <= stats.solve_rate <= 1.0

    def test_fitness_distribution(self, all_trajectories):
        stats = CorpusStatistics(all_trajectories)
        dist = stats.fitness_distribution()
        assert isinstance(dist, dict)
        assert sum(dist.values()) == 3

    def test_operator_distribution(self, all_trajectories):
        stats = CorpusStatistics(all_trajectories)
        dist = stats.operator_distribution()
        assert "mutation" in dist or "init" in dist

    def test_generation_stats(self, all_trajectories):
        stats = CorpusStatistics(all_trajectories)
        gs = stats.generation_stats()
        assert "mean" in gs
        assert "std" in gs
        assert gs["mean"] > 0

    def test_generation_stats_empty(self):
        stats = CorpusStatistics()
        gs = stats.generation_stats()
        assert gs["mean"] == 0.0

    def test_individual_stats(self, all_trajectories):
        stats = CorpusStatistics(all_trajectories)
        is_ = stats.individual_stats()
        assert is_["total"] > 0

    def test_improvement_chain_stats(self, all_trajectories):
        stats = CorpusStatistics(all_trajectories)
        ics = stats.improvement_chain_stats()
        assert "mean" in ics

    def test_difficulty_distribution(self, all_trajectories):
        stats = CorpusStatistics(all_trajectories)
        dd = stats.difficulty_distribution()
        assert isinstance(dd, dict)

    def test_full_report(self, all_trajectories):
        stats = CorpusStatistics(all_trajectories)
        report = stats.full_report()
        assert "total_trajectories" in report
        assert "solve_rate" in report

    def test_add_and_add_many(self):
        stats = CorpusStatistics()
        assert stats.total == 0
        traj = SearchTrajectory(best_fitness=0.5)
        stats.add(traj)
        assert stats.total == 1
        stats.add_many([SearchTrajectory(), SearchTrajectory()])
        assert stats.total == 3

    def test_solve_rate_empty(self):
        stats = CorpusStatistics()
        assert stats.solve_rate == 0.0
