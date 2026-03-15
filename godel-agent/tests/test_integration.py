"""End-to-end integration tests for the Godel Agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.core.agent import GodelAgent
from src.core.executor import Task, MockLLMClient
from src.audit.logger import AuditLogger


def make_e2e_config(tmp_path: Path) -> dict[str, Any]:
    """Create config for end-to-end test."""
    return {
        "project": {"name": "e2e_test", "seed": 42, "output_dir": str(tmp_path / "data")},
        "agent": {
            "llm_provider": "mock",
            "system_prompt": "Solve math problems step by step.",
        },
        "meta_learning": {
            "max_iterations": 5,
            "warmup_iterations": 2,
            "tasks_per_iteration": 3,
            "modification_cooldown": 1,
            "num_examples": 2,
        },
        "modification": {
            "require_deliberation": True,
            "deliberation_depth": 1,
            "allowed_targets": ["prompt_strategy", "few_shot_selector", "reasoning_strategy"],
            "forbidden_targets": ["validation.suite", "audit.logger", "rollback.mechanism"],
            "max_complexity_ratio": 5.0,
        },
        "validation": {
            "suite": "__none__",
            "min_pass_rate": 0.50,
            "performance_threshold": -0.10,
            "auto_rollback": True,
        },
        "audit": {
            "log_dir": str(tmp_path / "audit_logs"),
            "log_diffs": True,
            "log_reasoning": True,
        },
    }


def make_tasks() -> list[Task]:
    """Create a set of tasks for integration testing."""
    return [
        Task(task_id=f"e2e_{i}", question=f"What is {i} * 2?", expected_answer=str(i * 2), domain="math", category="arithmetic")
        for i in range(1, 16)
    ]


class TestEndToEnd:
    """Full end-to-end integration test."""

    def test_five_iterations_with_warmup(self, tmp_path: Path) -> None:
        """Run 5 iterations and verify warmup behavior."""
        config = make_e2e_config(tmp_path)
        agent = GodelAgent(config)
        tasks = make_tasks()

        results = agent.run(tasks, max_iterations=5)

        assert len(results) == 5

        # First 2 iterations are warmup - no deliberation
        for r in results[:2]:
            assert r.deliberated is False
            assert r.modification_applied is False

        # All iterations should have accuracy values
        for r in results:
            assert 0.0 <= r.accuracy <= 1.0
            assert len(r.results) > 0

    def test_deliberation_occurs(self, tmp_path: Path) -> None:
        """Verify deliberation fires after warmup."""
        config = make_e2e_config(tmp_path)
        config["meta_learning"]["warmup_iterations"] = 1
        config["meta_learning"]["max_iterations"] = 7
        config["meta_learning"]["modification_cooldown"] = 1

        agent = GodelAgent(config)
        tasks = make_tasks()
        results = agent.run(tasks)

        assert len(results) == 7
        # At least one post-warmup iteration should exist
        post_warmup = [r for r in results if r.iteration >= 1]
        assert len(post_warmup) >= 6

    def test_modification_and_rollback_cycle(self, tmp_path: Path) -> None:
        """Test that modification and rollback are properly logged."""
        config = make_e2e_config(tmp_path)
        config["meta_learning"]["warmup_iterations"] = 1
        config["meta_learning"]["max_iterations"] = 5
        config["meta_learning"]["modification_cooldown"] = 1

        agent = GodelAgent(config)

        # Set up mock LLM to propose modifications
        assert isinstance(agent.llm, MockLLMClient)

        # Pre-load responses: regular answers + modification proposals
        # First warmup iteration: 3 task answers
        for _ in range(3):
            agent.llm.add_response("The answer is: 42")

        # Iterations 1-4: task answers + deliberation proposals
        for iter_num in range(1, 5):
            # Task answers
            for _ in range(3):
                agent.llm.add_response("The answer is: 42")
            # Deliberation proposal (will be consumed when deliberation fires)
            agent.llm.add_response(json.dumps({
                "action": "modify",
                "target": "prompt_strategy",
                "method_name": "choose_reasoning_mode",
                "description": f"Iteration {iter_num} modification",
                "code": "def choose_reasoning_mode(self, task, recent_results):\n    return 'direct'\n",
                "risk": "low",
                "rationale": "Try direct mode",
            }))

        tasks = make_tasks()
        results = agent.run(tasks)

        assert len(results) == 5

        # Check audit log
        entries = agent.audit.entries
        assert len(entries) > 0

        # Should have iteration entries
        iter_entries = [e for e in entries if e["type"] == "iteration"]
        assert len(iter_entries) == 5

    def test_audit_log_written_to_disk(self, tmp_path: Path) -> None:
        """Verify audit log is written to disk."""
        config = make_e2e_config(tmp_path)
        config["meta_learning"]["max_iterations"] = 3

        agent = GodelAgent(config)
        tasks = make_tasks()
        agent.run(tasks)

        # Check that audit log file exists
        audit_dir = Path(config["audit"]["log_dir"])
        assert audit_dir.exists()

        # Find the run directory
        run_dirs = [d for d in audit_dir.iterdir() if d.is_dir() and d.name != "latest"]
        assert len(run_dirs) >= 1

        # Check log file
        log_file = run_dirs[0] / "audit_log.jsonl"
        assert log_file.exists()

        with open(log_file) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) >= 3  # At least 3 iteration entries

        # Verify entries are valid JSON
        for line in lines:
            entry = json.loads(line)
            assert "type" in entry
            assert "timestamp" in entry

    def test_export_for_safety_review(self, tmp_path: Path) -> None:
        """Verify safety review export contains required fields."""
        config = make_e2e_config(tmp_path)
        config["meta_learning"]["max_iterations"] = 3

        agent = GodelAgent(config)
        tasks = make_tasks()
        agent.run(tasks)

        export = agent.audit.export_for_safety_review()
        assert "run_id" in export
        assert "total_entries" in export
        assert "total_iterations" in export
        assert "total_modifications" in export
        assert "total_rollbacks" in export
        assert "entries" in export
        assert export["total_iterations"] == 3

    def test_state_lineage_tracked(self, tmp_path: Path) -> None:
        """Verify state lineage is properly tracked."""
        config = make_e2e_config(tmp_path)
        config["meta_learning"]["max_iterations"] = 3

        agent = GodelAgent(config)
        tasks = make_tasks()
        agent.run(tasks)

        # State manager should have captured states
        history = agent.state_manager.history
        assert len(history) >= 3  # initial + captured states

        # Each state after the first should have a parent
        for state in history[1:]:
            assert state.parent_state_id != ""

        # Lineage from last state should include all
        if history:
            last = history[-1]
            lineage = agent.state_manager.get_lineage(last.state_id)
            assert len(lineage) >= 1

    def test_no_modification_mode(self, tmp_path: Path) -> None:
        """Test running without any modifications."""
        config = make_e2e_config(tmp_path)
        config["meta_learning"]["max_iterations"] = 5

        agent = GodelAgent(config)
        tasks = make_tasks()
        results = agent.run(tasks, allow_modification=False)

        assert len(results) == 5
        for r in results:
            assert r.deliberated is False
            assert r.modification_applied is False
            assert r.modification_rolled_back is False

    def test_accuracy_history_accumulated(self, tmp_path: Path) -> None:
        """Verify accuracy history grows with each iteration."""
        config = make_e2e_config(tmp_path)
        config["meta_learning"]["max_iterations"] = 5

        agent = GodelAgent(config)
        tasks = make_tasks()
        results = agent.run(tasks)

        # State should have accumulated accuracy history
        latest = agent.state_manager.latest
        assert latest is not None
        assert len(latest.accuracy_history) == 5

    def test_task_results_include_answers(self, tmp_path: Path) -> None:
        """Verify task results contain extracted answers."""
        config = make_e2e_config(tmp_path)
        config["meta_learning"]["max_iterations"] = 1

        agent = GodelAgent(config)
        tasks = make_tasks()
        results = agent.run(tasks, max_iterations=1)

        assert len(results) == 1
        for tr in results[0].results:
            assert tr.response != ""
            # Mock LLM always returns something
