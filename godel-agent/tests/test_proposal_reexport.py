"""Tests for re-export modules: proposal, base (tasks)."""

from __future__ import annotations


class TestProposalReExport:
    def test_import_modification_proposal(self) -> None:
        from src.modification.proposal import ModificationProposal
        assert ModificationProposal is not None
        # Verify it's the same class
        from src.modification.modifier import ModificationProposal as MP2
        assert ModificationProposal is MP2


class TestTaskBaseReExport:
    def test_import_task(self) -> None:
        from src.tasks.base import Task
        assert Task is not None
        from src.core.executor import Task as T2
        assert Task is T2

    def test_import_task_result(self) -> None:
        from src.tasks.base import TaskResult
        assert TaskResult is not None
        from src.core.executor import TaskResult as TR2
        assert TaskResult is TR2
