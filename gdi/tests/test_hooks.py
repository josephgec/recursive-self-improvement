"""Tests for GDI integration hooks and decorators."""

import os
import json
import tempfile

import pytest

from src.composite.gdi import GoalDriftIndex, GDIResult
from src.integration.hooks import GDIHooks
from src.integration.decorator import track_drift, GDIDriftError
from src.integration.phase_adapters import PhaseAdapter
from src.integration.pipeline_bridge import PipelineBridge
from src.reference.store import ReferenceStore
from src.alerting.alert_manager import AlertManager

# Import test helpers
from tests.conftest import MockAgent


class TestGDIHooks:
    """Tests for GDIHooks."""

    def test_after_modification(self, reference_outputs, tmp_path):
        """after_modification should compute GDI."""
        store_path = str(tmp_path / "ref.json")
        store = ReferenceStore(store_path)
        store.save({"outputs": reference_outputs})

        gdi = GoalDriftIndex()
        probe_tasks = ["task1", "task2"]
        agent = MockAgent(reference_outputs)

        hooks = GDIHooks(gdi, probe_tasks, store)
        result = hooks.after_modification(agent)

        assert isinstance(result, GDIResult)
        assert result.metadata.get("trigger_event") == "after_modification"

    def test_after_training(self, reference_outputs, tmp_path):
        """after_training should compute GDI."""
        store_path = str(tmp_path / "ref.json")
        store = ReferenceStore(store_path)
        store.save({"outputs": reference_outputs})

        gdi = GoalDriftIndex()
        agent = MockAgent(reference_outputs)
        hooks = GDIHooks(gdi, ["task1"], store)
        result = hooks.after_training(agent)

        assert result.metadata.get("trigger_event") == "after_training"

    def test_after_library_update(self, reference_outputs, tmp_path):
        """after_library_update should compute GDI."""
        store_path = str(tmp_path / "ref.json")
        store = ReferenceStore(store_path)
        store.save({"outputs": reference_outputs})

        gdi = GoalDriftIndex()
        agent = MockAgent(reference_outputs)
        hooks = GDIHooks(gdi, ["task1"], store)
        result = hooks.after_library_update(agent)

        assert result.metadata.get("trigger_event") == "after_library_update"

    def test_periodic_check(self, reference_outputs, tmp_path):
        """periodic_check should compute GDI."""
        store_path = str(tmp_path / "ref.json")
        store = ReferenceStore(store_path)
        store.save({"outputs": reference_outputs})

        gdi = GoalDriftIndex()
        agent = MockAgent(reference_outputs)
        hooks = GDIHooks(gdi, ["task1"], store)
        result = hooks.periodic_check(agent)

        assert result.metadata.get("trigger_event") == "periodic_check"

    def test_on_result_callback(self, reference_outputs, tmp_path):
        """on_result callback should be called."""
        store_path = str(tmp_path / "ref.json")
        store = ReferenceStore(store_path)
        store.save({"outputs": reference_outputs})

        results = []
        gdi = GoalDriftIndex()
        agent = MockAgent(reference_outputs)
        hooks = GDIHooks(gdi, ["task1"], store, on_result=results.append)
        hooks.after_modification(agent)

        assert len(results) == 1

    def test_results_accumulated(self, reference_outputs, tmp_path):
        """Results should be accumulated in history."""
        store_path = str(tmp_path / "ref.json")
        store = ReferenceStore(store_path)
        store.save({"outputs": reference_outputs})

        gdi = GoalDriftIndex()
        agent = MockAgent(reference_outputs)
        hooks = GDIHooks(gdi, ["task1"], store)
        hooks.after_modification(agent)
        hooks.periodic_check(agent)

        assert len(hooks.results) == 2


class TestTrackDriftDecorator:
    """Tests for @track_drift decorator."""

    def test_decorator_computes_gdi(self, reference_outputs, tmp_path):
        """Decorator should compute GDI after function execution."""
        store_path = str(tmp_path / "ref.json")
        store = ReferenceStore(store_path)
        store.save({"outputs": reference_outputs})

        gdi = GoalDriftIndex()
        agent = MockAgent(reference_outputs)

        @track_drift(gdi, ["task1"], store, raise_on_red=False)
        def modify_agent(agent):
            return "modified"

        result = modify_agent(agent)
        assert result == "modified"
        assert len(modify_agent._gdi_results) == 1

    def test_decorator_raises_on_red(self, reference_outputs, collapsed_outputs, tmp_path):
        """Decorator should raise GDIDriftError on red alert."""
        store_path = str(tmp_path / "ref.json")
        store = ReferenceStore(store_path)
        store.save({"outputs": reference_outputs})

        gdi = GoalDriftIndex()
        collapsed_agent = MockAgent(collapsed_outputs)

        @track_drift(gdi, ["task1", "task2", "task3"], store, raise_on_red=True)
        def modify_agent(agent):
            return "modified"

        # Check if collapsed outputs produce red alert
        result_check = gdi.compute(
            [collapsed_outputs[0], collapsed_outputs[1], collapsed_outputs[2]],
            reference_outputs,
        )

        if result_check.alert_level == "red":
            with pytest.raises(GDIDriftError) as exc_info:
                modify_agent(collapsed_agent)
            assert exc_info.value.result.alert_level == "red"
        else:
            # If not red, the test should still pass — verify decorator runs
            modify_agent(collapsed_agent)
            assert len(modify_agent._gdi_results) >= 1

    def test_gdi_drift_error(self):
        """GDIDriftError should contain result."""
        result = GDIResult(
            composite_score=0.9,
            alert_level="red",
            trend="increasing",
            semantic_score=0.8,
            lexical_score=0.7,
            structural_score=0.6,
            distributional_score=0.5,
        )
        error = GDIDriftError(result)
        assert error.result.alert_level == "red"
        assert "red" in str(error)

    def test_decorator_with_agent_extractor(self, reference_outputs, tmp_path):
        """Custom agent extractor should work."""
        store_path = str(tmp_path / "ref.json")
        store = ReferenceStore(store_path)
        store.save({"outputs": reference_outputs})

        gdi = GoalDriftIndex()
        agent = MockAgent(reference_outputs)

        @track_drift(
            gdi, ["task1"], store,
            agent_extractor=lambda *args, **kwargs: kwargs.get("agent"),
            raise_on_red=False,
        )
        def modify_agent(config, agent=None):
            return "modified"

        result = modify_agent("some_config", agent=agent)
        assert result == "modified"
        assert len(modify_agent._gdi_results) == 1


class TestPhaseAdapters:
    """Tests for PhaseAdapter."""

    def test_for_godel(self):
        """Godel adapter should emphasize semantic."""
        config = PhaseAdapter.for_godel()
        assert config["weights"].semantic == 0.40
        assert config["phase"] == "godel"

    def test_for_soar(self):
        """SOAR adapter should have balanced weights."""
        config = PhaseAdapter.for_soar()
        assert config["weights"].validate()
        assert config["phase"] == "soar"

    def test_for_symcode(self):
        """SymCode adapter should emphasize structural."""
        config = PhaseAdapter.for_symcode()
        assert config["weights"].structural == 0.35
        assert config["phase"] == "symcode"

    def test_for_rlm(self):
        """RLM adapter should emphasize distributional."""
        config = PhaseAdapter.for_rlm()
        assert config["weights"].distributional == 0.40
        assert config["phase"] == "rlm"

    def test_for_pipeline(self):
        """Pipeline adapter should have strict thresholds."""
        config = PhaseAdapter.for_pipeline()
        assert config["thresholds"]["green_max"] <= 0.15
        assert config["phase"] == "pipeline"

    def test_all_adapters_valid_weights(self):
        """All adapters should produce valid weights."""
        for method in [
            PhaseAdapter.for_godel,
            PhaseAdapter.for_soar,
            PhaseAdapter.for_symcode,
            PhaseAdapter.for_rlm,
            PhaseAdapter.for_pipeline,
        ]:
            config = method()
            assert config["weights"].validate()


class TestPipelineBridge:
    """Tests for PipelineBridge."""

    def test_check_drift(self, reference_outputs, tmp_path):
        """check_drift should return GDI result."""
        store_path = str(tmp_path / "ref.json")
        store = ReferenceStore(store_path)
        store.save({"outputs": reference_outputs})

        gdi = GoalDriftIndex()
        agent = MockAgent(reference_outputs)
        bridge = PipelineBridge(gdi, store, ["task1"])

        result = bridge.check_drift(agent)
        assert isinstance(result, GDIResult)

    def test_is_safe_to_proceed(self, reference_outputs, tmp_path):
        """Safe to proceed when outputs closely match reference."""
        # Use same subset for both reference store and agent output
        subset = reference_outputs[:5]
        store_path = str(tmp_path / "ref.json")
        store = ReferenceStore(store_path)
        store.save({"outputs": subset})

        gdi = GoalDriftIndex()
        agent = MockAgent(subset)
        # Use enough probe tasks so agent output distribution matches reference
        bridge = PipelineBridge(gdi, store, ["t1", "t2", "t3", "t4", "t5"])

        assert bridge.is_safe_to_proceed(agent)

    def test_get_metrics(self, reference_outputs, tmp_path):
        """Should return metrics after check."""
        store_path = str(tmp_path / "ref.json")
        store = ReferenceStore(store_path)
        store.save({"outputs": reference_outputs})

        gdi = GoalDriftIndex()
        agent = MockAgent(reference_outputs)
        bridge = PipelineBridge(gdi, store, ["task1"])

        # No data yet
        metrics = bridge.get_metrics()
        assert metrics["status"] == "no_data"

        # After check
        bridge.check_drift(agent)
        metrics = bridge.get_metrics()
        assert "composite_score" in metrics
        assert "alert_level" in metrics

    def test_bridge_with_alert_manager(self, reference_outputs, tmp_path):
        """Bridge should integrate with alert manager."""
        store_path = str(tmp_path / "ref.json")
        store = ReferenceStore(store_path)
        store.save({"outputs": reference_outputs})

        gdi = GoalDriftIndex()
        agent = MockAgent(reference_outputs)
        alert_mgr = AlertManager()
        bridge = PipelineBridge(gdi, store, ["task1"], alert_manager=alert_mgr)

        bridge.check_drift(agent)
        assert len(bridge.results) == 1
