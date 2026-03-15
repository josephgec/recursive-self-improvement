"""Integration test — simulates a mini RSI loop using all subsystems.

The test uses only local backends (LocalREPL, subprocess symbolic runners,
LocalTracker) so it runs without Docker, Modal, or wandb.

Test flow:
  1. Start tracking run.
  2. Create REPL, execute a "model" (simple transform function).
  3. Generate "synthetic data", verify via symbolic.
  4. Compute GDI, check constraints.
  5. Log everything, verify all infrastructure works together.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from repl.src.local_repl import LocalREPL
from repl.src.sandbox import ExecutionResult
from sdk.config import InfraConfig
from sdk.repl_client import REPLClient
from sdk.symbolic_client import SymbolicClient
from sdk.tracking_client import TrackingClient
from tracking.src.local_backend import LocalTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _make_tracking_client(tmp_tracking_dir: Path) -> TrackingClient:
    """Create a TrackingClient writing to a temp directory."""
    config = InfraConfig.from_yaml(_PROJECT_ROOT / "configs" / "local.yaml")
    client = TrackingClient.from_config(config)
    if isinstance(client.tracker, LocalTracker):
        client.tracker._base_dir = tmp_tracking_dir
    return client


# ---------------------------------------------------------------------------
# Integration test: mini RSI loop
# ---------------------------------------------------------------------------

class TestMiniRSILoop:
    """Simulate a minimal recursive self-improvement loop across all systems."""

    def test_full_loop(self, tmp_tracking_dir: Path) -> None:
        """End-to-end: REPL + symbolic + tracking cooperate in a loop."""
        config = InfraConfig.from_yaml(_PROJECT_ROOT / "configs" / "local.yaml")

        # ---- 1. Initialise tracking ----
        tracker = _make_tracking_client(tmp_tracking_dir)
        tracker.start_run("mini-rsi-loop", {"approach": "integration-test"})

        # Set reference texts for drift computation
        reference_texts = [
            "The model transforms inputs by doubling and adding one.",
            "Expected behaviour is deterministic and verifiable.",
        ]
        tracker.set_reference(reference_texts)

        # ---- 2. Create REPL and execute "model" ----
        repl = LocalREPL(config=config.repl_config)
        try:
            # Define a simple "model" function
            model_code = """
def transform(x):
    return 2 * x + 1
"""
            r = repl.execute(model_code)
            assert r.success, f"Model definition failed: {r.error_message}"

            # Execute the model on test inputs
            test_code = """
results = [transform(i) for i in range(5)]
"""
            r = repl.execute(test_code)
            assert r.success, f"Model execution failed: {r.error_message}"
            results = repl.get_variable("results")
            assert results == [1, 3, 5, 7, 9], f"Unexpected results: {results}"

            # Log generation 0 metrics
            gen0_metrics = {
                "loss": 0.5,
                "accuracy": 0.9,
                "safety_score": 0.95,
                "perplexity": 10.0,
            }
            tracker.log_generation(0, gen0_metrics)

            # ---- 3. Generate "synthetic data" and verify ----
            # Simulate model generating outputs
            gen_code = """
synthetic = [transform(x) for x in [10, 20, 30]]
"""
            r = repl.execute(gen_code)
            assert r.success
            synthetic = repl.get_variable("synthetic")
            assert synthetic == [21, 41, 61]

            # ---- 4. Verify with symbolic engine ----
            sym_client = SymbolicClient.from_config(config)

            # Verify that our transform matches 2*x+1
            vr = sym_client.verify_code("result = 2 * 10 + 1", expected="21")
            assert vr.passed, f"Symbolic verification failed: {vr.details}"

            # Solve the inverse: find x such that 2x+1 = 21
            solve_result = sym_client.solve("2*x + 1 - 21", "x")
            assert solve_result.success, f"SymPy solve failed: {solve_result.error}"

            # Z3 implication: if x > 0, then 2*x + 1 > 1
            implies = sym_client.check_implication(["x > 0"], "2*x + 1 > 1")
            assert implies, "Z3 implication check failed"

            # ---- 5. Log generation 1 with updated metrics ----
            gen1_metrics = {
                "loss": 0.3,
                "accuracy": 0.92,
                "safety_score": 0.94,
                "perplexity": 8.0,
            }
            tracker.log_generation(1, gen1_metrics)

            # Generated texts for drift
            gen1_texts = [
                "The model transforms inputs by doubling and adding one.",
                "The transform function produces deterministic outputs.",
            ]

            safety = tracker.check_safety(1, gen1_texts, gen1_metrics)
            assert safety.drift is not None, "No drift measurement"
            assert safety.constraints is not None, "No constraints report"
            assert safety.constraints.all_passed, (
                f"Constraints failed: {safety.constraints.recommendation}"
            )

            # ---- 6. Simulate generation 2 with spawn child ----
            child = repl.spawn_child()
            try:
                # "Improve" the model in the child
                child.execute("""
def transform(x):
    return 2 * x + 1  # same function, verified correct
""")
                r = child.execute("improved_results = [transform(i) for i in range(5)]")
                assert r.success
                improved = child.get_variable("improved_results")
                assert improved == [1, 3, 5, 7, 9]

                # Parent unaffected
                assert repl.get_variable("results") == [1, 3, 5, 7, 9]
            finally:
                child.shutdown()

            gen2_metrics = {
                "loss": 0.25,
                "accuracy": 0.94,
                "safety_score": 0.93,
                "perplexity": 7.0,
            }
            tracker.log_generation(2, gen2_metrics)

            gen2_texts = [
                "The model transforms inputs by doubling and adding one.",
                "Outputs are verified and deterministic for all inputs.",
            ]
            safety2 = tracker.check_safety(2, gen2_texts, gen2_metrics)
            assert safety2.constraints is not None
            assert safety2.constraints.all_passed

        finally:
            repl.shutdown()

        # ---- 7. Finish tracking and verify artefacts ----
        tracker.finish()

        run_dir = tmp_tracking_dir / "mini-rsi-loop"
        assert run_dir.exists(), "Run directory was not created"
        assert (run_dir / "run_meta.json").exists()
        assert (run_dir / "metrics.jsonl").exists()
        assert (run_dir / "drift.jsonl").exists()
        assert (run_dir / "constraints.jsonl").exists()
        assert (run_dir / "run_finished.json").exists()

        # Verify metrics.jsonl has multiple entries
        with open(run_dir / "metrics.jsonl") as f:
            lines = f.readlines()
        assert len(lines) >= 3, f"Expected >=3 metric entries, got {len(lines)}"

        # Verify drift.jsonl has entries
        with open(run_dir / "drift.jsonl") as f:
            drift_lines = f.readlines()
        assert len(drift_lines) >= 2, f"Expected >=2 drift entries, got {len(drift_lines)}"

        # Parse first drift entry and check structure
        first_drift = json.loads(drift_lines[0])
        assert "goal_drift_index" in first_drift
        assert "semantic_drift" in first_drift

    def test_repl_security_in_loop(self, tmp_tracking_dir: Path) -> None:
        """Verify that security holds during a simulated loop iteration."""
        config = InfraConfig.from_yaml(_PROJECT_ROOT / "configs" / "local.yaml")
        repl = LocalREPL(config=config.repl_config)

        try:
            # Good code works
            r = repl.execute("x = 42")
            assert r.success

            # Dangerous code is blocked
            r = repl.execute("import os; os.system('whoami')")
            assert not r.success
            assert "blocked" in (r.error_message or "").lower()

            # After a security block, REPL still works
            r = repl.execute("y = x + 1")
            assert r.success
            assert repl.get_variable("y") == 43
        finally:
            repl.shutdown()

    def test_constraint_violation_detection(self, tmp_tracking_dir: Path) -> None:
        """Verify that constraint violations are detected and reported."""
        tracker = _make_tracking_client(tmp_tracking_dir)
        tracker.start_run("constraint-violation-test")

        tracker.set_reference(["reference text for testing"])

        # Generation with bad safety score (below 0.9 threshold)
        bad_metrics = {
            "loss": 0.5,
            "accuracy": 0.8,
            "safety_score": 0.5,  # Below the 0.9 halt threshold
        }
        tracker.log_generation(0, bad_metrics)
        safety = tracker.check_safety(
            0,
            ["some generated text"],
            bad_metrics,
        )
        assert safety.constraints is not None
        assert not safety.constraints.all_passed
        assert safety.constraints.recommendation == "halt"
        assert not safety.safe

        tracker.finish()

    def test_drift_detection(self, tmp_tracking_dir: Path) -> None:
        """Verify that goal drift is computed and tracked."""
        tracker = _make_tracking_client(tmp_tracking_dir)
        tracker.start_run("drift-detection-test")

        # Reference: math-focused text
        ref = [
            "Solve quadratic equations using the quadratic formula.",
            "Factor polynomials to find roots.",
        ]
        tracker.set_reference(ref)

        # Generation 1: similar text (low drift)
        similar = [
            "Use the quadratic formula to solve equations.",
            "Find polynomial roots by factoring.",
        ]
        metrics1 = {"accuracy": 0.9, "safety_score": 0.95}
        tracker.log_generation(1, metrics1)
        safety1 = tracker.check_safety(1, similar, metrics1)
        assert safety1.drift is not None
        low_gdi = safety1.drift.goal_drift_index

        # Generation 2: very different text (higher drift)
        different = [
            "The weather is sunny today with a high of 75 degrees.",
            "Cooking recipes involve precise measurements of ingredients.",
        ]
        metrics2 = {"accuracy": 0.85, "safety_score": 0.93}
        tracker.log_generation(2, metrics2)
        safety2 = tracker.check_safety(2, different, metrics2)
        assert safety2.drift is not None
        high_gdi = safety2.drift.goal_drift_index

        # Divergent text should have higher drift
        assert high_gdi > low_gdi, (
            f"Expected higher drift for divergent text: {high_gdi} vs {low_gdi}"
        )

        tracker.finish()

    def test_capability_alignment_ratio(self, tmp_tracking_dir: Path) -> None:
        """Verify that CAR is computed across generations."""
        tracker = _make_tracking_client(tmp_tracking_dir)
        tracker.start_run("car-test")

        # Gen 0
        tracker.log_generation(0, {"accuracy": 0.8, "safety_score": 0.9})
        # Gen 1: capability up, alignment stable
        tracker.log_generation(1, {"accuracy": 0.85, "safety_score": 0.9})

        trajectory = tracker.car_tracker.get_trajectory()
        assert len(trajectory) == 1
        m = trajectory[0]
        assert m.generation == 1
        assert m.capability_gain > 0
        assert m.pareto_improving  # alignment didn't degrade

        tracker.finish()


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    """Test the InfraConfig loader."""

    def test_from_yaml_local(self) -> None:
        config = InfraConfig.from_yaml(_PROJECT_ROOT / "configs" / "local.yaml")
        assert config.repl_backend == "local"
        assert config.symbolic_backend == "subprocess"
        assert config.tracking_backend == "local"

    def test_from_yaml_default(self) -> None:
        config = InfraConfig.from_yaml(_PROJECT_ROOT / "configs" / "default.yaml")
        assert config.repl_config.timeout_seconds == 300
        assert config.repl_pool_size == 4

    def test_from_yaml_overlay(self) -> None:
        """CI overlay should override pool_size."""
        config = InfraConfig.from_yaml(_PROJECT_ROOT / "configs" / "ci.yaml")
        assert config.repl_pool_size == 1
        assert config.repl_config.timeout_seconds == 30

    def test_from_env_defaults_to_local(self, monkeypatch) -> None:
        monkeypatch.delenv("RSI_ENV", raising=False)
        config = InfraConfig.from_env()
        assert config.repl_backend == "local"

    def test_from_env_override(self, monkeypatch) -> None:
        monkeypatch.setenv("RSI_ENV", "local")
        monkeypatch.setenv("RSI_REPL_BACKEND", "docker")
        config = InfraConfig.from_env()
        assert config.repl_backend == "docker"


# ---------------------------------------------------------------------------
# REPL client tests
# ---------------------------------------------------------------------------

class TestREPLClient:
    """Test the REPLClient SDK layer."""

    def test_session_lifecycle(self, repl_client: REPLClient) -> None:
        loop = asyncio.get_event_loop()

        async def _run():
            async with repl_client.session() as repl:
                r = repl.execute("x = 42")
                assert r.success
                assert repl.get_variable("x") == 42

        loop.run_until_complete(_run())

    def test_execute_convenience(self, repl_client: REPLClient) -> None:
        loop = asyncio.get_event_loop()

        async def _run():
            r = await repl_client.execute("x = 7 * 6")
            assert r.success
            assert r.variables.get("x") == 42

        loop.run_until_complete(_run())

    def test_child_isolation(self, repl_client: REPLClient) -> None:
        loop = asyncio.get_event_loop()

        async def _run():
            async with repl_client.session() as parent:
                parent.execute("val = 100")
                async with repl_client.child(parent) as child:
                    child.execute("val = 999")
                    assert child.get_variable("val") == 999
                assert parent.get_variable("val") == 100

        loop.run_until_complete(_run())

    def test_session_sync(self, repl_client: REPLClient) -> None:
        """Synchronous session context manager works."""
        with repl_client.session_sync() as repl:
            result = repl.execute("x = 7 * 7")
            assert result.success
            assert repl.get_variable("x") == 49

    def test_execute_sync(self, repl_client: REPLClient) -> None:
        """Synchronous one-shot execute works."""
        result = repl_client.execute_sync("x = 6 * 7")
        assert result.success
        assert result.variables.get("x") == 42

    def test_child_sync(self, repl_client: REPLClient) -> None:
        """Synchronous child context manager works."""
        with repl_client.session_sync() as parent:
            parent.execute("val = 50")
            with repl_client.child_sync(parent) as child:
                child.execute("val = 999")
                assert child.get_variable("val") == 999
            assert parent.get_variable("val") == 50

    def test_shutdown_sync(self, repl_client: REPLClient) -> None:
        """Synchronous shutdown does not raise."""
        # Create a separate client to avoid interfering with fixture cleanup
        from sdk.config import InfraConfig
        config = InfraConfig.from_yaml(_PROJECT_ROOT / "configs" / "local.yaml")
        client2 = REPLClient.from_config(config)
        client2.shutdown_sync()  # Should not raise

    def test_pool_property(self, repl_client: REPLClient) -> None:
        """pool property returns the underlying REPLPool."""
        from repl.src.pool import REPLPool
        assert isinstance(repl_client.pool, REPLPool)


# ---------------------------------------------------------------------------
# Symbolic client tests
# ---------------------------------------------------------------------------

class TestSymbolicClient:
    """Test the SymbolicClient SDK layer."""

    def test_solve(self, symbolic_client: SymbolicClient) -> None:
        result = symbolic_client.solve("x**2 - 9", "x")
        assert result.success
        assert result.expression is not None

    def test_verify_code(self, symbolic_client: SymbolicClient) -> None:
        vr = symbolic_client.verify_code("result = 6 * 7", expected="42")
        assert vr.passed

    def test_check_implication(self, symbolic_client: SymbolicClient) -> None:
        assert symbolic_client.check_implication(
            ["x > 5"], "x > 3"
        )

    def test_check_sat(self, symbolic_client: SymbolicClient) -> None:
        code = """
s = Solver()
x = Int('x')
s.add(x > 0, x < 5)
"""
        result = symbolic_client.check_sat(code)
        assert result.satisfiable is True


# ---------------------------------------------------------------------------
# Tracking client tests
# ---------------------------------------------------------------------------

class TestTrackingClient:
    """Test the TrackingClient SDK layer."""

    def test_start_log_finish(self, tracking_client: TrackingClient, tmp_tracking_dir: Path) -> None:
        tracking_client.start_run("basic-test")
        tracking_client.log_generation(0, {"loss": 1.0, "accuracy": 0.5})
        tracking_client.finish()

        run_dir = tmp_tracking_dir / "basic-test"
        assert (run_dir / "run_meta.json").exists()
        assert (run_dir / "metrics.jsonl").exists()
        assert (run_dir / "run_finished.json").exists()

    def test_safety_check(self, tracking_client: TrackingClient) -> None:
        tracking_client.start_run("safety-test")
        # Use identical texts so drift is ~0
        ref_text = "The model should solve math problems accurately."
        tracking_client.set_reference([ref_text])
        tracking_client.log_generation(0, {"accuracy": 0.9, "safety_score": 0.95})

        safety = tracking_client.check_safety(
            0,
            [ref_text],  # identical to reference => negligible drift
            {"accuracy": 0.9, "safety_score": 0.95},
        )
        assert safety.drift is not None
        assert safety.constraints is not None
        assert safety.constraints.all_passed
        assert safety.safe
        tracking_client.finish()
