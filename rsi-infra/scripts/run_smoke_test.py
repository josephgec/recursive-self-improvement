#!/usr/bin/env python3
"""Smoke test for rsi-infra — validates all subsystems work end-to-end.

Run directly::

    python scripts/run_smoke_test.py

Or with specific tests::

    python scripts/run_smoke_test.py repl
    python scripts/run_smoke_test.py symbolic
    python scripts/run_smoke_test.py tracking
    python scripts/run_smoke_test.py cross
    python scripts/run_smoke_test.py all
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import typer
except ImportError:
    typer = None  # type: ignore[assignment]

from sdk.config import InfraConfig
from sdk.repl_client import REPLClient
from sdk.symbolic_client import SymbolicClient
from sdk.tracking_client import TrackingClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_passed = 0
_failed = 0


def _ensure_event_loop() -> None:
    """Ensure an event loop exists on the current thread (needed for Python 3.9)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


def _report(name: str, ok: bool, detail: str = "") -> None:
    global _passed, _failed
    status = "PASS" if ok else "FAIL"
    suffix = f" -- {detail}" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    if ok:
        _passed += 1
    else:
        _failed += 1


# ---------------------------------------------------------------------------
# Test: REPL lifecycle
# ---------------------------------------------------------------------------

def test_repl_lifecycle() -> None:
    """Create -> execute -> persist var -> spawn child -> verify isolation -> shutdown."""
    print("\n=== REPL lifecycle ===")
    _ensure_event_loop()
    config = InfraConfig.from_yaml(_PROJECT_ROOT / "configs" / "local.yaml")
    client = REPLClient.from_config(config)

    async def _run() -> None:
        # 1. Basic execution
        async with client.session() as repl:
            r = repl.execute("x = 40 + 2")
            _report("execute basic", r.success)

            # 2. Persistent variable
            r2 = repl.execute("y = x * 2")
            _report("persist variable", r2.success and r2.variables.get("y") == 84)

            # 3. Spawn child
            async with client.child(repl) as child:
                # Child inherits x
                r3 = child.execute("z = x + 1")
                _report("child inherits parent var", r3.success and r3.variables.get("z") == 43)

                # Mutate in child
                child.execute("x = 999")
                _report("child mutation isolated",
                        repl.get_variable("x") == 42)

            # 4. Parent still works after child shutdown
            r4 = repl.execute("w = x + 10")
            _report("parent ok after child", r4.success and r4.variables.get("w") == 52)

        await client.shutdown()

    asyncio.get_event_loop().run_until_complete(_run())


# ---------------------------------------------------------------------------
# Test: REPL security
# ---------------------------------------------------------------------------

def test_repl_security() -> None:
    """Blocked imports, blocked builtins."""
    print("\n=== REPL security ===")
    _ensure_event_loop()
    config = InfraConfig.from_yaml(_PROJECT_ROOT / "configs" / "local.yaml")
    client = REPLClient.from_config(config)

    async def _run() -> None:
        async with client.session() as repl:
            # Blocked import
            r = repl.execute("import os")
            _report("block import os", not r.success)

            # Blocked subprocess
            r2 = repl.execute("import subprocess")
            _report("block import subprocess", not r2.success)

            # Allowed import
            r3 = repl.execute("import math; pi = math.pi")
            _report("allow import math", r3.success)

        await client.shutdown()

    asyncio.get_event_loop().run_until_complete(_run())


# ---------------------------------------------------------------------------
# Test: Symbolic pipeline
# ---------------------------------------------------------------------------

def test_symbolic_pipeline() -> None:
    """SymPy solve -> Z3 check -> verify."""
    print("\n=== Symbolic pipeline ===")
    config = InfraConfig.from_yaml(_PROJECT_ROOT / "configs" / "local.yaml")
    client = SymbolicClient.from_config(config)

    # 1. SymPy solve
    result = client.solve("x**2 - 4", "x")
    _report("sympy solve x^2-4", result.success and result.expression is not None)

    # 2. Z3 implication check
    holds = client.check_implication(["x > 0", "x < 10"], "x < 10")
    _report("z3 implication", holds)

    # 3. Verify code against expected
    vr = client.verify_code("result = 2 + 2", expected="4")
    _report("verify_code 2+2=4", vr.passed)


# ---------------------------------------------------------------------------
# Test: Tracking pipeline
# ---------------------------------------------------------------------------

def test_tracking_pipeline() -> None:
    """Init -> log -> GDI -> constraints -> finish."""
    print("\n=== Tracking pipeline ===")
    tmp = tempfile.mkdtemp(prefix="rsi_smoke_")
    try:
        config = InfraConfig.from_yaml(_PROJECT_ROOT / "configs" / "local.yaml")
        client = TrackingClient.from_config(config)
        # Point tracker at temp dir
        from tracking.src.local_backend import LocalTracker
        if isinstance(client.tracker, LocalTracker):
            client.tracker._base_dir = Path(tmp)

        client.start_run("smoke-test", {"lr": 0.001})
        _report("start_run", True)

        # Log generation 0
        gen0_metrics = {"loss": 2.0, "accuracy": 0.7, "safety_score": 0.95}
        client.log_generation(0, gen0_metrics)
        _report("log_generation 0", True)

        # Set reference texts
        ref_texts = [
            "The model should solve math problems accurately.",
            "Safety constraints must be preserved across generations.",
        ]
        client.set_reference(ref_texts)
        _report("set_reference", True)

        # Check safety on generation 1
        gen1_texts = [
            "The model should solve math problems accurately and efficiently.",
            "Safety constraints must be preserved across all iterations.",
        ]
        gen1_metrics = {"loss": 1.8, "accuracy": 0.75, "safety_score": 0.93}
        client.log_generation(1, gen1_metrics)
        safety = client.check_safety(1, gen1_texts, gen1_metrics)
        _report("check_safety gen 1", safety.drift is not None and safety.constraints is not None)
        _report("constraints passed", safety.constraints.all_passed if safety.constraints else False)

        client.finish()
        _report("finish", True)

        # Verify files created
        run_dir = Path(tmp) / "smoke-test"
        _report("tracking files exist",
                (run_dir / "run_meta.json").exists()
                and (run_dir / "metrics.jsonl").exists())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test: Cross-system
# ---------------------------------------------------------------------------

def test_cross_system() -> None:
    """Execute code in REPL that does math, log result via tracking."""
    print("\n=== Cross-system ===")
    _ensure_event_loop()
    tmp = tempfile.mkdtemp(prefix="rsi_cross_")
    try:
        config = InfraConfig.from_yaml(_PROJECT_ROOT / "configs" / "local.yaml")
        repl_client = REPLClient.from_config(config)
        tracking_client = TrackingClient.from_config(config)

        from tracking.src.local_backend import LocalTracker
        if isinstance(tracking_client.tracker, LocalTracker):
            tracking_client.tracker._base_dir = Path(tmp)

        tracking_client.start_run("cross-system-test")

        async def _run() -> None:
            async with repl_client.session() as repl:
                # Execute a math computation
                r = repl.execute("import math\nresult = math.factorial(10)")
                _report("repl compute factorial", r.success)

                result_val = r.variables.get("result")
                _report("factorial(10) == 3628800", result_val == 3628800)

                # Log it
                tracking_client.log_generation(0, {
                    "accuracy": 1.0 if result_val == 3628800 else 0.0,
                    "safety_score": 0.99,
                    "loss": 0.1,
                })
                _report("cross-system log", True)

            await repl_client.shutdown()

        asyncio.get_event_loop().run_until_complete(_run())

        tracking_client.finish()
        _report("cross-system finish", True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_TESTS = {
    "repl": [test_repl_lifecycle, test_repl_security],
    "symbolic": [test_symbolic_pipeline],
    "tracking": [test_tracking_pipeline],
    "cross": [test_cross_system],
}


def run_tests(suite: str = "all") -> None:
    """Run smoke tests.  *suite* is one of: all, repl, symbolic, tracking, cross."""
    global _passed, _failed
    _passed = 0
    _failed = 0

    if suite == "all":
        tests = [t for group in ALL_TESTS.values() for t in group]
    elif suite in ALL_TESTS:
        tests = ALL_TESTS[suite]
    else:
        print(f"Unknown suite '{suite}'. Choose from: all, {', '.join(ALL_TESTS)}")
        sys.exit(1)

    for test_fn in tests:
        try:
            test_fn()
        except Exception:
            _report(test_fn.__name__, False, traceback.format_exc().splitlines()[-1])

    print(f"\n{'='*40}")
    print(f"Results: {_passed} passed, {_failed} failed")
    if _failed > 0:
        sys.exit(1)


def main() -> None:
    if typer is not None:
        app = typer.Typer(add_completion=False)

        @app.command()
        def smoke(suite: str = typer.Argument("all", help="Test suite: all, repl, symbolic, tracking, cross")):
            """Run rsi-infra smoke tests."""
            run_tests(suite)

        app()
    else:
        suite = sys.argv[1] if len(sys.argv) > 1 else "all"
        run_tests(suite)


if __name__ == "__main__":
    main()
