"""Shared subprocess execution helper for symbolic runners.

Runs arbitrary Python code in a fresh subprocess with timeout isolation,
so that hangs, segfaults, or excessive memory usage cannot affect the
parent process.
"""

from __future__ import annotations

import multiprocessing
import pickle
import time
import traceback
from typing import Any


def _worker(
    code: str,
    namespace_setup: str,
    result_queue: multiprocessing.Queue,
) -> None:
    """Target function executed inside a child process.

    Runs *namespace_setup* to populate globals, then executes *code*.
    Only variables introduced or modified by *code* (not setup) are returned.
    Puts a result dict onto *result_queue*.
    """
    namespace: dict[str, Any] = {}
    try:
        # Execute setup code (e.g. "from sympy import *")
        if namespace_setup:
            exec(namespace_setup, namespace)  # noqa: S102

        # Snapshot keys that exist after setup so we can exclude them later
        setup_keys = set(namespace.keys())

        exec(code, namespace)  # noqa: S102

        # Collect only variables introduced/modified by user code
        variables: dict[str, Any] = {}
        for k, v in namespace.items():
            if k in setup_keys:
                continue
            if k.startswith("__"):
                continue
            try:
                pickle.dumps(v)
                variables[k] = v
            except Exception:
                variables[k] = repr(v)

        result_queue.put({
            "success": True,
            "variables": variables,
            "error": None,
        })
    except Exception as exc:  # noqa: BLE001
        result_queue.put({
            "success": False,
            "variables": {},
            "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        })


def run_in_subprocess(
    code: str,
    namespace_setup: str = "",
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Run *code* in a fresh subprocess with *timeout* seconds.

    Parameters
    ----------
    code:
        Python source to execute after setup.
    namespace_setup:
        Python source executed first to populate the namespace (e.g. imports).
    timeout:
        Maximum wall-clock seconds; the child process is killed if exceeded.

    Returns
    -------
    dict with keys ``success`` (bool), ``variables`` (dict), ``error`` (str|None),
    and ``execution_time_ms`` (float).
    """
    ctx = multiprocessing.get_context("spawn")
    result_queue: multiprocessing.Queue = ctx.Queue()

    proc = ctx.Process(
        target=_worker,
        args=(code, namespace_setup, result_queue),
        daemon=True,
    )

    start = time.perf_counter()
    proc.start()
    proc.join(timeout=timeout)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)
        return {
            "success": False,
            "variables": {},
            "error": f"Execution timed out after {timeout}s",
            "execution_time_ms": elapsed_ms,
        }

    if proc.exitcode != 0 and result_queue.empty():
        return {
            "success": False,
            "variables": {},
            "error": f"Subprocess exited with code {proc.exitcode}",
            "execution_time_ms": elapsed_ms,
        }

    if result_queue.empty():
        return {
            "success": False,
            "variables": {},
            "error": "No result returned from subprocess",
            "execution_time_ms": elapsed_ms,
        }

    result = result_queue.get_nowait()
    result["execution_time_ms"] = elapsed_ms
    return result
