#!/usr/bin/env python3
"""Stress test the REPL pool."""

import sys
import os
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pool.pool import REPLPool
from src.safety.policy import SafetyPolicy


def worker(pool, worker_id, iterations):
    """Worker thread that acquires and uses REPLs."""
    for i in range(iterations):
        try:
            repl = pool.acquire(timeout=10)
            result = repl.execute(f"x = {worker_id} * {i}")
            pool.release(repl)
        except Exception as e:
            print(f"Worker {worker_id} error: {e}")


def main():
    num_workers = 8
    iterations = 10
    pool_size = 4

    print(f"Stress test: {num_workers} workers, {iterations} iterations each, pool size {pool_size}")

    policy = SafetyPolicy()
    pool = REPLPool(size=pool_size, policy=policy)

    start = time.time()

    threads = []
    for i in range(num_workers):
        t = threading.Thread(target=worker, args=(pool, i, iterations))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    elapsed = time.time() - start
    metrics = pool.get_metrics()

    print(f"\nCompleted in {elapsed:.2f}s")
    print(f"Total acquires: {metrics.total_acquires}")
    print(f"Total releases: {metrics.total_releases}")

    pool.shutdown()


if __name__ == "__main__":
    main()
