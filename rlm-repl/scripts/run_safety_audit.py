#!/usr/bin/env python3
"""Run a safety audit against malicious code samples."""

import sys
import os
import json
import time
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backends.local import LocalREPL
from src.safety.policy import SafetyPolicy
from src.safety.ast_scanner import ASTScanner
from src.interface.errors import ForbiddenCodeError


def main():
    samples_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "malicious_samples",
    )

    policy = SafetyPolicy()
    scanner = ASTScanner(policy)

    print("RLM-REPL Safety Audit")
    print("=" * 50)

    results = []
    sample_files = sorted(glob.glob(os.path.join(samples_dir, "*.py")))

    for filepath in sample_files:
        name = os.path.basename(filepath)
        with open(filepath, "r") as f:
            code = f.read()

        # Test AST scanner
        scan_result = scanner.scan(code)

        # Test REPL execution
        repl = LocalREPL(policy=policy)
        blocked = False
        error_msg = ""
        try:
            result = repl.execute(code)
            if result.error:
                blocked = True
                error_msg = result.error
        except ForbiddenCodeError as e:
            blocked = True
            error_msg = str(e)
        except Exception as e:
            blocked = True
            error_msg = str(e)
        finally:
            repl.shutdown()

        status = "BLOCKED" if (not scan_result.safe or blocked) else "ESCAPED"
        print(f"  {name}: {status}")
        if scan_result.violations:
            for v in scan_result.violations[:2]:
                print(f"    - {v}")

        results.append({
            "sample": name,
            "ast_safe": scan_result.safe,
            "violations": len(scan_result.violations),
            "execution_blocked": blocked,
            "status": status,
        })

    # Summary
    total = len(results)
    blocked = sum(1 for r in results if r["status"] == "BLOCKED")
    print(f"\nSummary: {blocked}/{total} samples blocked")

    # Save report
    report_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "audit_reports", "latest.json",
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump({"timestamp": time.time(), "results": results}, f, indent=2)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
