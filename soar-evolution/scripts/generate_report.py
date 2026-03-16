#!/usr/bin/env python3
"""Generate a report from search results."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.report import ReportGenerator
from src.utils.logging import setup_logging


def main():
    setup_logging(level="INFO")

    results_file = sys.argv[1] if len(sys.argv) > 1 else "data/results/search_results.json"
    results_path = Path(results_file)

    if not results_path.exists():
        print(f"No results file found at {results_path}")
        return

    with open(results_path) as f:
        data = json.load(f)

    gen = ReportGenerator()
    text = gen.format_text_report(data)
    print(text)

    output_path = "data/reports/report.txt"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(text)
    print(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    main()
