.PHONY: install test lint download-data run-debug run-math500 run-olympiad run-comparison report clean

install:
	pip install -e .

test:
	pytest tests/ -v --tb=short

lint:
	python -m py_compile src/**/*.py

download-data:
	python scripts/download_benchmarks.py

run-debug:
	python scripts/run_benchmark.py --benchmark math500 --pipeline both --max-problems 10 --config configs/debug.yaml

run-math500:
	python scripts/run_benchmark.py --benchmark math500 --pipeline both --report

run-olympiad:
	python scripts/run_benchmark.py --benchmark olympiad --pipeline both --report

run-comparison:
	python scripts/run_comparison.py

report:
	python scripts/generate_report.py

clean:
	rm -rf data/results data/reports data/prompts_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
