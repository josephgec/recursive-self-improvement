.PHONY: install test lint run-debug run-1b run-7b run-all clean

install:
	pip install -e .

test:
	pytest tests/ -v --tb=short

lint:
	python -m py_compile src/**/*.py

run-debug:
	python scripts/run_full_experiment.py --scale 1b --schedules zero_alpha --config configs/debug.yaml

run-1b:
	python scripts/run_full_experiment.py --scale 1b --schedules all

run-7b:
	python scripts/run_full_experiment.py --scale 7b --schedules all

run-all:
	python scripts/run_full_experiment.py --scale both --schedules all

clean:
	rm -rf data/synthetic data/checkpoints data/metrics data/reports
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
