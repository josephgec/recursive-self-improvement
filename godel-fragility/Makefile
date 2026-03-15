.PHONY: install test lint run-quick run-full sweep report clean

install:
	pip install -e .

test:
	pytest tests/ -v --tb=short

lint:
	python -m py_compile src/**/*.py

run-quick:
	python scripts/run_stress_test.py --config configs/quick.yaml --report

run-full:
	python scripts/run_stress_test.py --config configs/full.yaml --complexity-sweep --report

sweep:
	python scripts/run_complexity_sweep.py

report:
	python scripts/generate_report.py

clean:
	rm -rf data/results data/checkpoints data/reports
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
