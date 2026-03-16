.PHONY: test coverage lint clean install

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v

coverage:
	python -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html -v

lint:
	python -m py_compile src/benchmarks/task.py
	python -m py_compile src/benchmarks/registry.py

clean:
	rm -rf __pycache__ .pytest_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

run-all:
	python scripts/run_all_benchmarks.py

run-scaling:
	python scripts/run_scaling_experiment.py

report:
	python scripts/generate_report.py
