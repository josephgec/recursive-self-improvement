.PHONY: install test test-cov lint clean build docker-build audit benchmark

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v

test-cov:
	python -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

lint:
	python -m flake8 src/ tests/
	python -m mypy src/

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

build:
	python -m build

docker-build:
	python scripts/build_sandbox_image.py

audit:
	python scripts/run_safety_audit.py

benchmark:
	python scripts/benchmark_backends.py

stress:
	python scripts/stress_test.py
