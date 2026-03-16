.PHONY: test lint clean install

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v --cov=src --cov-report=term-missing

test-quick:
	python -m pytest tests/ -x -q

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov

lint:
	python -m py_compile src/**/*.py
