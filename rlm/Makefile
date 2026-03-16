.PHONY: test lint coverage clean install dev

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v

coverage:
	python -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html -v

lint:
	python -m py_compile src/core/completion.py
	python -m py_compile src/core/session.py

clean:
	rm -rf __pycache__ .pytest_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
