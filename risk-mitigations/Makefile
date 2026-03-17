.PHONY: install test coverage lint clean

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v

coverage:
	python -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html -v

lint:
	python -m py_compile src/collapse/alpha_scheduler.py
	python -m py_compile src/collapse/data_reserve.py
	python -m py_compile src/collapse/collapse_forecaster.py
	python -m py_compile src/collapse/halt_and_diagnose.py
	python -m py_compile src/collapse/recovery.py

clean:
	rm -rf __pycache__ .pytest_cache htmlcov .coverage
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

check: test coverage

all: install test coverage
