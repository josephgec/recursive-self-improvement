.PHONY: test coverage lint clean run dry safety report

test:
	python -m pytest tests/ -v

coverage:
	python -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

lint:
	python -m py_compile src/pipeline/orchestrator.py
	python -m py_compile src/pipeline/state.py

clean:
	rm -rf __pycache__ .pytest_cache htmlcov .coverage
	find . -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true

run:
	python scripts/run_pipeline.py

dry:
	python scripts/run_dry.py

safety:
	python scripts/run_safety_audit.py

report:
	python scripts/generate_report.py
