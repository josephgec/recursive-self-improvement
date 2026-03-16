.PHONY: test coverage lint check headroom report calibrate clean

test:
	python -m pytest tests/ -v

coverage:
	python -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html -v

lint:
	python -m py_compile src/constraints/base.py
	python -m py_compile src/checker/suite.py
	python -m py_compile src/enforcement/gate.py

check:
	python scripts/run_check.py

headroom:
	python scripts/run_headroom.py

report:
	python scripts/generate_report.py

calibrate:
	python scripts/calibrate_thresholds.py

clean:
	rm -rf __pycache__ .pytest_cache htmlcov .coverage
	find . -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true
