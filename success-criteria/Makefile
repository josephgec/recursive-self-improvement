.PHONY: test coverage lint clean run collect verify verdict report all

test:
	python -m pytest tests/ -v

coverage:
	python -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

lint:
	python -m py_compile src/criteria/base.py
	python -m py_compile src/criteria/sustained_improvement.py
	python -m py_compile src/criteria/paradigm_improvement.py
	python -m py_compile src/criteria/gdi_bounds.py
	python -m py_compile src/criteria/publication_acceptance.py
	python -m py_compile src/criteria/auditability.py

clean:
	rm -rf __pycache__ .pytest_cache htmlcov .coverage
	find . -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true

run:
	python scripts/run_evaluation.py

collect:
	python scripts/collect_evidence.py

verify:
	python scripts/verify_integrity.py

verdict:
	python scripts/generate_verdict.py

report:
	python scripts/package_reproducibility.py

all: collect verify run verdict report
