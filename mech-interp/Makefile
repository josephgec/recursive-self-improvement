.PHONY: test coverage lint clean install

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v

coverage:
	python -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html -v

lint:
	python -m py_compile src/probing/probe_set.py
	python -m py_compile src/probing/extractor.py
	python -m py_compile src/probing/snapshot.py
	python -m py_compile src/probing/diff.py
	python -m py_compile src/probing/projector.py
	python -m py_compile src/attention/head_extractor.py
	python -m py_compile src/attention/specialization.py
	python -m py_compile src/attention/reward_correlation.py
	python -m py_compile src/attention/role_tracker.py
	python -m py_compile src/attention/dead_head_detector.py
	python -m py_compile src/anomaly/divergence_detector.py
	python -m py_compile src/anomaly/behavioral_similarity.py
	python -m py_compile src/anomaly/internal_distance.py
	python -m py_compile src/anomaly/ratio_monitor.py
	python -m py_compile src/anomaly/deceptive_alignment.py
	python -m py_compile src/integration/godel_hooks.py
	python -m py_compile src/integration/soar_hooks.py
	python -m py_compile src/integration/pipeline_hooks.py
	python -m py_compile src/integration/decorator.py
	python -m py_compile src/monitoring/dashboard.py
	python -m py_compile src/monitoring/alert_rules.py
	python -m py_compile src/monitoring/time_series.py
	python -m py_compile src/analysis/activation_analysis.py
	python -m py_compile src/analysis/head_evolution.py
	python -m py_compile src/analysis/anomaly_characterization.py
	python -m py_compile src/analysis/report.py

clean:
	rm -rf __pycache__ .pytest_cache htmlcov .coverage
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
