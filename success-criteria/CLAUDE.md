# Success Criteria - Month-18 GO/NO-GO Evaluation

## Overview
Fully self-contained evaluation framework with 5 pre-registered criteria for the Month-18 GO/NO-GO decision. All evidence is mock/synthetic for offline deterministic evaluation.

## Commands
- `make test` — Run all tests
- `make coverage` — Run tests with coverage report (target >= 90%)
- `make run` — Run full evaluation pipeline
- `make all` — Collect, verify, evaluate, verdict, report

## Architecture
- `src/criteria/` — 5 criterion implementations (ABC pattern)
- `src/evidence/` — Evidence collection and integrity verification
- `src/evaluation/` — Evaluator, confidence, sensitivity analysis
- `src/verdict/` — Final verdict logic (SUCCESS/PARTIAL/NOT_MET)
- `src/reporting/` — Executive summary, technical report, appendix
- `configs/` — Pre-registered thresholds, evidence paths, venues
- `scripts/` — CLI entry points
- `tests/` — Comprehensive test suite

## Key Design Decisions
- Mann-Kendall trend test implemented from scratch (no scipy)
- Paired t-test implemented with numpy
- Hash chain uses SHA-256
- All evidence is mock/synthetic — no external dependencies
- Deterministic outputs for reproducibility
