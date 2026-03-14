# Data Contamination Audit

Pipeline to detect AI-generated content in training corpora and build a clean data reserve.

## Quick Start
- `make install` — install dependencies
- `make test` — run tests
- `python scripts/run_audit.py --config configs/small_scale.yaml` — dev run
- `python scripts/run_audit.py --config configs/full_scale.yaml` — production run

## Architecture
- `src/data/` — Data ingestion (Common Crawl, Wikipedia)
- `src/embeddings/` — Transformer embeddings + cosine similarity
- `src/classifier/` — Human vs. synthetic classifier (XGBoost + stylometric features)
- `src/reserve/` — Clean data reserve filtering and export
- `src/reporting/` — Visualization and audit reports

## Key Patterns
- All modules use the `Document` dataclass from `src/data/common_crawl.py`
- Configuration via Hydra YAML in `configs/`
- Intermediate results cached to `data/` — pipeline is resumable
- Tests in `tests/` — run before committing

## Implementation Plan
See the 20-step implementation sequence in the project plan.
