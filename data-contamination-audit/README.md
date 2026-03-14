# Data Contamination Audit

A pipeline to detect AI-generated content in training corpora and build a clean data reserve. Processes Common Crawl and Wikipedia snapshots from 2013–2025, produces temporal contamination curves, and outputs a filtered reserve of high-confidence human-authored data with per-document authenticity scores.

## What It Does

1. **Measures corpus contamination** — Cosine similarity scoring over timestamped snapshots reveals how textual homogeneity has increased over time, with an inflection point corresponding to the rise of LLM-generated content.
2. **Classifies documents as human or synthetic** — An XGBoost classifier trained on perplexity, watermark detection, and stylometric features produces calibrated per-document authenticity scores.
3. **Builds a clean data reserve** — Threshold-based filtering, deduplication, and quality checks produce a Parquet dataset of high-confidence human-authored documents.

## Quick Start

```bash
# Install
pip install -e .

# Run tests
make test

# Development run (small scale)
python scripts/run_audit.py --config configs/small_scale.yaml

# Production run
python scripts/run_audit.py --config configs/full_scale.yaml
```

## Architecture

```
src/
├── data/           # Data ingestion (Common Crawl, Wikipedia, temporal sampling)
├── embeddings/     # Transformer embeddings, cosine similarity, temporal curves
├── classifier/     # Human vs. synthetic classifier (XGBoost + feature pipeline)
│   └── features/   # Perplexity, watermark detection, stylometry, ensemble
├── reserve/        # Clean data reserve filtering, quality checks, export
└── reporting/      # Temporal curve plots, feature distributions, audit reports
```

## Pipeline Steps

The pipeline is orchestrated by `scripts/run_audit.py` and runs in 7 resumable steps:

| Step | Description |
|------|-------------|
| `download` | Fetch Wikipedia and Common Crawl data for configured time bins |
| `embed` | Compute transformer embeddings for all documents |
| `features` | Extract perplexity, watermark, and stylometric features |
| `train` | Train the contamination classifier on labeled data |
| `classify` | Score all documents with the trained classifier |
| `filter` | Apply threshold + quality filters to build the reserve |
| `report` | Generate temporal curves and audit summary report |

```bash
# Run specific steps
python scripts/run_audit.py --steps embed,features,train

# Dry run (print what would be done)
python scripts/run_audit.py --dry-run
```

## Classifier Features

The binary classifier uses 17 features across three families:

- **Perplexity** (3 features) — Mean, standard deviation, and burstiness of per-chunk perplexity under GPT-2. AI text has lower, more uniform perplexity.
- **Watermark detection** (2 features) — Z-score and green-list fraction from the Kirchenbauer et al. (2023) statistical watermark detection scheme.
- **Stylometry** (12 features) — Vocabulary richness, hapax ratio, sentence/paragraph length variation, Yule's K, function word ratio, punctuation patterns, conjunction rate, passive voice ratio, and n-gram repetition.

## Configuration

Configs live in `configs/` as YAML files. Key settings:

```yaml
sampling:
  n_per_bin: 5000          # Documents per time bin
  bin_size: "year"         # year, half-year, or quarter

embeddings:
  model_name: "all-MiniLM-L6-v2"  # Sentence transformer model

classifier:
  model_type: "xgboost"
  hyperparameters:
    n_estimators: 500
    max_depth: 8

reserve:
  threshold: 0.90          # Minimum p_human to include
  deduplicate: true
  dedup_threshold: 0.95
```

## Output

The pipeline produces:

- **`data/reserve/reserve.parquet`** — Clean data reserve with authenticity scores
- **`data/reserve/summary.json`** — Reserve statistics including α_t (proportion of authentic data)
- **`data/reserve/audit_report.md`** — Full audit report with embedded visualizations
- **`data/reserve/temporal_similarity.png`** — Contamination curve over time
- **`data/reserve/contamination_rate.png`** — Per-year synthetic content fraction

## Testing

```bash
# Unit tests (fast, ~2 min)
make test

# With coverage
pytest tests/ --cov=src --cov-report=term-missing

# Integration test (slow, ~25 min, runs full pipeline on synthetic data)
pytest tests/test_integration.py -v
```

## Key Design Decisions

- **XGBoost over neural classifiers** — Interpretability. Feature importance rankings reveal which signals drive contamination detection.
- **Sentence-transformers over raw LLM embeddings** — Efficiency. Embedding millions of documents requires models optimized for throughput.
- **GPT-2 for perplexity scoring** — Neutrality. As a pre-contamination-era model, GPT-2 serves as an unbiased reference distribution.
- **Platt scaling for calibration** — Reliability. Ensures predicted probabilities are well-calibrated for threshold-based filtering.

## License

MIT
