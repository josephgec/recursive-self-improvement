# SOAR Hindsight

Converts evolutionary search trajectories into fine-tuning data, closing the SOAR self-improvement loop.

## Commands
- `make test` - Run all tests with coverage
- `make test-quick` - Run tests, stop on first failure
- `make install` - Install in dev mode
- `python -m pytest tests/test_integration.py` - Integration tests only

## Architecture
- `src/collection/` - Harvest and index search trajectories
- `src/synthesis/` - Convert trajectories to training pairs via strategies
- `src/finetuning/` - Mock fine-tuning orchestration (OpenAI + local)
- `src/iteration/` - SOAR loop: search -> collect -> train -> evaluate -> repeat
- `src/analysis/` - Data quality, transfer analysis, reporting
- `src/utils/` - Token counting, sampling utilities

## Conventions
- All tests are fully offline with mock data
- TrainingPair is the universal exchange format between synthesis and fine-tuning
- Strategies are pluggable via the synthesizer registry
