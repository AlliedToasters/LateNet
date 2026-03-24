# CLAUDE.md

## Project Overview

LateNet generates large-scale contrastive true/false statement pairs from WordNet for probing truth representations in LLM activations. This repo handles text dataset generation and validation only — activation extraction happens in the sibling `lmprobe` repo.

## Build & Run

```bash
# Install for development
pip install -e .

# Run tests
pytest

# CLI commands
latenet-generate --seed 42 --output candidates.parquet
latenet-validate --input candidates.parquet --output validated.parquet
latenet-export --input validated.parquet --output latenet_v1.parquet
```

## Dependencies

- Python 3.10+
- NLTK (WordNet corpus: `nltk.corpus.wordnet`)
- PyArrow / Pandas (parquet export)
- Anthropic SDK and external LLM APIs (validation pipeline only)

Download WordNet data: `python -c "import nltk; nltk.download('wordnet')"`

## Architecture

### Package layout: `latenet/`

- **wordnet/** — WordNet traversal (`walker.py`), relationship extraction (`relationships.py`), semantic distance (`distance.py`)
- **generation/** — Statement templates (`templates.py`), core pair generator (`generator.py`), false statement strategies (`negation.py`), difficulty tiers (`difficulty.py`)
- **validation/** — LLM ensemble voting (`voting.py`), dispute escalation (`escalation.py`), quality filters (`filters.py`)
- **augmentation/** — Non-WordNet generative scripts (`custom.py`)
- **io/** — Parquet export with standardized schema (`export.py`)

### CLI scripts: `scripts/`

Entry points defined in `pyproject.toml`. Each script is a thin CLI wrapper around library code.

## Key Design Decisions

- **Generation is deterministic** given a seed — always accept and propagate a `seed` parameter
- **Generation is cheap, validation is expensive** — they run as separate stages. Never couple them
- **Incremental by design** — generate a batch, validate it, append. Don't require full regeneration
- **Difficulty is semantic distance** — hard=sibling swap, medium=cousin swap, easy=distant subtree
- **Validation uses ensemble voting** — Llama 405B + Sonnet vote, Opus breaks ties on disagreement

## Code Conventions

- Use type hints throughout
- Parquet is the canonical export format
- All generated data includes `pair_id` to link true/false counterparts
- Template IDs must be stable across runs for reproducibility
- Keep validation API calls batched and retriable

## What Not To Add

- No activation extraction or model inference (that's `lmprobe`)
- No probe fitting or evaluation
- No model loading beyond what the validation voting pipeline needs
- This repo produces text datasets, not tensor datasets
