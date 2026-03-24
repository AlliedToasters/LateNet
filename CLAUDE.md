# CLAUDE.md

## Project Overview

LateNet generates large-scale contrastive true/false statement pairs for probing truth representations in LLM activations. It combines WordNet-derived taxonomic knowledge with domain-specific generators grounded in structured data sources (geospatial, temporal, chemical, linguistic, etc.). This repo handles text dataset generation and validation only — activation extraction happens in the sibling `lmprobe` repo.

## Build & Run

```bash
# Install for development
pip install -e .

# Run tests
pytest

# CLI commands
latenet-generate --seed 42 --output candidates.parquet
latenet-generate --generators wordnet --max-depth 3 --output small.parquet
latenet-validate --input candidates.parquet --output validated.parquet
latenet-export --input validated.parquet --output latenet_v1.parquet
latenet-stats --input candidates.parquet
```

## Dependencies

- Python 3.10+
- NLTK (WordNet corpus: `nltk.corpus.wordnet`)
- PyArrow / Pandas (parquet export)
- Anthropic SDK and external LLM APIs (validation pipeline only)

Download WordNet data: `python -c "import nltk; nltk.download('wordnet')"`

## Architecture

### Package layout: `latenet/`

- **generators/** — `BaseGenerator` ABC (`base.py`), WordNet generator (`wordnet_gen.py`), domain-specific generator stubs (`geography.py`, `temporal.py`, `chemistry.py`, `language.py`, `magnitude.py`, `authorship.py`, `biology.py`)
- **wordnet/** — WordNet traversal (`walker.py`), relationship extraction (`relationships.py`), semantic distance (`distance.py`)
- **templates/** — Statement templates per relation type (`templates.py`), template diversity (`diversity.py`)
- **negation/** — False statement strategies (`strategies.py`)
- **difficulty/** — Difficulty tiers and semantic distance scoring (`tiers.py`)
- **validation/** — LLM ensemble voting (`voting.py`), dispute escalation (`escalation.py`), quality filters (`filters.py`)
- **io/** — Parquet export with standardized schema (`export.py`)

### Generator contract

All generators subclass `BaseGenerator` and implement:
- `generate() -> Iterator[ContrastivePair]`
- `relation_types() -> list[str]`
- `domains() -> list[str]`
- `name -> str` (property)

### CLI scripts: `scripts/`

Entry points defined in `pyproject.toml`: `latenet-generate`, `latenet-validate`, `latenet-export`, `latenet-stats`.

## Key Design Decisions

- **Generation is deterministic** given a seed — always accept and propagate a `seed` parameter
- **Generation is cheap, validation is expensive** — they run as separate stages. Never couple them
- **Incremental by design** — generate a batch, validate it, append. Don't require full regeneration
- **Difficulty is semantic distance** — hard=sibling swap, medium=cousin swap, easy=distant subtree
- **Validation uses ensemble voting** — Llama 405B + Sonnet vote, Opus breaks ties on disagreement
- **Adding a new generator** — subclass BaseGenerator, implement the contract, register in `scripts/generate.py` GENERATORS dict

## Code Conventions

- Use type hints throughout
- Parquet is the canonical export format
- All generated data includes `pair_id` to link true/false counterparts
- Template IDs must be stable across runs for reproducibility
- Keep validation API calls batched and retriable
- All randomness through `random.Random(seed)` — no global RNG

## What Not To Add

- No activation extraction or model inference (that's `lmprobe`)
- No probe fitting or evaluation
- No model loading beyond what the validation voting pipeline needs
- This repo produces text datasets, not tensor datasets
