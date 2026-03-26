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

# Build a balanced, validated dataset (the main data product)
latenet-build --rows-per-stratum 50 --output latenet_v1.parquet
latenet-build --rows-per-stratum 10 --generators biology temporal --legs anthropic
```

## Canonical Dataset Build

The main data product is a balanced, LLM-validated parquet file produced by `latenet-build`. The process:

1. **Generate** — Each generator produces candidate contrastive pairs from structured data sources (WordNet, Wikidata, Natural Earth, mendeleev). Generation is deterministic per seed.
2. **Validate** — Every candidate is judged by two independent validators:
   - **Llama 405B Instruct** (logit-level True/False via NDIF/lmprobe, `logit_top_k=10`)
   - **Claude Sonnet** (text-level True/False via Anthropic API)
   - Sonnet disagreements are **escalated to Opus** for the dispute record.
3. **Accept** — Only rows where *both* Llama and Sonnet agree with the ground-truth label enter the clean dataset.
4. **Dispute** — Rows where any validator disagrees go to a sidecar `disputes.parquet` with full validation metadata (logit gaps, model responses, escalation results). These are not discarded — they're valuable for understanding generator failure modes.
5. **Loop** — If any (generator, difficulty) stratum is under-filled, generate a new batch with an incremented seed (`seed + round * 1000`) and repeat from step 2. The loop runs until all strata have exactly `rows_per_stratum` pairs, or `max_rounds` is reached.

```bash
# Build the canonical dataset: 50 pairs per (generator, difficulty) cell
latenet-build --rows-per-stratum 50 --seed 42 --output latenet_v1.parquet

# Subset of generators, Anthropic-only validation (skip NDIF)
latenet-build --rows-per-stratum 20 --generators biology temporal --legs anthropic

# Outputs:
#   latenet_v1.parquet           — clean dataset (balanced, all validators agree)
#   latenet_v1.disputes.parquet  — disputed rows with validation metadata
```

**Required env vars:** `ANTHROPIC_API_KEY`, `NDIF_API_KEY` (for the NDIF/Llama leg).

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
- **validation/** — LLM ensemble voting (`voting.py`), dispute escalation (`escalation.py`), quality filters (`filters.py`), stratified build loop (`stratified.py`)
- **io/** — Parquet export with standardized schema (`export.py`)

### Generator contract

All generators subclass `BaseGenerator` and implement:
- `generate() -> Iterator[ContrastivePair]`
- `relation_types() -> list[str]`
- `domains() -> list[str]`
- `name -> str` (property)

### CLI scripts: `scripts/`

Entry points defined in `pyproject.toml`: `latenet-generate`, `latenet-validate`, `latenet-build`, `latenet-export`, `latenet-stats`.

## Key Design Decisions

- **Generation is deterministic** given a seed — always accept and propagate a `seed` parameter
- **Generation is cheap, validation is expensive** — they run as separate stages. Never couple them
- **Incremental by design** — generate a batch, validate it, append. Don't require full regeneration
- **Difficulty is semantic distance** — hard=sibling swap, medium=cousin swap, easy=distant subtree
- **Validation uses ensemble voting** — Llama 405B (logit-level via NDIF/lmprobe) + Sonnet vote, Opus escalation on Sonnet disagreements only
- **Stratified build loop** — `latenet-build` generates, validates, and loops until per-(generator, difficulty) quotas are met. Only rows where *both* validators agree with GT are accepted. Disputes go to a sidecar file for analysis. Each round uses `seed + round * 1000` for new candidates
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
