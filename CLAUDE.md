# CLAUDE.md

## Project Overview

LateNet generates large-scale contrastive true/false statement pairs for probing truth representations in LLM activations. It combines WordNet-derived taxonomic knowledge with domain-specific generators grounded in structured data sources (geospatial, temporal, chemical, linguistic, etc.). This repo handles text dataset generation and validation only — activation extraction happens in the sibling `lmprobe` repo.

## Environment

Use the `.venv` virtualenv in the project root. Activate with `source .venv/bin/activate` or invoke directly via `.venv/bin/python`, `.venv/bin/pytest`, etc.

## Build & Run

```bash
# Install for development
pip install -e ".[validation]"

# Run tests
pytest

# Iterative workflow (the primary way to build the dataset)
latenet-batch --generators biology temporal --max-pairs 50 --seed 42
latenet-batch --generators chemistry --max-pairs 20 --seed 100
latenet-stats --ledger                         # inspect ledger state
latenet-curate --rows-per-stratum 50 --output latenet_v1.parquet

# One-shot build (legacy — fills strata in a single loop)
latenet-build --rows-per-stratum 50 --output latenet_v1.parquet

# Low-level tools
latenet-generate --seed 42 --output candidates.parquet
latenet-validate --input candidates.parquet --output validated.parquet
latenet-stats --input candidates.parquet
```

## Canonical Dataset Build

The main data product is a balanced, LLM-validated parquet file. The iterative workflow uses a **persistent ledger** (`data/ledger.parquet`) — an append-only record of every statement ever generated and validated, with full provenance (git hash, timestamp, batch ID).

### Iterative workflow (preferred)

1. **`latenet-batch`** — Generate a batch of candidates, validate with the LLM ensemble, append results to the ledger. Run repeatedly with different generators, seeds, and sizes.
2. **`latenet-stats --ledger`** — Inspect the ledger: per-stratum fill levels, acceptance rates, provenance breakdown. Spot problematic generators.
3. **Tweak generators** — Fix templates, add data sources, adjust difficulty calibration.
4. **Repeat** from (1) until the ledger has enough clean rows per stratum.
5. **`latenet-curate`** — Pull a balanced subset from the ledger to produce the final data product.

### One-shot build (legacy)

`latenet-build` runs a generate-validate-accept loop in a single invocation. The process:

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
- **io/** — Parquet export (`export.py`), persistent ledger (`ledger.py`), provenance stamping (`provenance.py`)

### Generator contract

All generators subclass `BaseGenerator` and implement:
- `generate() -> Iterator[ContrastivePair]`
- `relation_types() -> list[str]`
- `domains() -> list[str]`
- `name -> str` (property)

### CLI scripts: `scripts/`

Entry points defined in `pyproject.toml`: `latenet-batch`, `latenet-curate`, `latenet-build`, `latenet-generate`, `latenet-validate`, `latenet-stats`, `latenet-export`, `latenet-qa`.

## Key Design Decisions

- **Generation is deterministic** given a seed — always accept and propagate a `seed` parameter
- **Generation is cheap, validation is expensive** — they run as separate stages. Never couple them
- **Incremental by design** — generate a batch, validate it, append. Don't require full regeneration
- **Difficulty is semantic distance** — hard=sibling swap, medium=cousin swap, easy=distant subtree
- **Validation uses ensemble voting** — Llama 405B (logit-level via NDIF/lmprobe) + Sonnet vote, Opus escalation on Sonnet disagreements only
- **Ledger-based workflow** — `latenet-batch` generates, validates, and appends to a persistent ledger (`data/ledger.parquet`). Every row carries `git_hash`, `generated_at`, and `batch_id` for provenance. `latenet-curate` pulls balanced strata from the ledger to produce the final data product. `latenet-build` still available for one-shot stratified builds
- **Adding a new generator** — subclass BaseGenerator, implement the contract, register in `scripts/generate.py` GENERATORS dict

## Git / Merge Policy

- **Never squash commits on merge.** Every generated row in the ledger carries a `git_hash` linking it to the code version that produced it. Squashing rewrites commit hashes, which would break provenance tracing. Use merge commits or rebase (without squash) for PRs

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
