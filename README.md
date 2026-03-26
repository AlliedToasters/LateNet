# LateNet

A large-scale dataset of contrastive true/false statement pairs for probing truth representations in LLM activations.

LateNet combines WordNet-derived taxonomic knowledge at scale with domain-specific generators grounded in structured data sources (geospatial, temporal, chemical, linguistic, etc.), covering logical relation types that no single knowledge base can express alone. The output is a validated dataset of labeled true/false statement pairs ready for activation extraction.

## Scope

This repo generates and validates contrastive text pairs. Activation extraction and probe fitting happen in [lmprobe](https://github.com/AlliedToasters/lmprobe).

**This repo produces text datasets, not tensor datasets.**

## Installation

```bash
pip install latenet

# or for development
git clone https://github.com/AlliedToasters/LateNet.git
cd LateNet
pip install -e .
```

## Quick Start

```bash
# Build the canonical dataset (generate + validate + balance in one step)
latenet-build --rows-per-stratum 50 --seed 42 --output latenet_v1.parquet

# This produces:
#   latenet_v1.parquet           — clean, balanced dataset
#   latenet_v1.disputes.parquet  — rows where validators disagreed (for analysis)
```

Requires `ANTHROPIC_API_KEY` and `NDIF_API_KEY` environment variables for the validation pipeline.

### Other CLI tools

```bash
# Generate candidates without validation
latenet-generate --seed 42 --output candidates.parquet
latenet-generate --generators wordnet --max-depth 8 --output candidates.parquet

# Validate an existing dataset (one-shot, no stratification loop)
latenet-validate --input candidates.parquet --output validated.parquet

# Export final dataset
latenet-export --input validated.parquet --output latenet_v1.parquet

# View dataset statistics
latenet-stats --input candidates.parquet

# QA: sample statements stratified by generator and difficulty
latenet-qa --input candidates.parquet
latenet-qa --input candidates.parquet --n 5 --generators chemistry geography
```

## How It Works

LateNet uses a plugin architecture where each **generator** wraps a structured data source and produces contrastive true/false pairs for specific relation types and semantic domains.

The **WordNet generator** walks the noun hierarchy, generating true statements from real relationships (hypernymy, meronymy) and false statements by swapping concepts at controlled semantic distances — from hard (sibling swap) to easy (distant subtree swap).

**Domain-specific generators** extend coverage to relation types WordNet can't express: geospatial containment, temporal ordering, chemical properties, translation equivalence, magnitude comparisons, and author-work attribution. Each is grounded in a structured data source so ground truth is programmatic.

All generators share a common contract (`BaseGenerator`) and produce `ContrastivePair` objects with a standardized schema. Generated pairs are validated by an LLM ensemble and exported as labeled Parquet files.

### The Build Process

`latenet-build` is the canonical way to produce a dataset. It runs a generate-validate-accept loop:

1. **Generate** candidate pairs from structured data sources (deterministic per seed)
2. **Validate** each candidate with two independent LLM judges:
   - **Llama 3.1 405B Instruct** — logit-level True/False via [NDIF](https://ndif.us)/[lmprobe](https://github.com/AlliedToasters/lmprobe) (no local GPU needed)
   - **Claude Sonnet** — text-level True/False via Anthropic API
   - Sonnet disagreements are escalated to **Claude Opus** for the dispute record
3. **Accept** only rows where both validators agree with the ground-truth label
4. **Loop** with a new seed until every (generator × difficulty) stratum has the target number of pairs

Disputed rows are saved to a sidecar file with full validation metadata (logit gaps, model responses, Opus escalation results). The clean dataset has perfectly even representation across all generators and difficulty tiers.

## Generators

| Generator | Data Source | Relation Types |
|-----------|------------|----------------|
| **wordnet** | NLTK WordNet | hypernymy, meronymy, antonymy, sibling |
| **geography** | Natural Earth, GeoNames | contained-in, cardinal-direction, closer-to, population-greater, area-greater |
| **chemistry** | mendeleev (periodic table) | symbol-of, member-of-group, state-at-room-temp, in-block, atomic-number-greater, property-greater |
| **temporal** | Wikidata (events, people) | happened-before, born-before, occurred-in-century, lived-before-event, were-contemporaries |
| **biology** | Wikidata (taxonomy) | is-member-of (species→genus, genus→family, etc.) |
| language | Translation dictionaries | translates-to |
| magnitude | World Bank, reference tables | greater-than, less-than |
| authorship | Literary/scientific databases | written-by, proposed-by |

**Bold** = implemented. Others are stubbed with the `BaseGenerator` contract, ready for incremental development.

## Adding a Generator

Subclass `BaseGenerator`, implement `generate()`, `relation_types()`, `domains()`, and the `name` property, then register it in the CLI:

```python
from latenet.generators.base import BaseGenerator
from latenet.types import ContrastivePair

class MyGenerator(BaseGenerator):
    @property
    def name(self) -> str:
        return "my_domain"

    def relation_types(self) -> list[str]:
        return ["my-relation"]

    def domains(self) -> list[str]:
        return ["my-domain"]

    def generate(self):
        yield ContrastivePair(...)
```

## Project Structure

```
latenet/
├── latenet/
│   ├── generators/      # BaseGenerator + all generator implementations
│   ├── wordnet/         # WordNet hierarchy traversal and relationships
│   ├── templates/       # Statement templates per relation type
│   ├── negation/        # False statement generation strategies
│   ├── difficulty/      # Difficulty tiers and semantic distance
│   ├── validation/      # LLM ensemble voting, escalation, and stratified build loop
│   └── io/              # Export to parquet
├── scripts/             # CLI entry points
├── data_sources/        # Static reference data for domain generators
├── tests/
├── pyproject.toml
└── README.md
```

## QA Sampling

The `latenet-qa` script samples true/false pairs from a generated parquet file, stratified by **generator x difficulty**. This gives a quick human-readable view for manual spot-checking as new generators and datasets are added.

```bash
# Default: 3 samples per generator x difficulty bucket
latenet-qa --input candidates.parquet

# More samples, filtered to specific generators
latenet-qa --input candidates.parquet --n 5 --generators chemistry

# Reproducible sampling
latenet-qa --input candidates.parquet --seed 123
```

Example output:
```
======================================================================
Generator: chemistry
  Total pairs: 20
  Relation types: symbol_of
  Difficulties: hard, medium
======================================================================

--- chemistry / hard (10 pairs, showing 2) ---

  pair_id:    4c0efd185ba0
  relation:   symbol_of
  template:   chem_symbol_01
  difficulty:  hard
  negation:   sibling_swap
  TRUE:  The chemical symbol for Sulfur is S.
  FALSE: The chemical symbol for Sulfur is P.
```

## Design Principles

- **Deterministic generation** given a seed for reproducibility
- **Plugin architecture** — adding a generator is subclass + register
- **Incremental runs** — generate a batch, validate, append. No full regeneration required
- **Cheap generation, expensive validation** — generation is local; validation requires API calls
- **Two axes of diversity** — logical relation types (what kind of reasoning) x semantic domains (what the facts are about)
- **Target:** 200K+ validated pairs across 50+ domains and 7+ relation types

## Acknowledgments

LateNet is directly inspired by [The Geometry of Truth](https://arxiv.org/abs/2310.06824) (Marks & Tegmark, 2023), which demonstrated that LLMs represent truth as a linear feature in activation space using small, hand-crafted datasets of true/false statements. LateNet aims to scale that approach by orders of magnitude.

> Samuel Marks and Max Tegmark. "The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Statements." *arXiv preprint arXiv:2310.06824*, 2023.

The taxonomic backbone comes from [WordNet](https://wordnet.princeton.edu/) (Miller, 1995), and the name "LateNet" nods to [ImageNet](https://www.image-net.org/) (Deng et al., 2009), which similarly used WordNet's synset hierarchy to organize a large-scale dataset that transformed its field.

## License

MIT - see [LICENSE](LICENSE).

## Author

Michael Klear ([@AlliedToasters](https://github.com/AlliedToasters))
