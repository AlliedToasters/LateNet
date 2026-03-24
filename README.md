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
# Generate contrastive pairs (all generators)
latenet-generate --seed 42 --output candidates.parquet

# Or run specific generators
latenet-generate --generators wordnet --max-depth 8 --output candidates.parquet

# Validate with LLM ensemble voting
latenet-validate --input candidates.parquet --output validated.parquet

# Export final dataset
latenet-export --input validated.parquet --output latenet_v1.parquet

# View dataset statistics
latenet-stats --input candidates.parquet
```

## How It Works

LateNet uses a plugin architecture where each **generator** wraps a structured data source and produces contrastive true/false pairs for specific relation types and semantic domains.

The **WordNet generator** walks the noun hierarchy, generating true statements from real relationships (hypernymy, meronymy) and false statements by swapping concepts at controlled semantic distances — from hard (sibling swap) to easy (distant subtree swap).

**Domain-specific generators** extend coverage to relation types WordNet can't express: geospatial containment, temporal ordering, chemical properties, translation equivalence, magnitude comparisons, and author-work attribution. Each is grounded in a structured data source so ground truth is programmatic.

All generators share a common contract (`BaseGenerator`) and produce `ContrastivePair` objects with a standardized schema. Generated pairs are validated by an LLM ensemble and exported as labeled Parquet files.

## Generators

| Generator | Data Source | Relation Types |
|-----------|------------|----------------|
| **wordnet** | NLTK WordNet | hypernymy, meronymy, antonymy, sibling |
| geography | Natural Earth, GeoNames | contained-in, north-of, closer-to |
| temporal | Historical databases | before, after, century-of |
| chemistry | Periodic table, PubChem | symbol-of, property-of, group-membership |
| language | Translation dictionaries | translates-to |
| magnitude | World Bank, reference tables | greater-than, less-than |
| authorship | Literary/scientific databases | written-by, proposed-by |
| biology | NCBI taxonomy | is-a, part-of |

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
│   ├── validation/      # LLM ensemble voting and escalation
│   └── io/              # Export to parquet
├── scripts/             # CLI entry points
├── data_sources/        # Static reference data for domain generators
├── tests/
├── pyproject.toml
└── README.md
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
