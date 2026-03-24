# LateNet

A large-scale dataset of contrastive true/false statement pairs for probing truth representations in LLM activations.

LateNet draws its taxonomic structure from [WordNet](https://wordnet.princeton.edu/), mirroring how ImageNet used WordNet synsets for its category hierarchy. The output is a validated dataset of labeled true/false statement pairs ready for activation extraction.

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
# Generate contrastive pairs from WordNet
latenet-generate --seed 42 --max-depth 8 --output candidates.parquet

# Validate with LLM ensemble voting
latenet-validate --input candidates.parquet --output validated.parquet

# Export final dataset
latenet-export --input validated.parquet --output latenet_v1.parquet
```

## How It Works

LateNet walks WordNet's noun hierarchy. For each concept (synset), it generates **true** statements from real relationships (hypernymy, meronymy, antonymy) and **false** statements by swapping related concepts at controlled semantic distances — from hard (sibling swap) to easy (distant subtree swap). Statements use varied templates to prevent probes from shortcutting on surface phrasing.

Generated pairs are then validated by an LLM ensemble (with escalation on disagreement) and exported as labeled Parquet files.

## Project Structure

```
latenet/
├── latenet/
│   ├── wordnet/         # WordNet hierarchy traversal and relationships
│   ├── generation/      # Contrastive pair generation and templates
│   ├── validation/      # LLM ensemble voting and escalation
│   ├── augmentation/    # Non-WordNet generative scripts
│   └── io/              # Export to parquet
├── scripts/             # CLI entry points
├── tests/
├── pyproject.toml
└── README.md
```

## Design Principles

- **Deterministic generation** given a seed for reproducibility
- **Incremental runs** - generate a batch, validate, append. No full regeneration required
- **Cheap generation, expensive validation** - generation uses WordNet locally; validation requires API calls and runs as a batch job
- **Target:** 200K+ validated pairs across 50+ domains

## Acknowledgments

LateNet is directly inspired by [The Geometry of Truth](https://arxiv.org/abs/2310.06824) (Marks & Tegmark, 2023), which demonstrated that LLMs represent truth as a linear feature in activation space using small, hand-crafted datasets of true/false statements. LateNet aims to scale that approach by orders of magnitude.

> Samuel Marks and Max Tegmark. "The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Statements." *arXiv preprint arXiv:2310.06824*, 2023.

The taxonomic backbone comes from [WordNet](https://wordnet.princeton.edu/) (Miller, 1995), and the name "LateNet" nods to [ImageNet](https://www.image-net.org/) (Deng et al., 2009), which similarly used WordNet's synset hierarchy to organize a large-scale dataset that transformed its field.

## License

MIT - see [LICENSE](LICENSE).

## Author

Michael Klear ([@AlliedToasters](https://github.com/AlliedToasters))
