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

### Contrastive Pair Generation

LateNet walks WordNet's noun hierarchy. For each concept (synset), it generates **true** statements from real relationships and **false** statements by perturbing those relationships at controlled semantic distances.

**Relationship types:**
- **Hypernymy** ("is-a"): "A dog is a mammal" / false: "A dog is a reptile"
- **Meronymy** ("part-of"): "A wheel is part of a car" / false: "A wheel is part of a tree"
- **Antonymy**: adjective-based contrastive pairs
- **Properties/attributes** where available

### Difficulty Tiers

False statements are generated at controlled semantic distances from the true concept:

| Tier | Strategy | Example |
|------|----------|---------|
| **Hard** | Sibling swap (same parent) | "A dog is a cat" |
| **Medium** | Cousin swap (same grandparent) | "A dog is a lizard" |
| **Easy** | Distant subtree swap | "A dog is an airplane" |

### Negation Strategies

Each true statement can produce multiple false counterparts:

1. **Sibling swap** - replace the object with a sibling in the WordNet tree
2. **Distant swap** - replace with a node from a different subtree
3. **Direct negation** - syntactic negation of a true statement ("A dog is not a mammal")
4. **Reverse relation** - flip the relation where it becomes false ("A mammal is a dog")

### Template Diversity

Statements use varied templates to prevent probes from shortcutting on surface phrasing:

- "{entity} is a {category}"
- "A {entity} is a type of {category}"
- "{part} is part of {whole}"
- "{entity} belongs to the category of {category}"

### LLM Validation Pipeline

Generated pairs are validated by an ensemble of LLMs to catch polysemy issues, pragmatically weird statements, and edge cases:

- **Voters:** Llama 405B Instruct, Claude Sonnet
- **Escalation:** On disagreement, escalate to Claude Opus as tiebreaker
- **Output:** Each row gets a `consensus` column and per-model votes
- **Filtering:** Disputed pairs can be kept with low confidence weight or dropped

### Output Schema

Exported as Parquet. Each row contains:

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Unique identifier |
| `statement` | str | Natural language statement |
| `label` | bool | True or false |
| `pair_id` | str | Links true/false counterparts |
| `domain` | str | WordNet top-level category |
| `relationship_type` | str | hypernym, meronym, antonym, etc. |
| `difficulty` | str | hard, medium, easy |
| `semantic_distance` | int | Hops in WordNet tree between true and swapped concept |
| `source_synset` | str | WordNet synset ID for the subject |
| `target_synset` | str | WordNet synset ID for the object (true version) |
| `neg_synset` | str \| null | WordNet synset ID for the swapped object |
| `template_id` | str | Which template was used |
| `vote_llama405b` | bool \| null | Per-model vote |
| `vote_sonnet` | bool \| null | Per-model vote |
| `vote_opus` | bool \| null | Per-model vote (only if escalated) |
| `consensus` | str | agreed, disputed, escalated |
| `source` | str | "wordnet" or "augmentation" |

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
