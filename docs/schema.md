# LateNet Output Schema

Exported as Parquet. Each row represents a single statement.

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Unique identifier |
| `statement` | str | Natural language statement |
| `label` | bool | True or false |
| `pair_id` | str | Links true/false counterparts |
| `domain` | str | WordNet top-level category (animal, plant, artifact, etc.) |
| `relationship_type` | str | hypernym, meronym, antonym, etc. |
| `difficulty` | str | hard, medium, easy |
| `semantic_distance` | int | Hops in WordNet tree between true and swapped concept |
| `source_synset` | str | WordNet synset ID for the subject |
| `target_synset` | str | WordNet synset ID for the object (true version) |
| `neg_synset` | str \| null | WordNet synset ID for the swapped object (false version) |
| `template_id` | str | Which template was used |
| `vote_llama405b` | bool \| null | Per-model vote |
| `vote_sonnet` | bool \| null | Per-model vote |
| `vote_opus` | bool \| null | Per-model vote (only if escalated) |
| `consensus` | str | agreed, disputed, escalated |
| `source` | str | "wordnet" or "augmentation" |

## Difficulty Tiers

| Tier | Strategy | Semantic Distance |
|------|----------|-------------------|
| **Hard** | Sibling swap (same parent) | 2 hops |
| **Medium** | Cousin swap (same grandparent) | 4 hops |
| **Easy** | Distant subtree swap | 6+ hops |

## Negation Strategies

1. **Sibling swap** - replace the object with a sibling in the WordNet tree
2. **Distant swap** - replace with a node from a different subtree
3. **Direct negation** - syntactic negation of a true statement
4. **Reverse relation** - flip the relation where it becomes false

## Validation Pipeline

- **Voters:** Llama 405B Instruct, Claude Sonnet
- **Escalation:** On disagreement, escalate to Claude Opus as tiebreaker
- **Consensus values:** `agreed` (voters match), `escalated` (tiebreaker used), `disputed` (no resolution)
