# LateNet Output Schema

Exported as Parquet. Each row represents a single statement.

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Unique identifier |
| `statement` | str | Natural language statement |
| `label` | bool | True or false |
| `pair_id` | str | Links true/false counterparts |
| `domain` | str | Semantic domain (animal, geography, chemistry, etc.) |
| `relation_type` | str | Logical relation (hypernymy, contained-in, greater-than, etc.) |
| `difficulty` | str | hard, medium, easy |
| `tier` | int | Distribution shift tier (1=in-distribution, 2=adjacent, 3=OOD) |
| `semantic_distance` | int \| null | Hops in hierarchy (where applicable) |
| `source_synset` | str \| null | WordNet synset ID for the subject (WordNet pairs only) |
| `target_synset` | str \| null | WordNet synset ID for the object (WordNet pairs only) |
| `neg_synset` | str \| null | WordNet synset ID for the swapped object (WordNet pairs only) |
| `generator` | str | Which generator produced this pair (wordnet, geography, temporal, etc.) |
| `template_id` | str | Which template was used |
| `negation_strategy` | str | sibling_swap, distant_swap, direct_negation, reverse_relation |
| `vote_llama405b` | bool \| null | Per-model vote |
| `vote_sonnet` | bool \| null | Per-model vote |
| `vote_opus` | bool \| null | Per-model vote (only if escalated) |
| `consensus` | str | agreed, disputed, escalated |

## Difficulty Tiers (semantic distance)

| Tier | Strategy | Semantic Distance |
|------|----------|-------------------|
| **Hard** | Sibling swap (same parent) | 2 hops |
| **Medium** | Cousin swap (same grandparent) | 3-5 hops |
| **Easy** | Distant subtree swap | 6+ hops |

## Distribution Shift Tiers

| Tier | Description |
|------|-------------|
| **1** | In-distribution: same templates, same relation types, held-out entities |
| **2** | Distribution-adjacent: same logical relation, different domain |
| **3** | Out-of-distribution: entirely new relation types not seen in training |

## Negation Strategies

1. **Sibling swap** — replace the object with a semantically close alternative
2. **Distant swap** — replace with a semantically distant alternative
3. **Direct negation** — syntactic negation of a true statement
4. **Reverse relation** — flip the relation where it becomes false

## Generators

| Generator | Data Source | Relation Types |
|-----------|------------|----------------|
| `wordnet` | NLTK WordNet | hypernymy, meronymy, antonymy, sibling |
| `geography` | Natural Earth, GeoNames | contained-in, north-of, closer-to |
| `temporal` | Historical databases | before, after, century-of |
| `chemistry` | Periodic table, PubChem | symbol-of, property-of, group-membership |
| `language` | Translation dictionaries | translates-to |
| `magnitude` | World Bank, reference tables | greater-than, less-than |
| `authorship` | Literary/scientific databases | written-by, proposed-by |
| `biology` | NCBI taxonomy | is-a, part-of |

## Validation Pipeline

- **Voters:** Llama 405B Instruct, Claude Sonnet
- **Escalation:** On disagreement, escalate to Claude Opus as tiebreaker
- **Consensus values:** `agreed` (voters match), `escalated` (tiebreaker used), `disputed` (no resolution)
