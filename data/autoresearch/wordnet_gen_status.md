# WordNet Generator: Current State, Solutions, and Gaps

**Date:** 2026-03-29
**Context:** WordNet generator is excluded from v1 dataset (17% yield vs 80%+ for other generators). This document synthesizes the pipeline, what we've fixed, what's still broken, and what solutions exist.

## The Problem in One Sentence

64% of WordNet FALSE statements are flagged as awkward by validators — swapped synsets cross semantic boundaries that make statements sound nonsensical even though they're technically false.

## Pipeline Summary

```
WordNetWalker (walker.py)
  → Filter to physical_entity.n.01 subtree (concrete nouns only)
  → Sort by Brown corpus frequency (common concepts first)
  → Extract hypernymy relationships (source → target)
      ↓
WordNetGenerator (wordnet_gen.py)
  → Primary-sense filter (skip time.n.05, keep time.n.01)
  → Per-source cap (max 3 pairs per source synset)
  → Round-robin across relationship types
      ↓
apply_negation (strategies.py)
  → sibling_swap (HARD): pick sibling of target → "Dog is a cat"
  → cousin_swap (MEDIUM): pick cousin of target → "Dog is a lizard"
  → distant_swap (EASY): pick from different tree → "Dog is a building"
  → reverse_relation: swap subject/object → "Animal is a dog"
  → direct_negation: insert "not" → "Dog is not an animal"
      ↓
pick_negation_synset (tiers.py)
  → Wu-Palmer coherence filter (wup ≥ 0.55)
  → Lemma overlap exclusion (no "man is a man")
  → Frequency-weighted choice (log-flattened for distant)
  → Category-level filter for distant (attested, ≥2 hyponyms, depth ≤5)
```

## What We Fixed (PR #28)

| Fix | Impact |
|-----|--------|
| Physical_entity subtree restriction | Eliminated abstract nouns ("time is a group") |
| Primary-sense filter | Skips obscure senses (time.n.05 etc.) |
| Lemma overlap exclusion | No more "man is a type of man" tautologies |
| Wu-Palmer coherence (≥0.55) | Filters taxonomically distant swaps |
| Category-level distant_swap | Only real categories (attested, ≥2 hyponyms, depth ≤5) |
| Log-flattened frequency | No more repeated "is a person" |
| Distant_swap tautology fix | Source lemma exclusion was missing |

**Result:** Awkward rate dropped from 55.4% → 36.7%. Real progress, but not enough.

## What's Still Broken

### 1. FALSE statement awkwardness (64% of false, 9.4% of true)

The remaining awkwardness is structural — WordNet siblings/cousins are **taxonomically** related but not **semantically** coherent for human readers:

| Example | Why It's Awkward |
|---------|-----------------|
| "Jew is an animal" | WordNet: person.n.01 siblings include animal.n.01 |
| "County is an Earth" | WordNet: region.n.01 siblings include astronomical bodies |
| "Woman is a Black" | WordNet: person.n.01 has ethno-religious hyponyms |
| "Growth is a servant" | WordNet: organism.n.01 subtree includes abstract growth |
| "Water belongs to the category of geographical area" | Distant swap crosses domains |
| "Child belongs to the category of earth" | Same — physical_entity too broad |

**Root cause:** WordNet's taxonomy reflects IS-A relationships that are technically correct but violate human semantic expectations. "Person" and "animal" are both "organisms" — a valid taxonomy — but "Person is an animal" sounds wrong to most readers.

### 2. Low unique pair yield (64/100 at seed=42, 36% dupes)

The physical_entity filter + primary-sense filter + Wu-Palmer filter dramatically reduces the candidate pool. At 1000 max_pairs, only 64 unique pairs survive dedup. The generator is hitting the ceiling of its filtered synset space.

### 3. Slow generation (480s for 64 pairs = 8 min)

Wu-Palmer similarity computation on every candidate pair is expensive. The walker processes all physical_entity synsets (~25K) even though most are filtered out downstream.

## Why Heuristics Can't Fix This

The core issue is that **WordNet's taxonomy doesn't align with human semantic intuition for "X is a Y" statements**. No amount of filtering on:
- Depth (already ≤5)
- Frequency (already frequency-weighted)
- Wu-Palmer similarity (already ≥0.55)
- Physical entity (already restricted)

...can fix the fundamental mismatch. Wu-Palmer measures taxonomic distance, not semantic plausibility. Two synsets can be taxonomically close (wup=0.7) but semantically absurd as swap partners.

## Promising Solutions

### 1. NDIF Completion Lookup Table (highest impact, medium effort)

**Status:** Proof of concept on `autoresearch/mar28-coherence` branch (`experiments/ndif_completions.py`).

**Idea:** Instead of using WordNet siblings as false statement candidates, ask Llama 405B: "A dog is a type of ___" and take the top-20 completions. These are **naturally coherent** false categories because they come from the model's own semantic space:

```
"A dog is a type of" → animal, mammal, pet, carnivore, canine
"A house is a type of" → building, shelter, dwelling, structure
```

**Why it works:** The model's completions reflect human semantic expectations, not taxonomy. "Dog is a mammal" (true) vs "Dog is a reptile" (false but coherent) — the kind of statement humans find natural.

**Implementation:** Precompute a JSON lookup table mapping source synsets → top-20 Llama categories. Use these as the false statement pool instead of WordNet siblings. Cost: ~40 batched NDIF calls.

**Caveat:** NDIF rotates models every few days. The lookup table would be version-stamped and regenerated when the model changes. Consider caching locally.

### 2. Wikidata P31/P279 IS-A Chains (medium impact, low effort)

**Idea:** Wikidata's "instance of" (P31) and "subclass of" (P279) relationships are maintained by human encyclopedists, not linguists. They reflect how people actually categorize things:

```
Q144 (dog) → P31 → Q16521 (taxon)
Q144 (dog) → P279 → Q39201 (pet)
```

We already have wikistash with the full Wikidata graph. Could supplement or replace WordNet's taxonomy for concrete noun IS-A pairs. The biology generator already does this successfully for taxonomic relationships.

### 3. Embedding-Based Coherence Filter (low-medium impact, medium effort)

**Idea:** Replace Wu-Palmer (taxonomic distance) with cosine similarity in a semantic embedding space (GloVe, word2vec, or small sentence transformer).

**Status:** GloVe prototype was tested and was inconclusive — GloVe captures topical co-occurrence, not IS-A plausibility. A sentence-level model (e.g., encode "X is a Y" and check coherence) might work better but adds a dependency.

**Best as a complement to NDIF completions**, not a replacement.

### 4. Hybrid: WordNet Structure + LLM Coherence

**Idea:** Keep WordNet's taxonomy for the TRUE side (it's good at IS-A facts) but use LLM-derived candidates for the FALSE side. This preserves the diversity of WordNet's 80K+ noun synsets while fixing the false statement quality.

**Flow:**
1. Walk WordNet for source-target pairs (TRUE)
2. For each source, look up NDIF completions
3. Pick false category from completions that **isn't** in the source's hypernym chain
4. Render false statement with LLM-derived category

This decouples the two sides: WordNet handles breadth, LLM handles coherence.

## Gaps That Need Research

1. **NDIF model rotation:** How to handle the lookup table going stale when NDIF switches from Llama 405B to another model? Version-stamp + cache, or make it model-agnostic?

2. **Wikidata coverage:** Does wikistash have enough P31/P279 relationships for general nouns (not just biology)? Need to audit coverage for WordNet's top-1000 concrete nouns.

3. **Sentence-level coherence:** Can a small model (e.g., all-MiniLM-L6-v2) reliably score "X is a Y" plausibility? Would need a labeled eval set.

4. **Scaling:** If NDIF completions work, can we generate 10K+ unique WordNet pairs? The lookup table approach scales linearly with synset count.

5. **Difficulty calibration:** With LLM-derived false categories, how do we assign difficulty tiers? The current hop-based system (sibling=hard, cousin=medium, distant=easy) won't apply. May need confidence-based difficulty (model's own logit gap as proxy).

## Current Numbers

| Metric | Value |
|--------|-------|
| Aff contest rate | 13.3% |
| Aff awkward rate | 36.7% |
| TRUE awkward | 9.4% |
| FALSE awkward | 64.1% |
| Unique pair yield | 64/100 (36% dupes) |
| Generation time | 480s / 64 pairs |
| Yield after curation | 17.2% |
| Wu-Palmer threshold | 0.55 |

## Recommendation

**NDIF completion lookup is the path forward.** It directly addresses the root cause (semantic incoherence of taxonomy-derived swaps) rather than adding more filters to an inherently mismatched data source. The proof of concept works, the cost is trivial, and it's composable with existing Wu-Palmer and frequency filters as secondary checks.

WordNet remains valuable for the TRUE side — its taxonomy is the best structured source of IS-A facts at scale. The fix is specifically about how we generate false alternatives.
