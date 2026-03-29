# Autoresearch Progress Report

**Date:** 2026-03-28/29
**Merged:** AlliedToasters/LateNet#28 → main
**Experimental branch:** `autoresearch/mar28-coherence` (not merged, contains NDIF experiment)

## What Was Done

An autoresearch-inspired iteration loop over 6 WordNet-focused trials, plus one full-generator ablation trial. All work driven by the negation ablation study findings.

### Merged to Main (PR #28)

| Fix | File(s) | Impact |
|-----|---------|--------|
| WordNet lemma overlap filter | `tiers.py` | Eliminates tautologies ("A man is a type of man") |
| Wikidata cache removal | `wikidata.py` | Strips parquet cache; wikistash is sole backend |
| Name capitalization | `wikidata.py` | Normalizes lowercase Wikidata names ("gabriel" → "Gabriel") |
| Geography city/country collision | `geography.py` | Skips Djibouti-in-Djibouti type pairs |
| Anatomy negation grammar | `strategies.py` | "do not both belong to" (was "both do not belong to") |
| Noble gas singularization | `chemistry.py` | "noble gas" (was "noble gase" from `rstrip("s")` on "gases") |
| Physical_entity restriction | `walker.py`, `tiers.py` | WordNet sources restricted to concrete nouns |
| Primary-sense filter | `wordnet_gen.py` | Skip obscure senses (time.n.05 etc.) |
| Category-level distant_swap | `tiers.py` | ≥2 hyponyms, attested, depth ≤5, physical_entity |
| Wu-Palmer coherence filter | `tiers.py` | wup ≥ 0.55 on sibling/cousin swaps |
| Log-flatten distant frequency | `tiers.py` | Diverse replacements instead of repeated "person" |
| Distant_swap tautology fix | `tiers.py` | Lemma overlap check was missing |
| Batch stratification fix | `batch.py` | groupby relation_type not difficulty |

### Key Metrics

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| WordNet awkward (aff) | 55.4% | ~27-33% | **-22 to -28pp** |
| WordNet awkward (neg) | 11.9% | ~5-7% | **-5 to -7pp** |
| Erroneous output rate | 1.2% | <1% | **-0.2pp+** |
| Negation fallback | 0.1% | 0% | fixed |
| Overall contest (aff) | 10.5% | ~15-17% | +5pp (different composition) |

### What Didn't Work

1. **Biology rank template change** ("lower/higher" vs "more specific/general") — made neg contest worse (53% → 65.6%). Reverted. The Marks & Tegmark negation effect on hierarchy reasoning is inherent to negation processing, not vocabulary.

2. **Primary-sense filter** alone — no impact on awkward rate because problematic sources ARE primary senses. Only helped when combined with physical_entity restriction.

3. **GloVe similarity** — captures topical co-occurrence, not IS-A plausibility. Can't discriminate good from bad swaps.

4. **Wu-Palmer threshold 0.6** — too aggressive; degraded cousin_swap quality. 0.55 is the sweet spot.

## Remaining Awkwardness (~27-33%)

| Source | Rate | Root Cause |
|--------|------|-----------|
| Sibling_swap FALSE | ~27% | WordNet siblings taxonomically related but semantically odd |
| Distant_swap FALSE | ~45% | Physical_entity pool still spans very different domains |
| Cousin_swap FALSE | ~44% | Two-hop cousins cross semantic boundaries |
| TRUE statements | ~6% | Inherent WordNet taxonomy ("Location is an object") |

## Promising Next Steps

### 1. NDIF Completion Lookup Table (highest impact, medium effort)

Proof of concept on `autoresearch/mar28-coherence` branch (`experiments/ndif_completions.py`). Llama 405B completions for "X is a type of ___" produce excellent natural categories:

```
"A dog is a type of" → animal, mammal, pet, carnivore, canine
"A house is a type of" → building, shelter, dwelling, structure
```

**Approach:** Precompute a JSON lookup table mapping each source synset to its top-20 Llama-suggested categories. Use these as the false statement pool instead of WordNet siblings. Cost: ~40 batched NDIF calls (trivial).

**Caveat:** NDIF rotates models every few days. The lookup table would be version-stamped and regenerated when the model changes. Consider caching locally.

### 2. Wikidata P31/P279 IS-A Chains (medium impact, low effort)

We already have wikistash. Wikidata's "instance of" (P31) and "subclass of" (P279) relationships are more natural than WordNet's taxonomy because they're maintained by humans writing encyclopedic descriptions. Could supplement or replace WordNet for concrete noun IS-A pairs.

### 3. Embedding-Based Coherence (low-medium impact, medium effort)

For sibling_swap: check cosine similarity between source and replacement using GloVe or a small sentence transformer. Not as promising as NDIF completions (GloVe prototype was inconclusive) but could serve as a lightweight backup.

### 4. Source Synset Curation (low effort, diminishing returns)

The remaining ~6% awkward TRUE statements come from a small set of problematic source synsets. A curated blocklist of ~10 synsets would eliminate them, but this is a band-aid — the physical_entity + primary-sense filters already handle the structural issue.

## Archived Data

Trial ledgers in `data/graveyard/`:
- `ledger_trial0.parquet` — baseline (main branch, pre-fixes)
- `ledger_trial1.parquet` — trial-1 (5 QA fixes)
- `ledger_trial2a.parquet` — trial-2a (distant_swap filters)
- `ledger_trial2b.parquet` — trial-2b (depth filter)
- `ledger_trial3.parquet` — trial-3 (primary-sense filter)
- `ledger_trial4.parquet` — trial-4 (physical_entity restriction)
- `ledger_trial5.parquet` — trial-5 (Wu-Palmer filter)

Trial reports in `data/autoresearch/`:
- `trial-1-report.md` — full ablation with 5 fixes, spot check
- `trial-2-report.md` — WordNet-focused, distant_swap improvements
- `trial-4-report.md` — physical_entity restriction results
- `coherence-experiments-report.md` — GloVe, Wu-Palmer, NDIF experiments
