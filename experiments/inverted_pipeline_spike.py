"""Spike: Inverted pipeline — model generates false candidates, WordNet validates.

Instead of WordNet picking candidates and the model scoring them, we:
1. Prompt Llama 70B base: "True or false? A {word} is a ___"
2. Decode top-N completions into words
3. Map words → WordNet noun synsets
4. Keep only synsets NOT in source's hypernym chain (= genuinely false)
5. Measure: yield rate, falseness rate, quality, difficulty distribution

Key question: does letting the model generate candidates produce inherently
coherent false statements, eliminating the need for heuristic filtering?
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer
from nltk.corpus import wordnet as wn

from lmprobe.extraction import ActivationExtractor
from lmprobe.retry import retry_with_backoff

from latenet.difficulty.tiers import _is_physical_entity
from latenet.generators.wordnet_gen import _is_primary_sense
from latenet.wordnet.distance import semantic_distance

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LLAMA_MODEL = "meta-llama/Llama-3.1-70B"
LLAMA_TOKENIZER = "meta-llama/Llama-3.1-70B"
LOGIT_TOP_K = 10000  # fetch top 10K tokens per word

# How many top tokens to decode and attempt synset mapping
DECODE_TOP_N = 200

# Test words — mix of concrete/abstract, common/uncommon
TEST_WORDS = [
    # Common concrete nouns
    "dog", "cat", "car", "tree", "bird", "fish", "horse",
    # Less common but still concrete
    "diamond", "mushroom", "violin", "sword", "whale",
    # Previously problematic (abstract-ish or obscure senses)
    "bridge", "chair", "potato", "lamp", "corn",
    # Stress test — words with many senses
    "bank", "spring", "bat",
]

OUTPUT_DIR = Path("experiments/spike_results")


def _base_prompt(word: str) -> str:
    return f"True or false? A {word} is a"


def _get_primary_synset(word: str):
    """Get primary noun synset, or None if not a physical entity."""
    synsets = wn.synsets(word, pos="n")
    if not synsets:
        return None
    s = synsets[0]
    if not _is_physical_entity(s):
        return None
    if not _is_primary_sense(s):
        return None
    return s


def _all_hypernym_names(synset) -> set[str]:
    """Get ALL synset names in the hypernym closure (ancestors + self)."""
    names = set()
    queue = [synset]
    while queue:
        s = queue.pop()
        if s.name() in names:
            continue
        names.add(s.name())
        queue.extend(s.hypernyms())
    return names


def _token_to_synsets(decoded_word: str) -> list:
    """Map a decoded token string to WordNet noun synsets.

    Tries the word as-is, then lowercased. Returns empty list if no match.
    """
    # Clean up: strip whitespace and punctuation
    word = decoded_word.strip().strip(".,;:!?\"'()[]{}").lower()
    if not word or len(word) < 2:
        return []

    # Skip obvious non-nouns (articles, prepositions, etc.)
    SKIP_WORDS = {
        "the", "a", "an", "of", "in", "on", "at", "to", "for", "and",
        "or", "not", "is", "are", "was", "were", "be", "been", "being",
        "very", "most", "more", "less", "much", "many", "few", "some",
        "type", "kind", "form", "sort", "class", "group",  # generic classifiers
        "living", "natural", "common", "large", "small", "new", "old",
    }
    if word in SKIP_WORDS:
        return []

    synsets = wn.synsets(word, pos="n")
    return synsets


def main():
    if "NNSIGHT_API_KEY" not in os.environ and "NDIF_API_KEY" in os.environ:
        os.environ["NNSIGHT_API_KEY"] = os.environ["NDIF_API_KEY"]
    if "NNSIGHT_API_KEY" not in os.environ:
        print("ERROR: Set NDIF_API_KEY or NNSIGHT_API_KEY env var")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {LLAMA_TOKENIZER}")
    tok = AutoTokenizer.from_pretrained(LLAMA_TOKENIZER)
    print(f"  Vocab size: {tok.vocab_size}")

    print(f"Initializing ActivationExtractor: {LLAMA_MODEL}")
    ext = ActivationExtractor(
        model_name=LLAMA_MODEL, device="cpu", layers=[],
        backend="nnsight", remote=True,
    )

    # Resolve test words
    test_synsets = []
    for word in TEST_WORDS:
        syn = _get_primary_synset(word)
        if syn:
            test_synsets.append((word, syn))
        else:
            print(f"  SKIP {word}: not a primary physical entity noun")
    print(f"\n{len(test_synsets)} test words resolved\n")

    # Aggregate metrics
    agg = {
        "tokens_decoded": [],       # how many of top-N tokens we decoded
        "tokens_mapped": [],        # how many mapped to ≥1 WordNet synset
        "synsets_found": [],        # total unique synsets found
        "synsets_false": [],        # synsets NOT in hypernym chain
        "synsets_true": [],         # synsets IN hypernym chain (would be true statements)
        "synsets_physical": [],     # synsets under physical_entity
        "synsets_primary": [],      # synsets that are primary sense
        "yield_rate": [],           # tokens_mapped / tokens_decoded
        "falseness_rate": [],       # synsets_false / synsets_found
        "quality_false": [],        # false + physical + primary (usable candidates)
    }
    per_word_results = []

    for word, synset in test_synsets:
        print("=" * 70)
        print(f"SOURCE: {word} ({synset.name()})")
        hypernyms = synset.hypernyms()
        print(f"  Hypernyms: {[h.name() for h in hypernyms]}")

        # Build hypernym closure for this source
        hyp_closure = _all_hypernym_names(synset)
        print(f"  Hypernym closure size: {len(hyp_closure)}")

        # Fetch logits from NDIF
        prompt = _base_prompt(word)
        print(f"  Querying NDIF (top_k={LOGIT_TOP_K})...")

        try:
            logits, _mask, logits_indices = retry_with_backoff(
                lambda p=prompt: ext.extract_logits_only(
                    [p], remote=True, logit_top_k=LOGIT_TOP_K,
                ),
                max_retries=3, base_delay=3.0, max_delay=60.0,
                context=f"inverted:{word}",
            )
        except Exception as e:
            print(f"  NDIF ERROR: {e}\n")
            continue

        last_logits = logits[0, -1, :]
        last_indices = logits_indices[0, -1, :]

        # Sort by logit descending
        sorted_pairs = sorted(
            zip(last_indices.tolist(), last_logits.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

        # --- STEP 1: Decode top-N tokens ---
        decoded_tokens = []
        for tid, logit_val in sorted_pairs[:DECODE_TOP_N]:
            decoded = tok.decode([tid]).strip()
            decoded_tokens.append((tid, decoded, logit_val))

        print(f"\n  Top-10 raw completions:")
        for i, (tid, decoded, logit_val) in enumerate(decoded_tokens[:10]):
            print(f"    {i+1:2d}. '{decoded}'  logit={logit_val:.3f}")

        # --- STEP 2: Map to WordNet synsets ---
        mapped = []       # (decoded_word, synset_list, logit, rank)
        unmapped = []     # (decoded_word, logit, rank, reason)
        seen_synsets = set()

        for rank, (tid, decoded, logit_val) in enumerate(decoded_tokens):
            synsets = _token_to_synsets(decoded)
            if synsets:
                # Deduplicate: only count each synset once
                new_synsets = [s for s in synsets if s.name() not in seen_synsets]
                if new_synsets:
                    for s in new_synsets:
                        seen_synsets.add(s.name())
                    mapped.append((decoded, new_synsets, logit_val, rank))
                else:
                    unmapped.append((decoded, logit_val, rank, "duplicate_synsets"))
            else:
                unmapped.append((decoded, logit_val, rank, "no_synsets"))

        # --- STEP 3: Classify synsets as TRUE or FALSE ---
        true_candidates = []   # in hypernym chain → would make true statements
        false_candidates = []  # NOT in chain → genuinely false

        for decoded, synset_list, logit_val, rank in mapped:
            for s in synset_list:
                in_chain = s.name() in hyp_closure
                is_physical = _is_physical_entity(s)
                is_primary = _is_primary_sense(s)
                dist = semantic_distance(synset, s)

                entry = {
                    "decoded": decoded,
                    "synset": s.name(),
                    "lemma": s.lemma_names()[0].replace("_", " "),
                    "logit": logit_val,
                    "rank": rank,
                    "in_hypernym_chain": in_chain,
                    "is_physical_entity": is_physical,
                    "is_primary_sense": is_primary,
                    "semantic_distance": dist,
                    "min_depth": s.min_depth(),
                }

                if in_chain:
                    true_candidates.append(entry)
                else:
                    false_candidates.append(entry)

        # --- Usable false candidates: false + physical + primary sense ---
        usable = [c for c in false_candidates
                   if c["is_physical_entity"] and c["is_primary_sense"]]

        # Classify by difficulty (semantic distance from source)
        hard = [c for c in usable if c["semantic_distance"] <= 2]
        medium = [c for c in usable if 2 < c["semantic_distance"] <= 5]
        easy = [c for c in usable if c["semantic_distance"] > 5]

        # --- Report ---
        n_decoded = len(decoded_tokens)
        n_mapped = len(mapped)
        n_synsets = len(seen_synsets)
        n_false = len(false_candidates)
        n_true = len(true_candidates)
        n_usable = len(usable)

        print(f"\n  Pipeline funnel:")
        print(f"    Tokens decoded:     {n_decoded}")
        print(f"    Mapped to synsets:   {n_mapped} ({n_mapped/n_decoded:.0%} yield)")
        print(f"    Unique synsets:      {n_synsets}")
        print(f"    TRUE (in chain):     {n_true} ({n_true/max(n_synsets,1):.0%})")
        print(f"    FALSE (not in chain):{n_false} ({n_false/max(n_synsets,1):.0%})")
        print(f"    Usable FALSE:        {n_usable} (physical + primary)")
        print(f"    Difficulty: {len(hard)} hard, {len(medium)} medium, {len(easy)} easy")

        # Show top unmapped tokens
        no_synset = [u for u in unmapped if u[3] == "no_synsets"][:5]
        if no_synset:
            print(f"\n  Top unmapped tokens: {', '.join(repr(u[0]) for u in no_synset)}")

        # Show top TRUE candidates (these are the "too true" problem)
        if true_candidates:
            print(f"\n  Top TRUE candidates (model thinks these are correct categories):")
            for c in sorted(true_candidates, key=lambda x: x["logit"], reverse=True)[:5]:
                print(f"    '{c['lemma']}'  logit={c['logit']:.3f}  {c['synset']}  dist={c['semantic_distance']}")

        # Show top USABLE FALSE candidates (the money shots)
        if usable:
            print(f"\n  Top usable FALSE candidates (these become false statements):")
            for c in sorted(usable, key=lambda x: x["logit"], reverse=True)[:10]:
                diff = "hard" if c["semantic_distance"] <= 2 else "med" if c["semantic_distance"] <= 5 else "easy"
                print(f"    'A {word} is a {c['lemma']}'  logit={c['logit']:.3f}  dist={c['semantic_distance']}  [{diff}]  {c['synset']}")

        # Show example statements
        hyp_name = hypernyms[0].lemma_names()[0].replace("_", " ") if hypernyms else "?"
        print(f"\n  Example statements:")
        print(f"    TRUE:  A {word} is a type of {hyp_name}")
        if usable:
            best = sorted(usable, key=lambda x: x["logit"], reverse=True)[0]
            print(f"    FALSE: A {word} is a type of {best['lemma']}  [logit={best['logit']:.3f}, dist={best['semantic_distance']}]")

        # Aggregate
        agg["tokens_decoded"].append(n_decoded)
        agg["tokens_mapped"].append(n_mapped)
        agg["synsets_found"].append(n_synsets)
        agg["synsets_false"].append(n_false)
        agg["synsets_true"].append(n_true)
        agg["synsets_physical"].append(sum(1 for c in false_candidates if c["is_physical_entity"]))
        agg["synsets_primary"].append(sum(1 for c in false_candidates if c["is_primary_sense"]))
        agg["yield_rate"].append(n_mapped / n_decoded if n_decoded else 0)
        agg["falseness_rate"].append(n_false / n_synsets if n_synsets else 0)
        agg["quality_false"].append(n_usable)

        per_word_results.append({
            "word": word,
            "synset": synset.name(),
            "hypernym_closure_size": len(hyp_closure),
            "tokens_decoded": n_decoded,
            "tokens_mapped": n_mapped,
            "unique_synsets": n_synsets,
            "true_candidates": n_true,
            "false_candidates": n_false,
            "usable_false": n_usable,
            "difficulty_breakdown": {
                "hard": len(hard), "medium": len(medium), "easy": len(easy),
            },
            "top_usable_false": sorted(usable, key=lambda x: x["logit"], reverse=True)[:10],
            "top_true": sorted(true_candidates, key=lambda x: x["logit"], reverse=True)[:5],
        })

        print()
        time.sleep(0.5)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("SUMMARY: INVERTED PIPELINE FUNNEL")
    print("=" * 70)

    def _avg(lst):
        return sum(lst) / len(lst) if lst else 0

    print(f"\n  Across {len(per_word_results)} source words (top-{DECODE_TOP_N} tokens each):")
    print(f"    Avg tokens decoded:     {_avg(agg['tokens_decoded']):.0f}")
    print(f"    Avg mapped to synsets:  {_avg(agg['tokens_mapped']):.1f}  ({_avg(agg['yield_rate']):.0%} yield)")
    print(f"    Avg unique synsets:     {_avg(agg['synsets_found']):.1f}")
    print(f"    Avg TRUE (in chain):    {_avg(agg['synsets_true']):.1f}")
    print(f"    Avg FALSE (not chain):  {_avg(agg['synsets_false']):.1f}  ({_avg(agg['falseness_rate']):.0%} of synsets)")
    print(f"    Avg usable FALSE:       {_avg(agg['quality_false']):.1f}")
    print()

    # Per-word table
    print(f"  {'Word':<12s} {'Decoded':>7s} {'Mapped':>6s} {'Yield':>6s} {'Syns':>5s} {'TRUE':>5s} {'FALSE':>5s} {'Usable':>6s}")
    print(f"  {'-'*12} {'-'*7} {'-'*6} {'-'*6} {'-'*5} {'-'*5} {'-'*5} {'-'*6}")
    for r in per_word_results:
        yield_pct = r["tokens_mapped"] / r["tokens_decoded"] if r["tokens_decoded"] else 0
        print(f"  {r['word']:<12s} {r['tokens_decoded']:>7d} {r['tokens_mapped']:>6d} {yield_pct:>5.0%} {r['unique_synsets']:>5d} {r['true_candidates']:>5d} {r['false_candidates']:>5d} {r['usable_false']:>6d}")

    # Difficulty distribution across all words
    total_hard = sum(r["difficulty_breakdown"]["hard"] for r in per_word_results)
    total_med = sum(r["difficulty_breakdown"]["medium"] for r in per_word_results)
    total_easy = sum(r["difficulty_breakdown"]["easy"] for r in per_word_results)
    total_usable = total_hard + total_med + total_easy
    print(f"\n  Difficulty distribution (all usable FALSE):")
    print(f"    Hard (dist≤2):   {total_hard}  ({total_hard/max(total_usable,1):.0%})")
    print(f"    Medium (dist≤5): {total_med}  ({total_med/max(total_usable,1):.0%})")
    print(f"    Easy (dist>5):   {total_easy}  ({total_easy/max(total_usable,1):.0%})")

    # Showcase: best false statement per word
    print(f"\n  SHOWCASE — best model-generated false statement per word:")
    print(f"  {'Word':<12s} {'TRUE statement':<35s} {'BEST FALSE':<40s} {'Dist':>4s}")
    print(f"  {'-'*12} {'-'*35} {'-'*40} {'-'*4}")
    for r in per_word_results:
        hyps = wn.synset(r["synset"]).hypernyms()
        true_cat = hyps[0].lemma_names()[0].replace("_", " ") if hyps else "?"
        true_stmt = f"A {r['word']} is a type of {true_cat}"
        if r["top_usable_false"]:
            best = r["top_usable_false"][0]
            false_stmt = f"A {r['word']} is a type of {best['lemma']}"
            dist = str(best["semantic_distance"])
        else:
            false_stmt = "(no usable candidates)"
            dist = "-"
        print(f"  {r['word']:<12s} {true_stmt:<35s} {false_stmt:<40s} {dist:>4s}")

    # Save results
    results_path = OUTPUT_DIR / "inverted_pipeline_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "model": LLAMA_MODEL,
            "logit_top_k": LOGIT_TOP_K,
            "decode_top_n": DECODE_TOP_N,
            "summary": {
                "n_words": len(per_word_results),
                "avg_yield_rate": round(_avg(agg["yield_rate"]), 4),
                "avg_falseness_rate": round(_avg(agg["falseness_rate"]), 4),
                "avg_usable_false": round(_avg(agg["quality_false"]), 1),
                "difficulty_distribution": {
                    "hard": total_hard, "medium": total_med, "easy": total_easy,
                },
            },
            "per_word": per_word_results,
        }, f, indent=2)
    print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    main()
