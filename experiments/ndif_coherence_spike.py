"""Spike: NDIF logit-guided false statement sampling for WordNet generator.

Uses Llama 70B base (via lmprobe/NDIF) with FULL VOCAB logits to:
1. Score every WordNet false candidate by logit value
2. Compute coverage@k curves to find optimal top_k parameter
3. Compare logit-weighted vs random false picks qualitatively
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer
from nltk.corpus import wordnet as wn

# lmprobe for NDIF access (NOT raw nnsight)
from lmprobe.extraction import ActivationExtractor
from lmprobe.retry import retry_with_backoff

# Reuse existing LateNet infrastructure
from latenet.difficulty.tiers import (
    _siblings_of,
    _cousins_of,
    _is_physical_entity,
    MIN_WUP_SIMILARITY,
)
from latenet.generators.wordnet_gen import _is_primary_sense

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LLAMA_MODEL = "meta-llama/Llama-3.1-70B"
LLAMA_TOKENIZER = "meta-llama/Llama-3.1-70B"

# K values to evaluate for coverage@k
K_VALUES = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]

# Hand-picked common physical-entity nouns for the spike
TEST_WORDS = [
    "dog", "cat", "car", "tree", "house", "fish", "bird", "chair",
    "river", "mountain", "flower", "horse", "knife", "piano", "snake",
    "bridge", "diamond", "airplane", "potato", "whale",
]

OUTPUT_DIR = Path("experiments/spike_results")


def _base_completion_prompt(word: str) -> str:
    """Plain text prompt for base model completion — no chat format."""
    return f"True or false? A {word} is a"


def _get_synset_for_word(word: str):
    """Get the primary noun synset for a word, or None."""
    synsets = wn.synsets(word, pos="n")
    if not synsets:
        return None
    s = synsets[0]
    if not _is_physical_entity(s):
        return None
    if not _is_primary_sense(s):
        return None
    return s


def _get_false_candidates(synset):
    """Get sibling + cousin false candidates for a synset, with Wu-Palmer filter."""
    source_lemmas = {l.name().lower() for l in synset.lemmas()}

    def _no_overlap(pool):
        return [
            s for s in pool
            if s.name() != synset.name()
            and not source_lemmas & {l.name().lower() for l in s.lemmas()}
        ]

    siblings = _no_overlap(_siblings_of(synset))
    cousins = _no_overlap(_cousins_of(synset))

    def _wup_filter(pool):
        return [s for s in pool if (synset.wup_similarity(s) or 0) >= MIN_WUP_SIMILARITY]

    coherent_siblings = _wup_filter(siblings)
    coherent_cousins = _wup_filter(cousins)

    return {
        "siblings": coherent_siblings if coherent_siblings else siblings[:10],
        "cousins": coherent_cousins if coherent_cousins else cousins[:10],
    }


def _tokenize_candidates(candidates: list, tokenizer) -> dict:
    """Tokenize candidate lemma names. Returns token_map and stats."""
    token_map: dict[int, list[tuple]] = defaultdict(list)
    single_token = 0
    multi_token = 0

    for syn in candidates:
        lemma = syn.lemma_names()[0].replace("_", " ")
        token_ids = tokenizer.encode(f" {lemma}", add_special_tokens=False)
        first_id = token_ids[0]
        token_map[first_id].append((syn, lemma, token_ids))
        if len(token_ids) == 1:
            single_token += 1
        else:
            multi_token += 1

    collision_groups = sum(1 for group in token_map.values() if len(group) > 1)

    return {
        "token_map": dict(token_map),
        "stats": {
            "single_token": single_token,
            "multi_token": multi_token,
            "total": single_token + multi_token,
            "collision_groups": collision_groups,
            "unique_first_tokens": len(token_map),
        },
    }


def _coverage_at_k(candidate_first_token_ids: set[int], sorted_vocab_ids: list[int], k_values: list[int]) -> dict[int, float]:
    """Compute coverage of candidate first-tokens at each k value.

    sorted_vocab_ids: all vocab IDs sorted by logit descending.
    Returns {k: fraction_of_candidates_covered}.
    """
    if not candidate_first_token_ids:
        return {k: 0.0 for k in k_values}

    results = {}
    for k in k_values:
        top_k_set = set(sorted_vocab_ids[:k])
        covered = len(candidate_first_token_ids & top_k_set)
        results[k] = covered / len(candidate_first_token_ids)
    return results


def main():
    # Bridge NDIF_API_KEY → NNSIGHT_API_KEY
    if "NNSIGHT_API_KEY" not in os.environ and "NDIF_API_KEY" in os.environ:
        os.environ["NNSIGHT_API_KEY"] = os.environ["NDIF_API_KEY"]

    if "NNSIGHT_API_KEY" not in os.environ:
        print("ERROR: Set NDIF_API_KEY or NNSIGHT_API_KEY env var")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {LLAMA_TOKENIZER}")
    tok = AutoTokenizer.from_pretrained(LLAMA_TOKENIZER)
    vocab_size = tok.vocab_size
    print(f"  Vocab size: {vocab_size}")

    print(f"Initializing ActivationExtractor: {LLAMA_MODEL}")
    ext = ActivationExtractor(
        model_name=LLAMA_MODEL,
        device="cpu",
        layers=[],
        backend="nnsight",
        remote=True,
    )

    # Resolve test words → synsets
    test_synsets = []
    for word in TEST_WORDS:
        syn = _get_synset_for_word(word)
        if syn:
            test_synsets.append((word, syn))
        else:
            print(f"  SKIP {word}: not a primary physical entity noun")
    print(f"\n{len(test_synsets)} test synsets resolved\n")

    # Aggregate stats across all words
    all_coverage_at_k: dict[int, list[float]] = {k: [] for k in K_VALUES}
    all_candidate_ranks: list[int] = []  # rank of each candidate's first token
    per_word_results = []

    for word, synset in test_synsets:
        print("=" * 70)
        print(f"SOURCE: {word} ({synset.name()})")
        print(f"  Hypernyms: {[h.name() for h in synset.hypernyms()]}")

        # Get false candidates
        candidates = _get_false_candidates(synset)
        all_candidates = candidates["siblings"] + candidates["cousins"]
        if not all_candidates:
            print("  No false candidates found, skipping\n")
            continue

        print(f"  Candidates: {len(candidates['siblings'])} siblings, {len(candidates['cousins'])} cousins")

        # Tokenize
        tok_info = _tokenize_candidates(all_candidates, tok)
        stats = tok_info["stats"]
        token_map = tok_info["token_map"]
        candidate_first_tids = set(token_map.keys())

        print(f"  Token stats: {stats['single_token']} single-token, {stats['multi_token']} multi-token")
        print(f"  Unique first tokens: {stats['unique_first_tokens']} / {stats['total']} candidates")
        print(f"  Collision groups: {stats['collision_groups']}")

        # Query NDIF — FULL VOCAB (logit_top_k=None)
        prompt = _base_completion_prompt(word)
        print(f"\n  Querying NDIF (full vocab, ~{vocab_size} logits)...")

        try:
            logits, _mask, logits_indices = retry_with_backoff(
                lambda p=prompt: ext.extract_logits_only(
                    [p], remote=True, logit_top_k=None,
                ),
                max_retries=3,
                base_delay=3.0,
                max_delay=120.0,
                context=f"spike:{word}",
            )
        except Exception as e:
            print(f"  NDIF ERROR: {e}\n")
            continue

        # Full vocab: logits shape is (batch, seq, vocab_size), logits_indices is None
        last_logits = logits[0, -1, :]  # shape (vocab_size,)

        # Sort vocab by logit descending
        sorted_indices = last_logits.argsort(descending=True).tolist()

        # Build rank lookup: token_id → rank (0-indexed)
        rank_lookup = {tid: rank for rank, tid in enumerate(sorted_indices)}

        # Top-10 model completions
        print(f"\n  Top-10 model completions for '{word} is a ___':")
        for rank in range(10):
            tid = sorted_indices[rank]
            logit_val = last_logits[tid].item()
            decoded = tok.decode([tid]).strip()
            print(f"    {rank+1:2d}. {decoded:<20s}  logit={logit_val:.3f}")

        # Coverage@k
        cov_at_k = _coverage_at_k(candidate_first_tids, sorted_indices, K_VALUES)
        print(f"\n  Coverage@k (unique first tokens):")
        for k in K_VALUES:
            bar = "#" * int(cov_at_k[k] * 40)
            print(f"    k={k:<6d}  {cov_at_k[k]:5.1%}  {bar}")
            all_coverage_at_k[k].append(cov_at_k[k])

        # Score every candidate (full coverage guaranteed)
        scored = []
        for first_tid, group in token_map.items():
            logit_val = last_logits[first_tid].item()
            rank = rank_lookup[first_tid]
            for syn, lemma, full_ids in group:
                scored.append({
                    "synset": syn.name(),
                    "lemma": lemma,
                    "logit": logit_val,
                    "rank": rank,
                    "n_tokens": len(full_ids),
                })
                all_candidate_ranks.append(rank)

        scored.sort(key=lambda x: x["logit"], reverse=True)

        # Top-5 and bottom-3
        print(f"\n  Top-5 logit-ranked false candidates:")
        for s in scored[:5]:
            marker = "" if s["n_tokens"] == 1 else f" ({s['n_tokens']} tok)"
            print(f"    {s['lemma']:<25s}  logit={s['logit']:>8.3f}  rank={s['rank']:<6d}  {s['synset']}{marker}")

        print(f"\n  Bottom-3 logit-ranked:")
        for s in scored[-3:]:
            marker = "" if s["n_tokens"] == 1 else f" ({s['n_tokens']} tok)"
            print(f"    {s['lemma']:<25s}  logit={s['logit']:>8.3f}  rank={s['rank']:<6d}  {s['synset']}{marker}")

        # Sample statements
        best = scored[0]
        worst = scored[-1]
        hyp_name = synset.hypernyms()[0].lemma_names()[0].replace("_", " ") if synset.hypernyms() else "?"
        print(f"\n  Statements:")
        print(f"    TRUE:   A {word} is a {hyp_name}")
        print(f"    BEST:   A {word} is a {best['lemma']}  (logit={best['logit']:.3f}, rank={best['rank']})")
        print(f"    WORST:  A {word} is a {worst['lemma']}  (logit={worst['logit']:.3f}, rank={worst['rank']})")

        import random
        rng = random.Random(42)
        random_pick = rng.choice(all_candidates)
        random_lemma = random_pick.lemma_names()[0].replace("_", " ")
        random_tid = tok.encode(f" {random_lemma}", add_special_tokens=False)[0]
        random_rank = rank_lookup.get(random_tid, -1)
        random_logit = last_logits[random_tid].item() if random_tid < len(last_logits) else float("-inf")
        print(f"    RANDOM: A {word} is a {random_lemma}  (logit={random_logit:.3f}, rank={random_rank})")

        # Logit spread
        logit_vals = [s["logit"] for s in scored]
        print(f"\n  Logit spread: max={max(logit_vals):.3f}, min={min(logit_vals):.3f}, range={max(logit_vals)-min(logit_vals):.3f}")

        per_word_results.append({
            "word": word,
            "synset": synset.name(),
            "n_candidates": stats["total"],
            "n_unique_tokens": stats["unique_first_tokens"],
            "coverage_at_k": {str(k): round(v, 4) for k, v in cov_at_k.items()},
            "top5": scored[:5],
            "bottom3": scored[-3:],
        })

        print()
        time.sleep(0.5)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print("=" * 70)
    print("SUMMARY: COVERAGE@K (averaged across all words)")
    print("=" * 70)
    print(f"\n  {'k':>8s}  {'Avg':>6s}  {'Min':>6s}  {'Max':>6s}  {'Chart'}")
    print(f"  {'---':>8s}  {'---':>6s}  {'---':>6s}  {'---':>6s}  {'---'}")
    for k in K_VALUES:
        vals = all_coverage_at_k[k]
        if vals:
            avg = sum(vals) / len(vals)
            mn = min(vals)
            mx = max(vals)
            bar = "#" * int(avg * 50)
            print(f"  {k:>8d}  {avg:5.1%}  {mn:5.1%}  {mx:5.1%}  {bar}")

    if all_candidate_ranks:
        import statistics
        print(f"\n  Candidate rank stats (across all words):")
        print(f"    Median rank: {statistics.median(all_candidate_ranks):.0f}")
        print(f"    Mean rank:   {statistics.mean(all_candidate_ranks):.0f}")
        print(f"    P90 rank:    {sorted(all_candidate_ranks)[int(len(all_candidate_ranks)*0.9)]}")
        print(f"    P95 rank:    {sorted(all_candidate_ranks)[int(len(all_candidate_ranks)*0.95)]}")
        print(f"    P99 rank:    {sorted(all_candidate_ranks)[int(len(all_candidate_ranks)*0.99)]}")
        print(f"    Max rank:    {max(all_candidate_ranks)}")
        print(f"    Total candidates scored: {len(all_candidate_ranks)}")

    # Save results JSON
    results_path = OUTPUT_DIR / "full_vocab_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "model": LLAMA_MODEL,
            "k_values": K_VALUES,
            "avg_coverage_at_k": {
                str(k): round(sum(v) / len(v), 4) if v else 0
                for k, v in all_coverage_at_k.items()
            },
            "per_word": per_word_results,
            "candidate_rank_percentiles": {
                "p50": statistics.median(all_candidate_ranks) if all_candidate_ranks else None,
                "p90": sorted(all_candidate_ranks)[int(len(all_candidate_ranks)*0.9)] if all_candidate_ranks else None,
                "p95": sorted(all_candidate_ranks)[int(len(all_candidate_ranks)*0.95)] if all_candidate_ranks else None,
                "p99": sorted(all_candidate_ranks)[int(len(all_candidate_ranks)*0.99)] if all_candidate_ranks else None,
            },
        }, f, indent=2)
    print(f"\n  Results saved to {results_path}")
    print()


if __name__ == "__main__":
    main()
