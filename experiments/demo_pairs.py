"""Demo: generate true/false contrastive pairs using NDIF logit-weighted false selection.

Outputs formatted pairs for spot inspection. True side from WordNet hypernymy
(no NDIF call), false side weighted by Llama 70B base logits at k=10,000.

For each source word, generates multiple pairs at different "plausibility tiers":
- top-1 logit pick (most plausible false)
- sampled from top-5 (softmax-weighted)
- sampled from top-20 (softmax-weighted)
- random baseline (no logit weighting)
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from collections import defaultdict

from transformers import AutoTokenizer
from nltk.corpus import wordnet as wn

from lmprobe.extraction import ActivationExtractor
from lmprobe.retry import retry_with_backoff

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
LOGIT_TOP_K = 10000

TEST_WORDS = [
    "dog", "cat", "car", "tree", "house", "fish", "bird", "chair",
    "river", "mountain", "flower", "horse", "knife", "piano", "snake",
    "bridge", "diamond", "airplane", "potato",
    # extras to stress-test
    "hammer", "eagle", "mushroom", "violin", "tiger", "lamp", "boat",
    "spider", "corn", "sword",
]

RNG = random.Random(42)


def _base_prompt(word: str) -> str:
    return f"True or false? A {word} is a"


def _get_synset(word: str):
    synsets = wn.synsets(word, pos="n")
    if not synsets:
        return None
    s = synsets[0]
    if not _is_physical_entity(s) or not _is_primary_sense(s):
        return None
    return s


def _get_candidates(synset):
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

    cs = _wup_filter(siblings)
    cc = _wup_filter(cousins)
    return (cs if cs else siblings[:10]) + (cc if cc else cousins[:10])


def _lemma(synset) -> str:
    return synset.lemma_names()[0].replace("_", " ")


def _softmax_sample(items: list[tuple], rng: random.Random) -> tuple:
    """Sample from (item, logit) pairs using softmax weights."""
    if not items:
        return None
    max_logit = max(l for _, l in items)
    weights = [math.exp(l - max_logit) for _, l in items]
    return rng.choices([it for it, _ in items], weights=weights, k=1)[0]


def main():
    if "NNSIGHT_API_KEY" not in os.environ and "NDIF_API_KEY" in os.environ:
        os.environ["NNSIGHT_API_KEY"] = os.environ["NDIF_API_KEY"]
    if "NNSIGHT_API_KEY" not in os.environ:
        print("ERROR: Set NDIF_API_KEY or NNSIGHT_API_KEY env var")
        sys.exit(1)

    print(f"Loading tokenizer + extractor...")
    tok = AutoTokenizer.from_pretrained(LLAMA_TOKENIZER)
    ext = ActivationExtractor(
        model_name=LLAMA_MODEL, device="cpu", layers=[],
        backend="nnsight", remote=True,
    )

    # Resolve words
    test_synsets = []
    for w in TEST_WORDS:
        s = _get_synset(w)
        if s:
            test_synsets.append((w, s))
        else:
            print(f"  SKIP {w}")
    print(f"{len(test_synsets)} words resolved\n")

    all_pairs = []

    for word, synset in test_synsets:
        candidates = _get_candidates(synset)
        if not candidates:
            continue

        # True statement
        hyps = synset.hypernyms()
        if not hyps:
            continue
        true_cat = _lemma(hyps[0])
        true_stmt = f"A {word} is a {true_cat}"

        # Tokenize candidates → first token IDs
        cand_tokens = []
        for c in candidates:
            lemma = _lemma(c)
            tids = tok.encode(f" {lemma}", add_special_tokens=False)
            cand_tokens.append((c, lemma, tids[0]))

        # Query NDIF
        prompt = _base_prompt(word)
        try:
            logits, _, logits_indices = retry_with_backoff(
                lambda p=prompt: ext.extract_logits_only(
                    [p], remote=True, logit_top_k=LOGIT_TOP_K,
                ),
                max_retries=3, base_delay=3.0, max_delay=60.0,
                context=f"demo:{word}",
            )
        except Exception as e:
            print(f"  NDIF ERROR for {word}: {e}")
            continue

        last_logits = logits[0, -1, :]
        last_indices = logits_indices[0, -1, :]

        # Build lookup
        logit_lookup = {}
        for i, vid in enumerate(last_indices.tolist()):
            logit_lookup[vid] = last_logits[i].item()

        # Score candidates
        scored = []
        for c, lemma, first_tid in cand_tokens:
            if first_tid in logit_lookup:
                scored.append((c, lemma, logit_lookup[first_tid]))
            # Skip candidates outside top-k (rare/obscure)

        if not scored:
            continue

        scored.sort(key=lambda x: x[2], reverse=True)

        # Generate pairs at different tiers
        # 1. Top-1 (most plausible false)
        top1 = scored[0]
        # 2. Softmax sample from top-5
        top5_pick = _softmax_sample([(s, l) for s, _, l in scored[:5]], RNG)
        # 3. Softmax sample from top-20
        top20_pick = _softmax_sample([(s, l) for s, _, l in scored[:20]], RNG)
        # 4. Random (no weighting)
        random_pick = RNG.choice(candidates)

        pairs = {
            "word": word,
            "true": true_stmt,
            "top1_false": f"A {word} is a {top1[1]}",
            "top1_logit": top1[2],
            "top5_false": f"A {word} is a {_lemma(top5_pick)}" if top5_pick else "N/A",
            "top20_false": f"A {word} is a {_lemma(top20_pick)}" if top20_pick else "N/A",
            "random_false": f"A {word} is a {_lemma(random_pick)}",
            "n_scored": len(scored),
            "n_candidates": len(candidates),
        }
        all_pairs.append(pairs)
        time.sleep(0.3)

    # ---------------------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DEMO PAIRS — spot-inspect for coherence")
    print("=" * 80)

    for p in all_pairs:
        print(f"\n--- {p['word'].upper()} ({p['n_scored']}/{p['n_candidates']} candidates scored) ---")
        print(f"  TRUE:      {p['true']}")
        print(f"  TOP-1:     {p['top1_false']}  [logit={p['top1_logit']:.1f}]")
        print(f"  TOP-5 SAM: {p['top5_false']}")
        print(f"  TOP-20 SAM:{p['top20_false']}")
        print(f"  RANDOM:    {p['random_false']}")

    # Compact table for quick scan
    print("\n\n" + "=" * 80)
    print("COMPACT TABLE: TRUE vs TOP-1 FALSE vs RANDOM FALSE")
    print("=" * 80)
    print(f"{'Word':<12s} {'TRUE statement':<35s} {'TOP-1 FALSE':<35s} {'RANDOM FALSE':<35s}")
    print("-" * 117)
    for p in all_pairs:
        print(f"{p['word']:<12s} {p['true']:<35s} {p['top1_false']:<35s} {p['random_false']:<35s}")

    print(f"\n{len(all_pairs)} pairs generated.")


if __name__ == "__main__":
    main()
