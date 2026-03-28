"""Difficulty tiers based on semantic distance (hard/medium/easy)."""

from __future__ import annotations

import random
from collections import defaultdict
from functools import lru_cache

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset

from latenet.types import Difficulty

MAX_SIBLING_POOL = 20


def _max_lemma_count(synset: Synset) -> int:
    """Max lemma frequency for a synset (Brown corpus)."""
    return max((lemma.count() for lemma in synset.lemmas()), default=0)


def _frequency_weighted_choice(pool: list[Synset], rng: random.Random) -> Synset:
    """Pick from pool with probability proportional to lemma frequency.

    Uses floor=1 so zero-count synsets still have a chance.
    """
    weights = [max(_max_lemma_count(s), 1) for s in pool]
    return rng.choices(pool, weights=weights, k=1)[0]


def classify_difficulty(hop_count: int | None) -> Difficulty:
    """Map hop distance to difficulty tier."""
    if hop_count is None:
        return Difficulty.EASY
    if hop_count <= 2:
        return Difficulty.HARD
    if hop_count <= 5:
        return Difficulty.MEDIUM
    return Difficulty.EASY


@lru_cache(maxsize=1)
def _synsets_by_root() -> dict[str, list[Synset]]:
    """Build a cached index of noun synsets grouped by root hypernym name."""
    index: dict[str, list[Synset]] = defaultdict(list)
    for s in wn.all_synsets("n"):
        roots = s.root_hypernyms()
        root_name = roots[0].name() if roots else "unknown"
        index[root_name].append(s)
    return dict(index)


def _siblings_of(synset: Synset) -> list[Synset]:
    """Get hyponyms of synset's parents, excluding synset itself."""
    siblings = []
    seen = set()
    for parent in synset.hypernyms():
        for child in parent.hyponyms():
            if child.name() != synset.name() and child.name() not in seen:
                seen.add(child.name())
                siblings.append(child)
    return siblings


def _cousins_of(synset: Synset) -> list[Synset]:
    """Get synsets sharing a grandparent but not a parent with synset."""
    parent_names = {p.name() for p in synset.hypernyms()}
    cousins = []
    seen = {synset.name()}
    for parent in synset.hypernyms():
        for grandparent in parent.hypernyms():
            for uncle in grandparent.hyponyms():
                if uncle.name() in parent_names:
                    continue
                for cousin in uncle.hyponyms():
                    if cousin.name() not in seen:
                        seen.add(cousin.name())
                        cousins.append(cousin)
    return cousins


def _is_category_level(synset: Synset) -> bool:
    """True if synset is a general category (has hyponyms and attested lemmas).

    Filters out leaf synsets like 'Sioux.n.01' or 'sixteen.n.01' that produce
    nonsensical false statements when used as categories in "X is a Y" templates.
    """
    return len(synset.hyponyms()) >= 2 and _max_lemma_count(synset) > 0


def _distant_synset(synset: Synset, rng: random.Random) -> Synset | None:
    """Pick a noun synset from a different root hypernym tree, weighted by frequency.

    Only considers category-level synsets (have hyponyms, attested in corpus) to
    avoid nonsensical false statements like 'Day is a Sioux'.
    """
    source_roots = {r.name() for r in synset.root_hypernyms()}
    by_root = _synsets_by_root()

    other_roots = [k for k in by_root if k not in source_roots]
    if not other_roots:
        all_synsets = [s for ss in by_root.values() for s in ss
                       if s.name() != synset.name() and _is_category_level(s)]
        return _frequency_weighted_choice(all_synsets, rng) if all_synsets else None

    root = rng.choice(other_roots)
    pool = [s for s in by_root[root] if _is_category_level(s)]
    if not pool:
        # Fallback: relax to just attested synsets
        pool = [s for s in by_root[root] if _max_lemma_count(s) > 0]
    return _frequency_weighted_choice(pool, rng) if pool else None


def pick_negation_synset(
    source: Synset,
    target: Synset,
    difficulty: Difficulty,
    rng: random.Random,
) -> Synset | None:
    """Pick a replacement synset for the false statement at the desired difficulty."""
    source_lemmas = {l.name().lower() for l in source.lemmas()}

    def _no_overlap(pool: list) -> list:
        """Exclude synsets sharing any lemma with source (avoids tautologies)."""
        return [
            s for s in pool
            if s.name() != source.name()
            and not source_lemmas & {l.name().lower() for l in s.lemmas()}
        ]

    if difficulty == Difficulty.HARD:
        pool = _no_overlap(_siblings_of(target))
    elif difficulty == Difficulty.MEDIUM:
        pool = _no_overlap(_cousins_of(target))
    elif difficulty == Difficulty.EASY:
        return _distant_synset(target, rng)
    else:
        return None

    if not pool:
        return None

    if len(pool) > MAX_SIBLING_POOL:
        pool = rng.sample(pool, MAX_SIBLING_POOL)

    return _frequency_weighted_choice(pool, rng)
