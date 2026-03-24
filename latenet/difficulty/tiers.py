"""Difficulty tiers based on semantic distance (hard/medium/easy)."""

from __future__ import annotations

import random
from collections import defaultdict
from functools import lru_cache

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset

from latenet.types import Difficulty

MAX_SIBLING_POOL = 20


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


def _distant_synset(synset: Synset, rng: random.Random) -> Synset | None:
    """Pick a random noun synset from a different root hypernym tree."""
    source_roots = {r.name() for r in synset.root_hypernyms()}
    by_root = _synsets_by_root()

    other_roots = [k for k in by_root if k not in source_roots]
    if not other_roots:
        all_synsets = [s for ss in by_root.values() for s in ss if s.name() != synset.name()]
        return rng.choice(all_synsets) if all_synsets else None

    root = rng.choice(other_roots)
    return rng.choice(by_root[root])


def pick_negation_synset(
    source: Synset,
    target: Synset,
    difficulty: Difficulty,
    rng: random.Random,
) -> Synset | None:
    """Pick a replacement synset for the false statement at the desired difficulty."""
    if difficulty == Difficulty.HARD:
        pool = _siblings_of(target)
        pool = [s for s in pool if s.name() != source.name()]
    elif difficulty == Difficulty.MEDIUM:
        pool = _cousins_of(target)
        pool = [s for s in pool if s.name() != source.name()]
    elif difficulty == Difficulty.EASY:
        return _distant_synset(target, rng)
    else:
        return None

    if not pool:
        return None

    if len(pool) > MAX_SIBLING_POOL:
        pool = rng.sample(pool, MAX_SIBLING_POOL)

    return rng.choice(pool)
