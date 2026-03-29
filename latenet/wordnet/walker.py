"""WordNet hierarchy traversal."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field

from nltk.corpus import wordnet as wn

from latenet.types import Relationship, RelationshipType
from latenet.wordnet.relationships import extract_all


def _domain_for_synset(synset, target_depth: int = 2) -> str:
    """Get the domain for a synset by finding its ancestor at target_depth.

    Walks up the hypernym chain and picks the ancestor at the specified depth
    in the WordNet hierarchy. Falls back to the synset itself if it's already
    at or above target_depth.
    """
    # Walk up to find the ancestor at target_depth
    current = synset
    path = [current]
    while current.hypernyms():
        current = current.hypernyms()[0]
        path.append(current)

    # path goes from synset → root. Reverse to go root → synset.
    path.reverse()

    # Pick the node at target_depth (or the deepest available if short)
    if len(path) > target_depth:
        ancestor = path[target_depth]
    else:
        ancestor = path[-1]

    return ancestor.lemma_names()[0].replace("_", " ")


def _max_lemma_count(synset) -> int:
    """Return the maximum lemma frequency count for any lemma in a synset.

    Uses the Brown corpus counts bundled with NLTK's WordNet.
    """
    return max((lemma.count() for lemma in synset.lemmas()), default=0)


_CONCRETE_ROOTS: frozenset[str] | None = None


def _concrete_synset_names() -> frozenset[str]:
    """Return names of all synsets under physical_entity.n.01.

    Restricting to this subtree avoids abstract nouns (time, group, relation)
    whose hypernym chains produce awkward IS-A statements.
    """
    global _CONCRETE_ROOTS
    if _CONCRETE_ROOTS is not None:
        return _CONCRETE_ROOTS

    root = wn.synset("physical_entity.n.01")
    names: set[str] = set()
    queue = [root]
    while queue:
        s = queue.pop()
        if s.name() in names:
            continue
        names.add(s.name())
        queue.extend(s.hyponyms())
    _CONCRETE_ROOTS = frozenset(names)
    return _CONCRETE_ROOTS


@dataclass
class WalkerConfig:
    seed: int = 42
    max_depth: int = 8
    min_examples_per_domain: int = 5
    rel_types: set[RelationshipType] = field(
        default_factory=lambda: set(RelationshipType)
    )
    pos: str = "n"
    frequency_weighted: bool = True
    concrete_only: bool = True


class WordNetWalker:
    def __init__(self, config: WalkerConfig | None = None):
        self.config = config or WalkerConfig()
        self.rng = random.Random(self.config.seed)

    def walk(self) -> list[Relationship]:
        """Traverse WordNet and return extracted relationships.

        When frequency_weighted is True, synsets are traversed in an order
        biased toward high-frequency lemmas (Brown corpus counts), so that
        well-known concepts are explored first and dominate generation when
        max_pairs is set.
        """
        synsets = list(wn.all_synsets(self.config.pos))
        synsets = [s for s in synsets if s.min_depth() <= self.config.max_depth]

        if self.config.concrete_only:
            concrete = _concrete_synset_names()
            synsets = [s for s in synsets if s.name() in concrete]

        if self.config.frequency_weighted:
            # Sort synsets by lemma frequency (descending) with random
            # tiebreaking within each frequency bucket. This ensures
            # high-frequency synsets are explored first, so when max_pairs
            # truncates generation the output is biased toward well-known
            # concepts. Zero-count synsets are shuffled at the tail.
            self.rng.shuffle(synsets)  # random tiebreaking
            synsets.sort(key=lambda s: _max_lemma_count(s), reverse=True)
        else:
            self.rng.shuffle(synsets)

        all_rels: list[Relationship] = []
        for synset in synsets:
            domain = _domain_for_synset(synset)
            rels = extract_all(synset, self.config.rel_types)
            for rel in rels:
                rel.domain = domain
            all_rels.extend(rels)

        if self.config.min_examples_per_domain > 0:
            all_rels = self._filter_by_domain(all_rels)

        return all_rels

    def _filter_by_domain(self, rels: list[Relationship]) -> list[Relationship]:
        """Drop relationships from domains with too few examples."""
        domain_counts: dict[str, int] = defaultdict(int)
        for rel in rels:
            domain_counts[rel.domain] += 1
        min_count = self.config.min_examples_per_domain
        return [r for r in rels if domain_counts[r.domain] >= min_count]
