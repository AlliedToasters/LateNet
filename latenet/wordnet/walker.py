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


@dataclass
class WalkerConfig:
    seed: int = 42
    max_depth: int = 8
    min_examples_per_domain: int = 5
    rel_types: set[RelationshipType] = field(
        default_factory=lambda: set(RelationshipType)
    )
    pos: str = "n"


class WordNetWalker:
    def __init__(self, config: WalkerConfig | None = None):
        self.config = config or WalkerConfig()
        self.rng = random.Random(self.config.seed)

    def walk(self) -> list[Relationship]:
        """Traverse WordNet and return extracted relationships."""
        synsets = list(wn.all_synsets(self.config.pos))
        synsets = [s for s in synsets if s.min_depth() <= self.config.max_depth]
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
