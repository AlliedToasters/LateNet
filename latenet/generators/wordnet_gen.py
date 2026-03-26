"""Contrastive pair generation from WordNet structure."""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from collections.abc import Iterator

from latenet.types import ContrastivePair, Difficulty, Relationship, RelationshipType

from latenet.generators.base import BaseGenerator
from latenet.wordnet.walker import WalkerConfig, WordNetWalker
from latenet.wordnet.distance import semantic_distance
from latenet.templates.templates import get_templates, render, slot_values_for_relationship
from latenet.negation.strategies import apply_negation

logger = logging.getLogger(__name__)


class WordNetGenerator(BaseGenerator):
    """Generate contrastive pairs from WordNet's noun hierarchy."""

    def __init__(
        self,
        seed: int = 42,
        max_pairs: int | None = None,
        max_depth: int = 8,
        min_examples_per_domain: int = 5,
        rel_types: set[RelationshipType] | None = None,
        max_false_per_true: int = 3,
    ):
        super().__init__(seed=seed, max_pairs=max_pairs)
        self.max_depth = max_depth
        self.min_examples_per_domain = min_examples_per_domain
        self.rel_types = rel_types
        self.max_false_per_true = max_false_per_true

    @property
    def name(self) -> str:
        return "wordnet"

    def relation_types(self) -> list[str]:
        return ["hypernymy", "meronymy", "antonymy", "sibling"]

    def domains(self) -> list[str]:
        # Domains are discovered dynamically during generation
        return []

    def _pairs_from_relationship(self, rel: Relationship) -> Iterator[ContrastivePair]:
        """Yield all contrastive pairs for a single relationship."""
        templates = get_templates(rel.rel_type)
        for template in templates:
            slots = slot_values_for_relationship(template, rel)
            true_statement = render(template, slots)

            negations = apply_negation(rel, template, true_statement, self.rng)
            if len(negations) > self.max_false_per_true:
                negations = negations[: self.max_false_per_true]

            if not negations:
                continue

            for i, (false_stmt, strategy, neg_synset) in enumerate(negations):
                if neg_synset is not None:
                    dist = semantic_distance(rel.target, neg_synset)
                else:
                    dist = 0

                pair_id = _make_pair_id(
                    rel.source.name(), rel.target.name(),
                    template.id, strategy.value, i,
                )

                yield ContrastivePair(
                    true_statement=true_statement,
                    false_statement=false_stmt,
                    pair_id=pair_id,
                    domain=rel.domain,
                    relation_type=rel.rel_type.value,
                    difficulty=_difficulty_for_distance(dist),
                    semantic_distance=dist,
                    generator=self.name,
                    template_id=template.id,
                    negation_strategy=strategy.value,
                    source_synset=rel.source.name(),
                    target_synset=rel.target.name(),
                    neg_synset=neg_synset.name() if neg_synset else None,
                )

    def generate(self) -> Iterator[ContrastivePair]:
        """Walk WordNet and yield contrastive pairs."""
        walker_config = WalkerConfig(
            seed=self.seed,
            max_depth=self.max_depth,
            min_examples_per_domain=self.min_examples_per_domain,
            rel_types=self.rel_types or set(RelationshipType),
        )
        walker = WordNetWalker(walker_config)
        relationships = walker.walk()

        # Group relationships by type for round-robin generation
        by_type: dict[RelationshipType, list[Relationship]] = defaultdict(list)
        for rel in relationships:
            by_type[rel.rel_type].append(rel)

        # Create an iterator of pairs per relationship type
        def _iter_for_type(rels: list[Relationship]) -> Iterator[ContrastivePair]:
            for rel in rels:
                yield from self._pairs_from_relationship(rel)

        rel_types = list(by_type.keys())
        iterators = [_iter_for_type(by_type[rt]) for rt in rel_types]
        counts_by_type: dict[str, int] = {rt.value: 0 for rt in rel_types}
        active = list(range(len(iterators)))
        count = 0

        # Round-robin across relationship types so max_pairs doesn't starve later types
        while active:
            next_active = []
            for idx in active:
                try:
                    pair = next(iterators[idx])
                except StopIteration:
                    continue
                yield pair
                count += 1
                counts_by_type[rel_types[idx].value] += 1
                next_active.append(idx)
                if self.max_pairs is not None and count >= self.max_pairs:
                    logger.info("WordNet generator pair counts: %s", counts_by_type)
                    return
            active = next_active

        logger.info(
            "WordNet generator produced %d total pairs. Per type: %s",
            count, counts_by_type,
        )


def _make_pair_id(
    source_name: str, target_name: str, template_id: str,
    strategy: str, index: int,
) -> str:
    """Deterministic pair ID."""
    key = f"{source_name}:{target_name}:{template_id}:{strategy}:{index}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _difficulty_for_distance(dist: int) -> str:
    if dist == 0:
        return ""
    if dist <= 2:
        return Difficulty.HARD.value
    if dist <= 5:
        return Difficulty.MEDIUM.value
    return Difficulty.EASY.value
