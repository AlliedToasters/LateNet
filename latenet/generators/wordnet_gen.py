"""Contrastive pair generation from WordNet structure."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator

from latenet.types import ContrastivePair, Difficulty, RelationshipType
from latenet.generators.base import BaseGenerator
from latenet.wordnet.walker import WalkerConfig, WordNetWalker
from latenet.wordnet.distance import semantic_distance
from latenet.templates.templates import get_templates, render, slot_values_for_relationship
from latenet.negation.strategies import apply_negation


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

        count = 0
        for rel in relationships:
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

                    pair = ContrastivePair(
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
                    yield pair

                    count += 1
                    if self.max_pairs is not None and count >= self.max_pairs:
                        return


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
