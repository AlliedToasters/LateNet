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


_lemma_count_cache: dict[str, int] = {}


def _max_lemma_count(synset) -> int:
    """Return the maximum lemma frequency count for any lemma in a synset."""
    key = synset.name()
    if key not in _lemma_count_cache:
        _lemma_count_cache[key] = max((lemma.count() for lemma in synset.lemmas()), default=0)
    return _lemma_count_cache[key]


def _synset_is_attested(synset) -> bool:
    """True if any lemma in the synset has a non-zero frequency count."""
    return _max_lemma_count(synset) > 0


def _is_primary_sense(synset) -> bool:
    """True if synset is the most common noun sense for its primary lemma.

    Filters out obscure senses like time.n.05 (a clock reading) that produce
    awkward statements because readers interpret the lemma as its primary sense.
    """
    from nltk.corpus import wordnet as wn
    lemma_name = synset.lemmas()[0].name()
    all_noun_senses = wn.synsets(lemma_name, pos="n")
    if not all_noun_senses:
        return True
    return all_noun_senses[0].name() == synset.name()


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
        min_lemma_frequency: int = 0,
        max_pairs_per_source: int = 3,
    ):
        super().__init__(seed=seed, max_pairs=max_pairs)
        self.max_depth = max_depth
        self.min_examples_per_domain = min_examples_per_domain
        # Exclude meronymy — noisy data, 35% contest rate, low signal
        if rel_types is None:
            self.rel_types = {RelationshipType.HYPERNYMY, RelationshipType.ANTONYMY, RelationshipType.SIBLING}
        else:
            self.rel_types = rel_types - {RelationshipType.MERONYMY}
        self.max_false_per_true = max_false_per_true
        self.min_lemma_frequency = min_lemma_frequency
        self.max_pairs_per_source = max_pairs_per_source

    @property
    def name(self) -> str:
        return "wordnet"

    def relation_types(self) -> list[str]:
        return ["hypernymy", "antonymy", "sibling"]

    def domains(self) -> list[str]:
        # Domains are discovered dynamically during generation
        return []

    def _should_skip_synset(self, synset, difficulty: str) -> bool:
        """Check if a synset should be skipped based on frequency filtering.

        Soft filter: synsets with all zero-count lemmas are only usable for
        easy tier (where the model is more likely to get it right even with
        weak representations).

        Hard filter: if min_lemma_frequency > 0, synsets below that threshold
        are always skipped.
        """
        max_count = _max_lemma_count(synset)
        # Hard cutoff
        if self.min_lemma_frequency > 0 and max_count < self.min_lemma_frequency:
            return True
        # Soft filter: unattested synsets only allowed for easy tier
        if max_count == 0 and difficulty != Difficulty.EASY.value:
            return True
        return False

    def _pairs_from_relationship(self, rel: Relationship) -> Iterator[ContrastivePair]:
        """Yield all contrastive pairs for a single relationship."""
        # Skip non-primary senses — they produce awkward statements because
        # readers interpret the lemma as its most common meaning
        if not _is_primary_sense(rel.source) or not _is_primary_sense(rel.target):
            return

        # Pre-check source and target synset attestation for logging
        if not _synset_is_attested(rel.source) and not _synset_is_attested(rel.target):
            logger.debug(
                "Unattested synsets: %s, %s — restricting to easy tier",
                rel.source.name(), rel.target.name(),
            )

        templates = get_templates(rel.rel_type)
        for template in templates:
            slots, synset_map = slot_values_for_relationship(template, rel)
            true_statement = render(template, slots, synset_map=synset_map)

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

                difficulty = _difficulty_for_distance(dist)

                # Frequency-based soft filter: skip unattested synsets
                # for non-easy tiers
                if self._should_skip_synset(rel.source, difficulty):
                    continue
                if self._should_skip_synset(rel.target, difficulty):
                    continue
                if neg_synset is not None and self._should_skip_synset(neg_synset, difficulty):
                    continue

                pair_id = _make_pair_id(
                    rel.source.name(), rel.target.name(),
                    template.id, strategy.value, i,
                )

                yield ContrastivePair(
                    true_statement=true_statement,
                    false_statement=false_stmt,
                    pair_id=pair_id,
                    domain="wordnet",
                    relation_type=rel.rel_type.value,
                    difficulty="mixed",
                    semantic_distance=dist,
                    generator=self.name,
                    template_id=template.id,
                    negation_strategy=strategy.value,
                    source_synset=rel.source.name(),
                    target_synset=rel.target.name(),
                    neg_synset=neg_synset.name() if neg_synset else None,
                    gen_params={
                        "source_synset": rel.source.name(),
                        "target_synset": rel.target.name(),
                        "neg_synset": neg_synset.name() if neg_synset else None,
                        "semantic_distance": dist,
                        "source_name": rel.source_name,
                        "target_name": rel.target_name,
                        "swap_distance": difficulty,
                    },
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

        # Per-source-synset cap to prevent a single holonym/hypernym from
        # dominating the output (e.g., "outer space" producing 45% of meronymy)
        source_counts: dict[str, int] = defaultdict(int)

        # Round-robin across relationship types so max_pairs doesn't starve later types
        while active:
            next_active = []
            for idx in active:
                try:
                    pair = next(iterators[idx])
                except StopIteration:
                    continue
                # Enforce per-source diversity
                src = pair.source_synset or ""
                if source_counts[src] >= self.max_pairs_per_source:
                    next_active.append(idx)
                    continue
                source_counts[src] += 1
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
    if dist <= 2:
        return Difficulty.HARD.value
    if dist <= 5:
        return Difficulty.MEDIUM.value
    return Difficulty.EASY.value
