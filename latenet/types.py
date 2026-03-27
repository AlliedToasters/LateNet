"""Shared types for LateNet."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum

from nltk.corpus.reader.wordnet import Synset

from latenet.sanitize import sanitize_statement


class RelationshipType(Enum):
    HYPERNYMY = "hypernymy"
    MERONYMY = "meronymy"
    ANTONYMY = "antonymy"
    SIBLING = "sibling"


class Difficulty(Enum):
    HARD = "hard"
    MEDIUM = "medium"
    EASY = "easy"


class NegationStrategy(Enum):
    SIBLING_SWAP = "sibling_swap"
    DISTANT_SWAP = "distant_swap"
    DIRECT_NEGATION = "direct_negation"
    REVERSE_RELATION = "reverse_relation"


@dataclass
class Relationship:
    """A single extracted WordNet relationship."""

    source: Synset
    target: Synset
    rel_type: RelationshipType
    source_name: str
    target_name: str
    domain: str = ""


@dataclass
class ContrastivePair:
    """Universal output type for all generators."""

    true_statement: str
    false_statement: str
    pair_id: str
    domain: str
    relation_type: str
    difficulty: str
    semantic_distance: int | None
    generator: str
    template_id: str
    negation_strategy: str
    source_synset: str | None = None
    target_synset: str | None = None
    neg_synset: str | None = None
    tier: int = 1
    gen_params: dict | None = None

    def __post_init__(self) -> None:
        self.true_statement = sanitize_statement(self.true_statement)
        self.false_statement = sanitize_statement(self.false_statement)

    def to_rows(self) -> list[dict]:
        """Expand into true + false row dicts for DataFrame export."""
        base = {
            "pair_id": self.pair_id,
            "domain": self.domain,
            "relation_type": self.relation_type,
            "difficulty": self.difficulty,
            "tier": self.tier,
            "semantic_distance": self.semantic_distance,
            "source_synset": self.source_synset,
            "target_synset": self.target_synset,
            "neg_synset": self.neg_synset,
            "generator": self.generator,
            "template_id": self.template_id,
            "negation_strategy": self.negation_strategy,
            "gen_params": json.dumps(self.gen_params) if self.gen_params is not None else None,
        }
        true_row = {
            "id": f"{self.pair_id}_true",
            "statement": self.true_statement,
            "label": True,
            **base,
        }
        false_row = {
            "id": f"{self.pair_id}_false",
            "statement": self.false_statement,
            "label": False,
            **base,
        }
        return [true_row, false_row]
