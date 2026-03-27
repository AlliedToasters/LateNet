"""Strategies for generating false statements (sibling swap, distant swap, direct negation)."""

from __future__ import annotations

import random
import re

from nltk.corpus.reader.wordnet import Synset

from latenet.types import Difficulty, NegationStrategy, Relationship, RelationshipType
from latenet.templates.templates import Template, render, slot_values_for_relationship
from latenet.difficulty.tiers import pick_negation_synset


def _swap_target_name(
    relationship: Relationship,
    template: Template,
    neg_synset: Synset,
) -> str:
    """Render a statement with the target replaced by neg_synset's lemma name."""
    neg_name = neg_synset.lemma_names()[0].replace("_", " ")
    slots, synset_map = slot_values_for_relationship(template, relationship)
    if template.rel_type == RelationshipType.HYPERNYMY:
        slots["category"] = neg_name
        synset_map["category"] = neg_synset.name()
    elif template.rel_type == RelationshipType.MERONYMY:
        slots["part"] = neg_name
        synset_map["part"] = neg_synset.name()
    elif template.rel_type == RelationshipType.SIBLING:
        slots["entity_b"] = neg_name
        synset_map["entity_b"] = neg_synset.name()
    elif template.rel_type == RelationshipType.ANTONYMY:
        slots["word_b"] = neg_name
        synset_map["word_b"] = neg_synset.name()
    return render(template, slots, synset_map=synset_map)


def sibling_swap(
    relationship: Relationship,
    template: Template,
    rng: random.Random,
) -> tuple[str, NegationStrategy, Synset] | None:
    """Generate false statement by swapping target with a sibling (hard)."""
    neg = pick_negation_synset(relationship.source, relationship.target, Difficulty.HARD, rng)
    if neg is None:
        return None
    stmt = _swap_target_name(relationship, template, neg)
    return stmt, NegationStrategy.SIBLING_SWAP, neg


def distant_swap(
    relationship: Relationship,
    template: Template,
    rng: random.Random,
) -> tuple[str, NegationStrategy, Synset] | None:
    """Generate false statement by swapping target with a distant synset (easy)."""
    neg = pick_negation_synset(relationship.source, relationship.target, Difficulty.EASY, rng)
    if neg is None:
        return None
    stmt = _swap_target_name(relationship, template, neg)
    return stmt, NegationStrategy.DISTANT_SWAP, neg


def direct_negation(
    true_statement: str,
) -> tuple[str, NegationStrategy, None]:
    """Insert 'not' into the true statement to make it false."""
    stmt = true_statement
    stmt = re.sub(r"\bis a\b", "is not a", stmt, count=1)
    stmt = re.sub(r"\bis an\b", "is not an", stmt, count=1)
    stmt = re.sub(r"\bhas a\b", "does not have a", stmt, count=1)
    stmt = re.sub(r"\bhas an\b", "does not have an", stmt, count=1)
    stmt = re.sub(r"\bare both\b", "are not both", stmt, count=1)
    stmt = re.sub(r"\bare antonyms\b", "are not antonyms", stmt, count=1)
    stmt = re.sub(r"\bbelongs to\b", "does not belong to", stmt, count=1)

    # Bare "has" without article (uncountable nouns: "Outer space has interstellar space")
    if stmt == true_statement:
        stmt = re.sub(r"\bhas\b", "does not have", stmt, count=1)

    if stmt == true_statement:
        stmt = f"It is not true that {true_statement[0].lower()}{true_statement[1:]}"

    return stmt, NegationStrategy.DIRECT_NEGATION, None


def reverse_relation(
    relationship: Relationship,
    template: Template,
) -> tuple[str, NegationStrategy, None] | None:
    """Swap subject/object to produce a false statement. Only for asymmetric relations."""
    if template.rel_type not in (RelationshipType.HYPERNYMY, RelationshipType.MERONYMY):
        return None

    slots, synset_map = slot_values_for_relationship(template, relationship)

    if template.rel_type == RelationshipType.HYPERNYMY:
        slots["entity"], slots["category"] = slots["category"], slots["entity"]
        synset_map["entity"], synset_map["category"] = synset_map["category"], synset_map["entity"]
    elif template.rel_type == RelationshipType.MERONYMY:
        slots["whole"], slots["part"] = slots["part"], slots["whole"]
        synset_map["whole"], synset_map["part"] = synset_map["part"], synset_map["whole"]

    stmt = render(template, slots, synset_map=synset_map)
    return stmt, NegationStrategy.REVERSE_RELATION, None


def apply_negation(
    relationship: Relationship,
    template: Template,
    true_statement: str,
    rng: random.Random,
) -> list[tuple[str, NegationStrategy, Synset | None]]:
    """Apply all applicable negation strategies. Returns list of (statement, strategy, neg_synset)."""
    results = []

    if template.rel_type in (RelationshipType.HYPERNYMY, RelationshipType.MERONYMY):
        sib = sibling_swap(relationship, template, rng)
        if sib:
            results.append(sib)

        dist = distant_swap(relationship, template, rng)
        if dist:
            results.append(dist)

        rev = reverse_relation(relationship, template)
        if rev:
            results.append(rev)

    if template.supports_negation:
        results.append(direct_negation(true_statement))

    return results
