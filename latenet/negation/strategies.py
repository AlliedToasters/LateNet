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
    coherence_scorer: object | None = None,
) -> tuple[str, NegationStrategy, Synset] | None:
    """Generate false statement by swapping target with a sibling (hard)."""
    neg = pick_negation_synset(relationship.source, relationship.target, Difficulty.HARD, rng, coherence_scorer=coherence_scorer)
    if neg is None:
        return None
    stmt = _swap_target_name(relationship, template, neg)
    return stmt, NegationStrategy.SIBLING_SWAP, neg


def cousin_swap(
    relationship: Relationship,
    template: Template,
    rng: random.Random,
    coherence_scorer: object | None = None,
) -> tuple[str, NegationStrategy, Synset] | None:
    """Generate false statement by swapping target with a cousin synset (medium)."""
    neg = pick_negation_synset(relationship.source, relationship.target, Difficulty.MEDIUM, rng, coherence_scorer=coherence_scorer)
    if neg is None:
        return None
    stmt = _swap_target_name(relationship, template, neg)
    return stmt, NegationStrategy.COUSIN_SWAP, neg


def distant_swap(
    relationship: Relationship,
    template: Template,
    rng: random.Random,
    coherence_scorer: object | None = None,
) -> tuple[str, NegationStrategy, Synset] | None:
    """Generate false statement by swapping target with a distant synset (easy)."""
    neg = pick_negation_synset(relationship.source, relationship.target, Difficulty.EASY, rng, coherence_scorer=coherence_scorer)
    if neg is None:
        return None
    stmt = _swap_target_name(relationship, template, neg)
    return stmt, NegationStrategy.DISTANT_SWAP, neg


def negate_statement(statement: str) -> str:
    """Mechanically negate a statement by inserting 'not'.

    Tries a series of pattern-specific rules covering all generator domains.
    Falls back to 'It is not true that ...' if no pattern matches.
    Returns the negated string.
    """
    original = statement
    s = statement

    # --- Copula / linking verb patterns (WordNet, biology, chemistry, etc.) ---
    s = re.sub(r"\bis a\b", "is not a", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bis an\b", "is not an", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bis classified as\b", "is not classified as", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bis a type of\b", "is not a type of", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bis a kind of\b", "is not a kind of", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bis a component of\b", "is not a component of", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bis a more specific rank\b", "is not a more specific rank", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bis a more general rank\b", "is not a more general rank", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bis a higher level\b", "is not a higher level", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bis a factor of\b", "is not a factor of", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bis divisible by\b", "is not divisible by", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bis located\b", "is not located", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bis in the same\b", "is not in the same", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bis nearer to\b", "is not nearer to", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bis part of\b", "is not part of", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bis closer to\b", "is not closer to", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bis larger in area\b", "is not larger in area", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bis larger than\b", "is not larger than", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bis greater than\b", "is not greater than", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bis found in\b", "is not found in", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bis in the\b", "is not in the", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bis classified under\b", "is not classified under", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bis represented by\b", "is not represented by", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bis brighter than\b", "is not brighter than", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bis (north|south|east|west) of\b", r"is not \1 of", s, count=1)
    if s != original:
        return s

    # --- "are" patterns (siblings, contemporaries) ---
    s = re.sub(r"\bare both\b", "are not both", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bare in the same\b", "are not in the same", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bare antonyms\b", "are not antonyms", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bare coprime\b", "are not coprime", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bare classified as\b", "are not classified as", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bare a\b", "are not a", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bare an\b", "are not an", s, count=1)
    if s != original:
        return s

    # --- Verb patterns (temporal, authorship) ---
    s = re.sub(r"\bwas born before\b", "was not born before", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bwas born in\b", "was not born in", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bwas born earlier\b", "was not born earlier", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bwas composed by\b", "was not composed by", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bwas written by\b", "was not written by", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bwas directed by\b", "was not directed by", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bwas painted by\b", "was not painted by", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bwas alive\b", "was not alive", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bwas an\b", "was not an", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bwas a\b", "was not a", s, count=1)
    if s != original:
        return s

    # --- Active verb patterns ---
    s = re.sub(r"\bwrote\b", "did not write", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bcomposed\b", "did not compose", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bdirected\b", "did not direct", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bpainted\b", "did not paint", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bcreated\b", "did not create", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bworked as\b", "did not work as", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bpredates\b", "does not predate", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bpostdates\b", "does not postdate", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bpreceded\b", "did not precede", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bexceeds\b", "does not exceed", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bcovers more\b", "does not cover more", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\borbits\b", "does not orbit", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\brevolves around\b", "does not revolve around", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bcontains\b", "does not contain", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\btranslates to\b", "does not translate to", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bequals\b", "does not equal", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bdivides\b", "does not divide", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bexists as\b", "does not exist as", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bcomes from\b", "does not come from", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bappears brighter\b", "does not appear brighter", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bmeans\b", "does not mean", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bdied (before|after)\b", r"did not die \1", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\btook place\b", "did not take place", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bhappened\b", "did not happen", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\blived at\b", "did not live at", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\blived before\b", "did not live before", s, count=1)
    if s != original:
        return s

    # --- "belongs to" / "has" / "lies" / "share" patterns ---
    s = re.sub(r"\bbelongs to\b", "does not belong to", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bboth belong to\b", "do not both belong to", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bbelong to\b", "do not belong to", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\blies to the\b", "does not lie to the", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bcan be found\b", "cannot be found", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bcan be classified\b", "cannot be classified", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bshare a\b", "do not share a", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bhave the greatest\b", "do not have the greatest", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bhas a\b", "does not have a", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bhas an\b", "does not have an", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bhas\b", "does not have", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bwere contemporaries\b", "were not contemporaries", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\boccurred in\b", "did not occur in", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\boccurred before\b", "did not occur before", s, count=1)
    if s != original:
        return s

    # --- General copula catch-all (handles remaining "is"/"are" patterns) ---
    s = re.sub(r"\bis\b", "is not", s, count=1)
    if s != original:
        return s
    s = re.sub(r"\bare\b", "are not", s, count=1)
    if s != original:
        return s

    # --- Fallback ---
    return f"It is not true that {original[0].lower()}{original[1:]}"


def direct_negation(
    true_statement: str,
) -> tuple[str, NegationStrategy, None]:
    """Insert 'not' into the true statement to make it false."""
    stmt = negate_statement(true_statement)
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
    coherence_scorer: object | None = None,
) -> list[tuple[str, NegationStrategy, Synset | None]]:
    """Apply all applicable negation strategies. Returns list of (statement, strategy, neg_synset)."""
    results = []

    if template.rel_type in (RelationshipType.HYPERNYMY, RelationshipType.MERONYMY):
        sib = sibling_swap(relationship, template, rng, coherence_scorer=coherence_scorer)
        if sib:
            results.append(sib)

        cuz = cousin_swap(relationship, template, rng, coherence_scorer=coherence_scorer)
        if cuz:
            results.append(cuz)

        dist = distant_swap(relationship, template, rng, coherence_scorer=coherence_scorer)
        if dist:
            results.append(dist)

        rev = reverse_relation(relationship, template)
        if rev:
            results.append(rev)

    if template.supports_negation:
        results.append(direct_negation(true_statement))

    return results
