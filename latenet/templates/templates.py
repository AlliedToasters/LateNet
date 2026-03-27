"""Statement templates per relationship type."""

from __future__ import annotations

from dataclasses import dataclass

from latenet.sanitize import render_template as _render_template
from latenet.types import RelationshipType


@dataclass(frozen=True)
class Template:
    id: str
    rel_type: RelationshipType
    pattern: str
    slots: tuple[str, ...]
    supports_negation: bool = True


# --- Hypernymy templates ---
_HYPERNYMY_TEMPLATES = [
    Template("hyp_01", RelationshipType.HYPERNYMY, "{entity} is {a_category}", ("entity", "category")),
    Template("hyp_02", RelationshipType.HYPERNYMY, "{a_entity} is a type of {category}", ("entity", "category")),
    Template("hyp_03", RelationshipType.HYPERNYMY, "{entity} belongs to the category of {category}", ("entity", "category")),
    Template("hyp_04", RelationshipType.HYPERNYMY, "{entity} is a kind of {category}", ("entity", "category")),
    Template("hyp_05", RelationshipType.HYPERNYMY, "{a_entity} is an example of {a_category}", ("entity", "category")),
]

# --- Meronymy templates ---
_MERONYMY_TEMPLATES = [
    Template("mer_01", RelationshipType.MERONYMY, "{part} is part of {whole}", ("part", "whole")),
    Template("mer_02", RelationshipType.MERONYMY, "{a_whole} has {a_part}", ("whole", "part")),
    Template("mer_03", RelationshipType.MERONYMY, "{part} is a component of {whole}", ("part", "whole")),
    Template("mer_04", RelationshipType.MERONYMY, "One of the parts of {a_whole} is {a_part}", ("whole", "part")),
]

# --- Sibling templates ---
_SIBLING_TEMPLATES = [
    Template(
        "sib_01", RelationshipType.SIBLING,
        "{entity_a} and {entity_b} are both types of {parent}",
        ("entity_a", "entity_b", "parent"),
        supports_negation=False,
    ),
    Template(
        "sib_02", RelationshipType.SIBLING,
        "Like {entity_a}, {entity_b} is {a_parent}",
        ("entity_a", "entity_b", "parent"),
        supports_negation=False,
    ),
]

# --- Antonymy templates ---
_ANTONYMY_TEMPLATES = [
    Template(
        "ant_01", RelationshipType.ANTONYMY,
        "{word_a} is the opposite of {word_b}",
        ("word_a", "word_b"),
        supports_negation=False,
    ),
    Template(
        "ant_02", RelationshipType.ANTONYMY,
        "{word_a} and {word_b} are antonyms",
        ("word_a", "word_b"),
        supports_negation=False,
    ),
]

TEMPLATES: dict[RelationshipType, list[Template]] = {
    RelationshipType.HYPERNYMY: _HYPERNYMY_TEMPLATES,
    RelationshipType.MERONYMY: _MERONYMY_TEMPLATES,
    RelationshipType.SIBLING: _SIBLING_TEMPLATES,
    RelationshipType.ANTONYMY: _ANTONYMY_TEMPLATES,
}


def get_templates(rel_type: RelationshipType) -> list[Template]:
    """Return all templates for a given relationship type."""
    return TEMPLATES.get(rel_type, [])


def render(
    template: Template,
    slot_values: dict[str, str],
    synset_map: dict[str, str] | None = None,
) -> str:
    """Fill template slots with values, resolving {a_X} article tokens.

    If *synset_map* is provided, it maps slot names to WordNet synset names
    so that uncountable nouns can skip the indefinite article.
    """
    return _render_template(template.pattern, synset_map=synset_map, **slot_values)


def slot_values_for_relationship(
    template: Template, relationship,
) -> tuple[dict[str, str], dict[str, str]]:
    """Build slot values and synset map from a Relationship.

    Returns ``(slot_values, synset_map)`` where *synset_map* maps slot
    names to WordNet synset name strings (e.g. ``"dog.n.01"``).  The
    synset map is used by :func:`render` to skip indefinite articles for
    uncountable nouns.
    """
    from latenet.types import Relationship

    rel: Relationship = relationship
    if template.rel_type == RelationshipType.HYPERNYMY:
        slots = {"entity": rel.source_name, "category": rel.target_name}
        smap = {"entity": rel.source.name(), "category": rel.target.name()}
        return slots, smap
    elif template.rel_type == RelationshipType.MERONYMY:
        slots = {"whole": rel.source_name, "part": rel.target_name}
        smap = {"whole": rel.source.name(), "part": rel.target.name()}
        return slots, smap
    elif template.rel_type == RelationshipType.SIBLING:
        parents = rel.source.hypernyms()
        parent_synset = parents[0] if parents else None
        parent_name = parent_synset.lemma_names()[0].replace("_", " ") if parent_synset else "thing"
        slots = {"entity_a": rel.source_name, "entity_b": rel.target_name, "parent": parent_name}
        smap = {
            "entity_a": rel.source.name(),
            "entity_b": rel.target.name(),
            "parent": parent_synset.name() if parent_synset else "",
        }
        return slots, smap
    elif template.rel_type == RelationshipType.ANTONYMY:
        slots = {"word_a": rel.source_name, "word_b": rel.target_name}
        smap = {"word_a": rel.source.name(), "word_b": rel.target.name()}
        return slots, smap
    else:
        slots = {"entity": rel.source_name, "category": rel.target_name}
        smap = {"entity": rel.source.name(), "category": rel.target.name()}
        return slots, smap
