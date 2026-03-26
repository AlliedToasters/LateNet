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


def render(template: Template, slot_values: dict[str, str]) -> str:
    """Fill template slots with values, resolving {a_X} article tokens."""
    return _render_template(template.pattern, **slot_values)


def slot_values_for_relationship(template: Template, relationship) -> dict[str, str]:
    """Build slot values dict from a Relationship, based on the template's rel_type."""
    from latenet.types import Relationship

    rel: Relationship = relationship
    if template.rel_type == RelationshipType.HYPERNYMY:
        return {"entity": rel.source_name, "category": rel.target_name}
    elif template.rel_type == RelationshipType.MERONYMY:
        return {"whole": rel.source_name, "part": rel.target_name}
    elif template.rel_type == RelationshipType.SIBLING:
        parents = rel.source.hypernyms()
        parent_name = parents[0].lemma_names()[0].replace("_", " ") if parents else "thing"
        return {"entity_a": rel.source_name, "entity_b": rel.target_name, "parent": parent_name}
    elif template.rel_type == RelationshipType.ANTONYMY:
        return {"word_a": rel.source_name, "word_b": rel.target_name}
    else:
        return {"entity": rel.source_name, "category": rel.target_name}
