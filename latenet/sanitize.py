"""Shared statement sanitizer and article resolution.

Layer 0: Basic text normalization (whitespace, punctuation, capitalization).
Layer 1: Article resolution via inflect (a/an agreement).

Layer 0 is applied automatically in ContrastivePair.__post_init__.
Layer 1 is applied at template render time via render_template().
"""

from __future__ import annotations

import re

import inflect

_engine = inflect.engine()
_article_cache: dict[str, str] = {}


def resolve_article(entity: str) -> str:
    """Return 'a {entity}' or 'an {entity}' with correct article.

    Uses inflect for reliable a/an resolution. Handles edge cases
    like silent-h words, consonant-sound vowel words, etc.
    """
    if not entity:
        return entity
    if entity in _article_cache:
        return _article_cache[entity]
    result = _engine.a(entity)
    _article_cache[entity] = result
    return result


def sanitize_statement(text: str) -> str:
    """Normalize a generated statement for grammar and formatting.

    Applied automatically to all ContrastivePair statements. Fixes:
    1. Whitespace: collapse runs of spaces, strip leading/trailing.
    2. Trailing period: add if missing, collapse '..' to '.'.
    3. Capitalization: uppercase first character.
    """
    if not text:
        return text

    # 1. Whitespace normalization.
    text = re.sub(r" {2,}", " ", text).strip()

    # 2. Trailing period.
    if text.endswith("..") and not text.endswith("..."):
        text = text[:-1]
    if not text.endswith("."):
        text += "."

    # 3. Capitalization.
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    return text


_ARTICLE_SLOT_RE = re.compile(r"\{a_(\w+)\}")


def render_template(pattern: str, **kwargs: str) -> str:
    """Render a template pattern with article-aware slot resolution.

    Slots named ``{a_X}`` are resolved by looking up the value for ``X``
    in kwargs and calling ``resolve_article(value)``, producing the
    correct 'a'/'an' + value string.

    Regular ``{X}`` slots are filled via standard str.format().

    Example::

        >>> render_template("{entity} is {a_category}.", entity="cat", category="animal")
        'cat is an animal.'
    """
    expanded = dict(kwargs)
    for match in _ARTICLE_SLOT_RE.finditer(pattern):
        base_slot = match.group(1)
        if base_slot in kwargs:
            expanded[f"a_{base_slot}"] = resolve_article(kwargs[base_slot])
    return pattern.format(**expanded)
