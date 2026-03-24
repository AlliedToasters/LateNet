"""Translation equivalence across language pairs.

Data sources: translation dictionaries.
Relations: translates-to.
Domains: spanish, french, chinese, german (extensible).
"""

from __future__ import annotations

from collections.abc import Iterator

from latenet.generators.base import BaseGenerator
from latenet.types import ContrastivePair


class LanguageGenerator(BaseGenerator):
    """Generate contrastive pairs from translation equivalences."""

    @property
    def name(self) -> str:
        return "language"

    def relation_types(self) -> list[str]:
        return ["translates-to"]

    def domains(self) -> list[str]:
        return ["spanish", "french", "chinese", "german"]

    def generate(self) -> Iterator[ContrastivePair]:
        raise NotImplementedError("Language generator not yet implemented")
