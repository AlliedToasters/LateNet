"""Author-work attribution.

Data sources: structured literary/scientific databases.
Relations: written-by, proposed-by.
Domains: literature, music, science, philosophy.
"""

from __future__ import annotations

from collections.abc import Iterator

from latenet.generators.base import BaseGenerator
from latenet.types import ContrastivePair


class AuthorshipGenerator(BaseGenerator):
    """Generate contrastive pairs from author-work attribution data."""

    @property
    def name(self) -> str:
        return "authorship"

    def relation_types(self) -> list[str]:
        return ["written-by", "proposed-by"]

    def domains(self) -> list[str]:
        return ["literature", "music", "science", "philosophy"]

    def generate(self) -> Iterator[ContrastivePair]:
        raise NotImplementedError("Authorship generator not yet implemented")
