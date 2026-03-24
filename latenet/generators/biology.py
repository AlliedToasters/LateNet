"""Taxonomic facts from sources beyond WordNet.

Data sources: NCBI taxonomy, biological databases.
Relations: is-a (taxonomic), part-of (anatomical).
Domains: species, anatomy, ecology.
"""

from __future__ import annotations

from collections.abc import Iterator

from latenet.generators.base import BaseGenerator
from latenet.types import ContrastivePair


class BiologyGenerator(BaseGenerator):
    """Generate contrastive pairs from biological taxonomy data."""

    @property
    def name(self) -> str:
        return "biology"

    def relation_types(self) -> list[str]:
        return ["is-a", "part-of"]

    def domains(self) -> list[str]:
        return ["species", "anatomy", "ecology"]

    def generate(self) -> Iterator[ContrastivePair]:
        raise NotImplementedError("Biology generator not yet implemented")
