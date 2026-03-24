"""Element properties, symbols, periodic table relationships.

Data sources: periodic table data, PubChem.
Relations: symbol-of, property-of, group-membership.
Domains: elements, compounds, states of matter.
"""

from __future__ import annotations

from collections.abc import Iterator

from latenet.generators.base import BaseGenerator
from latenet.types import ContrastivePair


class ChemistryGenerator(BaseGenerator):
    """Generate contrastive pairs from chemistry data."""

    @property
    def name(self) -> str:
        return "chemistry"

    def relation_types(self) -> list[str]:
        return ["symbol-of", "property-of", "group-membership"]

    def domains(self) -> list[str]:
        return ["elements", "compounds", "states_of_matter"]

    def generate(self) -> Iterator[ContrastivePair]:
        raise NotImplementedError("Chemistry generator not yet implemented")
