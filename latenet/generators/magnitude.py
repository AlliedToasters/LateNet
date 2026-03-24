"""Numerical comparisons: population, area, distance, physical constants.

Data sources: World Bank, census data, reference tables.
Relations: greater-than, less-than.
Domains: population, area, distance, physical_constants.
Uses wide margins (>3x ratio) to avoid ambiguity.
"""

from __future__ import annotations

from collections.abc import Iterator

from latenet.generators.base import BaseGenerator
from latenet.types import ContrastivePair


class MagnitudeGenerator(BaseGenerator):
    """Generate contrastive pairs from numerical magnitude comparisons."""

    @property
    def name(self) -> str:
        return "magnitude"

    def relation_types(self) -> list[str]:
        return ["greater-than", "less-than"]

    def domains(self) -> list[str]:
        return ["population", "area", "distance", "physical_constants"]

    def generate(self) -> Iterator[ContrastivePair]:
        raise NotImplementedError("Magnitude generator not yet implemented")
