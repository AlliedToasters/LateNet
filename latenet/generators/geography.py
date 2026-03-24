"""Geospatial: containment, proximity, cardinal direction.

Data sources: geopandas, natural earth, geonames.
Relations: contained-in, north-of/south-of, closer-to.
Domains: countries, cities, rivers, mountains, continents.
"""

from __future__ import annotations

from collections.abc import Iterator

from latenet.generators.base import BaseGenerator
from latenet.types import ContrastivePair


class GeographyGenerator(BaseGenerator):
    """Generate contrastive pairs from geospatial data."""

    @property
    def name(self) -> str:
        return "geography"

    def relation_types(self) -> list[str]:
        return ["contained-in", "north-of", "south-of", "closer-to"]

    def domains(self) -> list[str]:
        return ["countries", "cities", "rivers", "mountains", "continents"]

    def generate(self) -> Iterator[ContrastivePair]:
        raise NotImplementedError("Geography generator not yet implemented")
