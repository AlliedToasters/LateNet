"""Temporal ordering: historical events, inventions, births.

Data sources: curated timeline data, historical databases.
Relations: before, after, century-of.
Domains: wars, inventions, births, historical events.
"""

from __future__ import annotations

from collections.abc import Iterator

from latenet.generators.base import BaseGenerator
from latenet.types import ContrastivePair


class TemporalGenerator(BaseGenerator):
    """Generate contrastive pairs from temporal ordering data."""

    @property
    def name(self) -> str:
        return "temporal"

    def relation_types(self) -> list[str]:
        return ["before", "after", "century-of"]

    def domains(self) -> list[str]:
        return ["wars", "inventions", "births", "historical_events"]

    def generate(self) -> Iterator[ContrastivePair]:
        raise NotImplementedError("Temporal generator not yet implemented")
