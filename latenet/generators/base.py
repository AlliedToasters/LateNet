"""Abstract base class for all generators."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Iterator

from latenet.types import ContrastivePair


class BaseGenerator(ABC):
    """All generators produce ContrastivePair objects from structured data sources."""

    def __init__(self, seed: int = 42, max_pairs: int | None = None):
        self.seed = seed
        self.max_pairs = max_pairs
        self.rng = random.Random(seed)

    @abstractmethod
    def generate(self) -> Iterator[ContrastivePair]:
        """Yield contrastive pairs from this generator's data source."""
        ...

    @abstractmethod
    def relation_types(self) -> list[str]:
        """Return the logical relation types this generator covers."""
        ...

    @abstractmethod
    def domains(self) -> list[str]:
        """Return the semantic domains this generator covers."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this generator (e.g. 'wordnet', 'geography')."""
        ...
