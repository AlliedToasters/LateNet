"""Abstract base class for all generators."""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence

from latenet.types import ContrastivePair

logger = logging.getLogger(__name__)


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

    # --- Round-robin generation -----------------------------------------------

    def _round_robin_generate(
        self,
        sub_generators: list[Callable[[], Iterator[ContrastivePair]]],
    ) -> Iterator[ContrastivePair]:
        """Interleave pairs from multiple sub-generators.

        Cycles through sub-generators one pair at a time so that max_pairs
        doesn't starve later relation types.  Exhausted sub-generators are
        dropped from the rotation.
        """
        iterators = [fn() for fn in sub_generators]
        active = list(range(len(iterators)))
        count = 0

        while active:
            next_active = []
            for idx in active:
                try:
                    pair = next(iterators[idx])
                except StopIteration:
                    continue
                yield pair
                count += 1
                next_active.append(idx)
                if self.max_pairs is not None and count >= self.max_pairs:
                    return
            active = next_active

    # --- Sitelink-weighted sampling ------------------------------------------
    # Reusable across any generator whose entities have a notability signal
    # (Wikipedia sitelinks, population, citation count, etc.).

    @staticmethod
    def build_weights(values: Sequence[float], floor: float = 1.0) -> list[float]:
        """Convert raw notability values to normalized sampling weights.

        Parameters
        ----------
        values : sequence of numbers
            One per entity (e.g. sitelink counts).  NaN / <=0 are clamped to *floor*.
        floor : float
            Minimum weight — prevents zero-probability entities.

        Returns
        -------
        list[float]
            Normalized weights summing to 1.0.
        """
        clamped = [max(v if v == v else floor, floor) for v in values]  # v != v catches NaN
        total = sum(clamped)
        if total == 0:
            n = len(clamped)
            return [1.0 / n] * n
        return [v / total for v in clamped]

    def weighted_sample(
        self,
        indices: Sequence[int],
        weights: Sequence[float],
        n: int,
    ) -> list[int]:
        """Sample *n* unique indices with probability proportional to *weights*."""
        idx_list = list(indices)
        w_list = list(weights)
        sampled: list[int] = []
        seen: set[int] = set()
        while len(sampled) < n and len(seen) < len(idx_list):
            batch = self.rng.choices(idx_list, weights=w_list, k=min(n * 2, len(idx_list)))
            for i in batch:
                if i not in seen:
                    seen.add(i)
                    sampled.append(i)
                    if len(sampled) >= n:
                        break
        return sampled

    def weighted_pick(
        self,
        candidate_indices: Sequence[int],
        weights: Sequence[float],
        k: int = 1,
    ) -> list[int]:
        """Pick *k* unique indices from *candidate_indices*, weighted.

        *weights* must be the **full** weight array (indexed by entity id);
        this method selects the relevant subset for the candidates.
        """
        if not candidate_indices or k <= 0:
            return []
        sub_w = [weights[i] for i in candidate_indices]
        total = sum(sub_w)
        if total == 0:
            return self.rng.sample(list(candidate_indices), min(k, len(candidate_indices)))
        sub_w = [w / total for w in sub_w]
        cand = list(candidate_indices)
        picked: list[int] = []
        seen: set[int] = set()
        while len(picked) < k and len(seen) < len(cand):
            batch = self.rng.choices(cand, weights=sub_w, k=min(k * 2, len(cand)))
            for i in batch:
                if i not in seen:
                    seen.add(i)
                    picked.append(i)
                    if len(picked) >= k:
                        break
        return picked
