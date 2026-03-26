"""Anatomy generator: body system membership, regional containment, structure classification.

Data source: curated hardcoded data in anat_data.py (Gray's Anatomy / Netter's sourced).
Relations: in_system, in_region, is_structure_type, same_region, same_system.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass

from latenet.generators.base import BaseGenerator
from latenet.generators.anat_data import (
    ADJACENT_REGIONS,
    ALL_STRUCTURE_TYPES,
    ALL_SYSTEMS,
    MEDIUM_STRUCTURE_TYPES,
    REGION_LABELS,
    RELATED_STRUCTURE_TYPES,
    RELATED_SYSTEMS,
    STRUCTURES,
    STRUCTURE_TYPE_LABELS,
    SYSTEM_LABELS,
    AnatomicalStructure,
)
from latenet.sanitize import render_template
from latenet.types import ContrastivePair, Difficulty, NegationStrategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnatTemplate:
    id: str
    relation: str
    pattern: str


_SYSTEM_TEMPLATES = [
    AnatTemplate("anat_system_01", "in_system",
                 "The {structure} is part of the {system} system."),
    AnatTemplate("anat_system_02", "in_system",
                 "The {structure} belongs to the {system} system."),
    AnatTemplate("anat_system_03", "in_system",
                 "The {structure} is a component of the {system} system."),
    AnatTemplate("anat_system_04", "in_system",
                 "The {structure} is classified under the {system} system."),
]

_REGION_TEMPLATES = [
    AnatTemplate("anat_region_01", "in_region",
                 "The {structure} is located in the {region}."),
    AnatTemplate("anat_region_02", "in_region",
                 "The {structure} is found in the {region}."),
    AnatTemplate("anat_region_03", "in_region",
                 "The {structure} is in the {region}."),
    AnatTemplate("anat_region_04", "in_region",
                 "The {structure} is a structure of the {region}."),
]

_TYPE_TEMPLATES = [
    AnatTemplate("anat_type_01", "is_structure_type",
                 "The {structure} is {a_type}."),
    AnatTemplate("anat_type_02", "is_structure_type",
                 "The {structure} is classified as {a_type}."),
    AnatTemplate("anat_type_03", "is_structure_type",
                 "The {structure} is a type of {type}."),
]

_COLOC_TEMPLATES = [
    AnatTemplate("anat_coloc_01", "same_region",
                 "The {structureA} and the {structureB} are both in the {region}."),
    AnatTemplate("anat_coloc_02", "same_region",
                 "The {structureA} and the {structureB} are both located in the {region}."),
    AnatTemplate("anat_coloc_03", "same_region",
                 "Both the {structureA} and the {structureB} can be found in the {region}."),
]

_COMEM_TEMPLATES = [
    AnatTemplate("anat_comem_01", "same_system",
                 "The {structureA} and the {structureB} are both part of the {system} system."),
    AnatTemplate("anat_comem_02", "same_system",
                 "The {structureA} and the {structureB} both belong to the {system} system."),
]

_ALL_TEMPLATES: dict[str, list[AnatTemplate]] = {
    "in_system": _SYSTEM_TEMPLATES,
    "in_region": _REGION_TEMPLATES,
    "is_structure_type": _TYPE_TEMPLATES,
    "same_region": _COLOC_TEMPLATES,
    "same_system": _COMEM_TEMPLATES,
}


def _make_pair_id(parts: list[str]) -> str:
    key = ":".join(parts)
    return hashlib.sha256(key.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class AnatomyGenerator(BaseGenerator):
    """Generate contrastive pairs from curated human anatomy data."""

    def __init__(
        self,
        seed: int = 42,
        max_pairs: int | None = None,
    ):
        super().__init__(seed=seed, max_pairs=max_pairs)

        # Precompute indices
        self._structures = list(STRUCTURES)
        self._by_system: dict[str, list[AnatomicalStructure]] = {}
        self._by_region: dict[str, list[AnatomicalStructure]] = {}
        self._by_type: dict[str, list[AnatomicalStructure]] = {}
        # Structures eligible for system membership (no related_systems ambiguity)
        self._system_eligible: list[AnatomicalStructure] = []
        # Structures eligible for regional containment (don't span regions)
        self._region_eligible: list[AnatomicalStructure] = []

        for s in self._structures:
            self._by_system.setdefault(s.body_system, []).append(s)
            self._by_region.setdefault(s.body_region, []).append(s)
            self._by_type.setdefault(s.structure_type, []).append(s)
            if not s.related_systems:
                self._system_eligible.append(s)
            if not s.spans_regions:
                self._region_eligible.append(s)

        logger.info(
            "Anatomy data: %d structures, %d system-eligible, %d region-eligible",
            len(self._structures), len(self._system_eligible), len(self._region_eligible),
        )
        # Log multi-system exclusions
        excluded = [s.name for s in self._structures if s.related_systems]
        if excluded:
            logger.info("Multi-system exclusions (system membership): %s", excluded)
        spans = [s.name for s in self._structures if s.spans_regions]
        if spans:
            logger.info("Spans-regions exclusions (regional containment): %s", spans)

    @property
    def name(self) -> str:
        return "anatomy"

    def relation_types(self) -> list[str]:
        return ["in_system", "in_region", "is_structure_type", "same_region", "same_system"]

    def domains(self) -> list[str]:
        return ["anatomy"]

    # --- Main generate ---

    def generate(self) -> Iterator[ContrastivePair]:
        count = 0
        generators = [
            self._generate_system_membership,
            self._generate_regional_containment,
            self._generate_structure_type,
            self._generate_same_region,
            self._generate_same_system,
        ]
        # Round-robin across sub-generators so max_pairs doesn't starve later relations
        iterators = [gen_fn() for gen_fn in generators]
        counts_by_relation: dict[str, int] = {fn.__name__: 0 for fn in generators}
        active = list(range(len(iterators)))

        while active:
            next_active = []
            for idx in active:
                try:
                    pair = next(iterators[idx])
                except StopIteration:
                    continue
                yield pair
                count += 1
                counts_by_relation[generators[idx].__name__] += 1
                next_active.append(idx)
                if self.max_pairs is not None and count >= self.max_pairs:
                    logger.info("Anatomy generator pair counts: %s", counts_by_relation)
                    return
            active = next_active

        logger.info(
            "Anatomy generator produced %d total pairs. Per relation: %s",
            count, counts_by_relation,
        )

    # --- Helpers ---

    def _pick_template(self, relation: str) -> AnatTemplate:
        templates = _ALL_TEMPLATES[relation]
        return templates[self.rng.randint(0, len(templates) - 1)]

    def _region_distance(self, r1: str, r2: str) -> int:
        """BFS distance between two regions in the adjacency graph."""
        if r1 == r2:
            return 0
        visited = {r1}
        frontier = [r1]
        dist = 0
        while frontier:
            dist += 1
            next_frontier = []
            for r in frontier:
                for neighbor in ADJACENT_REGIONS.get(r, []):
                    if neighbor == r2:
                        return dist
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier
        return 99  # unreachable

    # --- System membership ---

    def _generate_system_membership(self) -> Iterator[ContrastivePair]:
        """'{structure} is part of the {system} system' pairs."""
        eligible = list(self._system_eligible)
        self.rng.shuffle(eligible)

        for struct in eligible:
            true_system = struct.body_system
            true_label = SYSTEM_LABELS[true_system]
            other_systems = [s for s in ALL_SYSTEMS if s != true_system]

            # Hard: same region, different system
            same_region_systems = {
                s.body_system for s in self._structures
                if s.body_region == struct.body_region and s.body_system != true_system
            }
            hard_systems = sorted(same_region_systems)

            # Medium: functionally related system
            related = RELATED_SYSTEMS.get(true_system, [])
            medium_systems = [s for s in related if s not in same_region_systems]

            # Easy: completely unrelated system
            used = same_region_systems | set(related)
            easy_systems = [s for s in other_systems if s not in used]

            for candidates, difficulty, strategy in [
                (hard_systems, Difficulty.HARD.value, NegationStrategy.SIBLING_SWAP.value),
                (medium_systems, Difficulty.MEDIUM.value, NegationStrategy.SIBLING_SWAP.value),
                (easy_systems, Difficulty.EASY.value, NegationStrategy.DISTANT_SWAP.value),
            ]:
                if not candidates:
                    continue
                wrong_system = self.rng.choice(candidates)
                wrong_label = SYSTEM_LABELS[wrong_system]

                template = self._pick_template("in_system")
                true_stmt = render_template(template.pattern,structure=struct.name, system=true_label)
                false_stmt = render_template(template.pattern,structure=struct.name, system=wrong_label)

                pair_id = _make_pair_id([
                    "anat", "system", struct.name, wrong_system, template.id, difficulty,
                ])
                yield ContrastivePair(
                    true_statement=true_stmt,
                    false_statement=false_stmt,
                    pair_id=pair_id,
                    domain="anatomy",
                    relation_type="in_system",
                    difficulty=difficulty,
                    semantic_distance=None,
                    generator=self.name,
                    template_id=template.id,
                    negation_strategy=strategy,
                )

    # --- Regional containment ---

    def _generate_regional_containment(self) -> Iterator[ContrastivePair]:
        """'{structure} is located in the {region}' pairs."""
        eligible = list(self._region_eligible)
        self.rng.shuffle(eligible)

        all_regions = list(ADJACENT_REGIONS.keys())

        for struct in eligible:
            true_region = struct.body_region
            true_label = REGION_LABELS[true_region]
            other_regions = [r for r in all_regions if r != true_region]

            # Hard: adjacent region
            adjacent = ADJACENT_REGIONS.get(true_region, [])
            hard_regions = [r for r in adjacent]

            # Medium: non-adjacent, same half of body
            # Trunk: thorax, abdomen, pelvis, back, neck
            # Extremities: upper_limb, lower_limb, head
            trunk = {"thorax", "abdomen", "pelvis", "back", "neck"}
            if true_region in trunk:
                same_half = [r for r in trunk if r != true_region and r not in adjacent]
            else:
                same_half = [r for r in other_regions if r not in trunk and r not in adjacent]
            medium_regions = same_half

            # Easy: distant region
            used = set(adjacent) | set(medium_regions)
            easy_regions = [r for r in other_regions if r not in used]

            for candidates, difficulty, strategy in [
                (hard_regions, Difficulty.HARD.value, NegationStrategy.SIBLING_SWAP.value),
                (medium_regions, Difficulty.MEDIUM.value, NegationStrategy.SIBLING_SWAP.value),
                (easy_regions, Difficulty.EASY.value, NegationStrategy.DISTANT_SWAP.value),
            ]:
                if not candidates:
                    continue
                wrong_region = self.rng.choice(candidates)
                wrong_label = REGION_LABELS[wrong_region]

                template = self._pick_template("in_region")
                true_stmt = render_template(template.pattern,structure=struct.name, region=true_label)
                false_stmt = render_template(template.pattern,structure=struct.name, region=wrong_label)

                pair_id = _make_pair_id([
                    "anat", "region", struct.name, wrong_region, template.id, difficulty,
                ])
                yield ContrastivePair(
                    true_statement=true_stmt,
                    false_statement=false_stmt,
                    pair_id=pair_id,
                    domain="anatomy",
                    relation_type="in_region",
                    difficulty=difficulty,
                    semantic_distance=None,
                    generator=self.name,
                    template_id=template.id,
                    negation_strategy=strategy,
                )

    # --- Structure type classification ---

    def _generate_structure_type(self) -> Iterator[ContrastivePair]:
        """'{structure} is a {type}' pairs."""
        structures = list(self._structures)
        self.rng.shuffle(structures)

        for struct in structures:
            true_type = struct.structure_type
            true_label = STRUCTURE_TYPE_LABELS[true_type]
            other_types = [t for t in ALL_STRUCTURE_TYPES if t != true_type]

            # Hard: related structure type
            hard_types = [t for t in RELATED_STRUCTURE_TYPES.get(true_type, []) if t in other_types]

            # Medium: medium-relatedness structure type
            medium_types = [
                t for t in MEDIUM_STRUCTURE_TYPES.get(true_type, [])
                if t in other_types and t not in hard_types
            ]

            # Easy: maximally different type
            used = set(hard_types) | set(medium_types)
            easy_types = [t for t in other_types if t not in used]

            for candidates, difficulty, strategy in [
                (hard_types, Difficulty.HARD.value, NegationStrategy.SIBLING_SWAP.value),
                (medium_types, Difficulty.MEDIUM.value, NegationStrategy.SIBLING_SWAP.value),
                (easy_types, Difficulty.EASY.value, NegationStrategy.DISTANT_SWAP.value),
            ]:
                if not candidates:
                    continue
                wrong_type = self.rng.choice(candidates)
                wrong_label = STRUCTURE_TYPE_LABELS[wrong_type]

                template = self._pick_template("is_structure_type")
                true_stmt = render_template(template.pattern,structure=struct.name, type=true_label)
                false_stmt = render_template(template.pattern,structure=struct.name, type=wrong_label)

                pair_id = _make_pair_id([
                    "anat", "type", struct.name, wrong_type, template.id, difficulty,
                ])
                yield ContrastivePair(
                    true_statement=true_stmt,
                    false_statement=false_stmt,
                    pair_id=pair_id,
                    domain="anatomy",
                    relation_type="is_structure_type",
                    difficulty=difficulty,
                    semantic_distance=None,
                    generator=self.name,
                    template_id=template.id,
                    negation_strategy=strategy,
                )

    # --- Regional co-location ---

    def _generate_same_region(self) -> Iterator[ContrastivePair]:
        """'{A} and {B} are both in the {region}' pairs."""
        # Build region-eligible groups
        region_groups: dict[str, list[AnatomicalStructure]] = {}
        for s in self._region_eligible:
            region_groups.setdefault(s.body_region, []).append(s)

        regions = list(region_groups.keys())
        self.rng.shuffle(regions)

        for region in regions:
            members = region_groups[region]
            if len(members) < 2:
                continue

            # Pick pairs from this region
            member_list = list(members)
            self.rng.shuffle(member_list)

            for i in range(0, len(member_list) - 1, 2):
                a = member_list[i]
                b = member_list[i + 1]
                region_label = REGION_LABELS[region]

                template = self._pick_template("same_region")
                true_stmt = render_template(template.pattern,
                    structureA=a.name, structureB=b.name, region=region_label,
                )

                # False: swap B with a structure from a different region
                other_structs = [
                    s for s in self._region_eligible if s.body_region != region
                ]
                if not other_structs:
                    continue

                # Difficulty based on how close the swap region is
                adjacent = set(ADJACENT_REGIONS.get(region, []))
                # Medium: non-adjacent but same body half (trunk vs extremities)
                trunk = {"thorax", "abdomen", "pelvis", "back", "neck"}
                if region in trunk:
                    same_half = {r for r in trunk if r != region} - adjacent
                else:
                    same_half = {r for r in ADJACENT_REGIONS if r not in trunk and r != region} - adjacent

                hard_swaps = [s for s in other_structs if s.body_region in adjacent]
                medium_swaps = [s for s in other_structs if s.body_region in same_half]
                distant_swaps = [
                    s for s in other_structs
                    if s.body_region not in adjacent and s.body_region not in same_half
                ]

                for candidates, difficulty, strategy in [
                    (hard_swaps, Difficulty.HARD.value, NegationStrategy.SIBLING_SWAP.value),
                    (medium_swaps, Difficulty.MEDIUM.value, NegationStrategy.SIBLING_SWAP.value),
                    (distant_swaps, Difficulty.EASY.value, NegationStrategy.DISTANT_SWAP.value),
                ]:
                    if not candidates:
                        continue
                    c = self.rng.choice(candidates)
                    false_stmt = render_template(template.pattern,
                        structureA=a.name, structureB=c.name, region=region_label,
                    )

                    pair_id = _make_pair_id([
                        "anat", "coloc", region, a.name, b.name, c.name, template.id, difficulty,
                    ])
                    yield ContrastivePair(
                        true_statement=true_stmt,
                        false_statement=false_stmt,
                        pair_id=pair_id,
                        domain="anatomy",
                        relation_type="same_region",
                        difficulty=difficulty,
                        semantic_distance=None,
                        generator=self.name,
                        template_id=template.id,
                        negation_strategy=strategy,
                    )

    # --- System co-membership ---

    def _generate_same_system(self) -> Iterator[ContrastivePair]:
        """'{A} and {B} are both part of the {system} system' pairs."""
        # Use system-eligible structures only
        system_groups: dict[str, list[AnatomicalStructure]] = {}
        for s in self._system_eligible:
            system_groups.setdefault(s.body_system, []).append(s)

        systems = list(system_groups.keys())
        self.rng.shuffle(systems)

        for system in systems:
            members = system_groups[system]
            if len(members) < 2:
                continue

            member_list = list(members)
            self.rng.shuffle(member_list)

            for i in range(0, len(member_list) - 1, 2):
                a = member_list[i]
                b = member_list[i + 1]
                system_label = SYSTEM_LABELS[system]

                template = self._pick_template("same_system")
                true_stmt = render_template(template.pattern,
                    structureA=a.name, structureB=b.name, system=system_label,
                )

                # False: swap B with a structure from a different system
                other_structs = [
                    s for s in self._system_eligible if s.body_system != system
                ]
                if not other_structs:
                    continue

                related = set(RELATED_SYSTEMS.get(system, []))
                # Medium: systems that share a body region but aren't functionally related
                system_regions = {s.body_region for s in system_groups[system]}
                co_regional_systems = {
                    s.body_system for s in self._system_eligible
                    if s.body_system != system and s.body_region in system_regions
                } - related

                hard_swaps = [s for s in other_structs if s.body_system in related]
                medium_swaps = [
                    s for s in other_structs
                    if s.body_system in co_regional_systems and s.body_system not in related
                ]
                distant_swaps = [
                    s for s in other_structs
                    if s.body_system not in related and s.body_system not in co_regional_systems
                ]

                for candidates, difficulty, strategy in [
                    (hard_swaps, Difficulty.HARD.value, NegationStrategy.SIBLING_SWAP.value),
                    (medium_swaps, Difficulty.MEDIUM.value, NegationStrategy.SIBLING_SWAP.value),
                    (distant_swaps, Difficulty.EASY.value, NegationStrategy.DISTANT_SWAP.value),
                ]:
                    if not candidates:
                        continue
                    c = self.rng.choice(candidates)
                    false_stmt = render_template(template.pattern,
                        structureA=a.name, structureB=c.name, system=system_label,
                    )

                    pair_id = _make_pair_id([
                        "anat", "comem", system, a.name, b.name, c.name, template.id, difficulty,
                    ])
                    yield ContrastivePair(
                        true_statement=true_stmt,
                        false_statement=false_stmt,
                        pair_id=pair_id,
                        domain="anatomy",
                        relation_type="same_system",
                        difficulty=difficulty,
                        semantic_distance=None,
                        generator=self.name,
                        template_id=template.id,
                        negation_strategy=strategy,
                    )
