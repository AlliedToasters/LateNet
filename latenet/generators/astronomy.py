"""Astronomy generator: solar system relations, stellar properties, constellation membership.

Data source: curated hardcoded data in astro_data.py (NASA/IAU sourced).
Relations: orbits, closer_to_sun, property_greater, is_type, star_property, in_constellation.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass

from latenet.generators.base import BaseGenerator
from latenet.generators.astro_data import (
    CONSTELLATIONS,
    DWARF_PLANETS,
    MOONS,
    PLANET_TYPES,
    PLANETS,
    STAR_TO_CONSTELLATION,
    STARS,
    TYPE_LABELS,
    Moon,
)
from latenet.types import ContrastivePair, Difficulty, NegationStrategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AstroTemplate:
    id: str
    relation: str
    pattern: str


_ORBIT_TEMPLATES = [
    AstroTemplate("astro_orbit_01", "orbits", "{moon} orbits {planet}."),
    AstroTemplate("astro_orbit_02", "orbits", "{moon} is a moon of {planet}."),
    AstroTemplate("astro_orbit_03", "orbits", "{moon} is a natural satellite of {planet}."),
    AstroTemplate("astro_orbit_04", "orbits", "{moon} revolves around {planet}."),
]

_PLANET_ORBIT_TEMPLATES = [
    AstroTemplate("astro_orbit_05", "orbits", "{planet} orbits the Sun."),
    AstroTemplate("astro_orbit_06", "orbits", "{planet} revolves around the Sun."),
]

_ORDER_TEMPLATES = [
    AstroTemplate("astro_order_01", "closer_to_sun",
                  "{planetA} is closer to the Sun than {planetB}."),
    AstroTemplate("astro_order_02", "closer_to_sun",
                  "{planetA} orbits closer to the Sun than {planetB}."),
    AstroTemplate("astro_order_03", "closer_to_sun",
                  "In the solar system, {planetA} is nearer to the Sun than {planetB}."),
]

_ORDINAL_TEMPLATES = [
    AstroTemplate("astro_order_04", "closer_to_sun",
                  "{planet} is the {ordinal} planet from the Sun."),
]

_MAGNITUDE_TEMPLATES = [
    AstroTemplate("astro_magnitude_01", "property_greater",
                  "{bodyA} is more massive than {bodyB}."),
    AstroTemplate("astro_magnitude_02", "property_greater",
                  "{bodyA} is larger than {bodyB}."),
    AstroTemplate("astro_magnitude_03", "property_greater",
                  "{bodyA} has a greater {property} than {bodyB}."),
]

_TYPE_TEMPLATES = [
    AstroTemplate("astro_type_01", "is_type", "{planet} is {type}."),
    AstroTemplate("astro_type_02", "is_type", "{planet} is classified as {type}."),
    AstroTemplate("astro_type_03", "is_type", "{planet} is categorized as {type}."),
]

_STAR_PROP_TEMPLATES = [
    AstroTemplate("astro_star_01", "star_property",
                  "{starA} appears brighter than {starB} as seen from Earth."),
    AstroTemplate("astro_star_02", "star_property",
                  "{starA} is closer to Earth than {starB}."),
]

_CONSTELLATION_TEMPLATES = [
    AstroTemplate("astro_constellation_01", "in_constellation",
                  "{star} is in the constellation {constellation}."),
    AstroTemplate("astro_constellation_02", "in_constellation",
                  "{star} belongs to the constellation {constellation}."),
    AstroTemplate("astro_constellation_03", "in_constellation",
                  "The star {star} is part of {constellation}."),
]

_ALL_TEMPLATES: dict[str, list[AstroTemplate]] = {
    "orbits": _ORBIT_TEMPLATES,
    "orbits_planet": _PLANET_ORBIT_TEMPLATES,
    "closer_to_sun": _ORDER_TEMPLATES,
    "closer_to_sun_ordinal": _ORDINAL_TEMPLATES,
    "property_greater": _MAGNITUDE_TEMPLATES,
    "is_type": _TYPE_TEMPLATES,
    "star_property": _STAR_PROP_TEMPLATES,
    "in_constellation": _CONSTELLATION_TEMPLATES,
}

_ORDINALS = {
    1: "first", 2: "second", 3: "third", 4: "fourth",
    5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth",
}

# Property display names for magnitude comparisons
_PROPERTY_LABELS = {
    "mass": "mass",
    "diameter": "diameter",
    "distance": "distance from the Sun",
    "moons": "number of moons",
}


def _make_pair_id(parts: list[str]) -> str:
    key = ":".join(parts)
    return hashlib.sha256(key.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class AstronomyGenerator(BaseGenerator):
    """Generate contrastive pairs from curated astronomical data."""

    def __init__(
        self,
        seed: int = 42,
        max_pairs: int | None = None,
        magnitude_ratio: float = 3.0,
        moon_count_ratio: float = 2.0,
    ):
        super().__init__(seed=seed, max_pairs=max_pairs)
        self.magnitude_ratio = magnitude_ratio
        self.moon_count_ratio = moon_count_ratio

    @property
    def name(self) -> str:
        return "astronomy"

    def relation_types(self) -> list[str]:
        return [
            "orbits", "closer_to_sun", "property_greater",
            "is_type", "star_property", "in_constellation",
        ]

    def domains(self) -> list[str]:
        return ["astronomy"]

    # --- Main generate ---

    def generate(self) -> Iterator[ContrastivePair]:
        count = 0
        generators = [
            self._generate_orbits,
            self._generate_ordering,
            self._generate_magnitude,
            self._generate_classification,
            self._generate_star_property,
            self._generate_constellation,
        ]
        counts_by_relation: dict[str, int] = {}
        for gen_fn in generators:
            for pair in gen_fn():
                counts_by_relation[pair.relation_type] = counts_by_relation.get(pair.relation_type, 0) + 1
                yield pair
                count += 1
                if self.max_pairs is not None and count >= self.max_pairs:
                    logger.info("Astronomy pair counts: %s", counts_by_relation)
                    return
        logger.info("Astronomy pair counts: %s", counts_by_relation)

    # --- Helpers ---

    def _pick_template(self, key: str) -> AstroTemplate:
        templates = _ALL_TEMPLATES[key]
        return templates[self.rng.randint(0, len(templates) - 1)]

    # --- Orbital containment ---

    def _generate_orbits(self) -> Iterator[ContrastivePair]:
        """Moon→planet and planet→Sun orbital relationships."""
        # Moon-planet pairs
        moons = list(MOONS)
        self.rng.shuffle(moons)

        all_parent_names = sorted({m.parent_planet for m in MOONS})

        for moon in moons:
            true_parent = moon.parent_planet
            template = self._pick_template("orbits")
            true_stmt = template.pattern.format(moon=moon.name, planet=true_parent)

            # Build swap candidates at different difficulties
            swaps = self._orbit_swaps(moon, all_parent_names)
            for wrong_parent, difficulty, strategy in swaps:
                false_stmt = template.pattern.format(moon=moon.name, planet=wrong_parent)
                pair_id = _make_pair_id([
                    "astro", "orbit", moon.name, wrong_parent, template.id, difficulty,
                ])
                yield ContrastivePair(
                    true_statement=true_stmt,
                    false_statement=false_stmt,
                    pair_id=pair_id,
                    domain="astronomy",
                    relation_type="orbits",
                    difficulty=difficulty,
                    semantic_distance=None,
                    generator=self.name,
                    template_id=template.id,
                    negation_strategy=strategy,
                )

        # Planet-Sun pairs (reverse relation: "{planet} orbits the Sun" / "{planet} orbits Jupiter")
        planets = list(PLANETS)
        self.rng.shuffle(planets)
        for planet in planets:
            template = self._pick_template("orbits_planet")
            true_stmt = template.pattern.format(planet=planet.name)
            # False: planet orbits another planet
            other_planets = [p for p in PLANETS if p.name != planet.name]
            if not other_planets:
                continue
            wrong = self.rng.choice(other_planets)
            false_stmt = template.pattern.format(planet=planet.name).replace("the Sun", wrong.name)
            pair_id = _make_pair_id([
                "astro", "orbit_sun", planet.name, wrong.name, template.id,
            ])
            yield ContrastivePair(
                true_statement=true_stmt,
                false_statement=false_stmt,
                pair_id=pair_id,
                domain="astronomy",
                relation_type="orbits",
                difficulty=Difficulty.EASY.value,
                semantic_distance=None,
                generator=self.name,
                template_id=template.id,
                negation_strategy=NegationStrategy.DISTANT_SWAP.value,
            )

    def _orbit_swaps(
        self, moon: Moon, all_parents: list[str],
    ) -> list[tuple[str, str, str]]:
        """Pick swap parents for a moon at hard/medium/easy difficulty."""
        true_parent = moon.parent_planet
        planet_order = {p.name: p.order_from_sun for p in PLANETS}
        true_order = planet_order.get(true_parent)

        swaps: list[tuple[str, str, str]] = []
        other_parents = [p for p in all_parents if p != true_parent]

        if not other_parents:
            return swaps

        if true_order is not None:
            # Hard: adjacent planet
            adjacent = [
                p for p in other_parents
                if p in planet_order and abs(planet_order[p] - true_order) == 1
            ]
            if adjacent:
                pick = self.rng.choice(adjacent)
                swaps.append((pick, Difficulty.HARD.value, NegationStrategy.SIBLING_SWAP.value))

            # Medium: 2-3 positions apart
            medium = [
                p for p in other_parents
                if p in planet_order and 2 <= abs(planet_order[p] - true_order) <= 3
            ]
            if medium:
                pick = self.rng.choice(medium)
                swaps.append((pick, Difficulty.MEDIUM.value, NegationStrategy.SIBLING_SWAP.value))

            # Easy: 4+ positions apart
            distant = [
                p for p in other_parents
                if p in planet_order and abs(planet_order[p] - true_order) >= 4
            ]
            if distant:
                pick = self.rng.choice(distant)
                swaps.append((pick, Difficulty.EASY.value, NegationStrategy.DISTANT_SWAP.value))
        else:
            # Parent not a major planet (e.g. Pluto) — just pick random
            pick = self.rng.choice(other_parents)
            swaps.append((pick, Difficulty.MEDIUM.value, NegationStrategy.SIBLING_SWAP.value))

        return swaps

    # --- Solar system ordering ---

    def _generate_ordering(self) -> Iterator[ContrastivePair]:
        """Planet ordering by distance from the Sun."""
        planets = list(PLANETS)

        # Pairwise comparisons
        self.rng.shuffle(planets)
        for i, a in enumerate(planets):
            for j in range(i + 1, len(planets)):
                b = planets[j]
                closer, farther = (a, b) if a.order_from_sun < b.order_from_sun else (b, a)
                gap = abs(a.order_from_sun - b.order_from_sun)

                template = self._pick_template("closer_to_sun")
                true_stmt = template.pattern.format(planetA=closer.name, planetB=farther.name)
                false_stmt = template.pattern.format(planetA=farther.name, planetB=closer.name)

                if gap == 1:
                    difficulty = Difficulty.HARD.value
                elif gap <= 3:
                    difficulty = Difficulty.MEDIUM.value
                else:
                    difficulty = Difficulty.EASY.value

                pair_id = _make_pair_id([
                    "astro", "order", closer.name, farther.name, template.id,
                ])
                yield ContrastivePair(
                    true_statement=true_stmt,
                    false_statement=false_stmt,
                    pair_id=pair_id,
                    domain="astronomy",
                    relation_type="closer_to_sun",
                    difficulty=difficulty,
                    semantic_distance=None,
                    generator=self.name,
                    template_id=template.id,
                    negation_strategy=NegationStrategy.REVERSE_RELATION.value,
                )

        # Ordinal position statements
        template = self._pick_template("closer_to_sun_ordinal")
        for planet in planets:
            true_ordinal = _ORDINALS[planet.order_from_sun]
            true_stmt = template.pattern.format(planet=planet.name, ordinal=true_ordinal)

            # Pick a wrong ordinal
            wrong_orders = [o for o in _ORDINALS if o != planet.order_from_sun]
            if not wrong_orders:
                continue
            wrong_order = self.rng.choice(wrong_orders)
            false_stmt = template.pattern.format(planet=planet.name, ordinal=_ORDINALS[wrong_order])

            gap = abs(wrong_order - planet.order_from_sun)
            if gap == 1:
                difficulty = Difficulty.HARD.value
            elif gap <= 3:
                difficulty = Difficulty.MEDIUM.value
            else:
                difficulty = Difficulty.EASY.value

            pair_id = _make_pair_id([
                "astro", "ordinal", planet.name, str(wrong_order), template.id,
            ])
            yield ContrastivePair(
                true_statement=true_stmt,
                false_statement=false_stmt,
                pair_id=pair_id,
                domain="astronomy",
                relation_type="closer_to_sun",
                difficulty=difficulty,
                semantic_distance=None,
                generator=self.name,
                template_id=template.id,
                negation_strategy=NegationStrategy.SIBLING_SWAP.value,
            )

    # --- Property magnitude ---

    def _generate_magnitude(self) -> Iterator[ContrastivePair]:
        """Property comparisons: mass, diameter, distance, moon count."""
        planets = list(PLANETS)

        comparisons: list[tuple[str, str]] = [
            ("mass_kg", "mass"),
            ("diameter_km", "diameter"),
            ("average_distance_from_sun_au", "distance"),
            ("number_of_moons", "moons"),
        ]

        for attr, prop_key in comparisons:
            prop_label = _PROPERTY_LABELS[prop_key]
            ratio_threshold = self.moon_count_ratio if prop_key == "moons" else self.magnitude_ratio

            pairs_list = list(planets)
            self.rng.shuffle(pairs_list)

            for i, a in enumerate(pairs_list):
                for j in range(i + 1, len(pairs_list)):
                    b = pairs_list[j]
                    va = getattr(a, attr)
                    vb = getattr(b, attr)
                    if va == 0 and vb == 0:
                        continue

                    big, small = (a, b) if va > vb else (b, a)
                    v_big = max(va, vb)
                    v_small = min(va, vb)

                    if v_small <= 0 or v_big / v_small < ratio_threshold:
                        continue

                    if prop_key == "mass":
                        template = _ALL_TEMPLATES["property_greater"][0]  # "more massive"
                    elif prop_key == "diameter":
                        template = _ALL_TEMPLATES["property_greater"][1]  # "larger"
                    else:
                        template = _ALL_TEMPLATES["property_greater"][2]  # "greater {property}"

                    true_stmt = template.pattern.format(
                        bodyA=big.name, bodyB=small.name, property=prop_label,
                    )
                    false_stmt = template.pattern.format(
                        bodyA=small.name, bodyB=big.name, property=prop_label,
                    )

                    # Difficulty based on ratio
                    ratio = v_big / v_small
                    if ratio < 5:
                        difficulty = Difficulty.HARD.value
                    elif ratio < 20:
                        difficulty = Difficulty.MEDIUM.value
                    else:
                        difficulty = Difficulty.EASY.value

                    pair_id = _make_pair_id([
                        "astro", "magnitude", prop_key, big.name, small.name, template.id,
                    ])
                    yield ContrastivePair(
                        true_statement=true_stmt,
                        false_statement=false_stmt,
                        pair_id=pair_id,
                        domain="astronomy",
                        relation_type="property_greater",
                        difficulty=difficulty,
                        semantic_distance=None,
                        generator=self.name,
                        template_id=template.id,
                        negation_strategy=NegationStrategy.REVERSE_RELATION.value,
                    )

    # --- Classification ---

    def _generate_classification(self) -> Iterator[ContrastivePair]:
        """Planet type classification."""
        all_bodies = list(PLANETS) + list(DWARF_PLANETS)
        self.rng.shuffle(all_bodies)

        for body in all_bodies:
            true_type = body.planet_type
            true_label = TYPE_LABELS[true_type]
            template = self._pick_template("is_type")
            true_stmt = template.pattern.format(planet=body.name, type=true_label)

            wrong_types = [t for t in PLANET_TYPES if t != true_type]

            # Hard: most similar type
            similar_map: dict[str, list[str]] = {
                "terrestrial": ["ice_giant"],
                "gas_giant": ["ice_giant"],
                "ice_giant": ["gas_giant"],
                "dwarf_planet": ["terrestrial"],
            }
            # Medium: intermediate similarity
            medium_map: dict[str, list[str]] = {
                "terrestrial": ["dwarf_planet"],
                "gas_giant": ["terrestrial"],
                "ice_giant": ["terrestrial", "dwarf_planet"],
                "dwarf_planet": ["ice_giant"],
            }
            # Easy: maximally different swap
            distant_map: dict[str, list[str]] = {
                "terrestrial": ["gas_giant"],
                "gas_giant": ["dwarf_planet"],
                "ice_giant": ["gas_giant"],
                "dwarf_planet": ["gas_giant"],
            }

            hard_candidates = [t for t in similar_map.get(true_type, []) if t in wrong_types]
            medium_candidates = [t for t in medium_map.get(true_type, []) if t in wrong_types]
            easy_candidates = [t for t in distant_map.get(true_type, []) if t in wrong_types]

            for candidates, difficulty, strategy in [
                (hard_candidates, Difficulty.HARD.value, NegationStrategy.SIBLING_SWAP.value),
                (medium_candidates, Difficulty.MEDIUM.value, NegationStrategy.SIBLING_SWAP.value),
                (easy_candidates, Difficulty.EASY.value, NegationStrategy.DISTANT_SWAP.value),
            ]:
                if not candidates:
                    continue
                wrong = self.rng.choice(candidates)
                wrong_label = TYPE_LABELS[wrong]
                false_stmt = template.pattern.format(planet=body.name, type=wrong_label)
                pair_id = _make_pair_id([
                    "astro", "type", body.name, wrong, template.id, difficulty,
                ])
                yield ContrastivePair(
                    true_statement=true_stmt,
                    false_statement=false_stmt,
                    pair_id=pair_id,
                    domain="astronomy",
                    relation_type="is_type",
                    difficulty=difficulty,
                    semantic_distance=None,
                    generator=self.name,
                    template_id=template.id,
                    negation_strategy=strategy,
                )

    # --- Stellar properties ---

    def _generate_star_property(self) -> Iterator[ContrastivePair]:
        """Star brightness and distance comparisons."""
        # Exclude the Sun from stellar comparisons (it's a special case)
        stars = [s for s in STARS if s.name != "Sun"]
        self.rng.shuffle(stars)

        # Apparent magnitude (lower = brighter)
        for i, a in enumerate(stars):
            for j in range(i + 1, len(stars)):
                b = stars[j]
                # Lower apparent magnitude = brighter
                brighter, dimmer = (a, b) if a.apparent_magnitude < b.apparent_magnitude else (b, a)
                mag_diff = abs(a.apparent_magnitude - b.apparent_magnitude)

                if mag_diff < 0.5:
                    continue  # Too close to compare meaningfully

                template = _ALL_TEMPLATES["star_property"][0]  # brightness
                true_stmt = template.pattern.format(starA=brighter.name, starB=dimmer.name)
                false_stmt = template.pattern.format(starA=dimmer.name, starB=brighter.name)

                if mag_diff < 1.5:
                    difficulty = Difficulty.HARD.value
                elif mag_diff < 4.0:
                    difficulty = Difficulty.MEDIUM.value
                else:
                    difficulty = Difficulty.EASY.value

                pair_id = _make_pair_id([
                    "astro", "star_bright", brighter.name, dimmer.name, template.id,
                ])
                yield ContrastivePair(
                    true_statement=true_stmt,
                    false_statement=false_stmt,
                    pair_id=pair_id,
                    domain="astronomy",
                    relation_type="star_property",
                    difficulty=difficulty,
                    semantic_distance=None,
                    generator=self.name,
                    template_id=template.id,
                    negation_strategy=NegationStrategy.REVERSE_RELATION.value,
                )

        # Distance comparisons
        for i, a in enumerate(stars):
            for j in range(i + 1, len(stars)):
                b = stars[j]
                closer, farther = (a, b) if a.distance_ly < b.distance_ly else (b, a)

                if farther.distance_ly <= 0 or closer.distance_ly <= 0:
                    continue
                ratio = farther.distance_ly / closer.distance_ly
                if ratio < self.magnitude_ratio:
                    continue

                template = _ALL_TEMPLATES["star_property"][1]  # distance
                true_stmt = template.pattern.format(starA=closer.name, starB=farther.name)
                false_stmt = template.pattern.format(starA=farther.name, starB=closer.name)

                if ratio < 5:
                    difficulty = Difficulty.HARD.value
                elif ratio < 20:
                    difficulty = Difficulty.MEDIUM.value
                else:
                    difficulty = Difficulty.EASY.value

                pair_id = _make_pair_id([
                    "astro", "star_dist", closer.name, farther.name, template.id,
                ])
                yield ContrastivePair(
                    true_statement=true_stmt,
                    false_statement=false_stmt,
                    pair_id=pair_id,
                    domain="astronomy",
                    relation_type="star_property",
                    difficulty=difficulty,
                    semantic_distance=None,
                    generator=self.name,
                    template_id=template.id,
                    negation_strategy=NegationStrategy.REVERSE_RELATION.value,
                )

    # --- Constellation membership ---

    def _generate_constellation(self) -> Iterator[ContrastivePair]:
        """Star-to-constellation containment."""
        star_constellation_pairs = [
            (star_name, const_name)
            for star_name, const_name in STAR_TO_CONSTELLATION.items()
        ]
        self.rng.shuffle(star_constellation_pairs)

        all_constellations = [c.name for c in CONSTELLATIONS]

        for star_name, true_const in star_constellation_pairs:
            template = self._pick_template("in_constellation")
            true_stmt = template.pattern.format(star=star_name, constellation=true_const)

            wrong_consts = [c for c in all_constellations if c != true_const]
            if not wrong_consts:
                continue

            # Classify wrong constellations by hemisphere relationship
            true_hemisphere = next(
                (c.hemisphere for c in CONSTELLATIONS if c.name == true_const), None
            )
            const_hemi = {c.name: c.hemisphere for c in CONSTELLATIONS}

            # Hard: same specific hemisphere (northern/southern match, not "both")
            hard_consts = [
                c for c in wrong_consts
                if const_hemi.get(c) == true_hemisphere
                and true_hemisphere != "both"
            ]
            # Medium: "both" hemisphere constellations, or true is "both" and wrong is specific
            medium_consts = [
                c for c in wrong_consts
                if const_hemi.get(c) == "both" or (true_hemisphere == "both" and const_hemi.get(c) != "both")
            ]
            # Easy: opposite hemisphere
            easy_consts = [
                c for c in wrong_consts
                if c not in hard_consts and c not in medium_consts
            ]

            for candidates, difficulty, strategy in [
                (hard_consts, Difficulty.HARD.value, NegationStrategy.SIBLING_SWAP.value),
                (medium_consts, Difficulty.MEDIUM.value, NegationStrategy.SIBLING_SWAP.value),
                (easy_consts, Difficulty.EASY.value, NegationStrategy.DISTANT_SWAP.value),
            ]:
                if not candidates:
                    continue
                wrong = self.rng.choice(candidates)
                false_stmt = template.pattern.format(star=star_name, constellation=wrong)
                pair_id = _make_pair_id([
                    "astro", "constellation", star_name, wrong, template.id, difficulty,
                ])
                yield ContrastivePair(
                    true_statement=true_stmt,
                    false_statement=false_stmt,
                    pair_id=pair_id,
                    domain="astronomy",
                    relation_type="in_constellation",
                    difficulty=difficulty,
                    semantic_distance=None,
                    generator=self.name,
                    template_id=template.id,
                    negation_strategy=strategy,
                )
