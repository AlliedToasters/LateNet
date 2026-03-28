"""Geography generator: containment, cardinal direction, proximity, magnitude.

Data sources: Natural Earth (countries, places, rivers), GeoNames (cities).
Relations: contained_in, cardinal_direction, closer_to, population_greater, area_greater.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass

import geopandas as gpd
import pandas as pd
from geopy.distance import geodesic

from latenet.generators.base import BaseGenerator
from latenet.generators.geo_data import (
    get_clean_cities,
    get_clean_countries,
    get_country_code_to_name,
)
from latenet.sanitize import render_template
from latenet.types import ContrastivePair, Difficulty, NegationStrategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GeoTemplate:
    id: str
    relation: str
    pattern: str
    supports_negation: bool = True


# Containment templates
_CONTAINMENT_TEMPLATES = [
    GeoTemplate("geo_contain_01", "contained_in", "{city} is located in {country}"),
    GeoTemplate("geo_contain_02", "contained_in", "{city} is a city in {country}"),
    GeoTemplate("geo_contain_03", "contained_in", "The city of {city} is in {country}"),
    GeoTemplate("geo_contain_04", "contained_in", "{city} can be found in {country}"),
]

# Cardinal direction templates
_CARDINAL_TEMPLATES = [
    GeoTemplate("geo_cardinal_01", "cardinal_direction", "{countryA} is {direction} of {countryB}"),
    GeoTemplate("geo_cardinal_02", "cardinal_direction", "{countryA} is located {direction} of {countryB}"),
    GeoTemplate("geo_cardinal_03", "cardinal_direction", "{countryA} lies to the {direction} of {countryB}"),
]

# Proximity templates
_PROXIMITY_TEMPLATES = [
    GeoTemplate("geo_closer_01", "closer_to", "{city} is closer to {near_city} than to {far_city}"),
    GeoTemplate("geo_closer_02", "closer_to", "{city} is nearer to {near_city} than to {far_city}"),
]

# Population magnitude templates
_POP_TEMPLATES = [
    GeoTemplate("geo_pop_01", "population_greater", "{big} has a larger population than {small}"),
    GeoTemplate("geo_pop_02", "population_greater", "{big}'s population is greater than {small}'s"),
    GeoTemplate("geo_pop_03", "population_greater", "In terms of population, {big} exceeds {small}"),
]

# Area magnitude templates
_AREA_TEMPLATES = [
    GeoTemplate("geo_area_01", "area_greater", "{big} is larger in area than {small}"),
    GeoTemplate("geo_area_02", "area_greater", "{big} covers more land area than {small}"),
    GeoTemplate("geo_area_03", "area_greater", "In terms of area, {big} exceeds {small}"),
]

_ALL_TEMPLATES: dict[str, list[GeoTemplate]] = {
    "contained_in": _CONTAINMENT_TEMPLATES,
    "cardinal_direction": _CARDINAL_TEMPLATES,
    "closer_to": _PROXIMITY_TEMPLATES,
    "population_greater": _POP_TEMPLATES,
    "area_greater": _AREA_TEMPLATES,
}

# Direction opposites for false statement generation
_OPPOSITE_DIRECTION = {
    "north": "south",
    "south": "north",
    "east": "west",
    "west": "east",
}


def _make_pair_id(parts: list[str]) -> str:
    key = ":".join(parts)
    return hashlib.sha256(key.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class GeographyGenerator(BaseGenerator):
    """Generate contrastive pairs from geospatial data."""

    def __init__(
        self,
        seed: int = 42,
        max_pairs: int | None = None,
        min_country_pop: int = 100_000,
        min_city_pop: int = 50_000,
        min_proximity_pop: int = 200_000,
        lat_margin: float = 5.0,
        lon_margin: float = 10.0,
        distance_ratio: float = 3.0,
        magnitude_ratio: float = 3.0,
        max_false_per_true: int = 2,
    ):
        super().__init__(seed=seed, max_pairs=max_pairs)
        self.min_country_pop = min_country_pop
        self.min_city_pop = min_city_pop
        self.min_proximity_pop = min_proximity_pop
        self.lat_margin = lat_margin
        self.lon_margin = lon_margin
        self.distance_ratio = distance_ratio
        self.magnitude_ratio = magnitude_ratio
        self.max_false_per_true = max_false_per_true

        # Loaded lazily
        self._countries: gpd.GeoDataFrame | None = None
        self._cities: pd.DataFrame | None = None
        self._cc_to_name: dict[str, str] | None = None
        self._neighbor_map: dict[str, list[str]] | None = None

    @property
    def name(self) -> str:
        return "geography"

    def relation_types(self) -> list[str]:
        return ["contained_in", "cardinal_direction", "closer_to", "population_greater", "area_greater"]

    def domains(self) -> list[str]:
        return ["geography"]

    # --- Data loading ---

    def _load_data(self) -> None:
        if self._countries is not None:
            return
        logger.info("Loading geography data...")
        self._countries = get_clean_countries(min_pop=self.min_country_pop)
        self._cities = get_clean_cities(min_pop=self.min_city_pop)
        self._cc_to_name = get_country_code_to_name(self._countries)
        self._neighbor_map = self._build_neighbor_map()

        # Compute country centroids and areas once
        name_col = "NAME" if "NAME" in self._countries.columns else "ADMIN"
        self._countries = self._countries.copy()
        self._countries["_centroid"] = self._countries.geometry.centroid
        self._countries["_lat"] = self._countries["_centroid"].y
        self._countries["_lon"] = self._countries["_centroid"].x
        # Expand abbreviated country names from Natural Earth
        _NAME_FIXES = {
            "S. Sudan": "South Sudan",
            "N. Korea": "North Korea",
            "S. Korea": "South Korea",
            "Dem. Rep. Congo": "Democratic Republic of the Congo",
            "Central African Rep.": "Central African Republic",
            "Dominican Rep.": "Dominican Republic",
            "Eq. Guinea": "Equatorial Guinea",
            "eSwatini": "Eswatini",
            "Bosnia and Herz.": "Bosnia and Herzegovina",
            "Solomon Is.": "Solomon Islands",
        }
        self._countries["_name"] = self._countries[name_col].replace(_NAME_FIXES)

        # Area in km² using Mollweide equal-area projection
        mollweide = self._countries.to_crs("+proj=moll")
        self._countries["_area_km2"] = mollweide.geometry.area / 1e6

        # Build continent lookup
        cont_col = "CONTINENT" if "CONTINENT" in self._countries.columns else None
        if cont_col:
            self._country_continent = dict(
                zip(self._countries["_name"], self._countries[cont_col])
            )
        else:
            self._country_continent = {}

        # Map city country_code to country name & continent
        self._city_country = {}
        for _, row in self._cities.iterrows():
            cc = row["country_code"]
            if cc in self._cc_to_name:
                self._city_country[row["name"]] = self._cc_to_name[cc]

        # Build log-population weights for countries and cities
        country_pops = self._countries["POP_EST"].tolist()
        log_country_pops = [
            math.log(max(float(p) if pd.notna(p) else 1.0, 1.0))
            for p in country_pops
        ]
        self._country_weights: list[float] = self.build_weights(log_country_pops)
        self._country_name_to_idx: dict[str, int] = {
            n: i for i, n in enumerate(self._countries["_name"].tolist())
        }

        city_pops = self._cities["population"].tolist()
        log_city_pops = [
            math.log(max(float(p) if pd.notna(p) else 1.0, 1.0))
            for p in city_pops
        ]
        self._city_weights: list[float] = self.build_weights(log_city_pops)

        logger.info(
            "Geography data loaded: %d countries, %d cities",
            len(self._countries), len(self._cities),
        )

    def _build_neighbor_map(self) -> dict[str, list[str]]:
        """Build country name -> list of neighboring country names via geometry touching."""
        gdf = self._countries
        name_col = "NAME" if "NAME" in gdf.columns else "ADMIN"
        neighbor_map: dict[str, list[str]] = {}
        names = gdf[name_col].tolist()
        for i, row_i in gdf.iterrows():
            n_i = row_i[name_col]
            neighbors = []
            for j, row_j in gdf.iterrows():
                if i == j:
                    continue
                if row_i.geometry.touches(row_j.geometry) or row_i.geometry.intersects(row_j.geometry):
                    neighbors.append(row_j[name_col])
            neighbor_map[n_i] = neighbors
        logger.info("Built neighbor map for %d countries", len(neighbor_map))
        return neighbor_map

    # --- Main generate ---

    def generate(self) -> Iterator[ContrastivePair]:
        self._load_data()
        count = 0

        generators = [
            self._generate_containment,
            self._generate_cardinal,
            self._generate_proximity,
            self._generate_population,
            self._generate_area,
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
                    logger.info("Geography generator pair counts: %s", counts_by_relation)
                    return
            active = next_active

        logger.info(
            "Geography generator produced %d total pairs. Per relation: %s",
            count, counts_by_relation,
        )

    # --- Containment ---

    def _generate_containment(self) -> Iterator[ContrastivePair]:
        """City in Country pairs with swapped-country false statements."""
        cities = self._cities.copy()

        # Build city -> true country mapping
        city_rows = []
        for _, row in cities.iterrows():
            cc = row["country_code"]
            if cc not in self._cc_to_name:
                continue
            country_name = self._cc_to_name[cc]
            city_rows.append({
                "city": row["name"],
                "country": country_name,
                "lat": row["latitude"],
                "lon": row["longitude"],
                "population": float(row["population"]) if pd.notna(row["population"]) else 1.0,
            })

        # Order by log-population weight so large cities appear first when max_pairs truncates
        log_pops = [math.log(max(r["population"], 1.0)) for r in city_rows]
        city_w = self.build_weights(log_pops)
        ordered_indices = self.weighted_sample(list(range(len(city_rows))), city_w, n=len(city_rows))
        city_rows = [city_rows[i] for i in ordered_indices]

        country_names = list(self._countries["_name"])
        templates = _ALL_TEMPLATES["contained_in"]

        for city_info in city_rows:
            city = city_info["city"]
            true_country = city_info["country"]
            continent = self._country_continent.get(true_country)

            # Pick negation countries at different difficulty levels
            swaps = self._pick_containment_swaps(true_country, continent, country_names)
            if not swaps:
                continue

            template = templates[self.rng.randint(0, len(templates) - 1)]
            true_stmt = render_template(template.pattern,city=city, country=true_country)

            for swap_country, difficulty, strategy in swaps[:self.max_false_per_true]:
                false_stmt = render_template(template.pattern,city=city, country=swap_country)
                pair_id = _make_pair_id([
                    "geo", "contain", city, true_country, swap_country, template.id
                ])
                yield ContrastivePair(
                    true_statement=true_stmt,
                    false_statement=false_stmt,
                    pair_id=pair_id,
                    domain="geography",
                    relation_type="contained_in",
                    difficulty=difficulty,
                    semantic_distance=None,
                    generator=self.name,
                    template_id=template.id,
                    negation_strategy=strategy,
                    gen_params={
                        "city": city,
                        "true_country": true_country,
                        "false_country": swap_country,
                        "continent": continent,
                    },
                )

    def _pick_containment_swaps(
        self, true_country: str, continent: str | None, all_countries: list[str],
    ) -> list[tuple[str, str, str]]:
        """Pick swap countries at hard/medium/easy difficulty."""
        swaps = []

        # Hard: neighboring country — pick by log-population weight
        neighbors = self._neighbor_map.get(true_country, [])
        if neighbors:
            neighbor_idx = [self._country_name_to_idx[c] for c in neighbors if c in self._country_name_to_idx]
            if neighbor_idx:
                picked = self.weighted_pick(neighbor_idx, self._country_weights, k=1)
                if picked:
                    swaps.append((all_countries[picked[0]], Difficulty.HARD.value, NegationStrategy.SIBLING_SWAP.value))

        # Medium: same continent, non-neighbor — pick by log-population weight
        neighbor_set = set(neighbors)
        if continent:
            same_cont_idx = [
                self._country_name_to_idx[c]
                for c in all_countries
                if self._country_continent.get(c) == continent
                and c != true_country
                and c not in neighbor_set
                and c in self._country_name_to_idx
            ]
            if same_cont_idx:
                picked = self.weighted_pick(same_cont_idx, self._country_weights, k=1)
                if picked:
                    swaps.append((all_countries[picked[0]], Difficulty.MEDIUM.value, NegationStrategy.SIBLING_SWAP.value))

        # Easy: different continent — pick by log-population weight
        if continent:
            diff_cont_idx = [
                self._country_name_to_idx[c]
                for c in all_countries
                if self._country_continent.get(c) != continent
                and self._country_continent.get(c) is not None
                and c in self._country_name_to_idx
            ]
        else:
            diff_cont_idx = [
                self._country_name_to_idx[c]
                for c in all_countries
                if c != true_country and c in self._country_name_to_idx
            ]
        if diff_cont_idx:
            picked = self.weighted_pick(diff_cont_idx, self._country_weights, k=1)
            if picked:
                swaps.append((all_countries[picked[0]], Difficulty.EASY.value, NegationStrategy.DISTANT_SWAP.value))

        return swaps

    # --- Cardinal direction ---

    def _generate_cardinal(self) -> Iterator[ContrastivePair]:
        """Country A is {direction} of Country B."""
        countries = self._countries
        names = list(countries["_name"])
        lat_map = dict(zip(countries["_name"], countries["_lat"]))
        lon_map = dict(zip(countries["_name"], countries["_lon"]))

        # Generate pairs for north/south and east/west
        # Order countries by log-population weight so prominent countries appear first
        pairs_seen: set[tuple[str, str, str]] = set()
        indices = self.weighted_sample(list(range(len(names))), self._country_weights, n=len(names))

        templates = _ALL_TEMPLATES["cardinal_direction"]

        for i in indices:
            for j in indices:
                if i == j:
                    continue
                a, b = names[i], names[j]

                lat_a, lat_b = lat_map[a], lat_map[b]
                lon_a, lon_b = lon_map[a], lon_map[b]

                # North/South
                lat_diff = lat_a - lat_b
                if abs(lat_diff) >= self.lat_margin:
                    direction = "north" if lat_diff > 0 else "south"
                    key = (a, b, direction)
                    if key not in pairs_seen:
                        pairs_seen.add(key)
                        template = templates[self.rng.randint(0, len(templates) - 1)]
                        true_stmt = render_template(template.pattern,
                            countryA=a, countryB=b, direction=direction
                        )
                        opp = _OPPOSITE_DIRECTION[direction]
                        false_stmt = render_template(template.pattern,
                            countryA=a, countryB=b, direction=opp
                        )
                        pair_id = _make_pair_id([
                            "geo", "cardinal", a, b, direction, template.id
                        ])

                        # Difficulty: larger margin = easier
                        diff = self._cardinal_difficulty(abs(lat_diff))

                        yield ContrastivePair(
                            true_statement=true_stmt,
                            false_statement=false_stmt,
                            pair_id=pair_id,
                            domain="geography",
                            relation_type="cardinal_direction",
                            difficulty=diff,
                            semantic_distance=None,
                            generator=self.name,
                            template_id=template.id,
                            negation_strategy=NegationStrategy.REVERSE_RELATION.value,
                            gen_params={
                                "country_a": a,
                                "country_b": b,
                                "direction": direction,
                                "margin": round(abs(lat_diff), 2),
                            },
                        )

                # East/West — skip antimeridian-ambiguous pairs
                lon_diff = lon_a - lon_b
                # Normalize to [-180, 180]
                if lon_diff > 180:
                    lon_diff -= 360
                elif lon_diff < -180:
                    lon_diff += 360

                # Skip if either country spans the antimeridian (rough heuristic: |lon| > 170)
                if abs(lon_a) > 170 or abs(lon_b) > 170:
                    continue

                if abs(lon_diff) >= self.lon_margin:
                    direction = "east" if lon_diff > 0 else "west"
                    key = (a, b, direction)
                    if key not in pairs_seen:
                        pairs_seen.add(key)
                        template = templates[self.rng.randint(0, len(templates) - 1)]
                        true_stmt = render_template(template.pattern,
                            countryA=a, countryB=b, direction=direction
                        )
                        opp = _OPPOSITE_DIRECTION[direction]
                        false_stmt = render_template(template.pattern,
                            countryA=a, countryB=b, direction=opp
                        )
                        pair_id = _make_pair_id([
                            "geo", "cardinal", a, b, direction, template.id
                        ])
                        diff = self._cardinal_difficulty(abs(lon_diff))

                        yield ContrastivePair(
                            true_statement=true_stmt,
                            false_statement=false_stmt,
                            pair_id=pair_id,
                            domain="geography",
                            relation_type="cardinal_direction",
                            difficulty=diff,
                            semantic_distance=None,
                            generator=self.name,
                            template_id=template.id,
                            negation_strategy=NegationStrategy.REVERSE_RELATION.value,
                            gen_params={
                                "country_a": a,
                                "country_b": b,
                                "direction": direction,
                                "margin": round(abs(lon_diff), 2),
                            },
                        )

    def _cardinal_difficulty(self, margin: float) -> str:
        """Difficulty based on how close the comparison is to threshold."""
        if margin < 10:
            return Difficulty.HARD.value
        if margin < 30:
            return Difficulty.MEDIUM.value
        return Difficulty.EASY.value

    # --- Proximity (closer_to) ---

    def _generate_proximity(self) -> Iterator[ContrastivePair]:
        """City A is closer to City B than to City C."""
        cities = self._cities
        # Use higher population threshold for proximity to avoid obscure cities
        # that LLMs have weak geographic representations for
        if self.min_proximity_pop > self.min_city_pop:
            cities = cities[cities["population"] >= self.min_proximity_pop]
        city_names = list(cities["name"])
        city_coords = {
            row["name"]: (row["latitude"], row["longitude"])
            for _, row in cities.iterrows()
        }
        city_pops_prox = [
            float(p) if pd.notna(p) else 1.0
            for p in cities["population"].tolist()
        ]
        log_pops_prox = [math.log(max(p, 1.0)) for p in city_pops_prox]
        prox_weights = self.build_weights(log_pops_prox)

        templates = _ALL_TEMPLATES["closer_to"]

        # Sample pool weighted by log-population; cap to avoid combinatorial explosion
        pool_size = min(len(city_names), 500)
        pool_indices = self.weighted_sample(list(range(len(city_names))), prox_weights, n=pool_size)
        sample = [city_names[i] for i in pool_indices]
        sample_weights = [prox_weights[i] for i in pool_indices]
        # Normalize sample weights for within-pool picks
        sample_w_total = sum(sample_weights)
        sample_weights_norm = [w / sample_w_total for w in sample_weights]

        for i, anchor in enumerate(sample):
            # Pick two other cities weighted by log-population
            other_idx = [j for j in range(len(sample)) if sample[j] != anchor]
            if len(other_idx) < 2:
                continue

            near_picks = self.rng.choices(other_idx, weights=[sample_weights_norm[j] for j in other_idx], k=1)
            near_i = near_picks[0]
            near = sample[near_i]

            far_idx = [j for j in other_idx if j != near_i]
            if not far_idx:
                continue
            far_picks = self.rng.choices(far_idx, weights=[sample_weights_norm[j] for j in far_idx], k=1)
            far = sample[far_picks[0]]

            coord_a = city_coords[anchor]
            coord_near = city_coords[near]
            coord_far = city_coords[far]

            d_near = geodesic(coord_a, coord_near).km
            d_far = geodesic(coord_a, coord_far).km

            # Ensure near is actually closer; if not, swap
            if d_near > d_far:
                near, far = far, near
                d_near, d_far = d_far, d_near

            # Enforce ratio threshold
            if d_near == 0 or d_far / d_near < self.distance_ratio:
                continue

            template = templates[self.rng.randint(0, len(templates) - 1)]
            true_stmt = render_template(template.pattern,
                city=anchor, near_city=near, far_city=far
            )
            # False: swap near and far
            false_stmt = render_template(template.pattern,
                city=anchor, near_city=far, far_city=near
            )
            pair_id = _make_pair_id([
                "geo", "closer", anchor, near, far, template.id
            ])

            # Difficulty based on ratio
            ratio = d_far / d_near
            if ratio < 5:
                diff = Difficulty.HARD.value
            elif ratio < 15:
                diff = Difficulty.MEDIUM.value
            else:
                diff = Difficulty.EASY.value

            yield ContrastivePair(
                true_statement=true_stmt,
                false_statement=false_stmt,
                pair_id=pair_id,
                domain="geography",
                relation_type="closer_to",
                difficulty=diff,
                semantic_distance=None,
                generator=self.name,
                template_id=template.id,
                negation_strategy=NegationStrategy.REVERSE_RELATION.value,
                gen_params={
                    "anchor": anchor,
                    "near": near,
                    "far": far,
                    "distance_near_km": round(d_near, 1),
                    "distance_far_km": round(d_far, 1),
                    "ratio": round(ratio, 2),
                },
            )

    # --- Population magnitude ---

    def _generate_population(self) -> Iterator[ContrastivePair]:
        """Country A has larger population than Country B."""
        countries = self._countries
        pop_map = dict(zip(countries["_name"], countries["POP_EST"]))
        valid_idx = [
            i for i, n in enumerate(countries["_name"])
            if pd.notna(pop_map.get(n)) and pop_map[n] > 0
        ]
        ordered = self.weighted_sample(valid_idx, self._country_weights, n=len(valid_idx))
        names = [countries["_name"].iloc[i] for i in ordered]

        templates = _ALL_TEMPLATES["population_greater"]
        # Cap per-country appearances to prevent tiny/huge countries from dominating
        country_counts: dict[str, int] = {}
        max_per_country = 5

        for i, a in enumerate(names):
            for j in range(i + 1, len(names)):
                b = names[j]
                pop_a, pop_b = pop_map[a], pop_map[b]
                big, small = (a, b) if pop_a > pop_b else (b, a)
                pop_big, pop_small = max(pop_a, pop_b), min(pop_a, pop_b)

                if pop_small <= 0 or pop_big / pop_small < self.magnitude_ratio:
                    continue

                # Enforce per-country cap
                if country_counts.get(a, 0) >= max_per_country:
                    break
                if country_counts.get(b, 0) >= max_per_country:
                    continue

                template = templates[self.rng.randint(0, len(templates) - 1)]
                true_stmt = render_template(template.pattern,big=big, small=small)
                false_stmt = render_template(template.pattern,big=small, small=big)
                pair_id = _make_pair_id([
                    "geo", "pop", big, small, template.id
                ])

                ratio = pop_big / pop_small
                if ratio < 5:
                    diff = Difficulty.HARD.value
                elif ratio < 20:
                    diff = Difficulty.MEDIUM.value
                else:
                    diff = Difficulty.EASY.value

                country_counts[a] = country_counts.get(a, 0) + 1
                country_counts[b] = country_counts.get(b, 0) + 1

                yield ContrastivePair(
                    true_statement=true_stmt,
                    false_statement=false_stmt,
                    pair_id=pair_id,
                    domain="geography",
                    relation_type="population_greater",
                    difficulty=diff,
                    semantic_distance=None,
                    generator=self.name,
                    template_id=template.id,
                    negation_strategy=NegationStrategy.REVERSE_RELATION.value,
                    gen_params={
                        "big": big,
                        "small": small,
                        "big_population": int(pop_big),
                        "small_population": int(pop_small),
                        "ratio": round(ratio, 2),
                    },
                )

    # --- Area magnitude ---

    def _generate_area(self) -> Iterator[ContrastivePair]:
        """Country A is larger in area than Country B."""
        countries = self._countries
        area_map = dict(zip(countries["_name"], countries["_area_km2"]))
        valid_idx = [
            i for i, n in enumerate(countries["_name"])
            if pd.notna(area_map.get(n)) and area_map[n] > 0
        ]
        ordered = self.weighted_sample(valid_idx, self._country_weights, n=len(valid_idx))
        names = [countries["_name"].iloc[i] for i in ordered]

        templates = _ALL_TEMPLATES["area_greater"]
        # Cap per-country appearances to prevent tiny countries from dominating
        country_counts: dict[str, int] = {}
        max_per_country = 5

        for i, a in enumerate(names):
            for j in range(i + 1, len(names)):
                b = names[j]
                area_a, area_b = area_map[a], area_map[b]
                big, small = (a, b) if area_a > area_b else (b, a)
                area_big, area_small = max(area_a, area_b), min(area_a, area_b)

                if area_small <= 0 or area_big / area_small < self.magnitude_ratio:
                    continue

                # Enforce per-country cap
                if country_counts.get(a, 0) >= max_per_country:
                    break  # a is saturated, skip rest of inner loop
                if country_counts.get(b, 0) >= max_per_country:
                    continue

                template = templates[self.rng.randint(0, len(templates) - 1)]
                true_stmt = render_template(template.pattern,big=big, small=small)
                false_stmt = render_template(template.pattern,big=small, small=big)
                pair_id = _make_pair_id([
                    "geo", "area", big, small, template.id
                ])

                ratio = area_big / area_small
                if ratio < 5:
                    diff = Difficulty.HARD.value
                elif ratio < 20:
                    diff = Difficulty.MEDIUM.value
                else:
                    diff = Difficulty.EASY.value

                country_counts[a] = country_counts.get(a, 0) + 1
                country_counts[b] = country_counts.get(b, 0) + 1

                yield ContrastivePair(
                    true_statement=true_stmt,
                    false_statement=false_stmt,
                    pair_id=pair_id,
                    domain="geography",
                    relation_type="area_greater",
                    difficulty=diff,
                    semantic_distance=None,
                    generator=self.name,
                    template_id=template.id,
                    negation_strategy=NegationStrategy.REVERSE_RELATION.value,
                    gen_params={
                        "big": big,
                        "small": small,
                        "big_area_km2": round(area_big, 1),
                        "small_area_km2": round(area_small, 1),
                        "ratio": round(ratio, 2),
                    },
                )
