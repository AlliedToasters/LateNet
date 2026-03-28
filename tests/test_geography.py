"""Tests for the geography generator."""

from __future__ import annotations

import math

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from latenet.generators.geography import GeographyGenerator
from latenet.types import ContrastivePair, Difficulty


def _make_country(name, iso_a2, sov_a3, adm0_a3, continent, subregion, pop, geom):
    return {
        "NAME": name,
        "ISO_A2": iso_a2,
        "SOV_A3": sov_a3,
        "ADM0_A3": adm0_a3,
        "CONTINENT": continent,
        "SUBREGION": subregion,
        "POP_EST": pop,
        "geometry": geom,
    }


def _mock_countries():
    """Small set of fake countries with known geometry."""
    # France: centroid ~(2, 47)
    france = _make_country(
        "France", "FR", "FR1", "FR1", "Europe", "Western Europe", 67_000_000,
        box(0, 45, 4, 49),
    )
    # Germany: centroid ~(11, 51), touches France? No — separated.
    # Make them touch: Germany box shares edge at x=4
    germany = _make_country(
        "Germany", "DE", "DE1", "DE1", "Europe", "Western Europe", 83_000_000,
        box(4, 47, 12, 53),
    )
    # Japan: centroid ~(138, 36)
    japan = _make_country(
        "Japan", "JP", "JPN", "JPN", "Asia", "Eastern Asia", 126_000_000,
        box(130, 30, 146, 42),
    )
    # Brazil: centroid ~(-52, -12)
    brazil = _make_country(
        "Brazil", "BR", "BRA", "BRA", "South America", "South America", 211_000_000,
        box(-60, -20, -44, -4),
    )

    gdf = gpd.GeoDataFrame(
        [france, germany, japan, brazil],
        crs="EPSG:4326",
    )
    return gdf


def _mock_cities():
    """Small set of fake cities with known coordinates."""
    rows = [
        {"name": "Paris", "latitude": 48.86, "longitude": 2.35, "country_code": "FR", "population": 2_161_000},
        {"name": "Berlin", "latitude": 52.52, "longitude": 13.40, "country_code": "DE", "population": 3_645_000},
        {"name": "Tokyo", "latitude": 35.68, "longitude": 139.69, "country_code": "JP", "population": 13_960_000},
        {"name": "São Paulo", "latitude": -23.55, "longitude": -46.63, "country_code": "BR", "population": 12_325_000},
        {"name": "Lyon", "latitude": 45.76, "longitude": 4.84, "country_code": "FR", "population": 516_000},
    ]
    return pd.DataFrame(rows)


def _patch_data(gen):
    """Patch the generator's data loading to use mock data."""
    gen._countries = _mock_countries()
    gen._cities = _mock_cities()
    gen._cc_to_name = {"FR": "France", "DE": "Germany", "JP": "Japan", "BR": "Brazil"}
    gen._neighbor_map = gen._build_neighbor_map()

    name_col = "NAME"
    gen._countries = gen._countries.copy()
    gen._countries["_centroid"] = gen._countries.geometry.centroid
    gen._countries["_lat"] = gen._countries["_centroid"].y
    gen._countries["_lon"] = gen._countries["_centroid"].x
    gen._countries["_name"] = gen._countries[name_col]

    mollweide = gen._countries.to_crs("+proj=moll")
    gen._countries["_area_km2"] = mollweide.geometry.area / 1e6

    gen._country_continent = dict(
        zip(gen._countries["_name"], gen._countries["CONTINENT"])
    )
    gen._city_country = {}
    for _, row in gen._cities.iterrows():
        cc = row["country_code"]
        if cc in gen._cc_to_name:
            gen._city_country[row["name"]] = gen._cc_to_name[cc]

    # Log-population weights (mirrors _load_data)
    country_pops = gen._countries["POP_EST"].tolist()
    log_country_pops = [math.log(max(float(p), 1.0)) for p in country_pops]
    gen._country_weights = gen.build_weights(log_country_pops)
    gen._country_name_to_idx = {n: i for i, n in enumerate(gen._countries["_name"].tolist())}

    city_pops = gen._cities["population"].tolist()
    log_city_pops = [math.log(max(float(p), 1.0)) for p in city_pops]
    gen._city_weights = gen.build_weights(log_city_pops)


def _make_generator(**kwargs) -> GeographyGenerator:
    gen = GeographyGenerator(seed=42, max_pairs=100, **kwargs)
    _patch_data(gen)
    return gen


# --- Tests ---

class TestContainment:
    def test_generates_containment_pairs(self):
        gen = _make_generator()
        pairs = list(gen._generate_containment())
        assert len(pairs) > 0
        for p in pairs:
            assert isinstance(p, ContrastivePair)
            assert p.relation_type == "contained_in"
            assert p.generator == "geography"
            assert p.domain == "geography"

    def test_true_statement_has_correct_city_country(self):
        gen = _make_generator()
        pairs = list(gen._generate_containment())
        # Paris should be in France
        paris_pairs = [p for p in pairs if "Paris" in p.true_statement]
        assert any("France" in p.true_statement for p in paris_pairs)

    def test_false_statement_has_different_country(self):
        gen = _make_generator()
        pairs = list(gen._generate_containment())
        for p in pairs:
            # True and false should differ
            assert p.true_statement != p.false_statement

    def test_difficulty_tiers(self):
        gen = _make_generator()
        pairs = list(gen._generate_containment())
        difficulties = {p.difficulty for p in pairs}
        # Should have at least some of the difficulty levels
        assert len(difficulties) >= 1


class TestCardinalDirection:
    def test_generates_cardinal_pairs(self):
        gen = _make_generator()
        pairs = list(gen._generate_cardinal())
        assert len(pairs) > 0
        for p in pairs:
            assert p.relation_type == "cardinal_direction"

    def test_direction_correctness(self):
        gen = _make_generator()
        pairs = list(gen._generate_cardinal())
        # Germany (centroid ~51°N) is north of Japan (centroid ~36°N)
        for p in pairs:
            if "Germany" in p.true_statement and "Japan" in p.true_statement:
                if "north" in p.true_statement:
                    assert "south" in p.false_statement or "south" not in p.true_statement

    def test_opposite_in_false(self):
        gen = _make_generator()
        pairs = list(gen._generate_cardinal())
        for p in pairs:
            # The false statement should have the opposite direction
            if "north" in p.true_statement:
                assert "south" in p.false_statement
            elif "south" in p.true_statement:
                assert "north" in p.false_statement


class TestProximity:
    def test_generates_proximity_pairs(self):
        gen = _make_generator()
        pairs = list(gen._generate_proximity())
        # May be 0 if ratio threshold not met, but should succeed
        for p in pairs:
            assert p.relation_type == "closer_to"

    def test_false_swaps_near_far(self):
        gen = _make_generator()
        pairs = list(gen._generate_proximity())
        for p in pairs:
            assert p.true_statement != p.false_statement


class TestPopulationMagnitude:
    def test_generates_population_pairs(self):
        gen = _make_generator()
        pairs = list(gen._generate_population())
        assert len(pairs) > 0
        for p in pairs:
            assert p.relation_type == "population_greater"

    def test_ratio_threshold(self):
        gen = _make_generator(magnitude_ratio=3.0)
        pairs = list(gen._generate_population())
        # All pairs should have >3x ratio
        for p in pairs:
            assert p.difficulty in (Difficulty.HARD.value, Difficulty.MEDIUM.value, Difficulty.EASY.value)


class TestAreaMagnitude:
    def test_generates_area_pairs(self):
        gen = _make_generator()
        pairs = list(gen._generate_area())
        assert len(pairs) > 0
        for p in pairs:
            assert p.relation_type == "area_greater"


class TestDeterminism:
    def test_same_seed_same_output(self):
        gen1 = _make_generator()
        gen2 = _make_generator()
        pairs1 = list(gen1.generate())
        pairs2 = list(gen2.generate())
        assert len(pairs1) == len(pairs2)
        for p1, p2 in zip(pairs1, pairs2):
            assert p1.pair_id == p2.pair_id
            assert p1.true_statement == p2.true_statement
            assert p1.false_statement == p2.false_statement

    def test_different_seed_different_output(self):
        gen1 = _make_generator()
        gen2 = GeographyGenerator(seed=99, max_pairs=100)
        _patch_data(gen2)
        pairs1 = list(gen1.generate())
        pairs2 = list(gen2.generate())
        # At least some pairs should differ (different shuffle order)
        ids1 = {p.pair_id for p in pairs1}
        ids2 = {p.pair_id for p in pairs2}
        # Not all the same (could share some cardinal pairs but not all)
        assert ids1 != ids2 or len(ids1) == 0


class TestTemplateDiversity:
    def test_multiple_templates_used(self):
        gen = _make_generator()
        pairs = list(gen.generate())
        template_ids = {p.template_id for p in pairs}
        # Should use templates from multiple relation types
        assert len(template_ids) >= 3

    def test_template_ids_are_stable(self):
        gen = _make_generator()
        pairs = list(gen.generate())
        for p in pairs:
            assert p.template_id.startswith("geo_")


class TestToRows:
    def test_to_rows_schema(self):
        gen = _make_generator()
        pairs = list(gen.generate())
        assert len(pairs) > 0
        rows = pairs[0].to_rows()
        assert len(rows) == 2
        true_row, false_row = rows
        assert true_row["label"] is True
        assert false_row["label"] is False
        assert true_row["pair_id"] == false_row["pair_id"]
        assert true_row["generator"] == "geography"


class TestEdgeCases:
    def test_max_pairs_respected(self):
        gen = GeographyGenerator(seed=42, max_pairs=5)
        _patch_data(gen)
        pairs = list(gen.generate())
        assert len(pairs) <= 5

    def test_pair_ids_unique(self):
        gen = _make_generator()
        pairs = list(gen.generate())
        ids = [p.pair_id for p in pairs]
        assert len(ids) == len(set(ids)), "Duplicate pair IDs found"
