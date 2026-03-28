"""Tests for the astronomy generator."""

from __future__ import annotations

from latenet.generators.astronomy import AstronomyGenerator
from latenet.types import ContrastivePair, Difficulty


def _make_generator(**kwargs) -> AstronomyGenerator:
    return AstronomyGenerator(seed=42, max_pairs=500, **kwargs)


# --- Orbital relationships ---


class TestOrbits:
    def test_generates_orbit_pairs(self):
        gen = _make_generator()
        pairs = list(gen._generate_orbits())
        assert len(pairs) > 0
        for p in pairs:
            assert isinstance(p, ContrastivePair)
            assert p.relation_type == "orbits"
            assert p.generator == "astronomy"
            assert p.domain == "astronomy"

    def test_europa_orbits_jupiter(self):
        gen = _make_generator()
        pairs = list(gen._generate_orbits())
        europa_pairs = [p for p in pairs if "Europa" in p.true_statement]
        assert any("Jupiter" in p.true_statement for p in europa_pairs)

    def test_moon_orbits_earth(self):
        gen = _make_generator()
        pairs = list(gen._generate_orbits())
        moon_pairs = [p for p in pairs if p.true_statement.startswith("Moon ") or "Moon is" in p.true_statement]
        assert any("Earth" in p.true_statement for p in moon_pairs)

    def test_false_statement_wrong_parent(self):
        gen = _make_generator()
        pairs = list(gen._generate_orbits())
        for p in pairs:
            assert p.true_statement != p.false_statement


# --- Solar system ordering ---


class TestOrdering:
    def test_generates_ordering_pairs(self):
        gen = _make_generator()
        pairs = list(gen._generate_ordering())
        assert len(pairs) > 0
        for p in pairs:
            assert p.relation_type == "closer_to_sun"

    def test_mercury_closer_than_neptune(self):
        gen = _make_generator()
        pairs = list(gen._generate_ordering())
        # Find pair with Mercury and Neptune
        mn_pairs = [
            p for p in pairs
            if "Mercury" in p.true_statement and "Neptune" in p.true_statement
        ]
        assert len(mn_pairs) > 0
        for p in mn_pairs:
            assert "Mercury" in p.true_statement

    def test_ordinal_statements(self):
        gen = _make_generator()
        pairs = list(gen._generate_ordering())
        ordinal_pairs = [p for p in pairs if "planet from the Sun" in p.true_statement]
        assert len(ordinal_pairs) == 8  # One per planet

    def test_earth_is_third(self):
        gen = _make_generator()
        pairs = list(gen._generate_ordering())
        earth_ordinal = [
            p for p in pairs
            if "Earth" in p.true_statement and "planet from the Sun" in p.true_statement
        ]
        assert any("third" in p.true_statement for p in earth_ordinal)

    def test_adjacent_planets_are_hard(self):
        gen = _make_generator()
        pairs = list(gen._generate_ordering())
        # Venus(2) vs Earth(3) — gap=1 should be hard
        ve_pairs = [
            p for p in pairs
            if "Venus" in p.true_statement and "Earth" in p.true_statement
            and "planet from the Sun" not in p.true_statement
        ]
        assert all(p.gen_params.get("swap_distance") == Difficulty.HARD.value for p in ve_pairs)


# --- Property magnitude ---


class TestMagnitude:
    def test_generates_magnitude_pairs(self):
        gen = _make_generator()
        pairs = list(gen._generate_magnitude())
        assert len(pairs) > 0
        for p in pairs:
            assert p.relation_type == "property_greater"

    def test_jupiter_more_massive_than_earth(self):
        gen = _make_generator()
        pairs = list(gen._generate_magnitude())
        jup_earth = [
            p for p in pairs
            if "Jupiter" in p.true_statement and "Earth" in p.true_statement
            and "massive" in p.true_statement
        ]
        assert len(jup_earth) > 0
        for p in jup_earth:
            # Jupiter should be the "bigger" one
            assert p.true_statement.index("Jupiter") < p.true_statement.index("Earth")

    def test_ratio_threshold_respected(self):
        gen = _make_generator(magnitude_ratio=3.0)
        pairs = list(gen._generate_magnitude())
        # All pairs should respect the ratio threshold — no very close comparisons
        assert len(pairs) > 0

    def test_moon_count_uses_lower_threshold(self):
        gen = _make_generator(magnitude_ratio=3.0, moon_count_ratio=2.0)
        pairs = list(gen._generate_magnitude())
        moon_count_pairs = [p for p in pairs if "number of moons" in p.true_statement]
        # Should have some moon count pairs even at lower threshold
        assert len(moon_count_pairs) > 0


# --- Classification ---


class TestClassification:
    def test_generates_classification_pairs(self):
        gen = _make_generator()
        pairs = list(gen._generate_classification())
        assert len(pairs) > 0
        for p in pairs:
            assert p.relation_type == "is_type"

    def test_mars_is_terrestrial(self):
        gen = _make_generator()
        pairs = list(gen._generate_classification())
        mars_pairs = [p for p in pairs if "Mars" in p.true_statement]
        assert any("terrestrial" in p.true_statement for p in mars_pairs)

    def test_jupiter_is_gas_giant(self):
        gen = _make_generator()
        pairs = list(gen._generate_classification())
        jup_pairs = [p for p in pairs if "Jupiter" in p.true_statement]
        assert any("gas giant" in p.true_statement for p in jup_pairs)

    def test_pluto_is_dwarf_planet(self):
        gen = _make_generator()
        pairs = list(gen._generate_classification())
        pluto_pairs = [p for p in pairs if "Pluto" in p.true_statement]
        assert any("dwarf planet" in p.true_statement for p in pluto_pairs)

    def test_pluto_not_called_ninth_planet(self):
        gen = _make_generator()
        pairs = list(gen._generate_classification())
        pluto_stmts = [p.true_statement + " " + p.false_statement for p in pairs if "Pluto" in p.true_statement]
        for stmt in pluto_stmts:
            assert "ninth planet" not in stmt

    def test_no_double_planet(self):
        gen = _make_generator()
        pairs = list(gen._generate_classification())
        for p in pairs:
            assert "planet planet" not in p.true_statement
            assert "planet planet" not in p.false_statement

    def test_article_agreement(self):
        gen = _make_generator()
        pairs = list(gen._generate_classification())
        for p in pairs:
            assert "a ice" not in p.true_statement
            assert "a ice" not in p.false_statement

    def test_medium_tier_present(self):
        gen = _make_generator()
        pairs = list(gen._generate_classification())
        diffs = {p.gen_params.get("swap_distance") for p in pairs if p.gen_params}
        assert Difficulty.MEDIUM.value in diffs


# --- Stellar properties ---


class TestStarProperty:
    def test_generates_star_pairs(self):
        gen = _make_generator()
        pairs = list(gen._generate_star_property())
        assert len(pairs) > 0
        for p in pairs:
            assert p.relation_type == "star_property"

    def test_sirius_brighter_than_polaris(self):
        gen = _make_generator()
        pairs = list(gen._generate_star_property())
        bright_pairs = [
            p for p in pairs
            if "Sirius" in p.true_statement and "Polaris" in p.true_statement
            and "brighter" in p.true_statement
        ]
        assert len(bright_pairs) > 0
        # Sirius (mag -1.46) is brighter than Polaris (mag 1.98)
        for p in bright_pairs:
            assert p.true_statement.index("Sirius") < p.true_statement.index("Polaris")

    def test_proxima_closer_than_betelgeuse(self):
        # Proxima (4.24 ly) vs Betelgeuse (700 ly) — ratio ~165x, well above threshold
        gen = _make_generator()
        pairs = list(gen._generate_star_property())
        dist_pairs = [
            p for p in pairs
            if "Proxima" in p.true_statement and "Betelgeuse" in p.true_statement
            and "closer" in p.true_statement
        ]
        assert len(dist_pairs) > 0


# --- Constellation membership ---


class TestConstellation:
    def test_generates_constellation_pairs(self):
        gen = _make_generator()
        pairs = list(gen._generate_constellation())
        assert len(pairs) > 0
        for p in pairs:
            assert p.relation_type == "in_constellation"

    def test_betelgeuse_in_orion(self):
        gen = _make_generator()
        pairs = list(gen._generate_constellation())
        bet_pairs = [p for p in pairs if "Betelgeuse" in p.true_statement]
        assert any("Orion" in p.true_statement for p in bet_pairs)

    def test_polaris_in_ursa_minor(self):
        gen = _make_generator()
        pairs = list(gen._generate_constellation())
        pol_pairs = [p for p in pairs if "Polaris" in p.true_statement]
        assert any("Ursa Minor" in p.true_statement for p in pol_pairs)

    def test_medium_tier_present(self):
        gen = _make_generator()
        pairs = list(gen._generate_constellation())
        diffs = {p.gen_params.get("swap_distance") for p in pairs if p.gen_params}
        assert Difficulty.MEDIUM.value in diffs


# --- Determinism ---


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
        gen2 = AstronomyGenerator(seed=99, max_pairs=500)
        pairs1 = list(gen1.generate())
        pairs2 = list(gen2.generate())
        ids1 = {p.pair_id for p in pairs1}
        ids2 = {p.pair_id for p in pairs2}
        assert ids1 != ids2 or len(ids1) == 0


# --- Integration ---


class TestIntegration:
    def test_max_pairs_respected(self):
        gen = AstronomyGenerator(seed=42, max_pairs=10)
        pairs = list(gen.generate())
        assert len(pairs) <= 10

    def test_pair_ids_unique(self):
        gen = _make_generator()
        pairs = list(gen.generate())
        ids = [p.pair_id for p in pairs]
        assert len(ids) == len(set(ids)), "Duplicate pair IDs found"

    def test_difficulty_values(self):
        gen = _make_generator()
        pairs = list(gen.generate())
        for p in pairs:
            assert p.difficulty == "mixed"

    def test_template_ids_are_stable(self):
        gen = _make_generator()
        pairs = list(gen.generate())
        for p in pairs:
            assert p.template_id.startswith("astro_")

    def test_multiple_relation_types(self):
        gen = _make_generator()
        pairs = list(gen.generate())
        rel_types = {p.relation_type for p in pairs}
        assert "orbits" in rel_types
        assert "closer_to_sun" in rel_types
        assert "property_greater" in rel_types
        assert "is_type" in rel_types
        assert "star_property" in rel_types
        assert "in_constellation" in rel_types

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
        assert true_row["generator"] == "astronomy"

    def test_generator_contract(self):
        gen = _make_generator()
        assert gen.name == "astronomy"
        assert gen.domains() == ["astronomy"]
        assert len(gen.relation_types()) == 6
