"""Tests for the chemistry generator."""

from __future__ import annotations

import pandas as pd

from latenet.generators.chemistry import (
    ChemistryGenerator,
    _state_at_room_temp,
)
from latenet.types import ContrastivePair, Difficulty


def _mock_elements() -> pd.DataFrame:
    """Small set of fake elements with known properties."""
    rows = [
        {
            "atomic_number": 1, "symbol": "H", "name": "Hydrogen",
            "group_id": 1.0, "period": 1, "block": "s", "series_id": 1,
            "atomic_weight": 1.008, "en_pauling": 2.20, "density": 0.00009,
            "is_radioactive": 0,
            "melting_point": 14.01, "boiling_point": 20.28,
            "state_rt": "gas",
            "_series_name": "Nonmetals", "series_label": "nonmetal",
        },
        {
            "atomic_number": 2, "symbol": "He", "name": "Helium",
            "group_id": 18.0, "period": 1, "block": "s", "series_id": 2,
            "atomic_weight": 4.003, "en_pauling": None, "density": 0.000179,
            "is_radioactive": 0,
            "melting_point": 0.95, "boiling_point": 4.22,
            "state_rt": "gas",
            "_series_name": "Noble gases", "series_label": "noble gase",
        },
        {
            "atomic_number": 11, "symbol": "Na", "name": "Sodium",
            "group_id": 1.0, "period": 3, "block": "s", "series_id": 3,
            "atomic_weight": 22.990, "en_pauling": 0.93, "density": 0.968,
            "is_radioactive": 0,
            "melting_point": 370.95, "boiling_point": 1156.09,
            "state_rt": "solid",
            "_series_name": "Alkali metals", "series_label": "alkali metal",
        },
        {
            "atomic_number": 26, "symbol": "Fe", "name": "Iron",
            "group_id": 8.0, "period": 4, "block": "d", "series_id": 8,
            "atomic_weight": 55.845, "en_pauling": 1.83, "density": 7.874,
            "is_radioactive": 0,
            "melting_point": 1811.15, "boiling_point": 3134.15,
            "state_rt": "solid",
            "_series_name": "Transition metals", "series_label": "transition metal",
        },
        {
            "atomic_number": 29, "symbol": "Cu", "name": "Copper",
            "group_id": 11.0, "period": 4, "block": "d", "series_id": 8,
            "atomic_weight": 63.546, "en_pauling": 1.90, "density": 8.96,
            "is_radioactive": 0,
            "melting_point": 1357.77, "boiling_point": 2835.15,
            "state_rt": "solid",
            "_series_name": "Transition metals", "series_label": "transition metal",
        },
        {
            "atomic_number": 79, "symbol": "Au", "name": "Gold",
            "group_id": 11.0, "period": 6, "block": "d", "series_id": 8,
            "atomic_weight": 196.967, "en_pauling": 2.40, "density": 19.3,
            "is_radioactive": 0,
            "melting_point": 1337.33, "boiling_point": 3109.15,
            "state_rt": "solid",
            "_series_name": "Transition metals", "series_label": "transition metal",
        },
        {
            "atomic_number": 80, "symbol": "Hg", "name": "Mercury",
            "group_id": 12.0, "period": 6, "block": "d", "series_id": 7,
            "atomic_weight": 200.592, "en_pauling": 1.90, "density": 13.534,
            "is_radioactive": 0,
            "melting_point": 234.32, "boiling_point": 629.77,
            "state_rt": "liquid",
            "_series_name": "Poor metals", "series_label": "poor metal",
        },
        {
            "atomic_number": 9, "symbol": "F", "name": "Fluorine",
            "group_id": 17.0, "period": 2, "block": "p", "series_id": 6,
            "atomic_weight": 18.998, "en_pauling": 3.98, "density": 0.001696,
            "is_radioactive": 0,
            "melting_point": 53.48, "boiling_point": 85.03,
            "state_rt": "gas",
            "_series_name": "Halogens", "series_label": "halogen",
        },
    ]
    return pd.DataFrame(rows)


def _patch_data(gen: ChemistryGenerator) -> None:
    """Patch the generator's data loading to use mock data."""
    gen._elements = _mock_elements()
    gen._series_map = {
        1: "Nonmetals", 2: "Noble gases", 3: "Alkali metals",
        6: "Halogens", 7: "Poor metals", 8: "Transition metals",
    }


def _make_generator(**kwargs) -> ChemistryGenerator:
    gen = ChemistryGenerator(seed=42, max_pairs=200, **kwargs)
    _patch_data(gen)
    return gen


# --- Tests ---


class TestStateAtRoomTemp:
    def test_solid(self):
        assert _state_at_room_temp(1811.0, 3134.0) == "solid"

    def test_liquid(self):
        assert _state_at_room_temp(234.0, 630.0) == "liquid"

    def test_gas(self):
        assert _state_at_room_temp(14.0, 20.0) == "gas"

    def test_none_melting_point(self):
        assert _state_at_room_temp(None, 100.0) is None

    def test_none_boiling_point(self):
        assert _state_at_room_temp(100.0, None) is None


class TestSymbol:
    def test_generates_symbol_pairs(self):
        gen = _make_generator()
        pairs = list(gen._generate_symbol())
        assert len(pairs) > 0
        for p in pairs:
            assert isinstance(p, ContrastivePair)
            assert p.relation_type == "symbol_of"
            assert p.generator == "chemistry"
            assert p.domain == "chemistry"

    def test_true_statement_correct_symbol(self):
        gen = _make_generator()
        pairs = list(gen._generate_symbol())
        gold_pairs = [p for p in pairs if "Gold" in p.true_statement]
        assert any("Au" in p.true_statement for p in gold_pairs)

    def test_false_statement_wrong_symbol(self):
        gen = _make_generator()
        pairs = list(gen._generate_symbol())
        for p in pairs:
            assert p.true_statement != p.false_statement


class TestGroupMembership:
    def test_generates_group_pairs(self):
        gen = _make_generator()
        pairs = list(gen._generate_group())
        assert len(pairs) > 0
        for p in pairs:
            assert p.relation_type == "member_of_group"

    def test_sodium_is_alkali_metal(self):
        gen = _make_generator()
        pairs = list(gen._generate_group())
        sodium_pairs = [p for p in pairs if "Sodium" in p.true_statement]
        assert any("alkali metal" in p.true_statement for p in sodium_pairs)

    def test_false_has_different_series(self):
        gen = _make_generator()
        pairs = list(gen._generate_group())
        for p in pairs:
            assert p.true_statement != p.false_statement


class TestStateMatter:
    def test_generates_state_pairs(self):
        gen = _make_generator()
        pairs = list(gen._generate_state())
        assert len(pairs) > 0
        for p in pairs:
            assert p.relation_type == "state_at_room_temp"

    def test_mercury_is_liquid(self):
        gen = _make_generator()
        pairs = list(gen._generate_state())
        hg_pairs = [p for p in pairs if "Mercury" in p.true_statement]
        assert any("liquid" in p.true_statement for p in hg_pairs)

    def test_hydrogen_is_gas(self):
        gen = _make_generator()
        pairs = list(gen._generate_state())
        h_pairs = [p for p in pairs if "Hydrogen" in p.true_statement]
        assert any("gas" in p.true_statement for p in h_pairs)


class TestBlockMembership:
    def test_generates_block_pairs(self):
        gen = _make_generator()
        pairs = list(gen._generate_block())
        assert len(pairs) > 0
        for p in pairs:
            assert p.relation_type == "in_block"

    def test_iron_is_d_block(self):
        gen = _make_generator()
        pairs = list(gen._generate_block())
        fe_pairs = [p for p in pairs if "Iron" in p.true_statement]
        assert any("d" in p.true_statement for p in fe_pairs)

    def test_false_has_different_block(self):
        gen = _make_generator()
        pairs = list(gen._generate_block())
        for p in pairs:
            assert p.true_statement != p.false_statement


class TestAtomicNumber:
    def test_generates_atomic_number_pairs(self):
        gen = _make_generator()
        pairs = list(gen._generate_atomic_number())
        assert len(pairs) > 0
        for p in pairs:
            assert p.relation_type == "atomic_number_greater"

    def test_ratio_threshold(self):
        gen = _make_generator(magnitude_ratio=3.0)
        pairs = list(gen._generate_atomic_number())
        for p in pairs:
            assert p.difficulty == "mixed"


class TestPropertyComparison:
    def test_generates_property_pairs(self):
        gen = _make_generator()
        pairs = list(gen._generate_property())
        assert len(pairs) > 0
        for p in pairs:
            assert p.relation_type == "property_greater"

    def test_multiple_properties(self):
        gen = _make_generator()
        pairs = list(gen._generate_property())
        # Should have pairs for electronegativity, atomic weight, density
        stmts = " ".join(p.true_statement for p in pairs)
        assert "electronegativity" in stmts or "atomic weight" in stmts or "density" in stmts


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
        gen2 = ChemistryGenerator(seed=99, max_pairs=200)
        _patch_data(gen2)
        pairs1 = list(gen1.generate())
        pairs2 = list(gen2.generate())
        ids1 = {p.pair_id for p in pairs1}
        ids2 = {p.pair_id for p in pairs2}
        assert ids1 != ids2 or len(ids1) == 0


class TestTemplateDiversity:
    def test_multiple_templates_used(self):
        gen = _make_generator()
        pairs = list(gen.generate())
        template_ids = {p.template_id for p in pairs}
        assert len(template_ids) >= 3

    def test_template_ids_are_stable(self):
        gen = _make_generator()
        pairs = list(gen.generate())
        for p in pairs:
            assert p.template_id.startswith("chem_")


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
        assert true_row["generator"] == "chemistry"


class TestEdgeCases:
    def test_max_pairs_respected(self):
        gen = ChemistryGenerator(seed=42, max_pairs=5)
        _patch_data(gen)
        pairs = list(gen.generate())
        assert len(pairs) <= 5

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
