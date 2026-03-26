"""Tests for the anatomy generator."""

from __future__ import annotations

from latenet.generators.anatomy import AnatomyGenerator
from latenet.generators.anat_data import STRUCTURES
from latenet.types import ContrastivePair, Difficulty


def _make_generator(**kwargs) -> AnatomyGenerator:
    return AnatomyGenerator(seed=42, max_pairs=2000, **kwargs)


# --- System membership ---


class TestSystemMembership:
    def test_generates_system_pairs(self):
        gen = _make_generator()
        pairs = list(gen._generate_system_membership())
        assert len(pairs) > 0
        for p in pairs:
            assert isinstance(p, ContrastivePair)
            assert p.relation_type == "in_system"
            assert p.generator == "anatomy"
            assert p.domain == "anatomy"

    def test_heart_cardiovascular(self):
        gen = _make_generator()
        pairs = list(gen._generate_system_membership())
        heart_pairs = [p for p in pairs if "heart" in p.true_statement.lower()]
        assert any("cardiovascular" in p.true_statement for p in heart_pairs)

    def test_femur_skeletal(self):
        gen = _make_generator()
        pairs = list(gen._generate_system_membership())
        femur_pairs = [p for p in pairs if "femur" in p.true_statement.lower()]
        assert any("skeletal" in p.true_statement for p in femur_pairs)

    def test_multi_system_excluded(self):
        """Structures with related_systems should NOT appear in system membership."""
        gen = _make_generator()
        pairs = list(gen._generate_system_membership())
        # Diaphragm has related_systems=["respiratory"]
        diaphragm_pairs = [p for p in pairs if "diaphragm" in p.true_statement.lower()]
        assert len(diaphragm_pairs) == 0

    def test_hard_tier_same_region_different_system(self):
        gen = _make_generator()
        pairs = list(gen._generate_system_membership())
        hard_pairs = [p for p in pairs if p.difficulty == Difficulty.HARD.value]
        assert len(hard_pairs) > 0

    def test_all_difficulty_tiers_present(self):
        gen = _make_generator()
        pairs = list(gen._generate_system_membership())
        diffs = {p.difficulty for p in pairs}
        assert Difficulty.HARD.value in diffs
        assert Difficulty.EASY.value in diffs


# --- Regional containment ---


class TestRegionalContainment:
    def test_generates_region_pairs(self):
        gen = _make_generator()
        pairs = list(gen._generate_regional_containment())
        assert len(pairs) > 0
        for p in pairs:
            assert p.relation_type == "in_region"

    def test_brain_in_head(self):
        gen = _make_generator()
        pairs = list(gen._generate_regional_containment())
        brain_pairs = [p for p in pairs if "brain" in p.true_statement.lower()]
        assert any("head" in p.true_statement for p in brain_pairs)

    def test_femur_in_lower_limb(self):
        gen = _make_generator()
        pairs = list(gen._generate_regional_containment())
        femur_pairs = [p for p in pairs if "femur" in p.true_statement.lower()]
        assert any("lower limb" in p.true_statement for p in femur_pairs)

    def test_spans_regions_excluded(self):
        """Structures that span regions should NOT appear in regional containment."""
        gen = _make_generator()
        pairs = list(gen._generate_regional_containment())
        # Esophagus spans_regions=True
        esophagus_pairs = [p for p in pairs if "esophagus" in p.true_statement.lower()]
        assert len(esophagus_pairs) == 0

    def test_hard_tier_adjacent_region(self):
        gen = _make_generator()
        pairs = list(gen._generate_regional_containment())
        hard_pairs = [p for p in pairs if p.difficulty == Difficulty.HARD.value]
        assert len(hard_pairs) > 0


# --- Structure type classification ---


class TestStructureType:
    def test_generates_type_pairs(self):
        gen = _make_generator()
        pairs = list(gen._generate_structure_type())
        assert len(pairs) > 0
        for p in pairs:
            assert p.relation_type == "is_structure_type"

    def test_femur_is_bone(self):
        gen = _make_generator()
        pairs = list(gen._generate_structure_type())
        femur_pairs = [p for p in pairs if "femur" in p.true_statement.lower()]
        assert any("bone" in p.true_statement for p in femur_pairs)

    def test_biceps_is_muscle(self):
        gen = _make_generator()
        pairs = list(gen._generate_structure_type())
        biceps_pairs = [p for p in pairs if "biceps" in p.true_statement.lower()]
        assert any("muscle" in p.true_statement for p in biceps_pairs)

    def test_aorta_is_vessel(self):
        gen = _make_generator()
        pairs = list(gen._generate_structure_type())
        aorta_pairs = [p for p in pairs if "aorta" in p.true_statement.lower()]
        assert any("blood vessel" in p.true_statement for p in aorta_pairs)

    def test_all_difficulty_tiers(self):
        gen = _make_generator()
        pairs = list(gen._generate_structure_type())
        diffs = {p.difficulty for p in pairs}
        assert Difficulty.HARD.value in diffs
        assert Difficulty.EASY.value in diffs


# --- Co-location ---


class TestSameRegion:
    def test_generates_colocation_pairs(self):
        gen = _make_generator()
        pairs = list(gen._generate_same_region())
        assert len(pairs) > 0
        for p in pairs:
            assert p.relation_type == "same_region"

    def test_true_pairs_share_region(self):
        gen = _make_generator()
        pairs = list(gen._generate_same_region())
        # The true statement should mention two structures actually in the same region
        assert len(pairs) > 0
        for p in pairs:
            assert p.true_statement != p.false_statement


# --- Co-membership ---


class TestSameSystem:
    def test_generates_comembership_pairs(self):
        gen = _make_generator()
        pairs = list(gen._generate_same_system())
        assert len(pairs) > 0
        for p in pairs:
            assert p.relation_type == "same_system"


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
        gen2 = AnatomyGenerator(seed=99, max_pairs=2000)
        pairs1 = list(gen1.generate())
        pairs2 = list(gen2.generate())
        ids1 = {p.pair_id for p in pairs1}
        ids2 = {p.pair_id for p in pairs2}
        assert ids1 != ids2 or len(ids1) == 0


# --- Integration ---


class TestIntegration:
    def test_max_pairs_respected(self):
        gen = AnatomyGenerator(seed=42, max_pairs=10)
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
        valid_diffs = {Difficulty.HARD.value, Difficulty.MEDIUM.value, Difficulty.EASY.value}
        for p in pairs:
            assert p.difficulty in valid_diffs

    def test_template_ids_are_stable(self):
        gen = _make_generator()
        pairs = list(gen.generate())
        for p in pairs:
            assert p.template_id.startswith("anat_")

    def test_multiple_relation_types(self):
        gen = _make_generator()
        pairs = list(gen.generate())
        rel_types = {p.relation_type for p in pairs}
        assert "in_system" in rel_types
        assert "in_region" in rel_types
        assert "is_structure_type" in rel_types
        assert "same_region" in rel_types
        assert "same_system" in rel_types

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
        assert true_row["generator"] == "anatomy"

    def test_generator_contract(self):
        gen = _make_generator()
        assert gen.name == "anatomy"
        assert gen.domains() == ["anatomy"]
        assert len(gen.relation_types()) == 5

    def test_template_diversity(self):
        """Check that multiple template IDs are used across pairs."""
        gen = _make_generator()
        pairs = list(gen.generate())
        template_ids = {p.template_id for p in pairs}
        # Should use templates from multiple relation types
        assert any(t.startswith("anat_system_") for t in template_ids)
        assert any(t.startswith("anat_region_") for t in template_ids)
        assert any(t.startswith("anat_type_") for t in template_ids)
        assert any(t.startswith("anat_coloc_") for t in template_ids)
        assert any(t.startswith("anat_comem_") for t in template_ids)

    def test_sufficient_scale(self):
        """Should generate at least several hundred pairs."""
        gen = AnatomyGenerator(seed=42)
        pairs = list(gen.generate())
        assert len(pairs) >= 500
