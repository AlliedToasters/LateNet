"""Tests for the biology generator."""

from __future__ import annotations

import pandas as pd

from latenet.datasources.wikidata import RANK_LEVEL, RANK_ORDER
from latenet.generators.biology import (
    BiologyGenerator,
    _lowest_common_rank,
    _make_pair_id,
    _taxonomic_distance,
)
from latenet.types import ContrastivePair, Difficulty


def _mock_organisms() -> pd.DataFrame:
    """Small set of mock organisms with known taxonomy for testing."""
    rows = [
        {
            "qid": "Q144", "name": "Canis lupus familiaris", "common_name": "dog",
            "taxon_rank": "species", "parent_taxon_qid": "Q_canis",
            "display_name": "dog",
            "genus": "Canis", "family": "Canidae", "order": "Carnivora",
            "class_": "Mammalia", "phylum": "Chordata", "kingdom": "Animalia",
            "genus_qid": "Q_canis", "family_qid": "Q_canidae",
            "order_qid": "Q_carnivora", "class__qid": "Q_mammalia",
            "phylum_qid": "Q_chordata", "kingdom_qid": "Q_animalia",
            "ncbi_taxon_id": "9615", "article": "https://en.wikipedia.org/wiki/Dog",
            "has_wikipedia": True,
        },
        {
            "qid": "Q18498", "name": "Vulpes vulpes", "common_name": "red fox",
            "taxon_rank": "species", "parent_taxon_qid": "Q_vulpes",
            "display_name": "red fox",
            "genus": "Vulpes", "family": "Canidae", "order": "Carnivora",
            "class_": "Mammalia", "phylum": "Chordata", "kingdom": "Animalia",
            "genus_qid": "Q_vulpes", "family_qid": "Q_canidae",
            "order_qid": "Q_carnivora", "class__qid": "Q_mammalia",
            "phylum_qid": "Q_chordata", "kingdom_qid": "Q_animalia",
            "ncbi_taxon_id": "9627", "article": "https://en.wikipedia.org/wiki/Red_fox",
            "has_wikipedia": True,
        },
        {
            "qid": "Q146", "name": "Felis catus", "common_name": "cat",
            "taxon_rank": "species", "parent_taxon_qid": "Q_felis",
            "display_name": "cat",
            "genus": "Felis", "family": "Felidae", "order": "Carnivora",
            "class_": "Mammalia", "phylum": "Chordata", "kingdom": "Animalia",
            "genus_qid": "Q_felis", "family_qid": "Q_felidae",
            "order_qid": "Q_carnivora", "class__qid": "Q_mammalia",
            "phylum_qid": "Q_chordata", "kingdom_qid": "Q_animalia",
            "ncbi_taxon_id": "9685", "article": "https://en.wikipedia.org/wiki/Cat",
            "has_wikipedia": True,
        },
        {
            "qid": "Q23390", "name": "Equus caballus", "common_name": "horse",
            "taxon_rank": "species", "parent_taxon_qid": "Q_equus",
            "display_name": "horse",
            "genus": "Equus", "family": "Equidae", "order": "Perissodactyla",
            "class_": "Mammalia", "phylum": "Chordata", "kingdom": "Animalia",
            "genus_qid": "Q_equus", "family_qid": "Q_equidae",
            "order_qid": "Q_perissodactyla", "class__qid": "Q_mammalia",
            "phylum_qid": "Q_chordata", "kingdom_qid": "Q_animalia",
            "ncbi_taxon_id": "9796", "article": "https://en.wikipedia.org/wiki/Horse",
            "has_wikipedia": True,
        },
        {
            "qid": "Q780", "name": "Gallus gallus domesticus", "common_name": "chicken",
            "taxon_rank": "species", "parent_taxon_qid": "Q_gallus",
            "display_name": "chicken",
            "genus": "Gallus", "family": "Phasianidae", "order": "Galliformes",
            "class_": "Aves", "phylum": "Chordata", "kingdom": "Animalia",
            "genus_qid": "Q_gallus", "family_qid": "Q_phasianidae",
            "order_qid": "Q_galliformes", "class__qid": "Q_aves",
            "phylum_qid": "Q_chordata", "kingdom_qid": "Q_animalia",
            "ncbi_taxon_id": "9031", "article": "https://en.wikipedia.org/wiki/Chicken",
            "has_wikipedia": True,
        },
        {
            "qid": "Q5113", "name": "Columba livia", "common_name": "pigeon",
            "taxon_rank": "species", "parent_taxon_qid": "Q_columba",
            "display_name": "pigeon",
            "genus": "Columba", "family": "Columbidae", "order": "Columbiformes",
            "class_": "Aves", "phylum": "Chordata", "kingdom": "Animalia",
            "genus_qid": "Q_columba", "family_qid": "Q_columbidae",
            "order_qid": "Q_columbiformes", "class__qid": "Q_aves",
            "phylum_qid": "Q_chordata", "kingdom_qid": "Q_animalia",
            "ncbi_taxon_id": "8932", "article": "https://en.wikipedia.org/wiki/Pigeon",
            "has_wikipedia": True,
        },
        {
            "qid": "Q128685", "name": "Agaricus bisporus", "common_name": "button mushroom",
            "taxon_rank": "species", "parent_taxon_qid": "Q_agaricus",
            "display_name": "button mushroom",
            "genus": "Agaricus", "family": "Agaricaceae", "order": "Agaricales",
            "class_": "Agaricomycetes", "phylum": "Basidiomycota", "kingdom": "Fungi",
            "genus_qid": "Q_agaricus", "family_qid": "Q_agaricaceae",
            "order_qid": "Q_agaricales", "class__qid": "Q_agaricomycetes",
            "phylum_qid": "Q_basidiomycota", "kingdom_qid": "Q_fungi",
            "ncbi_taxon_id": "5341", "article": "https://en.wikipedia.org/wiki/Agaricus_bisporus",
            "has_wikipedia": True,
        },
        {
            "qid": "Q83093", "name": "Cantharellus cibarius", "common_name": "chanterelle",
            "taxon_rank": "species", "parent_taxon_qid": "Q_cantharellus",
            "display_name": "chanterelle",
            "genus": "Cantharellus", "family": "Cantharellaceae", "order": "Cantharellales",
            "class_": "Agaricomycetes", "phylum": "Basidiomycota", "kingdom": "Fungi",
            "genus_qid": "Q_cantharellus", "family_qid": "Q_cantharellaceae",
            "order_qid": "Q_cantharellales", "class__qid": "Q_agaricomycetes",
            "phylum_qid": "Q_basidiomycota", "kingdom_qid": "Q_fungi",
            "ncbi_taxon_id": "5348", "article": "https://en.wikipedia.org/wiki/Chanterelle",
            "has_wikipedia": True,
        },
    ]
    return pd.DataFrame(rows)


def _patch_data(gen: BiologyGenerator) -> None:
    """Patch the generator to use mock data instead of Wikidata."""
    df = _mock_organisms()
    gen._organisms = df
    gen._family_groups = gen._build_groups(df, "family")
    gen._order_groups = gen._build_groups(df, "order")
    gen._class_groups = gen._build_groups(df, "class_")


def _make_generator(**kwargs) -> BiologyGenerator:
    gen = BiologyGenerator(seed=42, max_pairs=500, **kwargs)
    _patch_data(gen)
    return gen


# --- Tests ---


class TestTaxonomicDistance:
    def test_same_rank(self):
        assert _taxonomic_distance("species", "species") == 0

    def test_adjacent_ranks(self):
        assert _taxonomic_distance("species", "genus") == 1

    def test_distant_ranks(self):
        assert _taxonomic_distance("species", "kingdom") == 6

    def test_unknown_rank(self):
        assert _taxonomic_distance("species", "unknown") == 99


class TestLowestCommonRank:
    def test_same_family(self):
        gen = _make_generator()
        dog = gen._organisms[gen._organisms["display_name"] == "dog"].iloc[0]
        fox = gen._organisms[gen._organisms["display_name"] == "red fox"].iloc[0]
        assert _lowest_common_rank(dog, fox) == "family"

    def test_same_order_different_family(self):
        gen = _make_generator()
        dog = gen._organisms[gen._organisms["display_name"] == "dog"].iloc[0]
        cat = gen._organisms[gen._organisms["display_name"] == "cat"].iloc[0]
        assert _lowest_common_rank(dog, cat) == "order"

    def test_same_class_different_order(self):
        gen = _make_generator()
        dog = gen._organisms[gen._organisms["display_name"] == "dog"].iloc[0]
        horse = gen._organisms[gen._organisms["display_name"] == "horse"].iloc[0]
        # Dog is Carnivora, horse is Perissodactyla, both Mammalia
        assert _lowest_common_rank(dog, horse) == "class"

    def test_different_kingdom(self):
        gen = _make_generator()
        dog = gen._organisms[gen._organisms["display_name"] == "dog"].iloc[0]
        mushroom = gen._organisms[gen._organisms["display_name"] == "button mushroom"].iloc[0]
        # Animalia vs Fungi — should not share any rank below kingdom
        lcr = _lowest_common_rank(dog, mushroom)
        assert lcr is None or lcr == "kingdom"


class TestMembership:
    def test_generates_membership_pairs(self):
        gen = _make_generator()
        pairs = list(gen._generate_membership())
        assert len(pairs) > 0
        for p in pairs:
            assert isinstance(p, ContrastivePair)
            assert p.relation_type == "is_member_of"
            assert p.generator == "biology"

    def test_dog_membership(self):
        gen = _make_generator()
        pairs = list(gen._generate_membership())
        # Find pairs about dog at family or order level (class_ excluded due to
        # 2021 ICNP reclassification ambiguity)
        dog_pairs = [
            p for p in pairs
            if "dog" in p.true_statement.lower()
        ]
        assert len(dog_pairs) > 0

    def test_false_has_different_taxon(self):
        gen = _make_generator()
        pairs = list(gen._generate_membership())
        for p in pairs:
            assert p.true_statement != p.false_statement


class TestSibling:
    def test_generates_sibling_pairs(self):
        gen = _make_generator()
        pairs = list(gen._generate_sibling())
        assert len(pairs) > 0
        for p in pairs:
            assert p.relation_type == "same_taxon"

    def test_dog_and_fox_same_family(self):
        gen = _make_generator()
        pairs = list(gen._generate_sibling())
        canidae_pairs = [
            p for p in pairs
            if "dog" in p.true_statement.lower() and "fox" in p.true_statement.lower()
            or "fox" in p.true_statement.lower() and "dog" in p.true_statement.lower()
        ]
        # Dog and fox are both Canidae, so should appear as siblings
        assert len(canidae_pairs) > 0


class TestRank:
    def test_generates_rank_pairs(self):
        gen = _make_generator()
        pairs = list(gen._generate_rank())
        assert len(pairs) > 0
        for p in pairs:
            assert p.relation_type == "has_rank"
            assert p.domain == "taxonomy"

    def test_species_more_specific_than_genus(self):
        gen = _make_generator()
        pairs = list(gen._generate_rank())
        species_genus = [
            p for p in pairs
            if "pecies" in p.true_statement and "genus" in p.true_statement
            and "more specific" in p.true_statement
        ]
        assert len(species_genus) > 0

    def test_rank_count(self):
        """7 ranks produce C(7,2) = 21 pairs, x2 templates = 42."""
        gen = _make_generator()
        pairs = list(gen._generate_rank())
        assert len(pairs) == 42


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
        gen2 = BiologyGenerator(seed=99, max_pairs=500)
        _patch_data(gen2)
        pairs1 = list(gen1.generate())
        pairs2 = list(gen2.generate())
        # Rank pairs are deterministic regardless of seed, but membership/sibling differ
        ids1 = {p.pair_id for p in pairs1}
        ids2 = {p.pair_id for p in pairs2}
        # At least some should differ (membership and sibling use rng)
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
            assert p.template_id.startswith("bio_")


class TestDifficulty:
    def test_difficulty_values(self):
        gen = _make_generator()
        pairs = list(gen.generate())
        valid_diffs = {Difficulty.HARD.value, Difficulty.MEDIUM.value, Difficulty.EASY.value}
        for p in pairs:
            assert p.difficulty in valid_diffs

    def test_rank_difficulty_tiers(self):
        gen = _make_generator()
        pairs = list(gen._generate_rank())
        hard = [p for p in pairs if p.difficulty == Difficulty.HARD.value]
        easy = [p for p in pairs if p.difficulty == Difficulty.EASY.value]
        assert len(hard) > 0
        assert len(easy) > 0


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
        assert true_row["generator"] == "biology"


class TestEdgeCases:
    def test_max_pairs_respected(self):
        gen = BiologyGenerator(seed=42, max_pairs=5)
        _patch_data(gen)
        pairs = list(gen.generate())
        assert len(pairs) <= 5

    def test_pair_ids_unique(self):
        gen = _make_generator()
        pairs = list(gen.generate())
        ids = [p.pair_id for p in pairs]
        assert len(ids) == len(set(ids)), "Duplicate pair IDs found"

    def test_empty_data_produces_no_pairs(self):
        gen = BiologyGenerator(seed=42, max_pairs=100)
        gen._organisms = pd.DataFrame()
        gen._family_groups = {}
        gen._order_groups = {}
        gen._class_groups = {}
        pairs = list(gen.generate())
        assert len(pairs) == 0
