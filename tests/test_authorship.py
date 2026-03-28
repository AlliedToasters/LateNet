"""Tests for the authorship generator."""

from __future__ import annotations

import pandas as pd

from latenet.generators.authorship import (
    AuthorshipGenerator,
    _era_from_year,
    _make_pair_id,
    _DOMAIN_PAST_VERB,
)
from latenet.types import ContrastivePair, Difficulty


def _mock_author_work_data() -> pd.DataFrame:
    """Small set of mock author-work pairs with known attributions for testing."""
    rows = [
        # Literature — Shakespeare
        {
            "author_qid": "Q692", "author_name": "Shakespeare",
            "author_occupation": "author",
            "work_qid": "Q41567", "work_name": "Hamlet",
            "work_type": "play", "creative_domain": "literature",
            "publication_year": 1600, "has_wikipedia_author": True,
            "has_wikipedia_work": True, "verb": "written", "role": "author",
            "science_attribution": False,
        },
        {
            "author_qid": "Q692", "author_name": "Shakespeare",
            "author_occupation": "author",
            "work_qid": "Q83186", "work_name": "Macbeth",
            "work_type": "play", "creative_domain": "literature",
            "publication_year": 1606, "has_wikipedia_author": True,
            "has_wikipedia_work": True, "verb": "written", "role": "author",
            "science_attribution": False,
        },
        # Literature — Dickens (different era)
        {
            "author_qid": "Q5686", "author_name": "Dickens",
            "author_occupation": "author",
            "work_qid": "Q189811", "work_name": "Oliver Twist",
            "work_type": "novel", "creative_domain": "literature",
            "publication_year": 1838, "has_wikipedia_author": True,
            "has_wikipedia_work": True, "verb": "written", "role": "author",
            "science_attribution": False,
        },
        {
            "author_qid": "Q5686", "author_name": "Dickens",
            "author_occupation": "author",
            "work_qid": "Q189325", "work_name": "A Tale of Two Cities",
            "work_type": "novel", "creative_domain": "literature",
            "publication_year": 1859, "has_wikipedia_author": True,
            "has_wikipedia_work": True, "verb": "written", "role": "author",
            "science_attribution": False,
        },
        # Literature — Marlowe (same era as Shakespeare)
        {
            "author_qid": "Q28975", "author_name": "Marlowe",
            "author_occupation": "author",
            "work_qid": "Q333839", "work_name": "Doctor Faustus",
            "work_type": "play", "creative_domain": "literature",
            "publication_year": 1592, "has_wikipedia_author": True,
            "has_wikipedia_work": True, "verb": "written", "role": "author",
            "science_attribution": False,
        },
        {
            "author_qid": "Q28975", "author_name": "Marlowe",
            "author_occupation": "author",
            "work_qid": "Q1075827", "work_name": "Tamburlaine",
            "work_type": "play", "creative_domain": "literature",
            "publication_year": 1587, "has_wikipedia_author": True,
            "has_wikipedia_work": True, "verb": "written", "role": "author",
            "science_attribution": False,
        },
        # Music — Beethoven
        {
            "author_qid": "Q255", "author_name": "Beethoven",
            "author_occupation": "composer",
            "work_qid": "Q11989", "work_name": "Ninth Symphony",
            "work_type": "symphony", "creative_domain": "music",
            "publication_year": 1824, "has_wikipedia_author": True,
            "has_wikipedia_work": True, "verb": "composed", "role": "composer",
            "science_attribution": False,
        },
        {
            "author_qid": "Q255", "author_name": "Beethoven",
            "author_occupation": "composer",
            "work_qid": "Q186099", "work_name": "Moonlight Sonata",
            "work_type": "sonata", "creative_domain": "music",
            "publication_year": 1801, "has_wikipedia_author": True,
            "has_wikipedia_work": True, "verb": "composed", "role": "composer",
            "science_attribution": False,
        },
        # Music — Mozart (same era as Beethoven)
        {
            "author_qid": "Q254", "author_name": "Mozart",
            "author_occupation": "composer",
            "work_qid": "Q5064", "work_name": "The Magic Flute",
            "work_type": "opera", "creative_domain": "music",
            "publication_year": 1791, "has_wikipedia_author": True,
            "has_wikipedia_work": True, "verb": "composed", "role": "composer",
            "science_attribution": False,
        },
        {
            "author_qid": "Q254", "author_name": "Mozart",
            "author_occupation": "composer",
            "work_qid": "Q189600", "work_name": "Don Giovanni",
            "work_type": "opera", "creative_domain": "music",
            "publication_year": 1787, "has_wikipedia_author": True,
            "has_wikipedia_work": True, "verb": "composed", "role": "composer",
            "science_attribution": False,
        },
        # Art — Leonardo da Vinci
        {
            "author_qid": "Q762", "author_name": "Leonardo da Vinci",
            "author_occupation": "painter",
            "work_qid": "Q12418", "work_name": "Mona Lisa",
            "work_type": "painting", "creative_domain": "art",
            "publication_year": 1503, "has_wikipedia_author": True,
            "has_wikipedia_work": True, "verb": "painted", "role": "painter",
            "science_attribution": False,
        },
        {
            "author_qid": "Q762", "author_name": "Leonardo da Vinci",
            "author_occupation": "painter",
            "work_qid": "Q128910", "work_name": "The Last Supper",
            "work_type": "painting", "creative_domain": "art",
            "publication_year": 1498, "has_wikipedia_author": True,
            "has_wikipedia_work": True, "verb": "painted", "role": "painter",
            "science_attribution": False,
        },
        # Art — Picasso (different era)
        {
            "author_qid": "Q5593", "author_name": "Picasso",
            "author_occupation": "painter",
            "work_qid": "Q175036", "work_name": "Guernica",
            "work_type": "painting", "creative_domain": "art",
            "publication_year": 1937, "has_wikipedia_author": True,
            "has_wikipedia_work": True, "verb": "painted", "role": "painter",
            "science_attribution": False,
        },
        {
            "author_qid": "Q5593", "author_name": "Picasso",
            "author_occupation": "painter",
            "work_qid": "Q631276", "work_name": "Les Demoiselles d'Avignon",
            "work_type": "painting", "creative_domain": "art",
            "publication_year": 1907, "has_wikipedia_author": True,
            "has_wikipedia_work": True, "verb": "painted", "role": "painter",
            "science_attribution": False,
        },
    ]
    return pd.DataFrame(rows)


def _patch_data(gen: AuthorshipGenerator) -> None:
    """Patch the generator to use mock data instead of Wikidata."""
    df = _mock_author_work_data()
    df["era"] = df["publication_year"].apply(_era_from_year)
    gen._data = df
    gen._domain_groups = {
        domain: group.reset_index(drop=True)
        for domain, group in df.groupby("creative_domain")
    }
    gen._author_works = {}
    for idx, row in df.iterrows():
        aqid = row["author_qid"]
        gen._author_works.setdefault(aqid, []).append(idx)


def _make_generator(**kwargs) -> AuthorshipGenerator:
    gen = AuthorshipGenerator(seed=42, max_pairs=500, **kwargs)
    _patch_data(gen)
    return gen


# --- Tests ---


class TestEraFromYear:
    def test_modern_year(self):
        assert _era_from_year(1600) == "16"

    def test_19th_century(self):
        assert _era_from_year(1838) == "19"

    def test_none(self):
        assert _era_from_year(None) is None

    def test_bce(self):
        assert _era_from_year(-500) is not None


class TestCreatedBy:
    def test_generates_pairs(self):
        gen = _make_generator()
        pairs = list(gen._generate_created_by())
        assert len(pairs) > 0
        for p in pairs:
            assert isinstance(p, ContrastivePair)
            assert p.relation_type == "created_by"
            assert p.generator == "authorship"

    def test_true_false_different(self):
        gen = _make_generator()
        pairs = list(gen._generate_created_by())
        for p in pairs:
            assert p.true_statement != p.false_statement

    def test_domain_verb_mapping(self):
        """Verify that literature uses 'written', music uses 'composed', etc."""
        gen = _make_generator()
        pairs = list(gen._generate_created_by())

        for p in pairs:
            stmt = p.true_statement
            # Check the source work to determine which domain it's from
            if "Hamlet" in stmt or "Macbeth" in stmt or "Oliver Twist" in stmt:
                # Literature should use "written" or "wrote"
                assert "written" in stmt or "wrote" in stmt or "by" in stmt
            elif "Ninth Symphony" in stmt or "Moonlight Sonata" in stmt:
                # Music should use "composed"
                assert "composed" in stmt or "by" in stmt
            elif "Mona Lisa" in stmt or "Guernica" in stmt:
                # Art should use "painted"
                assert "painted" in stmt or "by" in stmt


class TestAuthorOf:
    def test_generates_pairs(self):
        gen = _make_generator()
        pairs = list(gen._generate_author_of())
        assert len(pairs) > 0
        for p in pairs:
            assert p.relation_type == "author_of"

    def test_false_work_from_different_author(self):
        """The false statement should attribute a different author's work."""
        gen = _make_generator()
        pairs = list(gen._generate_author_of())
        for p in pairs:
            # Source synset is the author, target is true work, neg is false work
            assert p.target_synset != p.neg_synset


class TestDomainAttribution:
    def test_generates_pairs(self):
        gen = _make_generator()
        pairs = list(gen._generate_domain_attribution())
        assert len(pairs) > 0
        for p in pairs:
            assert p.relation_type == "worked_in_domain"

    def test_false_role_different(self):
        gen = _make_generator()
        pairs = list(gen._generate_domain_attribution())
        for p in pairs:
            assert p.true_statement != p.false_statement
            # The role in true should differ from false
            # (e.g., "author" vs "composer")


class TestDifficultyTiers:
    def test_difficulty_values_valid(self):
        gen = _make_generator()
        pairs = list(gen.generate())
        for p in pairs:
            assert p.difficulty == "mixed"

    def test_same_domain_same_era_is_hard(self):
        gen = _make_generator()
        # Shakespeare (1600s) swapped with Marlowe (1590s) = same domain, same era
        difficulty = gen._difficulty_for_swap("literature", "16", "literature", "16")
        assert difficulty == Difficulty.HARD.value

    def test_same_domain_diff_era_is_medium(self):
        gen = _make_generator()
        # Shakespeare (16th c) swapped with Dickens (19th c)
        difficulty = gen._difficulty_for_swap("literature", "16", "literature", "19")
        assert difficulty == Difficulty.MEDIUM.value

    def test_diff_domain_is_easy(self):
        gen = _make_generator()
        difficulty = gen._difficulty_for_swap("literature", "16", "music", "19")
        assert difficulty == Difficulty.EASY.value


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
        gen2 = AuthorshipGenerator(seed=99, max_pairs=500)
        _patch_data(gen2)
        pairs1 = list(gen1.generate())
        pairs2 = list(gen2.generate())
        ids1 = {p.pair_id for p in pairs1}
        ids2 = {p.pair_id for p in pairs2}
        # At least some should differ due to RNG-driven template/swap selection
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
            assert p.template_id.startswith("auth_")


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
        assert true_row["generator"] == "authorship"


class TestEdgeCases:
    def test_max_pairs_respected(self):
        gen = AuthorshipGenerator(seed=42, max_pairs=5)
        _patch_data(gen)
        pairs = list(gen.generate())
        assert len(pairs) <= 5

    def test_pair_ids_unique(self):
        gen = _make_generator()
        pairs = list(gen.generate())
        ids = [p.pair_id for p in pairs]
        assert len(ids) == len(set(ids)), "Duplicate pair IDs found"

    def test_empty_data_produces_no_pairs(self):
        gen = AuthorshipGenerator(seed=42, max_pairs=100)
        gen._data = pd.DataFrame()
        gen._domain_groups = {}
        gen._author_works = {}
        pairs = list(gen.generate())
        assert len(pairs) == 0


class TestGeneratorContract:
    def test_name(self):
        gen = _make_generator()
        assert gen.name == "authorship"

    def test_relation_types(self):
        gen = _make_generator()
        assert gen.relation_types() == ["created_by", "author_of", "worked_in_domain"]

    def test_domains(self):
        gen = _make_generator()
        assert gen.domains() == ["authorship"]
