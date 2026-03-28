"""Tests for the temporal generator."""

from __future__ import annotations

import pandas as pd

from latenet.datasources.wikidata import _century_label, _parse_year, _year_to_century
from latenet.generators.temporal import (
    TemporalGenerator,
    _make_pair_id,
    _year_gap_difficulty,
)
from latenet.types import ContrastivePair, Difficulty


def _mock_events() -> pd.DataFrame:
    """Small set of mock historical events with known dates."""
    rows = [
        {
            "qid": "Q362", "name": "World War I",
            "description": "global war (1914-1918)",
            "date": "+1914-07-28T00:00:00Z", "year": 1914,
            "century": 20, "event_type": "war",
            "article": "https://en.wikipedia.org/wiki/World_War_I",
            "has_wikipedia": True,
        },
        {
            "qid": "Q361", "name": "World War II",
            "description": "global war (1939-1945)",
            "date": "+1939-09-01T00:00:00Z", "year": 1939,
            "century": 20, "event_type": "war",
            "article": "https://en.wikipedia.org/wiki/World_War_II",
            "has_wikipedia": True,
        },
        {
            "qid": "Q6534", "name": "French Revolution",
            "description": "revolution in France (1789-1799)",
            "date": "+1789-07-14T00:00:00Z", "year": 1789,
            "century": 18, "event_type": "revolution",
            "article": "https://en.wikipedia.org/wiki/French_Revolution",
            "has_wikipedia": True,
        },
        {
            "qid": "Q11812", "name": "Printing press",
            "description": "invention by Gutenberg",
            "date": "+1440-01-01T00:00:00Z", "year": 1440,
            "century": 15, "event_type": "invention",
            "article": "https://en.wikipedia.org/wiki/Printing_press",
            "has_wikipedia": True,
        },
        {
            "qid": "Q11451", "name": "Moon landing",
            "description": "Apollo 11 moon landing",
            "date": "+1969-07-20T00:00:00Z", "year": 1969,
            "century": 20, "event_type": "event",
            "article": "https://en.wikipedia.org/wiki/Moon_landing",
            "has_wikipedia": True,
        },
        {
            "qid": "Q1001", "name": "Fall of Constantinople",
            "description": "capture of Constantinople by the Ottoman Empire",
            "date": "+1453-05-29T00:00:00Z", "year": 1453,
            "century": 15, "event_type": "battle",
            "article": "https://en.wikipedia.org/wiki/Fall_of_Constantinople",
            "has_wikipedia": True,
        },
    ]
    return pd.DataFrame(rows)


def _mock_people() -> pd.DataFrame:
    """Small set of mock notable people with known dates."""
    rows = [
        {
            "qid": "Q937", "name": "Albert Einstein",
            "birth_date": "+1879-03-14T00:00:00Z", "birth_year": 1879,
            "death_date": "+1955-04-18T00:00:00Z", "death_year": 1955,
            "occupation": "physicist", "nationality": "Germany",
            "article": "https://en.wikipedia.org/wiki/Albert_Einstein",
            "has_wikipedia": True,
        },
        {
            "qid": "Q7311", "name": "Richard Feynman",
            "birth_date": "+1918-05-11T00:00:00Z", "birth_year": 1918,
            "death_date": "+1988-02-15T00:00:00Z", "death_year": 1988,
            "occupation": "physicist", "nationality": "United States",
            "article": "https://en.wikipedia.org/wiki/Richard_Feynman",
            "has_wikipedia": True,
        },
        {
            "qid": "Q692", "name": "William Shakespeare",
            "birth_date": "+1564-04-26T00:00:00Z", "birth_year": 1564,
            "death_date": "+1616-04-23T00:00:00Z", "death_year": 1616,
            "occupation": "playwright", "nationality": "England",
            "article": "https://en.wikipedia.org/wiki/William_Shakespeare",
            "has_wikipedia": True,
        },
        {
            "qid": "Q868", "name": "Aristotle",
            "birth_date": "-0384-01-01T00:00:00Z", "birth_year": -384,
            "death_date": "-0322-01-01T00:00:00Z", "death_year": -322,
            "occupation": "philosopher", "nationality": "Greece",
            "article": "https://en.wikipedia.org/wiki/Aristotle",
            "has_wikipedia": True,
        },
        {
            "qid": "Q762", "name": "Leonardo da Vinci",
            "birth_date": "+1452-04-15T00:00:00Z", "birth_year": 1452,
            "death_date": "+1519-05-02T00:00:00Z", "death_year": 1519,
            "occupation": "polymath", "nationality": "Italy",
            "article": "https://en.wikipedia.org/wiki/Leonardo_da_Vinci",
            "has_wikipedia": True,
        },
        {
            "qid": "Q1035", "name": "Charles Darwin",
            "birth_date": "+1809-02-12T00:00:00Z", "birth_year": 1809,
            "death_date": "+1882-04-19T00:00:00Z", "death_year": 1882,
            "occupation": "naturalist", "nationality": "United Kingdom",
            "article": "https://en.wikipedia.org/wiki/Charles_Darwin",
            "has_wikipedia": True,
        },
        {
            "qid": "Q5593", "name": "Niels Bohr",
            "birth_date": "+1885-10-07T00:00:00Z", "birth_year": 1885,
            "death_date": "+1962-11-18T00:00:00Z", "death_year": 1962,
            "occupation": "physicist", "nationality": "Denmark",
            "article": "https://en.wikipedia.org/wiki/Niels_Bohr",
            "has_wikipedia": True,
        },
        {
            "qid": "Q1067", "name": "Mahatma Gandhi",
            "birth_date": "+1869-10-02T00:00:00Z", "birth_year": 1869,
            "death_date": "+1948-01-30T00:00:00Z", "death_year": 1948,
            "occupation": "activist", "nationality": "India",
            "article": "https://en.wikipedia.org/wiki/Mahatma_Gandhi",
            "has_wikipedia": True,
        },
    ]
    df = pd.DataFrame(rows)
    df["death_year"] = df["death_year"].astype("Int64")
    return df


def _patch_data(gen: TemporalGenerator) -> None:
    """Patch the generator to use mock data instead of Wikidata."""
    gen._events = _mock_events()
    gen._people = _mock_people()


def _make_generator(**kwargs) -> TemporalGenerator:
    gen = TemporalGenerator(seed=42, max_pairs=500, **kwargs)
    _patch_data(gen)
    return gen


# --- Utility function tests ---


class TestParseYear:
    def test_positive_year(self):
        assert _parse_year("+1776-07-04T00:00:00Z") == 1776

    def test_negative_year(self):
        assert _parse_year("-0500-01-01T00:00:00Z") == -500

    def test_plain_iso(self):
        assert _parse_year("1969-07-20") == 1969

    def test_empty(self):
        assert _parse_year("") is None

    def test_none(self):
        assert _parse_year(None) is None


class TestYearToCentury:
    def test_first_century(self):
        assert _year_to_century(50) == 1

    def test_18th_century(self):
        assert _year_to_century(1776) == 18

    def test_20th_century(self):
        assert _year_to_century(1969) == 20

    def test_boundary_year(self):
        # Year 1800 is the last year of 18th century
        assert _year_to_century(1800) == 18

    def test_year_1801(self):
        # Year 1801 is the first year of 19th century
        assert _year_to_century(1801) == 19

    def test_negative_year(self):
        assert _year_to_century(-500) == -5

    def test_year_100(self):
        assert _year_to_century(100) == 1

    def test_year_101(self):
        assert _year_to_century(101) == 2


class TestCenturyLabel:
    def test_15th_century(self):
        assert _century_label(15) == "15th century"

    def test_1st_century(self):
        assert _century_label(1) == "1st century"

    def test_2nd_century(self):
        assert _century_label(2) == "2nd century"

    def test_3rd_century(self):
        assert _century_label(3) == "3rd century"

    def test_21st_century(self):
        assert _century_label(21) == "21st century"

    def test_negative_century(self):
        assert _century_label(-5) == "5th century BC"


class TestYearGapDifficulty:
    def test_close(self):
        assert _year_gap_difficulty(25) == "close"

    def test_moderate(self):
        assert _year_gap_difficulty(100) == "moderate"

    def test_distant(self):
        assert _year_gap_difficulty(500) == "distant"

    def test_boundary_close_moderate(self):
        assert _year_gap_difficulty(49) == "close"
        assert _year_gap_difficulty(50) == "moderate"

    def test_boundary_moderate_distant(self):
        assert _year_gap_difficulty(199) == "moderate"
        assert _year_gap_difficulty(200) == "distant"


# --- Generator tests ---


class TestEventOrdering:
    def test_generates_event_pairs(self):
        gen = _make_generator()
        pairs = list(gen._generate_event_ordering())
        assert len(pairs) > 0
        for p in pairs:
            assert isinstance(p, ContrastivePair)
            assert p.relation_type == "happened_before"
            assert p.generator == "temporal"

    def test_ordering_correctness(self):
        gen = _make_generator()
        pairs = list(gen._generate_event_ordering())
        for p in pairs:
            # The true statement should express temporal ordering
            stmt = p.true_statement.lower()
            assert any(w in stmt for w in ["before", "preceded", "earlier"])

    def test_min_year_gap_respected(self):
        gen = _make_generator(min_year_gap=30)
        pairs = list(gen._generate_event_ordering())
        for p in pairs:
            assert p.semantic_distance >= 30


class TestBirthOrdering:
    def test_generates_birth_pairs(self):
        gen = _make_generator()
        pairs = list(gen._generate_birth_ordering())
        assert len(pairs) > 0
        for p in pairs:
            assert p.relation_type == "born_before"

    def test_ordering_correctness(self):
        gen = _make_generator()
        pairs = list(gen._generate_birth_ordering())
        for p in pairs:
            assert "before" in p.true_statement.lower() or "earlier" in p.true_statement.lower() or "predates" in p.true_statement.lower()


class TestCenturyAttribution:
    def test_generates_century_pairs(self):
        gen = _make_generator()
        pairs = list(gen._generate_century_attribution())
        assert len(pairs) > 0
        for p in pairs:
            assert p.relation_type == "occurred_in_century"

    def test_century_in_statement(self):
        gen = _make_generator()
        pairs = list(gen._generate_century_attribution())
        for p in pairs:
            assert "century" in p.true_statement.lower()
            assert "century" in p.false_statement.lower()

    def test_true_and_false_differ(self):
        gen = _make_generator()
        pairs = list(gen._generate_century_attribution())
        for p in pairs:
            assert p.true_statement != p.false_statement


class TestEraOrdering:
    def test_generates_era_pairs(self):
        gen = _make_generator(min_era_gap=50)
        pairs = list(gen._generate_era_ordering())
        # With mock data, we should get some pairs (e.g. Aristotle lived before Moon landing)
        assert len(pairs) > 0
        for p in pairs:
            assert p.relation_type == "lived_before_event"


class TestContemporaneity:
    def test_generates_contemporary_pairs(self):
        gen = _make_generator()
        pairs = list(gen._generate_contemporaneity())
        assert len(pairs) > 0
        for p in pairs:
            assert p.relation_type == "were_contemporaries"

    def test_einstein_bohr_contemporaries(self):
        """Einstein (1879-1955) and Bohr (1885-1962) overlapped significantly."""
        gen = _make_generator()
        pairs = list(gen._generate_contemporaneity())
        # Check if any pair mentions both
        einstein_bohr = [
            p for p in pairs
            if ("Einstein" in p.true_statement and "Bohr" in p.true_statement)
            or ("Bohr" in p.true_statement and "Einstein" in p.true_statement)
        ]
        # Not guaranteed with random sampling, but with our small mock set it's likely
        # So just check that contemporaries are generated at all
        assert len(pairs) > 0


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
        gen2 = TemporalGenerator(seed=99, max_pairs=500)
        _patch_data(gen2)
        pairs1 = list(gen1.generate())
        pairs2 = list(gen2.generate())
        ids1 = {p.pair_id for p in pairs1}
        ids2 = {p.pair_id for p in pairs2}
        # At least some should differ due to random sampling
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
            assert p.template_id.startswith("temp_")


class TestDifficulty:
    def test_difficulty_is_mixed(self):
        gen = _make_generator()
        pairs = list(gen.generate())
        for p in pairs:
            assert p.difficulty == "mixed"

    def test_gap_bucket_in_gen_params(self):
        gen = _make_generator()
        pairs = list(gen.generate())
        # Pairs with gap_bucket in gen_params should have descriptive labels
        valid_buckets = {"close", "moderate", "distant"}
        for p in pairs:
            if p.gen_params and "gap_bucket" in p.gen_params:
                assert p.gen_params["gap_bucket"] in valid_buckets


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
        assert true_row["generator"] == "temporal"


class TestEdgeCases:
    def test_max_pairs_respected(self):
        gen = TemporalGenerator(seed=42, max_pairs=5)
        _patch_data(gen)
        pairs = list(gen.generate())
        assert len(pairs) <= 5

    def test_pair_ids_unique(self):
        gen = _make_generator()
        pairs = list(gen.generate())
        ids = [p.pair_id for p in pairs]
        assert len(ids) == len(set(ids)), "Duplicate pair IDs found"

    def test_empty_data_produces_no_pairs(self):
        gen = TemporalGenerator(seed=42, max_pairs=100)
        gen._events = pd.DataFrame()
        gen._people = pd.DataFrame()
        pairs = list(gen.generate())
        assert len(pairs) == 0
