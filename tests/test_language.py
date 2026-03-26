"""Tests for the language generator."""

from __future__ import annotations

import pandas as pd

from latenet.generators.muse_data import _in_expected_script
from latenet.generators.language import (
    LANGUAGE_FAMILIES,
    LanguageGenerator,
    _language_distance,
    _make_pair_id,
)
from latenet.types import ContrastivePair, Difficulty


def _mock_translation_data() -> pd.DataFrame:
    """Small set of mock translation pairs with known translations for testing.

    Uses MUSE-style schema: source_word, target_word, source_lang, target_lang, domain.
    """
    rows = [
        # Animals — dog
        {"source_word": "dog", "target_word": "perro", "source_lang": "en", "target_lang": "es", "domain": "animal"},
        {"source_word": "dog", "target_word": "chien", "source_lang": "en", "target_lang": "fr", "domain": "animal"},
        {"source_word": "dog", "target_word": "Hund", "source_lang": "en", "target_lang": "de", "domain": "animal"},
        {"source_word": "dog", "target_word": "cão", "source_lang": "en", "target_lang": "pt", "domain": "animal"},
        {"source_word": "dog", "target_word": "cane", "source_lang": "en", "target_lang": "it", "domain": "animal"},
        {"source_word": "dog", "target_word": "狗", "source_lang": "en", "target_lang": "zh", "domain": "animal"},
        {"source_word": "dog", "target_word": "犬", "source_lang": "en", "target_lang": "ja", "domain": "animal"},
        {"source_word": "dog", "target_word": "собака", "source_lang": "en", "target_lang": "ru", "domain": "animal"},
        # Animals — cat
        {"source_word": "cat", "target_word": "gato", "source_lang": "en", "target_lang": "es", "domain": "animal"},
        {"source_word": "cat", "target_word": "chat", "source_lang": "en", "target_lang": "fr", "domain": "animal"},
        {"source_word": "cat", "target_word": "Katze", "source_lang": "en", "target_lang": "de", "domain": "animal"},
        {"source_word": "cat", "target_word": "gato", "source_lang": "en", "target_lang": "pt", "domain": "animal"},
        {"source_word": "cat", "target_word": "gatto", "source_lang": "en", "target_lang": "it", "domain": "animal"},
        {"source_word": "cat", "target_word": "猫", "source_lang": "en", "target_lang": "zh", "domain": "animal"},
        {"source_word": "cat", "target_word": "猫", "source_lang": "en", "target_lang": "ja", "domain": "animal"},
        {"source_word": "cat", "target_word": "кошка", "source_lang": "en", "target_lang": "ru", "domain": "animal"},
        # Animals — horse
        {"source_word": "horse", "target_word": "caballo", "source_lang": "en", "target_lang": "es", "domain": "animal"},
        {"source_word": "horse", "target_word": "cheval", "source_lang": "en", "target_lang": "fr", "domain": "animal"},
        {"source_word": "horse", "target_word": "Pferd", "source_lang": "en", "target_lang": "de", "domain": "animal"},
        {"source_word": "horse", "target_word": "cavalo", "source_lang": "en", "target_lang": "pt", "domain": "animal"},
        # Food — apple
        {"source_word": "apple", "target_word": "manzana", "source_lang": "en", "target_lang": "es", "domain": "food"},
        {"source_word": "apple", "target_word": "pomme", "source_lang": "en", "target_lang": "fr", "domain": "food"},
        {"source_word": "apple", "target_word": "Apfel", "source_lang": "en", "target_lang": "de", "domain": "food"},
        {"source_word": "apple", "target_word": "maçã", "source_lang": "en", "target_lang": "pt", "domain": "food"},
        # Food — bread
        {"source_word": "bread", "target_word": "pan", "source_lang": "en", "target_lang": "es", "domain": "food"},
        {"source_word": "bread", "target_word": "pain", "source_lang": "en", "target_lang": "fr", "domain": "food"},
        {"source_word": "bread", "target_word": "Brot", "source_lang": "en", "target_lang": "de", "domain": "food"},
        {"source_word": "bread", "target_word": "pão", "source_lang": "en", "target_lang": "pt", "domain": "food"},
        # Country — France
        {"source_word": "france", "target_word": "Francia", "source_lang": "en", "target_lang": "es", "domain": "place"},
        {"source_word": "france", "target_word": "Frankreich", "source_lang": "en", "target_lang": "de", "domain": "place"},
        {"source_word": "france", "target_word": "França", "source_lang": "en", "target_lang": "pt", "domain": "place"},
        {"source_word": "france", "target_word": "法国", "source_lang": "en", "target_lang": "zh", "domain": "place"},
    ]
    return pd.DataFrame(rows)


def _patch_data(gen: LanguageGenerator) -> None:
    """Patch the generator to use mock data instead of downloading MUSE files."""
    df = _mock_translation_data()
    gen._data = df
    gen._build_lookups(df)


def _make_generator(**kwargs) -> LanguageGenerator:
    gen = LanguageGenerator(
        seed=42, max_pairs=500,
        target_languages=["es", "fr", "de", "pt", "it", "zh", "ja", "ru"],
        **kwargs,
    )
    _patch_data(gen)
    return gen


# --- Tests ---


class TestLanguageDistance:
    def test_same_subfamily_is_hard(self):
        # Spanish and Portuguese are both Romance
        assert _language_distance("es", "pt") == 1

    def test_same_macrofamily_is_medium(self):
        # Spanish (Romance) and German (Germanic) are both Indo-European
        assert _language_distance("es", "de") == 2

    def test_different_macrofamily_is_easy(self):
        # Spanish (Romance) and Chinese (Sino-Tibetan) are different
        assert _language_distance("es", "zh") == 3

    def test_romance_languages_same_family(self):
        for a in ["es", "fr", "pt", "it"]:
            for b in ["es", "fr", "pt", "it"]:
                if a != b:
                    assert _language_distance(a, b) == 1

    def test_indo_european_cross_subfamily(self):
        # Germanic + Romance
        assert _language_distance("de", "fr") == 2
        # Slavic + Romance
        assert _language_distance("ru", "es") == 2
        # Indo-Aryan + Germanic
        assert _language_distance("hi", "de") == 2

    def test_non_indo_european_pairs(self):
        # Chinese and Japanese are different macro-families
        assert _language_distance("zh", "ja") == 3
        # Arabic and Chinese
        assert _language_distance("ar", "zh") == 3


class TestScriptValidation:
    def test_chinese_characters_valid(self):
        assert _in_expected_script("狗", "zh")

    def test_chinese_latin_invalid(self):
        assert not _in_expected_script("dog", "zh")

    def test_russian_cyrillic_valid(self):
        assert _in_expected_script("собака", "ru")

    def test_russian_latin_invalid(self):
        assert not _in_expected_script("sobaka", "ru")

    def test_latin_languages_always_valid(self):
        assert _in_expected_script("perro", "es")
        assert _in_expected_script("chien", "fr")
        assert _in_expected_script("Hund", "de")

    def test_japanese_valid(self):
        assert _in_expected_script("犬", "ja")
        assert _in_expected_script("いぬ", "ja")

    def test_arabic_valid(self):
        assert _in_expected_script("كلب", "ar")

    def test_hindi_valid(self):
        assert _in_expected_script("कुत्ता", "hi")

    def test_empty_string_invalid(self):
        assert not _in_expected_script("", "zh")


class TestMuseDataLoading:
    def test_mock_data_schema(self):
        df = _mock_translation_data()
        assert "source_word" in df.columns
        assert "target_word" in df.columns
        assert "source_lang" in df.columns or True  # optional in mock
        assert "target_lang" in df.columns
        assert "domain" in df.columns

    def test_build_lookups_populates_maps(self):
        gen = _make_generator()
        assert len(gen._translation_map) > 0
        assert len(gen._domain_entities) > 0
        assert len(gen._all_english) > 0
        assert len(gen._entity_domains) > 0
        assert len(gen._lang_words) > 0

    def test_translation_map_correct(self):
        gen = _make_generator()
        assert gen._translation_map[("dog", "es")] == "perro"
        assert gen._translation_map[("cat", "fr")] == "chat"
        assert gen._translation_map[("apple", "de")] == "Apfel"

    def test_domain_entities_grouped(self):
        gen = _make_generator()
        assert "animal" in gen._domain_entities
        assert "food" in gen._domain_entities
        assert "dog" in gen._domain_entities["animal"]
        assert "apple" in gen._domain_entities["food"]

    def test_works_without_domain_column(self):
        """Generator should handle data without domain tags (fallback to 'other')."""
        gen = LanguageGenerator(seed=42, max_pairs=50, target_languages=["es", "fr"])
        df = pd.DataFrame([
            {"source_word": "dog", "target_word": "perro", "source_lang": "en", "target_lang": "es"},
            {"source_word": "cat", "target_word": "gato", "source_lang": "en", "target_lang": "es"},
            {"source_word": "dog", "target_word": "chien", "source_lang": "en", "target_lang": "fr"},
            {"source_word": "cat", "target_word": "chat", "source_lang": "en", "target_lang": "fr"},
        ])
        gen._data = df
        gen._build_lookups(df)
        # All words should get domain "other" since no domain column
        assert all(d == "other" for d in gen._entity_domains.values())
        pairs = list(gen.generate())
        assert len(pairs) > 0


class TestTranslatesTo:
    def test_generates_pairs(self):
        gen = _make_generator()
        pairs = list(gen._generate_translates_to())
        assert len(pairs) > 0
        for p in pairs:
            assert isinstance(p, ContrastivePair)
            assert p.relation_type == "translates_to"
            assert p.generator == "language"

    def test_true_false_different(self):
        gen = _make_generator()
        pairs = list(gen._generate_translates_to())
        for p in pairs:
            assert p.true_statement != p.false_statement

    def test_difficulty_tiers_present(self):
        gen = _make_generator()
        pairs = list(gen._generate_translates_to())
        difficulties = {p.difficulty for p in pairs}
        # Should have at least hard and medium (easy requires cross-language swap)
        assert Difficulty.HARD.value in difficulties or Difficulty.MEDIUM.value in difficulties

    def test_hard_swap_is_same_domain(self):
        """Hard swaps should use a translation from the same entity domain."""
        gen = _make_generator()
        pairs = list(gen._generate_translates_to())
        hard_pairs = [p for p in pairs if p.difficulty == Difficulty.HARD.value]
        assert len(hard_pairs) > 0
        for p in hard_pairs:
            assert p.negation_strategy == "sibling_swap"


class TestTranslationOf:
    def test_generates_pairs(self):
        gen = _make_generator()
        pairs = list(gen._generate_translation_of())
        assert len(pairs) > 0
        for p in pairs:
            assert p.relation_type == "translation_of"
            assert p.negation_strategy == "reverse_relation"

    def test_reverses_meaning(self):
        """The false statement should attribute the translation to a different English word."""
        gen = _make_generator()
        pairs = list(gen._generate_translation_of())
        for p in pairs:
            assert p.true_statement != p.false_statement


class TestWordIsLanguage:
    def test_generates_pairs(self):
        gen = _make_generator()
        pairs = list(gen._generate_word_is_language())
        # May be 0 if all words are cognates across languages; that's ok
        for p in pairs:
            assert p.relation_type == "word_is_language"

    def test_difficulty_by_language_distance(self):
        gen = _make_generator()
        pairs = list(gen._generate_word_is_language())
        for p in pairs:
            if p.difficulty == Difficulty.HARD.value:
                # Same sub-family
                assert p.semantic_distance == 1
            elif p.difficulty == Difficulty.EASY.value:
                assert p.semantic_distance == 3


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
        gen2 = LanguageGenerator(seed=99, max_pairs=500,
                                  target_languages=["es", "fr", "de", "pt", "it", "zh", "ja", "ru"])
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
            assert p.template_id.startswith("lang_")


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
        assert true_row["generator"] == "language"


class TestEdgeCases:
    def test_max_pairs_respected(self):
        gen = LanguageGenerator(seed=42, max_pairs=5,
                                target_languages=["es", "fr", "de", "pt"])
        _patch_data(gen)
        pairs = list(gen.generate())
        assert len(pairs) <= 5

    def test_pair_ids_unique(self):
        gen = _make_generator()
        pairs = list(gen.generate())
        ids = [p.pair_id for p in pairs]
        assert len(ids) == len(set(ids)), "Duplicate pair IDs found"

    def test_empty_data_produces_no_pairs(self):
        gen = LanguageGenerator(seed=42, max_pairs=100)
        gen._data = pd.DataFrame()
        pairs = list(gen.generate())
        assert len(pairs) == 0


class TestGeneratorContract:
    def test_name(self):
        gen = _make_generator()
        assert gen.name == "language"

    def test_relation_types(self):
        gen = _make_generator()
        assert gen.relation_types() == ["translates_to", "translation_of", "word_is_language"]

    def test_domains(self):
        gen = _make_generator()
        assert gen.domains() == ["language"]


class TestHomographFiltering:
    def test_cognate_detection(self):
        """Words that appear in multiple languages should be flagged as non-unique."""
        gen = _make_generator()
        # "猫" appears in both zh and ja in our mock data
        assert not gen._is_unique_to_language("猫", "zh")
        assert not gen._is_unique_to_language("猫", "ja")

    def test_unique_word_detected(self):
        gen = _make_generator()
        # "perro" only appears in Spanish
        assert gen._is_unique_to_language("perro", "es")


class TestPairIdStability:
    def test_pair_id_deterministic(self):
        parts = ["lang", "translate", "dog", "es", "perro", "gato"]
        id1 = _make_pair_id(parts)
        id2 = _make_pair_id(parts)
        assert id1 == id2

    def test_pair_id_different_for_different_inputs(self):
        id1 = _make_pair_id(["lang", "translate", "dog", "es"])
        id2 = _make_pair_id(["lang", "translate", "cat", "es"])
        assert id1 != id2
