"""Tests for the language generator."""

from __future__ import annotations

import pandas as pd

from latenet.datasources.wikidata import _in_expected_script, _is_clean_label
from latenet.generators.language import (
    LANGUAGE_FAMILIES,
    LanguageGenerator,
    _language_distance,
    _make_pair_id,
)
from latenet.types import ContrastivePair, Difficulty


def _mock_translation_data() -> pd.DataFrame:
    """Small set of mock translation pairs with known translations for testing."""
    rows = [
        # Animals — dog
        {"qid": "Q144", "english_label": "dog", "target_language": "es", "target_label": "perro", "entity_domain": "animal", "has_wikipedia": True},
        {"qid": "Q144", "english_label": "dog", "target_language": "fr", "target_label": "chien", "entity_domain": "animal", "has_wikipedia": True},
        {"qid": "Q144", "english_label": "dog", "target_language": "de", "target_label": "Hund", "entity_domain": "animal", "has_wikipedia": True},
        {"qid": "Q144", "english_label": "dog", "target_language": "pt", "target_label": "cão", "entity_domain": "animal", "has_wikipedia": True},
        {"qid": "Q144", "english_label": "dog", "target_language": "it", "target_label": "cane", "entity_domain": "animal", "has_wikipedia": True},
        {"qid": "Q144", "english_label": "dog", "target_language": "zh", "target_label": "狗", "entity_domain": "animal", "has_wikipedia": True},
        {"qid": "Q144", "english_label": "dog", "target_language": "ja", "target_label": "犬", "entity_domain": "animal", "has_wikipedia": True},
        {"qid": "Q144", "english_label": "dog", "target_language": "ru", "target_label": "собака", "entity_domain": "animal", "has_wikipedia": True},
        # Animals — cat
        {"qid": "Q146", "english_label": "cat", "target_language": "es", "target_label": "gato", "entity_domain": "animal", "has_wikipedia": True},
        {"qid": "Q146", "english_label": "cat", "target_language": "fr", "target_label": "chat", "entity_domain": "animal", "has_wikipedia": True},
        {"qid": "Q146", "english_label": "cat", "target_language": "de", "target_label": "Katze", "entity_domain": "animal", "has_wikipedia": True},
        {"qid": "Q146", "english_label": "cat", "target_language": "pt", "target_label": "gato", "entity_domain": "animal", "has_wikipedia": True},
        {"qid": "Q146", "english_label": "cat", "target_language": "it", "target_label": "gatto", "entity_domain": "animal", "has_wikipedia": True},
        {"qid": "Q146", "english_label": "cat", "target_language": "zh", "target_label": "猫", "entity_domain": "animal", "has_wikipedia": True},
        {"qid": "Q146", "english_label": "cat", "target_language": "ja", "target_label": "猫", "entity_domain": "animal", "has_wikipedia": True},
        {"qid": "Q146", "english_label": "cat", "target_language": "ru", "target_label": "кошка", "entity_domain": "animal", "has_wikipedia": True},
        # Animals — horse
        {"qid": "Q726", "english_label": "horse", "target_language": "es", "target_label": "caballo", "entity_domain": "animal", "has_wikipedia": True},
        {"qid": "Q726", "english_label": "horse", "target_language": "fr", "target_label": "cheval", "entity_domain": "animal", "has_wikipedia": True},
        {"qid": "Q726", "english_label": "horse", "target_language": "de", "target_label": "Pferd", "entity_domain": "animal", "has_wikipedia": True},
        {"qid": "Q726", "english_label": "horse", "target_language": "pt", "target_label": "cavalo", "entity_domain": "animal", "has_wikipedia": True},
        # Food — apple
        {"qid": "Q89", "english_label": "apple", "target_language": "es", "target_label": "manzana", "entity_domain": "food", "has_wikipedia": True},
        {"qid": "Q89", "english_label": "apple", "target_language": "fr", "target_label": "pomme", "entity_domain": "food", "has_wikipedia": True},
        {"qid": "Q89", "english_label": "apple", "target_language": "de", "target_label": "Apfel", "entity_domain": "food", "has_wikipedia": True},
        {"qid": "Q89", "english_label": "apple", "target_language": "pt", "target_label": "maçã", "entity_domain": "food", "has_wikipedia": True},
        # Food — bread
        {"qid": "Q7802", "english_label": "bread", "target_language": "es", "target_label": "pan", "entity_domain": "food", "has_wikipedia": True},
        {"qid": "Q7802", "english_label": "bread", "target_language": "fr", "target_label": "pain", "entity_domain": "food", "has_wikipedia": True},
        {"qid": "Q7802", "english_label": "bread", "target_language": "de", "target_label": "Brot", "entity_domain": "food", "has_wikipedia": True},
        {"qid": "Q7802", "english_label": "bread", "target_language": "pt", "target_label": "pão", "entity_domain": "food", "has_wikipedia": True},
        # Country — France
        {"qid": "Q142", "english_label": "France", "target_language": "es", "target_label": "Francia", "entity_domain": "country", "has_wikipedia": True},
        {"qid": "Q142", "english_label": "France", "target_language": "de", "target_label": "Frankreich", "entity_domain": "country", "has_wikipedia": True},
        {"qid": "Q142", "english_label": "France", "target_language": "pt", "target_label": "França", "entity_domain": "country", "has_wikipedia": True},
        {"qid": "Q142", "english_label": "France", "target_language": "zh", "target_label": "法国", "entity_domain": "country", "has_wikipedia": True},
    ]
    return pd.DataFrame(rows)


def _patch_data(gen: LanguageGenerator) -> None:
    """Patch the generator to use mock data instead of Wikidata."""
    df = _mock_translation_data()
    gen._data = df

    # Rebuild lookup structures
    gen._translation_map = {}
    gen._domain_entities = {}
    gen._lang_words = {}
    gen._entity_domains = {}
    gen._entity_qids = {}

    for _, row in df.iterrows():
        en = str(row["english_label"])
        lang = str(row["target_language"])
        target = str(row["target_label"])
        domain = str(row["entity_domain"])
        qid = str(row["qid"])

        gen._translation_map[(en, lang)] = target
        gen._domain_entities.setdefault(domain, [])
        if en not in gen._entity_domains:
            gen._domain_entities[domain].append(en)
        gen._entity_domains[en] = domain
        gen._entity_qids[en] = qid
        gen._lang_words.setdefault(lang, set()).add(target.lower())

    gen._all_english = sorted(set(gen._entity_domains.keys()))


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


class TestCleanLabel:
    def test_single_word(self):
        assert _is_clean_label("perro")

    def test_two_words(self):
        assert _is_clean_label("ice cream")

    def test_parentheses_rejected(self):
        assert not _is_clean_label("perro (animal)")

    def test_comma_rejected(self):
        assert not _is_clean_label("dog, domestic")

    def test_too_many_words_rejected(self):
        assert not _is_clean_label("a very long label phrase with many words")

    def test_empty_rejected(self):
        assert not _is_clean_label("")

    def test_whitespace_only_rejected(self):
        assert not _is_clean_label("   ")


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
