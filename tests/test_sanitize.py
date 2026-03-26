"""Tests for the shared sanitizer and article resolution (Layers 0 & 1)."""

from latenet.sanitize import render_template, resolve_article, sanitize_statement
from latenet.types import ContrastivePair


class TestSanitizeStatement:
    """Layer 0: basic text normalization."""

    def test_collapses_double_spaces(self):
        assert sanitize_statement("Dogs  are  mammals.") == "Dogs are mammals."

    def test_strips_leading_trailing(self):
        assert sanitize_statement("  Dogs are mammals.  ") == "Dogs are mammals."

    def test_adds_trailing_period(self):
        assert sanitize_statement("Dogs are mammals") == "Dogs are mammals."

    def test_keeps_existing_period(self):
        assert sanitize_statement("Dogs are mammals.") == "Dogs are mammals."

    def test_collapses_double_period(self):
        assert sanitize_statement("Dr. Smith is a doctor..") == "Dr. Smith is a doctor."

    def test_preserves_ellipsis(self):
        assert sanitize_statement("And then...") == "And then..."

    def test_capitalizes_first_char(self):
        assert sanitize_statement("dogs are mammals.") == "Dogs are mammals."

    def test_already_capitalized(self):
        assert sanitize_statement("Dogs are mammals.") == "Dogs are mammals."

    def test_empty_string(self):
        assert sanitize_statement("") == ""

    def test_combined(self):
        assert sanitize_statement("  dogs  are  mammals") == "Dogs are mammals."


class TestResolveArticle:
    """Layer 1: inflect-based a/an resolution."""

    def test_consonant_start(self):
        result = resolve_article("dog")
        assert result.startswith("a ")
        assert "dog" in result

    def test_vowel_start(self):
        result = resolve_article("elephant")
        assert result.startswith("an ")

    def test_silent_h(self):
        result = resolve_article("hour")
        assert result.startswith("an ")

    def test_consonant_sound_u(self):
        result = resolve_article("uniform")
        assert result.startswith("a ")

    def test_empty(self):
        assert resolve_article("") == ""


class TestRenderTemplate:
    """Layer 1: {a_X} template token resolution."""

    def test_basic_article_slot(self):
        result = render_template("{entity} is {a_category}.", entity="cat", category="animal")
        assert result == "cat is an animal."

    def test_article_consonant(self):
        result = render_template("{entity} is {a_category}.", entity="cat", category="dog")
        assert result == "cat is a dog."

    def test_no_article_slots(self):
        result = render_template("{a} plus {b} equals {result}.", a="2", b="3", result="5")
        assert result == "2 plus 3 equals 5."

    def test_multiple_article_slots(self):
        result = render_template(
            "{a_entity} is {a_category}.", entity="elephant", category="animal",
        )
        assert "an elephant" in result.lower()
        assert "an animal" in result.lower()

    def test_mixed_regular_and_article_slots(self):
        result = render_template(
            "The {entity} is {a_category}.", entity="heart", category="organ",
        )
        assert result == "The heart is an organ."


class TestContrastivePairSanitization:
    """Verify sanitizer is applied in ContrastivePair.__post_init__."""

    def _make_pair(self, true_stmt: str, false_stmt: str) -> ContrastivePair:
        return ContrastivePair(
            true_statement=true_stmt,
            false_statement=false_stmt,
            pair_id="test123",
            domain="test",
            relation_type="hypernymy",
            difficulty="easy",
            semantic_distance=5,
            generator="test",
            template_id="t1",
            negation_strategy="sibling_swap",
        )

    def test_whitespace_normalized(self):
        pair = self._make_pair("Dogs  are  mammals", "Dogs  are  rocks")
        assert pair.true_statement == "Dogs are mammals."
        assert pair.false_statement == "Dogs are rocks."

    def test_capitalized(self):
        pair = self._make_pair("dogs are mammals.", "dogs are rocks.")
        assert pair.true_statement == "Dogs are mammals."

    def test_pair_id_unchanged(self):
        pair = self._make_pair("dogs are mammals", "dogs are rocks")
        assert pair.pair_id == "test123"


class TestArticleConsolidation:
    """Verify ad-hoc article handling has been removed from generators."""

    def test_astronomy_type_labels_no_articles(self):
        from latenet.generators.astro_data import TYPE_LABELS

        for key, label in TYPE_LABELS.items():
            assert not label.startswith("a "), f"TYPE_LABELS[{key!r}] still has hardcoded 'a '"
            assert not label.startswith("an "), f"TYPE_LABELS[{key!r}] still has hardcoded 'an '"

    def test_math_properties_no_articles(self):
        from latenet.generators.mathematics import _PROPERTIES

        for key, (label, _checker) in _PROPERTIES.items():
            assert not label.startswith("a "), f"_PROPERTIES[{key!r}] still has hardcoded 'a '"
            assert not label.startswith("an "), f"_PROPERTIES[{key!r}] still has hardcoded 'an '"

    def test_astronomy_renders_correct_article(self):
        from latenet.generators.astro_data import TYPE_LABELS
        result = render_template(
            "{planet} is {a_type}.", planet="Uranus", type=TYPE_LABELS["ice_giant"],
        )
        assert "an ice giant" in result

    def test_math_renders_correct_article(self):
        from latenet.generators.mathematics import _PROPERTIES
        label = _PROPERTIES["perfect_square"][0]
        result = render_template("{number} is {a_property}.", number="9", property=label)
        assert "a perfect square" in result

    def test_biology_membership_templates_use_article_tokens(self):
        from latenet.generators.biology import _MEMBERSHIP_TEMPLATES
        patterns = [t.pattern for t in _MEMBERSHIP_TEMPLATES]
        article_patterns = [p for p in patterns if "{a_organism}" in p or "{a_taxon}" in p]
        assert len(article_patterns) >= 2, "Expected bio templates to use {a_X} tokens"

    def test_no_pattern_format_in_generators(self):
        """Verify no generator uses .pattern.format() directly."""
        import ast
        from pathlib import Path

        gen_dir = Path("latenet/generators")
        for py_file in gen_dir.glob("*.py"):
            if py_file.name in ("__init__.py", "base.py", "geo_data.py",
                                "astro_data.py", "anat_data.py", "math_utils.py",
                                "muse_data.py"):
                continue
            source = py_file.read_text()
            assert ".pattern.format(" not in source, (
                f"{py_file.name} still uses .pattern.format() — should use render_template()"
            )
