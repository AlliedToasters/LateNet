"""Tests for the grammar pipeline (Layers 2 & 3).

Layer 2 (GrammarScanner) requires Java — tests skip if unavailable.
Layer 3 (GrammarCorrector) requires ANTHROPIC_API_KEY — tests skip if unavailable.
"""

from __future__ import annotations

import json

import pytest

from latenet.quality.grammar import (
    CorrectionResult,
    GrammarCorrector,
    GrammarFlag,
    GrammarScanner,
)
from latenet.types import ContrastivePair


# ---------------------------------------------------------------------------
# Layer 2: GrammarScanner
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scanner():
    s = GrammarScanner()
    if not s.available:
        pytest.skip("LanguageTool unavailable (Java not installed)")
    return s


class TestGrammarScanner:
    def test_clean_sentence_no_flags(self, scanner):
        flags = scanner.scan("The cat is a mammal.")
        assert flags == []

    def test_detects_article_error(self, scanner):
        flags = scanner.scan("The cat is a animal.")
        rule_ids = [f.rule_id for f in flags]
        assert any("EN_A_VS_AN" in r for r in rule_ids), f"Expected EN_A_VS_AN, got {rule_ids}"

    def test_does_not_flag_passive_voice(self, scanner):
        flags = scanner.scan("The ball was thrown by the player.")
        rule_ids = [f.rule_id for f in flags]
        assert "PASSIVE_VOICE" not in rule_ids

    def test_does_not_flag_sentence_fragment(self, scanner):
        flags = scanner.scan("Dogs are mammals.")
        assert flags == []

    def test_scan_pair(self, scanner):
        pair = ContrastivePair(
            true_statement="The cat is a mammal.",
            false_statement="The cat is a mineral.",
            pair_id="test123",
            domain="test",
            relation_type="hypernymy",
            difficulty="easy",
            semantic_distance=5,
            generator="test",
            template_id="t1",
            negation_strategy="sibling_swap",
        )
        result = scanner.scan_pair(pair)
        assert "true_flags" in result
        assert "false_flags" in result
        assert "needs_correction" in result
        assert isinstance(result["needs_correction"], bool)

    def test_unavailable_returns_empty(self):
        """A scanner with no Java should return empty flags."""
        s = GrammarScanner.__new__(GrammarScanner)
        s._tool = None
        s._available = False
        s._disabled_rules = frozenset()
        assert s.scan("Bad grammar is.") == []


# ---------------------------------------------------------------------------
# Layer 3: GrammarCorrector — parse_response unit tests
# ---------------------------------------------------------------------------

class TestCorrectorParsing:
    """Unit-test the response parser without hitting the API."""

    def _make_corrector(self):
        c = GrammarCorrector.__new__(GrammarCorrector)
        c.batch_size = 50
        c.model = "test"
        c.max_retries = 1
        c._client = None
        c._available = False
        return c

    def test_parses_corrected(self):
        c = self._make_corrector()
        raw = json.dumps({
            "statements": [{
                "index": 0,
                "original": "A apple is round.",
                "corrected": "An apple is round.",
                "status": "corrected",
                "changes": "Fixed article",
            }]
        })
        results = c._parse_response(raw, ["A apple is round."])
        assert len(results) == 1
        assert results[0].status == "corrected"
        assert results[0].corrected == "An apple is round."

    def test_parses_unchanged(self):
        c = self._make_corrector()
        raw = json.dumps({
            "statements": [{
                "index": 0,
                "original": "The cat sat.",
                "status": "unchanged",
                "changes": None,
            }]
        })
        results = c._parse_response(raw, ["The cat sat."])
        assert results[0].status == "unchanged"
        assert results[0].corrected is None

    def test_parses_flagged(self):
        c = self._make_corrector()
        raw = json.dumps({
            "statements": [{
                "index": 0,
                "original": "Is word order bad.",
                "status": "flagged",
                "changes": "Too malformed",
            }]
        })
        results = c._parse_response(raw, ["Is word order bad."])
        assert results[0].status == "flagged"
        assert results[0].changes == "Too malformed"

    def test_missing_index_falls_back(self):
        c = self._make_corrector()
        raw = json.dumps({"statements": []})
        results = c._parse_response(raw, ["Hello world."])
        assert len(results) == 1
        assert results[0].status == "unchanged"

    def test_invalid_json_falls_back(self):
        c = self._make_corrector()
        results = c._parse_response("not json at all", ["Hello."])
        assert len(results) == 1
        assert results[0].status == "unchanged"

    def test_rejects_wildly_different_length(self):
        c = self._make_corrector()
        raw = json.dumps({
            "statements": [{
                "index": 0,
                "original": "Short.",
                "corrected": "This is a very long correction that adds way too much content and is wildly different.",
                "status": "corrected",
                "changes": "Rewrote everything",
            }]
        })
        results = c._parse_response(raw, ["Short."])
        assert results[0].status == "unchanged"

    def test_batch_multiple(self):
        c = self._make_corrector()
        raw = json.dumps({
            "statements": [
                {"index": 0, "original": "A apple.", "corrected": "An apple.", "status": "corrected", "changes": "article"},
                {"index": 1, "original": "Good.", "status": "unchanged", "changes": None},
            ]
        })
        results = c._parse_response(raw, ["A apple.", "Good."])
        assert len(results) == 2
        assert results[0].status == "corrected"
        assert results[1].status == "unchanged"

    def test_unavailable_corrector_returns_unchanged(self):
        c = self._make_corrector()
        results = c.correct_batch(["Hello."])
        assert len(results) == 1
        assert results[0].status == "unchanged"
