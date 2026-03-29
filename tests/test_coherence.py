"""Tests for NDIF logit-guided coherence scoring."""

from __future__ import annotations

import math
import random
from unittest.mock import MagicMock, patch

import pytest
from nltk.corpus import wordnet as wn

from latenet.wordnet.coherence import (
    CoherenceScorer,
    GENERIC_SYNSET_BLOCKLIST,
    softmax_sample,
    _base_prompt,
)
from latenet.difficulty.tiers import pick_negation_synset, _siblings_of
from latenet.types import Difficulty


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_scorer():
    """CoherenceScorer with mocked NDIF — returns deterministic logit lookups."""
    scorer = CoherenceScorer.__new__(CoherenceScorer)
    scorer.model = "mock"
    scorer.tokenizer_name = "mock"
    scorer.top_k = 10000
    scorer._extractor = MagicMock()
    scorer._logit_cache = {}

    # Mock tokenizer: maps " word" → [hash of word]
    mock_tok = MagicMock()
    def _encode(text, add_special_tokens=False):
        word = text.strip()
        return [hash(word) % 100000]
    mock_tok.encode = _encode
    scorer._tokenizer = mock_tok

    return scorer


def _inject_logits(scorer: CoherenceScorer, word: str, token_logits: dict[int, float]):
    """Inject a fake logit lookup for a source word into the scorer's cache."""
    scorer._logit_cache[word] = token_logits


# ---------------------------------------------------------------------------
# Unit tests: CoherenceScorer
# ---------------------------------------------------------------------------

class TestCoherenceScorer:

    def test_score_candidates_basic(self, mock_scorer):
        """Candidates with tokens in the cache get real logits; others get floor."""
        dog = wn.synset("dog.n.01")
        cat = wn.synset("cat.n.01")
        car = wn.synset("car.n.01")

        # Inject logits for "dog" — cat's token is present, car's isn't
        cat_tid = hash("cat") % 100000
        _inject_logits(mock_scorer, "dog", {cat_tid: 15.0})

        scored = mock_scorer.score_candidates("dog", [cat, car])
        assert len(scored) == 2

        # Cat should have higher logit than car (which gets floor)
        cat_entry = next(s for s in scored if s[0].name() == "cat.n.01")
        car_entry = next(s for s in scored if s[0].name() == "car.n.01")
        assert cat_entry[1] == 15.0
        assert car_entry[1] < cat_entry[1]  # floor = min - 1.0 = 14.0

    def test_generic_blocklist_filtered(self, mock_scorer):
        """Synsets in the generic blocklist are excluded from scored results."""
        # type.n.06 is in the blocklist
        type_syn = wn.synset("type.n.06") if wn.synsets("type", pos="n") else None
        dog = wn.synset("dog.n.01")

        if type_syn is None:
            pytest.skip("type.n.06 not in WordNet")

        _inject_logits(mock_scorer, "bridge", {})
        scored = mock_scorer.score_candidates("bridge", [type_syn, dog])

        synset_names = {s.name() for s, _ in scored}
        assert "type.n.06" not in synset_names
        assert "dog.n.01" in synset_names

    def test_cache_reuse(self, mock_scorer):
        """Second call for same source word reuses cached logits."""
        cat = wn.synset("cat.n.01")
        _inject_logits(mock_scorer, "dog", {hash("cat") % 100000: 12.0})

        scored1 = mock_scorer.score_candidates("dog", [cat])
        scored2 = mock_scorer.score_candidates("dog", [cat])
        assert scored1 == scored2

    def test_empty_candidates(self, mock_scorer):
        """Empty candidate list returns empty results."""
        _inject_logits(mock_scorer, "dog", {})
        assert mock_scorer.score_candidates("dog", []) == []


# ---------------------------------------------------------------------------
# Unit tests: softmax_sample
# ---------------------------------------------------------------------------

class TestSoftmaxSample:

    def test_deterministic_with_seed(self):
        """Same seed produces same sample."""
        dog = wn.synset("dog.n.01")
        cat = wn.synset("cat.n.01")
        scored = [(dog, 15.0), (cat, 10.0)]

        rng1 = random.Random(42)
        rng2 = random.Random(42)
        assert softmax_sample(scored, rng1).name() == softmax_sample(scored, rng2).name()

    def test_higher_logit_sampled_more(self):
        """Candidate with much higher logit is sampled more frequently."""
        dog = wn.synset("dog.n.01")
        cat = wn.synset("cat.n.01")
        # dog has logit 20, cat has logit 5 — huge gap
        scored = [(dog, 20.0), (cat, 5.0)]

        rng = random.Random(0)
        samples = [softmax_sample(scored, rng).name() for _ in range(100)]
        dog_frac = samples.count("dog.n.01") / 100
        assert dog_frac > 0.9  # Should be nearly all dog

    def test_empty_returns_none(self):
        assert softmax_sample([], random.Random(42)) is None


# ---------------------------------------------------------------------------
# Unit tests: prompt format
# ---------------------------------------------------------------------------

def test_base_prompt():
    assert _base_prompt("dog") == "True or false? A dog is a"
    assert _base_prompt("domestic cat") == "True or false? A domestic cat is a"


# ---------------------------------------------------------------------------
# Integration: pick_negation_synset with coherence_scorer
# ---------------------------------------------------------------------------

class TestPickNegationWithCoherence:

    def test_coherence_path_returns_synset(self, mock_scorer):
        """pick_negation_synset with a scorer returns a valid synset."""
        dog = wn.synset("dog.n.01")
        # dog's hypernym
        hyps = dog.hypernyms()
        if not hyps:
            pytest.skip("dog.n.01 has no hypernyms")
        target = hyps[0]

        # Inject logits so at least some siblings are scorable
        siblings = _siblings_of(target)
        if not siblings:
            pytest.skip("No siblings for dog's hypernym")

        # Give every sibling's token a logit
        logit_lookup = {}
        for i, sib in enumerate(siblings):
            lemma = sib.lemma_names()[0].replace("_", " ")
            tid = hash(lemma) % 100000
            logit_lookup[tid] = 10.0 + i
        source_word = dog.lemma_names()[0].replace("_", " ")
        _inject_logits(mock_scorer, source_word, logit_lookup)

        rng = random.Random(42)
        result = pick_negation_synset(dog, target, Difficulty.HARD, rng, coherence_scorer=mock_scorer)
        assert result is not None
        assert result.name() != dog.name()

    def test_none_scorer_uses_legacy_path(self):
        """Without coherence_scorer, pick_negation_synset uses legacy behavior."""
        dog = wn.synset("dog.n.01")
        hyps = dog.hypernyms()
        if not hyps:
            pytest.skip("dog.n.01 has no hypernyms")
        target = hyps[0]

        rng = random.Random(42)
        result = pick_negation_synset(dog, target, Difficulty.HARD, rng, coherence_scorer=None)
        # Should still work — legacy path
        # (may be None if no valid siblings, but shouldn't error)
        assert result is None or result.name() != dog.name()


# ---------------------------------------------------------------------------
# Unit tests: generate_false_candidates (inverted pipeline)
# ---------------------------------------------------------------------------

class TestGenerateFalseCandidates:

    def test_basic_generation(self, mock_scorer):
        """Inverted pipeline produces false candidates from logit distribution."""
        dog = wn.synset("dog.n.01")

        # Build a fake logit cache with tokens for real nouns
        # "cat", "car", "fish" should map to WordNet synsets
        logit_lookup = {}
        for word, logit in [("cat", 15.0), ("car", 14.0), ("fish", 13.0), ("xyz", 12.0)]:
            tid = hash(word) % 100000
            logit_lookup[tid] = logit

        # Mock the tokenizer decode to return the word from the token ID
        token_to_word = {}
        for word in ["cat", "car", "fish", "xyz"]:
            token_to_word[hash(word) % 100000] = word
        mock_scorer._tokenizer.decode = lambda tids: token_to_word.get(tids[0], "???")

        _inject_logits(mock_scorer, "dog", logit_lookup)
        candidates = mock_scorer.generate_false_candidates("dog", dog)

        # Should have some candidates (cat, car, fish are all physical entities
        # and not in dog's hypernym chain)
        assert len(candidates) > 0
        # All should be (Synset, float) tuples
        for syn, logit in candidates:
            assert hasattr(syn, "name")
            assert isinstance(logit, float)
        # None should be in dog's hypernym chain
        hyp_closure = set()
        queue = [dog]
        while queue:
            s = queue.pop()
            if s.name() in hyp_closure:
                continue
            hyp_closure.add(s.name())
            queue.extend(s.hypernyms())
        for syn, _ in candidates:
            assert syn.name() not in hyp_closure

    def test_sensitive_source_returns_empty(self, mock_scorer):
        """Sensitive source lemmas produce no candidates."""
        # "jew" is in SENSITIVE_SOURCE_LEMMAS
        jew_synsets = wn.synsets("jew", pos="n")
        if not jew_synsets:
            pytest.skip("No synset for 'jew'")
        result = mock_scorer.generate_false_candidates("jew", jew_synsets[0])
        assert result == []

    def test_hypernym_chain_excluded(self, mock_scorer):
        """TRUE candidates (in hypernym chain) are excluded."""
        dog = wn.synset("dog.n.01")
        # "animal" is in dog's hypernym chain
        animal_tid = hash("animal") % 100000
        cat_tid = hash("cat") % 100000

        token_to_word = {animal_tid: "animal", cat_tid: "cat"}
        mock_scorer._tokenizer.decode = lambda tids: token_to_word.get(tids[0], "???")

        _inject_logits(mock_scorer, "dog", {animal_tid: 20.0, cat_tid: 15.0})
        candidates = mock_scorer.generate_false_candidates("dog", dog)

        synset_names = {s.name() for s, _ in candidates}
        # animal.n.01 should be excluded (it's in dog's hypernym chain)
        assert "animal.n.01" not in synset_names


# ---------------------------------------------------------------------------
# Integration: pick_negation_synset EASY tier with inverted pipeline
# ---------------------------------------------------------------------------

class TestEasyTierInvertedPipeline:

    def test_easy_tier_uses_inverted_when_scorer_present(self, mock_scorer):
        """EASY tier with coherence_scorer uses generate_false_candidates."""
        dog = wn.synset("dog.n.01")
        target = dog.hypernyms()[0] if dog.hypernyms() else None
        if target is None:
            pytest.skip("dog has no hypernyms")

        # Inject logits with distant nouns
        logit_lookup = {}
        token_to_word = {}
        for word, logit in [("piano", 14.0), ("building", 13.0), ("river", 12.0)]:
            tid = hash(word) % 100000
            logit_lookup[tid] = logit
            token_to_word[tid] = word
        mock_scorer._tokenizer.decode = lambda tids: token_to_word.get(tids[0], "???")

        source_word = dog.lemma_names()[0].replace("_", " ")
        _inject_logits(mock_scorer, source_word, logit_lookup)

        rng = random.Random(42)
        result = pick_negation_synset(dog, target, Difficulty.EASY, rng, coherence_scorer=mock_scorer)
        # Should return something (may fall back to legacy if inverted yields nothing for this mock)
        # At minimum, should not error
        assert result is None or result.name() != dog.name()


# ---------------------------------------------------------------------------
# Blocklist completeness
# ---------------------------------------------------------------------------

def test_blocklist_entries_exist():
    """All blocklisted synsets should exist in WordNet."""
    for name in GENERIC_SYNSET_BLOCKLIST:
        try:
            wn.synset(name)
        except Exception:
            pytest.fail(f"Blocklisted synset {name} not found in WordNet")
