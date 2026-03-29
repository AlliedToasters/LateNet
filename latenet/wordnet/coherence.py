"""NDIF logit-guided coherence scoring for WordNet false statement candidates.

Uses Llama 70B base (via lmprobe/NDIF) to score how semantically plausible
each candidate synset is as a completion of "True or false? A {word} is a ___".
Candidates with higher logits produce more natural-sounding false statements.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
from pathlib import Path

from nltk.corpus.reader.wordnet import Synset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LLAMA_MODEL = "meta-llama/Llama-3.1-70B"
LLAMA_TOKENIZER = "meta-llama/Llama-3.1-70B"
DEFAULT_TOP_K = 10000

# Generic classifier synsets that get high logits but make bad false categories.
# "A bridge is a type" is a natural completion but a nonsensical statement.
GENERIC_SYNSET_BLOCKLIST = frozenset({
    # Generic classifier words
    "type.n.01", "type.n.06",
    "kind.n.01",
    "form.n.01", "form.n.03",
    "class.n.01",
    "group.n.01",
    "thing.n.01", "thing.n.04",
    "part.n.01", "part.n.03",
    "set.n.01", "set.n.02",
    "system.n.01",
    "whole.n.01",
    # Abstract/figurative person subtypes that produce awkward pairings
    "personification.n.01", "personification.n.02",
    "personage.n.01",
})

# Culturally sensitive source lemmas that produce uncomfortable pairings
# regardless of the false candidate chosen.
SENSITIVE_SOURCE_LEMMAS = frozenset({
    "jew", "muslim", "christian", "hindu", "buddhist",
    "black", "white", "arab", "asian",
})


def _base_prompt(word: str) -> str:
    """Plain text prompt for base model completion."""
    return f"True or false? A {word} is a"


class CoherenceScorer:
    """Score WordNet candidate synsets using NDIF logit distributions.

    Wraps lmprobe ActivationExtractor for remote logit extraction.
    Caches scores per source word to avoid redundant NDIF calls
    across difficulty tiers (siblings/cousins/distant all share a source).
    """

    def __init__(
        self,
        model: str = LLAMA_MODEL,
        tokenizer_name: str = LLAMA_TOKENIZER,
        top_k: int = DEFAULT_TOP_K,
    ):
        self.model = model
        self.tokenizer_name = tokenizer_name
        self.top_k = top_k

        # Lazy init — only connect to NDIF when first needed
        self._extractor = None
        self._tokenizer = None

        # Cache: source_word → {token_id: logit_value}
        self._logit_cache: dict[str, dict[int, float]] = {}

    def _ensure_initialized(self) -> None:
        """Lazy init extractor and tokenizer on first use."""
        if self._extractor is not None:
            return

        from lmprobe.extraction import ActivationExtractor
        from transformers import AutoTokenizer

        # Bridge NDIF_API_KEY → NNSIGHT_API_KEY
        if "NNSIGHT_API_KEY" not in os.environ and "NDIF_API_KEY" in os.environ:
            os.environ["NNSIGHT_API_KEY"] = os.environ["NDIF_API_KEY"]

        logger.info("Initializing CoherenceScorer: model=%s, top_k=%d", self.model, self.top_k)
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self._extractor = ActivationExtractor(
            model_name=self.model,
            device="cpu",
            layers=[],
            backend="nnsight",
            remote=True,
        )

    def _fetch_logits(self, word: str) -> dict[int, float]:
        """Fetch top-k logit distribution for a source word from NDIF.

        Returns dict mapping vocab token_id → logit value for the last
        position of the prompt "True or false? A {word} is a".
        """
        from lmprobe.retry import retry_with_backoff

        self._ensure_initialized()

        prompt = _base_prompt(word)
        logits, _mask, logits_indices = retry_with_backoff(
            lambda p=prompt: self._extractor.extract_logits_only(
                [p], remote=True, logit_top_k=self.top_k,
            ),
            max_retries=3,
            base_delay=3.0,
            max_delay=60.0,
            context=f"coherence:{word}",
        )

        # Last position, top-k
        last_logits = logits[0, -1, :]
        last_indices = logits_indices[0, -1, :]

        lookup = {}
        for i, vid in enumerate(last_indices.tolist()):
            lookup[vid] = last_logits[i].item()

        logger.debug("Fetched %d logits for '%s'", len(lookup), word)
        return lookup

    def _get_logits(self, word: str) -> dict[int, float]:
        """Get logit distribution for word, using cache."""
        if word not in self._logit_cache:
            self._logit_cache[word] = self._fetch_logits(word)
        return self._logit_cache[word]

    def score_candidates(
        self,
        source_word: str,
        candidates: list[Synset],
    ) -> list[tuple[Synset, float]]:
        """Score candidate synsets by logit plausibility.

        Parameters
        ----------
        source_word : str
            The source entity (e.g., "dog").
        candidates : list[Synset]
            WordNet synsets to score as false categories.

        Returns
        -------
        list[tuple[Synset, float]]
            (synset, logit) pairs, sorted by logit descending.
            Candidates outside top-k get a floor score.
            Generic blocklisted synsets are excluded.
        """
        if source_word.lower() in SENSITIVE_SOURCE_LEMMAS:
            return []

        self._ensure_initialized()
        logit_lookup = self._get_logits(source_word)

        scored = []
        floor_logit = None

        for syn in candidates:
            if syn.name() in GENERIC_SYNSET_BLOCKLIST:
                continue

            lemma = syn.lemma_names()[0].replace("_", " ")
            token_ids = self._tokenizer.encode(f" {lemma}", add_special_tokens=False)
            first_tid = token_ids[0]

            if first_tid in logit_lookup:
                scored.append((syn, logit_lookup[first_tid]))
            else:
                # Track for floor assignment
                scored.append((syn, None))

        # Assign floor logit to candidates outside top-k
        real_logits = [l for _, l in scored if l is not None]
        if real_logits:
            floor_logit = min(real_logits) - 1.0
        else:
            floor_logit = 0.0

        scored = [(syn, logit if logit is not None else floor_logit) for syn, logit in scored]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def generate_false_candidates(
        self,
        source_word: str,
        source_synset: Synset,
        decode_top_n: int = 200,
    ) -> list[tuple[Synset, float]]:
        """Generate false candidate synsets from the model's top completions.

        Inverted pipeline: model generates plausible completions for
        "True or false? A {word} is a ___", then WordNet validates
        which ones are genuinely false (not in source's hypernym chain).

        Returns (synset, logit) pairs sorted by logit descending.
        Only includes primary-sense noun synsets not in the hypernym closure.
        """
        from nltk.corpus import wordnet as wn
        from latenet.difficulty.tiers import _is_physical_entity

        if source_word.lower() in SENSITIVE_SOURCE_LEMMAS:
            return []

        self._ensure_initialized()
        logit_lookup = self._get_logits(source_word)

        # Sort tokens by logit descending
        sorted_tokens = sorted(logit_lookup.items(), key=lambda x: x[1], reverse=True)

        # Build exclusion set: hypernym closure (ancestors) + hyponym closure
        # (descendants). Bidirectional check prevents near-misses like
        # "county is a type of region" (region is ancestor) and
        # "pond is a type of lake" (lake is ancestor — pond is hyponym of lake).
        exclusion_set: set[str] = set()
        # Upward: all ancestors
        queue = [source_synset]
        while queue:
            s = queue.pop()
            if s.name() in exclusion_set:
                continue
            exclusion_set.add(s.name())
            queue.extend(s.hypernyms())
        # Downward: immediate hyponyms (1 level — full closure too expensive)
        for child in source_synset.hyponyms():
            exclusion_set.add(child.name())

        source_lemmas = {l.name().lower() for l in source_synset.lemmas()}

        # Collect lemmas from the hypernym chain for substring near-miss check.
        # Catches "chlorine is a chemical" where chemical.n.01 isn't in the
        # chain but chemical_element.n.01 IS — "chemical" is a substring.
        chain_lemmas: set[str] = set()
        for name in exclusion_set:
            try:
                syn_obj = wn.synset(name)
                for lem in syn_obj.lemmas():
                    chain_lemmas.add(lem.name().lower().replace("_", " "))
            except Exception:
                pass

        scored: list[tuple[Synset, float]] = []
        seen_synsets: set[str] = set()

        for tid, logit_val in sorted_tokens[:decode_top_n]:
            decoded = self._tokenizer.decode([tid]).strip()
            word = decoded.strip(".,;:!?\"'()[]{}").lower()
            if not word or len(word) < 2:
                continue

            # Only take the primary (first) noun sense — avoids obscure-sense problem
            synsets = wn.synsets(word, pos="n")
            if not synsets:
                continue
            syn = synsets[0]

            if syn.name() in seen_synsets:
                continue
            seen_synsets.add(syn.name())

            # Skip if ANY sense of this word is in the exclusion set.
            # Prevents polysemy leaks: "canine" maps to tooth (n.01) but
            # readers interpret it as the animal (n.02), which IS true.
            if any(s.name() in exclusion_set for s in synsets):
                continue
            # Skip TRUE candidates (in hypernym/hyponym chain)
            if syn.name() in exclusion_set:
                continue
            # Skip taxonomically close candidates that humans perceive as
            # true even if not in the exact hypernym chain. WuP ≥ 0.7
            # catches cases like "chlorine is a chemical" (wup=0.57 is
            # fine, but county/region at 0.91 is not).
            wup = source_synset.wup_similarity(syn)
            if wup is not None and wup >= 0.7:
                continue
            # Skip blocklisted generic classifiers
            if syn.name() in GENERIC_SYNSET_BLOCKLIST:
                continue
            # Skip lemma overlap with source (tautologies)
            if source_lemmas & {l.name().lower() for l in syn.lemmas()}:
                continue
            # Skip non-physical-entity synsets (abstract nonsense)
            if not _is_physical_entity(syn):
                continue
            # Skip lemma-level near-misses: if candidate lemma is a
            # substring of any chain lemma (or vice versa), it's too close.
            # "chemical" ⊂ "chemical element" → skip.
            cand_lemma = syn.lemma_names()[0].lower().replace("_", " ")
            if any(
                (cand_lemma in cl and len(cand_lemma) >= 4) or
                (cl in cand_lemma and len(cl) >= 4)
                for cl in chain_lemmas
            ):
                continue

            scored.append((syn, logit_val))

        scored.sort(key=lambda x: x[1], reverse=True)
        logger.debug(
            "Inverted pipeline for '%s': %d usable false candidates from top-%d tokens",
            source_word, len(scored), decode_top_n,
        )
        return scored

    def source_is_coherent(
        self,
        source_word: str,
        hypernym_synsets: list[Synset],
        top_n: int = 20,
    ) -> bool:
        """Check if the model associates source_word with its WordNet hypernyms.

        Returns True if any hypernym's first token appears in the model's
        top_n completions for "True or false? A {source_word} is a ___".
        If the model doesn't associate the word with this taxonomic sense,
        the source synset is being used in an obscure meaning and will
        produce awkward statements.
        """
        if not hypernym_synsets:
            return True  # Can't check — allow

        self._ensure_initialized()
        logit_lookup = self._get_logits(source_word)

        # Get top_n token IDs by logit
        sorted_tids = sorted(logit_lookup, key=lambda t: logit_lookup[t], reverse=True)[:top_n]
        top_set = set(sorted_tids)

        for hyp in hypernym_synsets:
            for lemma_obj in hyp.lemmas():
                lemma = lemma_obj.name().replace("_", " ")
                tids = self._tokenizer.encode(f" {lemma}", add_special_tokens=False)
                if tids[0] in top_set:
                    return True

        return False

    def close(self) -> None:
        """Release resources."""
        self._extractor = None
        self._tokenizer = None
        self._logit_cache.clear()


def softmax_sample(
    scored: list[tuple[Synset, float]],
    rng: random.Random,
    temperature: float = 0.5,
) -> Synset | None:
    """Sample a synset from scored candidates using softmax over logits.

    Temperature < 1.0 sharpens the distribution toward high-logit candidates.
    Temperature = 1.0 is standard softmax. Temperature > 1.0 flattens.
    Default 0.5 concentrates on plausible candidates without hard cutoffs.
    """
    if not scored:
        return None
    max_logit = max(l for _, l in scored)
    weights = [math.exp((l - max_logit) / temperature) for _, l in scored]
    return rng.choices([s for s, _ in scored], weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Cache utilities
# ---------------------------------------------------------------------------

def save_coherence_cache(scorer: CoherenceScorer, path: Path) -> None:
    """Save the scorer's logit cache to disk."""
    with open(path, "w") as f:
        json.dump(scorer._logit_cache, f)
    logger.info("Saved coherence cache (%d words) to %s", len(scorer._logit_cache), path)


def load_coherence_cache(scorer: CoherenceScorer, path: Path) -> int:
    """Load a previously saved logit cache into the scorer. Returns count loaded."""
    if not path.exists():
        return 0
    with open(path) as f:
        data = json.load(f)
    # JSON keys are strings; token IDs need to be ints
    for word, lookup in data.items():
        scorer._logit_cache[word] = {int(k): v for k, v in lookup.items()}
    logger.info("Loaded coherence cache (%d words) from %s", len(data), path)
    return len(data)
