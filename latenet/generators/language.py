"""Translation equivalence across language pairs.

Data source: Meta's MUSE bilingual dictionaries (via muse_data).
Relations: translates_to, translation_of, word_is_language.
Domains: language.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd

from latenet.generators.base import BaseGenerator
from latenet.generators.muse_data import (
    DEFAULT_TARGET_LANGUAGES,
    LANGUAGE_DISPLAY_NAMES,
    load_dictionary,
    tag_domains,
)
from latenet.sanitize import render_template
from latenet.types import ContrastivePair, Difficulty, NegationStrategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Language family groupings for difficulty tiers
# ---------------------------------------------------------------------------

# Sub-family groupings
LANGUAGE_FAMILIES: dict[str, str] = {
    "es": "romance", "fr": "romance", "pt": "romance", "it": "romance",
    "de": "germanic", "en": "germanic",
    "ru": "slavic",
    "zh": "sino_tibetan",
    "ja": "japonic",
    "hi": "indo_aryan",
    "ar": "semitic",
}

# Macro-family: Indo-European includes romance, germanic, slavic, indo_aryan
_INDO_EUROPEAN = {"romance", "germanic", "slavic", "indo_aryan"}


def _language_distance(lang_a: str, lang_b: str) -> int:
    """Compute distance between two languages for difficulty assignment.

    Returns: 1 = same sub-family (hard), 2 = same macro-family (medium), 3 = different (easy).
    """
    fam_a = LANGUAGE_FAMILIES.get(lang_a, "unknown_a")
    fam_b = LANGUAGE_FAMILIES.get(lang_b, "unknown_b")
    if fam_a == fam_b:
        return 1  # same sub-family
    if fam_a in _INDO_EUROPEAN and fam_b in _INDO_EUROPEAN:
        return 2  # same macro-family
    return 3  # different macro-family


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LangTemplate:
    id: str
    relation: str
    pattern: str


_TRANSLATE_TEMPLATES = [
    LangTemplate("lang_translate_01", "translates_to",
                 "The {language} word for {english} is {translation}."),
    LangTemplate("lang_translate_02", "translates_to",
                 "In {language}, {english} is {translation}."),
    LangTemplate("lang_translate_03", "translates_to",
                 "{english} in {language} is {translation}."),
    LangTemplate("lang_translate_04", "translates_to",
                 "The {language} translation of {english} is {translation}."),
]

_REVERSE_TEMPLATES = [
    LangTemplate("lang_reverse_01", "translation_of",
                 "{translation} is the {language} word for {english}."),
    LangTemplate("lang_reverse_02", "translation_of",
                 "{translation} means {english} in {language}."),
    LangTemplate("lang_reverse_03", "translation_of",
                 "In {language}, {translation} means {english}."),
]

_IDENTIFY_TEMPLATES = [
    LangTemplate("lang_identify_01", "word_is_language",
                 "{word} is a {language} word."),
    LangTemplate("lang_identify_02", "word_is_language",
                 "{word} is a word in {language}."),
    LangTemplate("lang_identify_03", "word_is_language",
                 "The word {word} comes from {language}."),
]

_ALL_TEMPLATES: dict[str, list[LangTemplate]] = {
    "translates_to": _TRANSLATE_TEMPLATES,
    "translation_of": _REVERSE_TEMPLATES,
    "word_is_language": _IDENTIFY_TEMPLATES,
}


def _make_pair_id(parts: list[str]) -> str:
    key = ":".join(parts)
    return hashlib.sha256(key.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class LanguageGenerator(BaseGenerator):
    """Generate contrastive pairs from translation equivalences."""

    def __init__(
        self,
        seed: int = 42,
        max_pairs: int | None = None,
        target_languages: list[str] | None = None,
        per_language_cap: int | None = None,
        per_domain_cap: int | None = None,
        force_refresh: bool = False,
    ):
        super().__init__(seed=seed, max_pairs=max_pairs)
        self.target_languages = target_languages or DEFAULT_TARGET_LANGUAGES
        self.per_language_cap = per_language_cap
        self.per_domain_cap = per_domain_cap
        self.force_refresh = force_refresh

        # Loaded lazily
        self._data: pd.DataFrame | None = None
        # Lookup: (source_word, target_lang) -> target_word
        self._translation_map: dict[tuple[str, str], str] = {}
        # Lookup: domain -> list of source_words in that domain
        self._domain_entities: dict[str, list[str]] = {}
        # Lookup: target_lang -> set of target_words (for cognate detection)
        self._lang_words: dict[str, set[str]] = {}
        # All unique source words
        self._all_english: list[str] = []
        # Lookup: source_word -> domain
        self._entity_domains: dict[str, str] = {}

    @property
    def name(self) -> str:
        return "language"

    def relation_types(self) -> list[str]:
        return ["translates_to", "translation_of", "word_is_language"]

    def domains(self) -> list[str]:
        return ["language"]

    # --- Data loading ---

    def _load_data(self) -> None:
        if self._data is not None:
            return

        logger.info("Loading translation data from MUSE dictionaries...")
        dfs = []
        for lang in self.target_languages:
            try:
                df = load_dictionary(
                    source_lang="en",
                    target_lang=lang,
                    force_refresh=self.force_refresh,
                )
                dfs.append(df)
            except Exception:
                logger.warning("Failed to load MUSE dictionary for en-%s, skipping", lang)

        if not dfs:
            self._data = pd.DataFrame()
            return

        combined = pd.concat(dfs, ignore_index=True)

        # Tag domains via WordNet (best-effort)
        combined = tag_domains(combined)

        self._data = combined
        self._build_lookups(combined)

        logger.info(
            "Translation data loaded: %d pairs, %d entities, %d languages, %d domains",
            len(combined), len(self._all_english),
            combined["target_lang"].nunique(), len(self._domain_entities),
        )
        for domain, entities in sorted(self._domain_entities.items()):
            logger.info("  %s: %d entities", domain, len(entities))

    def _build_lookups(self, df: pd.DataFrame) -> None:
        """Build internal lookup structures from the loaded DataFrame."""
        self._translation_map = {}
        self._domain_entities = {}
        self._lang_words = {}
        self._entity_domains = {}

        # Filter out proper nouns, obscure terms, and unattested words
        _excluded = self._filter_obscure_words(df)

        for _, row in df.iterrows():
            en = str(row["source_word"])
            lang = str(row["target_lang"])
            target = str(row["target_word"])
            domain = str(row.get("domain", "other"))

            if en.lower() in _excluded:
                continue

            self._translation_map[(en, lang)] = target
            self._domain_entities.setdefault(domain, [])
            if en not in self._entity_domains:
                self._domain_entities[domain].append(en)
            self._entity_domains[en] = domain
            self._lang_words.setdefault(lang, set()).add(target.lower())

        self._all_english = sorted(set(self._entity_domains.keys()))

    @staticmethod
    def _filter_obscure_words(df: pd.DataFrame) -> set[str]:
        """Filter source words to well-known English vocabulary only.

        Uses WordNet synset existence and lemma frequency counts (from the
        Brown corpus) to exclude proper nouns, obscure terms, and jargon.
        This is a one-time quality gate on the MUSE data.

        Returns set of lowercase words to EXCLUDE.
        """
        from nltk.corpus import wordnet as wn

        candidates = set(df["source_word"].str.lower().unique())
        exclude = set()
        for word in candidates:
            if len(word) <= 2:
                exclude.add(word)
                continue
            if any(c.isdigit() for c in word):
                exclude.add(word)
                continue
            # Must have a WordNet synset (noun or verb)
            synsets = wn.synsets(word, pos=wn.NOUN) + wn.synsets(word, pos=wn.VERB)
            if not synsets:
                exclude.add(word)
                continue
            # Must have non-zero lemma frequency in at least one synset
            # (attested in the Brown corpus — filters out obscure/technical terms)
            max_freq = max(
                (lemma.count() for ss in synsets for lemma in ss.lemmas()
                 if lemma.name().lower() == word),
                default=0,
            )
            if max_freq == 0:
                exclude.add(word)

        return exclude

    # --- Helpers ---

    def _pick_template(self, relation: str) -> LangTemplate:
        templates = _ALL_TEMPLATES[relation]
        return templates[self.rng.randint(0, len(templates) - 1)]

    def _lang_display(self, lang_code: str) -> str:
        return LANGUAGE_DISPLAY_NAMES.get(lang_code, lang_code)

    def _get_translation(self, english: str, lang: str) -> str | None:
        return self._translation_map.get((english, lang))

    def _pick_same_domain_swap(self, english: str, lang: str) -> str | None:
        """Pick a swap translation from the same entity domain."""
        domain = self._entity_domains.get(english)
        if not domain:
            return None
        candidates = self._domain_entities.get(domain, [])
        valid = [
            e for e in candidates
            if e != english and (e, lang) in self._translation_map
        ]
        if not valid:
            return None
        pick = self.rng.choice(valid)
        return self._translation_map[(pick, lang)]

    def _pick_different_domain_swap(self, english: str, lang: str) -> str | None:
        """Pick a swap translation from a different entity domain."""
        domain = self._entity_domains.get(english)
        other_domains = [d for d in self._domain_entities if d != domain]
        if not other_domains:
            return None
        swap_domain = self.rng.choice(other_domains)
        candidates = [
            e for e in self._domain_entities[swap_domain]
            if (e, lang) in self._translation_map
        ]
        if not candidates:
            return None
        pick = self.rng.choice(candidates)
        return self._translation_map[(pick, lang)]

    def _pick_wrong_language_swap(self, english: str, lang: str) -> str | None:
        """Pick a translation of the same word but from a different language (easy tier)."""
        other_langs = [l for l in self.target_languages if l != lang]
        self.rng.shuffle(other_langs)
        for other_lang in other_langs:
            t = self._get_translation(english, other_lang)
            true_t = self._get_translation(english, lang)
            if t and true_t and t.lower() != true_t.lower():
                return t
        return None

    def _is_unique_to_language(self, word: str, true_lang: str) -> bool:
        """Check if a target word appears only in one language (not a cognate)."""
        word_lower = word.lower()
        for lang, words in self._lang_words.items():
            if lang != true_lang and word_lower in words:
                return False
        return True

    # --- Main generate ---

    def generate(self) -> Iterator[ContrastivePair]:
        self._load_data()
        if self._data is None or self._data.empty:
            return

        count = 0
        generators = [
            self._generate_translates_to,
            self._generate_translation_of,
            self._generate_word_is_language,
        ]

        for gen_fn in generators:
            for pair in gen_fn():
                yield pair
                count += 1
                if self.max_pairs is not None and count >= self.max_pairs:
                    return

    # --- translates_to ---

    def _generate_translates_to(self) -> Iterator[ContrastivePair]:
        """Generate 'The {language} word for {english} is {translation}' pairs."""
        entities = list(self._all_english)
        self.rng.shuffle(entities)

        per_lang_counts: dict[str, int] = {}
        per_domain_counts: dict[str, int] = {}

        for english in entities:
            domain = self._entity_domains[english]

            langs = list(self.target_languages)
            self.rng.shuffle(langs)

            for lang in langs:
                true_translation = self._get_translation(english, lang)
                if not true_translation:
                    continue

                # Enforce per-language and per-domain caps
                if self.per_language_cap and per_lang_counts.get(lang, 0) >= self.per_language_cap:
                    continue
                if self.per_domain_cap and per_domain_counts.get(domain, 0) >= self.per_domain_cap:
                    continue

                # Generate at each difficulty tier
                swaps = self._pick_translates_to_swaps(english, lang, true_translation)
                for swap_translation, difficulty, strategy in swaps:
                    template = self._pick_template("translates_to")
                    lang_name = self._lang_display(lang)

                    true_stmt = render_template(template.pattern,
                        language=lang_name, english=english, translation=true_translation,
                    )
                    false_stmt = render_template(template.pattern,
                        language=lang_name, english=english, translation=swap_translation,
                    )

                    pair_id = _make_pair_id([
                        "lang", "translate", english, lang,
                        true_translation, swap_translation, template.id,
                    ])

                    sem_dist = 1 if difficulty == Difficulty.HARD.value else (
                        2 if difficulty == Difficulty.MEDIUM.value else 3
                    )

                    per_lang_counts[lang] = per_lang_counts.get(lang, 0) + 1
                    per_domain_counts[domain] = per_domain_counts.get(domain, 0) + 1

                    yield ContrastivePair(
                        true_statement=true_stmt,
                        false_statement=false_stmt,
                        pair_id=pair_id,
                        domain="language",
                        relation_type="translates_to",
                        difficulty=difficulty,
                        semantic_distance=sem_dist,
                        generator=self.name,
                        template_id=template.id,
                        negation_strategy=strategy,
                        source_synset=english,
                        target_synset=lang,
                    )

    def _pick_translates_to_swaps(
        self, english: str, lang: str, true_translation: str,
    ) -> list[tuple[str, str, str]]:
        """Pick swap translations at each difficulty tier.

        Returns list of (swap_translation, difficulty, negation_strategy).
        """
        swaps = []

        # Hard: same domain, same language
        hard_swap = self._pick_same_domain_swap(english, lang)
        if hard_swap and hard_swap.lower() != true_translation.lower():
            swaps.append((hard_swap, Difficulty.HARD.value, NegationStrategy.SIBLING_SWAP.value))

        # Medium: different domain, same language
        medium_swap = self._pick_different_domain_swap(english, lang)
        if medium_swap and medium_swap.lower() != true_translation.lower():
            swaps.append((medium_swap, Difficulty.MEDIUM.value, NegationStrategy.SIBLING_SWAP.value))

        # Easy: wrong language word
        easy_swap = self._pick_wrong_language_swap(english, lang)
        if easy_swap and easy_swap.lower() != true_translation.lower():
            swaps.append((easy_swap, Difficulty.EASY.value, NegationStrategy.DISTANT_SWAP.value))

        return swaps

    # --- translation_of ---

    def _generate_translation_of(self) -> Iterator[ContrastivePair]:
        """Generate '{translation} is the {language} word for {english}' pairs."""
        entities = list(self._all_english)
        self.rng.shuffle(entities)

        for english in entities:
            domain = self._entity_domains[english]

            langs = list(self.target_languages)
            self.rng.shuffle(langs)

            for lang in langs:
                true_translation = self._get_translation(english, lang)
                if not true_translation:
                    continue

                # Swap the meaning: "perro is the Spanish word for cat"
                # Pick a different english word whose translation exists in this language
                same_domain = [
                    e for e in self._domain_entities.get(domain, [])
                    if e != english and (e, lang) in self._translation_map
                ]
                if not same_domain:
                    continue

                swap_english = self.rng.choice(same_domain)

                template = self._pick_template("translation_of")
                lang_name = self._lang_display(lang)

                true_stmt = render_template(template.pattern,
                    translation=true_translation, language=lang_name, english=english,
                )
                false_stmt = render_template(template.pattern,
                    translation=true_translation, language=lang_name, english=swap_english,
                )

                pair_id = _make_pair_id([
                    "lang", "reverse", english, lang,
                    true_translation, swap_english, template.id,
                ])

                yield ContrastivePair(
                    true_statement=true_stmt,
                    false_statement=false_stmt,
                    pair_id=pair_id,
                    domain="language",
                    relation_type="translation_of",
                    difficulty=Difficulty.HARD.value,
                    semantic_distance=1,
                    generator=self.name,
                    template_id=template.id,
                    negation_strategy=NegationStrategy.REVERSE_RELATION.value,
                    source_synset=english,
                    target_synset=lang,
                )

    # --- word_is_language ---

    def _generate_word_is_language(self) -> Iterator[ContrastivePair]:
        """Generate '{word} is a {language} word' pairs."""
        entities = list(self._all_english)
        self.rng.shuffle(entities)

        for english in entities:
            langs = list(self.target_languages)
            self.rng.shuffle(langs)

            for true_lang in langs:
                true_translation = self._get_translation(english, true_lang)
                if not true_translation:
                    continue

                # Skip cognates: only generate if the word is unique to this language
                if not self._is_unique_to_language(true_translation, true_lang):
                    continue

                # Pick a false language based on difficulty tiers
                swaps = self._pick_language_id_swaps(true_lang)
                for false_lang, difficulty in swaps:
                    template = self._pick_template("word_is_language")
                    true_lang_name = self._lang_display(true_lang)
                    false_lang_name = self._lang_display(false_lang)

                    true_stmt = render_template(template.pattern,
                        word=true_translation, language=true_lang_name,
                    )
                    false_stmt = render_template(template.pattern,
                        word=true_translation, language=false_lang_name,
                    )

                    pair_id = _make_pair_id([
                        "lang", "identify", english, true_lang,
                        true_translation, false_lang, template.id,
                    ])

                    dist = _language_distance(true_lang, false_lang)

                    yield ContrastivePair(
                        true_statement=true_stmt,
                        false_statement=false_stmt,
                        pair_id=pair_id,
                        domain="language",
                        relation_type="word_is_language",
                        difficulty=difficulty,
                        semantic_distance=dist,
                        generator=self.name,
                        template_id=template.id,
                        negation_strategy=NegationStrategy.DISTANT_SWAP.value,
                        source_synset=english,
                        target_synset=true_lang,
                        neg_synset=false_lang,
                    )

    def _pick_language_id_swaps(
        self, true_lang: str,
    ) -> list[tuple[str, str]]:
        """Pick false languages at each difficulty tier.

        Returns list of (false_lang, difficulty).
        """
        swaps = []
        other_langs = [l for l in self.target_languages if l != true_lang]

        hard_candidates = [l for l in other_langs if _language_distance(true_lang, l) == 1]
        medium_candidates = [l for l in other_langs if _language_distance(true_lang, l) == 2]
        easy_candidates = [l for l in other_langs if _language_distance(true_lang, l) == 3]

        if hard_candidates:
            swaps.append((self.rng.choice(hard_candidates), Difficulty.HARD.value))
        if medium_candidates:
            swaps.append((self.rng.choice(medium_candidates), Difficulty.MEDIUM.value))
        if easy_candidates:
            swaps.append((self.rng.choice(easy_candidates), Difficulty.EASY.value))

        return swaps
