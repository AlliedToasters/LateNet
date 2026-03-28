"""MUSE bilingual dictionary loader with caching, filtering, and WordNet domain tagging.

Data source: Meta's MUSE (Multilingual Unsupervised and Supervised Embeddings)
bilingual dictionaries — static TSV files, one per language pair.

URL pattern: https://dl.fbaipublicfiles.com/arrival/dictionaries/{src}-{tgt}.txt
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

CACHE_ROOT = Path.home() / ".cache" / "latenet" / "muse"

MUSE_URL_TEMPLATE = (
    "https://dl.fbaipublicfiles.com/arrival/dictionaries/{src}-{tgt}.txt"
)

DEFAULT_TARGET_LANGUAGES = ["es", "fr", "de", "pt", "it", "zh", "ja", "ru", "ar", "hi"]

LANGUAGE_DISPLAY_NAMES: dict[str, str] = {
    "es": "Spanish", "fr": "French", "de": "German", "pt": "Portuguese",
    "it": "Italian", "zh": "Chinese", "ja": "Japanese", "ru": "Russian",
    "ar": "Arabic", "hi": "Hindi", "en": "English",
}

# Expected Unicode script ranges for non-Latin languages
_SCRIPT_RANGES: dict[str, list[tuple[int, int]]] = {
    "zh": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x2E80, 0x2EFF), (0x3000, 0x303F)],
    "ja": [(0x3040, 0x309F), (0x30A0, 0x30FF), (0x4E00, 0x9FFF), (0x3400, 0x4DBF)],
    "ru": [(0x0400, 0x04FF), (0x0500, 0x052F)],
    "ar": [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "hi": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
}


def _in_expected_script(text: str, lang: str) -> bool:
    """Check that at least 50% of non-space characters are in the expected script."""
    ranges = _SCRIPT_RANGES.get(lang)
    if ranges is None:
        return True  # Latin-script languages: no check needed
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    in_script = sum(
        1 for c in chars
        if any(lo <= ord(c) <= hi for lo, hi in ranges)
    )
    return in_script / len(chars) >= 0.5


def _ensure_cache_dir() -> Path:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return CACHE_ROOT


def _download_dictionary(source_lang: str, target_lang: str) -> Path:
    """Download a MUSE dictionary file and return the local cache path."""
    _ensure_cache_dir()
    filename = f"{source_lang}-{target_lang}.txt"
    cache_path = CACHE_ROOT / filename
    url = MUSE_URL_TEMPLATE.format(src=source_lang, tgt=target_lang)

    logger.info("Downloading MUSE dictionary: %s -> %s", url, cache_path)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    cache_path.write_bytes(resp.content)
    logger.info("Downloaded %d bytes to %s", len(resp.content), cache_path)
    return cache_path


def load_dictionary(
    source_lang: str = "en",
    target_lang: str = "es",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load a MUSE bilingual dictionary.

    Downloads and caches the dictionary file, then applies filtering:
    - Single-word translations only (no spaces)
    - No homographs (source == target)
    - Script validation for non-Latin languages
    - Lowercase normalization on source words
    - Deduplication (keep first occurrence per source word)

    Returns DataFrame with columns:
        source_word, target_word, source_lang, target_lang
    """
    _ensure_cache_dir()
    filename = f"{source_lang}-{target_lang}.txt"
    cache_path = CACHE_ROOT / filename

    if force_refresh or not cache_path.exists():
        _download_dictionary(source_lang, target_lang)
    else:
        logger.info("Cache hit: %s", cache_path)

    # Parse dictionary file: MUSE files use either tab or space as delimiter.
    # Try tab first; if that produces only 1 column, fall back to space.
    df = pd.read_csv(
        cache_path, sep="\t", header=None, names=["source_word", "target_word"],
        dtype=str, na_filter=False,
    )
    if "target_word" not in df.columns or df["target_word"].isna().all() or (df["target_word"] == "").all():
        df = pd.read_csv(
            cache_path, sep=" ", header=None, names=["source_word", "target_word"],
            dtype=str, na_filter=False, usecols=[0, 1],
        )
    initial_count = len(df)

    # Filter: single-word only (no spaces in either column)
    df = df[
        ~df["source_word"].str.contains(" ", regex=False)
        & ~df["target_word"].str.contains(" ", regex=False)
    ]
    after_single_word = len(df)
    dropped_multiword = initial_count - after_single_word
    if dropped_multiword:
        logger.info("Dropped %d multi-word entries for %s-%s", dropped_multiword, source_lang, target_lang)

    # Lowercase source words
    df["source_word"] = df["source_word"].str.lower()

    # Filter: no homographs (source == target, case-insensitive)
    homograph_mask = df["source_word"].str.lower() == df["target_word"].str.lower()
    dropped_homographs = homograph_mask.sum()
    df = df[~homograph_mask]
    if dropped_homographs:
        logger.info("Dropped %d homographs for %s-%s", dropped_homographs, source_lang, target_lang)

    # Script validation for non-Latin target languages
    if target_lang in _SCRIPT_RANGES:
        valid_script = df["target_word"].apply(lambda w: _in_expected_script(w, target_lang))
        dropped_script = (~valid_script).sum()
        df = df[valid_script]
        if dropped_script:
            logger.info("Dropped %d entries failing script validation for %s-%s", dropped_script, source_lang, target_lang)

    # Deduplicate: keep first occurrence per source word
    before_dedup = len(df)
    df = df.drop_duplicates(subset="source_word", keep="first")
    dropped_dupes = before_dedup - len(df)
    if dropped_dupes:
        logger.info("Dropped %d duplicate source words for %s-%s", dropped_dupes, source_lang, target_lang)

    # Add language metadata columns
    df["source_lang"] = source_lang
    df["target_lang"] = target_lang

    df = df.reset_index(drop=True)
    logger.info(
        "Loaded %d clean pairs for %s-%s (from %d raw)",
        len(df), source_lang, target_lang, initial_count,
    )
    return df


def tag_domains(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'domain' column by looking up source_word in WordNet.

    Uses the first synset's lexicographer file name to determine a category
    (e.g., noun.animal -> animal, noun.food -> food, noun.artifact -> object).

    Words not found in WordNet or with ambiguous categories get domain = "other".
    """
    try:
        from nltk.corpus import wordnet as wn
    except ImportError:
        logger.warning("NLTK/WordNet not available; tagging all words as 'other'")
        df["domain"] = "other"
        return df

    # Map lexicographer file names to simpler domain labels
    _LEXNAME_TO_DOMAIN: dict[str, str] = {
        "noun.animal": "animal",
        "noun.food": "food",
        "noun.body": "body_part",
        "noun.artifact": "object",
        "noun.location": "place",
        "noun.attribute": "attribute",
        "noun.plant": "plant",
        "noun.substance": "substance",
        "noun.person": "person",
        "noun.act": "action",
        "noun.event": "event",
        "noun.state": "state",
        "noun.feeling": "feeling",
        "noun.quantity": "quantity",
        "noun.time": "time",
        "noun.shape": "shape",
        "noun.communication": "communication",
        "noun.cognition": "cognition",
        "noun.possession": "possession",
        "noun.object": "object",
        "noun.relation": "relation",
        "noun.group": "group",
        "noun.phenomenon": "phenomenon",
        "noun.process": "process",
        "noun.Tops": "other",
    }

    domains = []
    for word in df["source_word"]:
        synsets = wn.synsets(word, pos=wn.NOUN)
        if synsets:
            lexname = synsets[0].lexname()
            domain = _LEXNAME_TO_DOMAIN.get(lexname, "other")
        else:
            domain = "other"
        domains.append(domain)

    df = df.copy()
    df["domain"] = domains
    return df
