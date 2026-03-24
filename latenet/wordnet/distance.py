"""Semantic distance calculation between synsets."""

from nltk.corpus.reader.wordnet import Synset

MAX_DISTANCE = 20


def hop_distance(s1: Synset, s2: Synset) -> int | None:
    """Raw hop count via shortest_path_distance. None if disconnected."""
    return s1.shortest_path_distance(s2)


def semantic_distance(s1: Synset, s2: Synset) -> int:
    """Hop distance between two synsets, defaulting to MAX_DISTANCE if disconnected."""
    d = hop_distance(s1, s2)
    return d if d is not None else MAX_DISTANCE
