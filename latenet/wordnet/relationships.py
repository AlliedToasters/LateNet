"""Extract hypernym, meronym, antonym, and sibling relationships from WordNet."""

from __future__ import annotations

from nltk.corpus.reader.wordnet import Synset

from latenet.types import Relationship, RelationshipType


def _lemma_name(synset: Synset) -> str:
    """Human-readable name from a synset's first lemma."""
    return synset.lemma_names()[0].replace("_", " ")


def extract_hypernyms(synset: Synset) -> list[Relationship]:
    """Return direct hypernym (is-a) relationships."""
    name = _lemma_name(synset)
    return [
        Relationship(
            source=synset,
            target=hyp,
            rel_type=RelationshipType.HYPERNYMY,
            source_name=name,
            target_name=_lemma_name(hyp),
        )
        for hyp in synset.hypernyms()
    ]


def extract_meronyms(synset: Synset) -> list[Relationship]:
    """Return part/substance/member meronym relationships."""
    name = _lemma_name(synset)
    rels = []
    for mer in synset.part_meronyms() + synset.substance_meronyms() + synset.member_meronyms():
        rels.append(
            Relationship(
                source=synset,
                target=mer,
                rel_type=RelationshipType.MERONYMY,
                source_name=name,
                target_name=_lemma_name(mer),
            )
        )
    return rels


def extract_antonyms(synset: Synset) -> list[Relationship]:
    """Return antonym relationships (via lemma-level antonyms)."""
    name = _lemma_name(synset)
    rels = []
    seen = set()
    for lemma in synset.lemmas():
        for ant_lemma in lemma.antonyms():
            ant_synset = ant_lemma.synset()
            if ant_synset.name() not in seen:
                seen.add(ant_synset.name())
                rels.append(
                    Relationship(
                        source=synset,
                        target=ant_synset,
                        rel_type=RelationshipType.ANTONYMY,
                        source_name=name,
                        target_name=_lemma_name(ant_synset),
                    )
                )
    return rels


def extract_siblings(synset: Synset) -> list[Relationship]:
    """Return sibling relationships (synsets sharing a hypernym parent)."""
    name = _lemma_name(synset)
    rels = []
    seen = set()
    for parent in synset.hypernyms():
        for sibling in parent.hyponyms():
            if sibling.name() != synset.name() and sibling.name() not in seen:
                seen.add(sibling.name())
                rels.append(
                    Relationship(
                        source=synset,
                        target=sibling,
                        rel_type=RelationshipType.SIBLING,
                        source_name=name,
                        target_name=_lemma_name(sibling),
                    )
                )
    return rels


_EXTRACTORS = {
    RelationshipType.HYPERNYMY: extract_hypernyms,
    RelationshipType.MERONYMY: extract_meronyms,
    RelationshipType.ANTONYMY: extract_antonyms,
    RelationshipType.SIBLING: extract_siblings,
}


def extract_all(
    synset: Synset, rel_types: set[RelationshipType] | None = None
) -> list[Relationship]:
    """Extract all requested relationship types for a synset."""
    types = rel_types or set(RelationshipType)
    rels = []
    for rt in types:
        rels.extend(_EXTRACTORS[rt](synset))
    return rels
