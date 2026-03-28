"""Tests for the Wikidata data layer."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Disable wikistash for all tests so mocked HTTP tests work correctly
import latenet.datasources.wikidata as _wd
_wd._wikistash_checked = True
_wd._wikistash_stash = None

from latenet.datasources.wikidata import (
    _build_lineage,
    _extract_qid,
    _resolve_rank_label,
    filter_has_label,
    filter_has_wikipedia,
    sparql_query,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_sparql_response(bindings: list[dict]) -> dict:
    """Build a SPARQL JSON response from simplified bindings."""
    return {
        "results": {
            "bindings": [
                {k: {"value": v} for k, v in row.items()}
                for row in bindings
            ]
        }
    }


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestExtractQid:
    def test_full_uri(self):
        assert _extract_qid("http://www.wikidata.org/entity/Q7377") == "Q7377"

    def test_bare_qid(self):
        assert _extract_qid("Q12345") == "Q12345"

    def test_empty(self):
        assert _extract_qid("") == ""


class TestResolveRankLabel:
    def test_species(self):
        assert _resolve_rank_label("http://www.wikidata.org/entity/Q7432") == "species"

    def test_kingdom(self):
        assert _resolve_rank_label("http://www.wikidata.org/entity/Q36732") == "kingdom"

    def test_unknown(self):
        assert _resolve_rank_label("http://www.wikidata.org/entity/Q99999") == "unknown"


class TestFilterHasWikipedia:
    def test_filters_empty_article(self):
        df = pd.DataFrame({
            "name": ["Dog", "Mystery", "Cat"],
            "article": ["https://en.wikipedia.org/wiki/Dog", "", "https://en.wikipedia.org/wiki/Cat"],
        })
        result = filter_has_wikipedia(df)
        assert len(result) == 2
        assert "Mystery" not in result["name"].values

    def test_no_article_column(self):
        df = pd.DataFrame({"name": ["Dog"]})
        result = filter_has_wikipedia(df)
        assert len(result) == 1


class TestFilterHasLabel:
    def test_filters_missing_labels(self):
        df = pd.DataFrame({
            "name": ["Q123", "Cat"],
            "commonName": [None, "cat"],
        })
        result = filter_has_label(df)
        assert len(result) == 1
        assert result.iloc[0]["name"] == "Cat"


class TestSparqlQuery:
    @patch("latenet.datasources.wikidata.requests.get")
    def test_basic_query(self, mock_get):
        """Test that sparql_query parses JSON results correctly (remote fallback)."""
        response_data = _fake_sparql_response([
            {"item": "http://www.wikidata.org/entity/Q144", "itemLabel": "dog"},
            {"item": "http://www.wikidata.org/entity/Q146", "itemLabel": "cat"},
        ])

        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        query = "SELECT ?item ?itemLabel WHERE { ?item wdt:P31 wd:Q16521 } LIMIT 2"
        df = sparql_query(query)

        assert len(df) == 2
        assert "item" in df.columns
        assert df.iloc[0]["itemLabel"] == "dog"

    @patch("latenet.datasources.wikidata.requests.get")
    def test_retries_on_failure(self, mock_get):
        """Test that transient failures are retried."""
        import requests as real_requests

        mock_get.side_effect = [
            real_requests.ConnectionError("timeout"),
            real_requests.ConnectionError("timeout"),
            MagicMock(
                json=MagicMock(return_value=_fake_sparql_response([{"x": "1"}])),
                raise_for_status=MagicMock(),
            ),
        ]

        query = "SELECT ?x WHERE { ?x ?y ?z } LIMIT 1"
        df = sparql_query(query)

        assert len(df) == 1
        assert mock_get.call_count == 3


class TestBuildLineage:
    def test_resolves_parent_chain(self):
        """Test that _build_lineage walks the parent chain correctly."""
        df = pd.DataFrame([
            {
                "qid": "Q_dog", "name": "dog", "common_name": "dog",
                "taxon_rank": "species", "parent_taxon_qid": "Q_canis",
            },
            {
                "qid": "Q_canis", "name": "Canis", "common_name": "Canis",
                "taxon_rank": "genus", "parent_taxon_qid": "Q_canidae",
            },
            {
                "qid": "Q_canidae", "name": "Canidae", "common_name": "Canidae",
                "taxon_rank": "family", "parent_taxon_qid": "Q_carnivora",
            },
            {
                "qid": "Q_carnivora", "name": "Carnivora", "common_name": "Carnivora",
                "taxon_rank": "order", "parent_taxon_qid": "Q_mammalia",
            },
            {
                "qid": "Q_mammalia", "name": "Mammalia", "common_name": "mammal",
                "taxon_rank": "class", "parent_taxon_qid": "Q_chordata",
            },
            {
                "qid": "Q_chordata", "name": "Chordata", "common_name": "Chordata",
                "taxon_rank": "phylum", "parent_taxon_qid": "Q_animalia",
            },
            {
                "qid": "Q_animalia", "name": "Animalia", "common_name": "animal",
                "taxon_rank": "kingdom", "parent_taxon_qid": None,
            },
        ])

        result = _build_lineage(df)

        # Check that the dog row has its lineage filled in
        dog_row = result[result["qid"] == "Q_dog"].iloc[0]
        assert dog_row["genus"] == "Canis"
        assert dog_row["family"] == "Canidae"
        assert dog_row["order"] == "Carnivora"
        # class gets renamed to class_
        assert dog_row["class_"] == "mammal"
        assert dog_row["phylum"] == "Chordata"
        assert dog_row["kingdom"] == "animal"

    def test_handles_missing_ranks(self):
        """Test lineage with gaps."""
        df = pd.DataFrame([
            {
                "qid": "Q1", "name": "Org", "common_name": "org",
                "taxon_rank": "species", "parent_taxon_qid": "Q2",
            },
            {
                "qid": "Q2", "name": "Fam", "common_name": "fam",
                "taxon_rank": "family", "parent_taxon_qid": None,
            },
        ])

        result = _build_lineage(df)
        org_row = result[result["qid"] == "Q1"].iloc[0]
        assert org_row["family"] == "fam"
        assert pd.isna(org_row["genus"]) or org_row["genus"] is None

    def test_chains_through_bridge_ranks(self):
        """Test that intermediate ranks (subfamily, etc.) bridge the lineage chain.

        Simulates the real-world case: lion (species) → Panthera (genus) →
        Pantherinae (subfamily) → Felidae (family). The subfamily is not a
        canonical rank so it shouldn't get a lineage column, but the walker
        must chain through it to reach Felidae.
        """
        df = pd.DataFrame([
            {
                "qid": "Q140", "name": "lion", "common_name": "lion",
                "taxon_rank": "species", "parent_taxon_qid": "Q127960",
            },
            {
                "qid": "Q127960", "name": "Panthera", "common_name": "Panthera",
                "taxon_rank": "genus", "parent_taxon_qid": "Q230177",
            },
            {
                "qid": "Q230177", "name": "Pantherinae", "common_name": "Pantherinae",
                "taxon_rank": "subfamily", "parent_taxon_qid": "Q25265",
            },
            {
                "qid": "Q25265", "name": "Felidae", "common_name": "cat",
                "taxon_rank": "family", "parent_taxon_qid": "Q27070",
            },
            {
                "qid": "Q27070", "name": "Carnivora", "common_name": "Carnivora",
                "taxon_rank": "order", "parent_taxon_qid": None,
            },
        ])

        result = _build_lineage(df)
        lion = result[result["qid"] == "Q140"].iloc[0]
        assert lion["genus"] == "Panthera"
        assert lion["family"] == "cat"  # chains through subfamily bridge
        assert lion["order"] == "Carnivora"


# ---------------------------------------------------------------------------
# Common name resolution
# ---------------------------------------------------------------------------

class TestCommonNameResolution:
    """Tests for scientific-to-common-name resolution."""

    def test_common_names_json_loads(self):
        """The bundled common_names.json is valid and non-empty."""
        from latenet.datasources.wikidata import _COMMON_NAMES
        assert isinstance(_COMMON_NAMES, dict)
        assert len(_COMMON_NAMES) > 100

    def test_binomial_regex_matches_scientific(self):
        """The binomial regex matches Latin binomials but not common names."""
        from latenet.datasources.wikidata import _BINOMIAL_RE
        assert _BINOMIAL_RE.match("Canis lupus")
        assert _BINOMIAL_RE.match("Prunus persica")
        assert not _BINOMIAL_RE.match("red fox")
        assert not _BINOMIAL_RE.match("dog")
        assert not _BINOMIAL_RE.match("Q12345")

    def test_known_mappings(self):
        """Spot-check a few well-known scientific-to-common mappings."""
        from latenet.datasources.wikidata import _COMMON_NAMES
        assert _COMMON_NAMES.get("Canis lupus") == "wolf"
        assert _COMMON_NAMES.get("Ananas comosus") == "pineapple"
        assert _COMMON_NAMES.get("Prunus persica") == "peach"
