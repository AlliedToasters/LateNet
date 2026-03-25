"""Tests for the Wikidata data layer."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from latenet.datasources.wikidata import (
    CACHE_ROOT,
    _build_lineage,
    _cache_key,
    _cache_path,
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


class TestCacheKey:
    def test_deterministic(self):
        q = "SELECT * WHERE { ?s ?p ?o }"
        assert _cache_key(q) == _cache_key(q)

    def test_different_queries_different_keys(self):
        q1 = "SELECT * WHERE { ?s ?p ?o }"
        q2 = "SELECT * WHERE { ?x ?y ?z }"
        assert _cache_key(q1) != _cache_key(q2)


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
    def test_basic_query(self, mock_get, tmp_path):
        """Test that sparql_query parses JSON results correctly."""
        response_data = _fake_sparql_response([
            {"item": "http://www.wikidata.org/entity/Q144", "itemLabel": "dog"},
            {"item": "http://www.wikidata.org/entity/Q146", "itemLabel": "cat"},
        ])

        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        query = "SELECT ?item ?itemLabel WHERE { ?item wdt:P31 wd:Q16521 } LIMIT 2"

        with patch("latenet.datasources.wikidata.CACHE_ROOT", tmp_path):
            with patch("latenet.datasources.wikidata._cache_path") as mock_cp:
                cache_file = tmp_path / "test.parquet"
                mock_cp.return_value = cache_file

                df = sparql_query(query, force_refresh=True)

        assert len(df) == 2
        assert "item" in df.columns
        assert df.iloc[0]["itemLabel"] == "dog"

    @patch("latenet.datasources.wikidata.requests.get")
    def test_cache_hit(self, mock_get, tmp_path):
        """Test that cached results are returned without making a request."""
        cache_file = tmp_path / "cached.parquet"
        cached_df = pd.DataFrame({"item": ["Q1"], "label": ["Universe"]})
        cached_df.to_parquet(cache_file, index=False)

        query = "SELECT * WHERE { wd:Q1 ?p ?o }"

        with patch("latenet.datasources.wikidata._cache_path", return_value=cache_file):
            df = sparql_query(query, force_refresh=False)

        mock_get.assert_not_called()
        assert len(df) == 1
        assert df.iloc[0]["label"] == "Universe"

    @patch("latenet.datasources.wikidata.requests.get")
    def test_retries_on_failure(self, mock_get, tmp_path):
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

        with patch("latenet.datasources.wikidata.CACHE_ROOT", tmp_path):
            with patch("latenet.datasources.wikidata._cache_path") as mock_cp:
                mock_cp.return_value = tmp_path / "retry_test.parquet"
                with patch("latenet.datasources.wikidata._ensure_cache_dir", return_value=tmp_path):
                    df = sparql_query(query, force_refresh=True)

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
