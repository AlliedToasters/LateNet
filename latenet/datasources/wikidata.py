"""Wikidata SPARQL client with pagination, caching, and common knowledge filtering.

Provides reusable data loading for any generator backed by Wikidata.
Caches query results as parquet in ~/.cache/latenet/wikidata/.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

CACHE_ROOT = Path.home() / ".cache" / "latenet" / "wikidata"
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = (
    "LateNet/0.1 (https://github.com/AlliedToasters/latenet; research dataset generation)"
)
DEFAULT_PAGE_SIZE = 5000
DEFAULT_TIMEOUT = 120
MAX_RETRIES = 3
RATE_LIMIT_SECONDS = 1.0

# ---------------------------------------------------------------------------
# wikistash local backend (optional)
# ---------------------------------------------------------------------------

_wikistash_stash = None
_wikistash_checked = False


def _get_wikistash():
    """Lazily initialize wikistash connection. Returns None if unavailable."""
    global _wikistash_stash, _wikistash_checked
    if _wikistash_checked:
        return _wikistash_stash
    _wikistash_checked = True

    # Check for WIKISTASH_DB env var or well-known paths
    import os
    db_path = os.environ.get("WIKISTASH_DB")
    if not db_path:
        # Check common relative paths
        for candidate in [
            Path("../../wikistash/wikistash_partial.duckdb"),
            Path("../wikistash/wikistash_partial.duckdb"),
            Path.home() / "wikistash" / "wikistash_partial.duckdb",
        ]:
            if candidate.exists():
                db_path = str(candidate)
                break

    if not db_path:
        logger.debug("wikistash not found — using remote Wikidata SPARQL endpoint")
        return None

    try:
        from wikistash import Stash
        _wikistash_stash = Stash(
            local_db_path=db_path,
            enable_live_fallback=False,
        )
        logger.info("wikistash connected: %s", db_path)
        return _wikistash_stash
    except Exception as e:
        logger.warning("wikistash available but failed to connect: %s", e)
        return None


def _run_wikistash_query(query: str) -> pd.DataFrame | None:
    """Try to run a SPARQL query via wikistash. Returns None on failure."""
    stash = _get_wikistash()
    if stash is None:
        return None

    try:
        data = stash.sparql_json(query)
        bindings = data.get("results", {}).get("bindings", [])
        if not bindings:
            return pd.DataFrame()
        rows = []
        for b in bindings:
            row = {}
            for key, val in b.items():
                row[key] = val.get("value", "")
            rows.append(row)
        return pd.DataFrame(rows)
    except Exception as e:
        logger.warning("wikistash query failed, falling back to remote: %s", e)
        return None


# ---------------------------------------------------------------------------
# SPARQL client
# ---------------------------------------------------------------------------


def _ensure_cache_dir() -> Path:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return CACHE_ROOT


def _cache_key(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()


def _cache_path(query: str) -> Path:
    return _ensure_cache_dir() / f"{_cache_key(query)}.parquet"


def _manifest_path() -> Path:
    return _ensure_cache_dir() / "manifest.json"


def _load_manifest() -> dict:
    mp = _manifest_path()
    if mp.exists():
        return json.loads(mp.read_text())
    return {}


def _save_manifest(manifest: dict) -> None:
    _manifest_path().write_text(json.dumps(manifest, indent=2))


def _update_manifest(query: str, row_count: int) -> None:
    manifest = _load_manifest()
    manifest[_cache_key(query)] = {
        "query_hash": _cache_key(query),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "row_count": row_count,
        "query_preview": query[:200],
    }
    _save_manifest(manifest)


_last_request_time: float = 0.0


def _rate_limit() -> None:
    global _last_request_time
    now = time.monotonic()
    elapsed = now - _last_request_time
    if elapsed < RATE_LIMIT_SECONDS:
        time.sleep(RATE_LIMIT_SECONDS - elapsed)
    _last_request_time = time.monotonic()


def sparql_query(
    query: str,
    timeout: int = DEFAULT_TIMEOUT,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Execute a SPARQL query against Wikidata and return results as a DataFrame.

    Tries wikistash (local) first, falls back to remote SPARQL endpoint.
    Results are cached as parquet files keyed by query hash.
    """
    cp = _cache_path(query)
    if not force_refresh and cp.exists():
        logger.info("Cache hit for query %s", _cache_key(query)[:12])
        return pd.read_parquet(cp)

    # Try wikistash first (local, fast, no rate limits — skip cache)
    local_result = _run_wikistash_query(query)
    if local_result is not None:
        logger.info("wikistash returned %d rows for query %s", len(local_result), _cache_key(query)[:12])
        return local_result

    logger.debug("SPARQL query:\n%s", query)
    logger.info("Cache miss for query %s, fetching from Wikidata...", _cache_key(query)[:12])

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/sparql-results+json",
    }

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        _rate_limit()
        try:
            resp = requests.get(
                SPARQL_ENDPOINT,
                params={"query": query},
                headers=headers,
                timeout=timeout,
            )
            resp.raise_for_status()
            import json as _json
            try:
                data = resp.json()
            except ValueError:
                data = _json.loads(resp.text, strict=False)
            break
        except (requests.RequestException, ValueError) as e:
            last_error = e
            wait = 2 ** attempt
            logger.warning(
                "SPARQL request attempt %d/%d failed: %s. Retrying in %ds...",
                attempt, MAX_RETRIES, e, wait,
            )
            time.sleep(wait)
    else:
        raise RuntimeError(
            f"SPARQL query failed after {MAX_RETRIES} attempts: {last_error}"
        ) from last_error

    # Parse SPARQL JSON results into DataFrame
    bindings = data.get("results", {}).get("bindings", [])
    if not bindings:
        df = pd.DataFrame()
    else:
        rows = []
        for b in bindings:
            row = {}
            for key, val in b.items():
                row[key] = val.get("value", "")
            rows.append(row)
        df = pd.DataFrame(rows)

    # Cache result
    _ensure_cache_dir()
    df.to_parquet(cp, index=False)
    _update_manifest(query, len(df))
    logger.info("Cached %d rows for query %s", len(df), _cache_key(query)[:12])

    return df


def sparql_query_paginated(
    query_template: str,
    page_size: int = DEFAULT_PAGE_SIZE,
    timeout: int = DEFAULT_TIMEOUT,
    force_refresh: bool = False,
    max_pages: int | None = None,
) -> pd.DataFrame:
    """Execute a paginated SPARQL query using {limit} and {offset} placeholders.

    Concatenates all pages into a single DataFrame and caches the final result.
    When wikistash is available, runs the full query locally (no pagination needed).
    """
    # Cache the full result by hashing the template
    cp = _cache_path(query_template)
    if not force_refresh and cp.exists():
        logger.info("Cache hit for paginated query %s", _cache_key(query_template)[:12])
        return pd.read_parquet(cp)

    # Try wikistash — run full query without pagination (local is fast enough, skip cache)
    # Fill in a large LIMIT and OFFSET=0 to satisfy the template placeholders
    local_query = query_template.format(limit=1_000_000, offset=0)
    local_result = _run_wikistash_query(local_query)
    if local_result is not None:
        logger.info("wikistash returned %d rows for paginated query %s",
                     len(local_result), _cache_key(query_template)[:12])
        return local_result

    logger.info("Starting paginated query %s...", _cache_key(query_template)[:12])

    all_dfs = []
    offset = 0
    page_num = 0

    while True:
        page_query = query_template.format(limit=page_size, offset=offset)

        # Don't cache individual pages — only the full result
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "application/sparql-results+json",
        }

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            _rate_limit()
            try:
                resp = requests.get(
                    SPARQL_ENDPOINT,
                    params={"query": page_query},
                    headers=headers,
                    timeout=timeout,
                )
                resp.raise_for_status()
                import json as _json
                data = _json.loads(resp.text, strict=False)
                break
            except (requests.RequestException, ValueError) as e:
                last_error = e
                wait = 2 ** attempt
                logger.warning(
                    "Page %d attempt %d/%d failed: %s. Retrying in %ds...",
                    page_num, attempt, MAX_RETRIES, e, wait,
                )
                time.sleep(wait)
        else:
            # If we already have some data, log warning and stop pagination
            # rather than failing the entire query
            if all_dfs:
                logger.warning(
                    "Page %d failed after %d attempts; stopping pagination with %d rows collected",
                    page_num, MAX_RETRIES, sum(len(d) for d in all_dfs),
                )
                break
            raise RuntimeError(
                f"Paginated query page {page_num} failed after {MAX_RETRIES} attempts: "
                f"{last_error}"
            ) from last_error

        bindings = data.get("results", {}).get("bindings", [])
        if not bindings:
            break

        rows = []
        for b in bindings:
            row = {}
            for key, val in b.items():
                row[key] = val.get("value", "")
            rows.append(row)

        page_df = pd.DataFrame(rows)
        all_dfs.append(page_df)
        total_rows = sum(len(d) for d in all_dfs)
        page_num += 1
        logger.info("Fetched page %d (%d rows so far)...", page_num, total_rows)

        if len(rows) < page_size:
            break

        if max_pages is not None and page_num >= max_pages:
            logger.info("Reached max_pages=%d, stopping pagination", max_pages)
            break

        offset += page_size

    if all_dfs:
        df = pd.concat(all_dfs, ignore_index=True)
    else:
        df = pd.DataFrame()

    # Cache the full result
    _ensure_cache_dir()
    df.to_parquet(cp, index=False)
    _update_manifest(query_template, len(df))
    logger.info("Paginated query complete: %d total rows", len(df))

    return df


# ---------------------------------------------------------------------------
# Common knowledge filter
# ---------------------------------------------------------------------------


def filter_has_wikipedia(df: pd.DataFrame, article_col: str = "article") -> pd.DataFrame:
    """Filter to entities that have an English Wikipedia article.

    The article_col should contain the Wikipedia article URL (non-empty = has article).
    """
    before = len(df)
    if article_col not in df.columns:
        logger.warning("No '%s' column found — skipping Wikipedia filter", article_col)
        return df
    filtered = df[df[article_col].notna() & (df[article_col] != "")].copy()
    logger.info(
        "Wikipedia filter: %d -> %d entities (dropped %d without article)",
        before, len(filtered), before - len(filtered),
    )
    return filtered


def filter_has_label(df: pd.DataFrame, label_col: str = "commonName") -> pd.DataFrame:
    """Filter to entities that have a non-empty English label."""
    before = len(df)
    if label_col not in df.columns:
        logger.warning("No '%s' column found — skipping label filter", label_col)
        return df
    filtered = df[df[label_col].notna() & (df[label_col] != "")].copy()
    logger.info(
        "Label filter: %d -> %d entities (dropped %d without label)",
        before, len(filtered), before - len(filtered),
    )
    return filtered


# ---------------------------------------------------------------------------
# Domain-specific extracts
# ---------------------------------------------------------------------------

# Wikidata QIDs for taxonomic ranks
_RANK_QIDS = {
    "species": "Q7432",
    "genus": "Q34740",
    "family": "Q35409",
    "order": "Q36602",
    "class": "Q37517",
    "phylum": "Q38348",
    "kingdom": "Q36732",
}

# Intermediate ranks that sit between the canonical ranks above.
# These are loaded as "bridge" entities so the lineage walker can chain
# through them (e.g., Panthera → Pantherinae [subfamily] → Felidae).
# They don't get their own lineage columns — they just need to exist
# in qid_to_parent so the walk doesn't dead-end.
# QIDs verified against Wikidata 2026-03 dump.
_BRIDGE_RANK_QIDS = {
    # Between species and genus
    "subgenus": "Q3238261",       # 13,797 entities
    # Between genus and family
    "section": "Q10861426",       #  7,417
    "series": "Q3025161",         #  1,804
    "subsection": "Q5998839",     #  1,563
    "tribe": "Q227936",           #  8,425
    "subtribe": "Q3965313",       #  2,629
    "supertribe": "Q14817220",    #     63
    "subfamily": "Q164280",       #  9,502
    # Between family and order
    "superfamily": "Q2136103",    #  1,953
    # Between order and class
    "suborder": "Q5867959",       #  1,136
    "infraorder": "Q2889003",     #    331
    "parvorder": "Q6311258",      #     57
    "mirorder": "Q7506274",       #      8
    "grandorder": "Q6462265",     #      7
    "magnorder": "Q6054237",      #      7
    "superorder": "Q5868144",     #    304
    "cohort": "Q2981883",         #     16
    # Between class and phylum
    "subclass": "Q5867051",       #    412
    "infraclass": "Q2007442",     #     50
    "superclass": "Q3504061",     #     38
    # Between phylum and kingdom
    "subphylum": "Q1153785",      #     95
    "infraphylum": "Q2361851",    #     21
    "superphylum": "Q2111790",    #     27
    "division": "Q334460",        #     77  (botanical phylum equivalent)
    # Between kingdom and root
    "subkingdom": "Q2752679",     #     40
    "infrakingdom": "Q3150876",   #     17
}

# Ordered from most specific to most general
RANK_ORDER = ["species", "genus", "family", "order", "class", "phylum", "kingdom"]
RANK_LEVEL = {r: i for i, r in enumerate(RANK_ORDER)}


def _build_organism_query_for_rank(rank_qid: str, min_sitelinks: int = 0) -> str:
    """Build a paginated SPARQL query for organisms of a single taxonomic rank.

    Uses rdfs:label directly instead of SERVICE wikibase:label, which is
    dramatically faster for large result sets like species (~1.6M taxa).
    Always returns sitelink counts for popularity-weighted sampling.
    """
    sitelink_clause = ""
    if min_sitelinks > 0:
        sitelink_clause = f"\n  FILTER(?sitelinks >= {min_sitelinks})"

    return f"""
SELECT ?item ?itemLabel ?parentTaxon ?parentTaxonLabel ?sitelinks
WHERE {{{{
  ?item wdt:P31 wd:Q16521 ;
        wdt:P105 wd:{rank_qid} ;
        wdt:P171 ?parentTaxon ;
        wikibase:sitelinks ?sitelinks ;
        rdfs:label ?itemLabel .
  ?parentTaxon rdfs:label ?parentTaxonLabel .
  FILTER(LANG(?itemLabel) = "en")
  FILTER(LANG(?parentTaxonLabel) = "en"){sitelink_clause}
}}}}
LIMIT {{limit}} OFFSET {{offset}}
"""


def _extract_qid(uri: str) -> str:
    """Extract QID from a Wikidata entity URI."""
    if not isinstance(uri, str):
        return ""
    if "/" in uri:
        return uri.rsplit("/", 1)[-1]
    return uri


def _resolve_rank_label(rank_uri: str) -> str:
    """Map a Wikidata rank URI to our rank label."""
    qid = _extract_qid(rank_uri)
    for label, rqid in _RANK_QIDS.items():
        if qid == rqid:
            return label
    return "unknown"


def _build_lineage(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve full taxonomic lineage by walking parent_taxon links in Python.

    Adds columns: kingdom, phylum, class_, order, family, genus
    (each containing the common name or scientific name of that ancestor).
    """
    # Build QID -> row lookup
    qid_to_row = {}
    for _, row in df.iterrows():
        qid = row["qid"]
        if qid not in qid_to_row:
            qid_to_row[qid] = row

    # Build QID -> parent QID lookup
    qid_to_parent = {}
    for _, row in df.iterrows():
        qid = row["qid"]
        parent = row.get("parent_taxon_qid")
        if pd.notna(parent) and parent:
            qid_to_parent[qid] = parent

    # For each organism, walk up the tree to fill lineage
    lineage_cols = {r: [] for r in RANK_ORDER if r != "species"}
    lineage_qid_cols = {f"{r}_qid": [] for r in RANK_ORDER if r != "species"}

    for _, row in df.iterrows():
        ancestors = {r: None for r in RANK_ORDER if r != "species"}
        ancestor_qids = {f"{r}_qid": None for r in RANK_ORDER if r != "species"}

        current_qid = row.get("parent_taxon_qid")
        visited = set()
        max_hops = 50  # Safety limit

        while current_qid and current_qid not in visited and max_hops > 0:
            visited.add(current_qid)
            max_hops -= 1

            parent_row = qid_to_row.get(current_qid)
            if parent_row is not None:
                rank = parent_row.get("taxon_rank")
                if rank and rank in ancestors:
                    name = parent_row.get("common_name") or parent_row.get("name")
                    ancestors[rank] = name
                    ancestor_qids[f"{rank}_qid"] = current_qid

            current_qid = qid_to_parent.get(current_qid)

        for r, val in ancestors.items():
            lineage_cols[r].append(val)
        for r, val in ancestor_qids.items():
            lineage_qid_cols[r].append(val)

    for col, vals in lineage_cols.items():
        df[col] = vals
    for col, vals in lineage_qid_cols.items():
        df[col] = vals

    # Rename 'class' column to avoid Python keyword conflict
    if "class" in df.columns:
        df = df.rename(columns={"class": "class_"})

    return df


def load_organisms(
    require_common_name: bool = True,
    require_wikipedia: bool = True,
    force_refresh: bool = False,
    max_pages_per_rank: int = 10,
    min_sitelinks: int = 10,
) -> pd.DataFrame:
    """Load organisms with taxonomy from Wikidata.

    Queries each taxonomic rank separately to avoid Wikidata query planner
    timeouts, then merges and resolves lineage in Python.

    Returns DataFrame with columns:
        qid, name, common_name, taxon_rank,
        parent_taxon_qid, kingdom, phylum, class_, order, family, genus,
        ncbi_taxon_id, has_wikipedia
    """
    all_dfs: list[pd.DataFrame] = []

    # Query canonical ranks (species, genus, family, ...) AND bridge ranks
    # (subfamily, superfamily, tribe, ...). Bridge ranks don't get lineage
    # columns but their presence in the DataFrame lets the lineage walker
    # chain through intermediate taxonomy nodes.
    all_ranks = {**_RANK_QIDS, **_BRIDGE_RANK_QIDS}

    for rank_label, rank_qid in all_ranks.items():
        # Bridge ranks don't need sitelink filtering — they're structural
        sitelinks = min_sitelinks if rank_label in _RANK_QIDS else 0
        query_template = _build_organism_query_for_rank(rank_qid, min_sitelinks=sitelinks)
        try:
            raw = sparql_query_paginated(
                query_template, page_size=2000, force_refresh=force_refresh,
                max_pages=max_pages_per_rank,
            )
        except RuntimeError:
            logger.warning("Skipping rank %s (%s) due to query failure", rank_label, rank_qid)
            continue

        if raw.empty:
            continue

        chunk = pd.DataFrame()
        chunk["qid"] = raw.get("item", pd.Series(dtype=str)).apply(_extract_qid)
        chunk["name"] = raw.get("itemLabel", pd.Series(dtype=str))
        chunk["common_name"] = raw.get("itemLabel", pd.Series(dtype=str))  # use label as common name
        chunk["taxon_rank"] = rank_label
        chunk["parent_taxon_qid"] = raw.get("parentTaxon", pd.Series(dtype=str)).apply(_extract_qid)
        chunk["parent_taxon_label"] = raw.get("parentTaxonLabel", pd.Series(dtype=str))
        chunk["sitelinks"] = pd.to_numeric(raw.get("sitelinks", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(int)
        chunk["ncbi_taxon_id"] = ""
        chunk["article"] = ""
        chunk["has_wikipedia"] = False
        all_dfs.append(chunk)
        logger.info("  %s: %d raw rows", rank_label, len(chunk))

    if not all_dfs:
        logger.warning("No organisms returned from Wikidata queries")
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)

    # Deduplicate: keep one row per QID
    df = df.drop_duplicates(subset=["qid"], keep="first").reset_index(drop=True)

    # Filter out entities whose name looks like a QID (no label resolved)
    df = df[~df["name"].str.match(r"^Q\d+$", na=False)].copy()

    logger.info("Raw organisms: %d unique entities", len(df))

    # Build lineage
    df = _build_lineage(df)

    logger.info("Final organism dataset: %d entities", len(df))
    return df


# Event types available for temporal queries: QID -> label
_EVENT_TYPES: dict[str, str] = {
    "Q198": "war",
    "Q178561": "battle",
    "Q131569": "treaty",
    "Q10931": "revolution",
    "Q39546": "invention",
    "Q3024240": "historical event",
    "Q2001676": "skirmish",
    "Q12017920": "military operation",
}

# Default event types to exclude — individual battles and skirmishes produce
# statements too obscure for LLM validators (79% dispute rate).
DEFAULT_EXCLUDE_EVENT_TYPES: frozenset[str] = frozenset({
    "Q178561",   # battle
    "Q2001676",  # skirmish
    "Q12017920", # military operation
})


def _build_event_query(
    exclude_event_types: frozenset[str] = DEFAULT_EXCLUDE_EVENT_TYPES,
    min_sitelinks: int = 0,
) -> str:
    """Build the paginated SPARQL query for historical events with dates.

    Uses direct P31 (instance-of) only — no transitive subclass traversal
    to keep query fast.  Date precision is filtered in Python.

    Parameters
    ----------
    exclude_event_types
        Wikidata QIDs to exclude from the event type VALUES block.
        Defaults to battles and skirmishes.
    min_sitelinks
        Minimum sitelink count for an entity to be included. 0 disables.
    """
    included = {qid for qid in _EVENT_TYPES if qid not in exclude_event_types}
    values_block = "\n    ".join(f"wd:{qid}  # {_EVENT_TYPES[qid]}" for qid in sorted(included))
    sitelink_clause = ""
    if min_sitelinks > 0:
        sitelink_clause = f"""
  ?item wikibase:sitelinks ?sitelinks .
  FILTER(?sitelinks >= {min_sitelinks})"""

    return f"""
SELECT DISTINCT ?item ?itemLabel ?itemDescription ?date ?eventType ?eventTypeLabel ?article
WHERE {{{{
  VALUES ?eventType {{{{
    {values_block}
  }}}}
  ?item wdt:P31 ?eventType .
  ?item wdt:P585 ?date .{sitelink_clause}

  OPTIONAL {{{{
    ?article schema:about ?item .
    ?article schema:isPartOf <https://en.wikipedia.org/> .
  }}}}

  SERVICE wikibase:label {{{{ bd:serviceParam wikibase:language "en" . }}}}
}}}}
LIMIT {{limit}} OFFSET {{offset}}
"""


# Notable occupations to query for people — each queried separately
# to avoid Wikidata query planner timeouts with VALUES blocks.
_PEOPLE_OCCUPATIONS: dict[str, str] = {
    "Q169470": "physicist",
    "Q170790": "mathematician",
    "Q36180": "writer",
    "Q82955": "politician",
    "Q901": "scientist",
    "Q4964182": "philosopher",
    "Q189290": "military officer",
    "Q11063": "astronomer",
}


def _build_people_query_for_occupation(
    occupation_qid: str,
    min_sitelinks: int = 0,
) -> str:
    """Build a people query for a single occupation.

    Querying one occupation at a time avoids Wikidata query planner timeouts.
    """
    sitelink_clause = ""
    if min_sitelinks > 0:
        sitelink_clause = f"""
  ?item wikibase:sitelinks ?sitelinks .
  FILTER(?sitelinks >= {min_sitelinks})"""

    return f"""
SELECT ?item ?itemLabel ?birthDate ?deathDate
WHERE {{{{
  ?item wdt:P106 wd:{occupation_qid} ;
        wdt:P569 ?birthDate .{sitelink_clause}
  OPTIONAL {{{{ ?item wdt:P570 ?deathDate . }}}}
  SERVICE wikibase:label {{{{ bd:serviceParam wikibase:language "en" . }}}}
}}}}
LIMIT {{limit}} OFFSET {{offset}}
"""


def _parse_year(date_str: str) -> int | None:
    """Extract year from an ISO date string (e.g. '1776-07-04T00:00:00Z').

    Handles negative years (BCE) and Wikidata's date format.
    """
    if not date_str or pd.isna(date_str):
        return None
    date_str = str(date_str).strip()
    # Wikidata format: +1776-07-04T00:00:00Z or -0500-01-01T00:00:00Z
    try:
        if date_str.startswith(("-", "+")):
            year_part = date_str.split("-", 2 if date_str[0] == "+" else 3)
            if date_str[0] == "+":
                return int(year_part[0][1:]) if year_part[0][1:] else None
            else:
                # Negative year: e.g. -0500-01-01T00:00:00Z
                return -int(year_part[1])
        # Plain ISO date
        return int(date_str[:4])
    except (ValueError, IndexError):
        return None


def _year_to_century(year: int) -> int:
    """Convert a year to its century number.

    1-100 = 1st century, 101-200 = 2nd century, etc.
    Negative years: -500 to -401 = -5th century.
    """
    if year > 0:
        return (year - 1) // 100 + 1
    else:
        return -((-year - 1) // 100 + 1)


def _century_label(century: int) -> str:
    """Convert century number to display string like '15th century'."""
    abs_c = abs(century)
    if abs_c % 10 == 1 and abs_c % 100 != 11:
        suffix = "st"
    elif abs_c % 10 == 2 and abs_c % 100 != 12:
        suffix = "nd"
    elif abs_c % 10 == 3 and abs_c % 100 != 13:
        suffix = "rd"
    else:
        suffix = "th"
    if century < 0:
        return f"{abs_c}{suffix} century BC"
    return f"{abs_c}{suffix} century"


def load_historical_events(
    min_year: int = -3000,
    max_year: int = 2020,
    require_precise_date: bool = True,
    require_wikipedia: bool = True,
    force_refresh: bool = False,
    exclude_event_types: frozenset[str] = DEFAULT_EXCLUDE_EVENT_TYPES,
    min_sitelinks: int = 10,
) -> pd.DataFrame:
    """Load historical events with dates from Wikidata.

    Parameters
    ----------
    exclude_event_types
        Wikidata QIDs to exclude from the event type VALUES block.
        Defaults to battles and skirmishes.
    min_sitelinks
        Minimum number of Wikipedia sitelinks for an event to be included.
        Entities with articles in fewer language editions are likely too
        obscure for LLM validators.  Default 10.

    Returns DataFrame with columns:
        qid, name, description, date, year, century,
        event_type, has_wikipedia
    """
    query_template = _build_event_query(
        exclude_event_types=exclude_event_types,
        min_sitelinks=min_sitelinks,
    )
    raw = sparql_query_paginated(
        query_template, page_size=2000, force_refresh=force_refresh,
    )

    if raw.empty:
        logger.warning("No events returned from Wikidata query")
        return pd.DataFrame()

    # Normalize columns
    df = pd.DataFrame()
    df["qid"] = raw.get("item", pd.Series(dtype=str)).apply(_extract_qid)
    df["name"] = raw.get("itemLabel", pd.Series(dtype=str))
    df["description"] = raw.get("itemDescription", pd.Series(dtype=str))
    df["date"] = raw.get("date", pd.Series(dtype=str))
    df["event_type_uri"] = raw.get("eventType", pd.Series(dtype=str))
    df["event_type"] = raw.get("eventTypeLabel", pd.Series(dtype=str))
    df["article"] = raw.get("article", pd.Series(dtype=str))
    df["has_wikipedia"] = df["article"].notna() & (df["article"] != "")

    # Parse year
    df["year"] = df["date"].apply(_parse_year)
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)

    # Apply year range filter
    df = df[(df["year"] >= min_year) & (df["year"] <= max_year)].copy()

    # Compute century
    df["century"] = df["year"].apply(_year_to_century)

    # Deduplicate: keep one row per QID (prefer rows with description)
    df = df.sort_values(
        "description", na_position="last"
    ).drop_duplicates(subset=["qid"], keep="first").reset_index(drop=True)

    logger.info("Raw events: %d unique entities", len(df))

    # Apply filters
    if require_wikipedia:
        df = filter_has_wikipedia(df)

    # Filter out events whose name looks like a QID (no label resolved)
    df = df[~df["name"].str.match(r"^Q\d+$", na=False)].copy()

    logger.info("Final events dataset: %d entities", len(df))
    return df


def load_notable_people(
    min_birth_year: int = -500,
    max_birth_year: int = 2000,
    require_birth_date: bool = True,
    force_refresh: bool = False,
    min_sitelinks: int = 10,
) -> pd.DataFrame:
    """Load notable people with dates from Wikidata.

    Queries each occupation separately to avoid Wikidata query planner
    timeouts, then merges and deduplicates results.

    Parameters
    ----------
    min_sitelinks
        Minimum number of Wikipedia sitelinks for a person to be included.
        Default 10.

    Returns DataFrame with columns:
        qid, name, birth_date, birth_year, death_date, death_year,
        occupation, has_wikipedia
    """
    all_dfs: list[pd.DataFrame] = []

    for occ_qid, occ_label in _PEOPLE_OCCUPATIONS.items():
        query_template = _build_people_query_for_occupation(occ_qid, min_sitelinks=min_sitelinks)
        try:
            raw = sparql_query_paginated(
                query_template, page_size=1000, force_refresh=force_refresh,
                max_pages=5,
            )
        except RuntimeError:
            logger.warning("Skipping occupation %s (%s) due to query failure", occ_label, occ_qid)
            continue

        if raw.empty:
            continue

        chunk = pd.DataFrame()
        chunk["qid"] = raw.get("item", pd.Series(dtype=str)).apply(_extract_qid)
        chunk["name"] = raw.get("itemLabel", pd.Series(dtype=str))
        chunk["birth_date"] = raw.get("birthDate", pd.Series(dtype=str))
        chunk["death_date"] = raw.get("deathDate", pd.Series(dtype=str))
        chunk["occupation"] = occ_label
        all_dfs.append(chunk)
        logger.info("  %s: %d raw rows", occ_label, len(chunk))

    if not all_dfs:
        logger.warning("No people returned from Wikidata queries")
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    df["article"] = ""  # Not queried per-occupation for performance
    df["has_wikipedia"] = False

    # Parse years
    df["birth_year"] = df["birth_date"].apply(_parse_year)
    df["death_year"] = df["death_date"].apply(_parse_year)

    if require_birth_date:
        df = df.dropna(subset=["birth_year"]).copy()
        df["birth_year"] = df["birth_year"].astype(int)

    # Apply birth year range filter
    df = df[
        (df["birth_year"] >= min_birth_year) & (df["birth_year"] <= max_birth_year)
    ].copy()

    # Deduplicate: keep one row per QID (prefer rows with occupation)
    df = df.sort_values(
        "occupation", na_position="last"
    ).drop_duplicates(subset=["qid"], keep="first").reset_index(drop=True)

    logger.info("Raw people: %d unique entities", len(df))

    # Filter out people whose name looks like a QID (no label resolved)
    df = df[~df["name"].str.match(r"^Q\d+$", na=False)].copy()

    # Convert death_year to int where present
    if "death_year" in df.columns:
        df["death_year"] = df["death_year"].astype("Int64")

    logger.info("Final people dataset: %d entities", len(df))
    return df


# ---------------------------------------------------------------------------
# Author-work domain configuration
# ---------------------------------------------------------------------------

# Maps creative domain -> (creator property, work types as QID->label, verb, role)
_AUTHORSHIP_DOMAINS: dict[str, dict] = {
    "literature": {
        "creator_prop": "P50",
        "work_types": {
            "Q7725634": "novel",
            "Q25379": "play",
            "Q5185279": "poem",
            "Q49084": "short story",
            "Q35760": "essay",
        },
        "verb": "written",
        "role": "author",
    },
    "music": {
        "creator_prop": "P86",
        "work_types": {
            "Q9734": "symphony",
            "Q1344": "opera",
            "Q207628": "concerto",
            "Q131746": "sonata",
            "Q105543609": "musical composition",
        },
        "verb": "composed",
        "role": "composer",
    },
    "art": {
        "creator_prop": "P170",
        "work_types": {
            "Q3305213": "painting",
            "Q860861": "sculpture",
            "Q219423": "mural",
        },
        "verb": "painted",
        "role": "painter",
    },
    "film": {
        "creator_prop": "P57",
        "work_types": {
            "Q11424": "film",
        },
        "verb": "directed",
        "role": "director",
    },
    "science": {
        "creator_prop": "P61",
        "work_types": {
            "Q11348": "theorem",
            "Q131476": "scientific theory",
            "Q11023": "equation",
        },
        "verb": "proposed",
        "role": "discoverer",
        # Also use P138 (named after) for science
        "named_after_prop": "P138",
    },
}


def _build_author_work_query(
    domain: str,
    creator_prop: str,
    work_type_qids: list[str],
    min_work_sitelinks: int = 0,
    min_author_sitelinks: int = 0,
) -> str:
    """Build a paginated SPARQL query for author-work pairs in a creative domain.

    Requires English Wikipedia articles for both author and work,
    and filters to works with exactly one creator.
    """
    values_block = " ".join(f"wd:{qid}" for qid in work_type_qids)
    sitelink_clauses = ""
    if min_work_sitelinks > 0:
        sitelink_clauses += f"""
  ?work wikibase:sitelinks ?workSitelinks .
  FILTER(?workSitelinks >= {min_work_sitelinks})"""
    if min_author_sitelinks > 0:
        sitelink_clauses += f"""
  ?author wikibase:sitelinks ?authorSitelinks .
  FILTER(?authorSitelinks >= {min_author_sitelinks})"""

    return f"""
SELECT DISTINCT ?work ?workLabel ?workType ?workTypeLabel
       ?author ?authorLabel ?pubDate
       ?authorArticle ?workArticle
WHERE {{{{
  VALUES ?workType {{{{{ values_block } }}}}
  ?work wdt:P31 ?workType ;
        wdt:{creator_prop} ?author .
  ?work rdfs:label ?workLabel .
  ?author rdfs:label ?authorLabel .
  FILTER(LANG(?workLabel) = "en")
  FILTER(LANG(?authorLabel) = "en"){sitelink_clauses}

  ?authorArticle schema:about ?author ;
                 schema:isPartOf <https://en.wikipedia.org/> .
  ?workArticle schema:about ?work ;
               schema:isPartOf <https://en.wikipedia.org/> .

  OPTIONAL {{{{ ?work wdt:P577 ?pubDate . }}}}
}}}}
LIMIT {{limit}} OFFSET {{offset}}
"""


def _build_named_after_query(
    work_type_qids: list[str],
    min_work_sitelinks: int = 0,
    min_author_sitelinks: int = 0,
) -> str:
    """Build a SPARQL query for science items using P138 (named after)."""
    values_block = " ".join(f"wd:{qid}" for qid in work_type_qids)
    sitelink_clauses = ""
    if min_work_sitelinks > 0:
        sitelink_clauses += f"""
  ?work wikibase:sitelinks ?workSitelinks .
  FILTER(?workSitelinks >= {min_work_sitelinks})"""
    if min_author_sitelinks > 0:
        sitelink_clauses += f"""
  ?author wikibase:sitelinks ?authorSitelinks .
  FILTER(?authorSitelinks >= {min_author_sitelinks})"""

    return f"""
SELECT DISTINCT ?work ?workLabel ?workType ?workTypeLabel
       ?author ?authorLabel ?pubDate
       ?authorArticle ?workArticle
WHERE {{{{
  VALUES ?workType {{{{{ values_block } }}}}
  ?work wdt:P31 ?workType ;
        wdt:P138 ?author .
  ?author wdt:P31 wd:Q5 .
  ?work rdfs:label ?workLabel .
  ?author rdfs:label ?authorLabel .
  FILTER(LANG(?workLabel) = "en")
  FILTER(LANG(?authorLabel) = "en"){sitelink_clauses}

  ?authorArticle schema:about ?author ;
                 schema:isPartOf <https://en.wikipedia.org/> .
  ?workArticle schema:about ?work ;
               schema:isPartOf <https://en.wikipedia.org/> .

  OPTIONAL {{{{ ?work wdt:P577 ?pubDate . }}}}
}}}}
LIMIT {{limit}} OFFSET {{offset}}
"""


def load_authors_and_works(
    domains: list[str] | None = None,
    min_works_per_author: int = 2,
    use_pageview_filter: bool = False,
    pageview_threshold: int = 1000,
    force_refresh: bool = False,
    min_work_sitelinks: int = 5,
    min_author_sitelinks: int = 15,
) -> pd.DataFrame:
    """Load authors/creators and their works from Wikidata.

    Returns DataFrame with columns:
        author_qid, author_name, author_occupation,
        work_qid, work_name, work_type,
        creative_domain (literature, music, science, art, philosophy),
        publication_year (where available),
        has_wikipedia_author, has_wikipedia_work
    """
    target_domains = domains or list(_AUTHORSHIP_DOMAINS.keys())
    all_dfs: list[pd.DataFrame] = []

    for domain_name in target_domains:
        if domain_name not in _AUTHORSHIP_DOMAINS:
            logger.warning("Unknown authorship domain: %s, skipping", domain_name)
            continue

        domain_cfg = _AUTHORSHIP_DOMAINS[domain_name]
        work_type_qids = list(domain_cfg["work_types"].keys())
        work_type_labels = domain_cfg["work_types"]

        # Main creator property query
        query_template = _build_author_work_query(
            domain_name, domain_cfg["creator_prop"], work_type_qids,
            min_work_sitelinks=min_work_sitelinks,
            min_author_sitelinks=min_author_sitelinks,
        )
        try:
            raw = sparql_query_paginated(
                query_template, page_size=2000, force_refresh=force_refresh,
                max_pages=10,
            )
        except RuntimeError:
            logger.warning("Skipping domain %s due to query failure", domain_name)
            raw = pd.DataFrame()

        if not raw.empty:
            chunk = _normalize_author_work_df(raw, domain_name, work_type_labels, domain_cfg)
            all_dfs.append(chunk)
            logger.info("  %s (creator prop): %d raw rows", domain_name, len(chunk))

        # Science domain: also query named-after relationships
        if domain_name == "science" and "named_after_prop" in domain_cfg:
            named_query = _build_named_after_query(
                work_type_qids,
                min_work_sitelinks=min_work_sitelinks,
                min_author_sitelinks=min_author_sitelinks,
            )
            try:
                raw_named = sparql_query_paginated(
                    named_query, page_size=2000, force_refresh=force_refresh,
                    max_pages=5,
                )
            except RuntimeError:
                logger.warning("Skipping named-after query for science domain")
                raw_named = pd.DataFrame()

            if not raw_named.empty:
                chunk_named = _normalize_author_work_df(
                    raw_named, domain_name, work_type_labels, domain_cfg,
                )
                chunk_named["science_attribution"] = True
                all_dfs.append(chunk_named)
                logger.info("  %s (named after): %d raw rows", domain_name, len(chunk_named))

    if not all_dfs:
        logger.warning("No author-work pairs returned from Wikidata queries")
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)

    # Fill science_attribution for non-science rows
    if "science_attribution" not in df.columns:
        df["science_attribution"] = False
    df["science_attribution"] = df["science_attribution"].fillna(False)

    # Deduplicate: one row per (work_qid, author_qid) pair
    df = df.drop_duplicates(subset=["work_qid", "author_qid"], keep="first").reset_index(drop=True)

    # Filter out entries whose names look like QIDs (no label resolved)
    df = df[~df["author_name"].str.match(r"^Q\d+$", na=False)].copy()
    df = df[~df["work_name"].str.match(r"^Q\d+$", na=False)].copy()

    logger.info("Raw author-work pairs: %d", len(df))

    # Filter out ambiguous workshop/school attributions in author names
    _ATTRIBUTION_QUALIFIERS = [
        "workshop of", "circle of", "follower of", "school of",
        "attributed to", "manner of", "studio of", "copy of",
    ]
    before = len(df)
    lower_names = df["author_name"].str.lower()
    mask = pd.Series(False, index=df.index)
    qualifier_counts: dict[str, int] = {}
    for qualifier in _ATTRIBUTION_QUALIFIERS:
        hits = lower_names.str.contains(qualifier, na=False)
        qualifier_counts[qualifier] = int(hits.sum())
        mask = mask | hits
    # Also check "after " as a prefix (e.g. "after Raphael")
    after_hits = lower_names.str.startswith("after ")
    qualifier_counts["after "] = int(after_hits.sum())
    mask = mask | after_hits

    df = df[~mask].copy()
    dropped = before - len(df)
    if dropped > 0:
        logger.info(
            "Workshop/attribution filter: dropped %d rows. Per qualifier: %s",
            dropped,
            {k: v for k, v in qualifier_counts.items() if v > 0},
        )

    # Drop works with multiple creators (shared authorship)
    # Count how many distinct authors each work has
    author_counts = df.groupby("work_qid")["author_qid"].nunique()
    multi_author_works = set(author_counts[author_counts > 1].index)
    if multi_author_works:
        before = len(df)
        df = df[~df["work_qid"].isin(multi_author_works)].copy()
        logger.info("Dropped %d rows for %d multi-author works",
                     before - len(df), len(multi_author_works))

    # Filter to authors with min_works_per_author
    work_counts = df.groupby("author_qid")["work_qid"].nunique()
    valid_authors = set(work_counts[work_counts >= min_works_per_author].index)
    before = len(df)
    df = df[df["author_qid"].isin(valid_authors)].copy().reset_index(drop=True)
    logger.info("Author filter (min %d works): %d -> %d rows",
                 min_works_per_author, before, len(df))

    logger.info("Final author-work dataset: %d pairs across %d authors",
                 len(df), df["author_qid"].nunique())
    return df


def _normalize_author_work_df(
    raw: pd.DataFrame,
    domain_name: str,
    work_type_labels: dict[str, str],
    domain_cfg: dict,
) -> pd.DataFrame:
    """Normalize raw SPARQL results into the standard author-work schema."""
    chunk = pd.DataFrame()
    chunk["author_qid"] = raw.get("author", pd.Series(dtype=str)).apply(_extract_qid)
    chunk["author_name"] = raw.get("authorLabel", pd.Series(dtype=str))
    chunk["author_occupation"] = domain_cfg["role"]
    chunk["work_qid"] = raw.get("work", pd.Series(dtype=str)).apply(_extract_qid)
    chunk["work_name"] = raw.get("workLabel", pd.Series(dtype=str))

    # Resolve work type label from QID
    work_type_uris = raw.get("workType", pd.Series(dtype=str))
    chunk["work_type"] = work_type_uris.apply(
        lambda uri: work_type_labels.get(_extract_qid(uri), "work")
    )

    chunk["creative_domain"] = domain_name
    chunk["publication_year"] = raw.get("pubDate", pd.Series(dtype=str)).apply(_parse_year)
    chunk["has_wikipedia_author"] = True  # required in query
    chunk["has_wikipedia_work"] = True  # required in query
    chunk["verb"] = domain_cfg["verb"]
    chunk["role"] = domain_cfg["role"]
    chunk["science_attribution"] = False
    return chunk


# ---------------------------------------------------------------------------
# Translation extract — entity categories and SPARQL queries
# ---------------------------------------------------------------------------

# Concrete noun categories for translation pairs: QID -> (label, entity_domain)
_TRANSLATION_CATEGORIES: dict[str, tuple[str, str]] = {
    # Animals
    "Q729": ("animal", "animal"),
    "Q7377": ("mammal", "animal"),
    "Q5113": ("bird", "animal"),
    "Q152": ("fish", "animal"),
    "Q1390": ("insect", "animal"),
    "Q10811": ("reptile", "animal"),
    # Fruits & foods
    "Q3314483": ("fruit", "food"),
    "Q11004": ("vegetable", "food"),
    # Countries & geography
    "Q6256": ("country", "country"),
    "Q515": ("city", "country"),
    # Body & nature
    "Q712378": ("body part", "body"),
    "Q1075": ("color", "object"),
    # Common objects
    "Q39546": ("tool", "object"),
    "Q11460": ("clothing", "object"),
    "Q223557": ("musical instrument", "object"),
}

# Default target languages
DEFAULT_TARGET_LANGUAGES = ["es", "fr", "de", "pt", "it", "zh", "ja", "ru", "ar", "hi"]

# Full language display names
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


def _is_clean_label(label: str) -> bool:
    """Check that a label is a clean single-word or short-phrase translation.

    Rejects labels with parentheses, commas, or more than 4 words.
    """
    if not label or not label.strip():
        return False
    if "(" in label or ")" in label or "," in label:
        return False
    if len(label.split()) > 4:
        return False
    return True


def _build_translation_query(category_qid: str, lang_code: str) -> str:
    """Build a SPARQL query for entities of a category with labels in a target language.

    Requires entities to have an English Wikipedia article (common knowledge filter).
    """
    return f"""
SELECT DISTINCT ?item ?itemLabel ?targetLabel ?article
WHERE {{
  ?item wdt:P31/wdt:P279* wd:{category_qid} .
  ?item rdfs:label ?itemLabel .
  ?item rdfs:label ?targetLabel .
  FILTER(LANG(?itemLabel) = "en")
  FILTER(LANG(?targetLabel) = "{lang_code}")

  ?article schema:about ?item ;
           schema:isPartOf <https://en.wikipedia.org/> .
}}
LIMIT 5000
"""


def load_translations(
    source_language: str = "en",
    target_languages: list[str] | None = None,
    entity_domains: list[str] | None = None,
    min_languages_per_entity: int = 3,
    use_pageview_filter: bool = False,
    pageview_threshold: int = 1000,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load translation pairs from Wikidata multilingual labels.

    Queries concrete noun categories for entities with multilingual labels.
    Filters to clean, unambiguous single-word translations.

    Returns DataFrame with columns:
        qid, english_label, target_language, target_label,
        entity_domain, has_wikipedia
    """
    target_langs = target_languages or DEFAULT_TARGET_LANGUAGES
    all_rows: list[dict] = []

    # Determine which categories to query
    categories = _TRANSLATION_CATEGORIES
    if entity_domains:
        domain_set = set(entity_domains)
        categories = {
            qid: (cat_label, domain)
            for qid, (cat_label, domain) in _TRANSLATION_CATEGORIES.items()
            if domain in domain_set
        }

    for cat_qid, (cat_label, entity_domain) in categories.items():
        for lang_code in target_langs:
            query = _build_translation_query(cat_qid, lang_code)
            try:
                raw = sparql_query(query, force_refresh=force_refresh)
            except RuntimeError:
                logger.warning(
                    "Skipping category %s (%s) for language %s due to query failure",
                    cat_label, cat_qid, lang_code,
                )
                continue

            if raw.empty:
                continue

            for _, row in raw.iterrows():
                en_label = str(row.get("itemLabel", "")).strip()
                target_label = str(row.get("targetLabel", "")).strip()
                qid = _extract_qid(str(row.get("item", "")))

                # Skip if either label is empty or looks like a QID
                if not en_label or not target_label:
                    continue
                if en_label.startswith("Q") and en_label[1:].isdigit():
                    continue
                if target_label.startswith("Q") and target_label[1:].isdigit():
                    continue

                # Skip homographs (same word in both languages)
                if en_label.lower() == target_label.lower():
                    continue

                # Skip multi-word / disambiguated labels
                if not _is_clean_label(target_label):
                    continue
                if not _is_clean_label(en_label):
                    continue

                # Script validation for non-Latin languages
                if not _in_expected_script(target_label, lang_code):
                    continue

                all_rows.append({
                    "qid": qid,
                    "english_label": en_label,
                    "target_language": lang_code,
                    "target_label": target_label,
                    "entity_domain": entity_domain,
                    "has_wikipedia": True,
                })

        logger.info("Category %s (%s): %d translation rows so far",
                     cat_label, cat_qid, len(all_rows))

    if not all_rows:
        logger.warning("No translations returned from Wikidata queries")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Deduplicate: keep one row per (english_label, target_language, target_label)
    before = len(df)
    df = df.drop_duplicates(
        subset=["english_label", "target_language", "target_label"],
        keep="first",
    ).reset_index(drop=True)
    logger.info("Deduplication: %d -> %d rows", before, len(df))

    # Filter entities that don't have translations in enough target languages
    lang_counts = df.groupby("qid")["target_language"].nunique()
    valid_qids = set(lang_counts[lang_counts >= min_languages_per_entity].index)
    before = len(df)
    df = df[df["qid"].isin(valid_qids)].copy().reset_index(drop=True)
    logger.info(
        "Min-languages filter (>=%d): %d -> %d rows (%d entities)",
        min_languages_per_entity, before, len(df), len(valid_qids),
    )

    logger.info(
        "Final translations dataset: %d pairs, %d entities, %d languages",
        len(df), df["qid"].nunique(), df["target_language"].nunique(),
    )
    for lang in sorted(df["target_language"].unique()):
        lang_df = df[df["target_language"] == lang]
        logger.info("  %s: %d pairs", lang, len(lang_df))

    return df
