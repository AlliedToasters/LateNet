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
DEFAULT_TIMEOUT = 60
MAX_RETRIES = 3
RATE_LIMIT_SECONDS = 1.0


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

    Results are cached as parquet files keyed by query hash.
    """
    cp = _cache_path(query)
    if not force_refresh and cp.exists():
        logger.info("Cache hit for query %s", _cache_key(query)[:12])
        return pd.read_parquet(cp)

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
            data = resp.json()
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
) -> pd.DataFrame:
    """Execute a paginated SPARQL query using {limit} and {offset} placeholders.

    Concatenates all pages into a single DataFrame and caches the final result.
    """
    # Cache the full result by hashing the template
    cp = _cache_path(query_template)
    if not force_refresh and cp.exists():
        logger.info("Cache hit for paginated query %s", _cache_key(query_template)[:12])
        return pd.read_parquet(cp)

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
                data = resp.json()
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

# Ordered from most specific to most general
RANK_ORDER = ["species", "genus", "family", "order", "class", "phylum", "kingdom"]
RANK_LEVEL = {r: i for i, r in enumerate(RANK_ORDER)}


def _build_organism_query() -> str:
    """Build the paginated SPARQL query for organisms with taxonomy.

    Uses a two-step approach:
    1. Get organisms with their direct parent taxon, rank, common name, and Wikipedia link
    2. Resolve full lineage in Python (SPARQL property paths are too slow at scale)
    """
    rank_values = " ".join(f"wd:{qid}" for qid in _RANK_QIDS.values())

    return """
SELECT DISTINCT ?item ?itemLabel ?commonName ?taxonRank ?taxonRankLabel
       ?parentTaxon ?parentTaxonLabel ?ncbiId ?article
WHERE {{
  ?item wdt:P31 wd:Q16521 .
  ?item wdt:P105 ?taxonRank .
  ?item wdt:P171 ?parentTaxon .

  VALUES ?taxonRank {{ {ranks} }}

  OPTIONAL {{ ?item wdt:P1843 ?commonName . FILTER(LANG(?commonName) = "en") }}
  OPTIONAL {{ ?item wdt:P685 ?ncbiId . }}
  OPTIONAL {{
    ?article schema:about ?item .
    ?article schema:isPartOf <https://en.wikipedia.org/> .
  }}

  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}
LIMIT {{limit}} OFFSET {{offset}}
""".format(ranks=rank_values)


def _extract_qid(uri: str) -> str:
    """Extract QID from a Wikidata entity URI."""
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
) -> pd.DataFrame:
    """Load organisms with taxonomy from Wikidata.

    Returns DataFrame with columns:
        qid, name, common_name, taxon_rank,
        parent_taxon_qid, kingdom, phylum, class_, order, family, genus,
        ncbi_taxon_id, has_wikipedia
    """
    query_template = _build_organism_query()
    raw = sparql_query_paginated(
        query_template, page_size=DEFAULT_PAGE_SIZE, force_refresh=force_refresh,
    )

    if raw.empty:
        logger.warning("No organisms returned from Wikidata query")
        return pd.DataFrame()

    # Normalize columns
    df = pd.DataFrame()
    df["qid"] = raw.get("item", pd.Series(dtype=str)).apply(_extract_qid)
    df["name"] = raw.get("itemLabel", pd.Series(dtype=str))
    df["common_name"] = raw.get("commonName", pd.Series(dtype=str))
    df["taxon_rank_uri"] = raw.get("taxonRank", pd.Series(dtype=str))
    df["taxon_rank"] = df["taxon_rank_uri"].apply(_resolve_rank_label)
    df["parent_taxon_qid"] = raw.get("parentTaxon", pd.Series(dtype=str)).apply(_extract_qid)
    df["parent_taxon_label"] = raw.get("parentTaxonLabel", pd.Series(dtype=str))
    df["ncbi_taxon_id"] = raw.get("ncbiId", pd.Series(dtype=str))
    df["article"] = raw.get("article", pd.Series(dtype=str))
    df["has_wikipedia"] = df["article"].notna() & (df["article"] != "")

    # Deduplicate: keep one row per QID (prefer rows with common_name)
    df = df.sort_values(
        "common_name", na_position="last"
    ).drop_duplicates(subset=["qid"], keep="first").reset_index(drop=True)

    logger.info("Raw organisms: %d unique entities", len(df))

    # Apply filters
    if require_wikipedia:
        df = filter_has_wikipedia(df)
    if require_common_name:
        df = filter_has_label(df, label_col="common_name")

    # Build lineage
    df = _build_lineage(df)

    logger.info("Final organism dataset: %d entities", len(df))
    return df


def _build_event_query() -> str:
    """Build the paginated SPARQL query for historical events with dates."""
    return """
SELECT DISTINCT ?item ?itemLabel ?itemDescription ?date ?eventType ?eventTypeLabel ?article
WHERE {{
  VALUES ?eventType {{
    wd:Q1190554  # significant event
    wd:Q198      # war
    wd:Q178561   # battle
    wd:Q131569   # treaty
    wd:Q10931    # revolution
    wd:Q12772819 # discovery
    wd:Q39546    # invention
    wd:Q1462442  # founding
    wd:Q3024240  # historical event
    wd:Q1656682  # event
    wd:Q831663   # military campaign
    wd:Q7275     # political event
  }}
  ?item wdt:P31/wdt:P279* ?eventType .

  # Date: prefer point in time, fall back to start time
  {{
    ?item wdt:P585 ?date .
  }} UNION {{
    ?item wdt:P580 ?date .
    FILTER NOT EXISTS {{ ?item wdt:P585 [] }}
  }}

  # Require year precision or better (precision >= 9)
  ?item p:P585|p:P580 ?dateStatement .
  ?dateStatement psv:P585|psv:P580 ?dateValue .
  ?dateValue wikibase:timePrecision ?precision .
  FILTER(?precision >= 9)

  OPTIONAL {{
    ?article schema:about ?item .
    ?article schema:isPartOf <https://en.wikipedia.org/> .
  }}

  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}
LIMIT {{limit}} OFFSET {{offset}}
"""


def _build_people_query(min_sitelinks: int = 5) -> str:
    """Build the paginated SPARQL query for notable people with dates."""
    return """
SELECT DISTINCT ?item ?itemLabel ?birthDate ?deathDate
       ?occupation ?occupationLabel ?nationality ?nationalityLabel ?article
WHERE {{{{
  ?item wdt:P31 wd:Q5 .             # instance of: human
  ?item wdt:P569 ?birthDate .        # date of birth

  # Require year precision or better on birth date
  ?item p:P569 ?birthStatement .
  ?birthStatement psv:P569 ?birthDateValue .
  ?birthDateValue wikibase:timePrecision ?precision .
  FILTER(?precision >= 9)

  # Notability filter: require N+ sitelinks
  ?item wikibase:sitelinks ?sitelinks .
  FILTER(?sitelinks >= {min_sitelinks})

  OPTIONAL {{ ?item wdt:P570 ?deathDate . }}
  OPTIONAL {{ ?item wdt:P106 ?occupation . }}
  OPTIONAL {{ ?item wdt:P27 ?nationality . }}
  OPTIONAL {{
    ?article schema:about ?item .
    ?article schema:isPartOf <https://en.wikipedia.org/> .
  }}

  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}}}
LIMIT {{{{limit}}}} OFFSET {{{{offset}}}}
""".format(min_sitelinks=min_sitelinks)


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
) -> pd.DataFrame:
    """Load historical events with dates from Wikidata.

    Returns DataFrame with columns:
        qid, name, description, date, year, century,
        event_type, has_wikipedia
    """
    query_template = _build_event_query()
    raw = sparql_query_paginated(
        query_template, page_size=DEFAULT_PAGE_SIZE, force_refresh=force_refresh,
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
    require_wikipedia: bool = True,
    min_sitelinks: int = 5,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load notable people with dates from Wikidata.

    Returns DataFrame with columns:
        qid, name, birth_date, birth_year, death_date, death_year,
        occupation, nationality, has_wikipedia
    """
    query_template = _build_people_query(min_sitelinks=min_sitelinks)
    raw = sparql_query_paginated(
        query_template, page_size=DEFAULT_PAGE_SIZE, force_refresh=force_refresh,
    )

    if raw.empty:
        logger.warning("No people returned from Wikidata query")
        return pd.DataFrame()

    # Normalize columns
    df = pd.DataFrame()
    df["qid"] = raw.get("item", pd.Series(dtype=str)).apply(_extract_qid)
    df["name"] = raw.get("itemLabel", pd.Series(dtype=str))
    df["birth_date"] = raw.get("birthDate", pd.Series(dtype=str))
    df["death_date"] = raw.get("deathDate", pd.Series(dtype=str))
    df["occupation"] = raw.get("occupationLabel", pd.Series(dtype=str))
    df["nationality"] = raw.get("nationalityLabel", pd.Series(dtype=str))
    df["article"] = raw.get("article", pd.Series(dtype=str))
    df["has_wikipedia"] = df["article"].notna() & (df["article"] != "")

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

    # Apply filters
    if require_wikipedia:
        df = filter_has_wikipedia(df)

    # Filter out people whose name looks like a QID (no label resolved)
    df = df[~df["name"].str.match(r"^Q\d+$", na=False)].copy()

    # Convert death_year to int where present
    if "death_year" in df.columns:
        df["death_year"] = df["death_year"].astype("Int64")

    logger.info("Final people dataset: %d entities", len(df))
    return df


def load_authors_and_works(**kwargs) -> pd.DataFrame:
    """Load authors and their works from Wikidata.

    Not yet implemented — stub for authorship generator.
    """
    raise NotImplementedError("Authors and works extract not yet implemented")


def load_translations(**kwargs) -> pd.DataFrame:
    """Load translation pairs from Wikidata.

    Not yet implemented — stub for language generator.
    """
    raise NotImplementedError("Translations extract not yet implemented")
