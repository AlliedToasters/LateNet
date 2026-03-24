"""Data loading, caching, and downloading for the geography generator.

Lazy-downloads Natural Earth shapefiles and GeoNames dumps on first use.
Caches to ~/.cache/latenet/{naturalearth,geonames}/.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

logger = logging.getLogger(__name__)

CACHE_ROOT = Path.home() / ".cache" / "latenet"

# --- Natural Earth URLs (1:10m) ---
_NE_BASE = "https://naciscdn.org/naturalearth/10m"
_NE_COUNTRIES_URL = f"{_NE_BASE}/cultural/ne_10m_admin_0_countries.zip"
_NE_PLACES_URL = f"{_NE_BASE}/cultural/ne_10m_populated_places.zip"
_NE_RIVERS_URL = f"{_NE_BASE}/physical/ne_10m_rivers_lake_centerlines.zip"

# --- GeoNames URLs ---
_GEONAMES_CITIES_URL = "https://download.geonames.org/export/dump/cities15000.zip"
_GEONAMES_COUNTRY_INFO_URL = "https://download.geonames.org/export/dump/countryInfo.txt"

# GeoNames cities15000 column names (from readme.txt)
_GEONAMES_COLS = [
    "geonameid", "name", "asciiname", "alternatenames",
    "latitude", "longitude", "feature_class", "feature_code",
    "country_code", "cc2", "admin1_code", "admin2_code",
    "admin3_code", "admin4_code", "population", "elevation",
    "dem", "timezone", "modification_date",
]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download_and_extract_zip(url: str, cache_dir: Path, label: str) -> Path:
    """Download a zip file and extract to cache_dir. Returns cache_dir."""
    _ensure_dir(cache_dir)

    # Check if already extracted (presence of .shp or .txt files)
    existing = list(cache_dir.glob("*.shp")) + list(cache_dir.glob("*.txt"))
    if existing:
        logger.info("Cache hit for %s at %s", label, cache_dir)
        return cache_dir

    logger.info("Downloading %s from %s ...", label, url)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(cache_dir)
    logger.info("Extracted %s to %s", label, cache_dir)
    return cache_dir


def _download_file(url: str, dest: Path, label: str) -> Path:
    """Download a single file to dest."""
    _ensure_dir(dest.parent)
    if dest.exists():
        logger.info("Cache hit for %s at %s", label, dest)
        return dest

    logger.info("Downloading %s from %s ...", label, url)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    logger.info("Saved %s to %s", label, dest)
    return dest


# ---------------------------------------------------------------------------
# Public loaders — each returns a GeoDataFrame or DataFrame
# ---------------------------------------------------------------------------

def load_countries() -> gpd.GeoDataFrame:
    """Load Natural Earth 10m countries as a GeoDataFrame."""
    cache_dir = CACHE_ROOT / "naturalearth" / "countries"
    _download_and_extract_zip(_NE_COUNTRIES_URL, cache_dir, "NE countries")
    shp = next(cache_dir.glob("*.shp"))
    gdf = gpd.read_file(shp)
    logger.info("Loaded %d country features", len(gdf))
    return gdf


def load_populated_places() -> gpd.GeoDataFrame:
    """Load Natural Earth 10m populated places."""
    cache_dir = CACHE_ROOT / "naturalearth" / "places"
    _download_and_extract_zip(_NE_PLACES_URL, cache_dir, "NE populated places")
    shp = next(cache_dir.glob("*.shp"))
    gdf = gpd.read_file(shp)
    logger.info("Loaded %d populated place features", len(gdf))
    return gdf


def load_rivers() -> gpd.GeoDataFrame:
    """Load Natural Earth 10m rivers."""
    cache_dir = CACHE_ROOT / "naturalearth" / "rivers"
    _download_and_extract_zip(_NE_RIVERS_URL, cache_dir, "NE rivers")
    shp = next(cache_dir.glob("*.shp"))
    gdf = gpd.read_file(shp)
    logger.info("Loaded %d river features", len(gdf))
    return gdf


def load_geonames_cities() -> pd.DataFrame:
    """Load GeoNames cities15000 dump."""
    cache_dir = CACHE_ROOT / "geonames"
    _download_and_extract_zip(_GEONAMES_CITIES_URL, cache_dir, "GeoNames cities15000")
    txt = next(cache_dir.glob("cities15000.txt"))
    df = pd.read_csv(txt, sep="\t", header=None, names=_GEONAMES_COLS, low_memory=False)
    logger.info("Loaded %d GeoNames cities", len(df))
    return df


def load_geonames_country_info() -> pd.DataFrame:
    """Load GeoNames countryInfo.txt (skipping comment lines)."""
    dest = CACHE_ROOT / "geonames" / "countryInfo.txt"
    _download_file(_GEONAMES_COUNTRY_INFO_URL, dest, "GeoNames countryInfo")
    lines = dest.read_text().splitlines()
    # Skip comment lines starting with #
    header_idx = next(i for i, line in enumerate(lines) if line.startswith("ISO"))
    data_lines = [lines[header_idx]] + [l for l in lines[header_idx + 1:] if not l.startswith("#")]
    df = pd.read_csv(io.StringIO("\n".join(data_lines)), sep="\t")
    logger.info("Loaded %d GeoNames country records", len(df))
    return df


# ---------------------------------------------------------------------------
# Preprocessed views used by the geography generator
# ---------------------------------------------------------------------------

def get_clean_countries(min_pop: int = 100_000) -> gpd.GeoDataFrame:
    """Load countries, filter disputed territories, microstates, nulls."""
    gdf = load_countries()

    # Filter disputed territories: keep only where sovereignty == admin country
    if "SOV_A3" in gdf.columns and "ADM0_A3" in gdf.columns:
        gdf = gdf[gdf["SOV_A3"] == gdf["ADM0_A3"]].copy()

    # Filter by population
    if "POP_EST" in gdf.columns:
        gdf = gdf[gdf["POP_EST"] >= min_pop].copy()

    # Drop null geometry or null name
    gdf = gdf.dropna(subset=["geometry"])
    name_col = "NAME" if "NAME" in gdf.columns else "ADMIN"
    gdf = gdf.dropna(subset=[name_col]).copy()

    logger.info("Clean countries: %d (min_pop=%d)", len(gdf), min_pop)
    return gdf


def get_clean_cities(min_pop: int = 50_000) -> pd.DataFrame:
    """Load GeoNames cities, filter by population, drop nulls."""
    df = load_geonames_cities()
    df = df[df["population"] >= min_pop].copy()
    df = df.dropna(subset=["name", "latitude", "longitude", "country_code"]).copy()
    logger.info("Clean cities: %d (min_pop=%d)", len(df), min_pop)
    return df


def get_country_code_to_name(countries_gdf: gpd.GeoDataFrame) -> dict[str, str]:
    """Build ISO_A2 -> country name mapping from Natural Earth data."""
    mapping = {}
    name_col = "NAME" if "NAME" in countries_gdf.columns else "ADMIN"
    iso_col = "ISO_A2" if "ISO_A2" in countries_gdf.columns else "ISO_A2_EH"
    for _, row in countries_gdf.iterrows():
        code = row.get(iso_col)
        name = row.get(name_col)
        if pd.notna(code) and pd.notna(name) and code != "-99":
            mapping[code] = name
    return mapping
