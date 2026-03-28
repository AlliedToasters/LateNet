"""Persistent append-only ledger for validated statements.

The ledger is the central record of every statement we've ever generated
and validated. It grows over time as batches are appended, and is never
overwritten — only appended to.

Layout::

    data/
        ledger.parquet           — all validated rows (clean + disputed)
        disputes.parquet         — disputed rows only (subset of ledger)

Each row in the ledger carries:
- All generation columns (statement, label, pair_id, generator, difficulty, ...)
- Validation columns (llama_agrees, sonnet_agrees, opus_agrees, contested, ...)
- Provenance columns (git_hash, git_dirty, generated_at, batch_id)
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

from .provenance import get_git_dirty, get_git_hash, get_timestamp

logger = logging.getLogger(__name__)

DEFAULT_LEDGER_DIR = Path("data")
LEDGER_FILENAME = "ledger.parquet"
DISPUTES_FILENAME = "disputes.parquet"


def _make_batch_id(git_hash: str, timestamp: str, seed: int) -> str:
    """Deterministic batch ID from provenance fields."""
    raw = f"{git_hash}:{timestamp}:{seed}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def stamp_provenance(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Add provenance columns to a DataFrame of generated/validated rows.

    Columns added:
    - git_hash: full 40-char SHA of HEAD
    - git_dirty: whether the working tree had uncommitted changes
    - generated_at: UTC ISO 8601 timestamp
    - batch_id: deterministic hash of (git_hash, timestamp, seed)
    """
    git_hash = get_git_hash()
    git_dirty = get_git_dirty()
    timestamp = get_timestamp()
    batch_id = _make_batch_id(git_hash, timestamp, seed)

    out = df.copy()
    out["git_hash"] = git_hash
    out["git_dirty"] = git_dirty
    out["generated_at"] = timestamp
    out["batch_id"] = batch_id
    return out


def load_ledger(ledger_dir: Path = DEFAULT_LEDGER_DIR) -> pd.DataFrame:
    """Load the ledger, returning an empty DataFrame if it doesn't exist."""
    path = ledger_dir / LEDGER_FILENAME
    if path.exists():
        df = pd.read_parquet(path)
        # Backfill negated column for ledgers created before negation support
        if "negated" not in df.columns:
            df["negated"] = False
        logger.info("Loaded ledger: %d rows from %s", len(df), path)
        return df
    logger.info("No existing ledger at %s", path)
    return pd.DataFrame()


def load_disputes(ledger_dir: Path = DEFAULT_LEDGER_DIR) -> pd.DataFrame:
    """Load the disputes sidecar, returning an empty DataFrame if it doesn't exist."""
    path = ledger_dir / DISPUTES_FILENAME
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def append_to_ledger(
    new_rows: pd.DataFrame,
    ledger_dir: Path = DEFAULT_LEDGER_DIR,
) -> pd.DataFrame:
    """Append validated rows to the persistent ledger.

    Deduplicates against existing ledger rows by statement text
    (case-insensitive). Returns the full updated ledger.
    """
    ledger_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = ledger_dir / LEDGER_FILENAME
    disputes_path = ledger_dir / DISPUTES_FILENAME

    existing = load_ledger(ledger_dir)

    if len(existing) > 0 and len(new_rows) > 0:
        # Dedup by statement text
        existing_stmts = set(existing["statement"].str.strip().str.lower())
        mask = ~new_rows["statement"].str.strip().str.lower().isin(existing_stmts)
        dupes = (~mask).sum()
        if dupes > 0:
            logger.info("Deduplication: %d rows already in ledger, skipping", dupes)
        new_rows = new_rows[mask]

    if len(new_rows) == 0:
        logger.info("No new rows to append to ledger")
        return existing

    # Append
    if len(existing) > 0:
        # Align columns — new batches may have columns old ones don't
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows.copy()

    # Write full ledger
    combined.to_parquet(ledger_path, index=False)
    logger.info(
        "Ledger updated: %d rows (+%d new) at %s",
        len(combined), len(new_rows), ledger_path,
    )

    # Update disputes sidecar
    if "contested" in combined.columns:
        disputes = combined[combined["contested"] == True]  # noqa: E712
        if len(disputes) > 0:
            disputes.to_parquet(disputes_path, index=False)
            logger.info("Disputes sidecar: %d rows at %s", len(disputes), disputes_path)

    return combined


def get_seen_statements(ledger_dir: Path = DEFAULT_LEDGER_DIR) -> set[str]:
    """Return the set of all statement texts in the ledger (lowercased).

    Used for pre-generation dedup so we don't generate candidates
    that are already in the ledger.
    """
    existing = load_ledger(ledger_dir)
    if len(existing) == 0:
        return set()
    return set(existing["statement"].str.strip().str.lower())


def ledger_strata_counts(ledger_dir: Path = DEFAULT_LEDGER_DIR) -> pd.DataFrame:
    """Return per-stratum counts of clean (uncontested) pairs in the ledger."""
    existing = load_ledger(ledger_dir)
    if len(existing) == 0:
        return pd.DataFrame(columns=["generator", "difficulty", "clean_pairs", "disputed_pairs", "total_pairs"])

    clean = existing[existing.get("contested", False) != True]  # noqa: E712
    disputed = existing[existing.get("contested", False) == True]  # noqa: E712

    clean_counts = clean.groupby(["generator", "difficulty"])["pair_id"].nunique().reset_index(name="clean_pairs")
    disputed_counts = disputed.groupby(["generator", "difficulty"])["pair_id"].nunique().reset_index(name="disputed_pairs")

    counts = clean_counts.merge(disputed_counts, on=["generator", "difficulty"], how="outer").fillna(0)
    counts["disputed_pairs"] = counts["disputed_pairs"].astype(int)
    counts["clean_pairs"] = counts["clean_pairs"].astype(int)
    counts["total_pairs"] = counts["clean_pairs"] + counts["disputed_pairs"]
    return counts.sort_values(["generator", "difficulty"]).reset_index(drop=True)
