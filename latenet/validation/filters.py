"""Post-validation filtering and quality metrics.

Merges validation verdicts from all legs into the dataset, computes
consensus, and produces clean (uncontested) and full (annotated) outputs.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .voting import RowVerdict

logger = logging.getLogger(__name__)


def merge_verdicts(
    df: pd.DataFrame,
    ndif_verdicts: dict[int, RowVerdict] | None = None,
    anthropic_verdicts: dict[int, dict[str, RowVerdict]] | None = None,
) -> pd.DataFrame:
    """Merge validation verdicts into the DataFrame as new columns.

    Adds columns:
    - llama_says_true, llama_agrees, llama_confidence, llama_error
    - sonnet_says_true, sonnet_agrees, sonnet_error
    - opus_says_true, opus_agrees, opus_error (only for escalated rows)
    - contested (bool) — True if any validator disagreed with the label

    Parameters
    ----------
    df : pd.DataFrame
        Original dataset with 'statement' and 'label' columns.
    ndif_verdicts : dict | None
        Output from run_ndif_leg().
    anthropic_verdicts : dict | None
        Output from run_anthropic_leg().

    Returns
    -------
    pd.DataFrame
        Copy of df with validation columns appended.
    """
    out = df.copy()

    # NDIF / Llama columns
    if ndif_verdicts:
        out["llama_says_true"] = out.index.map(
            lambda i: ndif_verdicts[i].says_true if i in ndif_verdicts else None
        )
        out["llama_agrees"] = out.index.map(
            lambda i: ndif_verdicts[i].agrees_with_label if i in ndif_verdicts else None
        )
        out["llama_confidence"] = out.index.map(
            lambda i: ndif_verdicts[i].confidence if i in ndif_verdicts else None
        )
        out["llama_error"] = out.index.map(
            lambda i: ndif_verdicts[i].error if i in ndif_verdicts else None
        )

    # Anthropic columns
    if anthropic_verdicts:
        out["sonnet_says_true"] = out.index.map(
            lambda i: anthropic_verdicts[i]["sonnet"].says_true
            if i in anthropic_verdicts else None
        )
        out["sonnet_agrees"] = out.index.map(
            lambda i: anthropic_verdicts[i]["sonnet"].agrees_with_label
            if i in anthropic_verdicts else None
        )
        out["sonnet_error"] = out.index.map(
            lambda i: anthropic_verdicts[i]["sonnet"].error
            if i in anthropic_verdicts else None
        )
        out["sonnet_awkward"] = out.index.map(
            lambda i: anthropic_verdicts[i]["sonnet"].awkward
            if i in anthropic_verdicts else None
        )

        # Opus (only present for escalated rows)
        def _opus_field(i: int, field: str) -> Any:
            if i not in anthropic_verdicts:
                return None
            opus = anthropic_verdicts[i].get("opus")
            if opus is None:
                return None
            return getattr(opus, field)

        out["opus_says_true"] = out.index.map(lambda i: _opus_field(i, "says_true"))
        out["opus_agrees"] = out.index.map(lambda i: _opus_field(i, "agrees_with_label"))
        out["opus_error"] = out.index.map(lambda i: _opus_field(i, "error"))

    # Contested flag: any validator disagreed with the label
    contested = pd.Series(False, index=out.index)

    if "llama_agrees" in out.columns:
        contested = contested | (out["llama_agrees"] == False)  # noqa: E712

    if "sonnet_agrees" in out.columns:
        contested = contested | (out["sonnet_agrees"] == False)  # noqa: E712

    out["contested"] = contested
    return out


def filter_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return only uncontested rows (all validators agree with label).

    Parameters
    ----------
    df : pd.DataFrame
        Output of merge_verdicts() — must have 'contested' column.

    Returns
    -------
    pd.DataFrame
        Rows where no validator disagreed.
    """
    if "contested" not in df.columns:
        raise ValueError("DataFrame must have 'contested' column — run merge_verdicts() first")
    clean = df[~df["contested"]].copy()
    logger.info(
        "Filter: %d/%d rows clean (%.1f%%), %d contested dropped",
        len(clean), len(df), 100 * len(clean) / max(len(df), 1),
        len(df) - len(clean),
    )
    return clean


def validation_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Compute summary statistics from a validated DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of merge_verdicts().

    Returns
    -------
    dict
        Summary statistics.
    """
    total = len(df)
    stats: dict[str, Any] = {"total_rows": total}

    if "llama_agrees" in df.columns:
        llama_ok = (df["llama_agrees"] == True).sum()  # noqa: E712
        llama_err = df["llama_error"].notna().sum()
        stats["llama_agreement"] = f"{llama_ok}/{total} ({100*llama_ok/max(total,1):.1f}%)"
        stats["llama_errors"] = int(llama_err)

    if "sonnet_agrees" in df.columns:
        sonnet_ok = (df["sonnet_agrees"] == True).sum()  # noqa: E712
        sonnet_err = df["sonnet_error"].notna().sum()
        stats["sonnet_agreement"] = f"{sonnet_ok}/{total} ({100*sonnet_ok/max(total,1):.1f}%)"
        stats["sonnet_errors"] = int(sonnet_err)

    if "opus_agrees" in df.columns:
        escalated = df["opus_says_true"].notna().sum()
        opus_ok = (df["opus_agrees"] == True).sum()  # noqa: E712
        stats["opus_escalated"] = int(escalated)
        stats["opus_agreement"] = f"{opus_ok}/{escalated}" if escalated else "n/a"

    if "contested" in df.columns:
        contested = df["contested"].sum()
        clean = total - contested
        stats["contested_rows"] = int(contested)
        stats["clean_rows"] = f"{clean}/{total} ({100*clean/max(total,1):.1f}%)"

    return stats
